#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compact modular SAE LoRA fine-tuning script.

This version only splits out:
  - sae_finetune/data.py
  - sae_finetune/lora.py
  - sae_finetune/models.py
  - sae_finetune/utils.py

The following stay in this main file to avoid over-fragmentation:
  - CLI
  - encoding/fusion
  - losses
  - optimizer/scheduler
  - checkpoint naming/saving
  - trainer
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sae_finetune.data import build_dataloaders, filter_batch, resolve_dataset_names
from sae_finetune.lora import configure_trainable_params
from sae_finetune.models import MODEL_REGISTRY, build_model
from sae_finetune.utils import (
    count_trainable_params,
    safe_l2_normalize,
    set_seed,
    unwrap_model,
)


# =============================================================================
# Default paths
# =============================================================================

script_dir = Path(__file__).resolve().parent
sort_root = script_dir.parent
representation_root = sort_root / "representation_collection"
default_latent_root = representation_root / "latents"
default_ckpt_path = script_dir / "pre_sae_weights" / "openclip_ViT-B-32_VL_SAE_256_32_best.pth"
default_save_path = script_dir / "sae_weights"


# =============================================================================
# Encoding and modality fusion
# =============================================================================

def combine_modal_latents(
    text_lat: torch.Tensor,
    img_lat: torch.Tensor,
    text_mask: torch.Tensor,
    img_mask: torch.Tensor,
) -> torch.Tensor:
    assert text_lat.shape == img_lat.shape
    device = text_lat.device
    N, D = text_lat.shape

    out = torch.zeros((N, D), dtype=torch.float32, device=device)
    count = torch.zeros((N, 1), dtype=torch.float32, device=device)

    text_lat = safe_l2_normalize(text_lat)
    img_lat = safe_l2_normalize(img_lat)

    if text_mask.any():
        out[text_mask] += text_lat[text_mask]
        count[text_mask] += 1.0
    if img_mask.any():
        out[img_mask] += img_lat[img_mask]
        count[img_mask] += 1.0

    valid = count.squeeze(-1) > 0
    if valid.any():
        out[valid] = out[valid] / count[valid]
        out[valid] = safe_l2_normalize(out[valid])

    return out


class RetrievalEncoder:
    """
    Uniform adapter over SAE variants.

    Supported model APIs:
      - encode_t / encode_v for text / vision branches.
      - encode for shared encoder.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode_t"):
            return self.model.encode_t(x)
        return self.model.encode(x)

    def encode_vision(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "encode_v"):
            return self.model.encode_v(x)
        return self.model.encode(x)

    def encode_pair(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        q_text_lat = self.encode_text(batch["q_text_emb"])
        q_img_lat = self.encode_vision(batch["q_img_emb"])
        p_text_lat = self.encode_text(batch["p_text_emb"])
        p_img_lat = self.encode_vision(batch["p_img_emb"])

        q_lat = combine_modal_latents(
            q_text_lat,
            q_img_lat,
            batch["q_text_mask"],
            batch["q_img_mask"],
        )
        p_lat = combine_modal_latents(
            p_text_lat,
            p_img_lat,
            batch["p_text_mask"],
            batch["p_img_mask"],
        )

        return q_lat, p_lat


# =============================================================================
# Loss registry
# =============================================================================

LossFn = Callable[[torch.Tensor, torch.Tensor, float, Dict[str, Any]], torch.Tensor]
LOSS_REGISTRY: Dict[str, LossFn] = {}


def register_loss(name: str):
    def decorator(fn: LossFn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator


def info_nce_loss(q_lat: torch.Tensor, p_lat: torch.Tensor, temperature: float) -> torch.Tensor:
    q_lat = safe_l2_normalize(q_lat)
    p_lat = safe_l2_normalize(p_lat)

    logits = (q_lat @ p_lat.t()) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_q = nn.functional.cross_entropy(logits, labels)
    loss_p = nn.functional.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_q + loss_p)


@register_loss("info_nce")
def loss_info_nce(
    q_lat: torch.Tensor,
    p_lat: torch.Tensor,
    temperature: float,
    context: Dict[str, Any],
) -> torch.Tensor:
    return info_nce_loss(q_lat, p_lat, temperature)


@register_loss("matryoshka_info_nce")
def loss_matryoshka_info_nce(
    q_lat: torch.Tensor,
    p_lat: torch.Tensor,
    temperature: float,
    context: Dict[str, Any],
) -> torch.Tensor:
    """Multi-prefix InfoNCE for Matryoshka_VL_SAE."""
    prefix_widths = context.get("prefix_widths", None)
    prefix_weights = context.get("prefix_weights", None)

    if prefix_widths is None or prefix_weights is None:
        raise ValueError("matryoshka_info_nce requires prefix_widths and prefix_weights in context.")

    total = q_lat.new_tensor(0.0)
    weight_sum = 0.0
    latent_dim = q_lat.shape[1]

    for width, weight in zip(prefix_widths, prefix_weights):
        width = int(width)
        width = max(1, min(width, latent_dim))
        weight = float(weight)

        if weight <= 0:
            continue

        q_sub = q_lat[:, :width]
        p_sub = p_lat[:, :width]

        total = total + weight * info_nce_loss(q_sub, p_sub, temperature)
        weight_sum += weight

    if weight_sum <= 0:
        return info_nce_loss(q_lat, p_lat, temperature)

    return total / weight_sum


def get_matryoshka_prefix_config(model: nn.Module) -> Tuple[List[int], List[float]]:
    base_model = unwrap_model(model)

    if not hasattr(base_model, "prefix_widths_tensor"):
        raise AttributeError(
            "Model has no prefix_widths_tensor. "
            "Use loss-mode=matryoshka_info_nce only with Matryoshka_VL_SAE."
        )

    if not hasattr(base_model, "prefix_loss_weights_tensor"):
        raise AttributeError(
            "Model has no prefix_loss_weights_tensor. "
            "Matryoshka_VL_SAE should register this buffer internally."
        )

    prefix_widths = [
        int(x)
        for x in base_model.prefix_widths_tensor.detach().cpu().long().tolist()
    ]
    prefix_weights = [
        float(x)
        for x in base_model.prefix_loss_weights_tensor.detach().cpu().float().tolist()
    ]

    if len(prefix_widths) != len(prefix_weights):
        raise ValueError(
            f"prefix_widths and prefix_weights length mismatch: "
            f"{len(prefix_widths)} vs {len(prefix_weights)}"
        )

    return prefix_widths, prefix_weights


def resolve_loss_mode(args, model: nn.Module) -> str:
    if args.loss_mode != "auto":
        loss_mode = args.loss_mode
    else:
        loss_mode = MODEL_REGISTRY[args.sae_type].default_loss

    if loss_mode not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_mode={loss_mode}. Available: {sorted(LOSS_REGISTRY.keys())}")

    if loss_mode == "matryoshka_info_nce" and args.sae_type != "matryoshka_vlsae":
        raise ValueError("matryoshka_info_nce should only be used with sae_type=matryoshka_vlsae.")

    return loss_mode


def build_loss_context(args, model: nn.Module, loss_mode: str) -> Dict[str, Any]:
    context: Dict[str, Any] = {}

    if loss_mode == "matryoshka_info_nce":
        prefix_widths, prefix_weights = get_matryoshka_prefix_config(model)
        context["prefix_widths"] = prefix_widths
        context["prefix_weights"] = prefix_weights

        print(f"[INFO] loss_mode                 = {loss_mode}", flush=True)
        print(f"[INFO] matryoshka prefix_widths  = {prefix_widths}", flush=True)
        print(f"[INFO] matryoshka prefix_weights = {prefix_weights}", flush=True)
    else:
        print(f"[INFO] loss_mode                 = {loss_mode}", flush=True)

    return context


def compute_retrieval_loss(
    q_lat: torch.Tensor,
    p_lat: torch.Tensor,
    temperature: float,
    loss_mode: str,
    loss_context: Dict[str, Any],
) -> torch.Tensor:
    return LOSS_REGISTRY[loss_mode](
        q_lat=q_lat,
        p_lat=p_lat,
        temperature=temperature,
        context=loss_context,
    )


def infer_effective_loss_mode_from_args(args) -> str:
    if args.loss_mode != "auto":
        return str(args.loss_mode)
    return MODEL_REGISTRY[args.sae_type].default_loss


# =============================================================================
# Checkpoint naming and saving
# =============================================================================

def build_loss_tag(args) -> str:
    loss_mode = infer_effective_loss_mode_from_args(args)

    # Backward compatibility with the original default VLSAE name.
    if args.sae_type == "vlsae" and loss_mode == "info_nce":
        return ""

    return f"_loss_{loss_mode}"


def build_target_tag(args) -> str:
    if args.adapter_mode != "lora":
        return f"_{args.adapter_mode}"
    if not args.lora_target_modules:
        return ""
    joined = "-".join(args.lora_target_modules)
    return f"_target_{joined}"


def build_checkpoint_name(args, stage: str) -> str:
    if args.sae_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown sae_type={args.sae_type}")

    model_name = MODEL_REGISTRY[args.sae_type].display_name
    loss_tag = build_loss_tag(args)
    target_tag = build_target_tag(args)

    if stage == "best":
        if args.adapter_mode == "lora":
            suffix = f"finetune_lorar_{args.lora_r}_loraa_{args.lora_alpha}{target_tag}"
        else:
            suffix = f"finetune_{args.adapter_mode}"
    elif stage == "final":
        suffix = f"finetune_final{target_tag}"
    else:
        raise ValueError(f"Unknown checkpoint stage: {stage}")

    return (
        f"openclip_ViT-B-32_{model_name}_{args.topk}_{args.hidden_ratio}"
        f"_{suffix}{loss_tag}.pth"
    )


def build_checkpoint_path(args, stage: str) -> str:
    return os.path.join(args.save_path, build_checkpoint_name(args, stage=stage))


def save_json_log(logs: List[Dict[str, Any]], save_path: str, stage: str) -> None:
    log_path = os.path.join(save_path, f"train_log_{stage}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


# =============================================================================
# Optimizer and scheduler
# =============================================================================

def build_optimizer(args, model: nn.Module) -> optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found. Check --adapter-mode and LoRA target modules.")

    if args.optimizer == "adam":
        return optim.Adam(params, lr=args.initial_lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return optim.AdamW(params, lr=args.initial_lr, weight_decay=args.weight_decay)

    raise ValueError(f"Unknown optimizer={args.optimizer}")


def build_scheduler(args, optimizer: optim.Optimizer):
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.num_epochs - args.warmup_epochs),
        )
    if args.scheduler == "none":
        return None

    raise ValueError(f"Unknown scheduler={args.scheduler}")


# =============================================================================
# Trainer
# =============================================================================

class SAEFineTuner:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        args,
        dataset_names: List[str],
        loss_mode: str,
        loss_context: Dict[str, Any],
    ):
        self.model = model
        self.encoder = RetrievalEncoder(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.dataset_names = dataset_names
        self.loss_mode = loss_mode
        self.loss_context = loss_context
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.logs: List[Dict[str, Any]] = []

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        q_lat, p_lat = self.encoder.encode_pair(batch)
        return compute_retrieval_loss(
            q_lat=q_lat,
            p_lat=p_lat,
            temperature=self.args.temperature,
            loss_mode=self.loss_mode,
            loss_context=self.loss_context,
        )

    def step_lr(self, epoch: int) -> None:
        if epoch < self.args.warmup_epochs:
            lr = self.args.initial_lr * (epoch + 1) / max(1, self.args.warmup_epochs)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            return

        if self.scheduler is not None:
            self.scheduler.step()

    def run_epoch(self, epoch: int) -> float:
        self.model.train()
        self.step_lr(epoch)

        total_loss = 0.0
        total_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.args.num_epochs}"):
            batch = filter_batch(batch, self.args.device)
            if batch is None:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss = self.compute_loss(batch)
            loss.backward()

            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.args.max_grad_norm,
                )

            self.optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(1, total_batches)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self.val_loader:
            batch = filter_batch(batch, self.args.device)
            if batch is None:
                continue

            loss = self.compute_loss(batch)
            total_loss += float(loss.item())
            total_batches += 1

        if total_batches == 0:
            return float("inf")
        return total_loss / total_batches

    def fit(self) -> None:
        for epoch in range(self.args.num_epochs):
            train_loss = self.run_epoch(epoch)
            val_loss = self.validate()

            lr = self.optimizer.param_groups[0]["lr"]
            self.log_epoch(epoch, lr, train_loss, val_loss)

            improved = self.maybe_save_best(val_loss)
            if not improved and self.patience_counter >= self.args.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}", flush=True)
                break

        if self.args.save_final:
            self.save(stage="final")

    def log_epoch(self, epoch: int, lr: float, train_loss: float, val_loss: float) -> None:
        item = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        self.logs.append(item)

        print(
            f"Epoch [{epoch + 1}/{self.args.num_epochs}], "
            f"LR: {lr:.6e}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}",
            flush=True,
        )

    def maybe_save_best(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save(stage="best")
            return True

        self.patience_counter += 1
        return False

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "args": vars(self.args),
            "best_val_loss": self.best_val_loss,
            "logs": self.logs,
            "datasets": self.dataset_names,
            "loss_mode": self.loss_mode,
            "loss_context": self.loss_context,
            "model_registry_name": self.args.sae_type,
            "model_display_name": MODEL_REGISTRY[self.args.sae_type].display_name,
        }

    def save(self, stage: str) -> None:
        path = build_checkpoint_path(self.args, stage=stage)
        torch.save(self.checkpoint_payload(), path)
        save_json_log(self.logs, self.args.save_path, stage)
        print(f"[INFO] Saved {stage} checkpoint to: {path}", flush=True)


# =============================================================================
# CLI
# =============================================================================

def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "LoRA fine-tune SAE with InfoNCE / Matryoshka InfoNCE",
        add_help=True,
    )

    # Data
    parser.add_argument(
        "--latent-root",
        default=str(default_latent_root),
        type=str,
        help="Root containing <DATASET>/latents.pt or latent pt files.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Train on one dataset only; default is all datasets under latent-root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Train on multiple datasets; default is all datasets under latent-root.",
    )
    parser.add_argument("--training-ratio", default=0.9, type=float)
    parser.add_argument("--shuffle", action="store_true")

    # Model
    parser.add_argument(
        "--sae-type",
        default="vlsae",
        choices=sorted(MODEL_REGISTRY.keys()),
        type=str,
        help="Which SAE model to fine-tune.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=str(default_ckpt_path),
        type=str,
        help="Pretrained SAE checkpoint. Use empty string to skip loading.",
    )
    parser.add_argument("--input-dim", default=512, type=int)
    parser.add_argument("--hidden-ratio", default=32, type=int)
    parser.add_argument("--topk", default=256, type=int)
    parser.add_argument("--model-dropout", default=0.01, type=float)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--kron-num-heads", default=4, type=int)
    parser.add_argument("--kron-base-dim", default=16, type=int)
    parser.add_argument("--kron-extend-dim", default=None, type=int)

    # Adapter / LoRA
    parser.add_argument(
        "--adapter-mode",
        default="lora",
        choices=["none", "lora", "full"],
        type=str,
        help="none: freeze all; lora: train LoRA only; full: full fine-tuning.",
    )
    parser.add_argument("--lora-r", default=48, type=int)
    parser.add_argument("--lora-alpha", default=96, type=float)
    parser.add_argument("--lora-dropout", default=0.0, type=float)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help=(
            "Substrings of module names to apply LoRA to. "
            "Default: all nn.Linear modules. Examples: --lora-target-modules encoder decoder"
        ),
    )

    # Training
    parser.add_argument("--num-epochs", default=300, type=int)
    parser.add_argument("--warmup-epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--initial-lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--patience", default=16, type=int)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--max-grad-norm", default=None, type=float)
    parser.add_argument(
        "--optimizer",
        default="adam",
        choices=["adam", "adamw"],
        type=str,
    )
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "none"],
        type=str,
    )
    parser.add_argument(
        "--loss-mode",
        default="auto",
        choices=["auto"] + sorted(LOSS_REGISTRY.keys()),
        type=str,
        help=(
            "Training loss mode. "
            "auto: use each model's registered default loss."
        ),
    )

    # Device / saving
    parser.add_argument(
        "--device",
        default="cuda:4" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--save-path",
        default=str(default_save_path),
        type=str,
        help="Directory to save fine-tuned checkpoints.",
    )
    parser.add_argument("--save-final", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    return parser


# =============================================================================
# Main
# =============================================================================

def print_run_config(args, dataset_names: List[str]) -> None:
    print(f"[INFO] sae_type         = {args.sae_type}", flush=True)
    print(f"[INFO] model_name       = {MODEL_REGISTRY[args.sae_type].display_name}", flush=True)
    print(f"[INFO] latent_root      = {args.latent_root}", flush=True)
    print(f"[INFO] datasets         = {dataset_names}", flush=True)
    print(f"[INFO] ckpt_path        = {args.ckpt_path}", flush=True)
    print(f"[INFO] save_path        = {args.save_path}", flush=True)
    print(f"[INFO] device           = {args.device}", flush=True)
    print(f"[INFO] adapter_mode     = {args.adapter_mode}", flush=True)
    print(f"[INFO] input_dim        = {args.input_dim}", flush=True)
    print(f"[INFO] hidden_dim       = {args.input_dim * args.hidden_ratio}", flush=True)
    print(f"[INFO] topk             = {args.topk}", flush=True)


def main(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    dataset_names = resolve_dataset_names(args)
    if len(dataset_names) == 0:
        raise RuntimeError(f"No datasets found under: {args.latent_root}")

    print_run_config(args, dataset_names)

    train_loader, val_loader, n_train, n_val = build_dataloaders(args, dataset_names)
    print(f"[INFO] train samples    = {n_train}", flush=True)
    print(f"[INFO] val samples      = {n_val}", flush=True)

    model = build_model(args, device=args.device)
    configure_trainable_params(model, args)

    loss_mode = resolve_loss_mode(args, model)
    loss_context = build_loss_context(args, model, loss_mode)

    total_params, trainable_params = count_trainable_params(model)
    print(f"[INFO] total params     = {total_params}", flush=True)
    print(f"[INFO] trainable params = {trainable_params}", flush=True)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    trainer = SAEFineTuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        dataset_names=dataset_names,
        loss_mode=loss_mode,
        loss_context=loss_context,
    )
    trainer.fit()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
