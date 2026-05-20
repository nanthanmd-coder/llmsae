import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from .utils import normalize_state_dict_keys

# Expected placement:
#   sae_trainer/
#     sae_model.py
#     finetune_adaptive.py
#     sae_finetune/
#       models.py
trainer_root = Path(__file__).resolve().parent.parent
sys.path.append(str(trainer_root))

from sae_model import (  # noqa: E402
    VL_SAE,
    SAE_D,
    SAE_V,
    Matryoshka_VL_SAE,
    KronSAE,
    IKronSAE,
    SAE12,
)


@dataclass
class ModelSpec:
    name: str
    display_name: str
    build_fn: Callable[[Any, int], nn.Module]
    default_loss: str = "info_nce"


MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(name: str, display_name: str, default_loss: str = "info_nce"):
    def decorator(build_fn: Callable[[Any, int], nn.Module]):
        MODEL_REGISTRY[name] = ModelSpec(
            name=name,
            display_name=display_name,
            build_fn=build_fn,
            default_loss=default_loss,
        )
        return build_fn
    return decorator


@register_model("vlsae", "VL_SAE", default_loss="info_nce")
def build_vlsae(args: Any, hidden_dim: int) -> nn.Module:
    # Backward-compatible with the original VL_SAE constructor.
    return VL_SAE(
        args.input_dim,
        hidden_dim,
        topk=args.topk,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


@register_model("saed", "SAE_D", default_loss="info_nce")
def build_saed(args: Any, hidden_dim: int) -> nn.Module:
    return SAE_D(args.input_dim, hidden_dim, topk=args.topk)


@register_model("saev", "SAE_V", default_loss="info_nce")
def build_saev(args: Any, hidden_dim: int) -> nn.Module:
    return SAE_V(args.input_dim, hidden_dim, topk=args.topk)


@register_model("sae12", "SAE12", default_loss="info_nce")
def build_sae12(args: Any, hidden_dim: int) -> nn.Module:
    return SAE12(args.input_dim, hidden_dim, topk=args.topk)


@register_model("matryoshka_vlsae", "Matryoshka_VL_SAE", default_loss="matryoshka_info_nce")
def build_matryoshka_vlsae(args: Any, hidden_dim: int) -> nn.Module:
    return Matryoshka_VL_SAE(
        input_dim=args.input_dim,
        hidden_dim=hidden_dim,
        topk=args.topk,
        dropout=args.model_dropout,
        activation=args.activation,
    )


@register_model("kron_sae", "Kron_SAE", default_loss="info_nce")
def build_kron_sae(args: Any, hidden_dim: int) -> nn.Module:
    return KronSAE(
        input_dim=args.input_dim,
        hidden_dim=hidden_dim,
        num_heads=args.kron_num_heads,
        base_dim=args.kron_base_dim,
        extend_dim=args.kron_extend_dim,
        topk=args.topk,
        dropout=args.model_dropout,
        activation=args.activation,
    )


@register_model("i_kron_sae", "IKron_SAE", default_loss="info_nce")
def build_i_kron_sae(args: Any, hidden_dim: int) -> nn.Module:
    return IKronSAE(
        input_dim=args.input_dim,
        hidden_dim=hidden_dim,
        num_heads=args.kron_num_heads,
        base_dim=args.kron_base_dim,
        extend_dim=args.kron_extend_dim,
        topk=args.topk,
        dropout=args.model_dropout,
        activation=args.activation,
    )


def build_model(args: Any, device: str) -> nn.Module:
    if args.sae_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown sae_type={args.sae_type}. Available: {sorted(MODEL_REGISTRY.keys())}"
        )

    hidden_dim = args.input_dim * args.hidden_ratio
    model = MODEL_REGISTRY[args.sae_type].build_fn(args, hidden_dim).to(device)

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        ckpt = normalize_state_dict_keys(ckpt)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] missing keys: {missing}", flush=True)
        print(f"[INFO] unexpected keys: {unexpected}", flush=True)
    else:
        print("[WARN] --ckpt-path is empty. Training from current random initialization.", flush=True)

    return model
