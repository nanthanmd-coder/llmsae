import random
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_state_dict_keys(ckpt: Any) -> Any:
    """Normalize common checkpoint wrappers and prefixes."""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            ckpt = ckpt["model_state_dict"]

    if isinstance(ckpt, dict):
        new_ckpt = {}
        for k, v in ckpt.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("autoencoder."):
                nk = nk[len("autoencoder."):]
            new_ckpt[nk] = v
        ckpt = new_ckpt

    return ckpt


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def count_trainable_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def freeze_all_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_all_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def freeze_non_lora_params(module: nn.Module) -> None:
    for name, p in module.named_parameters():
        p.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / denom
