import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .utils import freeze_all_params, freeze_non_lora_params, unfreeze_all_params


class LoRALinear(nn.Module):
    """
    Lightweight LoRA wrapper for nn.Linear.

    y = base(x) + scale * B(A(dropout(x)))

    The base linear layer is frozen inside this wrapper.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0
        self.dropout = nn.Dropout(float(dropout))

        for p in self.base.parameters():
            p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r <= 0:
            return out

        orig_shape = x.shape
        x2 = self.dropout(x).reshape(-1, orig_shape[-1])
        delta = (x2 @ self.lora_A.t()) @ self.lora_B.t()
        delta = delta.reshape(*orig_shape[:-1], self.base.out_features)
        return out + self.scaling * delta


@dataclass
class AdapterConfig:
    mode: str
    r: int
    alpha: float
    dropout: float
    target_modules: Optional[List[str]] = None


def should_wrap_linear(module_name: str, target_modules: Optional[List[str]]) -> bool:
    if target_modules is None or len(target_modules) == 0:
        return True
    return any(target in module_name for target in target_modules)


def apply_lora(module: nn.Module, cfg: AdapterConfig, prefix: str = "") -> int:
    """
    Recursively replace selected nn.Linear modules with LoRALinear.

    Returns the number of wrapped Linear modules.
    """
    wrapped = 0

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, LoRALinear):
            continue

        if isinstance(child, nn.Linear) and should_wrap_linear(full_name, cfg.target_modules):
            setattr(
                module,
                name,
                LoRALinear(
                    base=child,
                    r=cfg.r,
                    alpha=cfg.alpha,
                    dropout=cfg.dropout,
                ),
            )
            wrapped += 1
        else:
            wrapped += apply_lora(child, cfg, prefix=full_name)

    return wrapped


def configure_trainable_params(model: nn.Module, args) -> None:
    """
    Unified trainable-parameter policy.

    adapter_mode:
      - none: freeze everything; useful for evaluation sanity checks.
      - lora: inject LoRA into selected Linear modules, train LoRA only.
      - full: train all SAE parameters.
    """
    if args.adapter_mode == "none":
        freeze_all_params(model)
        print("[INFO] adapter_mode=none. All parameters are frozen.", flush=True)
        return

    if args.adapter_mode == "full":
        unfreeze_all_params(model)
        print("[INFO] adapter_mode=full. All parameters are trainable.", flush=True)
        return

    if args.adapter_mode == "lora":
        cfg = AdapterConfig(
            mode="lora",
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        wrapped = apply_lora(model, cfg)
        freeze_non_lora_params(model)
        print(f"[INFO] adapter_mode=lora. Wrapped Linear modules: {wrapped}", flush=True)
        print(f"[INFO] lora_target_modules={args.lora_target_modules}", flush=True)
        return

    raise ValueError(f"Unknown adapter_mode={args.adapter_mode}")
