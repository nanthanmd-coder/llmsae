import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import json


class AuxiliaryAE(nn.Module):
    def __init__(self, vision_dim, text_dim, projection_dim=4096):
        super().__init__()
        self.vision_projection = nn.Linear(vision_dim, projection_dim)
        self.text_projection = nn.Linear(text_dim, projection_dim)
        
        self.vision_decoder = nn.Linear(projection_dim, vision_dim)
        self.text_decoder = nn.Linear(projection_dim, text_dim)

    def encoder(self, vision_features=None, text_features=None):
        vision_embed, text_embed = None, None
        if vision_features is not None:
            vision_embed = self.vision_projection(vision_features)
        if text_features is not None:
            text_embed = self.text_projection(text_features)
        return vision_embed, text_embed
    def decoder(self, vision_embed=None, text_embed=None):
        vision_recon, text_recon = None, None
        if vision_embed is not None:
            vision_recon = self.vision_decoder(vision_embed)
        if text_embed is not None:
            text_recon = self.text_decoder(text_embed)
        return vision_recon, text_recon

    def forward(self, vision_features=None, text_features=None):
        vision_embed, text_embed = self.encoder(vision_features, text_features)
        vision_recon, text_recon = self.decoder(vision_embed, text_embed)
        return vision_embed, text_embed, vision_recon, text_recon

class VL_SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0):
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_uniform_(self.encoder, a=math.sqrt(5))
        
        self.vision_decoder = nn.Linear(hidden_dim, input_dim)
        self.text_decoder = nn.Linear(hidden_dim, input_dim)

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings, topk):
        abs_feat = torch.abs(embeddings)
        thres = torch.kthvalue(abs_feat, k=(self.hidden_dim - topk), dim=1)[0]
        sub = abs_feat - thres.unsqueeze(-1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = torch.ones_like(n_sub)
        n_sub = torch.where(n_sub != 0, one_sub, n_sub)
        embeddings = embeddings * n_sub
        return embeddings

    def encode(self, embeddings, mode='eval'):
        weights = F.normalize(self.encoder, p=2, dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = torch.cdist(embeddings, weights, p=2)
        embeddings = 2 - embeddings
        return self.sparsify(embeddings, topk=self.topk)

    def forward(self, vision_embeddings=None, text_embeddings=None, mode='eval'):
        recon_vision_embeddings = None
        recon_text_embeddings = None
        latent_v = None
        latent_t = None
        if vision_embeddings is not None:
            latent_v = self.encode(vision_embeddings, mode=mode)
            recon_vision_embeddings = self.vision_decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode(text_embeddings, mode=mode)
            recon_text_embeddings = self.text_decoder(latent_t)
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t



class SAE_D(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1):
        super().__init__()
        self.v_encoder = nn.Linear(input_dim, hidden_dim)
        self.activations = nn.ReLU()
        
        self.t_encoder = nn.Linear(input_dim, hidden_dim)
        self.vision_decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            )
        self.text_decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            )


        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        abs_feat = torch.abs(embeddings)
        thres = torch.kthvalue(abs_feat.float(), k=(self.hidden_dim - self.topk), dim=1)[0]

        sub = abs_feat - thres.unsqueeze(-1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = torch.ones_like(n_sub)
        n_sub = torch.where(n_sub != 0, one_sub, n_sub)
        embeddings = embeddings * n_sub
   
        return embeddings
        

    def encode_v(self, embeddings):
        return self.sparsify(self.activations(self.v_encoder(embeddings)))
    
    def encode_t(self, embeddings):
        return self.sparsify(self.activations(self.t_encoder(embeddings)))

    def forward(self, vision_embeddings=None, text_embeddings=None):
        recon_vision_embeddings = None
        recon_text_embeddings = None
        latent_v = None
        latent_t = None
        if vision_embeddings is not None:
            latent_v = self.encode_v(vision_embeddings)
            recon_vision_embeddings = self.vision_decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode_t(text_embeddings)
            recon_text_embeddings = self.text_decoder(latent_t)
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t

class SAE_V(nn.Module):
    def __init__(self, input_dim, hidden_dim, topk=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activations = nn.ReLU()
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            )

        self.topk = topk
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def sparsify(self, embeddings):
        abs_feat = torch.abs(embeddings)
        thres = torch.kthvalue(abs_feat.float(), k=(self.hidden_dim - self.topk), dim=1)[0]

        sub = abs_feat - thres.unsqueeze(-1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = torch.ones_like(n_sub)
        n_sub = torch.where(n_sub != 0, one_sub, n_sub)
        embeddings = embeddings * n_sub
   
        return embeddings

    def encode(self, embeddings):
        return self.sparsify(self.activations(self.encoder(embeddings)))

    def forward(self, vision_embeddings=None, text_embeddings=None, mode='eval'):
        recon_vision_embeddings = None
        recon_text_embeddings = None
        latent_v = None
        latent_t = None
        if vision_embeddings is not None:
            latent_v = self.encode(vision_embeddings)
            recon_vision_embeddings = self.decoder(latent_v)
        if text_embeddings is not None:
            latent_t = self.encode(text_embeddings)
            recon_text_embeddings = self.decoder(latent_t)
        return recon_vision_embeddings, recon_text_embeddings, latent_v, latent_t


# ============================================================================
# 以下是为 llm_trainer 适配的 SAE 类(EleutherAI sparsify / multimodal-sae 格式)
# 与上面的 VL_SAE / SAE_D / SAE_V / AuxiliaryAE 完全独立, 互不影响。
# 用于加载 HuggingFace lmms-lab/llama3-llava-next-8b-hf-sae-131k 这类权重。
# ============================================================================
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import NamedTuple, Optional, Union


@dataclass
class SaeConfig:
    """对应 sparsify 的 SaeConfig"""
    expansion_factor: int = 32
    num_latents: int = 0
    normalize_decoder: bool = True
    k: int = 32
    multi_topk: bool = False
    skip_connection: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    top_indices: torch.Tensor


class ForwardOutput(NamedTuple):
    sae_out: torch.Tensor
    latent_acts: torch.Tensor
    latent_indices: torch.Tensor
    fvu: torch.Tensor
    auxk_loss: torch.Tensor
    multi_topk_fvu: torch.Tensor


def _decoder_impl_torch(top_indices, top_acts, W_dec_T):
    """纯 PyTorch 实现的稀疏解码 y = sum_k (top_acts * W_dec[top_indices])"""
    W_dec = W_dec_T.T                                       # [num_latents, d_in]
    selected = W_dec[top_indices]                           # [..., K, d_in]
    return (top_acts.unsqueeze(-1) * selected).sum(dim=-2)  # [..., d_in]


# 优先用 sparsify 自带的 triton kernel(更快),没有就 fallback 到纯 PyTorch
try:
    from sparsify.utils import decoder_impl as _decoder_impl
    _DECODER_BACKEND = "sparsify"
except Exception:
    _decoder_impl = _decoder_impl_torch
    _DECODER_BACKEND = "torch"


class SAE(nn.Module):
    """
    EleutherAI sparsify / multimodal-sae 风格的 TopK SAE。
    专门用来加载 lmms-lab/llama3-llava-next-8b-hf-sae-131k 这类权重。

    与同文件内的 VL_SAE / SAE_D / SAE_V 完全独立, 互不影响。

    最简单的用法:
        from sae_model import SAE
        sae = SAE.from_pretrained(
            "finetune_models/llama3-llava-next-8b-hf-sae-131k/model.layers.24"
        )
        out = sae(x)   # ForwardOutput(sae_out, latent_acts, latent_indices, ...)

    或者仅 encode/decode:
        top_acts, top_indices = sae.encode(x)
        x_hat = sae.decode(top_acts, top_indices)
    """

    def __init__(
        self,
        d_in: int,
        cfg: Optional[SaeConfig] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        *,
        decoder: bool = True,
        # 简化用法: 也允许直接传 num_latents/k 等参数
        num_latents: Optional[int] = None,
        k: Optional[int] = None,
        expansion_factor: Optional[int] = None,
        normalize_decoder: Optional[bool] = None,
        multi_topk: Optional[bool] = None,
    ):
        super().__init__()

        if cfg is None:
            cfg = SaeConfig()
        if num_latents is not None:
            cfg.num_latents = num_latents
        if k is not None:
            cfg.k = k
        if expansion_factor is not None:
            cfg.expansion_factor = expansion_factor
        if normalize_decoder is not None:
            cfg.normalize_decoder = normalize_decoder
        if multi_topk is not None:
            cfg.multi_topk = multi_topk

        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x):
        sae_in = x.to(self.dtype) - self.b_dec
        return F.relu(self.encoder(sae_in))

    def select_topk(self, latents):
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def encode(self, x):
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts, top_indices):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        y = _decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def forward(self, x, dead_mask=None):
        pre_acts = self.pre_acts(x)
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)

        e = sae_out - x
        total_variance = (x - x.mean(0)).pow(2).sum()

        if dead_mask is not None and int(dead_mask.sum()) > 0:
            num_dead = int(dead_mask.sum())
            k_aux = x.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        fvu = e.pow(2).sum() / total_variance

        if self.cfg.multi_topk:
            top_acts_m, top_indices_m = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out_m = self.decode(top_acts_m, top_indices_m)
            multi_topk_fvu = (sae_out_m - x).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(sae_out, top_acts, top_indices, fvu, auxk_loss, multi_topk_fvu)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None and self.W_dec.grad is not None
        parallel = (self.W_dec.grad * self.W_dec.data).sum(dim=-1)
        self.W_dec.grad -= parallel.unsqueeze(-1) * self.W_dec.data

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
        *,
        decoder: bool = True,
        strict: Optional[bool] = None,
    ) -> "SAE":
        """
        从目录加载(推荐入口)。目录布局:
            path/
                cfg.json             {d_in, num_latents, k, ...}
                sae.safetensors      权重
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"SAE checkpoint dir not found: {path}")

        cfg_path = path / "cfg.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"cfg.json not found under {path}")

        with open(cfg_path, "r") as f:
            cfg_dict = json.load(f)

        d_in = cfg_dict.pop("d_in")
        cfg = SaeConfig.from_dict(cfg_dict)

        sae = cls(d_in, cfg, device=device, dtype=dtype, decoder=decoder)

        # 找权重文件
        ckpt_file = path / "sae.safetensors"
        if not ckpt_file.is_file():
            candidates = (list(path.glob("*.safetensors"))
                          + list(path.glob("*.pt"))
                          + list(path.glob("*.bin")))
            if not candidates:
                raise FileNotFoundError(f"No weight file under {path}")
            ckpt_file = candidates[0]

        if ckpt_file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_model
            except ImportError:
                raise ImportError("safetensors not installed; run `pip install safetensors`")
            strict_flag = strict if strict is not None else decoder
            load_model(model=sae, filename=str(ckpt_file),
                       device=str(device), strict=strict_flag)
        else:
            state = torch.load(ckpt_file, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            sae.load_state_dict(state, strict=strict if strict is not None else decoder)

        if dtype is not None:
            sae = sae.to(dtype=dtype)
        sae = sae.to(device=device)

        print(f"[SAE] loaded from {path} | d_in={d_in}, "
              f"num_latents={sae.num_latents}, k={cfg.k} | "
              f"decoder_backend={_DECODER_BACKEND}")
        return sae

    # 兼容 sparsify 原命名
    load_from_disk = from_pretrained

    def save_to_disk(self, path: Union[Path, str]):
        try:
            from safetensors.torch import save_model
        except ImportError:
            raise ImportError("safetensors not installed; run `pip install safetensors`")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump({**self.cfg.to_dict(), "d_in": self.d_in}, f)


# 兼容命名: 有人可能用大写 Sae
Sae = SAE