"""
model.py
----------
路径布局(项目根: code/VL-SAE-main/lvlms/):
  pretrained_models/llama3-llava-next-8b-hf/     <- 基座 LLaVA(只读)
  sae_trainer/
    sae_model.py                                  <- 你已有的 SAE 类定义
    finetune_models/
      llama3-llava-next-8b-hf-sae-131k/           <- 训练好的 SAE 权重
  llm_trainer/
    model.py                                      <- 本文件

负责: LLaVA 加载 + LoRA 注入 + 复用 sae_trainer 里的 SAE 类 + hook 挂载到第 24 层
原则: 不修改 pretrained_models/ 与 sae_trainer/ 下任何文件,所有改动以 adapter/hook 形式叠加
"""
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)
from peft import LoraConfig, get_peft_model, PeftModel


# ============================================================
# 0. 路径常量(相对 llm_trainer/ 目录)
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_LVLMS_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

DEFAULT_LLAVA_PATH = os.path.join(_LVLMS_ROOT, "pretrained_models", "llama3-llava-next-8b-hf")
DEFAULT_SAE_MODULE = os.path.join(_LVLMS_ROOT, "sae_trainer", "sae_model.py")
DEFAULT_SAE_CKPT_DIR = os.path.join(_LVLMS_ROOT, "sae_trainer", "finetune_models",
                                    "llama3-llava-next-8b-hf-sae-131k")


# ============================================================
# 0.5 双向注意力改造 (causal -> bidirectional)
# ============================================================
def _find_llama_model(model):
    """
    从可能被 PeftModel / LlavaNext 多层包装的 model 中定位到真正的
    LlamaModel (含 .layers 和 ._update_causal_mask 的那个对象)。
    """
    # 先剥掉 PeftModel 的外壳
    base = model.base_model.model if isinstance(model, PeftModel) else model

    # 再从 LlavaNext 结构中找 language_model.model (= LlamaModel)
    for attr_chain in [
        ("language_model", "model"),
        ("language_model",),
        ("model", "language_model", "model"),
        ("model", "language_model"),
    ]:
        obj = base
        try:
            for a in attr_chain:
                obj = getattr(obj, a)
            # 确认找到的确实是 LlamaModel (有 layers 和 _update_causal_mask)
            if hasattr(obj, "layers") and hasattr(obj, "_update_causal_mask"):
                return obj
        except AttributeError:
            continue
    return None


def _enable_bidirectional_attention(model):
    """
    把 LLaMA 的因果注意力改成双向注意力,适配检索/embedding 任务。

    改动点:
      1) monkey-patch _update_causal_mask -> 返回 None (不施加下三角 mask)
         padding 位置的屏蔽由外部 attention_mask 处理,不受影响。
      2) 每层 self_attn.is_causal = False (SDPA 路径需要,否则它会自己加 causal mask)

    对 trainer / SAE hook 完全透明: 它们只看 hidden state 输出,不关心注意力类型。
    """
    base_lm = _find_llama_model(model)
    if base_lm is None:
        print("[model] WARNING: 无法定位 LlamaModel, 跳过双向注意力改造", flush=True)
        return

    # 1) 干掉 causal mask 生成
    def _no_causal_mask(self, *args, **kwargs):
        return None

    base_lm._update_causal_mask = types.MethodType(_no_causal_mask, base_lm)

    # 2) 各层 attention 的 is_causal 标志也关掉
    n_patched = 0
    for layer in base_lm.layers:
        if hasattr(layer.self_attn, "is_causal"):
            layer.self_attn.is_causal = False
            n_patched += 1

    print(f"[model] bidirectional attention enabled "
          f"(_update_causal_mask=None, is_causal=False on {n_patched} layers)",
          flush=True)


# ============================================================
# 1. 基座模型加载
# ============================================================
def load_model(cfg):
    """加载 LLaVA-Next 并按配置注入 LoRA"""
    dtype = getattr(torch, cfg.model.torch_dtype)
    base_path = cfg.model.base_path or DEFAULT_LLAVA_PATH

    print(f"[model] loading processor from {base_path}", flush=True)
    processor = LlavaNextProcessor.from_pretrained(base_path)

    print(f"[model] loading LLaVA-Next weights (this may take a few minutes)...",
          flush=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map=cfg.model.device_map,
    )
    print(f"[model] LLaVA loaded, dtype={dtype}, device_map={cfg.model.device_map}",
          flush=True)

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print(f"[model] gradient checkpointing enabled", flush=True)

    if cfg.model.lora.enable:
        lora_cfg = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            lora_dropout=cfg.model.lora.dropout,
            target_modules=list(cfg.model.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[model] LoRA injected: r={cfg.model.lora.r}, alpha={cfg.model.lora.alpha}, "
              f"targets={list(cfg.model.lora.target_modules)}", flush=True)
        model.print_trainable_parameters()

    # ---- 双向注意力: 默认关闭, 可通过 cfg.model.bidirectional=true 开启 ----
    if getattr(cfg.model, "bidirectional", False):
        _enable_bidirectional_attention(model)

    return model, processor


def load_adapter(model, adapter_path):
    """从已有 adapter 目录加载 LoRA 权重(用于续训/评估)"""
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=True)


# ============================================================
# 2. 动态导入 sae_trainer/sae_model.py 里的 SAE 类
# ============================================================
def _import_sae_module(module_path):
    """从绝对路径动态导入 sae_model.py,避免在 llm_trainer/ 里重复定义 SAE"""
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"sae_model.py not found: {module_path}")

    # 把 sae_trainer/ 加进 sys.path,让 sae_model.py 内部的相对导入也能工作
    sae_dir = os.path.dirname(module_path)
    if sae_dir not in sys.path:
        sys.path.insert(0, sae_dir)

    spec = importlib.util.spec_from_file_location("sae_model_external", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_sae_class(sae_module, class_name=None):
    """
    在导入的 module 里找 SAE 类。
    优先用 cfg 指定的 class_name,否则按常见命名尝试,最后兜底找 nn.Module 子类。
    """
    candidates = []
    if class_name:
        candidates.append(class_name)
    candidates += ["SAE", "SparseAutoencoder", "AutoEncoder", "VLSAE",
                   "TopKSAE", "JumpReLUSAE"]

    for name in candidates:
        if hasattr(sae_module, name):
            return getattr(sae_module, name)

    for name in dir(sae_module):
        obj = getattr(sae_module, name)
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
            return obj

    raise ImportError(
        f"未能在 {sae_module.__file__} 中找到 SAE 类。"
        f"请在 configs 中通过 sae.class_name 指定。"
    )


# ============================================================
# 3. SAE 权重加载
# ============================================================
def _find_ckpt_file(ckpt_dir):
    """在权重目录下找 .pt / .bin / .safetensors"""
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"SAE ckpt dir not found: {ckpt_dir}")

    preferred = ["sae.pt", "model.pt", "pytorch_model.bin",
                 "sae.safetensors", "model.safetensors", "checkpoint.pt"]
    for name in preferred:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p

    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith((".pt", ".bin", ".safetensors")):
            return os.path.join(ckpt_dir, f)

    raise FileNotFoundError(f"在 {ckpt_dir} 中找不到 SAE 权重文件")


def _load_state_dict(path):
    """统一加载 .pt / .bin / .safetensors"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def load_sae(cfg, device, dtype=torch.float32):
    """
    从 sae_trainer/finetune_models/llama3-llava-next-8b-hf-sae-131k/model.layers.X
    加载 SAE。

    流程: 动态导入 sae_model.py -> 找 SAE 类 ->
          如果有 from_pretrained 入口直接用它 (推荐路径) ->
          否则退化到通用实例化逻辑 (按签名匹配参数名)
    """
    if not cfg.sae.enable:
        return None

    # 1) 导入 sae_model.py
    module_path = cfg.sae.module_path or DEFAULT_SAE_MODULE
    sae_module = _import_sae_module(module_path)
    SAECls = _resolve_sae_class(sae_module, getattr(cfg.sae, "class_name", None))

    # 2) 解析权重路径
    ckpt_path = cfg.sae.ckpt_path or DEFAULT_SAE_CKPT_DIR

    # 自动兜底: 如果 ckpt_path 下没有 cfg.json 但有 model.layers.{idx} 子目录, 自动进入
    # (兼容 sparsify / multimodal-sae 的多层 SAE 仓库布局)
    if os.path.isdir(ckpt_path) and not os.path.isfile(os.path.join(ckpt_path, "cfg.json")):
        layer_subdir = f"model.layers.{cfg.sae.layer_idx}"
        candidate = os.path.join(ckpt_path, layer_subdir)
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "cfg.json")):
            print(f"[SAE] auto-resolved ckpt_path: {ckpt_path} -> {candidate}", flush=True)
            ckpt_path = candidate

    # 3) 优先路径: 如果 SAE 类提供了 from_pretrained / load_from_disk, 直接用
    #    这两个方法内部会自己读 cfg.json + 实例化 + 加载权重, 不需要外部知道任何参数
    for entry_name in ("from_pretrained", "load_from_disk"):
        entry = getattr(SAECls, entry_name, None)
        if callable(entry) and os.path.isdir(ckpt_path):
            print(f"[SAE] using {SAECls.__name__}.{entry_name}({ckpt_path})", flush=True)
            sae = entry(ckpt_path, device=device, dtype=dtype)
            for p in sae.parameters():
                p.requires_grad = bool(cfg.sae.trainable)
            return sae

    # 4) 退化路径: 通用实例化 (用于没有 from_pretrained 的自定义 SAE 类)
    ckpt_file = _find_ckpt_file(ckpt_path)
    state = _load_state_dict(ckpt_file)

    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        saved_cfg = state.get("config", {}) or state.get("cfg", {})
    else:
        sd = state
        saved_cfg = {}

    # 实例化 SAE: 根据真实 __init__ 签名做参数名映射
    import inspect
    sig = inspect.signature(SAECls.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}

    raw_kwargs = dict(getattr(cfg.sae, "init_kwargs", {}) or {})
    if not raw_kwargs and saved_cfg:
        raw_kwargs = dict(saved_cfg)
    if not raw_kwargs:
        for key in ["encoder.weight", "W_enc", "encoder.0.weight"]:
            if key in sd:
                w = sd[key]
                raw_kwargs = {"d_in": w.shape[1], "d_hidden": w.shape[0]}
                break

    aliases = {
        "d_in":     ["d_in", "input_dim", "in_features", "n_inputs"],
        "d_hidden": ["num_latents", "d_hidden", "hidden_dim", "hidden_size",
                     "d_sae", "n_latents", "expansion_factor"],
        "k":        ["k", "top_k", "n_active"],
    }
    init_kwargs = {}
    for canonical, values in raw_kwargs.items():
        candidates = aliases.get(canonical, [canonical])
        if canonical not in aliases:
            if canonical in accepted_params:
                init_kwargs[canonical] = values
            continue
        matched = next((c for c in candidates if c in accepted_params), None)
        if matched is not None:
            init_kwargs[matched] = values

    try:
        sae = SAECls(**init_kwargs)
    except TypeError as e:
        raise TypeError(
            f"实例化 {SAECls.__name__} 失败,init_kwargs={init_kwargs}, "
            f"该类 __init__ 接受的参数: {sorted(accepted_params)}。"
            f"建议: 给 SAE 类加一个 from_pretrained 类方法, "
            f"或在 configs 的 sae.init_kwargs 中显式指定。原始错误: {e}"
        )

    missing, unexpected = sae.load_state_dict(sd, strict=False)
    if missing:
        print(f"[SAE load] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}", flush=True)
    if unexpected:
        print(f"[SAE load] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}", flush=True)

    sae = sae.to(device=device, dtype=dtype)
    for p in sae.parameters():
        p.requires_grad = bool(cfg.sae.trainable)
    print(f"[SAE] loaded {SAECls.__name__} from {ckpt_file}"
          f" -> device={device}, trainable={cfg.sae.trainable}", flush=True)
    return sae


# ============================================================
# 4. SAE Hook 挂载(默认挂在第 24 层)
# ============================================================
class SAEHook:
    """
    通过 forward_hook 在指定 layer 抓取 hidden state,送入 SAE。
    SAE 输出缓存到 self.cache,供 trainer 计算 SAE 引导损失。

    兼容多种 SAE forward 签名:
      - 返回 (x_hat, z)
      - 返回 dict: {"x_hat": ..., "z": ...} / {"reconstruction": ..., "features": ...}
      - 返回单个 x_hat
    """
    def __init__(self, sae, replace=False):
        self.sae = sae
        self.replace = replace
        self.cache = {}
        self.handle = None

    @staticmethod
    def _parse_sae_output(out):
        if isinstance(out, tuple):
            x_hat = out[0]
            z = out[1] if len(out) > 1 else None
        elif isinstance(out, dict):
            x_hat = out.get("x_hat") or out.get("reconstruction") or out.get("recon")
            z = out.get("z") or out.get("features") or out.get("latents") or out.get("code")
        else:
            x_hat, z = out, None
        return x_hat, z

    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        sae_dtype = next(self.sae.parameters()).dtype

        # 关键: 不 detach! 即使 SAE 不可训练, 也要让梯度通过 SAE.encode 
        # 回传到上游 LLaVA (用于 SAE 引导的 retrieval loss)。
        # SAE 参数 requires_grad=False 时, 梯度只会穿过它而不更新它。
        x = hidden.to(sae_dtype)

        # 优先用 encode + decode 路径(跳过 SAE 的训练专用 loss 计算, 如 fvu/auxk_loss)
        if hasattr(self.sae, "encode") and hasattr(self.sae, "decode"):
            # 关键省显存: SAE encode 内部要算预激活 [M, L, num_latents=131072],
            # M=max_candidates+1 (如 17) 时一次性算 = [17,L,131072] ~13GB + relu 副本 -> OOM。
            # 按第 0 维 (M 条序列) 分块 encode, 每块算完 topk 得到稀疏小结果立刻释放预激活,
            # 峰值降到 1/n_chunks。保住 max_candidates 不变, 只是分批过 SAE。
            M = x.shape[0]
            # 每块最多 chunk_sz 条序列 (经验值: 4 条带图 L~1490 时预激活 ~3GB, 安全)
            chunk_sz = getattr(self, "encode_chunk_size", 4)
            if M <= chunk_sz:
                enc = self.sae.encode(x)
                if isinstance(enc, tuple):
                    top_acts, top_indices = enc[0], enc[1]
                else:
                    top_acts, top_indices = enc, None
            else:
                acts_list, idx_list = [], []
                for s in range(0, M, chunk_sz):
                    e = min(s + chunk_sz, M)
                    enc_c = self.sae.encode(x[s:e])
                    if isinstance(enc_c, tuple):
                        a_c, i_c = enc_c[0], enc_c[1]
                    else:
                        a_c, i_c = enc_c, None
                    acts_list.append(a_c)
                    if i_c is not None:
                        idx_list.append(i_c)
                top_acts = torch.cat(acts_list, dim=0)
                top_indices = torch.cat(idx_list, dim=0) if idx_list else None
                del acts_list, idx_list

            if top_indices is not None and self.replace:
                x_hat = self.sae.decode(top_acts, top_indices)
            elif self.replace:
                # 没法只解码,fallback 到完整 forward
                sae_out = self.sae(x)
                x_hat, top_acts = self._parse_sae_output(sae_out)
                top_indices = None
            else:
                # 不需要替换 hidden, 也就不必解码
                x_hat = None

            self.cache = {
                "x": x,
                "x_hat": x_hat,
                "z": top_acts,              # 稀疏 top-K 激活 [..., K]
                "z_indices": top_indices,   # 对应下标 [..., K]
            }
        else:
            # 没有 encode/decode 接口, 退回到完整 forward
            sae_out = self.sae(x)
            x_hat, z = self._parse_sae_output(sae_out)
            self.cache = {"x": x, "x_hat": x_hat, "z": z}

        if self.replace and self.cache.get("x_hat") is not None:
            new_hidden = self.cache["x_hat"].to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden
        return output

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.cache = {}


def attach_sae_hook(model, sae, cfg):
    """
    把 SAE 挂到 cfg.sae.layer_idx 指定的 transformer 层(默认第 24 层)。
    LLaVA-Next 的 LLM decoder 层在 model.language_model.model.layers[i] 下。
    """
    if sae is None:
        return None

    base = model.base_model.model if isinstance(model, PeftModel) else model

    layers = None
    for path in [
        lambda m: m.language_model.model.layers,    # LLaVA-Next 标准路径
        lambda m: m.language_model.layers,
        lambda m: m.model.language_model.model.layers,
        lambda m: m.model.layers,
    ]:
        try:
            layers = path(base)
            break
        except AttributeError:
            continue

    if layers is None:
        raise RuntimeError("无法定位 LLM 的 decoder layers,请检查模型结构后调整 attach_sae_hook")

    idx = cfg.sae.layer_idx
    if idx < 0 or idx >= len(layers):
        raise IndexError(f"layer_idx={idx} 越界, LLM 共 {len(layers)} 层")

    layer = layers[idx]
    hook = SAEHook(sae, replace=bool(getattr(cfg.sae, "replace_hidden", False)))
    hook.handle = layer.register_forward_hook(hook._hook_fn)
    print(f"[SAE] hook attached to LLM layer {idx} (replace={hook.replace})", flush=True)
    return hook