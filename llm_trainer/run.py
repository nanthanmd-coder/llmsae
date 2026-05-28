"""
run.py
----------
统一入口。支持单卡和多卡 DDP。

单卡:
  python run.py --config configs/exp_lora_sae.yaml --mode train

多卡 (2 张卡):
  accelerate launch --num_processes 2 --num_machines 1 \
      --mixed_precision bf16 \
      run.py --config configs/exp_lora_sae.yaml --mode train

评估:
  python run.py --config configs/exp_lora_sae.yaml --mode eval \
                --ckpt outputs/.../adapter_best
"""
import os
import argparse
import random
import numpy as np
import torch
from omegaconf import OmegaConf

from model import load_model, load_sae, attach_sae_hook, load_adapter
from data import MMEBRetrievalDataset, build_collate_fn
from trainer import Trainer

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        base_files = cfg.pop("defaults")
        merged = OmegaConf.create({})
        cfg_dir = os.path.dirname(path)
        for name in base_files:
            base_path = os.path.join(cfg_dir, f"{name}.yaml")
            merged = OmegaConf.merge(merged, OmegaConf.load(base_path))
        cfg = OmegaConf.merge(merged, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def is_distributed_launch():
    """检测是否通过 accelerate launch / torchrun 启动 (多卡)"""
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    # ---- 创建 accelerator (单卡多卡都创建, 单卡时它就是个空壳) ----
    accelerator = None
    if _HAS_ACCELERATE:
        # static_graph=True: 解决 "marked ready twice" 错误。原因是 SAE 路径和 orig
        #   路径都经过 layer 24 的 LoRA 参数, 两条路径 backward 让 DDP 看到该参数被标记
        #   两次。叠加 gradient checkpointing 的 reentrant backward, DDP 默认不支持。
        #   静态图模式让 DDP 知道每 iter 图结构不变, 正确处理参数复用。
        # 注意: static_graph 与 find_unused_parameters 不能同时为 True, 关掉后者
        #   (静态图模式下 DDP 自己会在第一个 iter 探测未用参数)。
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
            static_graph=True,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            kwargs_handlers=[ddp_kwargs],
        )

    is_main = accelerator is None or accelerator.is_main_process
    is_dist = is_distributed_launch()

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))
        np_ = accelerator.num_processes if accelerator else 1
        print(f"[run] num_processes={np_}, distributed={is_dist}, mode={args.mode}",
              flush=True)

    # ---- 模型加载 ----
    # 多卡 DDP: 每张卡加载完整模型到自己的 GPU
    # 单卡:     强制整个模型到 cuda:0 (device_map=auto 会把层拆到多设备甚至 CPU
    #           offload, 训练时 forward 在设备间来回搬, 极慢甚至卡死)
    if is_dist:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cfg.model.device_map = {"": local_rank}
        if is_main:
            print("[run] DDP mode: each process loads full model to its own GPU",
                  flush=True)
    else:
        # 单卡: 强制单 device, 不用 auto
        if torch.cuda.is_available():
            cfg.model.device_map = {"": 0}
            print("[run] 单卡: device_map -> {'': 0} (避免 auto 拆分/CPU offload 卡死)",
                  flush=True)

    # ---- 关键: 续训时不在 load_model 里注入 LoRA, 由 load_adapter 从 checkpoint 加载 ----
    if args.ckpt:
        cfg.model.lora.enable = False          # 禁止 load_model 注入新 LoRA
        model, processor = load_model(cfg)     # 只加载基座 LLaVA
        model = load_adapter(model, args.ckpt) # 从 checkpoint 加载 LoRA (is_trainable=True)
    else:
        model, processor = load_model(cfg)

    # ---- 关键省显存: 训练/评估只用中间层 hidden (通过 hook), 完全不用 lm_head 的 logits。
    # LLaVA forward 默认会算 logits [M, L, vocab=128256], transformers v4.46+ 训练时
    # 强制 FP32 -> 单这一项 ~13GB, DDP 的 _DDPSink 还会 clone 一份 -> ~26GB -> OOM。
    # 把 lm_head 换成 Identity: forward 到 layer 24 的 hook 照常抓 hidden, 但不再算
    # 大 logits, DDP clone 的也只是小张量。layer 0~24 计算图完整, 梯度照常回传 LoRA。
    def _replace_lm_head_with_identity(m):
        base = m.base_model.model if hasattr(m, "base_model") else m
        for path in [
            lambda x: x.language_model,           # LLaVA-Next: language_model.lm_head
            lambda x: x.model.language_model,
            lambda x: x,
        ]:
            try:
                lm = path(base)
                if hasattr(lm, "lm_head") and not isinstance(lm.lm_head, torch.nn.Identity):
                    old = lm.lm_head
                    lm.lm_head = torch.nn.Identity()
                    if is_main:
                        print(f"[run] lm_head -> Identity (省 logits 显存), "
                              f"原 lm_head: {type(old).__name__}", flush=True)
                    return True
            except AttributeError:
                continue
        if is_main:
            print("[run] 警告: 没找到 lm_head, 未替换 (logits 仍会算, 可能 OOM)", flush=True)
        return False
    _replace_lm_head_with_identity(model)

    # SAE 跟 model 放同一张卡
    device = next(model.parameters()).device
    sae = load_sae(cfg, device=device)
    sae_hook = attach_sae_hook(model, sae, cfg) if sae is not None else None

    # ---- 数据 ----
    collate_fn = build_collate_fn(processor, max_length=cfg.data.max_length)

    train_ds, eval_ds = None, None
    if cfg.data.train_datasets is not None or args.mode == "train":
        train_ds = MMEBRetrievalDataset(
            data_root=cfg.data.data_root,
            dataset_names=cfg.data.train_datasets,
            images_root=cfg.data.images_root,
            max_candidates=cfg.data.max_candidates,
            gold_idx_col=cfg.data.gold_idx_col,
        )
    if cfg.data.eval_datasets is not None:
        eval_ds = MMEBRetrievalDataset(
            data_root=cfg.data.data_root,
            dataset_names=cfg.data.eval_datasets,
            images_root=cfg.data.images_root,
            max_candidates=cfg.data.max_candidates,
            gold_idx_col=cfg.data.gold_idx_col,
        )

    trainer = Trainer(
        cfg=cfg, model=model, processor=processor,
        sae=sae, sae_hook=sae_hook,
        train_dataset=train_ds, eval_dataset=eval_ds,
        collate_fn=collate_fn,
        accelerator=accelerator,
    )

    if args.mode == "train":
        trainer.fit()
    elif args.mode == "eval":
        if eval_ds is None:
            raise ValueError("eval 模式需要在配置里设置 data.eval_datasets")
        trainer.evaluate()

    if sae_hook is not None:
        sae_hook.remove()


if __name__ == "__main__":
    main()