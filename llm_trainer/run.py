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
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    # 多卡 DDP: 每张卡加载完整模型, device_map 不能用 auto, 用单 device
    # 单卡:     沿用 cfg 里的 device_map (默认 auto)
    if is_dist:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cfg.model.device_map = {"": local_rank}
        if is_main:
            print("[run] DDP mode: each process loads full model to its own GPU",
                  flush=True)

    model, processor = load_model(cfg)
    if args.ckpt:
        model = load_adapter(model, args.ckpt)

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