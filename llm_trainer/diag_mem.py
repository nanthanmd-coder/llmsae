"""
diag_mem.py — candidate 处理流程内存诊断
------------------------------------------------
尽可能还原 eval_zero_shot.py 的 candidate 处理链路, 每一步打印:
  - 张量 shape (L, K 到底多大)
  - 该步 GPU 显存增量 + CPU 内存增量
  - 这一步为什么占内存的原因分析

用法 (单卡, 小数据集):
  CUDA_VISIBLE_DEVICES=0 python diag_mem.py --config configs/exp_lora_sae.yaml \
       --dataset A-OKVQA --n 20

输出自动同时写到 diag_mem.log (可改 --log)。
关注输出里的:
  [STEP] ... 每步内存增量
  [WHY]  ... 内存归因
  最后的 [VERDICT] 总结哪一步是大头 + 全量预估
"""
import os, argparse, time, gc, sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from model import load_model, load_sae, attach_sae_hook
from data import (_is_nonempty_str, _normalize_to_list,
                  _resolve_image_path, _find_parquet_files)


class _Tee:
    """把 stdout 同时写到终端和日志文件"""
    def __init__(self, logpath):
        self.terminal = sys.stdout
        self.log = open(logpath, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ---------- 内存工具 ----------
def cpu_mem_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        # 退化: 读 /proc/self/status
        try:
            with open(f"/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1e6  # kB -> GB
        except Exception:
            return -1.0


def gpu_mem_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / 1e9
    free, total = torch.cuda.mem_get_info()
    return alloc, (total - free) / 1e9   # (torch已分配, 整卡已用含其他进程)


class MemTracker:
    def __init__(self):
        self.last_gpu = gpu_mem_gb()[0]
        self.last_cpu = cpu_mem_gb()
        self.steps = []

    def step(self, name, why=""):
        g_alloc, g_used = gpu_mem_gb()
        c = cpu_mem_gb()
        dg = g_alloc - self.last_gpu
        dc = c - self.last_cpu
        print(f"\n[STEP] {name}")
        print(f"   GPU torch-alloc={g_alloc:.2f}GB (Δ{dg:+.2f}GB)  "
              f"整卡已用={g_used:.2f}GB")
        print(f"   CPU RSS={c:.2f}GB (Δ{dc:+.2f}GB)")
        if why:
            print(f"   [WHY] {why}")
        self.steps.append((name, g_alloc, c, dg, dc))
        self.last_gpu = g_alloc
        self.last_cpu = c


def load_config(path):
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        base_files = cfg.pop("defaults")
        merged = OmegaConf.create({})
        cfg_dir = os.path.dirname(path)
        for name in base_files:
            merged = OmegaConf.merge(merged, OmegaConf.load(
                os.path.join(cfg_dir, f"{name}.yaml")))
        cfg = OmegaConf.merge(merged, cfg)
    OmegaConf.resolve(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=20, help="只取前 n 个 candidate 诊断")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--log", default="diag_mem.log", help="日志文件路径")
    args = ap.parse_args()

    # 所有 print 同时写终端 + 日志文件
    sys.stdout = _Tee(args.log)
    print(f"[日志] 输出同时写到: {args.log}", flush=True)

    tr = MemTracker()
    print("=" * 60)
    print(f"诊断: dataset={args.dataset}, n_cand={args.n}, batch={args.batch_size}")
    print("=" * 60)
    tr.step("启动 (import 完成)",
            why="基线; CPU 里是 python+torch+库, GPU 还没东西")

    cfg = load_config(args.config)
    if torch.cuda.is_available():
        cfg.model.device_map = {"": 0}

    # ---- 模型 ----
    t = time.time()
    model, processor = load_model(cfg)
    device = next(model.parameters()).device
    tr.step(f"加载模型 ({time.time()-t:.0f}s)",
            why=f"LLaVA-Next-8B 权重上 GPU。device={device}。"
                f"bf16 下 8B≈16GB + 视觉塔")

    # ---- SAE ----
    sae = load_sae(cfg, device=device)
    sae_hook = attach_sae_hook(model, sae, cfg) if sae is not None else None
    tr.step("加载 SAE + hook",
            why="SAE 权重 (decoder/encoder 矩阵) 上 GPU。"
                "131k latents 的 encoder ~ 4096×131072×2B ≈ 1GB 量级")

    # ---- 取 candidate ----
    ds_dir = os.path.join(cfg.data.data_root, args.dataset)
    files = _find_parquet_files(ds_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cand_dict = {}
    for _, row in df.iterrows():
        tgt_texts = _normalize_to_list(row["tgt_text"])
        tgt_imgs = _normalize_to_list(row["tgt_img_path"])
        for k in range(max(len(tgt_texts), len(tgt_imgs))):
            t_ = tgt_texts[k] if k < len(tgt_texts) and _is_nonempty_str(tgt_texts[k]) else ""
            i_ = tgt_imgs[k] if k < len(tgt_imgs) and _is_nonempty_str(tgt_imgs[k]) else ""
            if not t_ and not i_:
                continue
            key = (t_, i_)
            if key not in cand_dict:
                cand_dict[key] = len(cand_dict)
            if len(cand_dict) >= args.n:
                break
        if len(cand_dict) >= args.n:
            break
    cands = [{"text": t_, "image_path":
              _resolve_image_path(i_, cfg.data.data_root, cfg.data.images_root) if i_ else None}
             for (t_, i_) in cand_dict.keys()]
    n_img = sum(1 for c in cands if c["image_path"])
    tr.step(f"读 parquet + 解析 {len(cands)} 个 candidate ({n_img} 个带图)",
            why="纯 CPU; parquet 元数据。图还没加载")

    # ---- 逐 batch forward, 详细看 ta.shape ----
    from eval_zero_shot import UnitDataset, build_collate
    collate = build_collate(processor, max_length=cfg.data.max_length)
    loader = DataLoader(UnitDataset([
        {**c, "image_path": c["image_path"], "cid": f"c_{i}"}
        for i, c in enumerate(cands)], ), batch_size=args.batch_size,
        shuffle=False, num_workers=2, collate_fn=collate)

    _cache = {}
    base = model.base_model.model if hasattr(model, "base_model") else model
    layers = None
    for path in [lambda m: m.language_model.model.layers,
                 lambda m: m.language_model.layers,
                 lambda m: m.model.language_model.model.layers]:
        try:
            layers = path(base); break
        except AttributeError:
            continue
    pool_idx = getattr(cfg.train, "pool_layer_idx", cfg.sae.layer_idx)
    def _hk(m, i, o):
        _cache["h"] = (o[0] if isinstance(o, tuple) else o).detach()
        return o
    handle = layers[pool_idx].register_forward_hook(_hk)

    saes_gpu_bytes = 0
    saes_nz_total = 0
    L_seen = []
    K_seen = None
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch.pop("ids")
            batch = {k: v.to(device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            seq_len = batch["input_ids"].shape[1]
            _cache.clear()
            # 抓 forward 期间的显存峰值 (含 SAE 预激活 [B,L,131072] 这种瞬时大张量)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                before_fwd = torch.cuda.memory_allocated() / 1e9
            _ = model(**batch, use_cache=False)
            if torch.cuda.is_available():
                peak_fwd = torch.cuda.max_memory_allocated() / 1e9
            if bi == 0:
                tr.step(f"第1个 batch forward (seq_len={seq_len})",
                        why=f"forward 激活值占显存; batch={args.batch_size} × "
                            f"seq_len={seq_len} × 隐藏维。这是 forward 瞬时峰值")
                if torch.cuda.is_available():
                    fwd_peak_delta = peak_fwd - before_fwd
                    # SAE 预激活理论值: [B, L, n_latents]
                    n_lat = 131072
                    pre_act_fp32 = args.batch_size * seq_len * n_lat * 4 / 1e9
                    print(f"\n   [SAE预激活] forward 瞬时峰值增量 = {fwd_peak_delta:.1f}GB")
                    print(f"   [SAE预激活] 理论 [B={args.batch_size}, L={seq_len}, "
                          f"131072] fp32 = {pre_act_fp32:.1f}GB")
                    print(f"   [SAE预激活] forward 后回落到 = "
                          f"{torch.cuda.memory_allocated()/1e9:.1f}GB")
                    print(f"   [WHY] SAE 算 top-k 前要先算全部 131072 个 latent 预激活,"
                          f" 这个瞬时张量 ∝ B×L, batch 越大越猛")

            if sae_hook is not None and sae_hook.cache:
                ta = sae_hook.cache.get("z")
                ti = sae_hook.cache.get("z_indices")
                if ta is not None:
                    B, L, K = ta.shape
                    K_seen = K
                    L_seen.append(L)
                    # 模拟旧版: 整个 [B, L*K] 留 GPU
                    bytes_old = B * L * K * (4 + 8)  # acts f32 + inds i64
                    # 模拟新版: 去 padding + 去零 + token->样本级 pool
                    am_b = batch["attention_mask"]
                    nz_raw = 0       # 去 padding/零后, pool 前
                    nz_pooled = 0    # token->样本级 pool 后 (unique feature)
                    for b in range(B):
                        valid = am_b[b].bool()
                        a = ta[b][valid].reshape(-1).float()
                        idx = ti[b][valid].reshape(-1).long()
                        nz_raw += int((a != 0).sum().item())
                        # pool: 同一 feature 累加
                        uniq, inv = torch.unique(idx, return_inverse=True)
                        pooled = torch.zeros(uniq.shape[0], device=a.device, dtype=a.dtype)
                        pooled.scatter_add_(0, inv, a)
                        nz_pooled += int((pooled != 0).sum().item())
                    if bi == 0:
                        print(f"\n   [关键] SAE z.shape = [B={B}, L={L}, K={K}]")
                        print(f"   [关键] 旧版每样本 L×K = {L*K} (含 padding+零)")
                        print(f"   [关键] 去padding+去零后每 batch = {nz_raw} "
                              f"(压缩 {B*L*K/max(nz_raw,1):.1f}x)")
                        print(f"   [关键] token->样本级 pool 后每 batch = {nz_pooled} "
                              f"(再压缩 {nz_raw/max(nz_pooled,1):.1f}x, "
                              f"总 {B*L*K/max(nz_pooled,1):.1f}x)")
                        print(f"   [WHY] 带图样本激活几乎全非零, 去零没用; "
                              f"但 pool 把同 feature 跨 token 合并, 大幅降 nnz")
                    saes_gpu_bytes += bytes_old
                    saes_nz_total += nz_pooled
            if bi >= 2:   # 诊断只跑前 3 个 batch
                break
    handle.remove()

    avg_L = sum(L_seen) / len(L_seen) if L_seen else 0
    tr.step("跑完 3 个 batch (诊断采样)",
            why="看上面 [关键] 行的 L/K/压缩比")

    # ---- 全量预估 ----
    N_full = len(cand_dict)   # 实际可能更多, 这里用诊断的 n
    # 用采样的平均推全量
    print("\n" + "=" * 60)
    print("[VERDICT] 内存归因 + 全量预估")
    print("=" * 60)
    print(f"  实测: L(有效/pad后)≈{avg_L:.0f}, K(top-k)={K_seen}")
    if K_seen:
        per_cand_old = avg_L * K_seen * 12 / 1e6      # MB, 旧版
        per_cand_nz = (saes_nz_total / max(len(L_seen)*args.batch_size,1)) * 12 / 1e6
        print(f"\n  旧版 (留GPU, 不去padding/零):")
        print(f"    每个 cand ≈ {per_cand_old:.1f} MB")
        for N in [args.n, 550, 1498, 5000]:
            print(f"    {N} 个 cand ≈ {per_cand_old*N/1024:.1f} GB"
                  + ("  <- 这就是爆 80G 的原因" if per_cand_old*N/1024 > 50 else ""))
        print(f"\n  新版 (去padding+去零+搬CPU):")
        print(f"    每个 cand 真非零 ≈ {per_cand_nz:.2f} MB (在 CPU)")
        for N in [args.n, 550, 1498, 5000]:
            print(f"    {N} 个 cand ≈ {per_cand_nz*N/1024:.2f} GB (CPU内存, GPU=0)")
        print(f"\n  CSR nnz 预估 (全量 {1498} cand):")
        nnz_per = saes_nz_total / max(len(L_seen)*args.batch_size, 1)
        for N in [550, 1498]:
            nnz = nnz_per * N
            print(f"    {N} cand: nnz≈{nnz/1e6:.1f}M, "
                  f"CSR显存≈{nnz*12/1e9:.2f}GB (临时峰值~{nnz*20/1e9:.2f}GB)")

    print(f"\n  [结论]")
    if K_seen and avg_L * K_seen * 12 / 1e6 * 550 / 1024 > 50:
        print(f"    旧版 SAE 激活留 GPU + 没去 padding = 主因。")
        print(f"    L={avg_L:.0f} 太大 (LLaVA anyres 高分图 + batch padding)。")
        print(f"    -> 用新版 (去padding+去零+搬CPU), GPU 应回落到 模型+SAE 基线")
    print("=" * 60)

    # 恢复 stdout, 提示日志位置
    if isinstance(sys.stdout, _Tee):
        logpath = sys.stdout.log.name
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
        print(f"\n[日志] 完整输出已保存到: {logpath}")
        print(f"[日志] 复制: cat {logpath}")


if __name__ == "__main__":
    main()