"""
eval_zero_shot.py
-----------------
零样本 baseline 评估 (GPU 全程版)。

每个 query 的 candidate 池来自该 query 自己的 tgt 列表 (tgt[0]=真值, tgt[1:]=负样本)。
不同 query 的 tgt 经常重复, 所以 unique candidate 只 forward 一次, 用 cid 索引复用。

抽样模拟全局: 抽 sample_ratio 的 candidate (保留真值) 算 rank, rank/ratio 还原全局 rank。
三路对比: orig (原始 hidden) / sae (SAE 稀疏) / avg (两路相似度平均)

性能要点 (全 GPU):
  - emb / SAE 激活 留在 GPU, 不搬 CPU
  - cos_sim / sparse_sim 全在 GPU 上算
  - SAE 用 build_sparse_lookup 整个 candidate 池一次性稠密化到 GPU [N, V]
  - DataLoader num_workers 多线程加载图片

用法:
  python eval_zero_shot.py --config configs/exp_lora_sae.yaml \
                           --datasets A-OKVQA --num_queries 200 --sample_ratio 0.2
"""
import os
import json
import time
import random
import argparse
import threading
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from PIL import Image

from model import load_model, load_sae, attach_sae_hook
from data import (
    _is_nonempty_str, _normalize_to_list,
    _resolve_image_path, _find_parquet_files
)


# ============================================================
# GPU 监控线程
# ============================================================
class GPUMonitor:
    def __init__(self, interval=300):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2)
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ], timeout=3).decode().strip()
                for line in out.splitlines():
                    idx, util, mu, mt, temp, pw = [x.strip() for x in line.split(",")]
                    print(f"  [gpu{idx}] util={util}%  mem={mu}/{mt}MiB  "
                          f"temp={temp}C  power={pw}W", flush=True)
            except Exception as e:
                print(f"  [gpu monitor error] {e}", flush=True)
            if self._stop.wait(self.interval):
                break


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


# ============================================================
# 收集 query + unique candidate (跨 query 去重)
# ============================================================
def collect_units(data_root, dataset_name, images_root=None,
                  num_query_sample=None, seed=42):
    ds_dir = os.path.join(data_root, dataset_name)
    files = _find_parquet_files(ds_dir)
    if not files:
        raise FileNotFoundError(f"no parquet in {ds_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    cand_dict = {}
    def _intern(text, image_path):
        key = (text, image_path)
        if key not in cand_dict:
            cand_dict[key] = f"c_{len(cand_dict)}"
        return cand_dict[key]

    queries = []
    qid2cand_cids = {}
    qid2gold_pos = {}

    for idx, row in df.iterrows():
        qid = f"q_{idx}"
        q_text = row["qry_text"] if _is_nonempty_str(row["qry_text"]) else ""
        q_img = row["qry_img_path"] if _is_nonempty_str(row["qry_img_path"]) else ""
        queries.append({
            "qid": qid, "text": q_text,
            "image_path": _resolve_image_path(q_img, data_root, images_root),
        })

        tgt_texts = _normalize_to_list(row["tgt_text"])
        tgt_imgs = _normalize_to_list(row["tgt_img_path"])
        K = max(len(tgt_texts), len(tgt_imgs))
        cand_cids = []
        for k in range(K):
            t = tgt_texts[k] if k < len(tgt_texts) and _is_nonempty_str(tgt_texts[k]) else ""
            i = tgt_imgs[k] if k < len(tgt_imgs) and _is_nonempty_str(tgt_imgs[k]) else ""
            if not t and not i:
                continue
            cand_cids.append(_intern(t, i))
        qid2cand_cids[qid] = cand_cids
        qid2gold_pos[qid] = 0

    cands = [
        {"cid": cid, "text": t,
         "image_path": _resolve_image_path(i, data_root, images_root) if i else None}
        for (t, i), cid in cand_dict.items()
    ]

    if num_query_sample is not None and num_query_sample < len(queries):
        rng = random.Random(seed)
        queries = rng.sample(queries, num_query_sample)

    return queries, cands, qid2cand_cids, qid2gold_pos


# ============================================================
# Dataset / collate
# ============================================================
class UnitDataset(Dataset):
    def __init__(self, units):
        self.units = units
    def __len__(self):
        return len(self.units)
    def __getitem__(self, idx):
        u = self.units[idx]
        image = None
        if u["image_path"]:
            try:
                img = Image.open(u["image_path"]).convert("RGB")
                w, h = img.size
                if w < 16 or h < 16 or w > 8192 or h > 8192:
                    image = None
                else:
                    image = img
            except Exception:
                image = None
        return {"text": u["text"], "image": image, "id": u.get("qid") or u.get("cid")}


def build_collate(processor, max_length=512):
    def collate(batch):
        ids = [b["id"] for b in batch]
        texts = [b["text"] if b["text"] else "" for b in batch]
        images = [b["image"] for b in batch]

        no_img = [i for i, im in enumerate(images) if im is None]
        with_img = [i for i, im in enumerate(images) if im is not None]
        imgs = [images[i] for i in with_img]

        r_no, r_with = None, None
        if no_img:
            r_no = processor.tokenizer(
                [texts[i] for i in no_img],
                padding="max_length", truncation=True, max_length=max_length,
                return_tensors="pt")
        if with_img:
            txt_img = []
            for i in with_img:
                t = texts[i]
                if "<image>" not in t:
                    t = "<image>\n" + t
                txt_img.append(t)
            try:
                r_with = processor(text=txt_img, images=imgs,
                                   padding=True, truncation=False, return_tensors="pt")
            except Exception as e:
                print(f"[collate] processor failed on images, fallback text-only: {e}",
                      flush=True)
                r_with = processor.tokenizer(
                    txt_img, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt")

        def _L(r): return r["input_ids"].shape[1] if r is not None else 0
        L = max(_L(r_no), _L(r_with))
        pad_id = processor.tokenizer.pad_token_id or 0

        def _pad(t, target, val):
            if t.shape[1] == target: return t
            return torch.cat([t, torch.full((t.shape[0], target - t.shape[1]),
                                            val, dtype=t.dtype)], dim=1)

        M = len(batch)
        input_ids = torch.full((M, L), pad_id, dtype=torch.long)
        attn = torch.zeros((M, L), dtype=torch.long)
        if r_no is not None:
            ii = _pad(r_no["input_ids"], L, pad_id)
            mm = _pad(r_no["attention_mask"], L, 0)
            for k, fi in enumerate(no_img):
                input_ids[fi] = ii[k]; attn[fi] = mm[k]
        out = {"input_ids": input_ids, "attention_mask": attn}
        if r_with is not None:
            ii = _pad(r_with["input_ids"], L, pad_id)
            mm = _pad(r_with["attention_mask"], L, 0)
            for k, fi in enumerate(with_img):
                input_ids[fi] = ii[k]; attn[fi] = mm[k]
            out["pixel_values"] = r_with["pixel_values"]
            if "image_sizes" in r_with:
                out["image_sizes"] = r_with["image_sizes"]
        out["ids"] = ids
        return out
    return collate


# ============================================================
# Extractor: emb 留 GPU, SAE 激活留 GPU
# ============================================================
class Extractor:
    def __init__(self, model, sae_hook, pool_layer_idx, device):
        self.model = model
        self.sae_hook = sae_hook
        self.device = device
        self._cache = {}
        base = model.base_model.model if hasattr(model, "base_model") else model
        layers = None
        for path in [
            lambda m: m.language_model.model.layers,
            lambda m: m.language_model.layers,
            lambda m: m.model.language_model.model.layers,
        ]:
            try:
                layers = path(base); break
            except AttributeError:
                continue
        def _hk(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            self._cache["h"] = h.detach()
            return output
        self._handle = layers[pool_layer_idx].register_forward_hook(_hk)

    @torch.no_grad()
    def encode(self, ds, batch_size, collate, desc="", num_workers=4,
               keep_emb_on_gpu=True):
        """
        返回:
          ids:   list[str]
          origs: [N, D] tensor (GPU if keep_emb_on_gpu)
          saes:  list[(acts_gpu, inds_gpu)]  每个样本的稀疏激活, 留 GPU
        """
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate,
                            pin_memory=True,
                            persistent_workers=(num_workers > 0))
        ids, origs, saes = [], [], []
        n = len(loader)
        t0 = time.time()
        self.model.eval()
        for i, batch in enumerate(loader):
            bids = batch.pop("ids")
            batch = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch.items() if isinstance(v, torch.Tensor)}
            self._cache.clear()
            _ = self.model(**batch, use_cache=False)
            h = self._cache["h"]                       # GPU
            am = batch["attention_mask"].to(h.dtype)
            emb = ((h * am.unsqueeze(-1)).sum(1) /
                   am.sum(1, keepdim=True).clamp_min(1e-6)).float()  # GPU
            if not keep_emb_on_gpu:
                emb = emb.cpu()
            origs.append(emb)

            if self.sae_hook is not None and self.sae_hook.cache:
                ta = self.sae_hook.cache.get("z")
                ti = self.sae_hook.cache.get("z_indices")
                if ta is not None and ti is not None:
                    mask = am.unsqueeze(-1).to(ta.dtype)
                    ta_m = ta * mask / am.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
                    B, L, K = ta.shape
                    pa = ta_m.reshape(B, L * K)        # GPU
                    pi = ti.reshape(B, L * K)          # GPU
                    for b in range(B):
                        # 留在 GPU
                        saes.append((pa[b].clone(), pi[b].clone()))

            ids.extend(bids)
            if (i + 1) % max(n // 10, 1) == 0 or i == n - 1:
                print(f"  [{desc}] {i+1}/{n} ({100*(i+1)/n:.0f}%) "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        origs = torch.cat(origs, dim=0)
        return ids, origs, saes

    def close(self):
        if self._handle is not None:
            self._handle.remove()


# ============================================================
# 相似度 (全 GPU)
# ============================================================
def cos_sim(q, c):
    """q:[D], c:[N,D] -> [N]  (在 q/c 所在 device 上算)"""
    q = F.normalize(q.float().unsqueeze(0), dim=-1).squeeze(0)
    c = F.normalize(c.float(), dim=-1)
    return c @ q


def build_sparse_lookup(sparse_list, device="cuda", chunk_size=128, verbose=True):
    """
    整个 candidate 池一次性稠密化到 GPU [N, V] (处理不等长样本):
      1. 求所有 indices 并集 V
      2. 分块批量 scatter
      3. L2 归一化
    返回: (dense_norm[N,V] GPU, idx_to_pos GPU, unique GPU)
    """
    if len(sparse_list) == 0:
        return None

    N = len(sparse_list)

    # 求并集 (在 GPU 上 cat + unique; sparse_list 已在 GPU)
    t = time.time()
    all_inds = torch.cat([s[1] for s in sparse_list])
    unique = torch.unique(all_inds)
    V = unique.shape[0]
    max_idx = int(unique.max().item())
    del all_inds
    if verbose:
        print(f"    union V={V}, max_idx={max_idx}, took {time.time()-t:.1f}s",
              flush=True)

    idx_to_pos = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
    idx_to_pos[unique] = torch.arange(V, device=device)

    mem_gb = N * V * 4 / 1e9
    if verbose:
        print(f"    allocating dense [N={N}, V={V}] fp32 = {mem_gb:.1f} GB", flush=True)
    dense = torch.zeros(N, V, dtype=torch.float32, device=device)

    t = time.time()
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        max_len = max(sparse_list[i][0].shape[0] for i in range(s, e))
        chunk = e - s
        acts_chunk = torch.zeros(chunk, max_len, dtype=torch.float32, device=device)
        inds_chunk = torch.full((chunk, max_len), int(unique[0].item()),
                                dtype=torch.long, device=device)
        for k, i in enumerate(range(s, e)):
            L_i = sparse_list[i][0].shape[0]
            acts_chunk[k, :L_i] = sparse_list[i][0].float()
            inds_chunk[k, :L_i] = sparse_list[i][1].long()
        pos_chunk = idx_to_pos[inds_chunk]
        dense[s:e].scatter_add_(1, pos_chunk, acts_chunk)
        del acts_chunk, inds_chunk, pos_chunk
        if verbose and (s // chunk_size) % 20 == 0:
            print(f"    scatter {e}/{N} ({100*e/N:.0f}%)", flush=True)
    if verbose:
        print(f"    scatter took {time.time()-t:.1f}s", flush=True)

    dense = dense / (dense.norm(dim=-1, keepdim=True) + 1e-8)
    torch.cuda.empty_cache()
    return dense, idx_to_pos, unique


def sparse_sim_with_lookup(q_sparse, lookup, cand_indices, device="cuda"):
    """用预算好的 lookup 算 query 与 cand 子集余弦 (全 GPU)"""
    c_dense_norm, idx_to_pos, unique = lookup
    acts, inds = q_sparse
    V = c_dense_norm.shape[1]

    acts = acts.float().to(device)
    inds = inds.to(device)
    max_idx_in_table = idx_to_pos.shape[0] - 1
    valid = (inds <= max_idx_in_table) & (idx_to_pos[inds.clamp(max=max_idx_in_table)] >= 0)
    acts_v = acts[valid]
    pos_v = idx_to_pos[inds[valid]]

    q_d = torch.zeros(V, dtype=torch.float32, device=device)
    q_d.scatter_add_(0, pos_v, acts_v)
    q_d = q_d / (q_d.norm() + 1e-8)

    c_subset = c_dense_norm[cand_indices]      # [N_subset, V] GPU
    sims = c_subset @ q_d                      # [N_subset] GPU
    return sims


# ============================================================
# 抽样 + 评估 (一个 query) - 全 GPU
# ============================================================
def eval_one_query(cand_cids, gold_pos, sample_ratio, rng,
                   q_orig, c_orig_all,
                   q_sae=None, sparse_lookup=None,
                   cid2idx=None, device="cuda"):
    K = len(cand_cids)
    min_pool = 20
    if K <= min_pool:
        n_sample = K
    else:
        n_sample = max(min_pool, int(K * sample_ratio))
        n_sample = min(n_sample, K)

    gold_cid = cand_cids[gold_pos]
    other_cids = cand_cids[:gold_pos] + cand_cids[gold_pos + 1:]
    n_dist = n_sample - 1
    if n_dist >= len(other_cids):
        sampled_dist = other_cids[:]
    else:
        sampled_dist = rng.sample(other_cids, n_dist)
    sample_cids = sampled_dist + [gold_cid]
    rng.shuffle(sample_cids)
    new_gold_pos = sample_cids.index(gold_cid)

    cand_idx_in_full = [cid2idx[c] for c in sample_cids]
    cand_idx_tensor = torch.tensor(cand_idx_in_full, dtype=torch.long, device=device)

    # orig 路径 (GPU)
    c_orig_subset = c_orig_all[cand_idx_tensor]      # GPU index
    sims_orig = cos_sim(q_orig, c_orig_subset)        # GPU
    gold_score_orig = sims_orig[new_gold_pos]
    rank_orig = int((sims_orig > gold_score_orig).sum().item()) + 1
    out = {"orig_rank_sample": rank_orig, "n_sample": n_sample, "K": K}

    # sae 路径 (GPU)
    if q_sae is not None and sparse_lookup is not None:
        sims_sae = sparse_sim_with_lookup(q_sae, sparse_lookup, cand_idx_tensor, device)
        gold_score_sae = sims_sae[new_gold_pos]
        rank_sae = int((sims_sae > gold_score_sae).sum().item()) + 1
        out["sae_rank_sample"] = rank_sae

        sims_avg = (sims_orig + sims_sae) / 2
        gold_score_avg = sims_avg[new_gold_pos]
        rank_avg = int((sims_avg > gold_score_avg).sum().item()) + 1
        out["avg_rank_sample"] = rank_avg

    actual_ratio = n_sample / K
    scale = 1.0 / actual_ratio
    out["scale"] = scale
    for key in ("orig", "sae", "avg"):
        sk = f"{key}_rank_sample"
        if sk in out:
            out[f"{key}_rank_global"] = out[sk] * scale
    return out


def aggregate(per_query_results, ks=(1, 5, 10)):
    metrics = {}
    for key in ("orig", "sae", "avg"):
        sk_sample = f"{key}_rank_sample"
        sk_global = f"{key}_rank_global"
        if sk_sample not in per_query_results[0]:
            continue
        ranks_sample = np.array([r[sk_sample] for r in per_query_results])
        ranks_global = np.array([r[sk_global] for r in per_query_results])
        n_samples = np.array([r["n_sample"] for r in per_query_results])

        m = {}
        for k in ks:
            m[f"top{k}"] = float((ranks_sample <= k).mean())
        m["mrr"] = float((1.0 / ranks_sample).mean())
        m["acc"] = float((1.0 - (ranks_sample - 1) / np.maximum(n_samples - 1, 1)).mean())
        for k in ks:
            m[f"top{k}_global"] = float((ranks_global <= k).mean())
        m["mrr_global"] = float((1.0 / ranks_global).mean())
        metrics[key] = m
    return metrics


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", default="outputs/zero_shot")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_sae", action="store_true")
    parser.add_argument("--gpu_monitor_interval", type=float, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[eval] sample_ratio={args.sample_ratio}", flush=True)

    out_json = os.path.join(args.output_dir, "zero_shot_results.json")
    all_results = {}
    if args.resume and os.path.isfile(out_json):
        try:
            with open(out_json) as f:
                all_results = json.load(f)
            print(f"[eval] resume: loaded {len(all_results)} datasets: "
                  f"{list(all_results.keys())}", flush=True)
        except Exception as e:
            print(f"[eval] resume load failed: {e}", flush=True)

    datasets_to_run = [d for d in args.datasets if d not in all_results]
    if not datasets_to_run:
        print("[eval] all datasets done", flush=True)
        try:
            plot_results(all_results, os.path.join(args.output_dir, "zero_shot.png"))
        except Exception as e:
            print(f"[plot] {e}", flush=True)
        return
    print(f"[eval] will run: {datasets_to_run}", flush=True)

    print("[eval] loading model + SAE ...", flush=True)
    if torch.cuda.is_available():
        orig_dm = cfg.model.device_map
        cfg.model.device_map = {"": 0}
        print(f"[eval] override device_map: {orig_dm} -> {{'': 0}}", flush=True)

    model, processor = load_model(cfg)
    if args.ckpt:
        from model import load_adapter
        model = load_adapter(model, args.ckpt)
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(
            f"Model on {device}, not GPU! 检查 torch.cuda / CUDA_VISIBLE_DEVICES / "
            f"device_map (当前 {cfg.model.device_map}). 可见 GPU: {torch.cuda.device_count()}")
    print(f"[eval] model device: {device}", flush=True)

    if args.no_sae:
        print("[eval] --no_sae: orig path only", flush=True)
        sae, sae_hook = None, None
    else:
        sae = load_sae(cfg, device=device)
        sae_hook = attach_sae_hook(model, sae, cfg) if sae is not None else None

    pool_layer_idx = getattr(cfg.train, "pool_layer_idx", cfg.sae.layer_idx)
    extractor = Extractor(model, sae_hook, pool_layer_idx, device)
    collate = build_collate(processor, max_length=cfg.data.max_length)

    gpu_monitor = None
    if args.gpu_monitor_interval > 0:
        gpu_monitor = GPUMonitor(interval=args.gpu_monitor_interval)
        gpu_monitor.start()
        print(f"[eval] GPU monitor every {args.gpu_monitor_interval}s", flush=True)

    for ds_name in datasets_to_run:
        print(f"\n[eval] ====== {ds_name} ======", flush=True)
        try:
            queries, cands, qid2cands, qid2gold_pos = collect_units(
                cfg.data.data_root, ds_name, cfg.data.images_root,
                num_query_sample=args.num_queries, seed=args.seed)
        except Exception as e:
            print(f"[eval] skip {ds_name}: {e}", flush=True)
            continue

        avg_K = sum(len(qid2cands[q["qid"]]) for q in queries) / len(queries)
        print(f"[eval]   queries: {len(queries)}, unique cands: {len(cands)}, "
              f"avg K: {avg_K:.0f}", flush=True)

        try:
            t_fwd = time.time()
            cid_list, c_orig, c_sae = extractor.encode(
                UnitDataset(cands), args.batch_size, collate,
                desc="cand", num_workers=args.num_workers)
            qid_list, q_orig, q_sae = extractor.encode(
                UnitDataset(queries), args.batch_size, collate,
                desc="query", num_workers=args.num_workers)
            print(f"[eval]   forward (encode) took {time.time()-t_fwd:.1f}s", flush=True)
            cid2idx = {c: i for i, c in enumerate(cid_list)}

            sparse_lookup = None
            if c_sae and q_sae:
                print("[eval]   building sparse lookup ...", flush=True)
                t_lk = time.time()
                sparse_lookup = build_sparse_lookup(c_sae, device=str(device))
                print(f"[eval]   sparse lookup built in {time.time()-t_lk:.1f}s",
                      flush=True)

            print(f"[eval]   computing ranks ...", flush=True)
            rng = random.Random(args.seed)
            per_q = []
            t_rank = time.time()
            for qi, qid in enumerate(qid_list):
                cand_cids = qid2cands[qid]
                if not cand_cids:
                    continue
                r = eval_one_query(
                    cand_cids, qid2gold_pos[qid], args.sample_ratio, rng,
                    q_orig=q_orig[qi], c_orig_all=c_orig,
                    q_sae=q_sae[qi] if q_sae else None,
                    sparse_lookup=sparse_lookup,
                    cid2idx=cid2idx, device=str(device))
                per_q.append(r)
            print(f"[eval]   ranks computed in {time.time()-t_rank:.1f}s", flush=True)

            del sparse_lookup
            torch.cuda.empty_cache()

            metrics = aggregate(per_q)
            for key, m in metrics.items():
                print(f"[eval]   {key}:  top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
                      f"top10={m['top10']:.4f}  mrr={m['mrr']:.4f}  acc={m['acc']:.4f}",
                      flush=True)

            all_results[ds_name] = {
                "n_queries": len(per_q), "avg_K": avg_K,
                "sample_ratio": args.sample_ratio, **metrics,
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[eval] {ds_name} failed: {e}, continue", flush=True)
            torch.cuda.empty_cache()
            continue

        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[eval]   saved -> {out_json}", flush=True)
        try:
            plot_results(all_results, os.path.join(args.output_dir, "zero_shot.png"))
        except Exception as e:
            print(f"[plot] partial failed: {e}", flush=True)

    print(f"\n[eval] all done. {out_json}", flush=True)
    extractor.close()
    if gpu_monitor is not None:
        gpu_monitor.stop()


def plot_results(results, out_path):
    import matplotlib.pyplot as plt
    datasets = list(results.keys())
    if not datasets: return
    metric_keys = [("top1", "Top-1"), ("top5", "Top-5"), ("top10", "Top-10"),
                   ("mrr", "MRR"), ("acc", "Acc (beat ratio)")]
    paths_info = [("orig", "Original Hidden", "#3b82f6", "o"),
                  ("sae", "SAE Sparse", "#f97316", "s"),
                  ("avg", "Avg (both)", "#10b981", "^")]
    fig, axes = plt.subplots(1, len(metric_keys),
                             figsize=(4.5 * len(metric_keys), 5), sharey=True)
    x = np.arange(len(datasets))
    for ax, (mkey, mname) in zip(axes, metric_keys):
        for pk, lbl, color, marker in paths_info:
            ys = [results[d].get(pk, {}).get(mkey) for d in datasets]
            vx = [xi for xi, y in zip(x, ys) if y is not None]
            vy = [y for y in ys if y is not None]
            if vy:
                ax.plot(vx, vy, marker=marker, color=color, label=lbl,
                        linewidth=2, markersize=8)
        ax.set_title(mname, fontsize=13)
        ax.set_xticks(x); ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    ratios = set(r["sample_ratio"] for r in results.values())
    fig.suptitle(f"Zero-shot retrieval baseline (sample_ratio="
                 f"{list(ratios)[0] if len(ratios)==1 else ratios})",
                 y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()