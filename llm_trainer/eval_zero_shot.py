"""
eval_zero_shot.py
-----------------
零样本 baseline 评估。

每个 query 的 candidate 池来自该 query 自己的 tgt 列表 (tgt[0]=真值, tgt[1:]=负样本)。
不同 query 的 tgt 经常重复, 所以 unique candidate 只 forward 一次, 用 cid 索引复用。

抽样模拟全局: 抽 20% candidate (保留真值) 算 rank, rank × (1/sample_ratio) 还原全局 rank。
报告: top-1 / top-5 / MRR (按近似还原后的 rank 算)。

三路对比: orig (原始 hidden) / sae (SAE 稀疏) / avg (两路相似度平均)

用法:
  python eval_zero_shot.py --config configs/exp_lora_sae.yaml \
                           --datasets A-OKVQA ChartQA \
                           --num_queries 200 --sample_ratio 0.2
"""
import os
import json
import time
import random
import argparse

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
    """
    返回:
      queries:        list[{qid, text, image_path}]
      cands:          list[{cid, text, image_path}]    所有 query 的 tgt 合并去重
      qid2cand_cids:  {qid: [cid, ...]}                 每个 query 自己的候选 cid 列表
      qid2gold_pos:   {qid: int}                        gold 在 qid2cand_cids[qid] 里的下标(总是 0)
    """
    ds_dir = os.path.join(data_root, dataset_name)
    files = _find_parquet_files(ds_dir)
    if not files:
        raise FileNotFoundError(f"no parquet in {ds_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    cand_dict = {}   # (text, image_path) -> cid
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
            "qid": qid,
            "text": q_text,
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
        qid2gold_pos[qid] = 0   # tgt[0] = gold

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
# Embedding 提取 (统一 forward 一遍 unique units)
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
                # 合理性检查: 太小或奇怪比例的图跳过
                # LLaVA-Next 处理时要求至少几十像素
                w, h = img.size
                if w < 16 or h < 16 or w > 8192 or h > 8192:
                    # 太小或太大都跳过, 算无图样本
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
                # 兜底: 万一某张图能打开但 processor 处理崩溃, 把整批退化为纯文本
                print(f"[collate] processor failed on images, fallback to text-only: {e}",
                      flush=True)
                r_with = processor.tokenizer(
                    txt_img,
                    padding="max_length", truncation=True, max_length=max_length,
                    return_tensors="pt")

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
    def encode(self, ds, batch_size, collate, desc=""):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
        ids, origs, saes = [], [], []
        n = len(loader)
        t0 = time.time()
        self.model.eval()
        for i, batch in enumerate(loader):
            bids = batch.pop("ids")
            batch = {k: v.to(self.device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            self._cache.clear()
            _ = self.model(**batch, use_cache=False)
            h = self._cache["h"]
            am = batch["attention_mask"].to(h.dtype)
            emb = ((h * am.unsqueeze(-1)).sum(1) /
                   am.sum(1, keepdim=True).clamp_min(1e-6)).float().cpu()
            origs.append(emb)

            if self.sae_hook is not None and self.sae_hook.cache:
                ta = self.sae_hook.cache.get("z")
                ti = self.sae_hook.cache.get("z_indices")
                if ta is not None and ti is not None:
                    mask = am.unsqueeze(-1).to(ta.dtype)
                    ta_m = ta * mask / am.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
                    B, L, K = ta.shape
                    pa = ta_m.reshape(B, L * K).cpu()
                    pi = ti.reshape(B, L * K).cpu()
                    for b in range(B):
                        saes.append((pa[b], pi[b]))

            ids.extend(bids)
            if (i + 1) % max(n // 10, 1) == 0 or i == n - 1:
                print(f"  [{desc}] {i+1}/{n} ({100*(i+1)/n:.0f}%) "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        return ids, torch.cat(origs, dim=0), saes

    def close(self):
        if self._handle is not None:
            self._handle.remove()


# ============================================================
# 相似度: 给定 q_emb 和 c_emb (子集), 算余弦; 稀疏的同理
# ============================================================
def cos_sim(q, c):
    """q:[D], c:[N,D]  -> [N]"""
    q = F.normalize(q.float().unsqueeze(0), dim=-1).squeeze(0)
    c = F.normalize(c.float(), dim=-1)
    return c @ q


def sparse_sim(q_sparse, c_sparse_list):
    """
    q_sparse: (acts[L*K], inds[L*K])
    c_sparse_list: list of (acts, inds), 长度 N
    返回 [N] 余弦相似度
    """
    # 收集本 q 涉及的所有 indices, 在它们的并集上稠密化
    all_inds = [q_sparse[1]] + [c[1] for c in c_sparse_list]
    cat = torch.cat(all_inds)
    unique = torch.unique(cat)
    V = unique.shape[0]
    max_idx = int(unique.max().item())
    idx_to_pos = torch.full((max_idx + 1,), -1, dtype=torch.long)
    idx_to_pos[unique] = torch.arange(V)

    def _to_dense(acts, inds):
        d = torch.zeros(V, dtype=torch.float32)
        d.scatter_add_(0, idx_to_pos[inds], acts.float())
        return d

    q_d = F.normalize(_to_dense(*q_sparse).unsqueeze(0), dim=-1).squeeze(0)
    N = len(c_sparse_list)
    sims = torch.zeros(N, dtype=torch.float32)
    for j, c in enumerate(c_sparse_list):
        c_d = _to_dense(*c)
        c_d = c_d / (c_d.norm() + 1e-8)
        sims[j] = c_d @ q_d
    return sims


# ============================================================
# 抽样 + 评估 (一个 query)
# ============================================================
def eval_one_query(cand_cids, gold_pos, sample_ratio, rng,
                   q_orig, c_orig_all,         # 原始路径
                   q_sae=None, c_sae_all=None, # SAE 路径
                   cid2idx=None):
    """
    抽 sample_ratio 比例的 candidate (保证 gold 在内), 算三路 rank.

    返回 dict: {"orig": rank_global_est, "sae": ..., "avg": ...}
    rank_global_est = rank_in_sample × (1 / sample_ratio)
    """
    K = len(cand_cids)
    # 抽样阈值: K 太小时直接用全集 (避免抽样退化成 1 个 candidate)
    # 经验值: 如果 K <= 20, 全用; 否则按 sample_ratio 抽
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

    # 拿出对应 embedding
    cand_idx_in_full = [cid2idx[c] for c in sample_cids]
    c_orig_subset = c_orig_all[cand_idx_in_full]

    sims_orig = cos_sim(q_orig, c_orig_subset)
    gold_score_orig = sims_orig[new_gold_pos]
    rank_orig = int((sims_orig > gold_score_orig).sum().item()) + 1

    out = {"orig_rank_sample": rank_orig, "n_sample": n_sample, "K": K}

    sims_sae = None
    if q_sae is not None and c_sae_all:
        c_sae_subset = [c_sae_all[i] for i in cand_idx_in_full]
        sims_sae = sparse_sim(q_sae, c_sae_subset)
        gold_score_sae = sims_sae[new_gold_pos]
        rank_sae = int((sims_sae > gold_score_sae).sum().item()) + 1
        out["sae_rank_sample"] = rank_sae

        sims_avg = (sims_orig + sims_sae) / 2
        gold_score_avg = sims_avg[new_gold_pos]
        rank_avg = int((sims_avg > gold_score_avg).sum().item()) + 1
        out["avg_rank_sample"] = rank_avg

    # 还原全局 rank: rank_global ≈ rank_sample / sample_ratio
    actual_ratio = n_sample / K
    scale = 1.0 / actual_ratio
    out["scale"] = scale
    for key in ("orig", "sae", "avg"):
        sk = f"{key}_rank_sample"
        if sk in out:
            out[f"{key}_rank_global"] = out[sk] * scale
    return out


def aggregate(per_query_results, ks=(1, 5, 10)):
    """从 per-query rank 列表算 top-k acc + MRR + acc(击败比例).

    指标:
      - top-k:    抽样池内 rank <= k 的比例
      - mrr:      抽样池内 1/rank 平均
      - acc:      gold 击败的 candidate 比例 = 1 - rank_sample / n_sample
                  (= 1 - rank_global / K, 数学等价)
                  rank=1 时 acc 最接近 1; rank=n_sample 时 acc=0
    """
    metrics = {}
    for key in ("orig", "sae", "avg"):
        sk_sample = f"{key}_rank_sample"
        sk_global = f"{key}_rank_global"
        if sk_sample not in per_query_results[0]:
            continue
        ranks_sample = np.array([r[sk_sample] for r in per_query_results])
        ranks_global = np.array([r[sk_global] for r in per_query_results])
        n_samples = np.array([r["n_sample"] for r in per_query_results])
        Ks = np.array([r["K"] for r in per_query_results])

        m = {}
        # 抽样池内的指标 (主指标)
        for k in ks:
            m[f"top{k}"] = float((ranks_sample <= k).mean())
        m["mrr"] = float((1.0 / ranks_sample).mean())
        # acc: gold 击败 candidate 的比例 (越大越好, 1=完美, 0=最差)
        # 用 n_sample-1 做分母, 排除 gold 自己; rank=1 时 acc=1
        m["acc"] = float((1.0 - (ranks_sample - 1) / np.maximum(n_samples - 1, 1)).mean())
        # 全局还原 (供参考)
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
    parser.add_argument("--sample_ratio", type=float, default=0.2,
                        help="每个 query 的 candidate 抽样比例 (0.2 = 抽 20%)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs/zero_shot")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="跳过 output_dir 下已有 JSON 中的数据集 (用于中断后续跑)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[eval] sample_ratio={args.sample_ratio}", flush=True)

    # ---- resume: 从已有 JSON 加载已完成的数据集 ----
    out_json = os.path.join(args.output_dir, "zero_shot_results.json")
    all_results = {}
    if args.resume and os.path.isfile(out_json):
        try:
            with open(out_json) as f:
                all_results = json.load(f)
            print(f"[eval] resume: loaded {len(all_results)} completed datasets "
                  f"from {out_json}", flush=True)
            print(f"[eval] already done: {list(all_results.keys())}", flush=True)
        except Exception as e:
            print(f"[eval] resume failed to load {out_json}: {e}", flush=True)

    # 过滤掉已完成的数据集
    datasets_to_run = [d for d in args.datasets if d not in all_results]
    if not datasets_to_run:
        print(f"[eval] all datasets already done, nothing to do", flush=True)
        try:
            plot_results(all_results, os.path.join(args.output_dir, "zero_shot.png"))
        except Exception as e:
            print(f"[plot] {e}", flush=True)
        return
    print(f"[eval] will run: {datasets_to_run}", flush=True)

    print("[eval] loading model + SAE ...", flush=True)
    model, processor = load_model(cfg)
    if args.ckpt:
        from model import load_adapter
        model = load_adapter(model, args.ckpt)
    device = next(model.parameters()).device
    sae = load_sae(cfg, device=device)
    sae_hook = attach_sae_hook(model, sae, cfg) if sae is not None else None

    pool_layer_idx = getattr(cfg.train, "pool_layer_idx", cfg.sae.layer_idx)
    extractor = Extractor(model, sae_hook, pool_layer_idx, device)
    collate = build_collate(processor, max_length=cfg.data.max_length)

    for ds_name in datasets_to_run:
        print(f"\n[eval] ====== {ds_name} ======", flush=True)
        try:
            queries, cands, qid2cands, qid2gold_pos = collect_units(
                cfg.data.data_root, ds_name, cfg.data.images_root,
                num_query_sample=args.num_queries, seed=args.seed,
            )
        except Exception as e:
            print(f"[eval] skip {ds_name}: {e}", flush=True)
            continue

        avg_K = sum(len(qid2cands[q["qid"]]) for q in queries) / len(queries)
        print(f"[eval]   queries: {len(queries)}, unique candidates: {len(cands)}, "
              f"avg K per query: {avg_K:.0f}", flush=True)

        try:
            # 一次性 forward 所有 unique candidate
            cid_list, c_orig, c_sae = extractor.encode(
                UnitDataset(cands), args.batch_size, collate, desc="cand")
            qid_list, q_orig, q_sae = extractor.encode(
                UnitDataset(queries), args.batch_size, collate, desc="query")
            cid2idx = {c: i for i, c in enumerate(cid_list)}

            # 逐 query 抽样 + 算 rank
            print(f"[eval]   computing ranks (sample_ratio={args.sample_ratio})...",
                  flush=True)
            rng = random.Random(args.seed)
            per_q = []
            for qi, qid in enumerate(qid_list):
                cand_cids = qid2cands[qid]
                if not cand_cids:
                    continue
                r = eval_one_query(
                    cand_cids, qid2gold_pos[qid], args.sample_ratio, rng,
                    q_orig=q_orig[qi], c_orig_all=c_orig,
                    q_sae=q_sae[qi] if q_sae else None,
                    c_sae_all=c_sae,
                    cid2idx=cid2idx,
                )
                per_q.append(r)

            metrics = aggregate(per_q)
            for key, m in metrics.items():
                print(f"[eval]   {key}:  top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
                      f"top10={m['top10']:.4f}  mrr={m['mrr']:.4f}  "
                      f"acc={m['acc']:.4f}", flush=True)

            all_results[ds_name] = {
                "n_queries": len(per_q),
                "avg_K": avg_K,
                "sample_ratio": args.sample_ratio,
                **metrics,
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[eval] {ds_name} failed: {e}, continuing to next dataset",
                  flush=True)
            continue

        # 每个数据集跑完就立即存盘, 避免后面崩了前面白干
        with open(out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[eval]   saved partial results -> {out_json}", flush=True)

        # 同时增量更新图 (方便中途观察)
        try:
            plot_results(all_results, os.path.join(args.output_dir, "zero_shot.png"))
        except Exception as e:
            print(f"[plot] partial plot failed: {e}", flush=True)

    print(f"\n[eval] all done. final results: {out_json}", flush=True)
    extractor.close()


def plot_results(results, out_path):
    import matplotlib.pyplot as plt
    datasets = list(results.keys())
    if not datasets: return
    metric_keys = [("top1", "Top-1"), ("top5", "Top-5"),
                   ("top10", "Top-10"), ("mrr", "MRR"),
                   ("acc", "Acc (beat ratio)")]
    paths_info = [("orig", "Original Hidden", "#3b82f6", "o"),
                  ("sae",  "SAE Sparse",      "#f97316", "s"),
                  ("avg",  "Avg (both)",      "#10b981", "^")]

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
    fig.suptitle(f"Zero-shot retrieval baseline "
                 f"(sample_ratio={list(ratios)[0] if len(ratios)==1 else ratios})",
                 y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()