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
from peft import PeftModel

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


def gpu_mem(tag="", device=None):
    """打印当前 GPU 显存占用 (已分配 / 已保留 / 剩余 / 总量)。
    在大操作前后调用, 方便定位过载点和判断余量。"""
    if not torch.cuda.is_available():
        return
    try:
        dev = 0 if device is None else (device.index if hasattr(device, "index")
                                        and device.index is not None else 0)
        alloc = torch.cuda.memory_allocated(dev) / 1e9
        reserved = torch.cuda.memory_reserved(dev) / 1e9
        free, total = torch.cuda.mem_get_info(dev)
        free_gb, total_gb = free / 1e9, total / 1e9
        print(f"    [mem{(' ' + tag) if tag else ''}] "
              f"allocated={alloc:.1f}GB reserved={reserved:.1f}GB "
              f"free={free_gb:.1f}GB / total={total_gb:.1f}GB", flush=True)
    except Exception as e:
        print(f"    [mem] error: {e}", flush=True)


def estimate_mem(desc, n_elements, bytes_per=4):
    """预测某个张量/操作需要多少显存, 并跟当前剩余对比给出提示。"""
    need_gb = n_elements * bytes_per / 1e9
    msg = f"    [mem-predict] {desc}: 需要 ~{need_gb:.2f}GB"
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            free_gb = free / 1e9
            ratio = need_gb / max(free_gb, 1e-6)
            flag = " ⚠️超过剩余70%!" if need_gb > free_gb * 0.7 else ""
            msg += f" (剩余 {free_gb:.1f}GB, 占 {ratio*100:.0f}%{flag})"
        except Exception:
            pass
    print(msg, flush=True)
    return need_gb


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
        base = model.base_model.model if isinstance(model, PeftModel) else model
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
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
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
                    am_b = batch["attention_mask"]            # [B, L] 0/1
                    denom = am_b.sum(1).clamp_min(1.0)        # [B]
                    B, L, K = ta.shape
                    for b in range(B):
                        valid_tok = am_b[b].bool()            # [L]
                        a = ta[b][valid_tok].reshape(-1).float()   # [L_valid*K] 激活值
                        idx = ti[b][valid_tok].reshape(-1).long()  # [L_valid*K] feature 编号
                        # 关键: 先把 token 级激活 pool 成样本级稀疏向量。
                        # 同一 feature 在多个 token 的激活累加 -> nnz 从 L*K
                        # 降到 unique feature 数 (带图样本能省几十~上百倍)。
                        uniq, inv = torch.unique(idx, return_inverse=True)
                        pooled = torch.zeros(uniq.shape[0], device=a.device, dtype=a.dtype)
                        pooled.scatter_add_(0, inv, a)
                        pooled = pooled / denom[b]            # mean-pool 归一化
                        nz = pooled != 0                      # 去掉 pool 后的 0
                        # 搬 CPU 存 (省 GPU 显存), build CSR 时再上 GPU
                        saes.append((pooled[nz].cpu(), uniq[nz].cpu()))

            # 清掉本 batch 的 GPU 中间量 (SAE 预激活 hook cache 等), 防 reserved 累积
            self._cache.clear()
            if self.sae_hook is not None and self.sae_hook.cache:
                self.sae_hook.cache.clear()
            del batch, h, am, emb
            if 'ta' in dir() and ta is not None:
                del ta, ti
            # 每 50 个 batch 强制把 reserved 显存还给系统 (防 caching allocator 占满)
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            ids.extend(bids)
            if (i + 1) % max(n // 10, 1) == 0 or i == n - 1:
                a = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                r = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                pk = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
                print(f"  [{desc}] {i+1}/{n} ({100*(i+1)/n:.0f}%) "
                      f"elapsed={time.time()-t0:.0f}s "
                      f"alloc={a:.1f}G reserved={r:.1f}G peak={pk:.1f}G "
                      f"free={free:.1f}G saes={len(saes)}", flush=True)
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

    mem_gb = N * V * 2 / 1e9   # fp16
    if verbose:
        print(f"    allocating dense [N={N}, V={V}] fp16 = {mem_gb:.1f} GB", flush=True)

    # 显存安全检查: 如果 dense 需要的显存超过剩余可用的 70%, 放弃 (返回 None)
    if device != "cpu" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1e9
        if mem_gb > free_gb * 0.7:
            print(f"    [WARN] dense 需要 {mem_gb:.1f}GB 但只剩 {free_gb:.1f}GB, "
                  f"跳过 SAE 路径 (该数据集只有 orig 结果)", flush=True)
            return None

    dense = torch.zeros(N, V, dtype=torch.float16, device=device)

    t = time.time()
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        max_len = max(sparse_list[i][0].shape[0] for i in range(s, e))
        chunk = e - s
        acts_chunk = torch.zeros(chunk, max_len, dtype=torch.float16, device=device)
        inds_chunk = torch.full((chunk, max_len), int(unique[0].item()),
                                dtype=torch.long, device=device)
        for k, i in enumerate(range(s, e)):
            L_i = sparse_list[i][0].shape[0]
            acts_chunk[k, :L_i] = sparse_list[i][0].half()
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


def sparse_sim_local(q_sparse, cand_sparse_list, device="cuda"):
    """
    只稠密化"这个 query 用到的 cand 子集"(n_sample 个), 不建全池 dense。
    显存只占 [n_sample, V_local], V_local 是这一小撮的 indices 并集, 远小于全池 V。

    q_sparse:         (acts, inds)            单个 query 的稀疏激活
    cand_sparse_list: list of (acts, inds)    该 query 抽样的 n_sample 个 cand
    返回: sims [n_sample]
    """
    q_acts, q_inds = q_sparse
    n = len(cand_sparse_list)

    # query + 这 n 个 cand 的局部 indices 并集 (小)
    all_inds = torch.cat([q_inds] + [c[1] for c in cand_sparse_list])
    unique = torch.unique(all_inds)
    V_local = unique.shape[0]
    max_idx = int(unique.max().item())
    idx_to_pos = torch.full((max_idx + 1,), -1, dtype=torch.long, device=device)
    idx_to_pos[unique] = torch.arange(V_local, device=device)

    # query 稠密化到 V_local
    q_d = torch.zeros(V_local, dtype=torch.float32, device=device)
    q_d.scatter_add_(0, idx_to_pos[q_inds], q_acts.float())
    q_d = q_d / (q_d.norm() + 1e-8)

    # cand 稠密化到 [n, V_local] (V_local 小, 逐个 scatter 也快)
    c_dense = torch.zeros(n, V_local, dtype=torch.float32, device=device)
    for j, (acts, inds) in enumerate(cand_sparse_list):
        c_dense[j].scatter_add_(0, idx_to_pos[inds], acts.float())
    c_dense = c_dense / (c_dense.norm(dim=-1, keepdim=True) + 1e-8)

    sims = c_dense @ q_d                  # [n]
    return sims


def build_sparse_csr(sparse_list, device="cuda", verbose=True):
    """
    用 torch.sparse 把整个 candidate 池建成一个 CSR 稀疏矩阵 [N, V_total], 行已 L2 归一化。
    全池只建一次, 所有 query 复用。底层只存非零, 显存 = O(非零总数), 不是 O(N*V)。

    sparse_list: list of (acts, inds)   每个 candidate 的稀疏激活 (在 GPU)
    返回: (csr_mat[N, V_total] 行归一化 CSR, V_total)
          显存不足或空时返回 None
    """
    if len(sparse_list) == 0:
        return None

    N = len(sparse_list)
    t = time.time()

    # 1) 拼所有非零的 (row, col, val)
    rows, cols, vals = [], [], []
    for i, (acts, inds) in enumerate(sparse_list):
        n_nz = inds.shape[0]
        rows.append(torch.full((n_nz,), i, dtype=torch.long, device=device))
        cols.append(inds.long().to(device))
        vals.append(acts.float().to(device))
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    vals = torch.cat(vals)
    V_total = int(cols.max().item()) + 1
    nnz = vals.shape[0]
    if verbose:
        mem_mb = nnz * (8 + 8 + 4) / 1e6   # row(long)+col(long)+val(float), COO 阶段
        print(f"    sparse CSR: N={N}, V_total={V_total}, nnz={nnz} "
              f"(~{mem_mb:.0f}MB COO)", flush=True)

    # 显存安全检查 (COO 阶段是峰值)
    if device != "cpu" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if nnz * 20 > free * 0.7:    # 粗估 COO+CSR+coalesce 中间峰值 ~20B/nnz
            print(f"    [WARN] 稀疏矩阵 nnz={nnz} 显存可能不足, 跳过 SAE 路径",
                  flush=True)
            return None

    # 2) COO -> coalesce (合并同一 (row,col) 的重复 index, 累加激活)
    coo = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (N, V_total)
    ).coalesce()
    del rows, cols, vals

    # 3) 行 L2 归一化: 算每行平方和 -> sqrt -> 除
    ci = coo.indices()           # [2, nnz_coalesced]
    cv = coo.values()            # [nnz_coalesced]
    row_idx = ci[0]
    row_sq = torch.zeros(N, dtype=torch.float32, device=device)
    row_sq.scatter_add_(0, row_idx, cv * cv)
    row_norm = (row_sq.sqrt() + 1e-8)
    cv_normed = cv / row_norm[row_idx]

    # 4) 用归一化后的值重建 -> 转 CSR (matmul 快)
    csr = torch.sparse_coo_tensor(ci, cv_normed, (N, V_total)).coalesce().to_sparse_csr()
    if verbose:
        print(f"    sparse CSR built in {time.time()-t:.1f}s", flush=True)
    return csr, V_total


def sparse_sim_csr(q_sparse, csr_mat, V_total, cand_indices, device="cuda"):
    """
    用预建 CSR 稀疏矩阵算 query 与 cand 子集余弦 (全 GPU, 复用 csr_mat)。

    注意: CSR 不支持 index_select 按行切, 所以改成
          "整个 CSR @ query 算全池 sims, 再用稠密索引取子集"。
          CSR @ dense 是 torch.sparse 最成熟的操作, 稳。

    q_sparse:     (acts, inds)   单个 query
    csr_mat:      [N, V_total] 行归一化 CSR
    cand_indices: [n_sample]    要取的 cand 行号
    返回: sims [n_sample]
    """
    q_acts, q_inds = q_sparse
    q_inds = q_inds.to(device)
    q_acts = q_acts.to(device)
    q_d = torch.zeros(V_total, dtype=torch.float32, device=device)
    valid = q_inds < V_total
    q_d.scatter_add_(0, q_inds[valid].long(), q_acts[valid].float())
    q_d = q_d / (q_d.norm() + 1e-8)

    # 整个 CSR @ q -> 全池 sims [N] (稠密), 再普通索引取子集
    all_sims = (csr_mat @ q_d.unsqueeze(1)).squeeze(1)   # [N] 稠密
    sims = all_sims[cand_indices]                         # 稠密索引, 一定支持
    return sims


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
    q_d = q_d.to(c_dense_norm.dtype)           # 跟 c_dense 同 dtype (可能 fp16)

    c_subset = c_dense_norm[cand_indices]      # [N_subset, V] GPU
    sims = (c_subset @ q_d).float()            # 算完转回 fp32 保证 rank 比较稳
    return sims


# ============================================================
# 抽样 + 评估 (一个 query) - 全 GPU
# ============================================================
def eval_one_query(cand_cids, gold_pos, sample_ratio, rng,
                   q_orig, c_orig_all,
                   q_sae=None, csr_mat=None, V_total=None,
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

    # sae 路径 (GPU, 用预建 CSR 稀疏矩阵复用)
    if q_sae is not None and csr_mat is not None:
        sims_sae = sparse_sim_csr(q_sae, csr_mat, V_total, cand_idx_tensor, device)
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
    parser.add_argument("--num_workers", type=int, default=48)
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
    gpu_mem("before model load")
    if torch.cuda.is_available():
        orig_dm = cfg.model.device_map
        cfg.model.device_map = {"": 0}
        print(f"[eval] override device_map: {orig_dm} -> {{'': 0}}", flush=True)


    if not args.ckpt:
        cfg.model.lora.enable = False   # 无 ckpt 时不注入 LoRA, 评估纯基座

    cfg.model.lora.enable = False       # 统一关掉，不管有没有 ckpt    
    model, processor = load_model(cfg)
    gpu_mem("after model load")
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
        gpu_mem("after SAE load")

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
            gpu_mem(f"{ds_name} before encode")
            t_fwd = time.time()
            cid_list, c_orig, c_sae = extractor.encode(
                UnitDataset(cands), args.batch_size, collate,
                desc="cand", num_workers=args.num_workers)
            torch.cuda.empty_cache()    # cand encode 后还 reserved, 给 query encode 腾地方
            qid_list, q_orig, q_sae = extractor.encode(
                UnitDataset(queries), args.batch_size, collate,
                desc="query", num_workers=args.num_workers)
            print(f"[eval]   forward (encode) took {time.time()-t_fwd:.1f}s", flush=True)
            torch.cuda.empty_cache()    # encode 占满的 reserved 还给系统, 给 CSR 腾地方
            gpu_mem(f"{ds_name} after encode (已 empty_cache)")
            cid2idx = {c: i for i, c in enumerate(cid_list)}

            # SAE 路径: 用 torch.sparse 全池建一个 CSR 稀疏矩阵 (只存非零, 省显存),
            # 所有 query 复用同一个矩阵 (保留 candidate 复用红利)
            csr_mat, V_total = None, None
            use_sae = bool(c_sae) and bool(q_sae)
            if use_sae:
                # 预测稀疏矩阵显存 (nnz × ~20B 中间峰值)
                nnz_est = sum(s[1].shape[0] for s in c_sae)
                estimate_mem(f"sparse CSR (nnz={nnz_est})", nnz_est, bytes_per=20)
                print("[eval]   building sparse CSR (torch.sparse) ...", flush=True)
                res = build_sparse_csr(c_sae, device=str(device))
                if res is None:
                    print("[eval]   sparse CSR skipped (显存不足), orig only", flush=True)
                    use_sae = False
                else:
                    csr_mat, V_total = res
                    gpu_mem(f"{ds_name} after build CSR")

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
                    q_sae=q_sae[qi] if (use_sae and qi < len(q_sae)) else None,
                    csr_mat=csr_mat if use_sae else None,
                    V_total=V_total,
                    cid2idx=cid2idx, device=str(device))
                per_q.append(r)
            print(f"[eval]   ranks computed in {time.time()-t_rank:.1f}s", flush=True)
            gpu_mem(f"{ds_name} after ranks")

            del csr_mat
            c_orig = c_sae = q_orig = q_sae = None   # 释放该数据集的 emb/激活
            torch.cuda.empty_cache()
            gpu_mem(f"{ds_name} after cleanup")

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
    datasets = sorted(results.keys())
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