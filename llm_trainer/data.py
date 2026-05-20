"""
data.py
----------
负责: 加载 MMEB 风格的检索数据(parquet),展开为 1 query + K candidates 的独立 forward 单元。

数据格式 (与 activation_collector.py 一致):
  qry_text:      str
  qry_img_path:  str
  tgt_text:      List[str]  长度 K
  tgt_img_path:  List[str]  长度 K
  其中 tgt_text[gold_idx] / tgt_img_path[gold_idx] 是真值 (默认 gold_idx=0)

设计要点:
  1. 一条 MMEB 样本 -> 一个 dict, 含 query 与 K 个 candidate
  2. collate_fn 把 batch 内所有 (1+K) 条样本展平成大 batch, 一次性喂给 LLaVA
  3. 用 sample_ids / role_ids 记录每条展平样本属于原 batch 哪条、是 query 还是哪个 candidate
  4. Trainer 拿到 LLaVA hidden state 后,按 sample_ids 切回 (query_emb, cand_embs)
     再算相似度 + InfoNCE
  5. 图文都有 -> 图 emb 和 文 emb 平均; 只有一种模态 -> 用那一种 (在 trainer 池化阶段处理)
"""
import os
import ast
import glob
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# 工具函数 (与 activation_collector.py 风格对齐)
# ============================================================
def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _normalize_to_list(x):
    """把 parquet 里可能是 list/ndarray/str("[...]")/None 的 tgt 字段统一成 list"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist") and not isinstance(x, str):
        try:
            y = x.tolist()
            return y if isinstance(y, list) else [y]
        except Exception:
            pass
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                y = ast.literal_eval(s)
                return y if isinstance(y, list) else [y]
            except Exception:
                return [x]
        return [x]
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    return [x]


def _resolve_image_path(rel_path, data_root, images_root=None):
    if not _is_nonempty_str(rel_path):
        return None
    rel_path = rel_path.strip()
    if os.path.basename(rel_path).startswith("._"):
        return None

    candidates = []
    if os.path.isabs(rel_path):
        candidates.append(rel_path)
    if images_root is not None:
        candidates.append(os.path.join(images_root, rel_path))
    candidates.append(os.path.join(data_root, rel_path))
    candidates.append(os.path.join(data_root, "images", rel_path))
    candidates.append(rel_path)

    for p in candidates:
        if p and os.path.exists(p) and not os.path.basename(p).startswith("._"):
            return p
    return None


def _find_parquet_files(dataset_dir):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True))
    return files


# ============================================================
# Dataset
# ============================================================
class MMEBRetrievalDataset(Dataset):
    """
    输入: 单个 MMEB 数据集目录 (含若干 .parquet),或多个目录的根 (按 dataset_names 选择)。
    输出 (__getitem__):
        {
            "query": {"text": str|None, "image": PIL.Image|None},
            "candidates": [
                {"text": str|None, "image": PIL.Image|None},
                ...   # 长度 K
            ],
            "gold_idx": int,             # 真值在 candidates 中的下标, 默认 0
            "meta": {"dataset": str, "sample_idx": int},
        }

    注意: 这里只到 PIL.Image 这一步; 真正的 tensor 化在 collate_fn 里用 processor 完成。
    """

    def __init__(
        self,
        data_root: str,
        dataset_names: Optional[List[str]] = None,
        images_root: Optional[str] = None,
        max_candidates: Optional[int] = None,
        gold_idx_col: Optional[str] = None,   # parquet 里若有指示真值列下标的列名,否则默认 0
    ):
        self.data_root = os.path.abspath(data_root)
        self.images_root = os.path.abspath(images_root) if images_root else None
        self.max_candidates = max_candidates
        self.gold_idx_col = gold_idx_col

        # 收集所有 parquet
        if dataset_names is None:
            dataset_names = self._list_dataset_dirs(self.data_root)

        print(f"[MMEB] loading from data_root={self.data_root}", flush=True)
        print(f"[MMEB] datasets to load: {dataset_names}", flush=True)

        all_rows = []
        for name in dataset_names:
            ds_dir = os.path.join(self.data_root, name)
            if not os.path.isdir(ds_dir):
                print(f"[WARN] skip non-existing dataset dir: {ds_dir}", flush=True)
                continue
            files = _find_parquet_files(ds_dir)
            if not files:
                print(f"[WARN] no parquet in {ds_dir}", flush=True)
                continue
            n_rows_before = sum(len(df) for df in all_rows)
            for f in files:
                df = pd.read_parquet(f)
                df["__dataset__"] = name
                all_rows.append(df)
            n_added = sum(len(df) for df in all_rows) - n_rows_before
            print(f"[MMEB]   {name}: {n_added} samples ({len(files)} files)", flush=True)

        if not all_rows:
            raise FileNotFoundError(f"No parquet found under {self.data_root} for {dataset_names}")

        self.df = pd.concat(all_rows, ignore_index=True)
        required = ["qry_text", "qry_img_path", "tgt_text", "tgt_img_path"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"[MMEB] total: {len(self.df)} samples across {len(dataset_names)} datasets",
              flush=True)

    @staticmethod
    def _list_dataset_dirs(data_root):
        names = []
        for x in sorted(os.listdir(data_root)):
            full = os.path.join(data_root, x)
            if not os.path.isdir(full):
                continue
            if x in {"images", "activations", "__pycache__"} or x.startswith("."):
                continue
            if _find_parquet_files(full):
                names.append(x)
        return names

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path) -> Optional[Image.Image]:
        path = _resolve_image_path(rel_path, self.data_root, self.images_root)
        if path is None:
            return None
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] failed to open image {path}: {e}")
            return None

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # query
        qry_text = row["qry_text"] if _is_nonempty_str(row["qry_text"]) else None
        qry_img = self._load_image(row["qry_img_path"])

        # candidates
        tgt_texts = _normalize_to_list(row["tgt_text"])
        tgt_imgs = _normalize_to_list(row["tgt_img_path"])
        K = max(len(tgt_texts), len(tgt_imgs))
        if self.max_candidates is not None:
            K = min(K, self.max_candidates)

        candidates = []
        for i in range(K):
            t = tgt_texts[i] if i < len(tgt_texts) and _is_nonempty_str(tgt_texts[i]) else None
            img_rel = tgt_imgs[i] if i < len(tgt_imgs) and _is_nonempty_str(tgt_imgs[i]) else ""
            img = self._load_image(img_rel) if img_rel else None
            candidates.append({"text": t, "image": img})

        gold_idx = 0
        if self.gold_idx_col and self.gold_idx_col in row and not pd.isna(row[self.gold_idx_col]):
            gold_idx = int(row[self.gold_idx_col])
            gold_idx = max(0, min(gold_idx, K - 1))

        return {
            "query": {"text": qry_text, "image": qry_img},
            "candidates": candidates,
            "gold_idx": gold_idx,
            "meta": {
                "dataset": row.get("__dataset__", "unknown"),
                "sample_idx": int(idx),
            },
        }


# ============================================================
# Collate: 把 batch 内 (1+K) 条全部展平
# ============================================================
def build_collate_fn(processor, max_length: int = 512):
    """
    返回 collate_fn。约定:
      batch_size = B 条 MMEB 样本; 每条 1 query + K_i 个候选 (K_i 可以不一致)
      展平后总条数 M = sum(1 + K_i)

    返回 dict:
      # 给 LLaVA 的输入 (展平后的大 batch)
      input_ids:      LongTensor  [M, L]
      attention_mask: LongTensor  [M, L]
      pixel_values:   FloatTensor [M_img, C, H, W]   (仅含有图的样本)
      image_sizes:    [M_img, 2]
      # 边界与角色信息 (trainer 切分用)
      sample_ids:     LongTensor [M]   原 batch 中的样本下标 (0..B-1)
      role_ids:       LongTensor [M]   0 = query, 1..K = 第 i 个 candidate
      img_present:    BoolTensor [M]   该条是否有图
      text_present:   BoolTensor [M]   该条是否有文本
      img_pos_in_flat: LongTensor [M_img]   pixel_values 中第 j 个对应展平 batch 的 idx
      # 监督
      gold_idx:       LongTensor [B]
      num_candidates: LongTensor [B]   每条样本的 K_i
    """
    def collate_fn(batch):
        flat_texts: List[str] = []
        flat_images: List[Optional[Image.Image]] = []
        sample_ids: List[int] = []
        role_ids: List[int] = []
        text_present: List[bool] = []
        img_present: List[bool] = []

        gold_idx_list: List[int] = []
        num_candidates_list: List[int] = []

        for b_idx, item in enumerate(batch):
            query = item["query"]
            cands = item["candidates"]
            gold_idx_list.append(item["gold_idx"])
            num_candidates_list.append(len(cands))

            # query (role=0)
            _append_unit(query, b_idx, 0, flat_texts, flat_images,
                         sample_ids, role_ids, text_present, img_present)

            # candidates (role=1..K)
            for c_idx, cand in enumerate(cands):
                _append_unit(cand, b_idx, c_idx + 1, flat_texts, flat_images,
                             sample_ids, role_ids, text_present, img_present)

        M = len(flat_texts)
        # 取出有图的条目用 processor 处理图像; 没图的只走 tokenizer
        img_pos_in_flat = [i for i, im in enumerate(flat_images) if im is not None]
        valid_images = [flat_images[i] for i in img_pos_in_flat]

        # 文本部分: 统一用 tokenizer 编码 (即使有图的条目也走 tokenizer,
        # 因为 LlavaNextProcessor 期望 <image> token 占位与 pixel_values 数量一致;
        # 我们这里图文是独立池化,不需要让 LLaVA 在 text 里看 image token)
        # 为了让 LLaVA 真正"看到"图,有图的样本仍需走 processor; 拆成两个分支:
        results_no_img = None
        results_with_img = None

        no_img_idx = [i for i in range(M) if flat_images[i] is None]
        with_img_idx = img_pos_in_flat  # alias

        if no_img_idx:
            texts_no_img = [flat_texts[i] if flat_texts[i] else "" for i in no_img_idx]
            results_no_img = processor.tokenizer(
                texts_no_img,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        if with_img_idx:
            texts_with_img = []
            for i in with_img_idx:
                t = flat_texts[i] if flat_texts[i] else ""
                # LLaVA-Next 需要 <image> 占位符
                if "<image>" not in t:
                    t = "<image>\n" + t
                texts_with_img.append(t)
            # 注意: 不能截断有图样本! 否则会砍掉 image tokens,
            # 让 input_ids 里的 <image> 数量与实际 num_images 对不上,
            # 导致 _merge_input_ids_with_image_features 报错。
            # LLaVA-Next 单图约需 ~2880 个 image tokens, 加上文本通常 ~3000+,
            # 配置 max_length 时务必给足空间(建议 >= 3500)。
            results_with_img = processor(
                text=texts_with_img,
                images=valid_images,
                padding=True,           # 只 pad 到 batch 内最长
                truncation=False,       # 关键: 不截断有图样本
                return_tensors="pt",
            )

        # 合并回完整的展平 batch
        out = _merge_branches(M, no_img_idx, with_img_idx, results_no_img, results_with_img,
                              processor.tokenizer.pad_token_id or 0)

        out["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        out["role_ids"] = torch.tensor(role_ids, dtype=torch.long)
        out["text_present"] = torch.tensor(text_present, dtype=torch.bool)
        out["img_present"] = torch.tensor(img_present, dtype=torch.bool)
        out["img_pos_in_flat"] = torch.tensor(img_pos_in_flat, dtype=torch.long)
        out["gold_idx"] = torch.tensor(gold_idx_list, dtype=torch.long)
        out["num_candidates"] = torch.tensor(num_candidates_list, dtype=torch.long)
        return out

    return collate_fn


def _append_unit(unit, b_idx, role, flat_texts, flat_images,
                 sample_ids, role_ids, text_present, img_present):
    text = unit.get("text")
    image = unit.get("image")
    flat_texts.append(text if text else "")
    flat_images.append(image)  # None 或 PIL.Image
    sample_ids.append(b_idx)
    role_ids.append(role)
    text_present.append(text is not None and text != "")
    img_present.append(image is not None)


def _merge_branches(M, no_img_idx, with_img_idx, results_no_img, results_with_img, pad_id):
    """
    把两个分支(无图/有图)按原展平顺序合并回 [M, L] 张量。
    pixel_values 只来自有图分支。
    """
    # 统一文本长度: 取两边最大 L
    def _get_L(r):
        return r["input_ids"].shape[1] if r is not None else 0
    L = max(_get_L(results_no_img), _get_L(results_with_img))

    def _pad_to(t, target_L, pad_value):
        if t.shape[1] == target_L:
            return t
        bsz, cur_L = t.shape
        pad = torch.full((bsz, target_L - cur_L), pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    input_ids = torch.full((M, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((M, L), dtype=torch.long)

    if results_no_img is not None:
        ids = _pad_to(results_no_img["input_ids"], L, pad_id)
        mask = _pad_to(results_no_img["attention_mask"], L, 0)
        for k, flat_i in enumerate(no_img_idx):
            input_ids[flat_i] = ids[k]
            attention_mask[flat_i] = mask[k]

    out = {"input_ids": input_ids, "attention_mask": attention_mask}

    if results_with_img is not None:
        ids = _pad_to(results_with_img["input_ids"], L, pad_id)
        mask = _pad_to(results_with_img["attention_mask"], L, 0)
        for k, flat_i in enumerate(with_img_idx):
            input_ids[flat_i] = ids[k]
            attention_mask[flat_i] = mask[k]

        out["pixel_values"] = results_with_img["pixel_values"]
        if "image_sizes" in results_with_img:
            out["image_sizes"] = results_with_img["image_sizes"]

    return out