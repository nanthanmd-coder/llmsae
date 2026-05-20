#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
llava_next_activation_collector.py

Collect token-level hidden activations from LLaVA-NeXT-Llama3-8B-HF
for MMEB parquet datasets.

This replaces the OpenCLIP collector pattern:

    OpenCLIP encode_text / encode_image -> [N, D]

with:

    LLaVA-NeXT image+text prompt -> hook language_model layer -> [N, S, D]

Default model:
    llava-hf/llama3-llava-next-8b-hf

Default hooked layer:
    layer 24

Saved format per dataset:
    <save_root>/<DATASET>/activations.pt

Saved keys:
{
    "dataset_name": str,
    "model_id": str,
    "layer_idx": int,
    "hidden_size": int,                  # D, usually 4096 for LLaMA3-8B
    "save_dtype": str,

    "num_samples": int,                  # N
    "num_targets": int,                  # M = unique valid target text-image pairs
    "num_candidates": int,               # K = max candidate count

    "q_emb": FloatTensor [N, Sq, D],     # padded query activations
    "q_emb_mask": BoolTensor [N, Sq],    # valid token positions in q_emb
    "q_seq_lens": LongTensor [N],
    "q_valid_mask": BoolTensor [N],
    "q_text": list[str],
    "q_img_url": list[str],

    "t_emb": FloatTensor [M, St, D],     # padded unique target-pair activations
    "t_emb_mask": BoolTensor [M, St],
    "t_seq_lens": LongTensor [M],
    "t_text": list[str],
    "t_img_url": list[str],

    "target_index": LongTensor [N, K],   # map each candidate -> row in t_emb, -1 if absent
    "candidate_counts": LongTensor [N],
}

Notes:
1. This is token-level activation collection, not pooled embedding collection.
2. Sq and St are padded sequence lengths, so they may differ.
3. With LLaVA-NeXT, image tokens expand inside the model. For robustness this script
   defaults to batch_size=1. You can increase it, but batch_size=1 is closest to
   the SAE training pipeline used by multimodal-sae.
4. NSD tensors can become very large. Use --max-samples for debugging first.
"""

import os
import ast
import glob
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


# =============================================================================
# Basic helpers copied/adapted from the OpenCLIP collector
# =============================================================================


def is_nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def normalize_to_list(x):
    if x is None:
        return []

    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)

    if hasattr(x, "tolist") and not isinstance(x, str):
        try:
            y = x.tolist()
            if isinstance(y, list):
                return y
            return [y]
        except Exception:
            pass

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                y = ast.literal_eval(s)
                if isinstance(y, list):
                    return y
                return [y]
            except Exception:
                return [x]
        return [x]

    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    return [x]


def image_stem_from_relpath(rel_path: str) -> str:
    if not is_nonempty_str(rel_path):
        return ""
    return Path(rel_path).stem


def resolve_image_path(
    rel_path: str,
    data_root: str,
    images_root: Optional[str] = None,
) -> Optional[str]:
    if not is_nonempty_str(rel_path):
        return None

    rel_path = rel_path.strip()

    # Skip AppleDouble metadata files like ._xxx.jpg
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
        if p and os.path.exists(p):
            if os.path.basename(p).startswith("._"):
                continue
            return p

    return None


def list_dataset_dirs(data_root: str) -> List[str]:
    names = []
    for x in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, x)
        if not os.path.isdir(full):
            continue
        if x in {"images", "activations", "__pycache__"}:
            continue
        if x.startswith("."):
            continue

        parquets = glob.glob(os.path.join(full, "*.parquet"))
        if len(parquets) == 0:
            parquets = glob.glob(os.path.join(full, "**", "*.parquet"), recursive=True)

        if len(parquets) > 0:
            names.append(x)

    return names


def find_parquet_files(dataset_dir: str) -> List[str]:
    parquets = sorted(glob.glob(os.path.join(dataset_dir, "*.parquet")))
    if len(parquets) == 0:
        parquets = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True))
    return parquets


def read_dataset_parquets(dataset_dir: str) -> pd.DataFrame:
    parquet_files = find_parquet_files(dataset_dir)
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet found under {dataset_dir}")

    dfs = []
    for p in parquet_files:
        df = pd.read_parquet(p)
        df["__source_parquet__"] = os.path.basename(p)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    required = ["qry_text", "qry_img_path", "tgt_text", "tgt_img_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} in {dataset_dir}")

    return df


def build_structured_lists(
    df: pd.DataFrame,
    data_root: str,
    images_root: Optional[str],
):
    """
    Build all query/target lists in one pass.

    Returns:
        q_texts: List[str] length N
        q_img_urls: List[str] length N
        q_img_resolved: List[Optional[str]] length N

        t_texts_nested: List[List[str]] length N
        t_img_urls_nested: List[List[str]] length N
        t_img_resolved_nested: List[List[Optional[str]]] length N

        candidate_counts: List[int] length N
    """
    q_texts = []
    q_img_urls = []
    q_img_resolved = []

    t_texts_nested = []
    t_img_urls_nested = []
    t_img_resolved_nested = []

    candidate_counts = []

    for _, row in df.iterrows():
        qry_text = row["qry_text"] if is_nonempty_str(row["qry_text"]) else ""
        qry_img_rel = row["qry_img_path"] if is_nonempty_str(row["qry_img_path"]) else ""

        q_texts.append(qry_text)
        q_img_urls.append(image_stem_from_relpath(qry_img_rel))
        q_img_resolved.append(resolve_image_path(qry_img_rel, data_root=data_root, images_root=images_root))

        tgt_texts = normalize_to_list(row["tgt_text"])
        tgt_imgs = normalize_to_list(row["tgt_img_path"])

        n = max(len(tgt_texts), len(tgt_imgs))
        candidate_counts.append(n)

        row_t_texts = []
        row_t_img_urls = []
        row_t_img_resolved = []

        for i in range(n):
            tt = tgt_texts[i] if i < len(tgt_texts) and is_nonempty_str(tgt_texts[i]) else ""
            ti_rel = tgt_imgs[i] if i < len(tgt_imgs) and is_nonempty_str(tgt_imgs[i]) else ""

            row_t_texts.append(tt)
            row_t_img_urls.append(image_stem_from_relpath(ti_rel))
            row_t_img_resolved.append(
                resolve_image_path(ti_rel, data_root=data_root, images_root=images_root)
            )

        t_texts_nested.append(row_t_texts)
        t_img_urls_nested.append(row_t_img_urls)
        t_img_resolved_nested.append(row_t_img_resolved)

    return (
        q_texts,
        q_img_urls,
        q_img_resolved,
        t_texts_nested,
        t_img_urls_nested,
        t_img_resolved_nested,
        candidate_counts,
    )


# =============================================================================
# LLaVA-NeXT activation collector
# =============================================================================


def get_save_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported save dtype: {name}")


def get_model_dtype(name: str) -> torch.dtype:
    return get_save_dtype(name)


def get_llm_layer(model, layer_idx: int):
    """
    Robustly find the LLM transformer layer in LlavaNextForConditionalGeneration.
    For llava-hf/llama3-llava-next-8b-hf this is usually:
        model.language_model.model.layers[layer_idx]
    """
    candidates = []

    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            candidates.append(lm.model.layers)
        if hasattr(lm, "layers"):
            candidates.append(lm.layers)

    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            candidates.append(inner.layers)
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            candidates.append(inner.model.layers)

    for layers in candidates:
        if layer_idx < len(layers):
            return layers[layer_idx]

    raise AttributeError(
        f"Unable to find LLM layer {layer_idx}. "
        "Inspect the model structure and update get_llm_layer()."
    )


class LlavaNextLayerHook:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.cache: Dict[str, torch.Tensor] = {}
        self.handle = None

    def __enter__(self):
        layer = get_llm_layer(self.model, self.layer_idx)

        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self.cache["hidden"] = output.detach()

        self.handle = layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def build_llava_prompt(
    processor,
    text: str,
    has_image: bool,
    empty_text_prompt: str,
) -> str:
    text = text.strip() if isinstance(text, str) else ""

    # If image-only, add a neutral text prompt so the chat template remains well-formed.
    if has_image and not text:
        text = empty_text_prompt

    # If neither text nor image exists, keep a placeholder text but the caller should mark it invalid.
    if not has_image and not text:
        text = empty_text_prompt

    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    # Fallback for older processors. Prefer the chat template whenever available.
    image_prefix = "<image>\n" if has_image else ""
    return f"USER: {image_prefix}{text}\nASSISTANT:"


def load_pil_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def collect_llava_hidden_batch(
    model,
    processor,
    hook: LlavaNextLayerHook,
    texts: List[str],
    image_paths: List[Optional[str]],
    device: str,
    empty_text_prompt: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Collect hidden activations for a homogeneous mini-batch:
      - either all examples have images
      - or none have images

    Returns:
        per_item_hidden: List[Tensor [S_i_or_S_batch, D]]
        per_item_mask:   List[BoolTensor [S_i_or_S_batch]]

    In practice, the model output is padded to one S per mini-batch.
    If the layer output length matches input attention_mask length, that mask is used.
    Otherwise all output positions are treated as valid. With batch_size=1 this is
    close to the official SAE training behavior.
    """
    assert len(texts) == len(image_paths)

    has_images = [p is not None for p in image_paths]
    if len(set(has_images)) != 1:
        raise ValueError("collect_llava_hidden_batch expects homogeneous image presence.")

    prompts = [
        build_llava_prompt(
            processor=processor,
            text=text,
            has_image=has_img,
            empty_text_prompt=empty_text_prompt,
        )
        for text, has_img in zip(texts, has_images)
    ]

    images = None
    if has_images[0]:
        images = [load_pil_image(p) for p in image_paths]  # type: ignore[arg-type]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    hook.cache.clear()
    _ = model(**inputs, use_cache=False)

    if "hidden" not in hook.cache:
        raise RuntimeError("Layer hook did not capture hidden states.")

    hidden = hook.cache["hidden"]  # [B, S, D]
    if hidden.ndim != 3:
        raise RuntimeError(f"Expected hidden [B,S,D], got shape={tuple(hidden.shape)}")

    B, S, _ = hidden.shape

    # The original input attention_mask may or may not match the merged multimodal sequence length.
    input_mask = inputs.get("attention_mask", None)
    if torch.is_tensor(input_mask) and input_mask.ndim == 2 and input_mask.shape[0] == B and input_mask.shape[1] == S:
        mask = input_mask.bool()
    else:
        mask = torch.ones((B, S), dtype=torch.bool, device=hidden.device)

    per_item_hidden = []
    per_item_mask = []
    for i in range(B):
        per_item_hidden.append(hidden[i].detach().cpu())
        per_item_mask.append(mask[i].detach().cpu())

    return per_item_hidden, per_item_mask


def pad_hidden_list(
    hidden_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
    save_dtype: torch.dtype,
    pad_to_multiple: int = 1,
) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Pad a list of [S_i, D] tensors into [N, S_max, D].
    """
    if len(hidden_list) == 0:
        return (
            torch.empty((0, 0, 0), dtype=save_dtype),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
        )

    D = hidden_list[0].shape[-1]
    seq_lens = torch.tensor([int(m.sum().item()) for m in mask_list], dtype=torch.long)
    max_s = max(int(x.shape[0]) for x in hidden_list)

    if pad_to_multiple and pad_to_multiple > 1:
        max_s = ((max_s + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

    out = torch.zeros((len(hidden_list), max_s, D), dtype=save_dtype)
    out_mask = torch.zeros((len(hidden_list), max_s), dtype=torch.bool)

    for i, (h, m) in enumerate(zip(hidden_list, mask_list)):
        s = h.shape[0]
        out[i, :s] = h.to(dtype=save_dtype)
        out_mask[i, :s] = m.bool()

    return out, out_mask, seq_lens


def collect_items_to_nsd(
    model,
    processor,
    hook: LlavaNextLayerHook,
    items: List[Dict[str, Any]],
    batch_size: int,
    device: str,
    save_dtype: torch.dtype,
    empty_text_prompt: str,
    desc: str,
    pad_to_multiple: int,
) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor, torch.BoolTensor]:
    """
    Collect [N,S,D] activations for a list of multimodal items.

    Each item:
        {
            "text": str,
            "image_path": Optional[str],
            "valid": bool,
        }

    Returns:
        emb: [N,S,D]
        emb_mask: [N,S]
        seq_lens: [N]
        valid_mask: [N]
    """
    all_hidden: List[Optional[torch.Tensor]] = [None] * len(items)
    all_masks: List[Optional[torch.Tensor]] = [None] * len(items)
    valid_mask = torch.tensor([bool(x.get("valid", True)) for x in items], dtype=torch.bool)

    for start in tqdm(range(0, len(items), batch_size), desc=desc):
        chunk = items[start:start + batch_size]

        # Split by image presence to avoid mixed processor alignment issues.
        groups = {
            True: [],
            False: [],
        }
        for local_idx, item in enumerate(chunk):
            has_image = item.get("image_path", None) is not None
            groups[has_image].append((local_idx, item))

        for has_image, group in groups.items():
            if len(group) == 0:
                continue

            local_indices = [x[0] for x in group]
            group_items = [x[1] for x in group]
            texts = [x.get("text", "") for x in group_items]
            image_paths = [x.get("image_path", None) for x in group_items]

            hiddens, masks = collect_llava_hidden_batch(
                model=model,
                processor=processor,
                hook=hook,
                texts=texts,
                image_paths=image_paths,
                device=device,
                empty_text_prompt=empty_text_prompt,
            )

            for j, local_idx in enumerate(local_indices):
                global_idx = start + local_idx
                if not valid_mask[global_idx]:
                    # Keep the shape compatible but mask out all tokens.
                    all_hidden[global_idx] = torch.zeros_like(hiddens[j])
                    all_masks[global_idx] = torch.zeros_like(masks[j], dtype=torch.bool)
                else:
                    all_hidden[global_idx] = hiddens[j]
                    all_masks[global_idx] = masks[j]

    if any(x is None for x in all_hidden) or any(x is None for x in all_masks):
        missing = [i for i, x in enumerate(all_hidden) if x is None]
        raise RuntimeError(f"Some items were not encoded: {missing[:10]}")

    emb, emb_mask, seq_lens = pad_hidden_list(
        hidden_list=[x for x in all_hidden if x is not None],
        mask_list=[x for x in all_masks if x is not None],
        save_dtype=save_dtype,
        pad_to_multiple=pad_to_multiple,
    )

    return emb, emb_mask, seq_lens, valid_mask


def build_target_pair_bank(
    t_texts_nested: List[List[str]],
    t_img_urls_nested: List[List[str]],
    t_img_resolved_nested: List[List[Optional[str]]],
) -> Tuple[List[Dict[str, Any]], List[str], List[str], torch.LongTensor, int]:
    """
    Build a unique target bank over text-image pairs, not separate text and image banks.

    Returns:
        target_items: List[{"text", "image_path", "valid"}], length M
        target_texts: List[str], length M
        target_img_urls: List[str], length M
        target_index: LongTensor [N,K]
        K: int
    """
    N = len(t_texts_nested)
    K = max((len(x) for x in t_texts_nested), default=0)

    target_index = torch.full((N, K), -1, dtype=torch.long)

    pair_to_idx: Dict[Tuple[str, str], int] = {}
    target_items: List[Dict[str, Any]] = []
    target_texts: List[str] = []
    target_img_urls: List[str] = []

    for i in range(N):
        row_texts = t_texts_nested[i]
        row_img_urls = t_img_urls_nested[i]
        row_img_paths = t_img_resolved_nested[i]

        row_k = max(len(row_texts), len(row_img_paths))
        for j in range(row_k):
            text = row_texts[j] if j < len(row_texts) and is_nonempty_str(row_texts[j]) else ""
            img_path = row_img_paths[j] if j < len(row_img_paths) else None
            img_url = row_img_urls[j] if j < len(row_img_urls) else ""

            # No text and no valid image: absent candidate.
            if not is_nonempty_str(text) and img_path is None:
                continue

            key = (text, img_path or "")
            if key not in pair_to_idx:
                pair_to_idx[key] = len(target_items)
                target_items.append(
                    {
                        "text": text,
                        "image_path": img_path,
                        "valid": True,
                    }
                )
                target_texts.append(text)
                target_img_urls.append(img_url)

            target_index[i, j] = pair_to_idx[key]

    return target_items, target_texts, target_img_urls, target_index, K


def collect_single_dataset(
    dataset_name: str,
    dataset_dir: str,
    data_root: str,
    images_root: Optional[str],
    save_root: str,
    model,
    processor,
    hook: LlavaNextLayerHook,
    batch_size: int,
    device: str,
    save_dtype: torch.dtype,
    save_dtype_name: str,
    empty_text_prompt: str,
    layer_idx: int,
    model_id: str,
    max_samples: Optional[int],
    pad_to_multiple: int,
):
    print(f"[INFO] Processing dataset: {dataset_name}", flush=True)

    df = read_dataset_parquets(dataset_dir)
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].reset_index(drop=True)

    num_samples = len(df)

    (
        q_texts,
        q_img_urls,
        q_img_resolved,
        t_texts_nested,
        t_img_urls_nested,
        t_img_resolved_nested,
        candidate_counts,
    ) = build_structured_lists(df, data_root=data_root, images_root=images_root)

    candidate_counts_tensor = torch.tensor(candidate_counts, dtype=torch.long)
    K_from_counts = int(candidate_counts_tensor.max().item()) if num_samples > 0 else 0

    print(
        f"[INFO] {dataset_name}: num_samples={num_samples}, "
        f"candidate_count_min={int(candidate_counts_tensor.min().item()) if num_samples > 0 else 0}, "
        f"candidate_count_max={K_from_counts}",
        flush=True,
    )

    # Query items: one joint text-image prompt per sample.
    query_items = []
    for text, img_path in zip(q_texts, q_img_resolved):
        valid = is_nonempty_str(text) or img_path is not None
        query_items.append(
            {
                "text": text,
                "image_path": img_path,
                "valid": valid,
            }
        )

    q_emb, q_emb_mask, q_seq_lens, q_valid_mask = collect_items_to_nsd(
        model=model,
        processor=processor,
        hook=hook,
        items=query_items,
        batch_size=batch_size,
        device=device,
        save_dtype=save_dtype,
        empty_text_prompt=empty_text_prompt,
        desc=f"{dataset_name} | query LLaVA layer {layer_idx}",
        pad_to_multiple=pad_to_multiple,
    )

    # Target items: unique joint text-image candidate bank.
    target_items, target_texts, target_img_urls, target_index, K_from_index = build_target_pair_bank(
        t_texts_nested=t_texts_nested,
        t_img_urls_nested=t_img_urls_nested,
        t_img_resolved_nested=t_img_resolved_nested,
    )

    if K_from_counts != K_from_index:
        raise RuntimeError(f"K mismatch: candidate_counts K={K_from_counts}, target_index K={K_from_index}")

    t_emb, t_emb_mask, t_seq_lens, t_valid_mask = collect_items_to_nsd(
        model=model,
        processor=processor,
        hook=hook,
        items=target_items,
        batch_size=batch_size,
        device=device,
        save_dtype=save_dtype,
        empty_text_prompt=empty_text_prompt,
        desc=f"{dataset_name} | target LLaVA layer {layer_idx}",
        pad_to_multiple=pad_to_multiple,
    )

    hidden_size = int(q_emb.shape[-1]) if q_emb.ndim == 3 and q_emb.numel() > 0 else int(t_emb.shape[-1])

    save_dict = {
        "dataset_name": dataset_name,
        "model_id": model_id,
        "layer_idx": layer_idx,
        "hidden_size": hidden_size,
        "save_dtype": save_dtype_name,

        "num_samples": num_samples,
        "num_targets": len(target_items),
        "num_candidates": K_from_counts,

        "q_emb": q_emb,                       # [N, Sq, D]
        "q_emb_mask": q_emb_mask,             # [N, Sq]
        "q_seq_lens": q_seq_lens,             # [N]
        "q_valid_mask": q_valid_mask,         # [N]
        "q_text": q_texts,
        "q_img_url": q_img_urls,

        "t_emb": t_emb,                       # [M, St, D]
        "t_emb_mask": t_emb_mask,             # [M, St]
        "t_seq_lens": t_seq_lens,             # [M]
        "t_valid_mask": t_valid_mask,         # [M]
        "t_text": target_texts,
        "t_img_url": target_img_urls,

        "target_index": target_index,         # [N, K]
        "candidate_counts": candidate_counts_tensor,

        # For compatibility checks only. This is not the same as old text_index/img_index.
        "format": "llava_next_layer_hidden_nsd",
    }

    save_dir = os.path.join(save_root, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "activations.pt")
    torch.save(save_dict, save_path)

    print(f"[SHAPE] {dataset_name} q_emb        : {tuple(q_emb.shape)}", flush=True)
    print(f"[SHAPE] {dataset_name} q_emb_mask   : {tuple(q_emb_mask.shape)}", flush=True)
    print(f"[SHAPE] {dataset_name} t_emb        : {tuple(t_emb.shape)}", flush=True)
    print(f"[SHAPE] {dataset_name} t_emb_mask   : {tuple(t_emb_mask.shape)}", flush=True)
    print(f"[SHAPE] {dataset_name} target_index : {tuple(target_index.shape)}", flush=True)
    print(f"[INFO ] {dataset_name} num_targets  : {len(target_items)}", flush=True)
    print(f"[INFO ] Saved to {save_path}", flush=True)


# =============================================================================
# CLI
# =============================================================================


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Collect LLaVA-NeXT layer activations from MMEB parquet datasets",
        add_help=True,
    )

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]

    default_data_root = project_root / "MMEB-eval"
    default_images_root = default_data_root / "images"
    default_save_root = script_dir / "llava_next_activations"

    parser.add_argument("--data_root", default=str(default_data_root), type=str)
    parser.add_argument("--images_root", default=str(default_images_root), type=str)
    parser.add_argument("--save_root", default=str(default_save_root), type=str)

    parser.add_argument("--dataset", default=None, type=str, help="Process only one dataset, e.g. ChartQA")
    parser.add_argument("--datasets", nargs="+", default=None, help="Process multiple datasets by name")

    parser.add_argument(
        "--model_id",
        default="llava-hf/llama3-llava-next-8b-hf",
        type=str,
        help="HF model id or local path for LLaVA-NeXT.",
    )
    parser.add_argument("--layer_idx", default=24, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--empty_text_prompt", default="Describe the image.", type=str)

    parser.add_argument(
        "--model_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        type=str,
    )
    parser.add_argument(
        "--save_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        type=str,
        help="Dtype used when saving q_emb/t_emb.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        type=str,
    )
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="Debug only: process first N samples from each dataset.",
    )
    parser.add_argument(
        "--pad_to_multiple",
        default=1,
        type=int,
        help="Pad S dimension to a multiple of this value.",
    )

    return parser


def main(args):
    data_root = os.path.abspath(args.data_root)
    images_root = None if args.images_root is None else os.path.abspath(args.images_root)
    save_root = os.path.abspath(args.save_root)

    os.makedirs(save_root, exist_ok=True)

    if args.dataset is not None and args.datasets is not None:
        raise ValueError("Use either --dataset or --datasets, not both.")

    if args.dataset is not None:
        dataset_names = [args.dataset]
    elif args.datasets is not None:
        dataset_names = list(args.datasets)
    else:
        dataset_names = list_dataset_dirs(data_root)

    if len(dataset_names) == 0:
        raise RuntimeError(f"No dataset directories found under {data_root}")

    model_dtype = get_model_dtype(args.model_dtype)
    save_dtype = get_save_dtype(args.save_dtype)

    print(f"[INFO] data_root    = {data_root}", flush=True)
    print(f"[INFO] images_root = {images_root}", flush=True)
    print(f"[INFO] save_root   = {save_root}", flush=True)
    print(f"[INFO] datasets    = {dataset_names}", flush=True)
    print(f"[INFO] model_id    = {args.model_id}", flush=True)
    print(f"[INFO] layer_idx   = {args.layer_idx}", flush=True)
    print(f"[INFO] device      = {args.device}", flush=True)
    print(f"[INFO] model_dtype = {args.model_dtype}", flush=True)
    print(f"[INFO] save_dtype  = {args.save_dtype}", flush=True)
    print(f"[INFO] batch_size  = {args.batch_size}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_id)

    # For activation collection, right padding is easier to reason about.
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(args.device)
    model.eval()

    with LlavaNextLayerHook(model, args.layer_idx) as hook:
        for dataset_name in dataset_names:
            dataset_dir = os.path.join(data_root, dataset_name)
            if not os.path.isdir(dataset_dir):
                print(f"[WARN] Skip non-existing dataset dir: {dataset_dir}", flush=True)
                continue

            collect_single_dataset(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                data_root=data_root,
                images_root=images_root,
                save_root=save_root,
                model=model,
                processor=processor,
                hook=hook,
                batch_size=args.batch_size,
                device=args.device,
                save_dtype=save_dtype,
                save_dtype_name=args.save_dtype,
                empty_text_prompt=args.empty_text_prompt,
                layer_idx=args.layer_idx,
                model_id=args.model_id,
                max_samples=args.max_samples,
                pad_to_multiple=args.pad_to_multiple,
            )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
s