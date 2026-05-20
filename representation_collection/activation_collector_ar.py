#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
activation_collector_ar.py

Arrow-based activation collector for LLaVA.

目标：
1. 保持最终保存产物不变：
   {
       'image_features': [...],
       'text_features': [...],
       'image_file': [...],
       'text': [...],
   }
2. 保持 save_result() 输出命名逻辑不变。
3. 提升 GPU 利用率：
   - 支持 batch 推理
   - 使用 DataLoader + workers 做并行解码/预处理
   - 减少主线程串行 CPU 供数
4. 提升 Arrow 读取稳健性：
   - 默认不用 memory_map
   - 遇到 ESTALE 可重试
"""

import io
import os
import os.path as osp
import re
import sys
import math
import time
import errno
import argparse
from pathlib import Path
from functools import partial

import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image
from tqdm import tqdm
from transformers import set_seed
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from hooks import InputHook


SHARD_RE = re.compile(r"cc3m-wds-train-(\d+)-of-(\d+)\.arrow$")


def open_arrow_table(path: str, use_mmap: bool = False, retries: int = 3, sleep_s: float = 1.0) -> pa.Table:
    last_err = None

    for attempt in range(retries):
        try:
            if use_mmap:
                with pa.memory_map(path, "r") as source:
                    try:
                        reader = ipc.open_stream(source)
                        return reader.read_all()
                    except Exception:
                        source.seek(0)
                        reader = ipc.open_file(source)
                        return reader.read_all()
            else:
                with pa.OSFile(path, "rb") as source:
                    try:
                        reader = ipc.open_stream(source)
                        return reader.read_all()
                    except Exception:
                        source.seek(0)
                        reader = ipc.open_file(source)
                        return reader.read_all()

        except OSError as e:
            last_err = e
            if e.errno == errno.ESTALE and attempt + 1 < retries:
                time.sleep(sleep_s * (attempt + 1))
                continue
            raise

    raise last_err


def parse_shard_info(path: Path):
    m = SHARD_RE.fullmatch(path.name)
    if not m:
        return None
    return {
        "path": path,
        "name": path.name,
        "index": int(m.group(1)),
        "total": int(m.group(2)),
    }


def list_arrow_files(arrow_dir: str, pattern: str, force_process_latest: bool = False):
    arrow_files = sorted(Path(arrow_dir).glob(pattern))
    if not arrow_files:
        return []

    parsed = [parse_shard_info(p) for p in arrow_files]
    parsed_ok = [x for x in parsed if x is not None]

    if len(parsed_ok) == len(arrow_files):
        parsed_ok.sort(key=lambda x: x["index"])
        arrow_files = [x["path"] for x in parsed_ok]
    else:
        arrow_files = sorted(arrow_files)

    if not force_process_latest and len(arrow_files) > 0:
        arrow_files = arrow_files[:-1]

    return arrow_files


def detect_text_col(table: pa.Table):
    for name in ("txt", "text", "caption"):
        if name in table.column_names:
            return name
    return None


def detect_key_col(table: pa.Table):
    for name in ("__key__", "key"):
        if name in table.column_names:
            return name
    return None


def detect_image_col(table: pa.Table):
    for name in ("jpg", "jpeg", "png", "webp", "image"):
        if name in table.column_names:
            return name

    for field in table.schema:
        if pa.types.is_struct(field.type):
            child_names = {field.type[i].name for i in range(field.type.num_fields)}
            if {"bytes", "path"}.issubset(child_names):
                return field.name

    return None


def choose_image_name(key, image_cell):
    key = str(key) if key is not None else "unknown"
    ext = ".jpg"

    if isinstance(image_cell, dict):
        original_path = image_cell.get("path")
        if original_path:
            original_ext = Path(original_path).suffix.lower()
            if original_ext:
                ext = original_ext

    return f"{key}{ext}"


def split_list(lst, n):
    if n <= 0:
        raise ValueError("n must be > 0")
    if len(lst) == 0:
        return []
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    if k < 0 or k >= len(chunks):
        return []
    return chunks[k]


def ensure_pil_image(image_cell):
    if image_cell is None:
        return None

    try:
        if isinstance(image_cell, dict):
            image_bytes = image_cell.get("bytes")
            if image_bytes is None:
                return None
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if isinstance(image_cell, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(image_cell)).convert("RGB")
    except Exception:
        return None

    return None


def build_prompt_text(model, caption: str):
    caption = "" if caption is None else str(caption)
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + caption
    return DEFAULT_IMAGE_TOKEN + "\n" + caption


def extract_records_from_arrow_table(table: pa.Table):
    text_col = detect_text_col(table)
    key_col = detect_key_col(table)
    image_col = detect_image_col(table)

    if text_col is None:
        raise ValueError(f"Cannot find text column in Arrow table. Available columns: {table.column_names}")
    if image_col is None:
        raise ValueError(f"Cannot find image column in Arrow table. Available columns: {table.column_names}")

    texts = table[text_col].to_pylist()
    image_cells = table[image_col].to_pylist()

    if key_col is not None:
        keys = table[key_col].to_pylist()
    else:
        keys = [None] * len(texts)

    records = []
    for text, key, image_cell in zip(texts, keys, image_cells):
        if text is None or image_cell is None:
            continue

        image_name = choose_image_name(key, image_cell)
        records.append(
            {
                "text": str(text),
                "image_cell": image_cell,
                "image_file": image_name,
            }
        )

    return records


class ArrowRecordDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample = self.records[idx]
        pil_image = ensure_pil_image(sample["image_cell"])
        if pil_image is None:
            return None

        return {
            "text": sample["text"],
            "image": pil_image,
            "image_file": sample["image_file"],
        }


def left_pad_1d_tensors(tensors, pad_value):
    if len(tensors) == 0:
        raise ValueError("tensors must not be empty")

    max_len = max(x.shape[0] for x in tensors)
    out = []

    for x in tensors:
        if x.shape[0] == max_len:
            out.append(x)
            continue
        pad_len = max_len - x.shape[0]
        pad = torch.full((pad_len,), pad_value, dtype=x.dtype)
        out.append(torch.cat([pad, x], dim=0))

    return torch.stack(out, dim=0)


def collate_samples(batch, tokenizer, model, image_processor, args):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    input_ids_list = []
    image_tensors = []
    texts = []
    image_files = []

    for sample in batch:
        text = build_prompt_text(model, sample["text"])

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )

        image_tensor = image_processor.preprocess(
            sample["image"],
            return_tensors='pt'
        )['pixel_values'][0]

        input_ids_list.append(input_ids)
        image_tensors.append(image_tensor)
        texts.append(text)
        image_files.append(sample["image_file"])

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = left_pad_1d_tensors(input_ids_list, pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()
    images = torch.stack(image_tensors, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "texts": texts,
        "image_files": image_files,
    }

def collect_batch_activation(
    model,
    batch,
    args,
):
    target_layer_name = args.target_layer_name
    image_tokens_seq_len = args.image_tokens_seq_len

    input_ids = batch["input_ids"].cuda(non_blocking=args.pin_memory)
    attention_mask = batch["attention_mask"].cuda(non_blocking=args.pin_memory)
    images = batch["images"].half().cuda(non_blocking=args.pin_memory)

    with InputHook(model, outputs=[target_layer_name], as_tensor=True) as h:
        with torch.inference_mode():
            _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        returned_features = h.layer_outputs[target_layer_name]
        if isinstance(returned_features, (list, tuple)):
            returned_features = returned_features[0]

        # 期望形状: [B, S_expanded, H]
        if returned_features.dim() != 3:
            raise RuntimeError(
                f"Unexpected hooked feature shape for {target_layer_name}: {tuple(returned_features.shape)}"
            )

    results = []
    batch_size = input_ids.shape[0]

    for i in range(batch_size):
        # 注意：不要用 attention_mask 去裁剪 returned_features
        # input_ids 还是“原始输入序列”，其中图像只有 1 个 IMAGE_TOKEN_INDEX
        # returned_features 是“模型内部展开后的序列”，图像位置已展开成 image_tokens_seq_len 个 token
        sample_input_ids = input_ids[i]
        sample_hidden = returned_features[i]

        image_positions = torch.where(sample_input_ids == IMAGE_TOKEN_INDEX)[0]
        if image_positions.numel() == 0:
            continue

        image_start_idx = int(image_positions[0].item())
        image_end_idx = image_start_idx + image_tokens_seq_len

        if image_end_idx > sample_hidden.shape[0]:
            print(
                f"[WARN] Skip sample {batch['image_files'][i]}: "
                f"expanded hidden length {sample_hidden.shape[0]} < image_end_idx {image_end_idx}",
                flush=True,
            )
            continue

        image_features = sample_hidden[image_start_idx:image_end_idx, :].mean(0)

        text_slice = sample_hidden[image_end_idx:, :]
        if text_slice.numel() == 0:
            text_features = torch.zeros_like(image_features)
        else:
            text_features = text_slice.mean(0)

        results.append(
            {
                "image_features": image_features.detach().cpu().numpy(),
                "text_features": text_features.detach().cpu().numpy(),
                "image_file": batch["image_files"][i],
                "text": batch["texts"][i],
            }
        )

    return results


def collect_from_arrow_files(args, tokenizer, model, image_processor, arrow_files):
    image_features_list = []
    text_features_list = []
    image_files = []
    texts = []

    collate_fn = partial(
        collate_samples,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        args=args,
    )

    for arrow_path in arrow_files:
        table = open_arrow_table(
            str(arrow_path),
            use_mmap=args.use_mmap,
            retries=args.arrow_open_retries,
            sleep_s=args.arrow_open_retry_sleep,
        )
        records = extract_records_from_arrow_table(table)
        dataset = ArrowRecordDataset(records)

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = args.persistent_workers
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        dataloader = DataLoader(**loader_kwargs)

        for batch in tqdm(dataloader, desc=f"Inference [{arrow_path.name}]"):
            if batch is None:
                continue

            try:
                batch_results = collect_batch_activation(
                    model=model,
                    batch=batch,
                    args=args,
                )
            except Exception as e:
                print(f"[WARN] Skip batch from {arrow_path.name}: {e}", flush=True)
                continue

            for result in batch_results:
                image_features_list.append(result["image_features"])
                text_features_list.append(result["text_features"])
                image_files.append(result["image_file"])
                texts.append(result["text"])

    return {
        'image_features': image_features_list,
        'text_features': text_features_list,
        'image_file': image_files,
        'text': texts,
    }


def save_result(feature_dict, save_path, target_layer_name, shard_name=None, prefix="llava_cc3m"):
    os.makedirs(save_path, exist_ok=True)

    if shard_name is None:
        out_file = osp.join(save_path, f"{prefix}_{target_layer_name}_embeddings.pt")
    else:
        out_file = osp.join(save_path, f"{prefix}_{shard_name}_{target_layer_name}_embeddings.pt")

    torch.save(feature_dict, out_file)
    print(f"[INFO] Saved embeddings to {out_file}", flush=True)


def get_args_parser():
    parser = argparse.ArgumentParser("Collect LLaVA activations from Arrow shards", add_help=True)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--arrow_dir", type=str, required=True)
    parser.add_argument("--arrow_pattern", type=str, default="cc3m-wds-train-*.arrow")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--save_per_shard", action="store_true")
    parser.add_argument("--force_process_latest", action="store_true")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1)

    parser.add_argument("--target_layer_name", type=str, default="model.layers.30")
    parser.add_argument("--image_tokens_seq_len", type=int, default=576)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_prefix", type=str, default="llava_cc3m")

    # 性能相关参数
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # Arrow 打开策略
    parser.add_argument("--use_mmap", action="store_true")
    parser.add_argument("--arrow_open_retries", type=int, default=3)
    parser.add_argument("--arrow_open_retry_sleep", type=float, default=1.0)

    return parser


def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
    )
    model.eval()

    arrow_files = list_arrow_files(
        arrow_dir=args.arrow_dir,
        pattern=args.arrow_pattern,
        force_process_latest=args.force_process_latest,
    )
    if len(arrow_files) == 0:
        raise FileNotFoundError(
            f"No usable arrow files found in {args.arrow_dir} with pattern {args.arrow_pattern}"
        )

    if args.num_chunks > 1:
        arrow_files = get_chunk(arrow_files, args.num_chunks, args.chunk_idx)
        if len(arrow_files) == 0:
            raise ValueError(
                f"No arrow files assigned to chunk {args.chunk_idx} when num_chunks={args.num_chunks}"
            )

    if args.save_per_shard:
        for arrow_path in arrow_files:
            shard_name = Path(arrow_path).stem
            feature_dict = collect_from_arrow_files(
                args=args,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                arrow_files=[arrow_path],
            )
            save_result(
                feature_dict=feature_dict,
                save_path=args.save_path,
                target_layer_name=args.target_layer_name,
                shard_name=shard_name,
                prefix=args.save_prefix,
            )
        return

    feature_dict = collect_from_arrow_files(
        args=args,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        arrow_files=arrow_files,
    )
    save_result(
        feature_dict=feature_dict,
        save_path=args.save_path,
        target_layer_name=args.target_layer_name,
        shard_name=None,
        prefix=args.save_prefix,
    )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)