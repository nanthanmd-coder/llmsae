import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def list_latent_datasets(latent_root: str) -> List[str]:
    """Scan latent_root and return dataset directories that contain a latent pt file."""
    names = []
    if not os.path.isdir(latent_root):
        return names

    for name in sorted(os.listdir(latent_root)):
        full = os.path.join(latent_root, name)
        if not os.path.isdir(full):
            continue
        if name.startswith(".") or name == "__pycache__":
            continue

        candidates = [
            os.path.join(full, "latents.pt"),
            os.path.join(full, "latent.pt"),
            os.path.join(full, "activations.pt"),
        ]
        if any(os.path.isfile(p) for p in candidates):
            names.append(name)

    return names


def resolve_dataset_names(args) -> List[str]:
    if args.dataset is not None and args.datasets is not None:
        raise ValueError("Use either --dataset or --datasets, not both.")
    if args.dataset is not None:
        return [args.dataset]
    if args.datasets is not None:
        return list(args.datasets)
    return list_latent_datasets(args.latent_root)


def resolve_dataset_pt_path(latent_root: str, dataset_name: str) -> str:
    full = os.path.join(latent_root, dataset_name)
    candidates = [
        os.path.join(full, "latents.pt"),
        os.path.join(full, "latent.pt"),
        os.path.join(full, "activations.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"No pt file found for dataset {dataset_name} under {full}")


def gather_rows(bank: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    N = index.shape[0]
    D = bank.shape[1]
    out = torch.zeros((N, D), dtype=bank.dtype)
    valid = index >= 0
    if valid.any():
        out[valid] = bank[index[valid]]
    return out


class SavedPtPairDataset(Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def build_samples_from_one_pt(pt_path: str) -> List[Dict[str, torch.Tensor]]:
    data = torch.load(pt_path, map_location="cpu")

    required_keys = [
        "q_text_emb",
        "q_img_emb",
        "t_text_emb",
        "t_img_emb",
        "text_index",
        "img_index",
        "candidate_counts",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in {pt_path}: {missing}")

    q_text_emb = data["q_text_emb"].float()
    q_img_emb = data["q_img_emb"].float()
    q_text_mask = data.get("q_text_mask", torch.ones(q_text_emb.shape[0], dtype=torch.bool)).bool()
    q_img_mask = data.get("q_img_mask", torch.ones(q_img_emb.shape[0], dtype=torch.bool)).bool()

    t_text_emb = data["t_text_emb"].float()
    t_img_emb = data["t_img_emb"].float()
    text_index = data["text_index"].long()
    img_index = data["img_index"].long()
    candidate_counts = data["candidate_counts"].long()

    pos_text_idx = text_index[:, 0].clone()
    pos_img_idx = img_index[:, 0].clone()

    invalid_samples = candidate_counts <= 0
    pos_text_idx[invalid_samples] = -1
    pos_img_idx[invalid_samples] = -1

    pos_text_emb = gather_rows(t_text_emb, pos_text_idx)
    pos_img_emb = gather_rows(t_img_emb, pos_img_idx)

    pos_text_mask = pos_text_idx >= 0
    pos_img_mask = pos_img_idx >= 0

    samples = []
    N = q_text_emb.shape[0]
    for i in range(N):
        samples.append({
            "q_text_emb": q_text_emb[i],
            "q_img_emb": q_img_emb[i],
            "q_text_mask": q_text_mask[i],
            "q_img_mask": q_img_mask[i],
            "p_text_emb": pos_text_emb[i],
            "p_img_emb": pos_img_emb[i],
            "p_text_mask": pos_text_mask[i],
            "p_img_mask": pos_img_mask[i],
        })

    return samples


def load_all_samples(latent_root: str, dataset_names: List[str]) -> List[Dict[str, torch.Tensor]]:
    all_samples = []
    for dataset_name in dataset_names:
        pt_path = resolve_dataset_pt_path(latent_root, dataset_name)
        samples = build_samples_from_one_pt(pt_path)
        all_samples.extend(samples)
        print(f"[INFO] Loaded {len(samples)} samples from {dataset_name}", flush=True)
    return all_samples


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([x[k] for x in batch], dim=0)
    return out


def filter_batch(batch: Dict[str, torch.Tensor], device: str) -> Optional[Dict[str, torch.Tensor]]:
    q_text_emb = batch["q_text_emb"].to(device, non_blocking=True)
    q_img_emb = batch["q_img_emb"].to(device, non_blocking=True)
    q_text_mask = batch["q_text_mask"].to(device, non_blocking=True).bool()
    q_img_mask = batch["q_img_mask"].to(device, non_blocking=True).bool()

    p_text_emb = batch["p_text_emb"].to(device, non_blocking=True)
    p_img_emb = batch["p_img_emb"].to(device, non_blocking=True)
    p_text_mask = batch["p_text_mask"].to(device, non_blocking=True).bool()
    p_img_mask = batch["p_img_mask"].to(device, non_blocking=True).bool()

    valid = (q_text_mask | q_img_mask) & (p_text_mask | p_img_mask)
    if valid.sum().item() <= 1:
        return None

    return {
        "q_text_emb": q_text_emb[valid],
        "q_img_emb": q_img_emb[valid],
        "q_text_mask": q_text_mask[valid],
        "q_img_mask": q_img_mask[valid],
        "p_text_emb": p_text_emb[valid],
        "p_img_emb": p_img_emb[valid],
        "p_text_mask": p_text_mask[valid],
        "p_img_mask": p_img_mask[valid],
    }


def split_samples(
    samples: List[Dict[str, torch.Tensor]],
    training_ratio: float,
    shuffle: bool,
    seed: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    train_size = int(len(samples) * training_ratio)

    train_samples = [samples[i] for i in indices[:train_size]]
    val_samples = [samples[i] for i in indices[train_size:]]

    if shuffle:
        random.shuffle(train_samples)
        random.shuffle(val_samples)

    return train_samples, val_samples


def build_dataloaders(args, dataset_names: List[str]) -> Tuple[DataLoader, DataLoader, int, int]:
    samples = load_all_samples(args.latent_root, dataset_names)
    if len(samples) == 0:
        raise RuntimeError("No samples loaded from pt files.")

    train_samples, val_samples = split_samples(
        samples=samples,
        training_ratio=args.training_ratio,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    train_loader = DataLoader(
        SavedPtPairDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        SavedPtPairDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, len(train_samples), len(val_samples)
