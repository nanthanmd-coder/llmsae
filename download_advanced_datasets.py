#!/usr/bin/env python3
"""
=====================================================================
  Download + normalize 4 Composed Image Retrieval (CIR) benchmarks
  into a *unified* schema.

  Datasets:
    - CIRCO         (ICCV 2023, miccunifi/CIRCO)
    - FashionIQ     (CVPR 2021, via Plachta/FashionIQ on HuggingFace, ~989MB)
    - HP-FashionIQ  (ICML 2025, jackwaky/QuRe)  - preference annotations on
                     FashionIQ val; images symlinked from FashionIQ.
    - PinPoint      (CVPR 2026, pinterest/pinpoint-dataset)  - SKIPPED by
                     default; pass `--only pinpoint` to fetch.

  Layout produced:
    ./advanced_datasets/
      <NAME>/
        images/                 image files, flat
        queries.json            unified query list (see schema below)
        image_index.json        list of filenames in images/
        _original/              untouched original files (annotations etc)

  Unified queries.json schema (one list element per query):
    {
      "id":          str/int,            # unique within this dataset
      "split":       "train"|"val"|"test",
      "subset":      str | null,         # FashionIQ: "dress"/"shirt"/"toptee"
      "reference":   str | [str],        # filename(s) in images/
      "captions":   [str, ...],          # 1+ relative captions
      "target":      str | null,         # primary GT filename, null if hidden
      "gt_set":     [str, ...],          # all relevant filenames (multi-GT)
      "negatives":  [str, ...] | null,   # hard negatives (PinPoint only)
      "extra":       { ... }             # dataset-specific bits
    }

  Usage:
      python download_advanced_datasets.py
          # downloads CIRCO + FashionIQ + HP-FashionIQ (NOT PinPoint)

      python download_advanced_datasets.py --only fashioniq
      python download_advanced_datasets.py --only pinpoint
      python download_advanced_datasets.py --skip-coco-images
          # for CIRCO: skip the ~19GB COCO images
      python download_advanced_datasets.py --root /data/cir
=====================================================================
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request


# -------------------------------------------------------------------
# tqdm is optional; fallback dummy if missing.
# -------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:                                              # type: ignore
        def __init__(self, total=None, **kw): self.total = total
        def update(self, n=1): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): self.close()
    print("[hint] `pip install tqdm` for nice progress bars\n")


CHUNK = 1 << 20   # 1 MiB


# =====================================================================
#  Generic helpers
# =====================================================================
def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def git_clone(url: str, dest: Path) -> None:
    """Shallow clone; skip if dest is already a git repo."""
    if (dest / ".git").exists():
        print(f"  [skip] {dest} already cloned")
        return
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError(
            f"{dest} exists and is not empty (and not a git repo). "
            f"Remove it or pick another --root.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", url, str(dest)])


def http_download(url: str, dest: Path, expected_min_size: int = 1,
                  ua: str = "Mozilla/5.0") -> None:
    """Streaming download with tqdm; skip if file exists & big enough."""
    if dest.exists() and dest.stat().st_size >= expected_min_size:
        print(f"  [skip] {dest.name} already downloaded "
              f"({dest.stat().st_size / 1e9:.2f} GB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    print(f"  -> {dest}")
    req = Request(url, headers={"User-Agent": ua})
    with urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                unit_divisor=1024, desc=dest.name, ncols=80) as pbar:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                pbar.update(len(buf))
        tmp.rename(dest)


def unzip_to(zip_path: Path, dest_dir: Path,
             marker: Path | None = None) -> None:
    if marker and marker.exists():
        print(f"  [skip] already extracted ({marker.name} exists)")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as z:
        members = z.namelist()
        with tqdm(total=len(members), unit="file",
                  desc=f"unzip {zip_path.name}", ncols=80) as pbar:
            for m in members:
                z.extract(m, dest_dir)
                pbar.update(1)


def unrar_to(rar_path: Path, dest_dir: Path,
             marker: Path | None = None) -> None:
    """
    Extract a .rar file. Tries (in order):
      1. python `rarfile` module
      2. `unrar x` command-line
      3. `7z x` command-line (handles rar)
      4. `unar` command-line
    """
    if marker and marker.exists():
        print(f"  [skip] already extracted ({marker} exists)")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. rarfile module
    try:
        import rarfile                                       # type: ignore
        print(f"  extracting {rar_path.name} via python rarfile -> {dest_dir}")
        with rarfile.RarFile(str(rar_path)) as rf:
            members = rf.namelist()
            with tqdm(total=len(members), unit="file",
                      desc=f"unrar {rar_path.name}", ncols=80) as pbar:
                for m in members:
                    rf.extract(m, str(dest_dir))
                    pbar.update(1)
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"  rarfile module failed ({e}); trying CLI tools...")

    # 2/3/4: try CLI tools
    for tool, args in [
        ("unrar", ["x", "-y", str(rar_path), str(dest_dir) + os.sep]),
        ("7z",    ["x", "-y", f"-o{dest_dir}", str(rar_path)]),
        ("unar",  ["-f", "-o", str(dest_dir), str(rar_path)]),
    ]:
        if have(tool):
            print(f"  extracting via `{tool}`...")
            run([tool] + args)
            return

    raise RuntimeError(
        "No way to extract .rar found. Install one of:\n"
        "    pip install rarfile             (needs `unar` or `unrar` on PATH)\n"
        "    brew install unar               (macOS)\n"
        "    apt-get install unrar unar p7zip-full   (Linux)")


def move_contents(src: Path, dst: Path, skip_names=(".git",)) -> None:
    """Move all entries from src into dst (without overwriting existing)."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip_names:
            continue
        target = dst / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))


def find_dir(root: Path, name: str, max_depth: int = 4) -> Path | None:
    """Find first sub-dir named `name` under root, BFS up to max_depth."""
    queue = [(root, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if not cur.is_dir():
            continue
        if cur.name == name:
            return cur
        if depth >= max_depth:
            continue
        try:
            for child in cur.iterdir():
                if child.is_dir():
                    queue.append((child, depth + 1))
        except PermissionError:
            pass
    return None


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =====================================================================
#  CIRCO
# =====================================================================
def download_circo(root: Path, skip_coco_images: bool = False) -> None:
    print("\n" + "=" * 70)
    print(" CIRCO  (ICCV 2023)")
    print("=" * 70)
    base = root / "CIRCO"
    original = base / "_original"

    # 1. Clone CIRCO repo into _original/
    if not (original / "annotations" / "val.json").exists():
        tmp_clone = root / "_tmp_circo_clone"
        git_clone("https://github.com/miccunifi/CIRCO.git", tmp_clone)
        original.mkdir(parents=True, exist_ok=True)
        move_contents(tmp_clone, original)
        shutil.rmtree(tmp_clone, ignore_errors=True)
    else:
        print(f"  [skip] CIRCO annotations already in {original}")

    # 2. COCO 2017 unlabeled
    if skip_coco_images:
        print("  --skip-coco-images set: skipping ~19GB COCO download.")
        return

    coco_dir = original / "COCO2017_unlabeled"
    coco_dir.mkdir(parents=True, exist_ok=True)

    img_zip = coco_dir / "unlabeled2017.zip"
    http_download(
        "http://images.cocodataset.org/zips/unlabeled2017.zip",
        img_zip, expected_min_size=18_000_000_000)
    unzip_to(img_zip, coco_dir, marker=coco_dir / "unlabeled2017")

    ann_zip = coco_dir / "image_info_unlabeled2017.zip"
    http_download(
        "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
        ann_zip, expected_min_size=1_000_000)
    unzip_to(ann_zip, coco_dir,
             marker=coco_dir / "annotations" / "image_info_unlabeled2017.json")


def normalize_circo(root: Path) -> None:
    base = root / "CIRCO"
    original = base / "_original"
    if not original.exists():
        print(f"  [warn] CIRCO original/ not found; skip normalize.")
        return

    # 1. Set up images/ as a SYMLINK to COCO unlabeled2017
    coco_imgs = original / "COCO2017_unlabeled" / "unlabeled2017"
    images = base / "images"
    if coco_imgs.exists() and not images.exists():
        try:
            images.symlink_to(coco_imgs.resolve(), target_is_directory=True)
            print(f"  symlinked images/ -> {coco_imgs}")
        except OSError as e:
            print(f"  [warn] symlink failed ({e}); "
                  f"point your code at {coco_imgs} directly.")
    elif not coco_imgs.exists():
        print(f"  [warn] {coco_imgs} not found "
              f"(was --skip-coco-images used?). images/ left absent.")

    # 2. queries.json
    queries = []
    for split_name in ["val", "test"]:
        ann_path = original / "annotations" / f"{split_name}.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            anns = json.load(f)
        for ann in anns:
            ref_id = ann.get("reference_img_id")
            tgt_id = ann.get("target_img_id")
            gt_ids = ann.get("gt_img_ids", [])
            caption = ann.get("relative_caption", "")
            queries.append({
                "id": f"circo_{split_name}_{ann.get('id')}",
                "split": split_name,
                "subset": None,
                "reference": _coco_filename(ref_id),
                "captions": [caption],
                "target": _coco_filename(tgt_id) if tgt_id else None,
                "gt_set": [_coco_filename(g) for g in gt_ids],
                "negatives": None,
                "extra": {
                    "shared_concept": ann.get("shared_concept"),
                    "semantic_aspects": ann.get("semantic_aspects"),
                    "original_id": ann.get("id"),
                },
            })
    write_json(queries, base / "queries.json")
    print(f"  wrote {len(queries)} queries -> {base/'queries.json'}")

    # 3. image_index.json
    image_index = []
    if coco_imgs.exists():
        image_index = sorted(p.name for p in coco_imgs.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg"))
    write_json(image_index, base / "image_index.json")
    print(f"  wrote {len(image_index)} image entries -> image_index.json")


def _coco_filename(image_id) -> str:
    """COCO image filename convention: 000000{id:06d}.jpg (12-digit pad)."""
    if image_id is None:
        return ""
    return f"{int(image_id):012d}.jpg"


# =====================================================================
#  FashionIQ
# =====================================================================
FIQ_RAR_URL = (
    "https://huggingface.co/datasets/Plachta/FashionIQ/"
    "resolve/main/fashionIQ_dataset.rar?download=true")


def download_fashioniq(root: Path) -> None:
    print("\n" + "=" * 70)
    print(" FashionIQ  (via Plachta/FashionIQ HuggingFace mirror)")
    print("=" * 70)
    base = root / "FashionIQ"
    original = base / "_original"
    original.mkdir(parents=True, exist_ok=True)

    rar_path = original / "fashionIQ_dataset.rar"
    try:
        http_download(FIQ_RAR_URL, rar_path, expected_min_size=900_000_000)
    except Exception as e:
        print(f"\n  [!] HuggingFace download failed: {e}")
        print("  Alternatives:")
        print("    1. Use the huggingface-cli (handles rate-limits better):")
        print("         pip install huggingface_hub")
        print("         huggingface-cli download Plachta/FashionIQ \\")
        print("             fashionIQ_dataset.rar --repo-type dataset \\")
        print(f"             --local-dir {original}")
        print("    2. QuRe's Google Drive mirror (FashionIQ + CIRR bundled):")
        print("         https://drive.google.com/drive/folders/"
              "17WgyAkiTb21y8BdTNJptyfEm4wjk4FPx")
        print(f"  Place fashionIQ_dataset.rar into {original} and re-run.")
        raise

    extract_marker = original / "_extracted_ok"
    if extract_marker.exists():
        print(f"  [skip] already extracted")
    else:
        unrar_to(rar_path, original)
        extract_marker.touch()


def normalize_fashioniq(root: Path) -> None:
    base = root / "FashionIQ"
    original = base / "_original"
    if not original.exists():
        print(f"  [warn] FashionIQ _original/ missing; skip normalize.")
        return

    captions_dir = find_dir(original, "captions")
    splits_dir   = find_dir(original, "image_splits")
    if captions_dir is None or splits_dir is None:
        print(f"  [warn] couldn't find captions/ or image_splits/ inside "
              f"{original}; archive layout may be unusual.")
        return

    images_root = find_dir(original, "images")
    per_category_dirs = {
        cat: find_dir(original, cat) for cat in ("dress", "shirt", "toptee")
    }

    # queries.json
    queries = []
    for cat in ("dress", "shirt", "toptee"):
        for split in ("train", "val", "test"):
            cap_file = captions_dir / f"cap.{cat}.{split}.json"
            if not cap_file.exists():
                continue
            with open(cap_file) as f:
                items = json.load(f)
            for i, item in enumerate(items):
                candidate = item.get("candidate", "")
                target    = item.get("target", "")
                caps      = item.get("captions", [])
                queries.append({
                    "id": f"fiq_{cat}_{split}_{i}",
                    "split": split,
                    "subset": cat,
                    "reference": f"{candidate}.jpg",
                    "captions": list(caps),
                    "target": f"{target}.jpg" if target else None,
                    "gt_set": [f"{target}.jpg"] if target else [],
                    "negatives": None,
                    "extra": {
                        "candidate_asin": candidate,
                        "target_asin": target,
                    },
                })
    write_json(queries, base / "queries.json")
    print(f"  wrote {len(queries)} queries -> {base/'queries.json'}")

    # images/  (symlink to upstream layout for zero-copy)
    images = base / "images"
    if not images.exists():
        if images_root is not None and images_root != base / "images":
            try:
                images.symlink_to(images_root.resolve(),
                                  target_is_directory=True)
                print(f"  symlinked images/ -> {images_root}")
            except OSError as e:
                print(f"  [warn] symlink failed ({e}); copying instead.")
                shutil.copytree(images_root, images)
        elif any(per_category_dirs.values()):
            images.mkdir(parents=True, exist_ok=True)
            count = 0
            for cat, d in per_category_dirs.items():
                if d is None:
                    continue
                for jpg in d.iterdir():
                    if jpg.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        continue
                    target_link = images / jpg.name
                    if target_link.exists():
                        continue
                    try:
                        target_link.symlink_to(jpg.resolve())
                    except OSError:
                        shutil.copy2(jpg, target_link)
                    count += 1
            print(f"  merged {count} per-category images into images/")
        else:
            print(f"  [warn] could not locate images inside {original}; "
                  f"populate {images} manually.")

    # image_index.json
    image_index = []
    if images.exists():
        image_index = sorted(
            p.name for p in images.iterdir()
            if (p.is_file() or p.is_symlink())
            and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    declared = set()
    for cat in ("dress", "shirt", "toptee"):
        for split in ("train", "val", "test"):
            sp = splits_dir / f"split.{cat}.{split}.json"
            if sp.exists():
                with open(sp) as f:
                    declared.update(f"{asin}.jpg" for asin in json.load(f))
    if declared and not image_index:
        image_index = sorted(declared)
        print(f"  [note] image files not findable; index uses declared ASINs.")
    write_json(image_index, base / "image_index.json")
    print(f"  wrote {len(image_index)} image entries -> image_index.json")


# =====================================================================
#  HP-FashionIQ
# =====================================================================
def download_hpfiq(root: Path) -> None:
    print("\n" + "=" * 70)
    print(" HP-FashionIQ  (ICML 2025)")
    print("=" * 70)
    base = root / "HP_FashionIQ"
    original = base / "_original"
    if (original / "hpfiq.json").exists():
        print(f"  [skip] HP-FashionIQ already present")
        return

    tmp_clone = root / "_tmp_qure_clone"
    git_clone("https://github.com/jackwaky/QuRe.git", tmp_clone)
    src = tmp_clone / "HP_FashionIQ"
    if not src.exists():
        raise RuntimeError(
            "HP_FashionIQ folder not found in QuRe clone; layout changed?")
    original.mkdir(parents=True, exist_ok=True)
    move_contents(src, original)
    shutil.rmtree(tmp_clone, ignore_errors=True)


# regex for paths like "/-/image_data/shirt/B000LUIBH8.jpg"
_HPFIQ_PATH_RE = re.compile(
    r"(?:^|/)image_data/(dress|shirt|toptee)/([^/]+\.jpg)$")


def _hpfiq_path_to_filename(path: str) -> tuple[str | None, str | None]:
    m = _HPFIQ_PATH_RE.search(path)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def normalize_hpfiq(root: Path) -> None:
    base = root / "HP_FashionIQ"
    original = base / "_original"
    hpfiq_json = original / "hpfiq.json"
    if not hpfiq_json.exists():
        print(f"  [warn] {hpfiq_json} missing; skip normalize.")
        return

    with open(hpfiq_json) as f:
        data = json.load(f)

    queries = []
    seen_files: set[str] = set()
    for user_key, qsets in data.items():
        for qset_key, qset in qsets.items():
            refs       = qset.get("ref_img_paths", [])
            targs      = qset.get("targ_img_paths", [])
            sentences  = qset.get("sentences", [])
            preferred  = qset.get("preferred set", [])
            rset1      = qset.get("retrieved_set1", [])
            rset2      = qset.get("retrieved_set2", [])

            n = min(len(refs), len(targs), len(sentences), len(preferred),
                    len(rset1), len(rset2))
            for i in range(n):
                ref_cat, ref_name = _hpfiq_path_to_filename(refs[i])
                tgt_cat, tgt_name = _hpfiq_path_to_filename(targs[i])
                if ref_name: seen_files.add(ref_name)
                if tgt_name: seen_files.add(tgt_name)

                set1_imgs = [_hpfiq_path_to_filename(p)[1]
                             for p in rset1[i].get("img_path", [])]
                set2_imgs = [_hpfiq_path_to_filename(p)[1]
                             for p in rset2[i].get("img_path", [])]
                set1_imgs = [f for f in set1_imgs if f]
                set2_imgs = [f for f in set2_imgs if f]
                seen_files.update(set1_imgs); seen_files.update(set2_imgs)

                queries.append({
                    "id": f"hpfiq_{user_key}_{qset_key}_q{i}",
                    "split": "val",
                    "subset": ref_cat or tgt_cat,
                    "reference": ref_name or "",
                    "captions": [sentences[i]],
                    "target": tgt_name,
                    "gt_set": [tgt_name] if tgt_name else [],
                    "negatives": None,
                    "extra": {
                        "task": "set_preference",
                        "retrieved_set1": set1_imgs,
                        "retrieved_set2": set2_imgs,
                        "user_score_set1": _safe_int(
                            rset1[i].get("user_score")),
                        "user_score_set2": _safe_int(
                            rset2[i].get("user_score")),
                        "preferred_set": _safe_int(preferred[i]),
                        "user": user_key,
                        "question_set": qset_key,
                    },
                })
    write_json(queries, base / "queries.json")
    print(f"  wrote {len(queries)} queries -> {base/'queries.json'}")

    # images/ symlink to FashionIQ/images
    images = base / "images"
    fiq_images = root / "FashionIQ" / "images"
    if not images.exists():
        if fiq_images.exists():
            try:
                images.symlink_to(fiq_images.resolve(),
                                  target_is_directory=True)
                print(f"  symlinked images/ -> {fiq_images} "
                      f"(shares FashionIQ images, saves ~1.2GB)")
            except OSError as e:
                print(f"  [warn] symlink failed ({e}); "
                      f"point your code at {fiq_images} directly.")
        else:
            print(f"  [warn] {fiq_images} not present yet. "
                  f"Run --only fashioniq first.")

    image_index = sorted(seen_files)
    write_json(image_index, base / "image_index.json")
    print(f"  wrote {len(image_index)} image entries -> image_index.json")


def _safe_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


# =====================================================================
#  PinPoint  (only when explicitly requested)
# =====================================================================
def download_pinpoint(root: Path) -> None:
    print("\n" + "=" * 70)
    print(" PinPoint  (CVPR 2026)")
    print("=" * 70)
    base = root / "PinPoint"
    original = base / "_original"

    if (original / "pinpoint_licensed.parquet").exists():
        print(f"  [skip] PinPoint parquet already present")
        return

    tmp_clone = root / "_tmp_pinpoint_clone"
    git_clone("https://github.com/pinterest/pinpoint-dataset.git", tmp_clone)
    original.mkdir(parents=True, exist_ok=True)
    move_contents(tmp_clone, original)
    shutil.rmtree(tmp_clone, ignore_errors=True)


def normalize_pinpoint(root: Path) -> None:
    base = root / "PinPoint"
    original = base / "_original"
    parquet = original / "pinpoint_licensed.parquet"
    if not parquet.exists():
        print(f"  [warn] {parquet} missing; skip normalize.")
        return
    try:
        import pandas as pd                                  # type: ignore
    except ImportError:
        print(f"  [warn] PinPoint normalization needs pandas + pyarrow.")
        print(f"        Install: pip install pandas pyarrow")
        return

    df = pd.read_parquet(parquet)
    queries = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        ref1 = row.get("query_image_signature")
        ref2 = row.get("query_image_signature2")
        # PinPoint parquet stores "missing second image" as the LITERAL STRING
        # "None" (not a real null), so the obvious check is wrong. Reject both
        # truly-null and the string sentinel.
        if not isinstance(ref2, str) or not ref2 or ref2.strip().lower() in (
                "none", "nan", "null", ""):
            ref2 = None

        # parquet list columns come back as numpy arrays; `or []` is ambiguous.
        pos_raw = row.get("positive_candidates")
        neg_raw = row.get("negative_candidates")
        pos = list(pos_raw) if pos_raw is not None else []
        neg = list(neg_raw) if neg_raw is not None else []

        def sig_to_fname(s):
            return f"{s}.jpg" if s else None

        ref_fnames = [sig_to_fname(ref1)]
        if ref2:
            ref_fnames.append(sig_to_fname(ref2))
        ref_fnames = [r for r in ref_fnames if r]
        for r in ref_fnames: seen.add(r)
        for p in pos:
            fp = sig_to_fname(p)
            if fp: seen.add(fp)
        for n in neg:
            fn = sig_to_fname(n)
            if fn: seen.add(fn)

        queries.append({
            "id": str(row["query_id"]),
            "split": "test",
            "subset": None,
            "reference": ref_fnames if len(ref_fnames) > 1 else (
                ref_fnames[0] if ref_fnames else ""),
            "captions": [row.get("instruction", "")],
            "target": sig_to_fname(pos[0]) if pos else None,
            "gt_set": [sig_to_fname(p) for p in pos if p],
            "negatives": [sig_to_fname(n) for n in neg if n],
            "extra": {
                "note": "images live on Pinterest CDN; filename here = "
                        "signature + '.jpg'. URL pattern: i.pinimg.com/736x/"
                        "<sig[0:2]>/<sig[2:4]>/<sig[4:6]>/<sig>.jpg",
            },
        })
    write_json(queries, base / "queries.json")
    print(f"  wrote {len(queries)} queries -> {base/'queries.json'}")

    idx_txt = original / "index_signatures.txt"
    image_index = list(seen)
    if idx_txt.exists():
        with open(idx_txt) as f:
            for line in f:
                sig = line.strip()
                if sig:
                    image_index.append(f"{sig}.jpg")
    image_index = sorted(set(f for f in image_index if f))
    write_json(image_index, base / "image_index.json")
    print(f"  wrote {len(image_index)} image entries -> image_index.json")

    (base / "images").mkdir(parents=True, exist_ok=True)
    print(f"  Note: images/ is empty. Images live on Pinterest CDN.")


# =====================================================================
#  Pipeline dispatch
# =====================================================================
DATASETS = {
    # key:        (display, download_fn,         normalize_fn,        default?)
    "circo":      ("CIRCO",        download_circo,     normalize_circo,    True),
    "fashioniq":  ("FashionIQ",    download_fashioniq, normalize_fashioniq, True),
    "hpfiq":      ("HP_FashionIQ", download_hpfiq,     normalize_hpfiq,    True),
    "pinpoint":   ("PinPoint",     download_pinpoint,  normalize_pinpoint, False),
}

# Order matters: FashionIQ before HP_FashionIQ so the symlink lands.
DEFAULT_ORDER = ["circo", "fashioniq", "hpfiq", "pinpoint"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root", type=Path, default=Path("./advanced_datasets"),
        help="root dir (default: ./advanced_datasets)")
    ap.add_argument(
        "--only", choices=list(DATASETS), default=None,
        help="download/normalize only this one dataset")
    ap.add_argument(
        "--skip-coco-images", action="store_true",
        help="for CIRCO: skip the ~19GB COCO images")
    ap.add_argument(
        "--no-normalize", action="store_true",
        help="only download, don't produce unified queries.json")
    args = ap.parse_args()

    if not have("git"):
        print("ERROR: `git` is required. Install git first.")
        sys.exit(1)

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {args.root.resolve()}")

    if args.only:
        targets = [args.only]
    else:
        targets = [k for k in DEFAULT_ORDER if DATASETS[k][3]]
        skipped = [k for k in DEFAULT_ORDER if not DATASETS[k][3]]
        if skipped:
            print(f"(not included by default: {', '.join(skipped)}; "
                  f"use --only to fetch)")

    failures = []
    for name in targets:
        display, dl_fn, norm_fn, _ = DATASETS[name]
        try:
            if name == "circo":
                dl_fn(args.root, skip_coco_images=args.skip_coco_images)
            else:
                dl_fn(args.root)
            if not args.no_normalize:
                print(f"\n  -- normalizing {display} --")
                norm_fn(args.root)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(130)
        except Exception as e:
            print(f"\n[!] FAILED for {name}: {e}")
            import traceback; traceback.print_exc()
            failures.append((name, str(e)))

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  Output: {args.root.resolve()}")
    for name in targets:
        ok = not any(n == name for n, _ in failures)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}  ({DATASETS[name][0]})")
    if failures:
        sys.exit(1)
    print("\nAll done. Each dataset directory contains:")
    print("  images/           image files (or symlink)")
    print("  queries.json      unified query list")
    print("  image_index.json  full image filename list")
    print("  _original/        untouched raw files")


if __name__ == "__main__":
    main()