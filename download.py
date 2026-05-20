"""
Download required model weights and datasets from Hugging Face Hub.

Targets:
    pretrained_models/llama3-llava-next-8b-hf/                     (LVLM)
    sae_trainer/finetune_models/llama3-llava-next-8b-hf-sae-131k/  (SAE)
    data/MMEB-eval/                                                (dataset)
        ├── <task>/test/*.parquet
        ├── images.zip
        └── images/  ← extracted from images.zip

Usage:
    python download.py                       # download everything
    python download.py --only base           # one item only
    python download.py --only sae
    python download.py --only mmeb
    python download.py --mirror              # hf-mirror.com (China)
    python download.py --token <HF_TOKEN>    # for gated models
    python download.py --delete-zip          # delete zip after extraction
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


ITEMS = [
    {
        "key": "base",
        "repo_id": "llava-hf/llama3-llava-next-8b-hf",
        "repo_type": "model",
        "local_dir": "pretrained_models/llama3-llava-next-8b-hf",
        "description": "LLaVA-NeXT Llama3 8B base model (~16 GB)",
    },
    {
        "key": "sae",
        "repo_id": "lmms-lab/llama3-llava-next-8b-hf-sae-131k",
        "repo_type": "model",
        "local_dir": "sae_trainer/finetune_models/llama3-llava-next-8b-hf-sae-131k",
        "description": "Trained SAE on LLaVA-NeXT (131k features)",
    },
    {
        "key": "mmeb",
        "repo_id": "TIGER-Lab/MMEB-eval",
        "repo_type": "dataset",
        "local_dir": "MMEB-eval",
        "post_extract": {
            "zip": "images.zip",
            "to": "images",  # extract to data/MMEB-eval/images/
        },
        "description": "MMEB evaluation dataset (parquet + images.zip)",
    },
]


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def dir_size(path):
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download_repo(repo_id, repo_type, local_dir, token):
    from huggingface_hub import snapshot_download

    local_path = Path(local_dir).resolve()
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Repo:        {repo_id}  ({repo_type})")
    print(f"Destination: {local_path}")
    print(f"{'=' * 70}")

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_path),
        token=token,
        max_workers=4,
    )

    print(f"  → {human_size(dir_size(local_path))} on disk")


def extract_zip(local_dir, zip_name, extract_to_name, delete_zip=False):
    local_path = Path(local_dir).resolve()
    zip_path = local_path / zip_name
    extract_path = local_path / extract_to_name

    if not zip_path.exists():
        print(f"  ⚠ {zip_name} not found in {local_path}, skipping extraction")
        return

    # Skip if already extracted (folder exists and non-empty)
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"  ⓘ {extract_path} already exists and non-empty, skipping extraction")
        return

    print(f"  → extracting {zip_name} → {extract_path}/")
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, name in enumerate(members, start=1):
            zf.extract(name, extract_path)
            if i % 1000 == 0 or i == total:
                pct = 100.0 * i / total
                print(f"    [{i}/{total}] {pct:.1f}% extracted")

    print(f"  ✓ extracted to {extract_path}")
    print(f"    extracted size: {human_size(dir_size(extract_path))}")

    if delete_zip:
        zip_path.unlink()
        print(f"  → removed {zip_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    keys = [m["key"] for m in ITEMS]
    parser.add_argument("--only", choices=keys, default=None,
                        help="Only process one of: " + ", ".join(keys))
    parser.add_argument("--mirror", action="store_true",
                        help="Use https://hf-mirror.com (recommended in China)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face access token (or set HF_TOKEN env)")
    parser.add_argument("--delete-zip", action="store_true",
                        help="Delete the zip file after extraction (saves disk)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Download only, don't extract any zips")
    args = parser.parse_args()

    # Anchor all paths to where this script lives
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f">>> Working directory: {script_dir}")

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(">>> Using mirror: https://hf-mirror.com")
    else:
        print(f">>> HF endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")

    if args.token:
        print(">>> Using HF token")
    else:
        print(">>> No HF token (only needed for gated models)")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    selected = ITEMS if args.only is None else [m for m in ITEMS if m["key"] == args.only]

    for item in selected:
        print(f"\n>>> {item['description']}")
        try:
            download_repo(
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                local_dir=item["local_dir"],
                token=args.token,
            )
        except Exception as e:
            print(f"  ✗ FAILED: {item['repo_id']}\n     {type(e).__name__}: {e}")
            print("     Tip: re-run the same command; downloads resume automatically.")
            return 1

        if not args.skip_extract and "post_extract" in item:
            try:
                extract_zip(
                    local_dir=item["local_dir"],
                    zip_name=item["post_extract"]["zip"],
                    extract_to_name=item["post_extract"]["to"],
                    delete_zip=args.delete_zip,
                )
            except Exception as e:
                print(f"  ✗ EXTRACTION FAILED: {type(e).__name__}: {e}")
                return 1

    print("\n✓ All tasks finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())