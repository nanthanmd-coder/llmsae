#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/finetune.py" \
  --sae-type matryoshka_vlsae \
  --hidden-ratio 8 \
  --ckpt-path "${SCRIPT_DIR}/pre_sae_weights/openclip_ViT-B-32_Matryoshka_VL_SAE_256_8_best.pth" \
  --device cuda:2 \
  --loss-mode matryoshka_info_nce