#!/usr/bin/env bash
set -euo pipefail

# Run LLaVA-NeXT layer-24 activation collection.
#
# Expected files:
#   llava_next_activation_collector.py
#
# Typical run:
#   bash run_llava_next_activation_collector.sh
#
# Debug run:
#   MAX_SAMPLES=8 bash run_llava_next_activation_collector.sh
#
# Multiple datasets:
#   DATASETS="ChartQA DocVQA InfoVQA" bash run_llava_next_activation_collector.sh
#
# Different GPU:
#   GPU_ID=1 bash run_llava_next_activation_collector.sh

# -----------------------------
# GPU / runtime
# -----------------------------
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Optional Hugging Face cache paths.
# Uncomment and edit if needed.
# export HF_HOME="/path/to/hf_cache"
# export TRANSFORMERS_CACHE="/path/to/hf_cache/transformers"

# -----------------------------
# Paths
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COLLECTOR="${COLLECTOR:-${SCRIPT_DIR}/llava_next_activation_collector.py}"

# If this script is placed in:
#   VL-SAE-main/cvlms/sort/sae_trainer/
# then ../../.. should be VL-SAE-main/.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/MMEB-eval}"
IMAGES_ROOT="${IMAGES_ROOT:-${DATA_ROOT}/images}"
SAVE_ROOT="${SAVE_ROOT:-${SCRIPT_DIR}/llava_next_activations}"

# -----------------------------
# Model / activation config
# -----------------------------
MODEL_ID="${MODEL_ID:-llava-hf/llama3-llava-next-8b-hf}"
LAYER_IDX="${LAYER_IDX:-24}"

# Strongly recommended for LLaVA-NeXT activation collection.
# This follows the simple batch-size-1 style used by many SAE activation pipelines.
BATCH_SIZE="${BATCH_SIZE:-1}"

MODEL_DTYPE="${MODEL_DTYPE:-float16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"

# Optional. Use a small number first to validate the pipeline.
# Example:
#   MAX_SAMPLES=8 bash run_llava_next_activation_collector.sh
MAX_SAMPLES="${MAX_SAMPLES:-}"

# Pad sequence length S to a multiple. Keep 1 unless you need tensor-core-friendly shapes.
PAD_TO_MULTIPLE="${PAD_TO_MULTIPLE:-1}"

EMPTY_TEXT_PROMPT="${EMPTY_TEXT_PROMPT:-Describe the image.}"

# -----------------------------
# Dataset config
# -----------------------------
# Option A: one dataset
DATASET="${DATASET:-}"

# Option B: multiple datasets, space-separated
# Example:
#   DATASETS="ChartQA DocVQA InfoVQA"
DATASETS="${DATASETS:-}"

# If both DATASET and DATASETS are empty, the collector scans all dataset dirs under DATA_ROOT.

# -----------------------------
# Build command
# -----------------------------
cmd=(
  python "${COLLECTOR}"
  --data_root "${DATA_ROOT}"
  --images_root "${IMAGES_ROOT}"
  --save_root "${SAVE_ROOT}"
  --model_id "${MODEL_ID}"
  --layer_idx "${LAYER_IDX}"
  --batch_size "${BATCH_SIZE}"
  --model_dtype "${MODEL_DTYPE}"
  --save_dtype "${SAVE_DTYPE}"
  --device "cuda:0"
  --pad_to_multiple "${PAD_TO_MULTIPLE}"
  --empty_text_prompt "${EMPTY_TEXT_PROMPT}"
)

if [[ -n "${DATASET}" ]]; then
  cmd+=(--dataset "${DATASET}")
fi

if [[ -n "${DATASETS}" ]]; then
  # shellcheck disable=SC2206
  dataset_array=(${DATASETS})
  cmd+=(--datasets "${dataset_array[@]}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max_samples "${MAX_SAMPLES}")
fi

# -----------------------------
# Print and run
# -----------------------------
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] IMAGES_ROOT=${IMAGES_ROOT}"
echo "[INFO] SAVE_ROOT=${SAVE_ROOT}"
echo "[INFO] MODEL_ID=${MODEL_ID}"
echo "[INFO] LAYER_IDX=${LAYER_IDX}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] MODEL_DTYPE=${MODEL_DTYPE}"
echo "[INFO] SAVE_DTYPE=${SAVE_DTYPE}"
echo "[INFO] MAX_SAMPLES=${MAX_SAMPLES:-<none>}"
echo "[INFO] Command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
