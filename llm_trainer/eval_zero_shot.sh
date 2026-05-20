#!/usr/bin/env bash
# eval_zero_shot.sh — 零样本检索 baseline
#
# 用法:
#   bash eval_zero_shot.sh                          # 默认
#   GPUS=5 bash eval_zero_shot.sh                   # 换卡
#   SAMPLE_RATIO=0.5 bash eval_zero_shot.sh         # 改抽样比例 (默认 0.2)
#   N_QUERIES=500 bash eval_zero_shot.sh            # 多抽 query
#   bash eval_zero_shot.sh A-OKVQA ChartQA          # 指定数据集
#   bash eval_zero_shot.sh --ckpt outputs/exp/adapter_best  # 评测训练后

set -u
GPUS="${GPUS:-4}"
N_QUERIES="${N_QUERIES:-200}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/zero_shot_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-1}"   # 默认开启 resume; 设 RESUME=0 强制从头跑

CKPT=""
DATASETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)     CKPT="$2"; shift 2 ;;
        --no-resume) RESUME=0; shift ;;
        --resume)   RESUME=1; shift ;;
        *)          DATASETS+=("$1"); shift ;;
    esac
done
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(CIRR FashionIQ Visual7W A-OKVQA GQA OK-VQA TextVQA WebQA Wiki-SS-NQ)
fi

[ -f "eval_zero_shot.py" ] || { echo "请在 llm_trainer/ 目录运行"; exit 1; }
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/eval.log"

echo "[zero-shot] GPUs=$GPUS, datasets=${DATASETS[*]}"
echo "[zero-shot] num_queries=$N_QUERIES, sample_ratio=$SAMPLE_RATIO"
[ -n "$CKPT" ] && echo "[zero-shot] ckpt: $CKPT"
echo "[zero-shot] output: $OUTPUT_DIR"

EXTRA=()
[ -n "$CKPT" ] && EXTRA=(--ckpt "$CKPT")
[ "$RESUME" = "1" ] && EXTRA+=(--resume)

CUDA_VISIBLE_DEVICES="$GPUS" python eval_zero_shot.py \
    --config configs/exp_lora_sae.yaml \
    --datasets "${DATASETS[@]}" \
    --num_queries "$N_QUERIES" \
    --sample_ratio "$SAMPLE_RATIO" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA[@]}" 2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
if [ $RC -eq 0 ]; then
    echo "[zero-shot] done: $OUTPUT_DIR/zero_shot.png"
fi
exit $RC