#!/usr/bin/env bash
# eval_ckpts.sh — 依次评估多个 checkpoint, 每个存到独立目录, 最后可对比
#
# 用法:
#   bash eval_ckpts.sh                      # 默认评估 step100 + step300
#   CKPTS="adapter_step100 adapter_step300 adapter_last" bash eval_ckpts.sh
#   GPUS=0,1 N_QUERIES=200 bash eval_ckpts.sh
#
# 每个 ckpt 的结果存到 outputs/eval_<ckpt名>/, 互不覆盖。

set -u

# 训练默认输出目录 (checkpoint 所在地)
TRAIN_DIR="${TRAIN_DIR:-outputs/exp_mmeb_lora_sae}"
# 要评估的 checkpoint 名 (空格分隔), 默认 step100 + step300
CKPTS="${CKPTS:-adapter_step450}"

GPUS="${GPUS:-0,1}"
N_QUERIES="${N_QUERIES:-200}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.2}"

[ -f "eval_zero_shot.sh" ] || { echo "请在 llm_trainer/ 目录运行"; exit 1; }

echo "[eval-ckpts] 训练目录: $TRAIN_DIR"
echo "[eval-ckpts] 要评估的 checkpoint: $CKPTS"
echo "[eval-ckpts] GPUs=$GPUS, N_QUERIES=$N_QUERIES"
echo ""


for ckpt_name in $CKPTS; do
    ckpt_path="$TRAIN_DIR/$ckpt_name"
    if [ ! -d "$ckpt_path" ]; then
        echo "[eval-ckpts] 跳过 $ckpt_name: 目录不存在 ($ckpt_path)"
        continue
    fi
    out_dir="outputs/eval_${ckpt_name}"
    echo "=========================================="
    echo "[eval-ckpts] 评估 $ckpt_name -> $out_dir"
    echo "=========================================="
    OUTPUT_DIR="$out_dir" GPUS="$GPUS" N_QUERIES="$N_QUERIES" \
        SAMPLE_RATIO="$SAMPLE_RATIO" \
        bash eval_zero_shot.sh --ckpt "$ckpt_path"
    echo ""
done

echo "[eval-ckpts] 全部完成。结果目录:"
for ckpt_name in $CKPTS; do
    out_dir="outputs/eval_${ckpt_name}"
    [ -f "$out_dir/zero_shot_results.json" ] && echo "  $out_dir/zero_shot_results.json"
done

# zero-shot 基线 (无 adapter)
echo "=========================================="
echo "[eval-ckpts] 评估 zero-shot 基线 (无 adapter)"
echo "=========================================="
GPUS="$GPUS" N_QUERIES="$N_QUERIES" SAMPLE_RATIO="$SAMPLE_RATIO" \
    bash eval_zero_shot.sh
 
echo ""
echo "[eval-ckpts] 全部完成。"