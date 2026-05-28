#!/usr/bin/env bash
# eval_zero_shot.sh — 零样本检索 baseline (多卡数据集分片版)
#
# 多卡策略: 不切 forward, 而是把数据集列表平均分给每张卡,
#           每张卡起一个独立进程跑自己那份数据集, 零跨卡通信。
#           各进程写各自的子 JSON, 最后合并 + 画图。
#
# 用法:
#   bash eval_zero_shot.sh                          # 用 GPUS 指定的所有卡
#   GPUS=0,1,2,3 bash eval_zero_shot.sh             # 4 卡
#   GPUS=0 bash eval_zero_shot.sh                   # 单卡
#   SAMPLE_RATIO=0.5 N_QUERIES=500 bash eval_zero_shot.sh
#   bash eval_zero_shot.sh A-OKVQA ChartQA          # 指定数据集
#   bash eval_zero_shot.sh --ckpt outputs/exp/adapter_best

set -u
GPUS="${GPUS:-0,1}"                      # 逗号分隔的卡号; 卡数决定并行进程数
N_QUERIES="${N_QUERIES:-200}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-48}"         # 128 核 / 卡数 左右; 单卡进程内的 DataLoader worker
OUTPUT_DIR="${OUTPUT_DIR:-outputs/zero_shot_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-1}"
NO_SAE="${NO_SAE:-0}"

CKPT=""
DATASETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)      CKPT="$2"; shift 2 ;;
        --no-resume) RESUME=0; shift ;;
        --resume)    RESUME=1; shift ;;
        --no-sae)    NO_SAE=1; shift ;;
        *)           DATASETS+=("$1"); shift ;;
    esac
done
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(CIRR FashionIQ Visual7W A-OKVQA GQA OK-VQA TextVQA WebQA Wiki-SS-NQ)
fi

[ -f "eval_zero_shot.py" ] || { echo "请在 llm_trainer/ 目录运行"; exit 1; }

# 缓解显存碎片: expandable_segments 让 caching allocator 更灵活归还,
# 避免 encode 阶段 reserved 占满后 build CSR 分配不出来
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 解析 GPU 列表
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
NDS=${#DATASETS[@]}

mkdir -p "$OUTPUT_DIR"

echo "[zero-shot] GPUs=$GPUS ($NGPU 张), datasets=$NDS 个: ${DATASETS[*]}"
echo "[zero-shot] num_queries=$N_QUERIES, sample_ratio=$SAMPLE_RATIO, batch_size=$BATCH_SIZE"
echo "[zero-shot] num_workers/卡=$NUM_WORKERS, no_sae=$NO_SAE"
[ -n "$CKPT" ] && echo "[zero-shot] ckpt: $CKPT"
echo "[zero-shot] output: $OUTPUT_DIR"

# ---- 把数据集 round-robin 分配到每张卡 ----
# 卡 g 拿到的数据集: 索引 g, g+NGPU, g+2*NGPU, ...
declare -a SHARD   # SHARD[g] = 该卡的数据集列表(空格分隔)
for ((g=0; g<NGPU; g++)); do SHARD[$g]=""; done
for ((i=0; i<NDS; i++)); do
    g=$(( i % NGPU ))
    SHARD[$g]="${SHARD[$g]} ${DATASETS[$i]}"
done

EXTRA_BASE=()
[ -n "$CKPT" ] && EXTRA_BASE+=(--ckpt "$CKPT")
[ "$RESUME" = "1" ] && EXTRA_BASE+=(--resume)
[ "$NO_SAE" = "1" ] && EXTRA_BASE+=(--no_sae)

# ---- 每张卡起一个后台进程 ----
PIDS=()
for ((g=0; g<NGPU; g++)); do
    gpu_id="${GPU_ARR[$g]}"
    shard_datasets="${SHARD[$g]}"
    if [ -z "$shard_datasets" ]; then
        echo "[gpu $gpu_id] 无数据集分配, 跳过"
        continue
    fi
    # 每张卡写自己的子目录, 避免 JSON 互相覆盖
    sub_out="$OUTPUT_DIR/shard_gpu${gpu_id}"
    log="$OUTPUT_DIR/shard_gpu${gpu_id}.log"
    echo "[gpu $gpu_id] 分到:$shard_datasets  -> $sub_out"

    if [ "$NGPU" -eq 1 ]; then
        # 单卡: 直接输出到终端 + 日志 (实时可见, 无交错问题)
        CUDA_VISIBLE_DEVICES="$gpu_id" python eval_zero_shot.py \
            --config configs/exp_lora_sae.yaml \
            --datasets $shard_datasets \
            --num_queries "$N_QUERIES" \
            --sample_ratio "$SAMPLE_RATIO" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --output_dir "$sub_out" \
            "${EXTRA_BASE[@]}" 2>&1 | tee "$log" &
        PIDS+=($!)
    else
        # 多卡: 每行加 [gpuN] 前缀, tee 到终端 + 日志
        # (用 stdbuf 关掉缓冲, 保证前缀实时刷新)
        CUDA_VISIBLE_DEVICES="$gpu_id" stdbuf -oL -eL python eval_zero_shot.py \
            --config configs/exp_lora_sae.yaml \
            --datasets $shard_datasets \
            --num_queries "$N_QUERIES" \
            --sample_ratio "$SAMPLE_RATIO" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --output_dir "$sub_out" \
            "${EXTRA_BASE[@]}" 2>&1 \
            | stdbuf -oL sed "s/^/[gpu${gpu_id}] /" \
            | tee "$log" &
        PIDS+=($!)
    fi
done

echo "[zero-shot] 已启动 ${#PIDS[@]} 个进程, PIDs: ${PIDS[*]}"
echo "[zero-shot] 实时查看某卡日志: tail -f $OUTPUT_DIR/shard_gpu<id>.log"

# ---- 等待所有进程 ----
# 注意: 管道模式下 wait 等的是整条管道(到 tee)结束, 即 python 已跑完
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

# 用"子 JSON 是否生成"来判断各卡是否成功 (比退出码可靠)
FAIL=0
for ((g=0; g<NGPU; g++)); do
    gpu_id="${GPU_ARR[$g]}"
    [ -z "${SHARD[$g]}" ] && continue
    if [ ! -f "$OUTPUT_DIR/shard_gpu${gpu_id}/zero_shot_results.json" ]; then
        echo "[zero-shot] 警告: 卡 $gpu_id 没有产出 JSON, 检查 $OUTPUT_DIR/shard_gpu${gpu_id}.log"
        FAIL=1
    fi
done

# ---- 合并所有 shard 的 JSON + 画总图 ----
echo "[zero-shot] 所有进程结束, 合并结果 ..."
python - "$OUTPUT_DIR" << 'PYEOF'
import os, sys, json, glob
out_dir = sys.argv[1]
merged = {}
for jf in glob.glob(os.path.join(out_dir, "shard_gpu*", "zero_shot_results.json")):
    try:
        with open(jf) as f:
            d = json.load(f)
        merged.update(d)
        print(f"  merged {len(d)} datasets from {jf}")
    except Exception as e:
        print(f"  skip {jf}: {e}")

merged_json = os.path.join(out_dir, "zero_shot_results.json")
with open(merged_json, "w") as f:
    json.dump(merged, f, indent=2)
print(f"  -> {merged_json} ({len(merged)} datasets total)")

# 画总图: 复用 eval_zero_shot.py 的 plot_results
try:
    sys.path.insert(0, os.getcwd())
    from eval_zero_shot import plot_results
    plot_results(merged, os.path.join(out_dir, "zero_shot.png"))
except Exception as e:
    print(f"  plot failed: {e}")
PYEOF

if [ $FAIL -eq 0 ]; then
    echo "[zero-shot] done: $OUTPUT_DIR/zero_shot.png"
fi
exit $FAIL