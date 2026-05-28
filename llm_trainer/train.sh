#!/usr/bin/env bash
# =============================================================================
# train.sh — llm_trainer 训练入口 (单卡 / 多卡 DDP 自动选择)
#
# 用法:
#   bash train.sh                          # 默认 GPU 4,5 多卡 DDP
#   GPUS=4 bash train.sh                   # 只用 GPU 4 (单卡)
#   GPUS=4,5,6,7 bash train.sh             # 4 卡 DDP
#   bash train.sh --debug                  # 调试模式 (A-OKVQA 单数据集)
#   bash train.sh --resume <ckpt_path>     # 续训
#   bash train.sh --eval-only <ckpt_path>  # 只评估
#   bash train.sh --config configs/xxx.yaml
#
# 自动判断: GPUS 里有几张卡, 1 张走单卡 python, >=2 张走 accelerate launch DDP
# =============================================================================

set -u

# 中文输出 + 显存碎片缓解 (报错信息直接建议 expandable_segments)
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# DDP 子进程完整 traceback (否则只显示 ChildFailedError 外壳, 看不到真错误)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json
export NCCL_DEBUG=WARN
# 调试时设 DEBUG_SYNC=1 开 CUDA_LAUNCH_BLOCKING (定位精确报错行, 但会拖慢训练)
[ "${DEBUG_SYNC:-0}" = "1" ] && export CUDA_LAUNCH_BLOCKING=1

# ----------------------------------------------------------------------------
# 默认参数
# ----------------------------------------------------------------------------
GPUS="${GPUS:-0,1}"
CONFIG="configs/exp_lora_sae.yaml"
MODE="train"
CKPT=""
DEBUG=0

# ----------------------------------------------------------------------------
# 解析命令行
# ----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)      CKPT="$2"; shift 2 ;;
        --eval-only)   MODE="eval"; CKPT="$2"; shift 2 ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --debug)       DEBUG=1; shift ;;
        -h|--help)     head -16 "$0" | sed 's/^# \?//'; exit 0 ;;
        *)             echo "未知参数: $1"; exit 1 ;;
    esac
done

# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[train]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; }

# ----------------------------------------------------------------------------
# 前置检查
# ----------------------------------------------------------------------------
[ -f "run.py" ] || { err "请在 llm_trainer/ 目录运行"; exit 1; }
[ -f "$CONFIG" ] || { err "配置文件不存在: $CONFIG"; exit 1; }

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    warn "当前 conda 环境是 '${CONDA_DEFAULT_ENV:-未设置}', 建议先 conda activate llmsae"
    read -p "继续? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

command -v nvidia-smi &> /dev/null || { err "nvidia-smi 不可用"; exit 1; }

# 算几张卡
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

# ----------------------------------------------------------------------------
# 调试模式
# ----------------------------------------------------------------------------
if [ "$DEBUG" = "1" ]; then
    log "调试模式: A-OKVQA 单数据集"
    DEBUG_CONFIG="configs/_debug.yaml"
    cat > "$DEBUG_CONFIG" <<'YAML'
defaults:
  - base
exp_name: debug_aokvqa
data:
  train_datasets: ["A-OKVQA"]
  eval_datasets: ["A-OKVQA"]
  max_candidates: 4
  max_length: 3500
  num_workers: 2
train:
  epochs: 1
  batch_size: 1
  grad_accum_steps: 8
  lr: 1.0e-4
  pool_layer_idx: 24
  info_nce_temp: 0.07
  log_steps: 5
  eval_steps: 100
  save_steps: 200
YAML
    CONFIG="$DEBUG_CONFIG"
fi

# ----------------------------------------------------------------------------
# 准备信息
# ----------------------------------------------------------------------------
echo ""
log "==================== 训练准备 ===================="
log "  config:       $CONFIG"
log "  mode:         $MODE"
log "  GPUs:         $GPUS  (共 $NUM_GPUS 张)"
log "  launch:       $([ $NUM_GPUS -gt 1 ] && echo 'accelerate launch (DDP)' || echo 'python (单卡)')"
log "  conda env:    ${CONDA_DEFAULT_ENV:-未知}"
[ -n "$CKPT" ] && log "  ckpt:         $CKPT"
echo ""

log "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader | \
    awk -F', ' -v gpus="$GPUS" '
    BEGIN { split(gpus, sel, ","); for (i in sel) selected[sel[i]] = 1 }
    { mark = (($1) in selected) ? " <- 将使用" : ""; 
      print "  [" $1 "] " $2 " | used=" $3 " free=" $4 mark }'
echo ""

read -t 10 -p "确认开始? [Y/n] (10 秒自动开始) " ans || ans="y"
[[ "$ans" =~ ^[Nn]$ ]] && { log "已取消"; exit 0; }

# ----------------------------------------------------------------------------
# 输出目录
# ----------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME_FROM_CFG=$(grep -E "^exp_name:" "$CONFIG" 2>/dev/null | awk '{print $2}' | tr -d '"')
EXP_NAME="${EXP_NAME:-${EXP_NAME_FROM_CFG:-default}}"
OUTPUT_DIR="outputs/${EXP_NAME}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"

log "训练日志: $LOG_FILE"
log "Tensorboard: tensorboard --logdir ${OUTPUT_DIR}/logs"
echo ""

trap 'echo ""; warn "中断信号, 等待清理..."; sleep 2; exit 130' INT TERM

# ----------------------------------------------------------------------------
# 主命令: 根据 GPU 数自动选择启动方式
# ----------------------------------------------------------------------------
EXTRA_ARGS=()
[ -n "$CKPT" ] && EXTRA_ARGS=(--ckpt "$CKPT")

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡: 直接 python
    CMD="CUDA_VISIBLE_DEVICES=$GPUS python run.py --config $CONFIG --mode $MODE ${EXTRA_ARGS[*]}"
    log "执行命令:"
    log "  $CMD"
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" python run.py \
        --config "$CONFIG" --mode "$MODE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    RC=${PIPESTATUS[0]}
else
    # 多卡: accelerate launch
    # 找一个未被占用的端口避免冲突
    MASTER_PORT=$((20000 + RANDOM % 10000))
    CMD="CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --num_processes $NUM_GPUS \
         --num_machines 1 --mixed_precision bf16 --main_process_port $MASTER_PORT \
         run.py --config $CONFIG --mode $MODE ${EXTRA_ARGS[*]}"
    log "执行命令:"
    log "  $CMD"
    echo ""
    CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch \
        --num_processes "$NUM_GPUS" \
        --num_machines 1 \
        --mixed_precision bf16 \
        --main_process_port "$MASTER_PORT" \
        run.py --config "$CONFIG" --mode "$MODE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    RC=${PIPESTATUS[0]}
fi

# ----------------------------------------------------------------------------
# 总结
# ----------------------------------------------------------------------------
echo ""
log "===================================================="
if [ $RC -eq 0 ]; then
    log "训练完成 ✓"
    log "  日志:  $LOG_FILE"
    log "  权重:  $OUTPUT_DIR/adapter_*/"
    log "  TB:    tensorboard --logdir $OUTPUT_DIR/logs"
elif [ $RC -eq 130 ]; then
    warn "训练被中断 (Ctrl+C)"
else
    err "训练失败, 退出码: $RC"
    err "  日志: $LOG_FILE"
    err "  尾部:"
    tail -80 "$LOG_FILE" | sed 's/^/    /'
fi

[ "$DEBUG" = "1" ] && rm -f configs/_debug.yaml

exit $RC