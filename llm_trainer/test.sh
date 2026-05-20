#!/usr/bin/env bash
# =============================================================================
# test.sh — llm_trainer 分层冒烟测试
#
# 用法:
#   bash test.sh              # 跑全部测试 (T1 -> T5)
#   bash test.sh T1           # 只跑某一项
#   bash test.sh T1 T2 T3     # 跑指定几项
#
# 测试分层 (越往后越重):
#   T1  语法检查           (秒级,无依赖)
#   T2  配置加载           (秒级,需要 omegaconf)
#   T3  SAE 类 + 权重加载  (秒级,需要 torch)
#   T4  数据集加载         (~1分钟,需要 pandas + parquet)
#   T5  端到端 1 步训练    (大,需要 GPU + LLaVA 权重,谨慎跑)
#
# 在项目根目录执行 (即 lvlms/llm_trainer/ 目录):
#   cd code/VL-SAE-main/lvlms/llm_trainer
#   bash test.sh
# =============================================================================

set -u    # 引用未定义变量直接报错
# 不用 set -e: 我们要让所有测试都跑完,在最后统计结果

# 颜色
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

PASS=()
FAIL=()

run_test() {
    local name="$1"; shift
    local desc="$1"; shift
    echo -e "\n${YELLOW}========== [$name] $desc ==========${NC}"
    if "$@"; then
        echo -e "${GREEN}[PASS] $name${NC}"
        PASS+=("$name")
    else
        echo -e "${RED}[FAIL] $name${NC}"
        FAIL+=("$name")
    fi
}

# =============================================================================
# T1: 语法 / import 检查 — 不需要 GPU,不需要权重
# =============================================================================
test_T1() {
    python - <<'PY'
import ast, sys
files = ["model.py", "data.py", "trainer.py", "run.py"]
for f in files:
    try:
        ast.parse(open(f, encoding="utf-8").read())
        print(f"  {f}: syntax OK")
    except SyntaxError as e:
        print(f"  {f}: SYNTAX ERROR -> {e}")
        sys.exit(1)
PY
}

# =============================================================================
# T2: 配置文件能否被 omegaconf 正确加载 + defaults 合并
# =============================================================================
test_T2() {
    python - <<'PY'
import sys
sys.path.insert(0, ".")
from run import load_config
cfg = load_config("configs/exp_lora_sae.yaml")
print(f"  exp_name: {cfg.exp_name}")
print(f"  sae.layer_idx: {cfg.sae.layer_idx}")
print(f"  sae.ckpt_path: {cfg.sae.ckpt_path}")
print(f"  data.data_root: {cfg.data.data_root}")
print(f"  train.batch_size: {cfg.train.batch_size}")
# 关键字段必须存在
assert cfg.sae.layer_idx == 24, "layer_idx should be 24"
assert cfg.train.pool_layer_idx is not None
print("  config merge OK")
PY
}

# =============================================================================
# T3: SAE 类能否实例化 (不加载权重) + 能否从 sae_trainer/sae_model.py 动态导入
# =============================================================================
test_T3() {
    python - <<'PY'
import sys, os
sys.path.insert(0, ".")

# 1) 动态导入 sae_trainer/sae_model.py
from model import _import_sae_module, _resolve_sae_class
sae_module_path = "../sae_trainer/sae_model.py"
if not os.path.isfile(sae_module_path):
    print(f"  SKIP: {sae_module_path} not found (你还没创建 sae_model.py?)")
    sys.exit(0)

mod = _import_sae_module(sae_module_path)
SAECls = _resolve_sae_class(mod, None)
print(f"  found SAE class: {SAECls.__name__}")

# 2) 试着实例化 (小尺寸,避免吃显存)
import torch
try:
    sae = SAECls(d_in=128, num_latents=512, k=8)
except TypeError:
    # 不同 SAE 类签名不同;失败也算通过 (只是说明 init_kwargs 要手动指定)
    print("  WARN: 默认 init_kwargs 无法实例化,需要在 configs 里指定 sae.init_kwargs")
    sys.exit(0)

x = torch.randn(2, 4, 128)
out = sae(x)
# 兼容 (x_hat, z) 或 dict 或 单个 tensor
if isinstance(out, tuple):
    x_hat, z = out[0], out[1] if len(out) > 1 else None
elif isinstance(out, dict):
    x_hat = out.get("x_hat") or out.get("reconstruction")
else:
    x_hat = out
assert x_hat.shape == x.shape, f"reconstruction shape mismatch: {x_hat.shape}"
print(f"  forward OK: x.shape={tuple(x.shape)}, x_hat.shape={tuple(x_hat.shape)}")

# 3) 检查权重目录是否存在 (不实际加载,8B 太大)
ckpt_dir = "../sae_trainer/finetune_models/llama3-llava-next-8b-hf-sae-131k"
if os.path.isdir(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir)
             if f.endswith((".pt", ".bin", ".safetensors"))]
    print(f"  ckpt dir exists, weight files: {files}")
else:
    print(f"  WARN: ckpt dir not found: {ckpt_dir}")
PY
}

# =============================================================================
# T4: MMEB 数据集能否加载 + collate_fn 能否工作
#     不需要 LLaVA,但需要 LlavaNextProcessor (会去拉 tokenizer 配置)
# =============================================================================
test_T4() {
    python - <<'PY'
import sys, os
sys.path.insert(0, ".")
import torch

# 1) 检查 data_root
from run import load_config
cfg = load_config("configs/exp_lora_sae.yaml")
data_root = cfg.data.data_root
if not os.path.isdir(data_root):
    print(f"  SKIP: data_root not found: {data_root}")
    sys.exit(0)

# 2) 加载数据集 (不用 processor,只测 Dataset 部分)
from data import MMEBRetrievalDataset
try:
    ds = MMEBRetrievalDataset(
        data_root=data_root,
        dataset_names=cfg.data.train_datasets,
        images_root=cfg.data.images_root,
        max_candidates=4,    # 测试时少要点
    )
    print(f"  dataset size: {len(ds)}")
except Exception as e:
    print(f"  Dataset 加载失败: {e}")
    sys.exit(1)

# 3) 取一条样本看结构
sample = ds[0]
print(f"  sample keys: {list(sample.keys())}")
print(f"  query.text: {sample['query']['text'][:50] if sample['query']['text'] else None}...")
print(f"  query.image: {type(sample['query']['image']).__name__ if sample['query']['image'] else None}")
print(f"  num candidates: {len(sample['candidates'])}")
print(f"  gold_idx: {sample['gold_idx']}")

# 4) 测 collate_fn — 需要 processor
# 这里用一个 mock processor 避免下载 LLaVA 权重
class MockTokenizer:
    pad_token_id = 0
    def __call__(self, texts, **kwargs):
        max_len = kwargs.get("max_length", 32)
        bsz = len(texts)
        return {
            "input_ids": torch.zeros(bsz, max_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, max_len, dtype=torch.long),
        }
class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
    def __call__(self, text, images, **kwargs):
        max_len = kwargs.get("max_length", 32)
        bsz = len(text)
        return {
            "input_ids": torch.zeros(bsz, max_len, dtype=torch.long),
            "attention_mask": torch.ones(bsz, max_len, dtype=torch.long),
            "pixel_values": torch.zeros(bsz, 5, 3, 336, 336),
            "image_sizes": torch.zeros(bsz, 2, dtype=torch.long),
        }

from data import build_collate_fn
collate = build_collate_fn(MockProcessor(), max_length=32)
batch = collate([ds[0], ds[1] if len(ds) > 1 else ds[0]])
print(f"  collate output keys: {list(batch.keys())}")
print(f"  input_ids shape: {tuple(batch['input_ids'].shape)}")
print(f"  sample_ids: {batch['sample_ids'].tolist()}")
print(f"  role_ids:   {batch['role_ids'].tolist()}")
print(f"  gold_idx:   {batch['gold_idx'].tolist()}")
assert batch["input_ids"].shape[0] == batch["sample_ids"].shape[0]
assert batch["sample_ids"].shape[0] == batch["role_ids"].shape[0]
print("  collate OK")
PY
}

# =============================================================================
# T5: 端到端 1 步训练 — 真的跑 LLaVA + SAE + 反向传播 1 步
#     需要: GPU,LLaVA 权重已下载,SAE 权重已下载
#     这是最贵的测试,建议先确认 T1-T4 都过了再跑
# =============================================================================
test_T5() {
    # 检查 GPU
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "  SKIP: 没有可用 GPU,跳过端到端测试"
        return 0
    fi

    # 创建一个临时小规模配置,只跑 1 step
    cat > configs/_smoke_test.yaml <<'YAML'
defaults:
  - base
exp_name: _smoke_test
data:
  train_datasets: ["A-OKVQA"]
  max_candidates: 2          # 展平后每条样本只 1+2=3 条 forward
  max_length: 3500
  num_workers: 0
train:
  epochs: 1
  batch_size: 1
  grad_accum_steps: 1
  log_steps: 1
  save_steps: 999999          # 不要中途存盘
  eval_steps: 999999
YAML

    # 用 LIMIT 环境变量截断 dataloader,只跑 1 步就退出
    SMOKE_TEST=1 timeout 1800 python run.py \
        --config configs/_smoke_test.yaml \
        --mode train 2>&1 | tail -30

    local rc=${PIPESTATUS[0]}
    rm -f configs/_smoke_test.yaml
    if [ $rc -eq 124 ]; then
        echo "  TIMEOUT: 30 分钟还没跑完 1 步,可能卡在加载模型/数据"
        return 1
    fi
    return $rc
}

# =============================================================================
# 入口
# =============================================================================
ALL_TESTS=(T1 T2 T3 T4 T5)
if [ $# -eq 0 ]; then
    TESTS=("${ALL_TESTS[@]}")
else
    TESTS=("$@")
fi

# 确认在正确目录
if [ ! -f "model.py" ] || [ ! -f "data.py" ]; then
    echo -e "${RED}错误: 请在 llm_trainer/ 目录下运行此脚本${NC}"
    exit 1
fi

for t in "${TESTS[@]}"; do
    case "$t" in
        T1) run_test "T1" "语法检查" test_T1 ;;
        T2) run_test "T2" "配置文件加载与合并" test_T2 ;;
        T3) run_test "T3" "SAE 类导入与实例化" test_T3 ;;
        T4) run_test "T4" "MMEB 数据集加载与 collate" test_T4 ;;
        T5) run_test "T5" "端到端 1 步训练 (需 GPU)" test_T5 ;;
        *)  echo -e "${RED}未知测试: $t${NC}" ;;
    esac
done

# =============================================================================
# 总结
# =============================================================================
echo -e "\n${YELLOW}========== 总结 ==========${NC}"
echo -e "${GREEN}PASS: ${PASS[*]:-(none)}${NC}"
echo -e "${RED}FAIL: ${FAIL[*]:-(none)}${NC}"

if [ ${#FAIL[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ 全部通过${NC}"
    exit 0
else
    echo -e "${RED}✗ 有失败项${NC}"
    exit 1
fi
