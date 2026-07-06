# SafeAnywhere

SafeAnywhere 当前复现第一轮 safety-think SFT pilot：

- SafeChain cold-start safety-think：1000 条
- HEx-PHI masked dangerous-prefix recovery：500 条
- mixed 1500 SFT：1350 train / 150 val
- LLaMA-Factory v1 span-level mask 训练

SFT 数据里的 `prompt` 固定为原始 user request。Teacher prompt 只用于生成 teacher target，不会写入 SFT/eval prompt。Dangerous-prefix 样本通过 `messages` 和 LLaMA-Factory content span 实现：`assistant_prefill` 为 `loss_mask=0`，只有 `<safety_think>...</safety_think>` recovery target 参与 loss。

## 1. SafeAnywhere 环境

```bash
cd /root/workspace/SafeAnywhere
uv sync --frozen
uv run python scripts/00_check_env.py --config configs/safechain_smoke_10.yaml
```

配置 teacher：

```bash
cp .env.example .env
```

`.env`：

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

检查 API：

```bash
uv run python scripts/00_check_env.py \
  --config configs/safechain_smoke_10.yaml \
  --require-api
```

## 2. 数据源

放到：

```text
data/UWNSL__SafeChain__train.jsonl
data/Harmful-HEx-PHI.jsonl
```

## 3. 生成 SFT 数据

SafeChain smoke：

```bash
uv run python scripts/01_build_dataset.py \
  --config configs/safechain_smoke_10.yaml \
  --mock
```

SafeChain 1k：

```bash
uv run python scripts/01_build_dataset.py \
  --config configs/safechain_pilot_1k.yaml \
  --workers 1
```

HEx-PHI dangerous-prefix 500：

```bash
uv run python scripts/02_build_dangerous_prefix.py \
  --config configs/hex_phi_prefix_500.yaml \
  --workers 1
```

中断后加 `--resume` 继续。

## 4. 合并与导出

```bash
uv run python scripts/03_merge_sft_pilot.py
uv run python scripts/04_export_llamafactory_v1.py
```

产物：

```text
build/mixed_safechain1k_prefix500/sft_train.jsonl
build/mixed_safechain1k_prefix500/sft_val.jsonl
build/mixed_safechain1k_prefix500/train_lf_v1_spanmasked.jsonl
build/mixed_safechain1k_prefix500/val_lf_v1_spanmasked.jsonl
train/llamafactory/dataset_safeanywhere_1500_train.yaml
train/llamafactory/dataset_safeanywhere_1500_val.yaml
```

检查没有 prompt wrapper 泄漏：

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("build/safechain_pilot_1k/sft_train.jsonl"),
    Path("build/safechain_pilot_1k/sft_val.jsonl"),
    Path("build/hex_phi_prefix_500/sft_train.jsonl"),
    Path("build/hex_phi_prefix_500/sft_val.jsonl"),
    Path("build/mixed_safechain1k_prefix500/sft_train.jsonl"),
    Path("build/mixed_safechain1k_prefix500/sft_val.jsonl"),
]
needles = ["You are SafeAnywhere", "Rules for <safety_think>", "User request:", "Assistant prefill:"]
bad = []
for path in paths:
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        row = json.loads(line)
        if any(s in json.dumps(row, ensure_ascii=False) for s in needles):
            bad.append((str(path), line_no, row.get("id")))
            break
print("prompt_wrapper_leak_count", len(bad))
if bad:
    print(bad[:10])
    raise SystemExit(1)
PY
```

## 5. LLaMA-Factory

```bash
cd /root/workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout a61cfa692a70fcced4ba32a846d1e2de95f2865e
python -m pip install -U pip
python -m pip install -e .
```

检查 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果 `torch.cuda.is_available()` 不是 `True`，按机器 CUDA 版本安装 GPU PyTorch，例如 CUDA 12.6：

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

复制 template：

```bash
cd /root/workspace/SafeAnywhere
cp integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py \
  /root/workspace/LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/templates/
```

## 6. Mask 校验

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/05_validate_llamafactory_masks.py --structure-only

USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  python scripts/05_validate_llamafactory_masks.py \
  --renderer llamafactory \
  --llamafactory-root /root/workspace/LLaMA-Factory \
  --max-render-checks 2000
```

预期：

```text
train.rows = 1350
train.dangerous_prefix = 450
val.rows = 150
val.dangerous_prefix = 50
```

## 7. SFT

默认模型路径：

```text
/root/workspace/models/Qwen3-0.6B
```

如需替换模型，修改：

```text
train/llamafactory/qwen3_lora_sft_debug.yaml
train/llamafactory/qwen3_lora_sft_1500_v1.yaml
```

Debug：

```bash
cd /root/workspace/SafeAnywhere
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft train/llamafactory/qwen3_lora_sft_debug.yaml
```

正式训练：

```bash
cd /root/workspace/SafeAnywhere
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft train/llamafactory/qwen3_lora_sft_1500_v1.yaml
```

输出：

```text
runs/qwen3_safeanywhere_lora_debug/
runs/qwen3_safeanywhere_lora_1500_v1/
```

## 8. 自定义评测

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/06_build_eval_sets.py
```

一键生成、打分、写报告：

```bash
bash scripts/13_run_eval_comparison.sh
```

产物：

```text
build/eval/safeanywhere_v1/README.md
build/eval/safeanywhere_v1/base_score_summary.json
build/eval/safeanywhere_v1/sft_score_summary.json
build/eval/safeanywhere_v1/compare_base_vs_sft.md
```

核心指标：

```text
prefix_recovery.prefix_recovery_success_rate  越高越好
prefix_recovery.unsafe_continuation_rate      越低越好
harmful_direct.harmful_safe_response_rate     越高越好
benign_utility.benign_answer_rate             越高越好
over_refusal.over_refusal_rate                越低越好
```

## 9. 外部 Benchmark

准备数据：

```bash
python scripts/10_prepare_external_benchmarks.py \
  --benchmarks advbench xstest jailbreakbench \
  --output-dir build/eval/external
```

base：

```bash
python scripts/11_run_external_generation_eval.py \
  --benchmarks advbench xstest jailbreakbench \
  --base-model ../models/Qwen3-0.6B \
  --run-name base \
  --output-dir build/eval/external
```

SFT：

```bash
python scripts/11_run_external_generation_eval.py \
  --benchmarks advbench xstest jailbreakbench \
  --base-model ../models/Qwen3-0.6B \
  --adapter runs/qwen3_safeanywhere_lora_1500_v1 \
  --run-name sft \
  --output-dir build/eval/external
```

HarmBench / lm-eval 辅助命令：

```bash
python scripts/12_write_external_benchmark_commands.py \
  --output-dir build/eval/external/commands

bash build/eval/external/commands/merge_lora.sh
bash build/eval/external/commands/lm_eval.sh
bash build/eval/external/commands/harmbench.sh
```

## 10. 常用检查

```bash
uv run ruff check .
python -m py_compile \
  src/safeanywhere/*.py \
  scripts/01_build_dataset.py \
  scripts/02_build_dangerous_prefix.py \
  scripts/03_merge_sft_pilot.py \
  scripts/04_export_llamafactory_v1.py \
  scripts/05_validate_llamafactory_masks.py \
  scripts/06_build_eval_sets.py \
  scripts/07_generate_eval_responses.py \
  scripts/08_score_eval_results.py \
  scripts/09_compare_eval_reports.py \
  scripts/10_prepare_external_benchmarks.py \
  scripts/11_run_external_generation_eval.py \
  scripts/12_write_external_benchmark_commands.py \
  scripts/14_write_eval_readme.py
bash -n scripts/13_run_eval_comparison.sh
```

## 11. Push

```bash
cd /root/workspace/SafeAnywhere
git status
git add README.md configs scripts src integrations train
git commit -m "Fix clean SFT prompt generation and eval workflow"
git push origin main
```
