# SafeAnywhere

SafeAnywhere 当前默认构建 prefix2800 版 safety-think SFT 数据：

- SafeChain cold-start safety-think：1000 条
- HEx-PHI masked dangerous-prefix recovery：1200 条
- SafeChain harmful/adversarial harmful masked prefix recovery：1600 条
- mixed 3800 SFT：3420 train / 380 val
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

## 3. 生成主 SFT 数据集

默认只跑这一条命令：

```bash
bash scripts/15_build_prefix2800_dataset.sh
```

它会按顺序完成：

```text
SafeChain cold-start 1000
HEx-PHI dangerous-prefix 1200
SafeChain harmful-prefix 1600
merge -> mixed_safechain1k_prefix2800
export -> LLaMA-Factory span-mask JSONL
validate -> structure-only mask check
```

关键约束不变：所有 dangerous-prefix 样本都是 `user=0 / assistant_prefill=0 / recovery_target=1`，也就是 prefix 只作为 masked context，不参与 loss。

Prefix type 扩展不是单独命令，已经写入 `configs/hex_phi_prefix_1200.yaml` 和 `configs/safechain_prefix_1600.yaml`。SafeChain 1600 默认覆盖 `affirmative_prefix`、`outline_prefix`、`materials_prefix`、`source_link_prefix`、`roleplay_acceptance_prefix`、`code_prefix`、`evasion_prefix`。

常用参数：

```bash
# 中断后续跑
RESUME=1 bash scripts/15_build_prefix2800_dataset.sh

# 并发 teacher 请求
WORKERS=2 bash scripts/15_build_prefix2800_dataset.sh

# 不调用 teacher，只验证流程和 mask
MOCK=1 bash scripts/15_build_prefix2800_dataset.sh

# 打印完整阶段输出；默认完整日志写入 build/mixed_safechain1k_prefix2800/build_prefix2800.log
VERBOSE=1 bash scripts/15_build_prefix2800_dataset.sh
```

主产物：

```text
build/mixed_safechain1k_prefix2800/sft_train.jsonl
build/mixed_safechain1k_prefix2800/sft_val.jsonl
build/mixed_safechain1k_prefix2800/train_lf_v1_spanmasked.jsonl
build/mixed_safechain1k_prefix2800/val_lf_v1_spanmasked.jsonl
train/llamafactory/dataset_safeanywhere_prefix2800_train.yaml
train/llamafactory/dataset_safeanywhere_prefix2800_val.yaml
```

报告：

```text
build/safechain_pilot_1k/report.json
build/hex_phi_prefix_1200/report.json
build/safechain_prefix_1600/report.json
build/mixed_safechain1k_prefix2800/report.json
build/mixed_safechain1k_prefix2800/llamafactory_v1_export_report.json
build/mixed_safechain1k_prefix2800/build_prefix2800.log
```

## 4. Legacy 1500 Pilot

只在复现早期 1500 条 pilot 时使用：

```bash
uv run python scripts/01_build_dataset.py --config configs/safechain_pilot_1k.yaml --workers 1
uv run python scripts/02_build_dangerous_prefix.py --config configs/hex_phi_prefix_500.yaml --workers 1
uv run python scripts/03_merge_sft_pilot.py
uv run python scripts/04_export_llamafactory_v1.py
```

对应产物仍是：

```text
build/mixed_safechain1k_prefix500/
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
    Path("build/hex_phi_prefix_1200/sft_train.jsonl"),
    Path("build/hex_phi_prefix_1200/sft_val.jsonl"),
    Path("build/safechain_prefix_1600/sft_train.jsonl"),
    Path("build/safechain_prefix_1600/sft_val.jsonl"),
    Path("build/mixed_safechain1k_prefix2800/sft_train.jsonl"),
    Path("build/mixed_safechain1k_prefix2800/sft_val.jsonl"),
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
uv run python scripts/05_validate_llamafactory_masks.py \
  --structure-only \
  --train build/mixed_safechain1k_prefix2800/train_lf_v1_spanmasked.jsonl \
  --val build/mixed_safechain1k_prefix2800/val_lf_v1_spanmasked.jsonl

USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  python scripts/05_validate_llamafactory_masks.py \
  --train build/mixed_safechain1k_prefix2800/train_lf_v1_spanmasked.jsonl \
  --val build/mixed_safechain1k_prefix2800/val_lf_v1_spanmasked.jsonl \
  --renderer llamafactory \
  --llamafactory-root /root/workspace/LLaMA-Factory \
  --max-render-checks 2000
```

预期：

```text
train.rows = 3420
train.dangerous_prefix = 2520
val.rows = 380
val.dangerous_prefix = 280
```

## 7. SFT

默认模型路径：

```text
/root/workspace/models/Qwen3-0.6B
```

如需替换模型，修改：

```text
train/llamafactory/qwen3_lora_sft_prefix2800_v1.yaml
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
  llamafactory-cli sft train/llamafactory/qwen3_lora_sft_prefix2800_v1.yaml
```

输出：

```text
runs/qwen3_safeanywhere_lora_prefix2800_v1/
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
  scripts/02b_build_safechain_prefix.py \
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
bash -n scripts/15_build_prefix2800_dataset.sh
```

## 11. Push

```bash
cd /root/workspace/SafeAnywhere
git status
git add README.md configs scripts src integrations train
git commit -m "Fix clean SFT prompt generation and eval workflow"
git push origin main
```
