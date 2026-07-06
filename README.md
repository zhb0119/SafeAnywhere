# SafeAnywhere

SafeAnywhere 当前用于复现第一轮 safety-think SFT pilot：

- SafeChain cold-start safety-think：1000 条
- HEx-PHI masked dangerous-prefix recovery：500 条
- mixed 1500 SFT：1350 train / 150 val
- LLaMA-Factory v1 span-mask JSONL 训练

Dangerous-prefix 样本使用一个 assistant turn 内的 span-level mask：`assistant_prefill` 只作为上下文，loss 只落在 `<safety_think>...</safety_think>` 和 recovery 上。

## 目录

```text
configs/             # 数据构造配置
scripts/             # 数据构造、导出、mask 验证脚本
src/safeanywhere/    # 数据采样、teacher、校验、导出逻辑
integrations/        # LLaMA-Factory custom template
train/llamafactory/  # LLaMA-Factory v1 训练配置
docs/                # 方法与数据设计文档
data/                # 本地数据源，Git 忽略
build/               # 构造产物，Git 忽略
runs/                # 训练产物，Git 忽略
```

## 1. 安装 SafeAnywhere 环境

```bash
cd /root/workspace/SafeAnywhere
uv sync --frozen
```

检查环境：

```bash
uv run python scripts/00_check_env.py --config configs/safechain_smoke_10.yaml
```

## 2. 配置 Teacher API

```bash
cd /root/workspace/SafeAnywhere
cp .env.example .env
```

在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

检查 API key：

```bash
uv run python scripts/00_check_env.py \
  --config configs/safechain_smoke_10.yaml \
  --require-api
```

## 3. 准备数据源

把本地数据放到：

```text
data/UWNSL__SafeChain__train.jsonl
data/Harmful-HEx-PHI.jsonl
```

如果路径不同，修改对应配置：

```text
configs/safechain_smoke_10.yaml
configs/safechain_pilot_1k.yaml
configs/hex_phi_prefix_500.yaml
```

## 4. 构造 SafeChain 数据

可选：先跑 mock smoke，不调用 API。

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/01_build_dataset.py \
  --config configs/safechain_smoke_10.yaml \
  --mock
```

真实 smoke：

```bash
uv run python scripts/01_build_dataset.py \
  --config configs/safechain_smoke_10.yaml \
  --workers 2
```

构造 SafeChain 1k：

```bash
uv run python scripts/01_build_dataset.py \
  --config configs/safechain_pilot_1k.yaml \
  --workers 1
```

中断后继续：

```bash
uv run python scripts/01_build_dataset.py \
  --config configs/safechain_pilot_1k.yaml \
  --workers 1 \
  --resume
```

## 5. 构造 HEx-PHI Dangerous-Prefix 数据

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/02_build_dangerous_prefix.py \
  --config configs/hex_phi_prefix_500.yaml \
  --workers 1
```

中断后继续：

```bash
uv run python scripts/02_build_dangerous_prefix.py \
  --config configs/hex_phi_prefix_500.yaml \
  --workers 1 \
  --resume
```

## 6. 合并 mixed 1500

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/03_merge_sft_pilot.py
```

产物：

```text
build/mixed_safechain1k_prefix500/sft_train.jsonl
build/mixed_safechain1k_prefix500/sft_val.jsonl
build/mixed_safechain1k_prefix500/report.json
```

## 7. 导出 LLaMA-Factory v1 数据

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/04_export_llamafactory_v1.py
```

产物：

```text
build/mixed_safechain1k_prefix500/train_lf_v1_spanmasked.jsonl
build/mixed_safechain1k_prefix500/val_lf_v1_spanmasked.jsonl
train/llamafactory/dataset_safeanywhere_1500_train.yaml
train/llamafactory/dataset_safeanywhere_1500_val.yaml
```

结构级 mask 验证：

```bash
uv run python scripts/05_validate_llamafactory_masks.py --structure-only
```

## 8. Clone LLaMA-Factory

```bash
cd /root/workspace
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout a61cfa692a70fcced4ba32a846d1e2de95f2865e
```

## 9. 安装 LLaMA-Factory 环境

如果当前 conda 环境已有可用 GPU PyTorch，直接复用当前环境：

```bash
cd /root/workspace/LLaMA-Factory
python -m pip install -U pip
python -m pip install -e .
```

检查 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

只有 `torch.cuda.is_available()` 不是 `True` 时才重装 GPU 版 PyTorch，例如 CUDA 12.6：

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

当前配置是普通 LoRA，不需要 `bitsandbytes` 或 DeepSpeed。需要 QLoRA 时再装：

```bash
python -m pip install bitsandbytes
```

需要 DeepSpeed/ZeRO 时再装：

```bash
python -m pip install -r requirements/deepspeed.txt
```

## 10. 复制 LLaMA-Factory Template

```bash
cd /root/workspace/SafeAnywhere
cp integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py \
  /root/workspace/LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/templates/
```

训练配置使用：

```yaml
template: safeanywhere_qwen3_nothink
```

## 11. 模型路径

当前训练配置默认使用：

```yaml
model: ../models/Qwen3-0.6B
```

在当前机器上对应：

```text
/root/workspace/models/Qwen3-0.6B
```

如果换模型，修改：

```text
train/llamafactory/qwen3_lora_sft_debug.yaml
train/llamafactory/qwen3_lora_sft_1500_v1.yaml
```

示例：

```yaml
model: /path/to/Qwen3-4B-Instruct-2507
model: ../models/Qwen3-4B-Instruct-2507
model: Qwen/Qwen3-4B-Instruct-2507
```

数据路径和输出路径已经是仓库内相对路径。训练命令必须从 SafeAnywhere 根目录执行：

```bash
cd /root/workspace/SafeAnywhere
```

## 12. 验证 LLaMA-Factory Renderer Mask

```bash
cd /root/workspace/SafeAnywhere
python scripts/05_validate_llamafactory_masks.py \
  --renderer llamafactory \
  --llamafactory-root /root/workspace/LLaMA-Factory
```

预期数量：

```text
train.rows = 1350
train.dangerous_prefix = 450
val.rows = 150
val.dangerous_prefix = 50
```

## 13. Debug SFT

```bash
cd /root/workspace/SafeAnywhere
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft train/llamafactory/qwen3_lora_sft_debug.yaml
```

输出目录：

```text
runs/qwen3_safeanywhere_lora_debug/
```

## 14. 正式 1500 SFT

```bash
cd /root/workspace/SafeAnywhere
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft train/llamafactory/qwen3_lora_sft_1500_v1.yaml
```

输出目录：

```text
runs/qwen3_safeanywhere_lora_1500_v1/
```

## 15. SafeAnywhere 自定义评测

构造 held-out eval。默认从 mixed val 生成 direct harmful、prefix recovery、benign utility、over-refusal 四类样本。

```bash
cd /root/workspace/SafeAnywhere
uv run python scripts/06_build_eval_sets.py
```

产物：

```text
build/eval/safeanywhere_v1/safeanywhere_eval.jsonl
build/eval/safeanywhere_v1/prefix_recovery.jsonl
build/eval/safeanywhere_v1/harmful_direct.jsonl
build/eval/safeanywhere_v1/benign_utility.jsonl
build/eval/safeanywhere_v1/over_refusal.jsonl
build/eval/safeanywhere_v1/report.json
```

先评测 base model：

```bash
cd /root/workspace/SafeAnywhere
python scripts/07_generate_eval_responses.py \
  --base-model ../models/Qwen3-0.6B \
  --output build/eval/safeanywhere_v1/base_predictions.jsonl

python scripts/08_score_eval_results.py \
  --input build/eval/safeanywhere_v1/base_predictions.jsonl \
  --scored-output build/eval/safeanywhere_v1/base_predictions_scored.jsonl \
  --summary-output build/eval/safeanywhere_v1/base_score_summary.json
```

再评测 SFT LoRA：

```bash
cd /root/workspace/SafeAnywhere
python scripts/07_generate_eval_responses.py \
  --base-model ../models/Qwen3-0.6B \
  --adapter runs/qwen3_safeanywhere_lora_1500_v1 \
  --output build/eval/safeanywhere_v1/sft_predictions.jsonl

python scripts/08_score_eval_results.py \
  --input build/eval/safeanywhere_v1/sft_predictions.jsonl \
  --scored-output build/eval/safeanywhere_v1/sft_predictions_scored.jsonl \
  --summary-output build/eval/safeanywhere_v1/sft_score_summary.json
```

对比 base 和 SFT：

```bash
python scripts/09_compare_eval_reports.py \
  --base build/eval/safeanywhere_v1/base_score_summary.json \
  --candidate build/eval/safeanywhere_v1/sft_score_summary.json
```

核心关注：

```text
prefix_recovery.prefix_recovery_success_rate  越高越好
prefix_recovery.unsafe_continuation_rate      越低越好
harmful_direct.harmful_safe_response_rate     越高越好
benign_utility.benign_answer_rate             越高越好
over_refusal.over_refusal_rate                越低越好
safety_think_rate                             看方法是否学会触发 <safety_think>
```

快速 smoke 可加 `--limit`：

```bash
python scripts/07_generate_eval_responses.py \
  --base-model ../models/Qwen3-0.6B \
  --adapter runs/qwen3_safeanywhere_lora_debug \
  --limit 20 \
  --output build/eval/safeanywhere_v1/debug_predictions.jsonl
```

这些脚本的启发式打分只用于本地 smoke。论文或正式报告建议再跑 HarmBench、JailbreakBench、XSTest 和 lm-eval。

## 16. 外部 Benchmark

准备外部 benchmark prompt。默认会下载/转换 AdvBench、XSTest、JailbreakBench 到统一 JSONL。

```bash
cd /root/workspace/SafeAnywhere
python scripts/10_prepare_external_benchmarks.py \
  --benchmarks advbench xstest jailbreakbench \
  --output-dir build/eval/external
```

产物：

```text
build/eval/external/advbench/advbench_eval.jsonl
build/eval/external/xstest/xstest_eval.jsonl
build/eval/external/jailbreakbench/jbb_eval.jsonl
build/eval/external/external_benchmarks_report.json
```

快速 smoke 可加 `--limit`：

```bash
python scripts/10_prepare_external_benchmarks.py \
  --benchmarks advbench xstest \
  --limit 20 \
  --output-dir build/eval/external_smoke
```

生成并启发式打分 base model：

```bash
python scripts/11_run_external_generation_eval.py \
  --benchmarks advbench xstest jailbreakbench \
  --base-model ../models/Qwen3-0.6B \
  --run-name base \
  --output-dir build/eval/external
```

生成并启发式打分 SFT LoRA：

```bash
python scripts/11_run_external_generation_eval.py \
  --benchmarks advbench xstest jailbreakbench \
  --base-model ../models/Qwen3-0.6B \
  --adapter runs/qwen3_safeanywhere_lora_1500_v1 \
  --run-name sft \
  --output-dir build/eval/external
```

对应 summary 文件：

```text
build/eval/external/advbench/base_score_summary.json
build/eval/external/advbench/sft_score_summary.json
build/eval/external/xstest/base_score_summary.json
build/eval/external/xstest/sft_score_summary.json
build/eval/external/jailbreakbench/base_score_summary.json
build/eval/external/jailbreakbench/sft_score_summary.json
```

对比某个 benchmark：

```bash
python scripts/09_compare_eval_reports.py \
  --base build/eval/external/advbench/base_score_summary.json \
  --candidate build/eval/external/advbench/sft_score_summary.json
```

生成 lm-eval / HarmBench 辅助命令：

```bash
python scripts/12_write_external_benchmark_commands.py \
  --output-dir build/eval/external/commands
```

会写出：

```text
build/eval/external/commands/merge_lora.sh
build/eval/external/commands/lm_eval.sh
build/eval/external/commands/harmbench.sh
```

执行顺序：

```bash
bash build/eval/external/commands/merge_lora.sh
bash build/eval/external/commands/lm_eval.sh
bash build/eval/external/commands/harmbench.sh
```

外部 benchmark 来源：

```text
HarmBench:       https://github.com/centerforaisafety/HarmBench
JailbreakBench: https://github.com/JailbreakBench/jailbreakbench
AdvBench:       https://github.com/llm-attacks/llm-attacks
XSTest:         https://github.com/paul-rottger/xstest
```

Utility：

```text
lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
MMLU / GSM8K / ARC / HellaSwag / TruthfulQA / IFEval
```

推荐最终表格至少包含：

```text
HarmBench ASR ↓
JailbreakBench ASR ↓
AdvBench harmful compliance ↓
XSTest safe refusal ↓
SafeAnywhere prefix recovery success ↑
SafeAnywhere unsafe continuation ↓
IFEval ↑
MMLU / GSM8K / ARC ↑
```

## 17. 常用检查

查看数据构造报告：

```bash
cat build/safechain_pilot_1k/report.json
cat build/hex_phi_prefix_500/report.json
cat build/mixed_safechain1k_prefix500/report.json
```

代码检查：

```bash
uv run ruff check .
python -m py_compile \
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
  integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py
```

## 18. Push 到 GitHub

当前远端：

```text
https://github.com/zhb0119/SafeAnywhere
```

建议提交代码和配置，不提交 `data/`、`build/`、`runs/`、`.env`。

```bash
cd /root/workspace/SafeAnywhere
git status
git add \
  .gitignore \
  README.md \
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
  integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py \
  train/llamafactory

git commit -m "Add LLaMA-Factory span-mask SFT workflow"
git push origin main
```
