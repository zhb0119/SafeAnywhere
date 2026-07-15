# SafeAnywhere

SafeAnywhere 用于构建 SFT 数据，重点是让模型在危险前缀下恢复到安全续写。
当前可复现快照在 `build/data_build/safeanywhere_sft_v1/`。

## 当前快照

- `safechain`：1000 条（900 train / 100 val）
- `harmful_prefix/safechain_harmful`：300 条（270 train / 30 val）
- 合并后 SFT：1170 train / 130 val
- LLaMA-Factory 导出：
  - `build/data_build/safeanywhere_sft_v1/train_lf_v1_spanmasked.jsonl`
  - `build/data_build/safeanywhere_sft_v1/val_lf_v1_spanmasked.jsonl`

Mask 约定：

- `safechain`：`user=0 / assistant=1`
- `dangerous_prefix`：`user=0 / assistant_prefill=0 / recovery_target=1`

## 环境

```bash
cd /root/workspace/SafeAnywhere
uv sync --frozen
cp .env.example .env
```

`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 用于 teacher。
`JUDGE_*` 只在评测时需要。

## 数据构建

从源数据重新生成当前快照：

```bash
uv run python scripts/build_sft_dataset.py \
  --config configs/data_build/safeanywhere_sft_v1.yaml \
  --workers 1 \
  --quiet
```

如果只想复用当前仓库里已落盘的数据，可直接合并、导出并校验：

```bash
python3 scripts/data/merge_sft.py \
  --safechain-train build/data_build/safeanywhere_sft_v1/safechain/sft_train.jsonl \
  --safechain-val build/data_build/safeanywhere_sft_v1/safechain/sft_val.jsonl \
  --prefix-train build/data_build/safeanywhere_sft_v1/harmful_prefix/safechain_harmful/sft_train.jsonl \
  --prefix-val build/data_build/safeanywhere_sft_v1/harmful_prefix/safechain_harmful/sft_val.jsonl \
  --output-dir build/data_build/safeanywhere_sft_v1 \
  --no-strict-counts

python3 scripts/data/export_llamafactory_v1.py \
  --input-dir build/data_build/safeanywhere_sft_v1 \
  --output-dir build/data_build/safeanywhere_sft_v1 \
  --train-dataset-yaml configs/sft/llamafactory/dataset_safeanywhere_sft_v1_train.yaml \
  --val-dataset-yaml configs/sft/llamafactory/dataset_safeanywhere_sft_v1_val.yaml

python3 scripts/data/validate_llamafactory_masks.py \
  --train build/data_build/safeanywhere_sft_v1/train_lf_v1_spanmasked.jsonl \
  --val build/data_build/safeanywhere_sft_v1/val_lf_v1_spanmasked.jsonl
```

当前 `configs/data_build/safeanywhere_sft_v1.yaml` 只构建 `safechain + harmful_prefix/safechain_harmful`。
如果要做完整 1500 条混合数据，再补建 `harmful_prefix/hex_phi`，并把 `pipeline.finalize` 设为 `true`。

## SFT

1. 复制 LLaMA-Factory 模板：

```bash
cp integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py \
  /root/workspace/LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/templates/
```

2. 训练：

```bash
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml
```

3. 如基础模型路径不同，修改 `configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml` 里的 `model`。

## 评测

```bash
bash scripts/eval/run_eval_comparison.sh --config configs/eval/safeanywhere_v1.yaml
```

## 目录

```text
configs/
  data_build/
  sft/llamafactory/
  eval/
integrations/llamafactory/templates/
scripts/
  data/
  eval/
build/data_build/safeanywhere_sft_v1/
```
