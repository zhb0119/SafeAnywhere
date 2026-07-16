# SafeAnywhere

SafeAnywhere 构建 safety-think SFT 数据，并训练模型在危险前缀或越狱式 assistant prefill 后恢复到安全续写。当前 pipeline 包含两条 cold-start SFT 分支：

- `sft`：直接用原 tokenizer 训练 `<safety_think>...</safety_think>` 文本块。
- `sft_special`：先把 `<safety_think>` 和 `</safety_think>` 加入词表作为 special tokens，再做 SFT 和 OPSD。

## Pipeline

```text
data build
  -> LLaMA-Factory span-masked SFT data
  -> cold-start SFT LoRA
  -> merged SFT checkpoint
  -> OPSD self-distillation LoRA
  -> eval matrix
```

核心配置：

```text
configs/data_build/safeanywhere_sft_v1.yaml
configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml
configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1_special_tokens.yaml
configs/opsd/safechain_qwen3_0_6b.yaml
configs/opsd/safechain_qwen3_0_6b_special_tokens.yaml
configs/eval/safeanywhere_v1.yaml
```

## Data

当前数据快照位于 `build/data_build/safeanywhere_sft_v1/`：

| split | train | val | total |
| --- | ---: | ---: | ---: |
| `safechain` | 900 | 100 | 1000 |
| `harmful_prefix` | 607 | 67 | 674 |
| **merged SFT** | **1507** | **167** | **1674** |

Loss mask：

| source | mask |
| --- | --- |
| `safechain` | `user=0 / assistant=1` |
| `harmful_prefix` | `user=0 / assistant_prefill=0 / recovery_target=1` |

LLaMA-Factory 导出使用 content span 的 `loss_weight` 保留上述 mask。

## Runs

```text
runs/
  sft/
    qwen3_0_6b_v1/
  sft_special/
    qwen3_0_6b_safety_think_base/
    qwen3_0_6b_v1/
  merged/
    qwen3_0_6b_sft_v1/
    qwen3_0_6b_sft_special_v1/
  opsd/
    qwen3_0_6b_opsd_v1/
    qwen3_0_6b_opsd_special_v1/
    archive/
```

`sft` 只用于 SFT 阶段或 SFT merged checkpoint；`opsd/` 下的目录名使用 `opsd`，表示已经进入自蒸馏阶段。

## Setup

```bash
cd /root/workspace/SafeAnywhere
uv sync --frozen
uv sync --extra opsd
```

复制 LLaMA-Factory template：

```bash
cp integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py \
  /root/workspace/LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/templates/
```

## 1. Build Data

复用已有 annotations 并重新导出 SFT/LLaMA-Factory 数据：

```bash
uv run python scripts/build_sft_dataset.py \
  --config configs/data_build/safeanywhere_sft_v1.yaml \
  --export-existing-only \
  --quiet
```

需要重新调用 teacher API 时，去掉 `--export-existing-only`。

主要输出：

```text
build/data_build/safeanywhere_sft_v1/sft_train.jsonl
build/data_build/safeanywhere_sft_v1/sft_val.jsonl
build/data_build/safeanywhere_sft_v1/train_lf_v1_spanmasked.jsonl
build/data_build/safeanywhere_sft_v1/val_lf_v1_spanmasked.jsonl
build/data_build/safeanywhere_sft_v1/report.json
```

## 2. SFT

普通 cold-start SFT：

```bash
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml
```

输出：`runs/sft/qwen3_0_6b_v1/`

special-token cold-start SFT：

```bash
uv run --extra opsd python scripts/sft/add_safety_think_special_tokens.py \
  --base-model ../models/Qwen3-0.6B \
  --output runs/sft_special/qwen3_0_6b_safety_think_base

USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1_special_tokens.yaml
```

输出：

```text
runs/sft_special/qwen3_0_6b_safety_think_base/
runs/sft_special/qwen3_0_6b_v1/
```

## 3. Merge

普通 SFT merge：

```bash
uv run --extra opsd python scripts/opsd/merge_lora.py \
  --base-model ../models/Qwen3-0.6B \
  --adapter runs/sft/qwen3_0_6b_v1 \
  --output runs/merged/qwen3_0_6b_sft_v1
```

special-token SFT merge：

```bash
uv run --extra opsd python scripts/opsd/merge_lora.py \
  --base-model runs/sft_special/qwen3_0_6b_safety_think_base \
  --adapter runs/sft_special/qwen3_0_6b_v1 \
  --output runs/merged/qwen3_0_6b_sft_special_v1
```

## 4. OPSD

普通 OPSD：

```bash
uv run --extra opsd python scripts/opsd/run_opsd.py \
  --config configs/opsd/safechain_qwen3_0_6b.yaml
```

输出：`runs/opsd/qwen3_0_6b_opsd_v1/`

special-token OPSD：

```bash
uv run --extra opsd python scripts/opsd/run_opsd.py \
  --config configs/opsd/safechain_qwen3_0_6b_special_tokens.yaml
```

输出：`runs/opsd/qwen3_0_6b_opsd_special_v1/`

## 5. Eval

```bash
uv run --extra opsd python scripts/eval/run_eval_matrix.py \
  --config configs/eval/safeanywhere_v1.yaml
```

评测配置中的模型目标由 `configs/eval/safeanywhere_v1.yaml` 管理，默认包含 base、SFT 和 OPSD checkpoints。

## Notes

- `scripts/sft/add_safety_think_special_tokens.py` 会将 `<safety_think>` 和 `</safety_think>` 注册为 special tokens，并 resize embedding。
- special-token 分支后续 merge/OPSD 必须使用 `runs/sft_special/qwen3_0_6b_safety_think_base` 或其 merged checkpoint，不能混用原始 base。
- OPSD 细节见 `docs/OPSD.md`。
