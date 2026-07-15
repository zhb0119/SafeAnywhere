# SafeAnywhere

SafeAnywhere 构建 safety-think SFT 数据，用于训练模型在危险前缀或越狱式 assistant prefill 后恢复到安全续写。

## 数据快照

当前快照：`build/data_build/safeanywhere_sft_v1/`

| 数据 | train | val | total |
| --- | ---: | ---: | ---: |
| `safechain` | 900 | 100 | 1000 |
| `harmful_prefix` | 607 | 67 | 674 |
| **合并 SFT** | **1507** | **167** | **1674** |

`harmful_prefix` 来源：

- `hex_phi`：600 条
- `safechain_harmful`：74 条

说明：当前本地 `safechain_harmful/annotations.jsonl` 只有 74 条；配置目标是 300 条。复用已有数据时不会补打 API。

## Loss Mask

- `safechain`：`user=0 / assistant=1`
- `harmful_prefix`：`user=0 / assistant_prefill=0 / recovery_target=1`
- LLaMA-Factory 导出使用 content span 的 `loss_weight` 保留上述 mask。

顶层 `harmful_prefix/annotations.jsonl` 和 `harmful_prefix/failed.jsonl` 不保留；原始文件只放在各 source 子目录。

## CLI

安装依赖：

```bash
cd /root/workspace/SafeAnywhere
uv sync --frozen
```

复用已有 annotations，重新导出 SFT 和 LLaMA-Factory 数据（不调用 API）：

```bash
uv run python scripts/build_sft_dataset.py \
  --config configs/data_build/safeanywhere_sft_v1.yaml \
  --export-existing-only \
  --quiet
```

从源数据重建 annotations、SFT 和导出文件（会调用 teacher API）：

```bash
uv run python scripts/build_sft_dataset.py \
  --config configs/data_build/safeanywhere_sft_v1.yaml \
  --workers 1 \
  --quiet
```

校验 LLaMA-Factory span mask：

```bash
python3 scripts/data/validate_llamafactory_masks.py \
  --train build/data_build/safeanywhere_sft_v1/train_lf_v1_spanmasked.jsonl \
  --val build/data_build/safeanywhere_sft_v1/val_lf_v1_spanmasked.jsonl
```

## SFT

复制 LLaMA-Factory template：

```bash
cp integrations/llamafactory/templates/safeanywhere_qwen3_nothink.py \
  /root/workspace/LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/templates/
```

启动训练：

```bash
USE_V1=1 PYTHONPATH=/root/workspace/LLaMA-Factory/src \
  llamafactory-cli sft configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml
```

如基础模型路径不同，修改 `configs/sft/llamafactory/qwen3_lora_sft_safeanywhere_sft_v1.yaml`。

## OPSD 自蒸馏

安装 OPSD 训练依赖：

```bash
cd /root/workspace/SafeAnywhere
uv sync --extra opsd
```

检查配置和数据：

```bash
uv run --extra opsd python scripts/opsd/run_opsd.py \
  --config configs/opsd/safechain_qwen3_0_6b.yaml \
  --dry-run
```

修改 `configs/opsd/safechain_qwen3_0_6b.yaml`：

```yaml
model:
  path: /path/to/merged_sft_hf_checkpoint
train:
  output_dir: runs/opsd/qwen3_safeanywhere_opsd_v1
```

启动训练：

```bash
uv run --extra opsd python scripts/opsd/run_opsd.py \
  --config configs/opsd/safechain_qwen3_0_6b.yaml
```

默认配置训练 `1000` 个 optimizer steps，输出到 `train.output_dir`。
更多说明见 `docs/OPSD.md`。

## 主要产物

```text
build/data_build/safeanywhere_sft_v1/sft_train.jsonl
build/data_build/safeanywhere_sft_v1/sft_val.jsonl
build/data_build/safeanywhere_sft_v1/train_lf_v1_spanmasked.jsonl
build/data_build/safeanywhere_sft_v1/val_lf_v1_spanmasked.jsonl
build/data_build/safeanywhere_sft_v1/report.json
```
