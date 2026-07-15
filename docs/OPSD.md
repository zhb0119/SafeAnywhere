# SafeAnywhere OPSD v1

This module implements a first-pass, prompt-level on-policy self-distillation
loop inside the SafeAnywhere repo. It does not depend on OPSA, NeMo-RL, Ray, or
vLLM.

## Scope

Supported in v1:

- SafeChain prompt-level OPSD.
- One local HF causal LM used as both student and prompt-elicited teacher.
- Teacher distribution is computed with a label-specific safety system prompt.
- Loss is computed only on the continuation sampled from the current student.
- LoRA OPSD by default, with optional full-parameter mode.

Not supported in v1:

- Dangerous-prefix recovery masks.
- Distributed Ray/vLLM rollout.
- EMA or frozen teacher refresh schedules.

## Algorithm

For each SafeChain prompt:

1. Render the student prompt as the normal user prompt.
2. Sample a continuation from the current model.
3. Render the teacher prompt as `system=safety_prompt(label) + user prompt`.
4. Append the exact same sampled continuation token ids to both prompts.
5. Compute teacher logits with `no_grad`.
6. Train the student logits toward the teacher logits with KL over continuation tokens.

This keeps the training distribution on-policy while using the same SFT model's
prompt-elicited safe behavior as the teacher.

## Environment

Default data/eval dependencies remain unchanged:

```bash
cd /root/workspace/SafeAnywhere
uv sync --frozen
```

OPSD training needs optional heavy dependencies:

```bash
uv sync --extra opsd
```

If you use a CUDA-specific PyTorch wheel, install the correct PyTorch build for
your machine before running OPSD.

## Dry Run

Dry-run validates config, data loading, label sampling, and teacher prompts
without importing or loading a model:

```bash
uv run python scripts/opsd/run_opsd.py \
  --config configs/opsd/safechain_qwen3_0_6b.yaml \
  --dry-run
```

## Training

Edit `configs/opsd/safechain_qwen3_0_6b.yaml`:

- `model.path`: base HF checkpoint when using `adapter_path`, or a merged SFT
  checkpoint when `adapter_path` is empty.
- `model.train_mode`: `lora` or `full`; the default config uses `lora`.
- `model.tokenizer_path`: optional tokenizer path; defaults to `model.path`.
- `model.adapter_path`: optional LoRA/PEFT adapter path. In `lora` mode this
  is loaded as trainable, so it can point to the cold-start SFT LoRA adapter.
- `model.lora`: LoRA rank, alpha, dropout, target modules, and bias settings
  used when `train_mode: lora` and `adapter_path` is empty.
- `train.output_dir`: where OPSD checkpoints and logs should be written.

Then run:

```bash
uv run python scripts/opsd/run_opsd.py \
  --config configs/opsd/safechain_qwen3_0_6b.yaml
```

Outputs:

```text
runs/opsd/qwen3_safeanywhere_opsd_lora_v1/
  resolved_config.json
  train_log.jsonl
  rollout_samples.jsonl
  checkpoint-step-*/
  checkpoint-final/
```

## Notes

- `chat_template: qwen3_nothink` manually renders the same no-think ChatML shape
  used by the SafeAnywhere LLaMA-Factory template.
- `loss.top_k` can reduce memory, but full-vocab KL is the cleaner first
  baseline for small models.
- Keep balanced `data.label_ratios` unless intentionally running a safety-only
  ablation; otherwise benign utility can regress through over-refusal.
