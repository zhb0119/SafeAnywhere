# Scripts

Top-level scripts are limited to public entrypoints. Implementation scripts live in the organized subdirectories below.

## Public Entrypoints

```text
build_sft_dataset.py       Build the default SafeAnywhere SFT dataset.
check_env.py               Check config, local files, and optional API access.
run_eval_comparison.sh     Run SafeAnywhere custom eval end to end.
```

## Organized Implementations

```text
data/       Dataset construction, merge, LLaMA-Factory export, mask validation.
eval/       Custom eval set building, generation, LLM judge scoring, reports.
external/   External benchmark preparation and generation helpers.
legacy/     Historical one-off workflows kept for reproducibility.
opsd/       Prompt-level OPSD training and LoRA merge helpers.
sft/        Cold-start SFT utilities, including safety-think special-token base creation and sparse-row SFT.
utils/      Small utility CLIs.
```

Current harmful-prefix data uses HEx-PHI `source_excerpt` prefixes and SafeChain harmful `generated_compliance` prefixes. The old redacted template path is retained only as a legacy-compatible mode.

Historical numbered commands have moved into the organized implementation directories. Use the paths in this README for new runs.
