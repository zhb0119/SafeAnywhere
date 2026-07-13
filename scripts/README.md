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
eval/       Custom eval set building, generation, heuristic scoring, LLM judge scoring, reports.
external/   External benchmark preparation and runner command generation.
legacy/     Historical one-off workflows kept for reproducibility.
utils/      Small utility CLIs.
```

Historical numbered commands have moved into the organized implementation directories. Use the paths in this README for new runs.
