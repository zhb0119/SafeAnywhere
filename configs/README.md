# SafeAnywhere Configs

This directory is the canonical entry point for reproducible runs.

```text
configs/
  data_build/          Dataset construction configs.
  sft/llamafactory/    LLaMA-Factory dataset and SFT training configs.
  eval/                SafeAnywhere evaluation configs.
```

Environment variables may still override eval config values for ad hoc runs.

Default eval output layout:

```text
build/data_build/eval/<name>/
  README.md
  eval_set/
    safeanywhere_eval.jsonl
    report.json
    tasks/
  runs/
    base/
    sft/
  comparisons/
```
