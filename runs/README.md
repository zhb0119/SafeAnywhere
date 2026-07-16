# SafeAnywhere Runs

`runs/` stores local model artifacts and training logs. Keep outputs grouped by
stage:

```text
runs/
  sft/          Cold-start SFT LoRA adapters.
  sft_special/ Special-token base checkpoints and SFT LoRA adapters.
  merged/       Merged HF checkpoints used as downstream bases.
  opsd/         OPSD LoRA adapters, checkpoints, and logs.
```

Use `sft` only for SFT artifacts or SFT merged checkpoints. OPSD outputs should
use `opsd` in the run name, for example `runs/opsd/qwen3_0_6b_opsd_v1`.
