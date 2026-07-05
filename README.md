# SafeAnywhere

SafeAnywhere is a compact data-construction pipeline for building safety-think SFT pilots from SafeChain-style samples. The final SFT files stay in ordinary `prompt` / `response` format, while selected assistant responses may contain a brief `<safety_think>...</safety_think>` block.

## Repository Layout

```text
.
  configs/             # Smoke and pilot dataset configs
  scripts/             # CLI entry points for environment checks and dataset builds
  src/safeanywhere/    # Sampling, teacher prompting, validation, and export logic
  docs/dataset/        # Dataset design notes and prompt specs
  docs/method/         # Method proposals and condensed notes
  build/               # Generated outputs, ignored by Git
  .env.example         # Local environment template
  pyproject.toml       # Python package metadata
```

The repository root is now the Python project root. There is no tracked nested `code/` project.

## Installation

```powershell
cd D:\paper\SafeAnywhere
uv sync
```

For real DeepSeek/OpenAI-compatible teacher calls, create a local `.env` file from `.env.example` and set:

```text
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

`.env` is ignored and must not be committed.

## Run

Check the environment and config:

```powershell
uv run python scripts/00_check_env.py --config configs/safechain_smoke_10.yaml --require-api
```

Build a mock smoke dataset without API calls:

```powershell
uv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --mock
```

Build with the real teacher endpoint:

```powershell
uv run python scripts/01_build_dataset.py --config configs/safechain_smoke_10.yaml --workers 2
```

The scripts show a tqdm progress bar by default. Use `--workers` to control concurrent API calls; start with `2` and only increase after the provider is stable. Add `--quiet` to disable the progress bar.

## Failure Handling

API empty responses, JSON parse failures, and safety-block validation failures are excluded from the training set. The builder will:

1. Write failed samples to `failed.jsonl` with `id`, `label`, `instruction`, and `error`.
2. Draw same-label replacements from the SafeChain pool.
3. Continue until each label reaches the configured target count or `sampling.max_replacements` is exhausted.

This keeps label counts balanced and avoids silently producing short datasets after individual failures.

## Teacher Prompt Routing

To reduce empty JSON outputs on simple benign samples, teacher prompts are routed by sample type:

```text
vanilla_benign + requires_safety_think=false
  -> short JSON prompt requiring only {"response": "..."}

adversarial_benign + requires_safety_think=false
  -> short JSON prompt emphasizing safe benign interpretation and no <safety_think>

requires_safety_think=true or harmful/adversarial_harmful
  -> full safety-block schema prompt
```

This keeps ordinary benign samples away from the longer safety schema while preserving strict format control for higher-risk samples.

## Safety Block Schema

`<safety_think>` uses a semi-controlled schema:

```text
risk={short_free_text}; trigger={short_free_text}; intent={intent}; decision={decision}; plan={short_free_text}
```

Small-vocabulary fields:

```text
intent: benign, ambiguous, dual_use, harmful, jailbreak
decision: answer, clarify, limit, refuse, recover
```

`risk`, `trigger`, and `plan` must be non-empty, compact free-text phrases. The validator canonicalizes common aliases, such as `harmless -> benign`, `allow -> answer`, and `decline -> refuse`, then writes the normalized response.

## Outputs

Each config writes generated files under `build/<dataset_name>/`:

```text
build/<dataset_name>/
  manifest.jsonl       # Initial samples and replacements
  annotations.jsonl    # Validated teacher outputs
  failed.jsonl         # Failed samples, only when failures exist
  sft_train.jsonl      # Final SFT training split
  sft_val.jsonl        # Final SFT validation split
  report.json          # Counts, failures, replacements, positions, and teacher metadata
```

`report.json` includes `safety_think_position`, which records whether safety blocks appeared at the beginning or middle of responses.

## Documentation

- Dataset design: `docs/dataset/`
- Method notes: `docs/method/`
