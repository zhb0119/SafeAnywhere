from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.io_utils import read_config, resolve_cli_path  # noqa: E402


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def task_limits(config: dict[str, Any]) -> str | None:
    per_task = nested(config, "eval_set", "sampling", "per_task")
    if not isinstance(per_task, dict):
        return None
    parts = []
    for task, limit in per_task.items():
        if limit is None or limit == "":
            continue
        parts.append(f"{task}={int(limit)}")
    return ",".join(parts) if parts else None


def emit(name: str, value: Any) -> None:
    if value is None:
        return
    env_value = os.environ.get(name)
    if env_value:
        value = env_value
    elif isinstance(value, bool):
        value = "1" if value else "0"
    else:
        value = str(value)
    print(f"{name}={shlex.quote(str(value))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit shell variables from a SafeAnywhere eval config.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = read_config(resolve_cli_path(args.config, ROOT))
    mapping = {
        "EVAL_DIR": nested(config, "paths", "eval_dir"),
        "EVAL_INPUT": nested(config, "paths", "eval_input"),
        "EVAL_SET_SUBDIR": nested(config, "paths", "eval_set_subdir"),
        "RUNS_SUBDIR": nested(config, "paths", "runs_subdir"),
        "COMPARISONS_SUBDIR": nested(config, "paths", "comparisons_subdir"),
        "BASE_MODEL": nested(config, "models", "base_model"),
        "CANDIDATE_ADAPTER": nested(config, "models", "candidate", "adapter"),
        "CANDIDATE_NAME": nested(config, "models", "candidate", "name"),
        "BASELINE_ADAPTER": nested(config, "models", "baseline", "adapter"),
        "BASELINE_NAME": nested(config, "models", "baseline", "name"),
        "MAX_PER_TASK": first_present(nested(config, "eval_set", "sampling", "max_per_task"), nested(config, "eval_set", "max_per_task")),
        "TASK_LIMITS": task_limits(config),
        "INCLUDE_PREFIX_DIRECT": nested(config, "eval_set", "include_prefix_direct"),
        "MAX_NEW_TOKENS": nested(config, "generation", "max_new_tokens"),
        "TEMPERATURE": nested(config, "generation", "temperature"),
        "TOP_P": nested(config, "generation", "top_p"),
        "THINKING_MODE": nested(config, "generation", "thinking_mode"),
        "DTYPE": nested(config, "generation", "dtype"),
        "DEVICE_MAP": nested(config, "generation", "device_map"),
        "GENERATION_LIMIT": nested(config, "generation", "limit"),
        "SYSTEM_PROMPT": nested(config, "generation", "system_prompt"),
        "OUTPUT_MODE": nested(config, "generation", "output_mode"),
        "INCLUDE_REFERENCE": nested(config, "generation", "include_reference"),
        "INCLUDE_RAW_PREDICTION": nested(config, "generation", "include_raw_prediction"),
        "INCLUDE_RENDERED_PROMPT": nested(config, "generation", "include_rendered_prompt"),
        "JUDGE_MODEL": nested(config, "judge", "model"),
        "JUDGE_API_KEY_ENV": nested(config, "judge", "api_key_env"),
        "JUDGE_BASE_URL_ENV": nested(config, "judge", "base_url_env"),
        "JUDGE_MODEL_ENV": nested(config, "judge", "model_env"),
        "JUDGE_MAX_FIELD_CHARS": nested(config, "judge", "max_field_chars"),
    }
    for name, value in mapping.items():
        emit(name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
