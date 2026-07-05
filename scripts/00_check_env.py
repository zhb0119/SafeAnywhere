from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.io_utils import load_dotenv, read_config, resolve_cli_path, resolve_config_paths  # noqa: E402
from safeanywhere.sampling import load_safechain_pool  # noqa: E402

load_dotenv(ROOT / ".env")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SafeAnywhere config, data path, deps, and teacher env.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-api", action="store_true")
    args = parser.parse_args()

    config_path = resolve_cli_path(args.config, ROOT)
    config = resolve_config_paths(read_config(config_path), ROOT)
    source = Path(config["paths"]["safechain_jsonl"])
    per_label = config["sampling"]["per_label"]
    teacher = config["teacher"]
    api_key_env = teacher["api_key_env"]

    errors: list[str] = []
    if not source.exists():
        errors.append(f"missing_source:{source}")
    if not has_module("yaml"):
        errors.append("missing_package:PyYAML")
    if not has_module("openai"):
        errors.append("missing_package:openai")
    if args.require_api and not os.environ.get(api_key_env):
        errors.append(f"missing_env:{api_key_env}")

    available = {}
    if source.exists():
        pools = load_safechain_pool(source)
        available = {label: len(pools.get(label, [])) for label in per_label}
        for label, n in per_label.items():
            if available.get(label, 0) < int(n):
                errors.append(f"not_enough_samples:{label}:{available.get(label, 0)}/{n}")

    report = {
        "ok": not errors,
        "config": str(config_path),
        "source_exists": source.exists(),
        "available_by_label": available,
        "requested_by_label": per_label,
        "teacher": {
            "provider": teacher.get("provider"),
            "model": os.environ.get(teacher["model_env"], teacher.get("default_model")),
            "base_url": os.environ.get(teacher["base_url_env"], teacher.get("default_base_url")),
            "api_key_present": bool(os.environ.get(api_key_env)),
        },
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
