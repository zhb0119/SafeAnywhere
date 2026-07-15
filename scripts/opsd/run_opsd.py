from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.opsd.config import resolve_opsd_config  # noqa: E402
from safeanywhere.opsd.trainer import OpsdTrainer  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/opsd/safechain_qwen3_0_6b.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SafeAnywhere prompt-level OPSD on SafeChain.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data/prompt wiring without loading a model.")
    args = parser.parse_args()

    config = resolve_opsd_config(args.config)
    trainer = OpsdTrainer(config)
    if args.dry_run:
        print(json.dumps(trainer.dry_run_report(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
