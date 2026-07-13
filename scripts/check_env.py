from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "scripts/utils/check_env.py"), run_name="__main__")
