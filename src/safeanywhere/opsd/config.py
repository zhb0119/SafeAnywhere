from __future__ import annotations

from pathlib import Path
from typing import Any

from safeanywhere.io_utils import read_config, resolve_project_path


ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def resolve_opsd_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = resolve_project_path(path, ROOT)
    config = read_config(path)
    if not isinstance(config, dict):
        raise ValueError(f"OPSD config must be a mapping: {config_path}")
    config["_config_path"] = str(path)
    return config


def resolve_config_path(value: str | Path, *, required: bool = False) -> Path:
    path = resolve_project_path(value, ROOT)
    if required and not path.exists():
        raise FileNotFoundError(f"Path not found: {repo_relative(path)}")
    return path
