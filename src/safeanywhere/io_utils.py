from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported config suffix: {path.suffix}")


def resolve_cli_path(path: str | Path, project_root: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (Path(project_root) / path).resolve()


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()


def resolve_config_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        return config
    resolved = dict(paths)
    for key, value in paths.items():
        if isinstance(value, str) and value:
            resolved[key] = str(resolve_project_path(value, project_root))
    return {**config, "paths": resolved}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            yield obj


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


class JsonlAppender:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._f = self.path.open("a", encoding="utf-8", newline="\n")

    def write(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def load_dotenv(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or not os.environ[key]):
            os.environ[key] = value
