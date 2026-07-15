from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from safeanywhere.io_utils import ensure_dir, read_config, read_jsonl, write_json, write_jsonl  # noqa: E402
from safeanywhere.export import make_sft_rows as make_cold_start_sft_rows, split_train_val as split_cold_start_train_val  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/data_build/safeanywhere_sft_v1.yaml"
PROMPT_WRAPPER_NEEDLES = [
    "You are SafeAnywhere, a helpful assistant",
    "Rules for <safety_think>:",
    "Current data type: dangerous_prefix.",
    "Already generated assistant prefill, context only and not part of the training target:",
    "Now output the safe continuation JSON.",
    "Return one valid JSON object only:",
]
PREFIX_SOURCE_BUILDERS = {
    "hex_phi": {
        "script": "scripts/data/build_hex_phi_prefix.py",
        "path_key": "hex_phi_jsonl",
        "output_subdir": "hex_phi",
        "source_kind": "harmful_hex_phi",
    },
    "safechain_harmful": {
        "script": "scripts/data/build_safechain_prefix.py",
        "path_key": "safechain_jsonl",
        "output_subdir": "safechain_harmful",
        "source_kind": "safechain_harmful_prefix",
    },
}
BUILD_TASKS = ("safechain", "harmful_prefix")
GENERATED_CONFIG_SUFFIX = ".yaml"


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_build_config_path(value: str | Path) -> Path:
    path = repo_path(value)
    if path.exists():
        return path

    raw = Path(value).expanduser()
    if not raw.is_absolute() and raw.parts and raw.parts[0] == "configs" and len(raw.parts) == 2:
        migrated = ROOT / "configs" / "data_build" / raw.name
        if migrated.exists():
            print(f"Config moved: {raw} -> {repo_relative(migrated)}", flush=True)
            return migrated.resolve()

    raise FileNotFoundError(f"Config not found: {repo_relative(path)}")


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_jsonl_list(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def count_masks(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("attack_type") != "dangerous_prefix":
            continue
        messages = row.get("messages")
        if isinstance(messages, list):
            counts["/".join(str(message.get("loss_mask")) for message in messages)] += 1
    return dict(counts)


def format_cmd(cmd: list[str]) -> str:
    parts = []
    for part in cmd:
        if "\n" in part:
            escaped = part.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
            parts.append(f"$'{escaped}'")
        else:
            parts.append(shlex.quote(part))
    return " ".join(parts)


def run_command(name: str, cmd: list[str], log_file: Path, verbose: bool) -> None:
    print(name, flush=True)
    print(f"  $ {format_cmd(cmd)}", flush=True)
    if verbose:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return

    ensure_dir(log_file.parent)
    with log_file.open("a", encoding="utf-8") as out:
        out.write(f"\n\n# {name}\n")
        out.write(f"$ {format_cmd(cmd)}\n")
        out.flush()
        result = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {format_cmd(cmd)}. See log: {repo_relative(log_file)}")


def selected_build_tasks(config: dict[str, Any]) -> list[str]:
    raw = config.get("pipeline", {}).get("tasks", list(BUILD_TASKS))
    if isinstance(raw, str) and raw in {"all", "*"}:
        return list(BUILD_TASKS)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("pipeline.tasks must be a non-empty list, 'all', or '*'.")
    tasks = list(dict.fromkeys(str(task) for task in raw))
    unknown = sorted(set(tasks) - set(BUILD_TASKS))
    if unknown:
        raise ValueError(f"Unsupported pipeline tasks: {unknown}. Supported tasks: {list(BUILD_TASKS)}")
    return tasks


def should_finalize(config: dict[str, Any], tasks: list[str]) -> bool:
    pipeline = config.get("pipeline", {})
    if "finalize" in pipeline:
        return bool(pipeline["finalize"])
    return set(tasks) == set(BUILD_TASKS)


def selected_harmful_prefix_sources(config: dict[str, Any]) -> list[str]:
    sources = config["tasks"]["harmful_prefix"]["sources"]
    raw = config.get("pipeline", {}).get("harmful_prefix_sources", list(sources))
    if isinstance(raw, str) and raw in {"all", "*"}:
        return list(sources)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("pipeline.harmful_prefix_sources must be a non-empty list, 'all', or '*'.")
    selected = list(dict.fromkeys(str(source) for source in raw))
    unknown = sorted(set(selected) - set(sources))
    if unknown:
        raise ValueError(f"Unsupported harmful_prefix sources: {unknown}. Supported sources: {list(sources)}")
    unsupported = sorted(set(selected) - set(PREFIX_SOURCE_BUILDERS))
    if unsupported:
        raise ValueError(f"Unsupported harmful_prefix builders: {unsupported}")
    return selected


def configured_harmful_prefix_source_dirs(config: dict[str, Any], harmful_dir: Path) -> dict[str, Path]:
    return {
        source_name: harmful_dir / PREFIX_SOURCE_BUILDERS[source_name]["output_subdir"]
        for source_name in config["tasks"]["harmful_prefix"]["sources"]
    }


def make_prefix_sft_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "prompt": row["instruction"],
                "response": row["response"],
                "messages": row["messages"],
                "source": row["source"],
                "source_id": row["source_id"],
                "attack_type": row["attack_type"],
                "label": row["label"],
                "requires_safety_think": row["requires_safety_think"],
                "prefix_depth": row["prefix_depth"],
                "prefix_type": row["prefix_type"],
                "prefix_mode": row.get("prefix_mode"),
                "prefix_tokenish_len": row.get("prefix_tokenish_len"),
                "assistant_prefill": row["assistant_prefill"],
            }
        )
    return out


def split_prefix_train_val(rows: list[dict[str, Any]], seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    groups = list(by_source.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_val = int(round(len(rows) * val_ratio))
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for group in groups:
        if len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)
    return train, val


def child_config_paths(generated_config_dir: Path) -> list[Path]:
    if not generated_config_dir.exists():
        return []
    return sorted(path for path in generated_config_dir.glob(f"*{GENERATED_CONFIG_SUFFIX}") if path.is_file())


def child_output_dir_from_config(config: dict[str, Any]) -> Path:
    return repo_path(config["paths"]["output_dir"])


def export_child_sft_split(config_path: Path) -> dict[str, Any] | None:
    config = read_config(config_path)
    output_dir = child_output_dir_from_config(config)
    annotations_path = output_dir / "annotations.jsonl"
    if not annotations_path.exists():
        return None

    rows = read_jsonl_list(annotations_path)
    seed = int(config["seed"])
    val_ratio = float(config.get("sampling", {}).get("val_ratio", config.get("split", {}).get("val_ratio", 0.1)))

    sampling = config.get("sampling", {})
    dataset_name = str(config.get("dataset_name", ""))
    is_prefix = (
        "prefix_mode" in sampling
        or "prefix_type_counts" in sampling
        or "harmful_prefix" in dataset_name
    )
    if is_prefix:
        sft_rows = make_prefix_sft_rows(rows, config)
        train, val = split_prefix_train_val(sft_rows, seed, val_ratio)
        data_type = "harmful_prefix"
    elif "per_label" in sampling:
        sft_rows = make_cold_start_sft_rows(rows, config)
        train, val = split_cold_start_train_val(sft_rows, seed, val_ratio)
        data_type = "safechain"
    else:
        return None

    write_jsonl(output_dir / "sft_train.jsonl", train)
    write_jsonl(output_dir / "sft_val.jsonl", val)

    return {
        "config": repo_relative(config_path),
        "output_dir": repo_relative(output_dir),
        "data_type": data_type,
        "rows": len(rows),
        "train": len(train),
        "val": len(val),
    }


def export_all_child_sft_splits(generated_config_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for config_path in child_config_paths(generated_config_dir):
        report = export_child_sft_split(config_path)
        if report is not None:
            reports.append(report)
    return reports


def prefix_source_name_from_config_path(config_path: Path) -> str:
    stem = config_path.stem
    if stem.startswith("harmful_prefix_"):
        return stem.removeprefix("harmful_prefix_")
    return stem


def validate_harmful_prefix_sampling(config: dict[str, Any], source_names: list[str]) -> None:
    for source_name in source_names:
        source = config["tasks"]["harmful_prefix"]["sources"][source_name]
        total = int(source["total"])
        depth_total = sum(int(value) for value in source["depth_counts"].values())
        if depth_total != total:
            raise ValueError(
                f"tasks.harmful_prefix.sources.{source_name}.depth_counts sum {depth_total} does not match total {total}"
            )
        if "prefix_type_counts" in source:
            type_total = sum(int(value) for value in source["prefix_type_counts"].values())
            if type_total != total:
                raise ValueError(
                    f"tasks.harmful_prefix.sources.{source_name}.prefix_type_counts sum {type_total} does not match total {total}"
                )


def child_common_args(output_dir: Path, args: argparse.Namespace) -> list[str]:
    out: list[str] = ["--workers", str(args.workers)]
    if args.mock:
        out.append("--mock")
    if args.quiet:
        out.append("--quiet")
    if args.resume:
        if (output_dir / "manifest.jsonl").exists():
            out.append("--resume")
        else:
            print(f"  resume requested, but {repo_relative(output_dir / 'manifest.jsonl')} is missing; starting fresh.", flush=True)
    return out


def common_teacher(config: dict[str, Any], *, prefix: bool) -> dict[str, Any]:
    teacher = dict(config["teacher"])
    if prefix:
        teacher.update(config.get("prefix_teacher", {}))
    return teacher


def build_safechain_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    task = config["tasks"]["safechain"]
    return {
        "dataset_name": f"{config['dataset_name']}_safechain",
        "seed": int(config["seed"]),
        "paths": {
            "safechain_jsonl": repo_relative(repo_path(config["paths"]["safechain_jsonl"])),
            "output_dir": repo_relative(output_dir),
        },
        "sampling": {
            "per_label": task["per_label"],
            "safety_think_probability": task["safety_think_probability"],
            "val_ratio": float(config.get("split", {}).get("val_ratio", 0.1)),
            "max_replacements": int(task.get("max_replacements", 100)),
        },
        "teacher": common_teacher(config, prefix=False),
        "safety_block": config["safety_block"],
    }


def build_prefix_source_config(config: dict[str, Any], source_name: str, output_dir: Path) -> dict[str, Any]:
    source = config["tasks"]["harmful_prefix"]["sources"][source_name]
    builder = PREFIX_SOURCE_BUILDERS[source_name]
    path_key = builder["path_key"]
    prefix_task = config["tasks"]["harmful_prefix"]
    sampling = {
        "total": int(source["total"]),
        "depth_counts": source["depth_counts"],
        "val_ratio": float(source.get("val_ratio", config.get("split", {}).get("val_ratio", 0.1))),
        "max_replacements": int(source.get("max_replacements", prefix_task.get("max_replacements", 200))),
    }
    if "prefix_type_counts" in source:
        sampling["prefix_type_counts"] = source["prefix_type_counts"]
    for key in ("prefix_mode",):
        if key in source:
            sampling[key] = source[key]
        elif key in prefix_task:
            sampling[key] = prefix_task[key]
    return {
        "dataset_name": f"{config['dataset_name']}_harmful_prefix_{source_name}",
        "seed": int(config["seed"]),
        "paths": {
            path_key: repo_relative(repo_path(config["paths"][path_key])),
            "output_dir": repo_relative(output_dir),
        },
        "sampling": sampling,
        "teacher": common_teacher(config, prefix=True),
        "safety_block": config["safety_block"],
        "validation": config.get("validation", {}),
    }


def combine_jsonl(paths: list[Path], output_path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        rows.extend(read_jsonl_list(path))
    if missing and not missing_ok:
        raise FileNotFoundError(f"Missing required JSONL files: {[repo_relative(path) for path in missing]}")
    if rows:
        write_jsonl(output_path, rows)
    else:
        output_path.unlink(missing_ok=True)
    return rows


def validate_unique_ids(rows: list[dict[str, Any]], label: str) -> None:
    counts = Counter(str(row.get("id")) for row in rows)
    duplicates = sorted(row_id for row_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate ids in {label}: {duplicates[:10]}")


def validate_train_val_no_source_leak(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> list[str]:
    def source_key(row: dict[str, Any]) -> str:
        source = row.get("source") or row.get("attack_type") or "unknown"
        source_id = row.get("source_id") or row.get("id")
        return f"{source}:{source_id}"

    train_keys = {source_key(row) for row in train}
    val_keys = {source_key(row) for row in val}
    return sorted(train_keys & val_keys)


def validate_prompt_leaks(paths: list[Path], needles: list[str]) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line_no, row in enumerate(read_jsonl(path), start=1):
            blob = json.dumps(row, ensure_ascii=False)
            matched = [needle for needle in needles if needle in blob]
            if matched:
                leaks.append({"path": repo_relative(path), "line": line_no, "id": row.get("id"), "matched": matched})
                break
    return leaks


def build_harmful_prefix_report(
    output_dir: Path,
    source_dirs: dict[str, Path],
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = train + val
    reports = {}
    for source_name, source_dir in source_dirs.items():
        report_path = source_dir / "report.json"
        if report_path.exists():
            reports[source_name] = load_json(report_path)
    return {
        "dataset_name": output_dir.name,
        "data_type": "harmful_prefix",
        "counts": {
            "train_total": len(train),
            "val_total": len(val),
            "total": len(rows),
        },
        "by_source": count_by(rows, "source"),
        "by_label": count_by(rows, "label"),
        "by_prefix_type": count_by(rows, "prefix_type"),
        "by_prefix_mode": count_by(rows, "prefix_mode"),
        "by_prefix_depth": count_by(rows, "prefix_depth"),
        "dangerous_prefix_masks": count_masks(rows),
        "source_reports": reports,
        "files": {
            "manifest": repo_relative(output_dir / "manifest.jsonl"),
            "annotations": repo_relative(output_dir / "annotations.jsonl"),
            "failed": repo_relative(output_dir / "failed.jsonl") if (output_dir / "failed.jsonl").exists() else None,
            "sft_train": repo_relative(output_dir / "sft_train.jsonl"),
            "sft_val": repo_relative(output_dir / "sft_val.jsonl"),
            "sources": {source_name: repo_relative(path) for source_name, path in source_dirs.items()},
        },
    }


def write_dataset_card(
    output_dir: Path,
    final_report: dict[str, Any],
    harmful_report: dict[str, Any],
    validation_report: dict[str, Any],
) -> None:
    counts = final_report.get("counts", {})
    lines = [
        "# SafeAnywhere SFT v1 Dataset",
        "",
        "## Files",
        "",
        f"- train: `{repo_relative(output_dir / 'sft_train.jsonl')}`",
        f"- val: `{repo_relative(output_dir / 'sft_val.jsonl')}`",
        f"- LLaMA-Factory train: `{repo_relative(output_dir / 'train_lf_v1_spanmasked.jsonl')}`",
        f"- LLaMA-Factory val: `{repo_relative(output_dir / 'val_lf_v1_spanmasked.jsonl')}`",
        "",
        "## Counts",
        "",
        f"- train total: `{counts.get('train_total')}`",
        f"- val total: `{counts.get('val_total')}`",
        f"- total: `{counts.get('total')}`",
        f"- safechain train/val: `{counts.get('safechain_train')}` / `{counts.get('safechain_val')}`",
        f"- harmful-prefix train/val: `{counts.get('prefix_train')}` / `{counts.get('prefix_val')}`",
        "",
        "## Data Types",
        "",
        "- `safechain`: cold-start safety-think targets from SafeChain labels.",
        "- `harmful_prefix`: masked assistant-prefix recovery targets from HEx-PHI source excerpts and SafeChain generated prefixes.",
        "",
        "## Mask Contract",
        "",
        "- safechain rows: `user=0 / assistant_target=1`",
        "- harmful-prefix rows: `user=0 / assistant_prefill=0 / recovery_target=1`",
        "- LLaMA-Factory export joins assistant prefill and recovery target into one assistant turn with span-level loss weights.",
        "",
        "## Validation",
        "",
        f"- prompt wrapper leak count: `{validation_report.get('prompt_wrapper_leak_count')}`",
        f"- train/val source leakage count: `{validation_report.get('train_val_source_leak_count')}`",
        f"- harmful-prefix masks: `{harmful_report.get('dangerous_prefix_masks')}`",
        "",
    ]
    (output_dir / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def run_builder_steps(
    config: dict[str, Any],
    args: argparse.Namespace,
    generated_config_dir: Path,
    log_file: Path,
    tasks: list[str],
    harmful_prefix_sources: list[str],
) -> dict[str, Path]:
    python_bin = args.python_bin or sys.executable
    output_dir = repo_path(config["paths"]["output_dir"])
    safechain_dir = output_dir / "safechain"
    harmful_dir = output_dir / "harmful_prefix"
    source_dirs: dict[str, Path] = {}

    if "safechain" in tasks:
        safechain_config = build_safechain_config(config, safechain_dir)
        safechain_config_path = generated_config_dir / "safechain.yaml"
        write_yaml(safechain_config_path, safechain_config)
        if (safechain_dir / "annotations.jsonl").exists():
            print("[1/6] Skip safechain annotations (existing annotations found)", flush=True)
        else:
            run_command(
                "[1/6] Build safechain annotations",
                [
                    python_bin,
                    "scripts/data/build_safechain.py",
                    "--config",
                    repo_relative(safechain_config_path),
                    *child_common_args(safechain_dir, args),
                ],
                log_file,
                args.verbose,
            )
    else:
        print("[1/6] Skip safechain annotations", flush=True)

    if "harmful_prefix" in tasks:
        for source_name in harmful_prefix_sources:
            if source_name not in PREFIX_SOURCE_BUILDERS:
                raise ValueError(f"Unsupported harmful_prefix source: {source_name}")
            builder = PREFIX_SOURCE_BUILDERS[source_name]
            source_dir = harmful_dir / builder["output_subdir"]
            source_dirs[source_name] = source_dir
            source_config = build_prefix_source_config(config, source_name, source_dir)
            source_config_path = generated_config_dir / f"harmful_prefix_{source_name}.yaml"
            write_yaml(source_config_path, source_config)
            if (source_dir / "annotations.jsonl").exists():
                print(f"[2/6] Skip harmful_prefix/{source_name} annotations (existing annotations found)", flush=True)
            else:
                run_command(
                    f"[2/6] Build harmful_prefix/{source_name} annotations",
                    [
                        python_bin,
                        builder["script"],
                        "--config",
                        repo_relative(source_config_path),
                        *child_common_args(source_dir, args),
                    ],
                    log_file,
                    args.verbose,
                )
    else:
        print("[2/6] Skip harmful_prefix annotations", flush=True)

    return {"safechain": safechain_dir, "harmful_prefix": harmful_dir, **{f"source:{k}": v for k, v in source_dirs.items()}}


def assemble_harmful_prefix(harmful_dir: Path, source_dirs: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ensure_dir(harmful_dir)
    source_paths = list(source_dirs.values())
    combine_jsonl([path / "manifest.jsonl" for path in source_paths], harmful_dir / "manifest.jsonl")
    combine_jsonl([path / "annotations.jsonl" for path in source_paths], harmful_dir / "annotations.jsonl")
    combine_jsonl([path / "failed.jsonl" for path in source_paths], harmful_dir / "failed.jsonl", missing_ok=True)
    train = combine_jsonl([path / "sft_train.jsonl" for path in source_paths], harmful_dir / "sft_train.jsonl")
    val = combine_jsonl([path / "sft_val.jsonl" for path in source_paths], harmful_dir / "sft_val.jsonl")
    validate_unique_ids(train + val, "harmful_prefix")
    report = build_harmful_prefix_report(harmful_dir, source_dirs, train, val)
    write_json(harmful_dir / "report.json", report)
    return train, val, report


def run_merge_export_validate(
    config: dict[str, Any],
    args: argparse.Namespace,
    dirs: dict[str, Path],
    harmful_report: dict[str, Any],
    log_file: Path,
) -> dict[str, Any]:
    python_bin = args.python_bin or sys.executable
    output_dir = repo_path(config["paths"]["output_dir"])
    export_cfg = config.get("export", {})
    train_dataset_yaml = repo_path(export_cfg.get("train_dataset_yaml", "configs/sft/llamafactory/dataset_safeanywhere_sft_v1_train.yaml"))
    val_dataset_yaml = repo_path(export_cfg.get("val_dataset_yaml", "configs/sft/llamafactory/dataset_safeanywhere_sft_v1_val.yaml"))

    run_command(
        "[3/6] Merge safechain + harmful_prefix",
        [
            python_bin,
            "scripts/data/merge_sft.py",
            "--safechain-train",
            repo_relative(dirs["safechain"] / "sft_train.jsonl"),
            "--safechain-val",
            repo_relative(dirs["safechain"] / "sft_val.jsonl"),
            "--prefix-train",
            repo_relative(dirs["harmful_prefix"] / "sft_train.jsonl"),
            "--prefix-val",
            repo_relative(dirs["harmful_prefix"] / "sft_val.jsonl"),
            "--output-dir",
            repo_relative(output_dir),
            "--no-strict-counts",
        ],
        log_file,
        args.verbose,
    )

    train_rows = read_jsonl_list(output_dir / "sft_train.jsonl")
    val_rows = read_jsonl_list(output_dir / "sft_val.jsonl")
    validate_unique_ids(train_rows + val_rows, "final_sft")
    source_leaks = validate_train_val_no_source_leak(train_rows, val_rows)
    needles = config.get("validation", {}).get("prompt_wrapper_needles", PROMPT_WRAPPER_NEEDLES)
    prompt_leaks = validate_prompt_leaks(
        [
            dirs["safechain"] / "sft_train.jsonl",
            dirs["safechain"] / "sft_val.jsonl",
            dirs["harmful_prefix"] / "sft_train.jsonl",
            dirs["harmful_prefix"] / "sft_val.jsonl",
            output_dir / "sft_train.jsonl",
            output_dir / "sft_val.jsonl",
        ],
        needles,
    )
    if source_leaks:
        raise ValueError(f"Train/val source leakage detected: {source_leaks[:10]}")
    if prompt_leaks:
        raise ValueError(f"Prompt wrapper leakage detected: {prompt_leaks[:10]}")

    export_report: dict[str, Any] | None = None
    if export_cfg.get("llamafactory", True) and not args.skip_export:
        run_command(
            "[4/6] Export LLaMA-Factory span-mask JSONL",
            [
                python_bin,
                "scripts/data/export_llamafactory_v1.py",
                "--input-dir",
                repo_relative(output_dir),
                "--output-dir",
                repo_relative(output_dir),
                "--train-dataset-yaml",
                repo_relative(train_dataset_yaml),
                "--val-dataset-yaml",
                repo_relative(val_dataset_yaml),
                "--prefill-separator",
                str(export_cfg.get("prefill_separator", "\n")),
            ],
            log_file,
            args.verbose,
        )
        export_report = load_json(output_dir / "llamafactory_v1_export_report.json")

    if export_cfg.get("llamafactory", True) and not args.skip_export and not args.skip_validate:
        run_command(
            "[5/6] Validate LLaMA-Factory span-mask structure",
            [
                python_bin,
                "scripts/data/validate_llamafactory_masks.py",
                "--structure-only",
                "--train",
                repo_relative(output_dir / "train_lf_v1_spanmasked.jsonl"),
                "--val",
                repo_relative(output_dir / "val_lf_v1_spanmasked.jsonl"),
            ],
            log_file,
            args.verbose,
        )

    final_report = load_json(output_dir / "report.json")
    validation_report = {
        "prompt_wrapper_leak_count": len(prompt_leaks),
        "prompt_wrapper_leaks": prompt_leaks,
        "train_val_source_leak_count": len(source_leaks),
        "train_val_source_leaks": source_leaks,
    }
    final_report["pipeline"] = {
        "config": repo_relative(args.config_path),
        "generated_configs": repo_relative(output_dir / "configs"),
        "data_types": ["safechain", "harmful_prefix"],
        "safechain_dir": repo_relative(dirs["safechain"]),
        "harmful_prefix_dir": repo_relative(dirs["harmful_prefix"]),
        "log_file": repo_relative(log_file),
    }
    final_report["harmful_prefix"] = {
        "counts": harmful_report.get("counts"),
        "by_source": harmful_report.get("by_source"),
        "by_prefix_type": harmful_report.get("by_prefix_type"),
        "by_prefix_mode": harmful_report.get("by_prefix_mode"),
        "by_prefix_depth": harmful_report.get("by_prefix_depth"),
        "dangerous_prefix_masks": harmful_report.get("dangerous_prefix_masks"),
    }
    final_report["validation"] = validation_report
    if export_report is not None:
        final_report["llamafactory_export"] = export_report
    write_json(output_dir / "report.json", final_report)
    write_dataset_card(output_dir, final_report, harmful_report, validation_report)

    print("[6/6] Dataset ready", flush=True)
    return final_report


def auto_export_subdatasets(
    generated_config_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    export_reports = export_all_child_sft_splits(generated_config_dir)
    harmful_split_path = output_dir / "harmful_prefix"
    if not harmful_split_path.exists() and not export_reports:
        return export_reports, None

    harmful_report = None
    source_dirs: dict[str, Path] = {}
    if harmful_split_path.exists():
        for config_path in child_config_paths(generated_config_dir):
            child_config = read_config(config_path)
            if "hex_phi_jsonl" not in child_config.get("paths", {}) and "prefix_mode" not in child_config.get("sampling", {}):
                continue
            source_dir = repo_path(child_config["paths"]["output_dir"])
            if (source_dir / "annotations.jsonl").exists():
                source_dirs[prefix_source_name_from_config_path(config_path)] = source_dir
        if source_dirs:
            _, _, harmful_report = assemble_harmful_prefix(harmful_split_path, source_dirs)

    return export_reports, harmful_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SafeAnywhere SFT data with one command.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for child builder scripts.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock teacher responses.")
    parser.add_argument("--resume", action="store_true", help="Resume any subtask that already has a manifest.")
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to child builders.")
    parser.add_argument("--verbose", action="store_true", help="Print child output instead of writing only to build.log.")
    parser.add_argument("--skip-export", action="store_true", help="Skip LLaMA-Factory export.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip exported span-mask validation.")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.config_path = resolve_build_config_path(args.config)
    return args


def main() -> int:
    args = parse_args()
    config = read_config(args.config_path)
    tasks = selected_build_tasks(config)
    finalize = should_finalize(config, tasks)
    harmful_prefix_sources = selected_harmful_prefix_sources(config) if "harmful_prefix" in tasks else []
    if "harmful_prefix" in tasks:
        validate_harmful_prefix_sampling(config, harmful_prefix_sources)
    output_dir = repo_path(config["paths"]["output_dir"])
    generated_config_dir = ensure_dir(output_dir / "configs")
    log_file = output_dir / "build.log"
    if not args.verbose:
        ensure_dir(log_file.parent)
        with log_file.open("a", encoding="utf-8") as out:
            out.write("\n\n# New build_sft_dataset invocation\n")
            out.write(f"# config={repo_relative(args.config_path)} tasks={tasks} finalize={finalize}\n")
        print(f"Log: {repo_relative(log_file)}", flush=True)

    print(f"Pipeline tasks: {tasks}", flush=True)
    if harmful_prefix_sources:
        print(f"Harmful-prefix sources: {harmful_prefix_sources}", flush=True)
    print(f"Finalize mixed dataset: {finalize}", flush=True)
    dirs = run_builder_steps(config, args, generated_config_dir, log_file, tasks, harmful_prefix_sources)
    export_reports, harmful_report = auto_export_subdatasets(generated_config_dir, output_dir)
    if harmful_report is None and (output_dir / "harmful_prefix" / "report.json").exists():
        harmful_report = load_json(output_dir / "harmful_prefix" / "report.json")
    if harmful_report is None:
        raise RuntimeError("Cannot finalize without harmful_prefix report.")
    final_report = run_merge_export_validate(config, args, dirs, harmful_report, log_file)
    if export_reports:
        final_report["auto_export"] = {
            "generated_configs": repo_relative(generated_config_dir),
            "child_exports": export_reports,
        }

    print(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("Done.", flush=True)
    print(f"  dataset card: {repo_relative(output_dir / 'dataset_card.md')}", flush=True)
    print(f"  report: {repo_relative(output_dir / 'report.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
