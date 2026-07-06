from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "build/eval/external"

BENCHMARK_FILES = {
    "advbench": "advbench/advbench_eval.jsonl",
    "xstest": "xstest/xstest_eval.jsonl",
    "jailbreakbench": "jailbreakbench/jbb_eval.jsonl",
}


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def bench_paths(output_dir: Path, benchmark: str, run_name: str) -> tuple[Path, Path, Path]:
    bench_dir = output_dir / benchmark
    return (
        bench_dir / f"{run_name}_predictions.jsonl",
        bench_dir / f"{run_name}_predictions_scored.jsonl",
        bench_dir / f"{run_name}_score_summary.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generation + heuristic scoring on prepared external benchmarks.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["advbench", "xstest", "jailbreakbench"],
        choices=sorted(BENCHMARK_FILES),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="../models/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--run-name", default=None, help="Defaults to 'sft' when adapter is set, otherwise 'base'.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_name = args.run_name or ("sft" if args.adapter else "base")
    for benchmark in args.benchmarks:
        eval_file = args.output_dir / BENCHMARK_FILES[benchmark]
        if not eval_file.exists():
            raise FileNotFoundError(f"Eval file not found. Run scripts/10_prepare_external_benchmarks.py first: {eval_file}")
        predictions, scored, summary = bench_paths(args.output_dir, benchmark, run_name)
        gen_cmd = [
            sys.executable,
            str(ROOT / "scripts/07_generate_eval_responses.py"),
            "--eval-file",
            str(eval_file),
            "--base-model",
            args.base_model,
            "--output",
            str(predictions),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
        ]
        if args.adapter:
            gen_cmd.extend(["--adapter", args.adapter])
        if args.limit is not None:
            gen_cmd.extend(["--limit", str(args.limit)])
        score_cmd = [
            sys.executable,
            str(ROOT / "scripts/08_score_eval_results.py"),
            "--input",
            str(predictions),
            "--scored-output",
            str(scored),
            "--summary-output",
            str(summary),
        ]
        run(gen_cmd, args.dry_run)
        run(score_cmd, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
