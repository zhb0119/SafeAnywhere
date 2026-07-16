from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "build/data_build/eval/external/commands"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def make_lm_eval_commands(base_model: str, adapter: str, output_dir: str, tasks: str) -> str:
    base_out = f"{output_dir}/lm_eval/base"
    sft_out = f"{output_dir}/lm_eval/sft"
    return f"""
cd /root/workspace
git clone https://github.com/EleutherAI/lm-evaluation-harness.git || true
cd lm-evaluation-harness
python -m pip install -e ".[hf]"

lm_eval \\
  --model hf \\
  --model_args pretrained={base_model},dtype=bfloat16,trust_remote_code=True \\
  --tasks {tasks} \\
  --batch_size auto \\
  --output_path {base_out}

lm_eval \\
  --model hf \\
  --model_args pretrained={base_model},peft={adapter},dtype=bfloat16,trust_remote_code=True \\
  --tasks {tasks} \\
  --batch_size auto \\
  --output_path {sft_out}
"""


def make_merge_lora_command(base_model: str, adapter: str, merged_model: str) -> str:
    code = f"""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = Path({base_model!r}).resolve()
adapter = Path({adapter!r}).resolve()
out = Path({merged_model!r}).resolve()

tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
tokenizer.save_pretrained(out)
print(out)
"""
    return f"""
cd /root/workspace/SafeAnywhere
python - <<'PY'
{code.strip()}
PY
"""


def make_harmbench_commands(merged_model: str, output_dir: str) -> str:
    model_name = "safeanywhere_sft"
    yaml = f"""

{model_name}:
  model:
    model_name_or_path: {merged_model}
    use_fast_tokenizer: True
    dtype: bfloat16
    trust_remote_code: True
    chat_template: qwen
    eos_token: <|im_end|>
  num_gpus: 1
  model_type: open_source
"""
    return f"""
cd /root/workspace
git clone https://github.com/centerforaisafety/HarmBench.git || true
cd HarmBench
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

cat >> configs/model_configs/models.yaml <<'YAML'
{yaml.rstrip()}
YAML

python ./scripts/run_pipeline.py \\
  --methods DirectRequest,ZeroShot \\
  --models {model_name} \\
  --step all \\
  --mode local \\
  --behaviors_path ./data/behavior_datasets/harmbench_behaviors_text_all.csv \\
  --base_save_dir {output_dir}/harmbench/sft \\
  --cls_path cais/HarmBench-Llama-2-13b-cls
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Write shell command files for external benchmark runners.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="/root/workspace/models/Qwen3-0.6B")
    parser.add_argument("--adapter", default="/root/workspace/SafeAnywhere/runs/sft/qwen3_0_6b_v1")
    parser.add_argument("--merged-model", default="/root/workspace/SafeAnywhere/runs/merged/qwen3_0_6b_sft_v1")
    parser.add_argument("--eval-output-dir", default="/root/workspace/SafeAnywhere/build/data_build/eval/external")
    parser.add_argument("--lm-eval-tasks", default="mmlu,ifeval,gsm8k,arc_challenge,hellaswag,truthfulqa_mc2")
    args = parser.parse_args()

    commands = {
        "merge_lora.sh": make_merge_lora_command(args.base_model, args.adapter, args.merged_model),
        "lm_eval.sh": make_lm_eval_commands(args.base_model, args.adapter, args.eval_output_dir, args.lm_eval_tasks),
        "harmbench.sh": make_harmbench_commands(args.merged_model, args.eval_output_dir),
    }
    for filename, text in commands.items():
        path = args.output_dir / filename
        write(path, text)
    report = {
        "output_dir": str(args.output_dir),
        "files": {filename: str(args.output_dir / filename) for filename in commands},
        "base_model": args.base_model,
        "adapter": args.adapter,
        "merged_model": args.merged_model,
    }
    write(args.output_dir / "report.json", json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
