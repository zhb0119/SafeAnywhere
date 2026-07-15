from __future__ import annotations

import json
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from safeanywhere.io_utils import ensure_dir, write_json
from safeanywhere.opsd.chat import render_prompt
from safeanywhere.opsd.config import resolve_config_path
from safeanywhere.opsd.data import PromptItem, SafeChainPromptPool
from safeanywhere.opsd.prompts import PromptBank


def _require_train_deps() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Missing OPSD training dependencies. Install them with `uv sync --extra opsd` "
            "or install torch/transformers manually in this environment."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _load_peft_model_if_needed(model: Any, adapter_path: str | None) -> Any:
    if not adapter_path:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError("Loading a LoRA/PEFT adapter requires `peft`. Install `uv sync --extra opsd`.") from exc
    return PeftModel.from_pretrained(model, adapter_path)


class OpsdTrainer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rng = random.Random(int(config.get("seed", 20260706)))

    def dry_run_report(self) -> dict[str, Any]:
        data_cfg = self.config["data"]
        pool = SafeChainPromptPool.from_jsonl(
            resolve_config_path(data_cfg["train_jsonl"], required=True),
            label_filter=data_cfg.get("label_filter"),
            max_samples=data_cfg.get("max_samples"),
        )
        prompts_cfg = self.config.get("teacher_prompts", {})
        prompt_path = prompts_cfg.get("path")
        bank = PromptBank.from_yaml(resolve_config_path(prompt_path, required=True)) if prompt_path else PromptBank.default()
        sample = pool.sample_batch(1, label_ratios=data_cfg.get("label_ratios"), rng=self.rng)[0]
        return {
            "mode": "dry_run",
            "config": self.config.get("_config_path"),
            "train_jsonl": str(resolve_config_path(data_cfg["train_jsonl"])),
            "rows": len(pool.items),
            "counts_by_label": pool.counts_by_label(),
            "sample": {
                "id": sample.id,
                "label": sample.label,
                "instruction": sample.instruction,
                "teacher_prompt": bank.for_label(sample.label),
            },
        }

    def train(self) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer = _require_train_deps()
        from safeanywhere.opsd.losses import distillation_kl

        cfg = self.config
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        gen_cfg = cfg.get("generation", {})
        loss_cfg = cfg.get("loss", {})

        output_dir = ensure_dir(resolve_config_path(train_cfg["output_dir"]))
        write_json(output_dir / "resolved_config.json", cfg)

        pool = SafeChainPromptPool.from_jsonl(
            resolve_config_path(data_cfg["train_jsonl"], required=True),
            label_filter=data_cfg.get("label_filter"),
            max_samples=data_cfg.get("max_samples"),
        )
        prompts_cfg = cfg.get("teacher_prompts", {})
        prompt_path = prompts_cfg.get("path")
        prompt_bank = PromptBank.from_yaml(resolve_config_path(prompt_path, required=True)) if prompt_path else PromptBank.default()

        device = model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = str(model_cfg.get("dtype", "bf16")).lower()
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_name, torch.bfloat16 if device == "cuda" else torch.float32)
        if device == "cpu":
            torch_dtype = torch.float32

        model_path = str(resolve_config_path(model_cfg["path"], required=True))
        tokenizer_value = model_cfg.get("tokenizer_path") or model_path
        tokenizer_path = str(resolve_config_path(tokenizer_value, required=True))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
        adapter_path = model_cfg.get("adapter_path")
        if adapter_path:
            adapter_path = str(resolve_config_path(adapter_path, required=True))
        model = _load_peft_model_if_needed(model, adapter_path)
        model.to(device)
        if train_cfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
        model.train()

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(train_cfg.get("learning_rate", 1e-6)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        log_path = output_dir / "train_log.jsonl"
        sample_log_path = output_dir / "rollout_samples.jsonl"
        max_steps = int(train_cfg["max_steps"])
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
        micro_batch_size = int(train_cfg.get("micro_batch_size", 1))
        max_prompt_tokens = int(train_cfg.get("max_prompt_tokens", 1536))
        max_total_tokens = int(train_cfg.get("max_total_tokens", 2048))
        log_steps = int(train_cfg.get("logging_steps", 10))
        save_steps = int(train_cfg.get("save_steps", 100))
        sample_log_steps = int(train_cfg.get("sample_logging_steps", 50))
        sample_log_limit = int(train_cfg.get("sample_logging_limit", 4))
        chat_template = str(cfg.get("chat_template", "qwen3_nothink"))

        def autocast_ctx() -> Any:
            if device == "cuda" and torch_dtype in {torch.float16, torch.bfloat16}:
                return torch.autocast(device_type="cuda", dtype=torch_dtype)
            return nullcontext()

        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_losses: list[float] = []
            step_labels: Counter[str] = Counter()
            logged_samples: list[dict[str, Any]] = []

            for _ in range(grad_accum):
                batch = pool.sample_batch(micro_batch_size, label_ratios=data_cfg.get("label_ratios"), rng=self.rng)
                loss, batch_samples, label_counts = self._train_micro_batch(
                    torch=torch,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_bank=prompt_bank,
                    batch=batch,
                    distillation_kl=distillation_kl,
                    device=device,
                    gen_cfg=gen_cfg,
                    loss_cfg=loss_cfg,
                    chat_template=chat_template,
                    max_prompt_tokens=max_prompt_tokens,
                    max_total_tokens=max_total_tokens,
                    autocast_ctx=autocast_ctx,
                )
                if loss is None:
                    continue
                (loss / grad_accum).backward()
                step_losses.append(float(loss.detach().cpu()))
                step_labels.update(label_counts)
                if len(logged_samples) < sample_log_limit:
                    logged_samples.extend(batch_samples[: sample_log_limit - len(logged_samples)])

            if not step_losses:
                raise RuntimeError("No valid OPSD samples were produced in this optimizer step.")

            max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            avg_loss = sum(step_losses) / len(step_losses)
            log_row = {
                "step": step,
                "loss": avg_loss,
                "micro_batches": len(step_losses),
                "labels": dict(step_labels),
            }
            self._append_jsonl(log_path, log_row)
            if step == 1 or step % log_steps == 0:
                print(json.dumps(log_row, ensure_ascii=False), flush=True)
            if step == 1 or step % sample_log_steps == 0:
                for sample in logged_samples:
                    self._append_jsonl(sample_log_path, {"step": step, **sample})
            if save_steps > 0 and step % save_steps == 0:
                self._save_checkpoint(model, tokenizer, output_dir / f"checkpoint-step-{step}")

        self._save_checkpoint(model, tokenizer, output_dir / "checkpoint-final")

    def _train_micro_batch(
        self,
        *,
        torch: Any,
        model: Any,
        tokenizer: Any,
        prompt_bank: PromptBank,
        batch: list[PromptItem],
        distillation_kl: Any,
        device: str,
        gen_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
        chat_template: str,
        max_prompt_tokens: int,
        max_total_tokens: int,
        autocast_ctx: Any,
    ) -> tuple[Any | None, list[dict[str, Any]], Counter[str]]:
        prepared: list[tuple[PromptItem, str, list[int]]] = []
        for item in batch:
            student_prompt = render_prompt(tokenizer, item.instruction, template=chat_template)
            student_prompt_ids = tokenizer.encode(student_prompt, add_special_tokens=False)
            if len(student_prompt_ids) > max_prompt_tokens:
                continue
            prepared.append((item, student_prompt, student_prompt_ids))
        if not prepared:
            return None, [], Counter()

        prompt_texts = [row[1] for row in prepared]
        tokenized = tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        model.eval()
        do_sample = float(gen_cfg.get("temperature", 0.7)) > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(gen_cfg.get("max_new_tokens", 384)),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))
            generate_kwargs["top_p"] = float(gen_cfg.get("top_p", 0.9))
        with torch.no_grad():
            generated = model.generate(**tokenized, **generate_kwargs)
        continuation_batch = generated[:, tokenized["input_ids"].shape[1] :]

        losses = []
        sample_logs: list[dict[str, Any]] = []
        labels: Counter[str] = Counter()
        model.train()
        for row_idx, (item, _prompt_text, student_prompt_ids) in enumerate(prepared):
            continuation_ids = self._trim_continuation(
                continuation_batch[row_idx].detach().cpu().tolist(),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if not continuation_ids:
                continue

            teacher_prompt = render_prompt(
                tokenizer,
                item.instruction,
                system_prompt=prompt_bank.for_label(item.label),
                template=chat_template,
            )
            teacher_prompt_ids = tokenizer.encode(teacher_prompt, add_special_tokens=False)
            if len(student_prompt_ids) + len(continuation_ids) > max_total_tokens:
                continuation_ids = continuation_ids[: max(0, max_total_tokens - len(student_prompt_ids))]
            if len(teacher_prompt_ids) + len(continuation_ids) > max_total_tokens:
                continuation_ids = continuation_ids[: max(0, max_total_tokens - len(teacher_prompt_ids))]
            if not continuation_ids:
                continue

            student_input_ids = torch.tensor([student_prompt_ids + continuation_ids], dtype=torch.long, device=device)
            teacher_input_ids = torch.tensor([teacher_prompt_ids + continuation_ids], dtype=torch.long, device=device)

            model.eval()
            with torch.no_grad():
                with autocast_ctx():
                    teacher_logits_full = model(teacher_input_ids).logits[0]
            model.train()
            with autocast_ctx():
                student_logits_full = model(student_input_ids).logits[0]

            s_start = len(student_prompt_ids) - 1
            t_start = len(teacher_prompt_ids) - 1
            length = len(continuation_ids)
            student_logits = student_logits_full[s_start : s_start + length]
            teacher_logits = teacher_logits_full[t_start : t_start + length]
            loss = distillation_kl(
                student_logits,
                teacher_logits,
                kind=loss_cfg.get("type", "mixed_kl"),
                mixed_kl_weight=float(loss_cfg.get("mixed_kl_weight", 0.5)),
                temperature=float(loss_cfg.get("temperature", 1.0)),
                top_k=loss_cfg.get("top_k"),
            )
            losses.append(loss)
            labels[item.label] += 1
            sample_logs.append(
                {
                    "id": item.id,
                    "label": item.label,
                    "instruction": item.instruction,
                    "student_response": tokenizer.decode(continuation_ids, skip_special_tokens=False),
                    "continuation_tokens": len(continuation_ids),
                }
            )

        if not losses:
            return None, sample_logs, labels
        return torch.stack(losses).mean(), sample_logs, labels

    @staticmethod
    def _trim_continuation(ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
        trimmed: list[int] = []
        for token_id in ids:
            if pad_token_id is not None and token_id == pad_token_id and eos_token_id != pad_token_id:
                break
            trimmed.append(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                break
        return trimmed

    @staticmethod
    def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            f.flush()

    @staticmethod
    def _save_checkpoint(model: Any, tokenizer: Any, path: Path) -> None:
        ensure_dir(path)
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
