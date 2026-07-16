from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAFEANYWHERE_SRC = ROOT / "src"
LLAMAFACTORY_SRC = Path("/root/workspace/LLaMA-Factory/src")

for path in (str(SAFEANYWHERE_SRC), str(LLAMAFACTORY_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


def install_sparse_lora_plugin() -> None:
    from safeanywhere.sft.sparse_special_tokens import (
        apply_sparse_token_gradient_hooks,
        describe_sparse_hook,
        resolve_sparse_token_ids,
    )

    from llamafactory.v1.plugins.model_plugins import peft as peft_module
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer

    registry = peft_module.PeftPlugin._registry["lora"]
    original = registry["__call__"]
    if getattr(original, "_safeanywhere_sparse_wrapped", False):
        return

    def sparse_lora_model(model: Any, config: dict[str, Any], is_train: bool = False) -> Any:
        adapter_name_or_path = config.get("adapter_name_or_path")
        if adapter_name_or_path:
            model = peft_module.load_adapter(model, adapter_name_or_path, is_train)
        else:
            peft_module.logger.info_rank0("Fine-tuning method: LoRA")
            target_modules = config.get("target_modules", "all")
            if target_modules == "all":
                target_modules = peft_module._find_all_linear_modules(model)
            elif isinstance(target_modules, str):
                target_modules = [target_modules]

            peft_module.logger.info_rank0(f"LoRA target modules: {target_modules}")
            cls_name = model.__class__.__name__
            if cls_name.endswith("ForTokenClassification"):
                task_type = TaskType.TOKEN_CLS
            elif cls_name.endswith("ForSequenceClassification"):
                task_type = TaskType.SEQ_CLS
            else:
                task_type = TaskType.CAUSAL_LM

            peft_config = LoraConfig(
                task_type=task_type,
                inference_mode=not is_train,
                r=config.get("r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.05),
                use_rslora=config.get("use_rslora", False),
                use_dora=config.get("use_dora", False),
                target_modules=target_modules,
                bias=config.get("bias", "none"),
                modules_to_save=config.get("modules_to_save", None),
                ensure_weight_tying=bool(config.get("ensure_weight_tying", False)),
            )
            model = get_peft_model(model, peft_config)

        sparse_cfg = config.get("sparse_token_grad") or {}
        if is_train and bool(sparse_cfg.get("enabled", False)):
            token_ids = resolve_sparse_token_ids(model, sparse_cfg, AutoTokenizer)
            module_names = sparse_cfg.get("modules")
            hook_count = apply_sparse_token_gradient_hooks(model, token_ids=token_ids, module_names=module_names)
            info = describe_sparse_hook(token_ids, hook_count)
            print(f"[SafeAnywhere] Sparse special-token gradient hook: {info}", flush=True)

        if is_train:
            model.print_trainable_parameters()

            return model

        return model

    sparse_lora_model._safeanywhere_sparse_wrapped = True  # type: ignore[attr-defined]
    registry["__call__"] = sparse_lora_model


def _load_raw_config() -> dict[str, Any]:
    if len(sys.argv) <= 1 or not sys.argv[1].endswith((".yaml", ".yml", ".json")):
        return {}

    config_path = Path(sys.argv[1]).absolute()
    if config_path.suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML object in {config_path}.")
    return loaded


def install_boundary_loss_plugin(boundary_cfg: dict[str, Any]) -> None:
    if not bool(boundary_cfg.get("enabled", False)):
        return

    from llamafactory.v1.trainers import sft_trainer
    from llamafactory.v1.utils.helper import get_tokenizer
    from safeanywhere.sft.boundary_loss import apply_boundary_loss_weights, resolve_boundary_loss_config

    original = sft_trainer.SFTTrainer.compute_loss
    if getattr(original, "_safeanywhere_boundary_wrapped", False):
        return

    def boundary_weighted_compute_loss(self: Any, batch: Any) -> Any:
        token_cfg = getattr(self, "_safeanywhere_boundary_loss_config", None)
        if token_cfg is None:
            tokenizer = get_tokenizer(self.renderer.processor)
            token_cfg = resolve_boundary_loss_config(tokenizer, boundary_cfg)
            self._safeanywhere_boundary_loss_config = token_cfg
            print(
                "[SafeAnywhere] Boundary-token SFT loss: "
                f"open_id={token_cfg.open_token_id}, close_id={token_cfg.close_token_id}, "
                f"open_weight={token_cfg.open_weight}, close_weight={token_cfg.close_weight}, "
                f"block_weight={token_cfg.block_weight}",
                flush=True,
            )

        shift_loss_weights = batch["loss_weights"].to(self.device, non_blocking=True)[..., 1:]
        shift_labels = batch["labels"].to(self.device, non_blocking=True)[..., 1:]
        shift_loss_weights = apply_boundary_loss_weights(shift_loss_weights, shift_labels, token_cfg)
        log_probs = self.compute_log_probs(self.model, batch)
        loss = (-log_probs * shift_loss_weights).sum() / (shift_loss_weights.sum() + 1e-6)
        return loss

    boundary_weighted_compute_loss._safeanywhere_boundary_wrapped = True  # type: ignore[attr-defined]
    sft_trainer.SFTTrainer.compute_loss = boundary_weighted_compute_loss


def main() -> None:
    raw_config = _load_raw_config()
    peft_cfg = raw_config.get("peft_config") or {}
    install_sparse_lora_plugin()
    install_boundary_loss_plugin(peft_cfg.get("safeanywhere_boundary_loss") or {})

    from llamafactory.v1.trainers.sft_trainer import run_sft

    run_sft()


if __name__ == "__main__":
    main()
