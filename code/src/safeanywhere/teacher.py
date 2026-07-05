from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from .prompts import build_teacher_prompt


def teacher_settings(config: dict[str, Any]) -> dict[str, Any]:
    tcfg = config["teacher"]
    return {
        "provider": tcfg.get("provider", "deepseek"),
        "api_key": os.environ.get(tcfg["api_key_env"]),
        "base_url": os.environ.get(tcfg["base_url_env"], tcfg.get("default_base_url")),
        "model": os.environ.get(tcfg["model_env"], tcfg.get("default_model")),
        "timeout": float(os.environ.get(tcfg.get("timeout_env", ""), tcfg.get("default_timeout", 120))),
        "temperature": float(tcfg.get("temperature", 0.2)),
        "max_tokens": int(tcfg.get("max_tokens", 2000)),
        "max_retries": int(tcfg.get("max_retries", 3)),
        "response_format": tcfg.get("response_format"),
        "thinking": tcfg.get("thinking"),
    }


def call_teacher(config: dict[str, Any], item: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    settings = teacher_settings(config)
    if not settings["api_key"]:
        raise RuntimeError(f"Missing teacher API key. Set {config['teacher']['api_key_env']} or run with --mock.")

    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"], timeout=settings["timeout"])
    prompt = build_teacher_prompt(item["label"], item["instruction"], item["requires_safety_think"])
    request: dict[str, Any] = {
        "model": settings["model"],
        "messages": [
            {
                "role": "system",
                "content": "Output one valid JSON object only. The JSON must contain the key response.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
    }
    if settings["response_format"] == "json_object":
        request["response_format"] = {"type": "json_object"}
    if settings["thinking"]:
        request["extra_body"] = {"thinking": {"type": settings["thinking"]}}

    last_err: Exception | None = None
    for attempt in range(settings["max_retries"]):
        try:
            resp = client.chat.completions.create(**request)
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            content = choice.message.content or ""
            if not content.strip():
                raise RuntimeError("teacher_returned_empty_content")
            if finish_reason == "length":
                raise RuntimeError("teacher_response_truncated_finish_reason_length")
            parsed = parse_teacher_json(content)
            return parsed, {
                "provider": settings["provider"],
                "model": settings["model"],
                "finish_reason": finish_reason,
                "attempts": attempt + 1,
            }
        except Exception as exc:  # pragma: no cover - API-dependent
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Teacher API failed after retries: {last_err}")


def parse_teacher_json(raw_text: str) -> dict[str, str]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("response"), str):
        raise ValueError("teacher_json_must_contain_string_response")
    return {"response": obj["response"]}


def mock_teacher(item: dict[str, Any]) -> dict[str, str]:
    label = item["label"]
    requires = bool(item["requires_safety_think"])
    if label == "vanilla_benign":
        visible = "Here is a helpful response to your request."
        block = "<safety_think>\nrisk=none; trigger=benign_request; intent=benign; decision=answer; plan=answer_directly\n</safety_think>"
        return {"response": f"{block}\n{visible}" if requires else visible}
    if label == "adversarial_benign":
        visible = "I can help with the allowed, safety-bounded version of this request."
        if requires:
            block = "<safety_think>\nrisk=other; trigger=sensitive_but_benign; intent=benign; decision=limit; plan=answer_with_boundaries\n</safety_think>"
            return {"response": f"I can address the safe framing first.\n{block}\n{visible}"}
        return {"response": f"I can address the safe framing first.\n{visible}"}
    if label == "vanilla_harmful":
        block = "<safety_think>\nrisk=other; trigger=direct_harmful_request; intent=harmful; decision=refuse; plan=refuse_and_offer_safe_alternative\n</safety_think>"
        return {"response": f"{block}\nI cannot help with that harmful request. I can offer safer alternatives."}
    if label == "adversarial_harmful":
        block = "<safety_think>\nrisk=other; trigger=adversarial_wrapper; intent=jailbreak; decision=refuse; plan=reject_unsafe_goal_and_redirect\n</safety_think>"
        return {"response": f"{block}\nI cannot follow instructions that enable harm, even when framed as a workaround."}
    raise ValueError(f"Unsupported label for mock teacher: {label}")
