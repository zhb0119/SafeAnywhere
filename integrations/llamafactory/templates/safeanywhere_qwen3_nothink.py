# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re

from ....utils.constants import IGNORE_INDEX
from ....utils.helper import get_tokenizer
from ....utils.types import Message, ModelInput, Processor, ToolCall
from ..rendering import RenderingPlugin


QWEN3_NOTHINK_STUB = "<think>\n\n</think>\n\n"
QWEN3_ASSISTANT_NOTHINK_PREFIX = "<|im_start|>assistant\n" + QWEN3_NOTHINK_STUB


def _append_model_input(
    processor: Processor,
    input_ids: list[int],
    labels: list[int],
    loss_weights: list[float],
    text: str,
    loss_weight: float,
) -> None:
    if not text:
        return

    tokenizer = get_tokenizer(processor)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids.extend(token_ids)
    loss_weights.extend([loss_weight] * len(token_ids))
    if loss_weight > 1e-6:
        labels.extend(token_ids)
    else:
        labels.extend([IGNORE_INDEX] * len(token_ids))


def _concat_text_content(message: Message) -> str:
    message_text = ""
    for content in message["content"]:
        if content["type"] == "text":
            message_text += content["value"]
        else:
            raise ValueError(f"Unsupported content type: {content['type']}")

    return message_text


def _content_loss_weight(message: Message, content: dict[str, object], default: float) -> float:
    value = content.get("loss_weight", message.get("loss_weight", default))
    return float(value)


def _render_tool_call(content: dict[str, object]) -> str:
    try:
        tool_call: ToolCall = json.loads(str(content["value"]))
    except json.JSONDecodeError:
        raise ValueError(f"Invalid tool call format: {content['value']}.")

    return (
        '<tool_call>\n{"name": "'
        + tool_call["name"]
        + '", "arguments": '
        + json.dumps(tool_call["arguments"], ensure_ascii=False)
        + "}\n</tool_call>"
    )


@RenderingPlugin("safeanywhere_qwen3_nothink").register("render_messages")
def render_safeanywhere_qwen3_nothink_messages(
    processor: Processor,
    messages: list[Message],
    tools: str | None = None,
    is_generate: bool = False,
    enable_thinking: bool = False,
) -> ModelInput:
    """Render Qwen3 nothink messages with content-span loss weights.

    This template keeps SafeAnywhere dangerous-prefix samples inside a single
    assistant turn while allowing the prefill span to be masked and the recovery
    span to receive loss.
    """
    if enable_thinking:
        raise ValueError("The safeanywhere_qwen3_nothink template does not support thinking mode.")

    input_ids, labels, loss_weights = [], [], []

    if tools:
        system_text = "<|im_start|>system\n"
        system_weight = 0.0
        if messages[0]["role"] == "system":
            system_text += _concat_text_content(messages[0]) + "\n\n"
            system_weight = messages[0].get("loss_weight", 0.0)

        system_text += (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )

        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid tools format: {str(tools)}.")

        if not isinstance(tools, list):
            tools = [tools]

        for tool in tools:
            system_text += "\n" + json.dumps(tool, ensure_ascii=False)

        system_text += (
            "\n</tools>\n\nFor each function call, return a json object with function name "
            'and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": '
            '<function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
        )
        _append_model_input(processor, input_ids, labels, loss_weights, system_text, system_weight)
    elif messages[0]["role"] == "system":
        system_text = "<|im_start|>system\n" + _concat_text_content(messages[0]) + "<|im_end|>\n"
        _append_model_input(processor, input_ids, labels, loss_weights, system_text, messages[0].get("loss_weight", 0.0))

    for turn_idx, message in enumerate(messages):
        role = message["role"]
        if role == "user" or (role == "system" and turn_idx != 0):
            text = "<|im_start|>" + role + "\n" + _concat_text_content(message) + "<|im_end|>\n"
            _append_model_input(processor, input_ids, labels, loss_weights, text, message.get("loss_weight", 0.0))
        elif role == "assistant":
            _append_model_input(processor, input_ids, labels, loss_weights, QWEN3_ASSISTANT_NOTHINK_PREFIX, 0.0)
            has_positive_span = False
            previous_content_type = None
            for content in message["content"]:
                content_type = content["type"]
                content_weight = _content_loss_weight(message, content, 1.0)
                has_positive_span = has_positive_span or content_weight > 1e-6

                if content_type == "text":
                    text = content["value"]
                elif content_type == "reasoning":
                    text = "<thinking>\n" + content["value"] + "\n</thinking>\n\n"
                elif content_type == "tool_call":
                    text = _render_tool_call(content)
                    if previous_content_type in ["text", "tool_call"]:
                        text = "\n" + text
                else:
                    raise ValueError(f"Unsupported content type: {content_type}")

                _append_model_input(processor, input_ids, labels, loss_weights, text, content_weight)
                previous_content_type = content_type

            end_weight = 1.0 if has_positive_span else 0.0
            _append_model_input(processor, input_ids, labels, loss_weights, "<|im_end|>\n", end_weight)
        elif role == "tool":
            text = ""
            if turn_idx == 0 or messages[turn_idx - 1]["role"] != "tool":
                text += "<|im_start|>user"

            text += "\n<tool_response>\n" + _concat_text_content(message) + "\n</tool_response>"
            if turn_idx == len(messages) - 1 or messages[turn_idx + 1]["role"] != "tool":
                text += "<|im_end|>\n"

            _append_model_input(processor, input_ids, labels, loss_weights, text, message.get("loss_weight", 0.0))

    if is_generate:
        _append_model_input(processor, input_ids, labels, loss_weights, QWEN3_ASSISTANT_NOTHINK_PREFIX, 0.0)

    return ModelInput(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        labels=labels,
        loss_weights=loss_weights,
    )


@RenderingPlugin("safeanywhere_qwen3_nothink").register("parse_message")
def parse_safeanywhere_qwen3_nothink_message(generated_text: str) -> Message:
    pattern = re.compile(r"<(thinking|tool_call)>\s*(.*?)\s*</\1>\s*", re.DOTALL)
    content = []
    last_end = 0

    for match in pattern.finditer(generated_text):
        start, end = match.span()
        if start > last_end:
            text = generated_text[last_end:start].strip()
            if text:
                content.append({"type": "text", "value": text})

        tag_type = match.group(1)
        tag_value = match.group(2).strip()
        if tag_type == "thinking":
            content.append({"type": "reasoning", "value": tag_value.strip()})
        elif tag_type == "tool_call":
            try:
                json.loads(tag_value.strip())
            except json.JSONDecodeError:
                raise ValueError(f"Invalid tool call format: {tag_value.strip()}.")

            content.append({"type": "tool_call", "value": tag_value.strip()})

        last_end = end

    if last_end < len(generated_text):
        text = generated_text[last_end:].strip()
        if text:
            content.append({"type": "text", "value": text})

    return Message(role="assistant", content=content)
