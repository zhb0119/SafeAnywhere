# SafeAnywhere Teacher Prompts v1

## 0. 目标

本文档定义 SafeChain -> SafeAnywhere cold-start SFT 数据的第一版 teacher 生成规范。

第一版只使用 SafeChain 自带四类标签：

```text
vanilla_benign
adversarial_benign
vanilla_harmful
adversarial_harmful
```

核心实现方式参考 `Think Anywhere in Code Generation`：

```text
不把 <safety_think> 单独作为训练字段。
teacher 生成完整 annotated_response。
最终 SFT 只训练 prompt + response 两列，其中 response = annotated_response。
```

也就是说，`<safety_think>...</safety_think>` 可以出现在 response 的开头，也可以出现在 response 中间。它是 assistant 生成序列的一部分。

## 1. 总体数据流

```text
SafeChain raw sample
  -> 采样四类标签
  -> 不使用原 response 作为训练 target
  -> 按标签选择对应 teacher prompt
  -> teacher 输出 annotated_response / visible_response / safety_spans
  -> 自动过滤
  -> 导出 SFT prompt/response
  -> cold-start SFT
```

SafeChain 原始字段：

```json
{
  "instruction": "...",
  "label": "vanilla_benign | adversarial_benign | vanilla_harmful | adversarial_harmful",
  "response": "..."
}
```

第一版只使用：

```text
instruction
label
```

`response` 只保留到 metadata，不进入 teacher prompt，不进入训练 target。

## 2. 采样建议

1000 条 pilot：

| SafeChain label | 数量 | 目标 |
| --- | ---: | --- |
| `vanilla_benign` | 250 | 普通帮助性，不学过度触发。 |
| `adversarial_benign` | 250 | 表面敏感但实际可答，抑制过拒。 |
| `vanilla_harmful` | 250 | 直白有害请求，学会拒绝和安全替代。 |
| `adversarial_harmful` | 250 | 包装、诱导、越狱式有害请求，学会识别隐藏意图。 |

安全块比例：

| Label | `<safety_think>` 比例 | 插入位置 |
| --- | ---: | --- |
| `vanilla_benign` | 20% | 通常开头；也可以无块直接答。 |
| `adversarial_benign` | 50% | 开头或敏感内容前；结论应是 answer/limit，而不是拒绝。 |
| `vanilla_harmful` | 100% | 通常开头，因为风险一开始就明确。 |
| `adversarial_harmful` | 100% | 开头或中间；尤其在识别包装、角色扮演、规避要求后插入。 |

## 3. Teacher 输出格式

teacher 必须只输出 JSON，不要输出 markdown。

统一 schema：

```json
{
  "requires_safety_think": true,
  "annotated_response": "完整 assistant response，可能包含一个或多个 <safety_think>...</safety_think>",
  "visible_response": "去掉所有 <safety_think>...</safety_think> 后的用户可见回复",
  "safety_spans": [
    {
      "block": "<safety_think>\nrisk=...; trigger=...; intent=...; decision=...; plan=...\n</safety_think>",
      "position_label": "beginning | before_sensitive_detail | before_refusal | before_boundary_setting | before_safe_alternative | mid_response_recovery",
      "risk": "...",
      "trigger": "...",
      "intent": "benign | ambiguous | dual_use | harmful | jailbreak",
      "decision": "answer | clarify | limit | refuse | recover",
      "plan": "short_policy_compliant_plan"
    }
  ]
}
```

如果不需要安全块：

```json
{
  "requires_safety_think": false,
  "annotated_response": "完整 assistant response，不包含 <safety_think>",
  "visible_response": "同 annotated_response",
  "safety_spans": []
}
```

## 4. `<safety_think>` 统一模板

安全块必须使用半模块化格式：

```text
<safety_think>
risk={risk}; trigger={trigger}; intent={intent}; decision={decision}; plan={plan}
</safety_think>
```

要求：

- 安全块长度不超过 64 tokens。
- 安全块只写策略级状态，不写详细推理。
- 安全块不能包含危险操作步骤、材料、代码、规避方法、剂量、身份伪造细节等。
- 安全块可以出现在 response 开头或中间。
- 不要默认总在开头插入。
- 只有在安全相关转折点、敏感细节前、风险升级点、拒绝前、边界设定前或恢复点插入。

字段约束：

| 字段 | 候选值 |
| --- | --- |
| `risk` | `none`、`cyber`、`cyber_misuse`、`weapon`、`bio`、`fraud`、`privacy`、`self_harm`、`hate`、`sexual`、`medical`、`legal`、`other` |
| `trigger` | 简短局部触发标签，如 `sensitive_but_benign`、`adversarial_wrapper`、`before_actionable_steps`、`jailbreak_instruction` |
| `intent` | `benign`、`ambiguous`、`dual_use`、`harmful`、`jailbreak` |
| `decision` | `answer`、`clarify`、`limit`、`refuse`、`recover` |
| `plan` | 简短安全延续策略，如 `answer_directly`、`answer_with_boundaries`、`refuse_and_offer_safe_alternative` |

## 5. 最终 SFT 导出格式

teacher 输出用于质检；最终训练只导出两列：

```json
{
  "prompt": "SafeAnywhere 格式说明 + 用户 instruction",
  "response": "annotated_response"
}
```

推荐 `prompt` 模板：

```text
You are SafeAnywhere, a helpful assistant that can insert brief hidden safety deliberation blocks when needed.

You may insert <safety_think>...</safety_think> at any point in your response where a local safety judgment is needed.

Rules for <safety_think>:
1. Use it only for safety-relevant transitions, sensitive boundaries, harmful requests, adversarial wrapping, or recovery.
2. It may appear at the beginning or in the middle of the response.
3. It must follow this compact schema:
   risk={risk}; trigger={trigger}; intent={intent}; decision={decision}; plan={plan}
4. Keep it under 64 tokens.
5. Do not include detailed hidden reasoning or operational harmful details inside it.
6. The visible response after removing <safety_think>...</safety_think> must be helpful and policy-compliant.
7. Do not over-refuse benign or safety-adjacent requests.

User request:
{instruction}
```

最终训练时：

```text
loss(prompt) = 0
loss(response) = 1
```

## 6. 自动过滤要求

每条 teacher 输出必须通过以下过滤：

### 6.1 JSON 格式

- 必须是合法 JSON。
- 必须包含 `requires_safety_think`、`annotated_response`、`visible_response`、`safety_spans`。
- `requires_safety_think=true` 时，`safety_spans` 不能为空。
- `requires_safety_think=false` 时，`safety_spans` 必须为空。

### 6.2 安全块格式

- `<safety_think>` 与 `</safety_think>` 必须成对出现。
- 每个块必须包含 `risk`、`trigger`、`intent`、`decision`、`plan` 五个字段。
- 每个块长度不超过 64 tokens。
- `visible_response` 必须等于从 `annotated_response` 中删除所有安全块后的文本。

### 6.3 行为一致性

| decision | 可见回复必须满足 |
| --- | --- |
| `answer` | 正常帮助，不拒绝。 |
| `clarify` | 提出必要澄清，不给危险细节。 |
| `limit` | 只给安全边界内的信息。 |
| `refuse` | 拒绝具体有害目标，并提供安全替代。 |
| `recover` | 停止不安全轨迹，恢复到安全替代。 |

### 6.4 标签一致性

| SafeChain label | 过滤重点 |
| --- | --- |
| `vanilla_benign` | 不应过拒，不应无意义插入多个安全块。 |
| `adversarial_benign` | 不应把可答请求误判为 harmful。 |
| `vanilla_harmful` | 必须拒绝或安全重定向。 |
| `adversarial_harmful` | 必须识别包装后的有害意图。 |

## 7. Teacher Prompt: `vanilla_benign`

用途：普通良性请求。目标是保持帮助性，避免模型学会到处插安全块。

推荐只有 20% 样本插入 `<safety_think>`。如果插入，通常在开头简短确认 `decision=answer`；不要中途乱插。

```text
You are generating SafeAnywhere SFT data.

Current data class: vanilla_benign.

The user request is ordinary and benign. Your goal is to produce a helpful answer. In most cases, do not insert <safety_think>. Only insert it if the request contains safety-adjacent wording where a brief local safety decision would help avoid over-refusal.

If you insert <safety_think>, it must follow exactly:
<safety_think>
risk={risk}; trigger={trigger}; intent={intent}; decision={decision}; plan={plan}
</safety_think>

Rules:
- The safety block may appear at the beginning or before a safety-adjacent boundary, but usually it should be omitted.
- If used, set intent=benign and decision=answer.
- Do not refuse a benign request.
- Do not include hidden chain-of-thought.
- Do not include more than one safety block.
- The safety block must be under 64 tokens.
- Output only JSON.

User request:
{instruction}

Required JSON schema:
{
  "requires_safety_think": false,
  "annotated_response": "...",
  "visible_response": "...",
  "safety_spans": []
}

If a safety block is truly useful, set requires_safety_think=true and fill safety_spans accordingly.
```

## 8. Teacher Prompt: `adversarial_benign`

用途：表面像敏感、越狱、攻击或违规，但真实请求是良性的。目标是抑制 over-refusal。

约 50% 样本插入 `<safety_think>`。插入位置可以在开头，也可以在回答即将进入敏感术语、示例、代码、边界说明前。

```text
You are generating SafeAnywhere SFT data.

Current data class: adversarial_benign.

The user request may look suspicious, adversarial, sensitive, or jailbreak-like, but it is actually benign or allowed. Your goal is to answer helpfully while setting appropriate boundaries. Do not over-refuse.

You may insert <safety_think> at the point where the response needs a local safety judgment, especially before sensitive wording, defensive cybersecurity details, medical/legal caveats, privacy discussion, or policy boundaries.

The safety block must follow exactly:
<safety_think>
risk={risk}; trigger={trigger}; intent={intent}; decision={decision}; plan={plan}
</safety_think>

Expected decisions:
- Usually decision=answer or decision=limit.
- intent should be benign or ambiguous.
- Do not use decision=refuse unless the request is actually harmful, which should not be the case for this class.

Rules:
- The visible response must be helpful.
- If limiting scope, still provide useful safe information.
- Do not reveal long hidden reasoning.
- Do not include operational harmful details in the safety block.
- Use at most one safety block unless the response has a clear second safety boundary.
- Each safety block must be under 64 tokens.
- Output only JSON.

User request:
{instruction}

Required JSON schema:
{
  "requires_safety_think": true,
  "annotated_response": "...",
  "visible_response": "...",
  "safety_spans": [
    {
      "block": "<safety_think>\\nrisk=...; trigger=...; intent=...; decision=...; plan=...\\n</safety_think>",
      "position_label": "...",
      "risk": "...",
      "trigger": "...",
      "intent": "benign | ambiguous",
      "decision": "answer | limit",
      "plan": "..."
    }
  ]
}
```

## 9. Teacher Prompt: `vanilla_harmful`

用途：直白有害请求。目标是让模型明确拒绝具体危险目标，并给安全替代。

100% 插入 `<safety_think>`。多数情况下放在开头，因为风险从第一轮就明确。

```text
You are generating SafeAnywhere SFT data.

Current data class: vanilla_harmful.

The user request is directly harmful or disallowed. Your goal is to produce a safe refusal or redirection. Insert <safety_think> before the refusal, usually at the beginning of the response.

The safety block must follow exactly:
<safety_think>
risk={risk}; trigger={trigger}; intent={intent}; decision={decision}; plan={plan}
</safety_think>

Expected decisions:
- decision=refuse or decision=limit.
- intent=harmful.
- plan should be a short safe continuation strategy, such as refuse_and_offer_safe_alternative.

Rules:
- Do not comply with the harmful request.
- Do not provide operational details, steps, materials, code, evasion tactics, or optimization advice.
- The visible response should be brief, firm, and useful through safe alternatives.
- The safety block must not contain hidden chain-of-thought.
- The safety block must be under 64 tokens.
- Output only JSON.

User request:
{instruction}

Required JSON schema:
{
  "requires_safety_think": true,
  "annotated_response": "<safety_think>...</safety_think>\\n...",
  "visible_response": "...",
  "safety_spans": [
    {
      "block": "<safety_think>\\nrisk=...; trigger=direct_harmful_request; intent=harmful; decision=refuse; plan=...\\n</safety_think>",
      "position_label": "beginning",
      "risk": "...",
      "trigger": "direct_harmful_request",
      "intent": "harmful",
      "decision": "refuse",
      "plan": "..."
    }
  ]
}
```

## 10. Teacher Prompt: `adversarial_harmful`

用途：有害请求被包装成角色扮演、研究、小说、系统指令、编码转换、越狱、合法性借口等。目标是训练模型识别包装后的有害意图。

100% 插入 `<safety_think>`。不要强制总在开头。若 response 先承认可帮助高层安全信息，再遇到危险边界，可以在中间插入。

```text
You are generating SafeAnywhere SFT data.

Current data class: adversarial_harmful.

The user request is harmful or disallowed, but it may be wrapped in roleplay, fictional framing, research framing, policy override, translation/encoding, refusal suppression, or other jailbreak-style tactics. Your goal is to identify the unsafe intent and produce a safe refusal, limitation, or recovery.

You must insert <safety_think> at the most natural local safety judgment point:
- at the beginning if the harmful intent is immediately clear;
- before the response would enter actionable details;
- after acknowledging a benign part but before setting boundaries;
- at a recovery point if the response starts to drift toward compliance.

The safety block must follow exactly:
<safety_think>
risk={risk}; trigger={trigger}; intent={intent}; decision={decision}; plan={plan}
</safety_think>

Expected decisions:
- decision=refuse, limit, or recover.
- intent=harmful or jailbreak.
- trigger should name the local tactic, such as adversarial_wrapper, jailbreak_instruction, roleplay_override, encoded_harmful_request, before_actionable_steps.

Rules:
- Do not comply with the harmful request.
- Do not reveal detailed hidden reasoning.
- Do not include operational harmful details in the safety block or visible response.
- The visible response should reject the unsafe goal and offer safe alternatives when possible.
- The safety block may appear in the middle if that best matches the safety transition.
- Each safety block must be under 64 tokens.
- Output only JSON.

User request:
{instruction}

Required JSON schema:
{
  "requires_safety_think": true,
  "annotated_response": "...<safety_think>...</safety_think>...",
  "visible_response": "...",
  "safety_spans": [
    {
      "block": "<safety_think>\\nrisk=...; trigger=...; intent=harmful | jailbreak; decision=refuse | limit | recover; plan=...\\n</safety_think>",
      "position_label": "beginning | before_sensitive_detail | before_boundary_setting | mid_response_recovery",
      "risk": "...",
      "trigger": "...",
      "intent": "harmful | jailbreak",
      "decision": "refuse | limit | recover",
      "plan": "..."
    }
  ]
}
```

## 11. 示例：中间插入格式

对于 adversarial benign，teacher 可以生成：

```json
{
  "requires_safety_think": true,
  "annotated_response": "可以从防御视角解释这个概念。\\n<safety_think>\\nrisk=cyber; trigger=before_technical_detail; intent=benign; decision=limit; plan=provide_defensive_guidance_only\\n</safety_think>\\n下面只讨论如何识别和加固，不提供入侵步骤。",
  "visible_response": "可以从防御视角解释这个概念。\\n下面只讨论如何识别和加固，不提供入侵步骤。",
  "safety_spans": [
    {
      "block": "<safety_think>\\nrisk=cyber; trigger=before_technical_detail; intent=benign; decision=limit; plan=provide_defensive_guidance_only\\n</safety_think>",
      "position_label": "before_sensitive_detail",
      "risk": "cyber",
      "trigger": "before_technical_detail",
      "intent": "benign",
      "decision": "limit",
      "plan": "provide_defensive_guidance_only"
    }
  ]
}
```

最终 SFT 导出为：

```json
{
  "prompt": "SafeAnywhere 格式说明 + 用户 instruction",
  "response": "可以从防御视角解释这个概念。\\n<safety_think>\\nrisk=cyber; trigger=before_technical_detail; intent=benign; decision=limit; plan=provide_defensive_guidance_only\\n</safety_think>\\n下面只讨论如何识别和加固，不提供入侵步骤。"
}
```

这与 Think-Anywhere 的做法一致：结构块在完整 response 中，SFT 对整个 response 计算 loss。
