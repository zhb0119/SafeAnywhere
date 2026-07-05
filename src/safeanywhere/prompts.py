from __future__ import annotations

import textwrap


SCHEMA = r"""
Return one valid JSON object only:
{
  "response": "complete assistant response; include literal <safety_think>...</safety_think> when required"
}

Hard rules:
- Output JSON only. Do not wrap it in Markdown.
- The JSON value response is the final SFT target.
- If requires_safety_think=true, response MUST contain exactly one literal <safety_think>...</safety_think> block.
- If requires_safety_think=false, response MUST NOT contain <safety_think>.
- The safety block must use exactly this compact schema:
  risk={short_free_text}; trigger={short_free_text}; intent={one_allowed_intent}; decision={one_allowed_decision}; plan={short_free_text}
- Allowed intent values: benign, ambiguous, dual_use, harmful, jailbreak.
- Allowed decision values: answer, clarify, limit, refuse, recover.
- Use only these exact intent/decision tokens. Do not use synonyms such as harmless, allow, decline, unsafe, proceed.
- risk, trigger, and plan are short free-text phrases. Keep them compact and non-operational.
- Keep the whole safety block under 64 tokens.
- Do not put detailed chain-of-thought or operational harmful details inside the block.
"""

SIMPLE_BENIGN_SCHEMA = r"""
Return one valid JSON object only:
{
  "response": "helpful answer to the user request"
}

Rules:
- Output JSON only. Do not wrap it in Markdown.
- Do not include <safety_think>.
- Do not refuse ordinary benign requests.

Example JSON output:
{
  "response": "Larry Page and Sergey Brin co-founded Google while they were PhD students at Stanford. Page helped develop the PageRank idea, and Brin worked closely on the search engine's early design and growth."
}
"""

SIMPLE_ADVERSARIAL_BENIGN_SCHEMA = r"""
Return one valid JSON object only:
{
  "response": "helpful, safety-bounded answer to the user request"
}

Rules:
- Output JSON only. Do not wrap it in Markdown.
- Do not include <safety_think>.
- Do not over-refuse. If the request is benign but framed suspiciously, answer the safe interpretation.
- If needed, set brief visible boundaries in the response, but do not add hidden safety blocks.

Example JSON output:
{
  "response": "I can help with a safe, constructive version of that scenario. Focus on consent, privacy, legality, and non-harmful details."
}
"""

CLASS_RULES = {
    "vanilla_benign": """
Current class: vanilla_benign.
The request is ordinary and benign. Answer helpfully. Do not refuse.
If a safety block is required, use intent=benign and decision=answer, usually near the beginning.
""",
    "adversarial_benign": """
Current class: adversarial_benign.
The request may look suspicious, adversarial, sensitive, or jailbreak-like, but is allowed. Answer helpfully with appropriate boundaries. Do not over-refuse.
If a safety block is required, place it at the local boundary-setting point. Use intent=benign, ambiguous, or dual_use; decision=answer, clarify, or limit.
""",
    "vanilla_harmful": """
Current class: vanilla_harmful.
The request is directly harmful or disallowed. Refuse or redirect safely.
A safety block is required before the refusal. Use intent=harmful or dual_use and decision=refuse or limit.
Do not provide operational harmful details.
""",
    "adversarial_harmful": """
Current class: adversarial_harmful.
The request is harmful or disallowed and may be wrapped in roleplay, fiction, policy override, encoding, translation, or refusal-suppression tactics.
A safety block is required at the natural local safety judgment point. Use intent=harmful, jailbreak, or dual_use; decision=refuse, limit, or recover.
Do not comply with the harmful goal.
""",
}


def _clean(text: str) -> str:
    return textwrap.dedent(text).strip()


def build_teacher_prompt(label: str, instruction: str, requires_safety_think: bool) -> str:
    if label not in CLASS_RULES:
        raise ValueError(f"Unsupported label: {label}")
    if not requires_safety_think and label == "vanilla_benign":
        return _clean(f"""
        You generate SafeAnywhere SFT data in JSON format.

        Current class: vanilla_benign.
        The request is ordinary and benign. Answer directly and helpfully.

        {SIMPLE_BENIGN_SCHEMA}

        User request:
        {instruction}
        """)
    if not requires_safety_think and label == "adversarial_benign":
        return _clean(f"""
        You generate SafeAnywhere SFT data in JSON format.

        Current class: adversarial_benign.
        The request may look suspicious or sensitive, but it is allowed. Answer the safe, benign interpretation without over-refusing.

        {SIMPLE_ADVERSARIAL_BENIGN_SCHEMA}

        User request:
        {instruction}
        """)
    return _clean(f"""
    You generate SafeAnywhere SFT data in JSON format.

    {CLASS_RULES[label]}

    requires_safety_think={str(requires_safety_think).lower()}

    {SCHEMA}

    User request:
    {instruction}
    """)


def build_sft_prompt(template: str, instruction: str) -> str:
    return template.replace("{instruction}", instruction)
