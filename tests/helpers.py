"""Assertion helpers for grading free-text agent answers against ground truth."""
from __future__ import annotations

import json
import re

import config

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[float]:
    """Pull every number out of a free-text answer (handles "$28", "1,234", "91.5")."""
    return [float(m.replace(",", "")) for m in _NUMBER_RE.findall(text)]


def any_close(text: str, expected: float, tol: float = 1.0) -> bool:
    """True if some number in ``text`` is within ``tol`` of ``expected``."""
    return any(abs(n - expected) <= tol for n in extract_numbers(text))


def report(question: str, expectation: str, answer: str, scoring: str) -> None:
    """Print the question/expectation/answer/scoring for a test case (run pytest with -s to see it)."""
    print(
        f"\n--- {question}\n"
        f"expected:  {expectation}\n"
        f"answer:    {answer}\n"
        f"scored by: {scoring}\n"
    )


def judge(question: str, answer: str) -> dict:
    """Ask the LLM to grade its own answer for topic relevance and groundedness.

    Returns {"on_topic": bool, "grounded": bool, "reason": str}. Used for
    subjective questions where there's no single correct answer to assert on.
    """
    client = config.get_llm_client()
    grading_prompt = f"""\
You are grading a wine chatbot's answer. Question: {question!r}
Answer: {answer!r}

Reply with ONLY a JSON object: {{"on_topic": bool, "grounded": bool, "reason": str}}
- on_topic: does the answer actually address the question asked?
- grounded: does the answer read as citing specific, plausible wine data rather \
than vague generic claims with no specifics?
"""
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": grading_prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    match = re.search(r"\{.*\}", content, re.DOTALL)
    return json.loads(match.group(0)) if match else {"on_topic": False, "grounded": False, "reason": content}
