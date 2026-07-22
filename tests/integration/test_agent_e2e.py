"""End-to-end answer-quality tests: real Azure OpenAI + real MCP + real Chroma index.

Each test asks a golden question through `run_agent` and checks the final answer
against ground truth computed independently via pandas, rather than asserting on
exact wording. Requires OPENAI_API_KEY and a built index (`uv run build-index`).
"""
from __future__ import annotations

import logging
import os

import pandas as pd
import pytest

import agent
import config

from helpers import any_close, judge, report

pytestmark = pytest.mark.integration

# Quiet routine INFO logs (this process's own + the MCP subprocess's) so each
# question/answer/scoring report prints cleanly next to its PASSED/FAILED line.
logging.getLogger().setLevel(logging.WARNING)
os.environ["MCP_LOG_LEVEL"] = "WARNING"

WINES_DF = pd.read_csv(config.DATA_CSV)


def test_mcp_aggregate_average_price():
    """Structured (MCP-only) question: a numeric aggregate should match pandas ground truth."""
    question = "What is the average price of wine from Italy? Give me the number."
    expected = WINES_DF.loc[WINES_DF["country"] == "Italy", "price"].mean()

    answer = agent.run_agent([{"role": "user", "content": question}])

    report(
        question,
        expectation=f"a number within $1.00 of the true mean (${expected:.2f})",
        answer=answer,
        scoring="any_close() extracts every number in the answer and checks one is within tol of expected",
    )
    assert any_close(answer, expected, tol=1.0), f"expected ~{expected:.2f} in: {answer!r}"


def test_mcp_top_wines_by_country():
    """Structured (MCP-only) question: cited wines must be real Italian wines whose
    points meet the true top-3 threshold.

    Note: doesn't assert an exact top-3 title match — many wines tie at the top
    score, and pandas' `nlargest` may break ties differently than the model's own
    sort, so either subset of the tied wines is a correct answer.
    """
    question = "What are the top 3 highest-rated wines from Italy? List their titles."
    italy = WINES_DF[WINES_DF["country"] == "Italy"]
    threshold = italy.nlargest(3, "points")["points"].min()

    answer = agent.run_agent([{"role": "user", "content": question}])
    answer_lower = answer.lower()

    cited = italy[italy["title"].astype(str).str.lower().apply(lambda t: t in answer_lower)]

    report(
        question,
        expectation=f"real Italian wine title(s) with points >= {threshold} (the true top-3 threshold)",
        answer=answer,
        scoring=f"cited titles matched against the dataset: {cited['title'].tolist() or '(none matched)'}",
    )
    assert not cited.empty, f"expected a real Italian wine title in: {answer!r}"
    assert (cited["points"] >= threshold).all(), (
        f"cited wine(s) below the true top-3 points threshold ({threshold}):\n{cited}"
    )


def test_rag_answer_is_grounded_in_retrieval():
    """Semantic (RAG-only) question: cited wines must be real, and their actual
    descriptions (not the model's claim) must mention the requested flavour notes.

    Note: doesn't assert the exact retrieval candidates the agent's own tool call
    returned — the model chooses its own query phrasing, which can shift ANN
    results even for a rephrased-but-equivalent query.
    """
    query = "earthy, forest-floor tasting notes"
    question = f"Find a couple of wines with {query}. Give me their titles."

    answer = agent.run_agent([{"role": "user", "content": question}])
    answer_lower = answer.lower()

    cited = WINES_DF[WINES_DF["title"].astype(str).str.lower().apply(lambda t: t in answer_lower)]

    report(
        question,
        expectation="real wine title(s) whose actual description mentions 'earth' or 'forest'",
        answer=answer,
        scoring=f"cited titles matched against the dataset: {cited['title'].tolist() or '(none matched)'}",
    )
    assert not cited.empty, f"expected a real wine title from the dataset in: {answer!r}"

    grounded = cited["description"].astype(str).str.contains("earth|forest", case=False, na=False)
    assert grounded.any(), (
        f"none of the cited wines' actual descriptions mention earthy/forest-floor notes:\n"
        f"{cited[['title', 'description']]}"
    )


def test_combined_rag_and_mcp_respects_constraints():
    """Combined question: cited wines must be real (exist in the dataset) and satisfy
    every structured constraint — not just plausible-sounding.

    Note: this doesn't assert the agent called the RAG tool for "earthy" — a Pinot
    Noir that matches the structured filters is a reasonable answer even without a
    separate semantic lookup, since "earthy" is a fairly generic Pinot Noir descriptor.
    """
    question = (
        "Find earthy Pinot Noirs under $30 with at least 90 points. "
        "List a couple with their price and points."
    )
    answer = agent.run_agent([{"role": "user", "content": question}])
    answer_lower = answer.lower()

    pinots = WINES_DF[WINES_DF["variety"].astype(str).str.contains("pinot noir", case=False, na=False)]
    cited = pinots[pinots["title"].astype(str).str.lower().apply(lambda t: t in answer_lower)]

    report(
        question,
        expectation="real Pinot Noir title(s) with price <= $30 and points >= 90",
        answer=answer,
        scoring=f"cited titles matched against the dataset: {cited['title'].tolist() or '(none matched)'}",
    )
    assert "pinot" in answer_lower, f"expected 'Pinot' mentioned in: {answer!r}"
    assert not cited.empty, f"expected a real Pinot Noir title from the dataset in: {answer!r}"
    assert (cited["price"] <= 30).all(), f"cited wine(s) violate the price constraint:\n{cited}"
    assert (cited["points"] >= 90).all(), f"cited wine(s) violate the points constraint:\n{cited}"


def test_subjective_question_is_on_topic_and_grounded():
    """Subjective (no single correct answer) question: graded by a second LLM call
    instead of asserting on exact wording."""
    question = "Describe a bold, tannic Cabernet Sauvignon, citing a specific wine."
    answer = agent.run_agent([{"role": "user", "content": question}])

    verdict = judge(question, answer)

    report(
        question,
        expectation="on-topic and grounded, per a second LLM call's judgment",
        answer=answer,
        scoring=f"judge() verdict: {verdict}",
    )
    assert verdict.get("on_topic"), f"answer not on-topic ({verdict.get('reason')}): {answer!r}"
    assert verdict.get("grounded"), f"answer not grounded ({verdict.get('reason')}): {answer!r}"
