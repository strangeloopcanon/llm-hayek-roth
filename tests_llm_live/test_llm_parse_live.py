from __future__ import annotations

import os

import pytest

from econ_llm_preferences_experiment.elicitation import parse_batch_with_gpt
from econ_llm_preferences_experiment.env import load_dotenv
from econ_llm_preferences_experiment.models import AgentTruth
from econ_llm_preferences_experiment.openai_client import OpenAIClient


@pytest.mark.llm_live
def test_llm_parses_minimal_batch() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY missing")

    client = OpenAIClient(max_calls=3)
    agents = [
        AgentTruth(
            agent_id="c000",
            side="customer",
            category="easy",
            weights=(0.35, 0.35, 0.10, 0.10, 0.05, 0.05),
        ),
        AgentTruth(
            agent_id="p000",
            side="provider",
            category="easy",
            weights=(0.10, 0.35, 0.10, 0.10, 0.25, 0.10),
        ),
    ]
    truth = {a.agent_id: a for a in agents}
    texts = {
        "c000": (
            "Side: customer\nCategory: easy\nStated priorities (top-1 only):\n"
            "- quality_focus is high importance\n"
            "Other dimensions are not specified."
        ),
        "p000": (
            "Side: provider\nCategory: easy\nStated priorities (top-1 only):\n"
            "- weirdness_tolerance is high importance\n"
            "Other dimensions are not specified."
        ),
    }
    parsed = parse_batch_with_gpt(client=client, texts_by_agent_id=texts, truth_by_agent_id=truth)
    assert len(parsed.inferred) == 2
    for a in parsed.inferred:
        assert abs(sum(a.weights) - 1.0) < 1e-6
