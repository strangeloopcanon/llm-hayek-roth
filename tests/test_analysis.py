from __future__ import annotations

from econ_llm_preferences_experiment.analysis import compute_metrics
from econ_llm_preferences_experiment.models import MatchOutcome


def test_compute_metrics_zero_matches() -> None:
    v = ((0.1, 0.2), (0.3, 0.4))
    out = MatchOutcome(matches=(), proposals=3, accept_decisions=0, rounds=2)
    m = compute_metrics(outcome=out, v_customer_true=v, v_provider_true=v)
    assert m.match_rate == 0.0
    assert m.mean_total_value == 0.0


def test_compute_metrics_nonzero_matches() -> None:
    v_c = ((0.9, 0.1), (0.2, 0.8))
    v_p = ((0.7, 0.2), (0.1, 0.95))
    out = MatchOutcome(matches=((0, 0), (1, 1)), proposals=4, accept_decisions=2, rounds=3)
    m = compute_metrics(outcome=out, v_customer_true=v_c, v_provider_true=v_p)
    assert 0.9 <= m.match_rate <= 1.0
    assert m.mean_total_value > 0.0
    assert m.proposals_per_match == 2.0
