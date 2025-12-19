from __future__ import annotations

from econ_llm_preferences_experiment.regime_sweep import run_regime_sweep
from econ_llm_preferences_experiment.simulation import MarketParams


def test_run_regime_sweep_shapes() -> None:
    params = MarketParams(n_customers=6, n_providers=6)
    rows, welfare_mat, lambda_star_mat = run_regime_sweep(
        category="hard",
        market_params=params,
        replications=5,
        seed=0,
        attention_cost=0.01,
        max_k=2,
    )
    assert len(rows) == 4  # 2x2 grid
    assert len(welfare_mat) == 2
    assert len(welfare_mat[0]) == 2
    assert len(lambda_star_mat) == 2
    assert len(lambda_star_mat[0]) == 2
    assert {"k_I", "k_J", "net_welfare_diff", "lambda_star"} <= set(rows[0].keys())
