from __future__ import annotations

import random
import sys
from pathlib import Path

from econ_llm_preferences_experiment.mechanisms import SearchParams
from econ_llm_preferences_experiment.publishable_field_sim import (
    infer_weights_from_truth,
    main,
    measure_recommendation_acceptance,
    simulate_cell,
)
from econ_llm_preferences_experiment.simulation import MarketParams


def test_infer_weights_standard_is_one_hot() -> None:
    w = (0.05, 0.40, 0.10, 0.10, 0.20, 0.15)
    inferred = infer_weights_from_truth(w, elicitation="standard")
    assert sum(inferred) == 1.0
    assert inferred[1] == 1.0
    assert sum(1 for x in inferred if x == 1.0) == 1


def test_measurement_acceptance_all_accept() -> None:
    v_c_true = ((0.9,),)
    v_p_true = ((0.9,),)
    v_c_hat = ((0.1,),)
    v_p_hat = ((0.1,),)
    m = measure_recommendation_acceptance(
        v_customer_true=v_c_true,
        v_provider_true=v_p_true,
        v_customer_hat=v_c_hat,
        v_provider_hat=v_p_hat,
        accept_threshold=0.25,
        rec_k=1,
    )
    assert m.d_hat_i_obs == 1.0
    assert m.d_hat_j_obs == 1.0
    assert m.reciprocity_obs == 1.0
    assert m.accept_decisions == 2
    assert m.accepted_adj == [[0]]


def test_simulate_cell_outputs_are_well_formed() -> None:
    out, customers, ranks = simulate_cell(
        rng=random.Random(0),
        cell_id="hard_0000",
        category="hard",
        elicitation="ai",
        mechanism="central",
        market_params=MarketParams(n_customers=8, n_providers=8),
        rec_k=3,
        search_params=SearchParams(max_rounds=5),
        attention_cost=0.01,
    )
    assert 0.0 <= out.match_rate <= 1.0
    assert 0.0 <= out.d_hat_i_obs <= 1.0
    assert 0.0 <= out.d_hat_j_obs <= 1.0
    assert 0.0 <= out.reciprocity_obs <= 1.0
    assert len(customers) == 8
    assert all(r["cell_id"] == "hard_0000" for r in ranks)


def test_main_writes_expected_artifacts(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed",
            "7",
            "--cells-per-category",
            "8",
            "--n-customers",
            "6",
            "--n-providers",
            "6",
            "--rec-k",
            "2",
            "--max-rounds",
            "6",
            "--attention-cost",
            "0.01",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "arm_summary.md").exists()
    assert (tmp_path / "reg_matched.md").exists()
    assert (tmp_path / "reg_total_value.md").exists()
    assert (tmp_path / "fig_easy_match_rate_by_arm.svg").exists()
    assert (tmp_path / "fig_hard_reciprocity_curve.svg").exists()
