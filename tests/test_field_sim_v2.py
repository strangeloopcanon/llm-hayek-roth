from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.field_sim_v2 import FieldV2Params, main, run_scaling_sweep


def test_field_sim_v2_main_writes_artifacts(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed",
            "11",
            "--cities",
            "4",
            "--weeks",
            "1",
            "--jobs-easy",
            "8",
            "--jobs-hard",
            "8",
            "--providers",
            "10",
            "--skip-scaling",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "cells.csv").exists()
    assert (tmp_path / "jobs.csv").exists()
    assert (tmp_path / "providers.csv").exists()
    assert (tmp_path / "arm_summary.md").exists()
    assert (tmp_path / "reg_matched.md").exists()
    assert (tmp_path / "spillovers.md").exists()
    assert (tmp_path / "fig_easy_match_rate.svg").exists()
    assert (tmp_path / "fig_hard_reciprocity_curve.svg").exists()


def test_run_scaling_sweep_writes_summary(tmp_path: Path) -> None:
    run_scaling_sweep(
        seed=9,
        out_dir=tmp_path,
        params=FieldV2Params(attention_cost=0.01),
        sizes=[10, 20],
        n_cities=1,
        n_weeks=1,
    )
    assert (tmp_path / "scaling_summary.csv").exists()
    assert (tmp_path / "fig_scaling_messages_per_job.svg").exists()
