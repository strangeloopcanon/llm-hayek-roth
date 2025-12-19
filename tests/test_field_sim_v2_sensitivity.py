from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.field_sim_v2_sensitivity import main


def test_field_sim_v2_sensitivity_writes_outputs(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed-base",
            "10",
            "--n-seeds",
            "1",
            "--cities",
            "4",
            "--weeks",
            "1",
            "--jobs-easy",
            "3",
            "--jobs-hard",
            "3",
            "--providers",
            "5",
            "--attention-cost",
            "0.25",
            "--central-rec-k",
            "3",
            "--search-k",
            "1",
            "--std-misclass-hard",
            "0.7",
            "--max-runs",
            "1",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "fig_triple_welfare_vs_rec_k.svg").exists()
