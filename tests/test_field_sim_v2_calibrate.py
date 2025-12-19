from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.field_sim_v2_calibrate import main


def test_calibration_writes_artifacts(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed",
            "3",
            "--eval-seed",
            "4",
            "--evals",
            "3",
            "--cities",
            "3",
            "--weeks",
            "1",
            "--jobs-easy",
            "6",
            "--jobs-hard",
            "6",
            "--providers",
            "10",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "fig_loss.svg").exists()
    assert (tmp_path / "targets.json").exists()
    assert (tmp_path / "best_params.json").exists()
    assert (tmp_path / "best_moments.json").exists()
    assert (tmp_path / "eval_settings.json").exists()
