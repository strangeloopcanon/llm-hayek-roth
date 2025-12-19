from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.field_sim_v2_intakes import main


def test_intakes_main_writes_artifacts(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed",
            "7",
            "--city",
            "city001",
            "--week",
            "0",
            "--n-easy",
            "2",
            "--n-hard",
            "2",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "intake_samples.json").exists()
