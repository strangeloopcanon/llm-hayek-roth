from __future__ import annotations

import sys
from pathlib import Path

from econ_llm_preferences_experiment.llm_ablations import main


def test_llm_ablations_smoke_writes_outputs(tmp_path: Path) -> None:
    argv_before = sys.argv[:]
    try:
        sys.argv = [
            "prog",
            "--out",
            str(tmp_path),
            "--seed",
            "7",
            "--replications",
            "10",
            "--attention-cost",
            "0.25",
            "--skip-llm",
        ]
        main()
    finally:
        sys.argv = argv_before

    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "ablations_summary_easy.md").exists()
    assert (tmp_path / "ablations_summary_hard.md").exists()
    assert (tmp_path / "parsing_quality_easy.md").exists()
    assert (tmp_path / "parsing_quality_hard.md").exists()
    assert (tmp_path / "fig_easy_search_net_welfare.svg").exists()
    assert (tmp_path / "fig_hard_central_net_welfare.svg").exists()
