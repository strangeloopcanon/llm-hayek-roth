from __future__ import annotations

import os

from econ_llm_preferences_experiment.env import load_dotenv


def test_load_dotenv_sets_values(tmp_path) -> None:
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "FOO=bar",
                'QUOTED="baz"',
                "EMPTY=",
                "",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_dotenv(env_file)
    assert loaded["FOO"] == "bar"
    assert os.environ["FOO"] == "bar"
    assert loaded["QUOTED"] == "baz"
    assert os.environ["EMPTY"] == ""


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env.test"
    env_file.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "from_env")

    loaded = load_dotenv(env_file, override=False)
    assert loaded["FOO"] == "from_env"
    assert os.environ["FOO"] == "from_env"
