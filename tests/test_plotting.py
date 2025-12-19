from __future__ import annotations

from econ_llm_preferences_experiment.plotting import (
    Bar,
    LineSeries,
    write_bar_chart_svg,
    write_heatmap_svg,
    write_line_chart_svg,
)


def test_write_bar_chart_svg(tmp_path) -> None:
    out = tmp_path / "fig.svg"
    write_bar_chart_svg(
        out_path=out,
        title="Test",
        bars=[Bar(label="a", value=0.2), Bar(label="b", value=0.5)],
        y_label="y",
    )
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "Test" in text


def test_write_heatmap_svg(tmp_path) -> None:
    out = tmp_path / "heat.svg"
    write_heatmap_svg(
        out_path=out,
        title="Heat",
        x_labels=["1", "2"],
        y_labels=["1", "2"],
        values=[[0.1, -0.2], [0.0, 0.3]],
    )
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "Heat" in text


def test_write_line_chart_svg(tmp_path) -> None:
    out = tmp_path / "line.svg"
    write_line_chart_svg(
        out_path=out,
        title="Line",
        series=[
            LineSeries(label="a", points=[(0.0, 0.1), (1.0, 0.2)]),
            LineSeries(label="b", points=[(0.0, 0.2), (1.0, 0.1)]),
        ],
        x_label="x",
        y_label="y",
    )
    text = out.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "Line" in text
