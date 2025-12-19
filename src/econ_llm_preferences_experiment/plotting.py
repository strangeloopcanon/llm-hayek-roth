from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    label: str
    value: float


@dataclass(frozen=True)
class LineSeries:
    label: str
    points: list[tuple[float, float]]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _rgb(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _lerp(a: int, b: int, t: float) -> int:
    t = _clamp01(t)
    return int(round(a + (b - a) * t))


def _diverging_color(value: float, *, max_abs: float) -> str:
    """
    Maps negative -> red, 0 -> white, positive -> blue.
    """
    if max_abs <= 0:
        return _rgb(255, 255, 255)
    t = _clamp01(abs(value) / max_abs)
    if value >= 0:
        return _rgb(_lerp(255, 37, t), _lerp(255, 99, t), _lerp(255, 235, t))
    return _rgb(_lerp(255, 220, t), _lerp(255, 38, t), _lerp(255, 38, t))


def write_bar_chart_svg(
    *,
    out_path: str | Path,
    title: str,
    bars: list[Bar],
    y_label: str,
    width: int = 900,
    height: int = 420,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    margin_l = 80
    margin_r = 20
    margin_t = 50
    margin_b = 80

    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    max_val = max([b.value for b in bars] + [1e-9])
    max_val = max_val * 1.10

    bar_w = chart_w / max(len(bars), 1)

    def y(v: float) -> float:
        return margin_t + chart_h * (1.0 - v / max_val)

    parts: list[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    parts.append("<style>")
    parts.append("text{font-family: ui-sans-serif, system-ui; fill:#111827;}")
    parts.append(".axis{stroke:#6b7280; stroke-width:1}")
    parts.append(".bar{fill:#2563eb}")
    parts.append(".title{font-size:18px; font-weight:700}")
    parts.append(".label{font-size:12px;}")
    parts.append("</style>")

    parts.append(f"<text class='title' x='{margin_l}' y='28'>{title}</text>")
    parts.append(
        f"<line class='axis' x1='{margin_l}' y1='{margin_t}' x2='{margin_l}' "
        f"y2='{margin_t + chart_h}'/>"
    )
    parts.append(
        f"<line class='axis' x1='{margin_l}' y1='{margin_t + chart_h}' x2='{margin_l + chart_w}' "
        f"y2='{margin_t + chart_h}'/>"
    )
    parts.append(f"<text class='label' x='10' y='{margin_t + 15}'>{y_label}</text>")

    for idx, bar in enumerate(bars):
        x0 = margin_l + idx * bar_w + bar_w * 0.15
        x1 = margin_l + (idx + 1) * bar_w - bar_w * 0.15
        y0 = y(bar.value)
        parts.append(
            f"<rect class='bar' x='{x0:.1f}' y='{y0:.1f}' width='{(x1 - x0):.1f}' "
            f"height='{(margin_t + chart_h - y0):.1f}'/>"
        )
        parts.append(
            f"<text class='label' x='{(x0 + x1) / 2:.1f}' y='{margin_t + chart_h + 18}' "
            f"text-anchor='middle'>{bar.label}</text>"
        )
        parts.append(
            f"<text class='label' x='{(x0 + x1) / 2:.1f}' y='{y0 - 6:.1f}' "
            f"text-anchor='middle'>{bar.value:.3f}</text>"
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_line_chart_svg(
    *,
    out_path: str | Path,
    title: str,
    series: list[LineSeries],
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 420,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    margin_l = 80
    margin_r = 20
    margin_t = 50
    margin_b = 80

    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b

    all_points = [pt for s in series for pt in s.points]
    if not all_points:
        out_path.write_text("", encoding="utf-8")
        return

    xs = [x for x, _y in all_points]
    ys = [y for _x, y in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0

    pad_y = 0.08 * (max_y - min_y)
    min_y -= pad_y
    max_y += pad_y

    def sx(x: float) -> float:
        return margin_l + chart_w * (x - min_x) / (max_x - min_x)

    def sy(y: float) -> float:
        return margin_t + chart_h * (1.0 - (y - min_y) / (max_y - min_y))

    palette = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c"]

    parts: list[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    parts.append("<style>")
    parts.append("text{font-family: ui-sans-serif, system-ui; fill:#111827;}")
    parts.append(".axis{stroke:#6b7280; stroke-width:1}")
    parts.append(".grid{stroke:#e5e7eb; stroke-width:1}")
    parts.append(".title{font-size:18px; font-weight:700}")
    parts.append(".label{font-size:12px;}")
    parts.append("</style>")

    parts.append(f"<text class='title' x='{margin_l}' y='28'>{title}</text>")
    parts.append(
        f"<line class='axis' x1='{margin_l}' y1='{margin_t}' x2='{margin_l}' "
        f"y2='{margin_t + chart_h}'/>"
    )
    parts.append(
        f"<line class='axis' x1='{margin_l}' y1='{margin_t + chart_h}' x2='{margin_l + chart_w}' "
        f"y2='{margin_t + chart_h}'/>"
    )
    parts.append(f"<text class='label' x='10' y='{margin_t + 15}'>{y_label}</text>")
    parts.append(
        f"<text class='label' x='{margin_l + chart_w / 2:.1f}' y='{height - 20}' "
        f"text-anchor='middle'>{x_label}</text>"
    )

    x_ticks = sorted(set(xs))
    for x in x_ticks:
        x0 = sx(x)
        parts.append(
            f"<line class='grid' x1='{x0:.1f}' y1='{margin_t:.1f}' "
            f"x2='{x0:.1f}' y2='{margin_t + chart_h:.1f}'/>"
        )
        parts.append(
            f"<text class='label' x='{x0:.1f}' y='{margin_t + chart_h + 18:.1f}' "
            f"text-anchor='middle'>{x:.2f}</text>"
        )

    y_ticks = 4
    for k in range(y_ticks + 1):
        yv = min_y + (max_y - min_y) * k / y_ticks
        y0 = sy(yv)
        parts.append(
            f"<line class='grid' x1='{margin_l:.1f}' y1='{y0:.1f}' "
            f"x2='{margin_l + chart_w:.1f}' y2='{y0:.1f}'/>"
        )
        parts.append(
            f"<text class='label' x='{margin_l - 8:.1f}' y='{y0 + 4:.1f}' "
            f"text-anchor='end'>{yv:.3f}</text>"
        )

    legend_x = margin_l + chart_w - 10
    legend_y = margin_t + 10
    for idx, s in enumerate(series):
        color = palette[idx % len(palette)]
        ly = legend_y + idx * 18
        parts.append(
            f"<rect x='{legend_x - 120}' y='{ly - 10}' width='12' height='12' fill='{color}'/>"
        )
        parts.append(
            f"<text class='label' x='{legend_x - 102}' y='{ly}' dominant-baseline='middle'>"
            f"{s.label}</text>"
        )

        pts = [(sx(x), sy(y)) for x, y in s.points]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{path}'/>")
        for x, y in pts:
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'/>")

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_heatmap_svg(
    *,
    out_path: str | Path,
    title: str,
    x_labels: list[str],
    y_labels: list[str],
    values: list[list[float]],
    cell_px: int = 70,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_x = len(x_labels)
    n_y = len(y_labels)
    if n_y != len(values) or any(len(row) != n_x for row in values):
        raise ValueError("values must be a (len(y_labels) × len(x_labels)) matrix")

    margin_l = 120
    margin_r = 20
    margin_t = 60
    margin_b = 90

    width = margin_l + n_x * cell_px + margin_r
    height = margin_t + n_y * cell_px + margin_b

    flat = [v for row in values for v in row]
    min_v = min(flat) if flat else 0.0
    max_v = max(flat) if flat else 0.0
    max_abs = max(abs(min_v), abs(max_v), 1e-9)

    parts: list[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    parts.append("<style>")
    parts.append("text{font-family: ui-sans-serif, system-ui; fill:#111827;}")
    parts.append(".title{font-size:18px; font-weight:700}")
    parts.append(".axis{font-size:12px; fill:#111827}")
    parts.append(".celltext{font-size:12px; fill:#111827}")
    parts.append("</style>")

    parts.append(f"<text class='title' x='{margin_l}' y='28'>{title}</text>")
    parts.append(f"<text class='axis' x='{margin_l}' y='48'>x: k_I (customer elicitation)</text>")
    parts.append("<text class='axis' x='10' y='48'>y: k_J (provider elicitation)</text>")

    for yi, ylab in enumerate(y_labels):
        y = margin_t + yi * cell_px
        parts.append(
            f"<text class='axis' x='{margin_l - 10}' y='{y + cell_px / 2:.1f}' "
            f"text-anchor='end' dominant-baseline='middle'>{ylab}</text>"
        )
        for xi, _xlab in enumerate(x_labels):
            x = margin_l + xi * cell_px
            v = values[yi][xi]
            color = _diverging_color(v, max_abs=max_abs)
            parts.append(
                f"<rect x='{x}' y='{y}' width='{cell_px}' height='{cell_px}' "
                f"fill='{color}' stroke='#e5e7eb'/>"
            )
            parts.append(
                f"<text class='celltext' x='{x + cell_px / 2:.1f}' y='{y + cell_px / 2:.1f}' "
                f"text-anchor='middle' dominant-baseline='middle'>{v:.3f}</text>"
            )

    for xi, xlab in enumerate(x_labels):
        x_center = margin_l + xi * cell_px + cell_px / 2
        y_text = margin_t + n_y * cell_px + 22
        parts.append(
            f"<text class='axis' x='{x_center:.1f}' y='{y_text:.1f}' "
            f"text-anchor='middle'>{xlab}</text>"
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
