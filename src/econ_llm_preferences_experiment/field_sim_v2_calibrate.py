from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import TypeAlias

from econ_llm_preferences_experiment.field_sim_v2 import (
    CityWeekAssignment,
    FieldV2Params,
    simulate_city_week,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.plotting import LineSeries, write_line_chart_svg

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


DEFAULT_TARGETS: dict[str, dict[str, float]] = {
    # These are deliberately "plausible moments" for a home-services marketplace baseline
    # (standard intake + decentralized search) over a short horizon.
    "easy": {
        "match_rate": 0.45,
        "cancel_rate": 0.05,
        "avg_price": 210.0,
        "messages_per_job": 9.5,
        "provider_inbox_per_day": 1.8,
        "consumer_surplus_per_job": 20.0,
        "provider_profit_per_job": 15.0,
    },
    "hard": {
        "match_rate": 0.35,
        "cancel_rate": 0.06,
        "avg_price": 320.0,
        "messages_per_job": 10.5,
        "provider_inbox_per_day": 2.0,
        "consumer_surplus_per_job": 30.0,
        "provider_profit_per_job": 20.0,
    },
}

DEFAULT_TOL: dict[str, float] = {
    "match_rate": 0.04,
    "cancel_rate": 0.02,
    "avg_price": 40.0,
    "messages_per_job": 2.0,
    "provider_inbox_per_day": 0.6,
    "consumer_surplus_per_job": 20.0,
    "provider_profit_per_job": 10.0,
}


def _simulate_baseline_moments(
    *,
    seed: int,
    params: FieldV2Params,
    cities: int,
    weeks: int,
    jobs_easy: int,
    jobs_hard: int,
    providers: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)  # nosec B311
    cells = []
    for city_idx in range(cities):
        city_id = f"cal_city{city_idx:03d}"
        for week in range(weeks):
            assignment = CityWeekAssignment(
                city_id=city_id,
                week=week,
                cell_easy=("standard", "search"),
                cell_hard=("standard", "search"),
            )
            city_rng = random.Random(rng.randrange(1_000_000_000))  # nosec B311
            cell_out, _jobs, _prov, _meas = simulate_city_week(
                rng=city_rng,
                assignment=assignment,
                params=params,
                n_jobs_easy=jobs_easy,
                n_jobs_hard=jobs_hard,
                n_providers=providers,
            )
            cells.extend(cell_out)

    out: dict[str, dict[str, float]] = {}
    for category in ("easy", "hard"):
        subset = [c for c in cells if c.category == category]
        out[category] = {
            "match_rate": _mean([c.match_rate for c in subset]),
            "cancel_rate": _mean([c.cancel_rate for c in subset]),
            "avg_price": _mean([c.avg_price for c in subset]),
            "messages_per_job": _mean([c.messages_per_job for c in subset]),
            "provider_inbox_per_day": _mean([c.provider_inbox_per_day for c in subset]),
            "consumer_surplus_per_job": _mean([c.consumer_surplus_per_job for c in subset]),
            "provider_profit_per_job": _mean([c.provider_profit_per_job for c in subset]),
        }
    return out


def _loss(
    *,
    moments: dict[str, dict[str, float]],
    targets: dict[str, dict[str, float]],
    tol: dict[str, float],
) -> float:
    loss = 0.0
    for category, tcat in targets.items():
        for metric, target in tcat.items():
            sim = moments[category][metric]
            scale = float(tol.get(metric, max(1e-6, abs(target) * 0.10)))
            loss += ((sim - target) / scale) ** 2
    return loss


def _sample_candidate(rng: random.Random, base: FieldV2Params) -> FieldV2Params:
    """
    A small, "calibration knob" subset chosen to primarily affect observable moments:
    prices, cancels, match rates, and capacity/congestion.
    """
    return FieldV2Params(
        **{
            **asdict(base),
            "base_markup": float(rng.uniform(0.15, 0.35)),
            "budget_markup": float(rng.uniform(0.00, 0.20)),
            "demand_markup": float(rng.uniform(0.10, 0.30)),
            "cancel_base": float(rng.uniform(0.03, 0.10)),
            "provider_weekly_capacity": int(rng.randint(4, 8)),
            "provider_daily_screen_cap": int(rng.randint(6, 12)),
            "accept_threshold": float(rng.uniform(0.22, 0.30)),
            "idiosyncratic_noise_sd": float(rng.uniform(0.06, 0.12)),
            "value_scale_easy": float(rng.uniform(600.0, 950.0)),
            "value_scale_hard": float(rng.uniform(1100.0, 1700.0)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/field_v2_calibration_latest")
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--eval-seed", type=int, default=901)
    parser.add_argument("--evals", type=int, default=60)
    parser.add_argument("--cities", type=int, default=18)
    parser.add_argument("--weeks", type=int, default=1)
    parser.add_argument("--jobs-easy", type=int, default=25)
    parser.add_argument("--jobs-hard", type=int, default=25)
    parser.add_argument("--providers", type=int, default=40)
    parser.add_argument("--targets-json", default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = DEFAULT_TARGETS
    tol = DEFAULT_TOL
    if args.targets_json:
        overrides = json.loads(Path(args.targets_json).read_text(encoding="utf-8"))
        targets = overrides.get("targets", targets)
        tol = overrides.get("tolerance", tol)

    rng = random.Random(args.seed)  # nosec B311
    base = FieldV2Params()

    rows: list[dict[str, RowValue]] = []

    best_loss = math.inf
    best_params = base
    best_moments: dict[str, dict[str, float]] = {}

    def record(iter_idx: int, params: FieldV2Params, moments: dict[str, dict[str, float]]) -> None:
        row: dict[str, RowValue] = {
            "iter": iter_idx,
            "loss": round(
                _loss(moments=moments, targets=targets, tol=tol),
                6,
            ),
            "base_markup": round(params.base_markup, 4),
            "budget_markup": round(params.budget_markup, 4),
            "demand_markup": round(params.demand_markup, 4),
            "cancel_base": round(params.cancel_base, 4),
            "provider_weekly_capacity": int(params.provider_weekly_capacity),
            "provider_daily_screen_cap": int(params.provider_daily_screen_cap),
            "accept_threshold": round(params.accept_threshold, 4),
            "idiosyncratic_noise_sd": round(params.idiosyncratic_noise_sd, 4),
            "value_scale_easy": round(params.value_scale_easy, 2),
            "value_scale_hard": round(params.value_scale_hard, 2),
        }
        for category in ("easy", "hard"):
            for metric, value in moments[category].items():
                row[f"{category}_{metric}"] = round(value, 4)
        rows.append(row)

    # Baseline evaluation first.
    base_eval_params = FieldV2Params(
        **{
            **asdict(base),
            "compliance_ai": 1.0,
            "contamination_ai_in_control": 0.0,
            "compliance_central": 1.0,
        }
    )
    base_moments = _simulate_baseline_moments(
        seed=args.eval_seed,
        params=base_eval_params,
        cities=args.cities,
        weeks=args.weeks,
        jobs_easy=args.jobs_easy,
        jobs_hard=args.jobs_hard,
        providers=args.providers,
    )
    record(0, base_eval_params, base_moments)

    best_loss = _loss(moments=base_moments, targets=targets, tol=tol)
    best_params = base_eval_params
    best_moments = base_moments

    for i in range(1, args.evals + 1):
        cand = _sample_candidate(rng, base_eval_params)
        moments = _simulate_baseline_moments(
            seed=args.eval_seed,
            params=cand,
            cities=args.cities,
            weeks=args.weeks,
            jobs_easy=args.jobs_easy,
            jobs_hard=args.jobs_hard,
            providers=args.providers,
        )
        cand_loss = _loss(moments=moments, targets=targets, tol=tol)
        record(i, cand, moments)
        if cand_loss < best_loss:
            best_loss = cand_loss
            best_params = cand
            best_moments = moments
            log(
                logger,
                20,
                "calibration_best_update",
                iter=i,
                loss=round(best_loss, 6),
                params={
                    "base_markup": best_params.base_markup,
                    "budget_markup": best_params.budget_markup,
                    "demand_markup": best_params.demand_markup,
                    "cancel_base": best_params.cancel_base,
                    "provider_weekly_capacity": best_params.provider_weekly_capacity,
                    "provider_daily_screen_cap": best_params.provider_daily_screen_cap,
                    "accept_threshold": best_params.accept_threshold,
                    "idiosyncratic_noise_sd": best_params.idiosyncratic_noise_sd,
                },
            )

    _write_csv(rows, out_dir / "runs.csv")

    # Loss plot.
    pts = [(float(r["iter"]), float(r["loss"])) for r in rows]
    write_line_chart_svg(
        out_path=out_dir / "fig_loss.svg",
        title="Calibration: loss vs iteration",
        x_label="iteration",
        y_label="loss",
        series=[LineSeries(label="loss", points=pts)],
    )

    (out_dir / "targets.json").write_text(
        json.dumps({"targets": targets, "tolerance": tol}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    best_overrides = {
        "base_markup": best_params.base_markup,
        "budget_markup": best_params.budget_markup,
        "demand_markup": best_params.demand_markup,
        "cancel_base": best_params.cancel_base,
        "provider_weekly_capacity": best_params.provider_weekly_capacity,
        "provider_daily_screen_cap": best_params.provider_daily_screen_cap,
        "accept_threshold": best_params.accept_threshold,
        "idiosyncratic_noise_sd": best_params.idiosyncratic_noise_sd,
        "value_scale_easy": best_params.value_scale_easy,
        "value_scale_hard": best_params.value_scale_hard,
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(best_overrides, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "eval_settings.json").write_text(
        json.dumps(
            {
                "note": "Calibration evaluates a baseline with perfect compliance to reduce noise.",
                "compliance_ai": base_eval_params.compliance_ai,
                "contamination_ai_in_control": base_eval_params.contamination_ai_in_control,
                "compliance_central": base_eval_params.compliance_central,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "best_moments.json").write_text(
        json.dumps(best_moments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    cmd = (
        "uv run python -m econ_llm_preferences_experiment.field_sim_v2 --params-json "
        f"{out_dir / 'best_params.json'}"
    )
    readme = "\n".join(
        [
            "# FieldSim v2 calibration (moment matching)",
            "",
            "So what: picks a small set of parameters so the simulated marketplace has",
            "plausible baseline moments (prices, cancels, match rates, congestion) before we",
            "interpret welfare and mechanism comparisons.",
            "",
            f"Runs: {len(rows)} (including iter=0 baseline)",
            f"- best loss: {best_loss:.4f}",
            "",
            "Best-fit baseline moments (standard intake + search):",
            f"- easy: {best_moments['easy']}",
            f"- hard: {best_moments['hard']}",
            "",
            "Artifacts:",
            "- `runs.csv`: iteration-by-iteration params + moments + loss",
            "- `fig_loss.svg`: loss curve",
            "- `targets.json`: target moments + tolerances",
            "- `best_params.json`: overrides to pass via `--params-json`",
            "- `eval_settings.json`: evaluation settings used during calibration",
            "",
            "Use best params in FieldSim v2:",
            f"- `{cmd}`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    log(logger, 20, "calibration_written", out_dir=str(out_dir), best_loss=round(best_loss, 6))


if __name__ == "__main__":
    main()
