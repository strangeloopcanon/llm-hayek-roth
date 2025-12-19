from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

from econ_llm_preferences_experiment.econometrics import ols_cluster_robust
from econ_llm_preferences_experiment.field_sim_v2 import (
    CityWeekAssignment,
    Elicitation,
    FieldV2Params,
    JobOutcome,
    Mechanism,
    simulate_city_week,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log
from econ_llm_preferences_experiment.plotting import LineSeries, write_line_chart_svg

logger = get_logger(__name__)

RowValue: TypeAlias = str | float | int


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    q = 0.0 if q < 0.0 else 1.0 if q > 1.0 else q
    ys = sorted(xs)
    idx = int(round(q * (len(ys) - 1)))
    return float(ys[idx])


def _write_csv(rows: list[dict[str, RowValue]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_list(arg: str, *, cast: type) -> list[Any]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [cast(p) for p in parts]


def _arm_label(ai: int, central: int) -> str:
    if ai == 0 and central == 0:
        return "standard_search"
    if ai == 0 and central == 1:
        return "standard_central"
    if ai == 1 and central == 0:
        return "ai_search"
    return "ai_central"


def _did_from_jobs(*, jobs: list[JobOutcome], outcome: str) -> dict[str, float]:
    """
    Simple DiD by category using job-level means.
    """
    by_cat_arm: dict[tuple[str, str], list[float]] = {}
    for j in jobs:
        key = (j.category, _arm_label(j.ai, j.central))
        by_cat_arm.setdefault(key, []).append(_as_float(getattr(j, outcome)))

    def m(cat: str, arm: str) -> float:
        return _mean(by_cat_arm.get((cat, arm), []))

    def did(cat: str) -> float:
        return (m(cat, "ai_central") - m(cat, "standard_central")) - (
            m(cat, "ai_search") - m(cat, "standard_search")
        )

    return {"did_easy": did("easy"), "did_hard": did("hard"), "triple": did("hard") - did("easy")}


def _triple_from_regression(*, jobs: list[JobOutcome], outcome: str) -> tuple[float, float]:
    y = [_as_float(getattr(j, outcome)) for j in jobs]
    x = []
    clusters = []
    for j in jobs:
        hard = 1.0 if j.category == "hard" else 0.0
        ai = float(j.ai)
        central = float(j.central)
        x.append(
            [
                1.0,
                ai,
                central,
                ai * central,
                hard,
                ai * hard,
                central * hard,
                ai * central * hard,
            ]
        )
        clusters.append(j.cell_id)
    res = ols_cluster_robust(y=y, x=x, clusters=clusters)
    triple_idx = 7
    return float(res.coef[triple_idx]), float(res.p[triple_idx])


def _run_once(
    *,
    seed: int,
    cities: int,
    weeks: int,
    jobs_easy: int,
    jobs_hard: int,
    providers: int,
    params: FieldV2Params,
) -> list[JobOutcome]:
    rng = random.Random(seed)  # nosec B311

    arms: list[tuple[Elicitation, Mechanism]] = [
        ("standard", "search"),
        ("standard", "central"),
        ("ai", "search"),
        ("ai", "central"),
    ]

    cell_keys: list[tuple[str, int]] = [
        (f"city{city_idx:03d}", week) for city_idx in range(cities) for week in range(weeks)
    ]
    reps = (len(cell_keys) + len(arms) - 1) // len(arms)
    easy_arms = (arms * reps)[: len(cell_keys)]
    hard_arms = (arms * reps)[: len(cell_keys)]
    rng.shuffle(easy_arms)
    rng.shuffle(hard_arms)

    assignments: list[CityWeekAssignment] = []
    for idx, (city_id, week) in enumerate(cell_keys):
        assignments.append(
            CityWeekAssignment(
                city_id=city_id,
                week=week,
                cell_easy=easy_arms[idx],
                cell_hard=hard_arms[idx],
            )
        )

    job_rows: list[JobOutcome] = []
    for a in assignments:
        city_rng = random.Random(rng.randrange(1_000_000_000))  # nosec B311
        _cells, jobs, _providers, _meas = simulate_city_week(
            rng=city_rng,
            assignment=a,
            params=params,
            n_jobs_easy=jobs_easy,
            n_jobs_hard=jobs_hard,
            n_providers=providers,
        )
        job_rows.extend(jobs)

    return job_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/field_v2_sensitivity_latest")
    parser.add_argument("--seed-base", type=int, default=300)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--cities", type=int, default=40)
    parser.add_argument("--weeks", type=int, default=2)
    parser.add_argument("--jobs-easy", type=int, default=30)
    parser.add_argument("--jobs-hard", type=int, default=30)
    parser.add_argument("--providers", type=int, default=45)
    parser.add_argument("--attention-cost", type=str, default="0.25")
    parser.add_argument("--central-rec-k", type=str, default="5")
    parser.add_argument("--search-k", type=str, default="2")
    parser.add_argument("--std-misclass-hard", type=str, default="0.55,0.70,0.85")
    parser.add_argument("--max-runs", type=int, default=80)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    attention_costs = _parse_list(args.attention_cost, cast=float)
    rec_ks = _parse_list(args.central_rec_k, cast=int)
    search_ks = _parse_list(args.search_k, cast=int)
    misclasses = _parse_list(args.std_misclass_hard, cast=float)

    seeds = [args.seed_base + i for i in range(args.n_seeds)]

    rows: list[dict[str, RowValue]] = []
    run_count = 0
    for attn in attention_costs:
        for rec_k in rec_ks:
            for search_k in search_ks:
                for misclass in misclasses:
                    for seed in seeds:
                        if run_count >= args.max_runs:
                            break
                        run_count += 1
                        params = FieldV2Params(
                            attention_cost=float(attn),
                            central_rec_k=int(rec_k),
                            search_k_per_day=int(search_k),
                            std_weight_misclass_hard=float(misclass),
                        )
                        log(
                            logger,
                            20,
                            "field_v2_sensitivity_run",
                            run=run_count,
                            seed=seed,
                            attention_cost=float(attn),
                            central_rec_k=int(rec_k),
                            search_k_per_day=int(search_k),
                            std_weight_misclass_hard=float(misclass),
                        )

                        jobs = _run_once(
                            seed=seed,
                            cities=args.cities,
                            weeks=args.weeks,
                            jobs_easy=args.jobs_easy,
                            jobs_hard=args.jobs_hard,
                            providers=args.providers,
                            params=params,
                        )

                        did_match = _did_from_jobs(jobs=jobs, outcome="matched")
                        did_welfare = _did_from_jobs(jobs=jobs, outcome="net_welfare")
                        triple_match, p_match = _triple_from_regression(
                            jobs=jobs, outcome="matched"
                        )
                        triple_w, p_w = _triple_from_regression(jobs=jobs, outcome="net_welfare")

                        rows.append(
                            {
                                "seed": seed,
                                "cities": args.cities,
                                "weeks": args.weeks,
                                "jobs_easy": args.jobs_easy,
                                "jobs_hard": args.jobs_hard,
                                "providers": args.providers,
                                "attention_cost": float(attn),
                                "central_rec_k": int(rec_k),
                                "search_k_per_day": int(search_k),
                                "std_weight_misclass_hard": float(misclass),
                                "did_easy_match": round(did_match["did_easy"], 4),
                                "did_hard_match": round(did_match["did_hard"], 4),
                                "triple_did_match": round(did_match["triple"], 4),
                                "triple_reg_match": round(triple_match, 4),
                                "p_triple_reg_match": round(p_match, 4),
                                "did_easy_welfare": round(did_welfare["did_easy"], 4),
                                "did_hard_welfare": round(did_welfare["did_hard"], 4),
                                "triple_did_welfare": round(did_welfare["triple"], 4),
                                "triple_reg_welfare": round(triple_w, 4),
                                "p_triple_reg_welfare": round(p_w, 4),
                            }
                        )
                    if run_count >= args.max_runs:
                        break
                if run_count >= args.max_runs:
                    break
            if run_count >= args.max_runs:
                break
        if run_count >= args.max_runs:
            break

    _write_csv(rows, out_dir / "runs.csv")

    # Aggregate by rec_k for a quick visual.
    by_rec: dict[int, list[float]] = {}
    for r in rows:
        by_rec.setdefault(int(_as_float(r["central_rec_k"])), []).append(
            _as_float(r["triple_reg_welfare"])
        )

    series = [
        LineSeries(
            label=f"attn={args.attention_cost}",
            points=[(float(k), _mean(v)) for k, v in sorted(by_rec.items())],
        )
    ]
    write_line_chart_svg(
        out_path=out_dir / "fig_triple_welfare_vs_rec_k.svg",
        title="Sensitivity: triple interaction on net_welfare vs rec_k",
        x_label="central_rec_k",
        y_label="coef(ai×central×hard) on net_welfare",
        series=series,
    )

    triples = [_as_float(r["triple_reg_welfare"]) for r in rows]
    pos_share = sum(1 for t in triples if t > 0.0) / len(triples) if triples else 0.0
    sig_share = (
        sum(1 for r in rows if _as_float(r["p_triple_reg_welfare"]) < 0.05) / len(rows)
        if rows
        else 0.0
    )

    summary = {
        "n_runs": len(rows),
        "pos_share_triple_welfare": pos_share,
        "sig_share_triple_welfare_p05": sig_share,
        "triple_welfare_p10": _quantile(triples, 0.10),
        "triple_welfare_p50": _quantile(triples, 0.50),
        "triple_welfare_p90": _quantile(triples, 0.90),
        "defaults": {
            "cities": args.cities,
            "weeks": args.weeks,
            "jobs_easy": args.jobs_easy,
            "jobs_hard": args.jobs_hard,
            "providers": args.providers,
        },
        "grid": {
            "attention_cost": attention_costs,
            "central_rec_k": rec_ks,
            "search_k_per_day": search_ks,
            "std_weight_misclass_hard": misclasses,
            "seeds": seeds,
        },
        "max_runs": args.max_runs,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    readme = "\n".join(
        [
            "# FieldSim v2 sensitivity sweep",
            "",
            "So what: checks whether the core mechanism interaction (ai×central×hard) is robust",
            "across seeds and a small grid of key knobs.",
            "",
            f"Runs: {len(rows)}",
            f"- share(triple_welfare>0): {pos_share:.2f}",
            f"- share(p<0.05): {sig_share:.2f}",
            f"- triple_welfare quantiles: p10={summary['triple_welfare_p10']:.2f}, "
            f"p50={summary['triple_welfare_p50']:.2f}, p90={summary['triple_welfare_p90']:.2f}",
            "",
            "Artifacts:",
            "- `runs.csv`: run-level DiD + regression triples",
            "- `summary.json`: aggregate quantiles and grid",
            "- `fig_triple_welfare_vs_rec_k.svg`: quick robustness plot",
            "",
            "Run:",
            "- `uv run python -m econ_llm_preferences_experiment.field_sim_v2_sensitivity`",
            "",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "params_defaults": asdict(FieldV2Params()),
                "args": vars(args),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    log(logger, 20, "field_v2_sensitivity_written", out_dir=str(out_dir))


if __name__ == "__main__":
    main()
