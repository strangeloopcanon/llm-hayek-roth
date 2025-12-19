from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from econ_llm_preferences_experiment.field_sim_v2 import FieldV2Params, _make_jobs_for_cell
from econ_llm_preferences_experiment.home_services import (
    ai_chat_job_intake,
    standard_form_job_intake,
    task_by_id,
)
from econ_llm_preferences_experiment.logging_utils import get_logger, log

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/intakes_latest")
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--city", default="city008")
    parser.add_argument("--week", type=int, default=0)
    parser.add_argument("--n-easy", type=int, default=6)
    parser.add_argument("--n-hard", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = FieldV2Params()
    rng = random.Random(args.seed)  # nosec B311

    easy_jobs, easy_std, easy_ai = _make_jobs_for_cell(
        rng=rng,
        city_id=args.city,
        week=args.week,
        category="easy",
        cell_id=f"{args.city}_w{args.week:02d}_easy",
        n_jobs=args.n_easy,
        params=params,
    )
    hard_jobs, hard_std, hard_ai = _make_jobs_for_cell(
        rng=rng,
        city_id=args.city,
        week=args.week,
        category="hard",
        cell_id=f"{args.city}_w{args.week:02d}_hard",
        n_jobs=args.n_hard,
        params=params,
    )

    samples = []
    for job in list(easy_jobs) + list(hard_jobs):
        task = task_by_id(job.task_id)
        sj_std = easy_std[job.job_id] if job.category == "easy" else hard_std[job.job_id]
        sj_ai = easy_ai[job.job_id] if job.category == "easy" else hard_ai[job.job_id]

        std_text = standard_form_job_intake(
            task=task,
            city_id=job.city_id,
            budget_reported=sj_std.budget_reported,
            schedule_slots_reported=sj_std.schedule_slots_reported,
            requires_license_reported=sj_std.requires_license_reported,
            requires_insurance_reported=sj_std.requires_insurance_reported,
            weights_hat=sj_std.weights_hat,
            horizon_days=params.horizon_days,
            slots_per_day=params.slots_per_day,
        )
        chat_text = ai_chat_job_intake(
            task=task,
            city_id=job.city_id,
            budget_reported=sj_ai.budget_reported,
            schedule_slots_true=job.schedule_slots,
            schedule_slots_reported=sj_ai.schedule_slots_reported,
            requires_license_true=job.requires_license,
            requires_insurance_true=job.requires_insurance,
            requires_license_reported=sj_ai.requires_license_reported,
            requires_insurance_reported=sj_ai.requires_insurance_reported,
            complexity_true=job.complexity,
            weirdness_true=job.weirdness,
            weights_hat=sj_ai.weights_hat,
            horizon_days=params.horizon_days,
            slots_per_day=params.slots_per_day,
            rng=rng,
        )

        samples.append(
            {
                "job_id": job.job_id,
                "category": job.category,
                "task_id": job.task_id,
                "task_label": job.task_label,
                "budget_true": round(job.budget_true, 2),
                "complexity_true": round(job.complexity, 3),
                "weirdness_true": round(job.weirdness, 3),
                "requires_license_true": bool(job.requires_license),
                "requires_insurance_true": bool(job.requires_insurance),
                "standard_form_text": std_text,
                "ai_chat_text": chat_text,
            }
        )

    (out_dir / "intake_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# Intake verisimilitude samples (FieldSim v2 primitives)",
        "",
        "So what: realistic, human-readable job intakes that correspond to the same underlying",
        "structured primitives used by FieldSim v2 (budget, schedule, licensing/insurance, etc.).",
        "",
        "Artifacts:",
        "- `intake_samples.json`: structured sample texts + latent primitives",
        "",
    ]
    for s in samples:
        md_lines.extend(
            [
                f"## {s['job_id']} ({s['category']}: {s['task_label']})",
                "",
                f"- budget_true: ${s['budget_true']}",
                f"- complexity_true: {s['complexity_true']}",
                f"- weirdness_true: {s['weirdness_true']}",
                f"- requires_license_true: {s['requires_license_true']}",
                f"- requires_insurance_true: {s['requires_insurance_true']}",
                "",
                "### Standard form",
                "```",
                str(s["standard_form_text"]),
                "```",
                "",
                "### AI chat",
                "```",
                str(s["ai_chat_text"]),
                "```",
                "",
            ]
        )
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    log(logger, 20, "intake_samples_written", out_dir=str(out_dir), n=len(samples))


if __name__ == "__main__":
    main()
