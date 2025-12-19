from __future__ import annotations

import math
import random
from dataclasses import dataclass

from econ_llm_preferences_experiment.models import DIMENSIONS, Category


@dataclass(frozen=True)
class HomeServicesTask:
    task_id: str
    label: str
    category: Category
    budget_median: float
    budget_sigma: float
    complexity_mu: float
    complexity_sd: float
    weirdness_base: float
    license_prob: float
    insurance_prob: float
    schedule_mean_share: float
    keywords: tuple[str, ...] = ()


_TASKS: tuple[HomeServicesTask, ...] = (
    # Easy / high-describability tasks.
    HomeServicesTask(
        task_id="tv_mounting",
        label="TV mounting",
        category="easy",
        budget_median=180.0,
        budget_sigma=0.30,
        complexity_mu=1.00,
        complexity_sd=0.12,
        weirdness_base=0.12,
        license_prob=0.03,
        insurance_prob=0.05,
        schedule_mean_share=0.60,
        keywords=("tv", "mount", "studs", "drywall"),
    ),
    HomeServicesTask(
        task_id="faucet_repair",
        label="Faucet repair / replacement",
        category="easy",
        budget_median=210.0,
        budget_sigma=0.32,
        complexity_mu=1.05,
        complexity_sd=0.14,
        weirdness_base=0.16,
        license_prob=0.04,
        insurance_prob=0.06,
        schedule_mean_share=0.55,
        keywords=("faucet", "leak", "sink"),
    ),
    HomeServicesTask(
        task_id="furniture_assembly",
        label="Furniture assembly",
        category="easy",
        budget_median=160.0,
        budget_sigma=0.35,
        complexity_mu=0.98,
        complexity_sd=0.10,
        weirdness_base=0.10,
        license_prob=0.02,
        insurance_prob=0.04,
        schedule_mean_share=0.65,
        keywords=("ikea", "assembly", "tools"),
    ),
    HomeServicesTask(
        task_id="house_cleaning",
        label="House cleaning",
        category="easy",
        budget_median=240.0,
        budget_sigma=0.28,
        complexity_mu=1.00,
        complexity_sd=0.10,
        weirdness_base=0.08,
        license_prob=0.02,
        insurance_prob=0.07,
        schedule_mean_share=0.55,
        keywords=("cleaning", "deep clean", "pets"),
    ),
    HomeServicesTask(
        task_id="drywall_patch",
        label="Drywall patch + paint touch-up",
        category="easy",
        budget_median=280.0,
        budget_sigma=0.33,
        complexity_mu=1.12,
        complexity_sd=0.16,
        weirdness_base=0.18,
        license_prob=0.04,
        insurance_prob=0.06,
        schedule_mean_share=0.50,
        keywords=("drywall", "patch", "paint"),
    ),
    # Hard / low-describability tasks.
    HomeServicesTask(
        task_id="electrical_panel",
        label="Electrical panel / wiring work",
        category="hard",
        budget_median=1100.0,
        budget_sigma=0.55,
        complexity_mu=1.45,
        complexity_sd=0.28,
        weirdness_base=0.42,
        license_prob=0.65,
        insurance_prob=0.55,
        schedule_mean_share=0.40,
        keywords=("breaker", "panel", "wiring", "permit"),
    ),
    HomeServicesTask(
        task_id="historic_staircase",
        label="Historic staircase repair",
        category="hard",
        budget_median=1600.0,
        budget_sigma=0.62,
        complexity_mu=1.65,
        complexity_sd=0.32,
        weirdness_base=0.55,
        license_prob=0.45,
        insurance_prob=0.45,
        schedule_mean_share=0.35,
        keywords=("staircase", "historic", "old house"),
    ),
    HomeServicesTask(
        task_id="foundation_crack",
        label="Foundation crack repair",
        category="hard",
        budget_median=2200.0,
        budget_sigma=0.70,
        complexity_mu=1.85,
        complexity_sd=0.36,
        weirdness_base=0.50,
        license_prob=0.55,
        insurance_prob=0.60,
        schedule_mean_share=0.30,
        keywords=("foundation", "crack", "structural"),
    ),
    HomeServicesTask(
        task_id="custom_tile",
        label="Custom tile work (bath/kitchen)",
        category="hard",
        budget_median=1300.0,
        budget_sigma=0.58,
        complexity_mu=1.55,
        complexity_sd=0.30,
        weirdness_base=0.40,
        license_prob=0.25,
        insurance_prob=0.45,
        schedule_mean_share=0.35,
        keywords=("tile", "grout", "waterproofing"),
    ),
    HomeServicesTask(
        task_id="asbestos",
        label="Asbestos testing / remediation",
        category="hard",
        budget_median=2600.0,
        budget_sigma=0.75,
        complexity_mu=2.05,
        complexity_sd=0.40,
        weirdness_base=0.62,
        license_prob=0.80,
        insurance_prob=0.70,
        schedule_mean_share=0.25,
        keywords=("asbestos", "testing", "abatement"),
    ),
)


def tasks_for_category(category: Category) -> tuple[HomeServicesTask, ...]:
    return tuple(t for t in _TASKS if t.category == category)


def task_by_id(task_id: str) -> HomeServicesTask:
    for t in _TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(f"Unknown task_id: {task_id}")


def sample_task(*, rng: random.Random, category: Category) -> HomeServicesTask:
    tasks = tasks_for_category(category)
    if not tasks:
        raise ValueError(f"No tasks for category={category}")
    return rng.choice(list(tasks))


def draw_budget(*, rng: random.Random, task: HomeServicesTask) -> float:
    # lognormal: median = exp(mu)
    return float(rng.lognormvariate(math.log(task.budget_median), task.budget_sigma))


def draw_complexity(*, rng: random.Random, task: HomeServicesTask) -> float:
    raw = rng.gauss(task.complexity_mu, task.complexity_sd)
    lo = 0.70 if task.category == "easy" else 0.80
    hi = 1.60 if task.category == "easy" else 2.60
    return float(max(lo, min(hi, raw)))


def draw_weirdness(*, rng: random.Random, task: HomeServicesTask, complexity: float) -> float:
    base = task.weirdness_base + (0.20 if task.category == "easy" else 0.28) * (complexity - 1.0)
    noise_sd = 0.12 if task.category == "easy" else 0.20
    x = base + rng.gauss(0.0, noise_sd)
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


_CITY_NAMES: tuple[str, ...] = (
    "Austin",
    "Boston",
    "Chicago",
    "Denver",
    "Los Angeles",
    "Miami",
    "New York",
    "Phoenix",
    "San Francisco",
    "Seattle",
)


def city_name(city_id: str) -> str:
    digits = "".join(ch for ch in city_id if ch.isdigit())
    idx = int(digits) if digits else sum(ord(ch) for ch in city_id)
    return _CITY_NAMES[idx % len(_CITY_NAMES)]


def _level(x: float) -> str:
    if x >= 0.30:
        return "high"
    if x >= 0.18:
        return "medium"
    if x >= 0.08:
        return "low"
    return "none"


def format_schedule_slots(
    *,
    slots: frozenset[int],
    horizon_days: int,
    slots_per_day: int,
    max_windows: int = 3,
) -> str:
    if not slots:
        return "no availability provided"
    labels = ["morning", "afternoon", "evening"]
    windows = []
    for s in sorted(slots):
        day = (s // slots_per_day) + 1
        part = labels[s % slots_per_day] if slots_per_day == 3 else f"slot{s % slots_per_day}"
        windows.append(f"day {day} {part}")
    if len(windows) > max_windows:
        windows = windows[:max_windows] + ["(more)"]
    return ", ".join(windows)


def standard_form_job_intake(
    *,
    task: HomeServicesTask,
    city_id: str,
    budget_reported: float,
    schedule_slots_reported: frozenset[int] | None,
    requires_license_reported: bool | None,
    requires_insurance_reported: bool | None,
    weights_hat: tuple[float, ...],
    horizon_days: int,
    slots_per_day: int,
) -> str:
    top = sorted(range(len(weights_hat)), key=lambda idx: weights_hat[idx], reverse=True)[:3]
    lines = [
        "Standard intake form submission",
        f"City: {city_name(city_id)}",
        f"Service: {task.label}",
        f"Budget (reported): ${budget_reported:.0f}",
        "Timeline / availability:",
    ]
    if schedule_slots_reported is None:
        lines.append("- not provided (free-text only)")
    else:
        sched = format_schedule_slots(
            slots=schedule_slots_reported,
            horizon_days=horizon_days,
            slots_per_day=slots_per_day,
        )
        lines.append(f"- {sched}")

    lines.append("Requirements:")
    lines.append(f"- licensed required? {requires_license_reported}")
    lines.append(f"- insured required? {requires_insurance_reported}")
    lines.append("Stated priorities (coarse):")
    for idx in top:
        lines.append(f"- {DIMENSIONS[idx]}: {_level(weights_hat[idx])}")
    lines.append("Notes box: (optional)")
    lines.append("- (left blank)")
    return "\n".join(lines)


def ai_chat_job_intake(
    *,
    task: HomeServicesTask,
    city_id: str,
    budget_reported: float,
    schedule_slots_true: frozenset[int],
    schedule_slots_reported: frozenset[int] | None,
    requires_license_true: bool,
    requires_insurance_true: bool,
    requires_license_reported: bool | None,
    requires_insurance_reported: bool | None,
    complexity_true: float,
    weirdness_true: float,
    weights_hat: tuple[float, ...],
    horizon_days: int,
    slots_per_day: int,
    rng: random.Random,
) -> str:
    def yn(x: bool) -> str:
        return "yes" if x else "no"

    def maybe(x: bool | None) -> str:
        if x is None:
            return "not sure"
        return yn(bool(x))

    city = city_name(city_id)
    urgency = (
        "ASAP" if weights_hat[2] >= 0.30 else "this week" if weights_hat[2] >= 0.18 else "flexible"
    )
    price_pref = (
        "I’m budget-conscious"
        if weights_hat[0] >= 0.30
        else "I can pay for quality"
        if weights_hat[1] >= 0.30
        else "I’m somewhat flexible on price"
    )
    comm_pref = "Please send updates." if weights_hat[3] >= 0.18 else "No need for many updates."

    true_sched = format_schedule_slots(
        slots=schedule_slots_true, horizon_days=horizon_days, slots_per_day=slots_per_day
    )
    rep_sched = (
        "not captured"
        if schedule_slots_reported is None
        else format_schedule_slots(
            slots=schedule_slots_reported, horizon_days=horizon_days, slots_per_day=slots_per_day
        )
    )
    note = ""
    if weirdness_true > 0.60 and rng.random() < 0.75:
        note = "It’s an old place and there might be surprises once you open it up."
    elif complexity_true > 1.60 and rng.random() < 0.70:
        note = "This might be more involved than it sounds; please sanity-check scope."

    lines = [
        "Agent: Hi — I’ll ask a few quick questions to write a clear request.",
        f"User: I’m in {city}. I need help with {task.label.lower()}.",
        "Agent: When do you want this done?",
        f"User: {urgency}.",
        "Agent: What budget range are you aiming for?",
        f"User: Around ${budget_reported:.0f}.",
        f"User: {price_pref}.",
        "Agent: Any constraints like licensing/insurance requirements?",
        f"User: licensed required? {yn(requires_license_true)}; "
        f"insured required? {yn(requires_insurance_true)}.",
        f"Agent: I heard: licensed={maybe(requires_license_reported)}, "
        f"insured={maybe(requires_insurance_reported)}.",
        "Agent: What times work for you?",
        f"User: {true_sched}.",
        f"Agent: Captured availability: {rep_sched}.",
        "Agent: Anything else I should include?",
        f"User: {comm_pref} {note}".strip(),
        "Agent: Great — I’ll send this as a structured request.",
    ]
    return "\n".join(lines)
