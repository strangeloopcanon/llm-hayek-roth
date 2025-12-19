from __future__ import annotations

import random
from dataclasses import dataclass

from econ_llm_preferences_experiment.models import DIMENSIONS, AgentTruth, Category, MarketInstance


def _normalize(weights: list[float]) -> tuple[float, ...]:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return tuple(1.0 / len(weights) for _ in weights)
    return tuple(max(0.0, w) / total for w in weights)


def _dirichlet(rng: random.Random, k: int, *, alpha: float = 1.0) -> tuple[float, ...]:
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    total = sum(draws)
    if total <= 0:
        return tuple(1.0 / k for _ in range(k))
    return tuple(x / total for x in draws)


def draw_agent_weights(rng: random.Random, *, category: Category) -> tuple[float, ...]:
    raw = [max(0.0, rng.gauss(0.0, 1.0)) for _ in DIMENSIONS]
    if category == "easy":
        top = sorted(range(len(raw)), key=lambda k: raw[k], reverse=True)[:3]
        raw = [raw[k] if k in top else 0.0 for k in range(len(raw))]
    return _normalize(raw)


@dataclass(frozen=True)
class MarketParams:
    n_customers: int = 30
    n_providers: int = 30
    epsilon: float = 0.1
    accept_threshold: float = 0.25
    idiosyncratic_noise_sd: float = 0.08


def generate_population(
    *,
    rng: random.Random,
    category: Category,
    n_customers: int,
    n_providers: int,
) -> tuple[tuple[AgentTruth, ...], tuple[AgentTruth, ...]]:
    customers = tuple(
        AgentTruth(
            agent_id=f"c{i:03d}",
            side="customer",
            category=category,
            weights=draw_agent_weights(rng, category=category),
        )
        for i in range(n_customers)
    )
    providers = tuple(
        AgentTruth(
            agent_id=f"p{j:03d}",
            side="provider",
            category=category,
            weights=draw_agent_weights(rng, category=category),
        )
        for j in range(n_providers)
    )
    return customers, providers


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def generate_market_instance(
    *,
    rng: random.Random,
    customers: tuple[AgentTruth, ...],
    providers: tuple[AgentTruth, ...],
    idiosyncratic_noise_sd: float,
) -> MarketInstance:
    k = len(DIMENSIONS)
    customer_attributes = tuple(_dirichlet(rng, k, alpha=1.2) for _ in customers)
    provider_attributes = tuple(_dirichlet(rng, k, alpha=1.2) for _ in providers)

    v_customer: list[list[float]] = []
    for c in customers:
        row: list[float] = []
        for y in provider_attributes:
            base = _dot(c.weights, y)
            noisy = base + rng.gauss(0.0, idiosyncratic_noise_sd)
            row.append(_clip01(noisy))
        v_customer.append(row)

    v_provider: list[list[float]] = []
    for p in providers:
        row = []
        for x in customer_attributes:
            base = _dot(p.weights, x)
            noisy = base + rng.gauss(0.0, idiosyncratic_noise_sd)
            row.append(_clip01(noisy))
        v_provider.append(row)

    return MarketInstance(
        customers=customers,
        providers=providers,
        customer_attributes=customer_attributes,
        provider_attributes=provider_attributes,
        v_customer=tuple(tuple(row) for row in v_customer),
        v_provider=tuple(tuple(row) for row in v_provider),
    )


def inferred_value_matrix(
    *,
    weights_by_agent: tuple[tuple[float, ...], ...],
    partner_attributes: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    mat: list[list[float]] = []
    for w in weights_by_agent:
        mat.append([_clip01(_dot(w, a)) for a in partner_attributes])
    return tuple(tuple(row) for row in mat)


def preference_density_proxy(
    *,
    v_true: tuple[tuple[float, ...], ...],
    v_hat: tuple[tuple[float, ...], ...],
    epsilon: float,
) -> float:
    total = 0
    close = 0
    for row_true, row_hat in zip(v_true, v_hat, strict=True):
        for t, h in zip(row_true, row_hat, strict=True):
            total += 1
            if abs(t - h) <= epsilon:
                close += 1
    return close / total if total else 0.0
