from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DIMENSIONS: tuple[str, ...] = (
    "price_sensitivity",
    "quality_focus",
    "speed_urgency",
    "communication_fit",
    "weirdness_tolerance",
    "schedule_flexibility",
)

Side = Literal["customer", "provider"]
Category = Literal["easy", "hard"]


@dataclass(frozen=True)
class AgentTruth:
    agent_id: str
    side: Side
    category: Category
    weights: tuple[float, ...]  # len == len(DIMENSIONS); sum to 1


@dataclass(frozen=True)
class AgentInferred:
    agent_id: str
    side: Side
    category: Category
    weights: tuple[float, ...]  # len == len(DIMENSIONS); sum to 1
    tags: tuple[str, ...]


@dataclass(frozen=True)
class MarketInstance:
    customers: tuple[AgentTruth, ...]
    providers: tuple[AgentTruth, ...]
    customer_attributes: tuple[tuple[float, ...], ...]  # x_i in [0,1]^K
    provider_attributes: tuple[tuple[float, ...], ...]  # y_j in [0,1]^K
    v_customer: tuple[tuple[float, ...], ...]  # v_i(j)
    v_provider: tuple[tuple[float, ...], ...]  # v_j(i)


@dataclass(frozen=True)
class MatchOutcome:
    matches: tuple[tuple[int, int], ...]  # (i_index, j_index)
    proposals: int
    accept_decisions: int
    rounds: int
