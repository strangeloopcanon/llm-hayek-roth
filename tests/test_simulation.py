from __future__ import annotations

import random

from econ_llm_preferences_experiment.simulation import (
    MarketParams,
    generate_market_instance,
    generate_population,
    preference_density_proxy,
)


def test_generate_population_weights_sum_to_one() -> None:
    rng = random.Random(0)
    customers, providers = generate_population(
        rng=rng, category="easy", n_customers=5, n_providers=5
    )
    for agent in customers + providers:
        assert abs(sum(agent.weights) - 1.0) < 1e-9


def test_preference_density_proxy_bounds() -> None:
    v_true = ((0.1, 0.2), (0.3, 0.4))
    v_hat = ((0.1, 0.25), (0.0, 0.5))
    d = preference_density_proxy(v_true=v_true, v_hat=v_hat, epsilon=0.05)
    assert 0.0 <= d <= 1.0


def test_generate_market_instance_shapes() -> None:
    rng = random.Random(1)
    params = MarketParams(n_customers=4, n_providers=3)
    customers, providers = generate_population(
        rng=rng, category="hard", n_customers=params.n_customers, n_providers=params.n_providers
    )
    market = generate_market_instance(
        rng=rng,
        customers=customers,
        providers=providers,
        idiosyncratic_noise_sd=params.idiosyncratic_noise_sd,
    )
    assert len(market.v_customer) == params.n_customers
    assert len(market.v_customer[0]) == params.n_providers
    assert len(market.v_provider) == params.n_providers
    assert len(market.v_provider[0]) == params.n_customers
