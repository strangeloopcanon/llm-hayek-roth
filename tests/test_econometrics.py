from __future__ import annotations

from econ_llm_preferences_experiment.econometrics import ols_cluster_robust


def test_ols_cluster_robust_recovers_treatment_effect() -> None:
    # Simple clustered design: y ~= 1 + 2*treat + small noise.
    y: list[float] = []
    x: list[list[float]] = []
    clusters: list[str] = []

    def add_cluster(cluster: str, treat: int, ys: list[float]) -> None:
        for val in ys:
            y.append(val)
            x.append([1.0, float(treat)])
            clusters.append(cluster)

    add_cluster("A", 0, [1.0, 1.1, 0.9, 1.2, 0.8])
    add_cluster("B", 0, [1.05, 0.95, 1.1, 0.9, 1.0])
    add_cluster("C", 1, [3.0, 2.9, 3.1, 3.05, 2.95])
    add_cluster("D", 1, [2.8, 3.2, 3.0, 3.1, 2.9])

    res = ols_cluster_robust(y=y, x=x, clusters=clusters)
    assert len(res.coef) == 2
    assert abs(res.coef[1] - 2.0) < 0.15
    assert res.n_clusters == 4
