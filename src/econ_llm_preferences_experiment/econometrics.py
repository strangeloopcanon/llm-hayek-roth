from __future__ import annotations

import math
from dataclasses import dataclass


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _transpose(x: list[list[float]]) -> list[list[float]]:
    if not x:
        return []
    n_cols = len(x[0])
    if any(len(row) != n_cols for row in x):
        raise ValueError("Matrix must be rectangular")
    return [[row[j] for row in x] for j in range(n_cols)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b:
        return []
    n = len(a)
    k = len(a[0])
    if any(len(row) != k for row in a):
        raise ValueError("Left matrix must be rectangular")
    if any(len(row) != len(b[0]) for row in b):
        raise ValueError("Right matrix must be rectangular")
    if len(b) != k:
        raise ValueError("Inner dimensions mismatch")
    m = len(b[0])
    out = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for t in range(k):
            ait = a[i][t]
            for j in range(m):
                out[i][j] += ait * b[t][j]
    return out


def _matvec(a: list[list[float]], v: list[float]) -> list[float]:
    if not a:
        return []
    if any(len(row) != len(v) for row in a):
        raise ValueError("Dimension mismatch")
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]


def _outer(u: list[float], v: list[float]) -> list[list[float]]:
    return [[ui * vj for vj in v] for ui in u]


def _invert_square(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("Expected a non-empty square matrix")

    aug = [
        [float(a[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]


@dataclass(frozen=True)
class OLSResult:
    coef: list[float]
    se: list[float]
    t: list[float]
    p: list[float]
    n_obs: int
    n_params: int
    n_clusters: int


def ols_cluster_robust(
    *, y: list[float], x: list[list[float]], clusters: list[str], df_correction: bool = True
) -> OLSResult:
    n = len(y)
    if n == 0:
        raise ValueError("Need at least one observation")
    if len(x) != n or len(clusters) != n:
        raise ValueError("Length mismatch: y, x, clusters")

    p = len(x[0])
    if any(len(row) != p for row in x):
        raise ValueError("Design matrix must be rectangular")
    if p == 0:
        raise ValueError("Need at least one regressor column")

    # beta = (X'X)^{-1} X'y
    xt = _transpose(x)
    xtx = _matmul(xt, x)
    xty = _matvec(xt, y)
    xtx_inv = _invert_square(xtx)
    beta = _matvec(xtx_inv, xty)

    residuals = [y[i] - sum(x[i][j] * beta[j] for j in range(p)) for i in range(n)]

    groups: dict[str, list[int]] = {}
    for idx, g in enumerate(clusters):
        groups.setdefault(g, []).append(idx)
    g_count = len(groups)

    meat = [[0.0 for _ in range(p)] for _ in range(p)]
    for g_idx in groups.values():
        s_g = [0.0 for _ in range(p)]
        for i in g_idx:
            ui = residuals[i]
            for j in range(p):
                s_g[j] += x[i][j] * ui
        og = _outer(s_g, s_g)
        for r in range(p):
            for c in range(p):
                meat[r][c] += og[r][c]

    v = _matmul(_matmul(xtx_inv, meat), xtx_inv)
    if df_correction and g_count > 1 and n > p:
        scale = (g_count / (g_count - 1.0)) * ((n - 1.0) / (n - float(p)))
        for r in range(p):
            for c in range(p):
                v[r][c] *= scale

    se = [math.sqrt(v[j][j]) if v[j][j] > 0 else 0.0 for j in range(p)]
    t = [beta[j] / se[j] if se[j] > 0 else 0.0 for j in range(p)]
    pvals = [2.0 * (1.0 - _normal_cdf(abs(tj))) for tj in t]

    return OLSResult(
        coef=beta,
        se=se,
        t=t,
        p=pvals,
        n_obs=n,
        n_params=p,
        n_clusters=g_count,
    )
