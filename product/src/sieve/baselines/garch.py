"""GARCH(1,1) baselines (verbatim recursion from research `generators.py`).

Parameters come from the suite (per-index joint MLE frozen at the research
commit); no fitting happens at evaluation time.
"""

from __future__ import annotations

import numpy as np

BURN = 500


def _path(n, rng, omega, alpha, beta, innov):
    total = n + BURN
    e = innov(total)
    r = np.empty(total)
    s2 = omega / max(1e-8, 1 - alpha - beta)
    for i in range(total):
        r[i] = np.sqrt(s2) * e[i]
        s2 = omega + alpha * r[i] ** 2 + beta * s2
    return r[BURN:]


def generate_norm(n: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    o, a, b = params["garch"]
    return _path(n, rng, o, a, b, lambda m: rng.standard_normal(m))


def generate_t(n: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    o, a, b, df = params["garch_t"]
    sc = np.sqrt(df / (df - 2.0))
    return _path(n, rng, o, a, b, lambda m: rng.standard_t(df, m) / sc)
