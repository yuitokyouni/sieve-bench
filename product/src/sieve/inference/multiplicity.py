"""Multiple-testing adjustments.

Both procedures are available; the suite manifest declares which applies and
the bundle records it. Holm controls FWER (suitable when the family is the
set of confirmatory suite tests); Benjamini-Hochberg controls FDR (the research
default for exploratory grids).
"""

from __future__ import annotations

import numpy as np


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    ok = np.where(np.isfinite(p))[0]
    adj = np.full(len(p), np.nan)
    m = len(ok)
    if m == 0:
        return adj.tolist()
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    ok = np.where(np.isfinite(p))[0]
    q = np.full(len(p), np.nan)
    if len(ok) == 0:
        return q.tolist()
    order = ok[np.argsort(p[ok])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1.0))
        q[i] = prev
    return q.tolist()


ADJUSTMENTS = {"holm": holm, "benjamini_hochberg": benjamini_hochberg}
