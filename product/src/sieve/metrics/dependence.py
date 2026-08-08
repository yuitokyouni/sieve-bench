"""Return-dependence and drift metrics.

Both were added to the research battery POST HOC, after calibrated ABMs
exposed failures the original battery could not see (mean reversion 3-5x too
strong; drift 4-10x the empirical level). That history is carried in their
MetricSpec.prespecification and shown in reports — disclosure, not confession
(spec §5.5).
"""

from __future__ import annotations

import numpy as np


def variance_ratio_20(r: np.ndarray, q: int = 20) -> float:
    """Lo & MacKinlay (1988) variance ratio at q=20; 1 under independence."""
    n = (len(r) // q) * q
    if n < q * 10:
        return float("nan")
    a = r[:n]
    v1 = a.var()
    if v1 <= 0:
        return float("nan")
    return float((a.reshape(-1, q).sum(axis=1).var() / q) / v1)


def drift(r: np.ndarray) -> float:
    """Mean return per unit standard deviation (window Sharpe)."""
    s = float(np.std(r))
    if s <= 0 or not np.isfinite(s):
        return float("nan")
    return float(np.mean(r) / s)
