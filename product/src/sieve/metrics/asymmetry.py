"""Sign/time asymmetry metrics."""

from __future__ import annotations

import numpy as np


def leverage(r: np.ndarray, lags: int = 5) -> float:
    """Correlation of r_t with future |r|; negative in equity indices.

    One of only three battery members that see the arrow of time (the research
    invariance audit found 14 of 17 statistics exactly reversal-invariant).
    Disclosed sensitivity: the lag-count knob moves the level by 0.94 IQR.
    """
    a = np.abs(r)
    out = []
    for k in range(1, lags + 1):
        x, y = r[:-k], a[k:]
        sx, sy = x.std(), y.std()
        if sx > 0 and sy > 0:
            out.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)))
    return float(np.mean(out)) if out else float("nan")
