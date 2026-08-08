"""Two-sample Kolmogorov-Smirnov statistic (verbatim from research
`resampling.py` @ 6ad237c)."""

from __future__ import annotations

import numpy as np


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))
