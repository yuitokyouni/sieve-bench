"""Shared helpers. Math is migrated verbatim from research `facts.py`
(sieve-bench @ 6ad237c); golden tests pin every value bit-for-bit."""

from __future__ import annotations

import numpy as np


def acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    d = (x * x).sum()
    if d <= 0:
        return float("nan")
    return float((x[lag:] * x[:-lag]).sum() / d)
