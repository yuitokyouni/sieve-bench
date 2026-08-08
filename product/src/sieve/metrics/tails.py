"""Tail-index metrics (Hill).

Known trap, disclosed in metadata: the estimate depends on the tail fraction
``frac``. The research sensitivity sweep measured a swing of 0.92 IQR for the
left tail across frac in (0.025, 0.05, 0.10).
"""

from __future__ import annotations

import numpy as np


def _hill(x: np.ndarray, frac: float) -> float:
    x = np.sort(x)[::-1]
    k = max(int(len(x) * frac), 10)
    if len(x) <= k or x[k] <= 0:
        return float("nan")
    return float(1.0 / np.mean(np.log(x[:k] / x[k])))


def hill_right(r: np.ndarray, frac: float = 0.05) -> float:
    return _hill(r[r > 0], frac)


def hill_left(r: np.ndarray, frac: float = 0.05) -> float:
    return _hill(-r[r < 0], frac)
