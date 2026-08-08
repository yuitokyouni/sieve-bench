"""Marginal-distribution metrics."""

from __future__ import annotations

import numpy as np


def excess_kurtosis(r: np.ndarray) -> float:
    """Tail weight of the marginal; 0 for a Gaussian."""
    z = (r - r.mean()) / r.std()
    return float((z ** 4).mean() - 3.0)
