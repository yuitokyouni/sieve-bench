"""Volatility-dynamics metrics."""

from __future__ import annotations

import numpy as np

from sieve.metrics._shared import acf


def acf_abs_1(r: np.ndarray) -> float:
    """Lag-1 autocorrelation of |r|: the direct volatility-clustering signal.

    Blind spot on record: a plain GARCH(1,1) passes this (research KS 0.12,
    p = 0.36 against real data). Reproducing it says nothing about mechanism.
    """
    return acf(np.abs(r), 1)


def acf_abs_20(r: np.ndarray) -> float:
    """Lag-20 autocorrelation of |r|: persistence of clustering."""
    return acf(np.abs(r), 20)
