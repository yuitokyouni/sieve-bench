"""Metric registry: id → (function, MetricSpec, evidence dimension).

Nothing in the runner hard-codes metric lists (spec §5.3); suites reference
``metric_id@major`` strings resolved here. Blind-spot metadata comes from the
research invariance audit — it is part of the method, not marketing copy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sieve.core.enums import Prespecification
from sieve.core.models import MetricSpec
from sieve.metrics import asymmetry, dependence, distribution, tails, volatility

_REVERSAL_BLIND = "time-reversal invariant: identical on a series and its reverse"
_LOCATION_BLIND = "location invariant: cannot see drift (by design; see drift metric)"


def _spec(metric_id: str, fn: Callable, display: str, signal: str,
          blind: list[str], pre: Prespecification,
          refs: list[str]) -> tuple[Callable, MetricSpec]:
    return fn, MetricSpec(
        metric_id=metric_id, version="1",
        display_name=display,
        function_path=f"{fn.__module__}.{fn.__name__}",
        input_contract="1d float array of log returns, length >= 300",
        scale_invariant=True, intended_signal=signal,
        known_blind_spots=blind, prespecification=pre, references=refs)


_ENTRIES: dict[str, tuple[Callable, MetricSpec, str]] = {}


def _register(dimension: str, fn: Callable, metric_id: str, display: str,
              signal: str, blind: list[str], pre: Prespecification,
              refs: list[str]) -> None:
    f, spec = _spec(metric_id, fn, display, signal, blind, pre, refs)
    _ENTRIES[metric_id] = (f, spec, dimension)


_register("marginal_distribution", distribution.excess_kurtosis,
          "excess_kurtosis", "Excess kurtosis",
          "heavy tails of the marginal return distribution",
          [_REVERSAL_BLIND, _LOCATION_BLIND,
           "cannot separate a jointly-fitted Student-t GARCH from real data "
           "(research KS 0.18, p=0.22)"],
          Prespecification.PRE_SPECIFIED, ["Cont (2001)"])

_register("tail_behavior", tails.hill_left, "hill_left",
          "Hill tail index (left)",
          "power-law weight of the loss tail",
          [_REVERSAL_BLIND, "depends on tail fraction k (swing 0.92 IQR "
           "across k in 2.5-10%)"],
          Prespecification.PRE_SPECIFIED, ["Hill (1975)", "Cont (2001)"])

_register("tail_behavior", tails.hill_right, "hill_right",
          "Hill tail index (right)",
          "power-law weight of the gain tail",
          [_REVERSAL_BLIND, "depends on tail fraction k"],
          Prespecification.PRE_SPECIFIED, ["Hill (1975)", "Cont (2001)"])

_register("volatility_dynamics", volatility.acf_abs_1, "acf_abs_1",
          "|r| autocorrelation, lag 1",
          "volatility clustering",
          [_REVERSAL_BLIND, _LOCATION_BLIND,
           "reproduced by GARCH(1,1): passing this says nothing about mechanism"],
          Prespecification.PRE_SPECIFIED, ["Cont (2001)"])

_register("volatility_dynamics", volatility.acf_abs_20, "acf_abs_20",
          "|r| autocorrelation, lag 20",
          "persistence of volatility clustering",
          [_REVERSAL_BLIND, "blind to all GARCH-family baselines in the "
           "research matrix (KS <= 0.18)"],
          Prespecification.PRE_SPECIFIED, ["Cont (2001)"])

_register("leverage_asymmetry", asymmetry.leverage, "leverage",
          "Leverage correlation",
          "sign-to-future-volatility coupling (arrow of time)",
          ["lag-count knob moves the level by 0.94 IQR",
           "retained by block bootstrap of real data (real blocks keep the "
           "real arrow)"],
          Prespecification.PRE_SPECIFIED,
          ["Bouchaud, Matacz & Potters (2001)"])

_register("return_dependence", dependence.variance_ratio_20,
          "variance_ratio_20", "Variance ratio (q=20)",
          "mean reversion / trend persistence of returns",
          [_REVERSAL_BLIND, _LOCATION_BLIND],
          Prespecification.POST_HOC,
          ["Lo & MacKinlay (1988)",
           "added after calibrated ABMs showed 3-5x over-strong mean reversion"])

_register("drift_nonstationarity", dependence.drift, "drift",
          "Drift (mean/sd)",
          "location of the return distribution (window Sharpe)",
          [_REVERSAL_BLIND,
           "weak against all research generators; strong against calibrated ABMs"],
          Prespecification.POST_HOC,
          ["added after calibrated ABMs showed 4-10x empirical drift"])


def resolve(ref: str) -> tuple[Callable, MetricSpec, str]:
    """Resolve ``metric_id@major`` (or bare id) to (fn, spec, dimension)."""
    metric_id, _, major = ref.partition("@")
    if metric_id not in _ENTRIES:
        raise KeyError(f"unknown metric: {metric_id}")
    fn, spec, dim = _ENTRIES[metric_id]
    if major and spec.version.split(".")[0] != major:
        raise KeyError(f"metric {metric_id} major version {major} not available "
                       f"(have {spec.version})")
    return fn, spec, dim


def all_specs() -> list[MetricSpec]:
    return [spec for _, spec, _ in _ENTRIES.values()]


def compute(ref: str, r: np.ndarray) -> float:
    fn, _, _ = resolve(ref)
    try:
        return float(fn(np.asarray(r, dtype=float)))
    except Exception:
        return float("nan")
