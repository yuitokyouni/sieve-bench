"""Baseline registry: generators with declared mechanisms (spec §2.3, §5.2).

Two forms of a baseline exist and both are first-class:

1. **Frozen distributions** shipped inside a suite (`baseline_stats.json`):
   per-run statistic values computed at a pinned research commit with
   per-index MLE parameters. This is what the offline golden path compares
   against — derived numbers, so no raw reference data is redistributed.
2. **Live generators** (this module): the same mathematics, runnable from the
   shipped parameters when a workflow wants fresh draws. The bootstrap pair
   cannot run without the local reference series and says so explicitly.

Every entry declares mechanisms_present / mechanisms_absent, because the whole
point of a baseline ladder is knowing what is missing (spec: "a baseline must
expose both its parameters and the mechanism it is intended to include/remove").
"""

from __future__ import annotations

from collections.abc import Callable

from sieve.baselines import bootstrap, garch, gaussian, student_t
from sieve.core.models import BaselineSpec

_ENTRIES: dict[str, tuple[Callable | None, BaselineSpec]] = {}


def _register(baseline_id: str, fn: Callable | None, desc: str,
              present: list[str], absent: list[str]) -> None:
    _ENTRIES[baseline_id] = (fn, BaselineSpec(
        baseline_id=baseline_id, version="1", description=desc,
        mechanisms_present=present, mechanisms_absent=absent,
        parameters={}, calibration_ref="per-index MLE, sieve-bench@6ad237c"))


_register("gaussian", gaussian.generate,
          "iid standard normal draws — the floor",
          [], ["heavy tails", "volatility clustering", "asymmetry",
               "time irreversibility"])
_register("student_t", student_t.generate,
          "iid standardized Student-t draws",
          ["heavy tails"], ["volatility clustering", "asymmetry",
                            "time irreversibility"])
_register("iid_bootstrap", bootstrap.iid_generate,
          "resample reference returns one by one (marginal preserved)",
          ["empirical marginal"], ["all temporal structure"])
_register("block_bootstrap", bootstrap.block_generate,
          "resample reference returns in 20-day blocks",
          ["empirical marginal", "sub-20-day temporal structure "
           "(including the real arrow of time within blocks)"],
          ["memory beyond 20 days"])
_register("garch_norm", garch.generate_norm,
          "GARCH(1,1), normal innovations",
          ["volatility clustering"], ["heavy conditional tails", "asymmetry"])
_register("garch_t", garch.generate_t,
          "GARCH(1,1), Student-t innovations, jointly fitted",
          ["volatility clustering", "heavy tails"], ["asymmetry"])


def resolve(ref: str) -> tuple[Callable | None, BaselineSpec]:
    baseline_id, _, major = ref.partition("@")
    if baseline_id not in _ENTRIES:
        raise KeyError(f"unknown baseline: {baseline_id}")
    fn, spec = _ENTRIES[baseline_id]
    if major and spec.version.split(".")[0] != major:
        raise KeyError(f"baseline {baseline_id} major {major} not available")
    return fn, spec


def all_specs() -> list[BaselineSpec]:
    return [spec for _, spec in _ENTRIES.values()]
