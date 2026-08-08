"""Design-preserving null for dependent reference windows.

Migrated verbatim from research `resampling.block_boot_test` @ 6ad237c.
Why this and not a plain permutation: the reference windows overlap and share
calendar shocks. Under a true null the naive permutation rejected 56% at
nominal 5% (research selftest). Block resampling preserves the design; the
residual over-rejection from having only ~6 independent calendar blocks is
handled by the calibrated decision line the suite declares (alpha = 0.01
delivering a true size of 3-5%).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def block_boot_test(a_vals, a_blocks, b_vals, stat: Callable,
                    rng: np.random.Generator, n_boot: int = 2000):
    """Return (statistic, p_value, null 95th pct, n_blocks)."""
    a = np.asarray(a_vals, float)
    b = np.asarray(b_vals, float)
    ma, mb = np.isfinite(a), np.isfinite(b)
    a, ab = a[ma], np.asarray(a_blocks)[ma]
    b = b[mb]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan"), 0

    groups = [np.where(ab == u)[0] for u in sorted(set(ab.tolist()))]
    k, nb = len(groups), len(b)
    obs = stat(a, b)
    null = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        astar = np.concatenate([a[groups[j]] for j in pick])
        bstar = rng.choice(a, nb, replace=True)
        null[i] = stat(astar, bstar)
    good = np.isfinite(null)
    pval = float((1.0 + np.sum(null[good] >= obs)) / (good.sum() + 1.0))
    return float(obs), pval, float(np.percentile(null[good], 95)), k
