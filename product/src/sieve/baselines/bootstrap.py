"""Bootstrap baselines.

These need the raw reference pool, which suites do not ship (it is not
redistributable). They run only when the caller supplies a local pool; the
offline golden path uses their frozen distributions instead.
"""

from __future__ import annotations

import numpy as np


class ReferencePoolRequired(RuntimeError):
    """Raised when a bootstrap baseline is invoked without a local pool."""


def _pool(params: dict) -> np.ndarray:
    pool = params.get("pool")
    if pool is None:
        raise ReferencePoolRequired(
            "bootstrap baselines resample the raw reference series, which is "
            "not shipped; fetch it locally (research fetch.py) or use the "
            "frozen distributions in the suite")
    return np.asarray(pool, float)


def iid_generate(n: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    return rng.choice(_pool(params), size=n, replace=True)


def block_generate(n: int, rng: np.random.Generator, params: dict,
                   block: int = 20) -> np.ndarray:
    pool = _pool(params)
    nb = n // block + 1
    starts = rng.integers(0, len(pool) - block, size=nb)
    return np.concatenate([pool[s:s + block] for s in starts])[:n]
