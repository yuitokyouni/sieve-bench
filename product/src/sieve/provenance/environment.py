"""Capture the execution environment and the seed tree.

Seed convention follows the research lab's ``make_rngs``: one master
``SeedSequence``, named children spawned once, every child recorded in the
run manifest so any stage can be replayed in isolation.
"""

from __future__ import annotations

import platform
import sys

import numpy as np

from sieve.core.models import SeedNode

SEED_CHILDREN = ("windows", "resampling", "baselines", "report")


def make_rngs(
        master_seed: int
) -> tuple[dict[str, np.random.Generator], list[SeedNode]]:
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(len(SEED_CHILDREN))
    rngs = {name: np.random.default_rng(child)
            for name, child in zip(SEED_CHILDREN, children)}
    nodes = [SeedNode(name=name, entropy=int(master_seed),
                      spawn_key=list(child.spawn_key))
             for name, child in zip(SEED_CHILDREN, children)]
    return rngs, nodes


def environment_fingerprint() -> dict[str, str]:
    import numpy
    import pydantic

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pydantic": pydantic.__version__,
    }
    try:
        import scipy
        out["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import polars
        out["polars"] = polars.__version__
    except ImportError:
        pass
    return out
