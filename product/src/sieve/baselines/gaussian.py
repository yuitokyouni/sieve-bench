from __future__ import annotations

import numpy as np


def generate(n: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    return rng.standard_normal(n)
