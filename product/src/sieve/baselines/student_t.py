from __future__ import annotations

import numpy as np


def generate(n: int, rng: np.random.Generator, params: dict) -> np.ndarray:
    df = float(params["t_df"])
    x = rng.standard_t(df, n)
    return x / np.sqrt(df / (df - 2.0))
