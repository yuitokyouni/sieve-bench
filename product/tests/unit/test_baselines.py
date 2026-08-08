"""Baseline mechanism smoke tests: each generator shows the mechanisms it
declares present and lacks the ones it declares absent (plan §5)."""

import numpy as np
import pytest

from sieve.baselines import registry
from sieve.baselines.bootstrap import (
    ReferencePoolRequired,
    block_generate,
    iid_generate,
)
from sieve.metrics.asymmetry import leverage
from sieve.metrics.distribution import excess_kurtosis
from sieve.metrics.volatility import acf_abs_1

PARAMS = {"garch": [0.017, 0.116, 0.863],
          "garch_t": [0.011, 0.118, 0.875, 6.47],
          "t_df": 2.65}
N = 50_000


def _gen(bid, seed=42, params=PARAMS, n=N):
    fn, _ = registry.resolve(bid)
    return fn(n, np.random.default_rng(seed), params)


def test_gaussian_floor_has_nothing():
    r = _gen("gaussian")
    assert abs(excess_kurtosis(r)) < 0.1
    assert abs(acf_abs_1(r)) < 0.02
    assert abs(leverage(r)) < 0.02


def test_student_t_adds_tails_only():
    r = _gen("student_t")
    assert excess_kurtosis(r) > 1.0
    assert abs(acf_abs_1(r)) < 0.02
    # unit variance normalization
    assert abs(r.std() - 1.0) < 0.15


def test_garch_adds_clustering():
    rn = _gen("garch_norm")
    rt = _gen("garch_t")
    assert acf_abs_1(rn) > 0.1
    assert acf_abs_1(rt) > 0.1
    # conditional t fattens tails beyond conditional normal
    assert excess_kurtosis(rt) > excess_kurtosis(rn)


def test_garch_lacks_asymmetry():
    r = _gen("garch_t", n=200_000)
    assert abs(leverage(r)) < 0.02


def test_bootstrap_requires_local_pool():
    rng = np.random.default_rng(0)
    with pytest.raises(ReferencePoolRequired):
        iid_generate(100, rng, {})
    with pytest.raises(ReferencePoolRequired):
        block_generate(100, rng, {})


def test_bootstrap_resamples_the_pool():
    pool = np.arange(1000, dtype=float)
    rng = np.random.default_rng(0)
    r = iid_generate(500, rng, {"pool": pool})
    assert set(r).issubset(set(pool))
    b = block_generate(500, rng, {"pool": pool}, block=20)
    # contiguous 20-blocks of an arange have constant unit increments inside
    inc = np.diff(b)
    assert (inc == 1).sum() >= len(b) * 0.9


def test_every_baseline_declares_mechanisms():
    specs = registry.all_specs()
    assert len(specs) == 6
    for s in specs:
        assert s.mechanisms_present or s.mechanisms_absent
        assert not (set(s.mechanisms_present) & set(s.mechanisms_absent))
        assert s.calibration_ref
