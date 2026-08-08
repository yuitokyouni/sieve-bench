import numpy as np

from sieve.inference.multiplicity import ADJUSTMENTS, benjamini_hochberg, holm


def test_holm_known_vector():
    assert np.allclose(holm([0.01, 0.02, 0.03, 0.04]),
                       [0.04, 0.06, 0.06, 0.06])


def test_bh_known_vector():
    assert np.allclose(benjamini_hochberg([0.01, 0.02, 0.03, 0.04]),
                       [0.04, 0.04, 0.04, 0.04])


def test_holm_dominates_bh():
    p = [0.001, 0.011, 0.02, 0.6, 0.04]
    h, b = holm(p), benjamini_hochberg(p)
    assert all(hv >= bv - 1e-12 for hv, bv in zip(h, b))


def test_nan_excluded_from_family():
    p = [float("nan"), 0.01, float("nan")]
    h = holm(p)
    assert np.isnan(h[0]) and np.isnan(h[2])
    assert h[1] == 0.01           # family of one: no penalty
    b = benjamini_hochberg(p)
    assert b[1] == 0.01


def test_capped_at_one():
    assert max(holm([0.9, 0.8, 0.7])) <= 1.0


def test_registry_names():
    assert set(ADJUSTMENTS) == {"holm", "benjamini_hochberg"}
