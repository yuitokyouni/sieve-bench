"""Known-case behavior of every suite metric (plan §5 / spec §16)."""

import numpy as np
import pytest

from sieve.metrics import registry
from sieve.metrics.asymmetry import leverage
from sieve.metrics.dependence import drift, variance_ratio_20
from sieve.metrics.distribution import excess_kurtosis
from sieve.metrics.tails import hill_left, hill_right
from sieve.metrics.volatility import acf_abs_1

RNG = np.random.default_rng(123)
GAUSS = RNG.standard_normal(200_000)
T5 = RNG.standard_t(5, 200_000)


def test_excess_kurtosis_gaussian_zero():
    assert abs(excess_kurtosis(GAUSS)) < 0.05


def test_excess_kurtosis_orders_tail_weight():
    assert excess_kurtosis(T5) > 1.0 > excess_kurtosis(GAUSS)


def test_hill_symmetric_tails_agree():
    # symmetric distribution: left and right tail indices agree loosely
    assert abs(hill_left(T5) - hill_right(T5)) < 0.5


def test_hill_orders_tail_heaviness():
    t3 = RNG.standard_t(3, 200_000)
    # heavier tail → smaller tail index
    assert hill_left(t3) < hill_left(GAUSS)


def test_acf_abs_iid_is_zero():
    assert abs(acf_abs_1(GAUSS[:50_000])) < 0.02


def test_acf_abs_sees_clustering():
    # alternate calm/wild regimes → |r| autocorrelation appears
    sd = np.repeat([0.5, 2.0], 50)
    sd = np.tile(sd, 500)
    clustered = RNG.standard_normal(len(sd)) * sd
    assert acf_abs_1(clustered) > 0.1


def test_leverage_iid_zero_and_sign():
    assert abs(leverage(GAUSS[:100_000])) < 0.02
    # negative return → higher next-day volatility, by construction
    n = 100_000
    e = RNG.standard_normal(n)
    sd = np.ones(n)
    for i in range(1, n):
        sd[i] = 1.0 + (1.0 if e[i - 1] < 0 else 0.0)
    r = e * sd
    assert leverage(r) < -0.05


def test_variance_ratio_iid_is_one():
    assert abs(variance_ratio_20(GAUSS[:100_000]) - 1.0) < 0.05


def test_variance_ratio_sees_mean_reversion():
    # AR(1) with negative coefficient → VR < 1
    n = 100_000
    r = np.empty(n)
    r[0] = 0.0
    e = RNG.standard_normal(n)
    for i in range(1, n):
        r[i] = -0.3 * r[i - 1] + e[i]
    assert variance_ratio_20(r) < 0.8


def test_drift_shifts_with_location():
    x = GAUSS[:10_000]
    assert drift(x + 0.5) > drift(x) + 0.4


def test_registry_resolves_all_suite_metrics():
    for mid in ("excess_kurtosis", "hill_left", "hill_right", "acf_abs_1",
                "acf_abs_20", "leverage", "variance_ratio_20", "drift"):
        fn, spec, dim = registry.resolve(f"{mid}@1")
        assert spec.metric_id == mid
        assert dim in ("marginal_distribution", "tail_behavior",
                       "volatility_dynamics", "leverage_asymmetry",
                       "return_dependence", "drift_nonstationarity")
        assert spec.known_blind_spots, f"{mid} must disclose blind spots"


def test_registry_rejects_unknown_and_wrong_major():
    with pytest.raises(KeyError):
        registry.resolve("no_such_metric@1")
    with pytest.raises(KeyError):
        registry.resolve("leverage@9")


def test_post_hoc_metrics_are_marked():
    from sieve.core.enums import Prespecification
    for mid in ("variance_ratio_20", "drift"):
        _, spec, _ = registry.resolve(mid)
        assert spec.prespecification is Prespecification.POST_HOC
    _, spec, _ = registry.resolve("leverage")
    assert spec.prespecification is Prespecification.PRE_SPECIFIED


def test_compute_maps_failure_to_nan():
    assert np.isnan(registry.compute("variance_ratio_20", np.ones(30)))
    assert np.isnan(registry.compute("drift", np.zeros(100)))
