"""Golden regression: the product preserves the sieve-bench science.

Three layers, strongest available first:

1. The suite ships byte-identical copies of the frozen fixtures.
2. Product metric functions equal the research functions bit-for-bit on
   arbitrary synthetic input (runs whenever the research repo is present —
   it is the parent directory in the current layout).
3. Recomputing the 124 reference windows with product metrics reproduces the
   frozen values exactly (runs only when the raw index data has been fetched
   locally; the data itself is never shipped).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PRODUCT = Path(__file__).resolve().parents[2]
RESEARCH = PRODUCT.parent
FIX = Path(__file__).resolve().parent / "fixtures"
SUITE = PRODUCT / "suites" / "financial-daily" / "1.0.0"

M1_METRICS = ["excess_kurtosis", "hill_left", "hill_right", "acf_abs_1",
              "acf_abs_20", "leverage", "variance_ratio_20", "drift"]


def product_fn(mid):
    from sieve.metrics import registry
    fn, _, _ = registry.resolve(mid)
    return fn


# ---- layer 1: the suite ships exactly the frozen fixtures ------------------

@pytest.mark.parametrize("name", ["reference_stats.json",
                                  "baseline_stats.json",
                                  "baseline_params.json"])
def test_suite_ships_frozen_fixture_bytes(name):
    assert (SUITE / name).read_bytes() == (FIX / name).read_bytes()


def test_fixture_shape():
    ref = json.loads((FIX / "reference_stats.json").read_text())
    assert len(ref["windows"]) == 124
    assert set(ref["values"]) == set(M1_METRICS)
    assert all(len(v) == 124 for v in ref["values"].values())
    assert len(ref["sources"]) == 6
    base = json.loads((FIX / "baseline_stats.json").read_text())
    assert set(base["values"]) == {"gaussian", "student_t", "iid_bootstrap",
                                   "block_bootstrap", "garch_norm", "garch_t"}


# ---- layer 2: product == research on synthetic input, bit for bit ----------

@pytest.mark.skipif(not (RESEARCH / "facts.py").exists(),
                    reason="research repo absent")
@pytest.mark.parametrize("mid", M1_METRICS)
def test_product_metric_equals_research_bit_for_bit(mid):
    sys.path.insert(0, str(RESEARCH))
    import facts
    rng = np.random.default_rng(2026)
    samples = [rng.standard_normal(1000),
               rng.standard_t(4, 1500),
               np.cumsum(rng.standard_normal(1200)) * 0.01
               + rng.standard_normal(1200)]
    from sieve.baselines.garch import generate_t
    samples.append(generate_t(2000, np.random.default_rng(5),
                              {"garch_t": [0.011, 0.118, 0.875, 6.47]}))
    for x in samples:
        a = product_fn(mid)(x)
        b = float(facts.BATTERY[mid](x))
        assert (np.isnan(a) and np.isnan(b)) or a == b, (mid, a, b)


# ---- layer 3: reference windows recompute exactly (needs local data) -------

@pytest.mark.skipif(not (RESEARCH / "data").is_dir()
                    or not any((RESEARCH / "data").glob("*.json")),
                    reason="raw index data not fetched locally")
def test_reference_values_recompute_exactly():
    sys.path.insert(0, str(RESEARCH))
    from windows import load_series, real_windows
    ref = json.loads((FIX / "reference_stats.json").read_text())
    wins = real_windows(load_series())
    assert len(wins) == len(ref["windows"])
    for mid in M1_METRICS:
        fn = product_fn(mid)
        got = [float(fn(w.values)) for w in wins]
        assert got == ref["values"][mid], mid
