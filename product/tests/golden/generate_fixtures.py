"""Freeze golden fixtures from the research repository at its current commit.

Run from the research repo root (sieve-bench). Regenerating requires the
research code and the locally fetched index data (python3 fetch.py there).

Produces, under product/tests/golden/fixtures/:
  research_commit.txt        - commit the fixtures were frozen from
  reference_stats.json       - per-window statistic values for the real data
                               (124 windows x M1 metrics) + calendar blocks
  baseline_stats.json        - per-run statistic values for the 6 baselines
                               (per-index fits, allocation as in separation.py)
  baseline_params.json       - fitted parameters per index (provenance)
  research_outputs/          - verbatim copies of the research result JSONs

reference_stats.json / baseline_stats.json are ALSO shipped inside the
financial-daily@1.0.0 suite: they are derived window-level statistics, not the
raw Yahoo series (which is not redistributable), which is what makes the M1
golden path fully offline.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from collections import Counter  # noqa: E402

import facts  # noqa: E402
import numpy as np  # noqa: E402
from generators import GENERATORS, build_contexts  # noqa: E402
from windows import (  # noqa: E402
    BLOCK_WIDTHS,
    STRIDE,
    WINDOW,
    calendar_blocks,
    load_series,
    real_windows,
)

# The M1 suite metric set (financial-daily@1.0.0). Names match facts.BATTERY.
M1_METRICS = ["excess_kurtosis", "hill_left", "hill_right", "acf_abs_1",
              "acf_abs_20", "leverage", "variance_ratio_20", "drift"]
M1_BASELINES = ["gaussian", "student_t", "iid_bootstrap", "block_bootstrap",
                "garch_norm", "garch_t"]
N_RUNS = 200
SEED = 20260802

FIX = os.path.join(HERE, "fixtures")
os.makedirs(os.path.join(FIX, "research_outputs"), exist_ok=True)

commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
open(os.path.join(FIX, "research_commit.txt"), "w").write(commit + "\n")

rng = np.random.default_rng(SEED)
series = load_series()
wins = real_windows(series)
blocks = calendar_blocks(wins, BLOCK_WIDTHS["span"])

ref = {"commit": commit, "window": WINDOW, "stride": STRIDE,
       "block_width_days": BLOCK_WIDTHS["span"], "metrics": M1_METRICS,
       "windows": [{"index": w.index, "start": str(w.start), "end": str(w.end),
                    "block": int(b)} for w, b in zip(wins, blocks)],
       "values": {m: [float(facts.BATTERY[m](w.values)) for w in wins]
                  for m in M1_METRICS}}


def _source_hash(path):
    """SHA-256 of the non-null (timestamp, close) pairs, canonical JSON.

    Lets anyone who fetches the same index window verify they hold the same
    source series the shipped statistics were derived from, without the raw
    series ever being redistributed.
    """
    d = json.load(open(path))["chart"]["result"][0]
    good = [[int(t), float(c)] for t, c in
            zip(d["timestamp"], d["indicators"]["quote"][0]["close"])
            if c is not None]
    body = json.dumps(good, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


ref["sources"] = {
    fn[:-5]: {"sha256_timestamp_close_pairs": _source_hash(
                  os.path.join(ROOT, "data", fn)),
              "transform": "log-diff of non-null closes, unit-sd scaled"}
    for fn in sorted(os.listdir(os.path.join(ROOT, "data")))
    if fn.endswith(".json")}
json.dump(ref, open(os.path.join(FIX, "reference_stats.json"), "w"), indent=1)

ctxs = build_contexts(series)
counts = Counter(w.index for w in wins)
alloc = {n: max(1, round(N_RUNS * counts[n] / len(wins))) for n in sorted(counts)}
base = {"commit": commit, "seed": SEED, "n_runs_nominal": N_RUNS,
        "alloc": alloc, "fit_per_index": True, "metrics": M1_METRICS,
        "values": {}}
for g in M1_BASELINES:
    rows = {m: [] for m in M1_METRICS}
    for name, k in alloc.items():
        for _ in range(k):
            x = GENERATORS[g](WINDOW, rng, ctxs[name])
            for m in M1_METRICS:
                rows[m].append(float(facts.BATTERY[m](x)))
    base["values"][g] = rows
    print(f"  {g}: {sum(alloc.values())} runs")
json.dump(base, open(os.path.join(FIX, "baseline_stats.json"), "w"), indent=1)

params = {n: {k: v for k, v in ctxs[n].items() if k != "pool"}
          for n in sorted(ctxs)}
json.dump({"commit": commit, "per_index": params},
          open(os.path.join(FIX, "baseline_params.json"), "w"), indent=1)

for f in ("separation.json", "knockout.json", "invariance.json",
          "sensitivity.json", "robustness.json", "probes_calibration.json",
          "power.json", "baselines.json"):
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(FIX, "research_outputs", f))
print("frozen at", commit)
