"""Regenerate the example input deterministically.

The example model is the product's own garch_t baseline run with the shipped
S&P 500 parameters (suite financial-daily@1.0.0), seed 7, 6000 days. That
makes the expected golden-path outcome legible: volatility clustering and
heavy tails are present, sign asymmetry (leverage) is absent by construction.

Run from anywhere:  python3 product/examples/csv_returns/generate.py
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parents[1]
sys.path.insert(0, str(PRODUCT / "src"))

from sieve.baselines.garch import generate_t  # noqa: E402

N, SEED = 6000, 7

params = json.loads(
    (PRODUCT / "suites/financial-daily/1.0.0/baseline_params.json").read_text())
gspc = params["per_index"]["gspc"]

rng = np.random.default_rng(SEED)
r = generate_t(N, rng, gspc)

day = dt.date(2000, 1, 3)
lines = ["timestamp,return"]
for x in r:
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    lines.append(f"{day.isoformat()},{float(x)!r}")
    day += dt.timedelta(days=1)
(HERE / "returns.csv").write_text("\n".join(lines) + "\n")
print(f"wrote {HERE / 'returns.csv'}: {N} returns, seed {SEED}")
