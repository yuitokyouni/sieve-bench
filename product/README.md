# Sieve

Evidence infrastructure for deciding what a simulation result **does and does
not** support.

You hand Sieve a simulated daily return series and a claim. It runs a
versioned test suite against frozen empirical reference distributions and
writes a run directory containing a human-readable report and a sealed,
machine-verifiable **evidence bundle**. Everything happens locally and
offline: nothing is uploaded, no network is touched.

```
pip install -e .          # or: uv pip install -e .
sieve test examples/csv_returns --suite financial-daily@1.0 --claim descriptive-market-dynamics
sieve verify .sieve/runs/<run-id>
```

The run directory contains:

```
manifest.json                     run identity, seed tree, environment
observations.parquet              per-window metric values (the raw evidence)
results.json                      per-metric tests: KS, p, adjusted p, status
findings.json                     FAILs turned into author-actionable findings
artifacts/baseline_context.json   which baselines each metric cannot separate
report/index.html                 the report, readable without Sieve
evidence_bundle.json              everything above, in one sealed schema
bundle.sha256                     sha256sum-compatible integrity sidecar
```

## What a result means

- **A claim, not a model, is evaluated.** `descriptive-market-dynamics` asks
  whether the input reproduces the descriptive dynamics of major equity-index
  daily returns — nothing else. Statuses answer "does the evidence support
  using this simulation *for this claim*".
- **Ten evidence dimensions, never averaged.** Each dimension ends at PASS,
  FAIL, WARN, NOT_TESTED or INSUFFICIENT. `NOT_TESTED` and `INSUFFICIENT`
  are first-class answers. There is no overall score anywhere in the system,
  by design and by test (`tests/unit/test_no_score.py`).
- **PASS is calibrated, and disclosed as weak where it is weak.** The decision
  line (α = 0.01, Holm-adjusted) is the nominal level whose *measured* true
  size is 3–5% under the reference dependence structure — a naive permutation
  test rejects 56% at nominal 5% on this design, so Sieve does not use one.
  Every PASS row also lists which shipped baselines that metric *cannot*
  separate from the reference: agreeing with reality on a metric that GARCH
  also matches is weak evidence, and the report says so on the row itself.
- **Post-hoc diagnostics are labeled.** Two suite metrics (`variance_ratio_20`,
  `drift`) were added to the research battery after calibrated agent-based
  models exposed failures the original battery could not see. They carry a
  POST HOC badge in every report — disclosure, not confession.
- **Bundles are sealed twice.** `bundle_hash` is a deterministic scientific
  seal: same input + same suite + same seed ⇒ the same hash, byte for byte
  (timestamps and run IDs are excluded). `bundle.sha256` pins the shipped
  files, including every artifact hash; `sha256sum -c bundle.sha256` works
  with no Sieve installed. `sieve verify` checks both layers and reports
  tampering as a result, not a crash.

## Where the science comes from

The metrics, baselines, reference statistics, calibrated inference and every
disclosed blind spot are migrated verbatim from the
[sieve-bench](../README.md) research repository, and golden regression tests
pin the product to it bit-for-bit (`tests/golden/`). The suite ships *derived*
window statistics (124 reference windows × 8 metrics, with calendar blocks and
per-index source hashes) — raw index data is neither shipped nor fetched.

Baselines declare their mechanisms explicitly (`sieve baselines list`):
`gaussian` (nothing), `student_t` (heavy tails only), `iid_bootstrap`
(marginal only), `block_bootstrap` (marginal + short memory), `garch_norm`
(clustering only), `garch_t` (clustering + heavy tails, no asymmetry). The
worked example (`examples/csv_returns`) is a garch_t path: it FAILs
`leverage_asymmetry` — the mechanism it genuinely lacks — and its PASSes on
volatility metrics carry the baseline-context caveat.

## What Sieve will not do

No reality score. No model rankings or leaderboards. No "certified" badges.
No generic LLM evaluation. No uploading of proprietary code or data. No
arbitrary remote code execution. These are product invariants (spec §0), not
roadmap gaps; the no-score invariant is enforced by tests.

## CLI

```
sieve doctor                 environment + suite check (offline, no network)
sieve test INPUT --suite S --claim C [--out DIR] [--seed N]
sieve verify RUN_DIR         exit 0 intact / 4 modified / 3 missing
sieve report RUN_DIR         re-render HTML from the stored bundle
sieve suites list|show       installed suites, their hashes and claims
sieve metrics list|show      metrics with dimensions and known blind spots
sieve baselines list         baselines with declared mechanisms
sieve schemas export         JSON Schema for every durable artifact
```

Exit codes separate evaluation from execution: a model FAILing a dimension is
a *result* (exit 0); only invalid input (2), missing pieces (3), tampering
(4) or a Sieve bug (1) are errors.

## Input contract (Tier 0)

`returns.csv` with columns `timestamp,return` (daily log returns), 50+ rows,
optionally a `manifest.yaml` with model identity, parameters, `git_commit`,
`code_uri`. Anything you do not state is recorded as absent — provenance gaps
appear in the report rather than being silently filled. Five non-overlapping
windows of 1000 observations (≈ 20 years) are needed for the statistical
tests; shorter inputs produce INSUFFICIENT, which is an answer, not an error.

## Status

M1: the offline golden path above is complete and tested (75 tests, including
determinism, tamper detection and bit-for-bit parity with the research repo).
See `STATUS.md` for the milestone ledger and `IMPLEMENTATION_PLAN.md` for the
full mapping from spec to code.
