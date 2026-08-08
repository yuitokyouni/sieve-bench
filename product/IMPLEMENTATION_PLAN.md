# Sieve — Implementation Plan (M0 + M1)

Per spec §22. Maps every M0/M1 requirement to concrete files and tests.
Spec: `SIEVE_PRODUCT_IMPLEMENTATION_SPEC` v0.1 (2026-08-09).

## 1. Current relevant files

### `sieve-bench` (research, this repo, frozen at `6ad237c`)

| File | Relevance |
|---|---|
| `facts.py` | 17 statistics + `SPEC` design contracts (must_invariant, params) |
| `generators.py` | 6+2 baselines, per-index MLE fits (`build_contexts`), cached in `baselines.json` |
| `windows.py` | date-aware windows, calendar blocks (the dependence unit) |
| `resampling.py` | KS, block bootstrap null (calibrated), paired block test, BH |
| `selftest.py` | test-of-the-test: size 56%→calibrated α=0.01 line, power table |
| `separation.py` | orchestration: real vs generators, joint energy test, reference contrasts |
| `probes.py` | declared probe family (GJR-t, numba) for reversal_learnability_gap |
| `invariance.py` / `sensitivity.py` | audit layers ① and knob-sensitivity |
| `knockout.py` | mechanism ablation (known omission × detector) |
| result JSONs | regression evidence → frozen into `tests/golden/fixtures/` |

### `financial-abm-lab` (read-only clone at `/workspace/yuitokyouni/financial-abm-lab`)

| Asset | Reused as |
|---|---|
| `src/fabm/rng.py` `make_rngs` (SeedSequence → named children) | seed-tree convention in `provenance/` |
| `imported/PRISM/src/prism/types.py` `ModelAdapter(Protocol)`, `content_hash` | Tier-1 adapter protocol shape (M3) + content-hash convention |
| config-driven experiment + run metadata conventions | `RunManifest` fields |

Not imported at runtime (spec §14.2): the monorepo stays upstream evidence.

## 2. Migration mapping (M1 scope)

| Research source | Product destination | Notes |
|---|---|---|
| `facts.py` 8 suite metrics | `src/sieve/metrics/{distribution,tails,volatility,asymmetry,dependence}.py` | verbatim math, + `MetricSpec` metadata incl. `known_blind_spots` from invariance.json and `prespecification` (VR20/drift = POST_HOC) |
| `generators.py` 6 baselines | `src/sieve/baselines/*` | verbatim math; parameters come from suite data, not runtime fits |
| `resampling.py` `ks_stat`, `block_boot_test`, `benjamini_hochberg` | `src/sieve/inference/{ks,blockboot,multiplicity}.py` | + Holm (suite declares `holm`) |
| calibrated decision line α=0.01 (selftest) | `suites/financial-daily/1.0.0/suite.yaml` `inference.alpha` | provenance note links the calibration |
| real-window stats + calendar blocks | `suites/.../reference_stats.json` (shipped) | derived values, not raw Yahoo series → offline + redistributable |
| baseline run stats (per-index fits) | `suites/.../baseline_stats.json` (shipped) | resolves the bootstrap-reference conflict (see §6) |

## 3. Golden fixtures frozen (done, commit `6ad237c`)

`tests/golden/fixtures/`: `research_commit.txt`, `reference_stats.json`
(124 windows × 8 metrics + blocks), `baseline_stats.json` (6 × 199 runs × 8),
`baseline_params.json`, `research_outputs/*.json` (all current result files).
Generator: `tests/golden/generate_fixtures.py` (requires the research repo + fetched data).

## 4. M0/M1 file tree

As spec §12, rooted at `product/` (see conflict C1). Files marked (M1) land in the second milestone:

```
product/
  README.md  IMPLEMENTATION_PLAN.md  STATUS.md  CHANGELOG.md  pyproject.toml
  src/sieve/
    core/models.py enums.py hashing.py serialization.py
    data/contracts.py ingest.py                      (M1)
    metrics/registry.py distribution.py tails.py volatility.py
            asymmetry.py dependence.py               (M1)
    baselines/registry.py gaussian.py student_t.py bootstrap.py garch.py (M1)
    inference/ks.py blockboot.py multiplicity.py     (M1)
    suites/registry.py loader.py
    adapters/protocol.py csv.py                      (M1)
    evaluation/runner.py profile.py findings.py      (M1)
    provenance/environment.py manifest.py bundle.py verify.py
    reporting/html.py templates/report.html.j2       (M1)
    cli/app.py
  suites/financial-daily/1.0.0/{suite.yaml,reference_stats.json,baseline_stats.json}
  schemas/*.schema.json (exported from pydantic)
  examples/csv_returns/{manifest.yaml,returns.csv}   (M1)
  tests/{unit,integration,golden}/                   fixtures frozen already
  .github/workflows/test.yml lint.yml
```

## 5. Tests per acceptance criterion (spec §16)

| Acceptance | Test |
|---|---|
| Determinism: byte-stable canonical JSON | `tests/integration/test_golden_path.py::test_deterministic_rerun` (two runs, compare canonical bytes minus excluded fields) |
| Bundle verification fails after mutation | `test_golden_path.py::test_verify_detects_tamper` |
| Seed tree in provenance | `test_golden_path.py::test_seed_lineage_recorded` |
| Metrics reproduce research values | `tests/golden/test_metric_parity.py` (every suite metric vs frozen per-window values, exact) |
| Metric unit tests on known synthetic cases | `tests/unit/test_metrics.py` (e.g. kurtosis(gaussian)≈0, leverage(iid)≈0, drift(x+c) shifts) |
| Baseline mechanism smoke tests | `tests/unit/test_baselines.py` (garch clusters: acf_abs_1 up; iid does not; block keeps short memory) |
| No aggregate score | `tests/unit/test_no_score.py` (schema+bundle scan: no `score`/rank field; profile has no mean) |
| NOT_TESTED / INSUFFICIENT first-class | golden path asserts profile contains NOT_TESTED dims; short-input run yields INSUFFICIENT |
| Post-hoc marked | `test_prespecification.py` (VR20/drift carry POST_HOC into results + report) |
| Offline | runner has no network imports; integration test runs with proxy vars unset |
| Report human-readable / verify UX | report renders; `sieve verify` exit codes; CLI help distinguishes eval-fail vs exec-fail |
| Canonical serialization round-trip (M0 exit) | `tests/unit/test_serialization.py` (hand-authored bundle → hash → verify) |

## 6. Assumptions and conflicts discovered

- **C1 — new repository vs session git rules.** Spec §12 wants a fresh repo; this
  session may only push to the designated `sieve-bench` branch. Resolution: the
  product lives at `product/` (self-contained package, no imports from the
  research root), structured as a future repo root — extractable later via
  `git subtree split -P product`. Runtime independence (spec §0.5) is preserved.
- **C2 — offline golden path vs bootstrap baselines.** `iid/block_bootstrap`
  resample the reference pool, but raw Yahoo series must not be redistributed.
  Resolution: ship *derived* per-run statistic values (`baseline_stats.json`)
  frozen from the research commit; baselines are compared as frozen
  distributions with full provenance. Runtime re-simulation stays possible for
  the parametric four via shipped per-index parameters.
- **C3 — Python 3.12 preferred; environment has 3.11.** Target `>=3.11`; no
  3.12-only syntax.
- **C4 — parquet layer.** `polars` chosen (single wheel, writes parquet);
  pandas/pyarrow not installed here.
- **C5 — suite says `holm`, research used BH.** Both implemented in
  `multiplicity.py`; the suite manifest is authoritative per run and the bundle
  records which was applied.
- **C6 — reference data provenance.** `reference_stats.json` carries the source
  hashes from the research `fetch.py` (SHA-256 of (timestamp, close) pairs) so
  a user who fetches the same window can re-derive and verify the shipped values.
- **C7 — language.** Product surfaces are English (outreach requirement);
  research repo stays as-is.
- **C8 — `uv` present** but the sandbox has a shared site-packages; `pyproject.toml`
  is uv-compatible, dev here uses the ambient interpreter. CI uses uv.
