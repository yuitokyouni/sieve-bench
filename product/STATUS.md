# Status

Updated: 2026-08-08. Spec: `SIEVE_PRODUCT_IMPLEMENTATION_SPEC` v0.1 (§16
acceptance criteria; §22 ordered M0 → M1 only).

## M0 — foundations: DONE

- [x] Package skeleton (`src/sieve`, `pyproject.toml`, `sieve` entry point)
- [x] Versioned schemas for every durable artifact (`core/models.py`,
      exported to `schemas/*.schema.json`)
- [x] Canonical serialization: sorted keys, compact, `\n`, non-finite
      forbidden, volatile fields nulled shape-preserving
      (`core/serialization.py`)
- [x] Two-layer seal + verify (`provenance/bundle.py`): deterministic
      scientific `bundle_hash` and sha256sum-compatible file sidecar
- [x] Seed protocol: master `SeedSequence` → named children, recorded in the
      run manifest (`provenance/environment.py`)
- [x] `sieve --help`, `sieve doctor`, `sieve schemas export`
- [x] M0 exit test: hand-authored bundle → seal → write → load → verify
      (`tests/unit/test_serialization.py`)

## M1 — offline golden path: DONE

`sieve test examples/csv_returns --suite financial-daily@1.0 --claim
descriptive-market-dynamics` deterministically produces `manifest.json`,
`observations.parquet`, `results.json`, `findings.json`,
`evidence_bundle.json`, `report/index.html`, `artifacts/`, `bundle.sha256` —
locally, offline, ~5 s.

- [x] Tier-0 CSV adapter with provenance-gap recording (`adapters/csv.py`)
- [x] 8 metrics migrated verbatim, with dimensions, known blind spots and
      POST_HOC disclosure (`metrics/`)
- [x] 6 baselines with declared mechanisms; frozen distributions shipped,
      parametric generators runnable from shipped per-index MLE parameters;
      bootstrap pair refuses to run without a local pool, explicitly
      (`baselines/`)
- [x] Calibrated inference: calendar-block bootstrap null, α = 0.01 line,
      Holm (suite-declared; BH also implemented) (`inference/`)
- [x] `financial-daily@1.0.0` suite: immutable content hash over manifest +
      shipped data; claim `descriptive-market-dynamics`
- [x] Ten-dimension profile with first-class NOT_TESTED / INSUFFICIENT;
      reproducibility scored from manifest completeness (`evaluation/profile.py`)
- [x] FAIL → findings with author questions (`evaluation/findings.py`)
- [x] HTML report in the spec §9.1 section order, POST HOC badges,
      baseline-context column, no aggregate anywhere (`reporting/`)
- [x] `sieve test/verify/report`, exit-code contract (eval ≠ exec failure)

### Acceptance criteria → tests (all passing, 75 total)

| Criterion | Test |
|---|---|
| Determinism (same input+suite+seed → same seal) | `test_golden_path.py::test_deterministic_rerun` |
| Verification fails after modification | `::test_verify_detects_{artifact,bundle}_tamper` |
| Seed lineage recorded | `::test_seed_lineage_recorded` |
| Metrics reproduce research values bit-for-bit | `tests/golden/test_metric_parity.py` (synthetic + 124 real windows) |
| Suite ships frozen fixture bytes | `test_metric_parity.py::test_suite_ships_frozen_fixture_bytes` |
| Known-case metric behavior | `tests/unit/test_metrics.py` |
| Baseline mechanism smoke | `tests/unit/test_baselines.py` |
| No aggregate score | `tests/unit/test_no_score.py` |
| NOT_TESTED / INSUFFICIENT first-class | `::test_profile_has_first_class_not_tested`, `::test_insufficient_on_short_input` |
| POST_HOC visible in results + report | `::test_post_hoc_disclosure_reaches_results_and_report` |
| Offline | `::test_offline_no_network_modules_in_product` |
| CLI exit-code contract | `tests/unit/test_cli.py` |

## Not in M1 (deliberately)

Tier-1/2 adapters, `sieve compare`, multiscale/regime/intervention dimensions
(they appear as NOT_TESTED, honestly), suite update tooling, any service
component. Next milestone per spec §13: M2 (comparison + more claims).

## Known limitations on record

- Reference dependence leaves ~6 independent calendar blocks: power against
  modest differences is limited, and every PASS carries that caveat.
- The example input FAILs `hill_left` as well as `leverage`: a single
  parameter set does not span the cross-index spread of the reference. Kept
  as-is — it demonstrates that distribution-over-windows testing is stricter
  than moment matching.
- `polars` writes `observations.parquet`; parquet bytes are not asserted
  byte-stable across polars versions (values are, via frame equality in
  tests; the deterministic seal covers results, not parquet bytes).
