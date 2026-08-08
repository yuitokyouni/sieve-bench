# Changelog

## 0.1.0 — 2026-08-08

First working release: M0 (foundations) + M1 (offline golden path).

- `sieve test <input> --suite financial-daily@1.0 --claim
  descriptive-market-dynamics` produces a complete, deterministic, verifiable
  run directory (report + sealed evidence bundle) locally and offline.
- 8 metrics, 6 mechanism-declared baselines, calibrated calendar-block
  inference — migrated verbatim from the sieve-bench research repository and
  pinned bit-for-bit by golden regression tests.
- Two-layer integrity: deterministic scientific `bundle_hash` (volatile IDs
  excluded) + sha256sum-compatible file sidecar; `sieve verify` reports
  tampering as a result, not a crash.
- Ten-dimension validation profile with first-class NOT_TESTED /
  INSUFFICIENT; POST HOC disclosure; per-row baseline blind-spot context; no
  aggregate score anywhere (enforced by test).
