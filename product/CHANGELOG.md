# Changelog

## 0.1.0 — 2026-08-09

First public release: M0 (foundations) + M1 (offline golden path) + M1.5
(public credibility release).

Pinned by this release:

- suite `financial-daily@1.0.0`, suite_hash
  `7808f705d610cfde4621d4ea5b46dced75aaa3816558d64b6e9051d24891df61`
- frozen example run `docs/example-run/`, bundle_hash
  `31bda432c319adfcc40a25e137f50dff15446513c886dfc85cbcf61138dbedf5`
- reproduction protocol and environment: `docs/reproduce.md`

Seal scope (M1.5): `bundle_hash` now excludes the platform fingerprint,
filesystem paths and CLI invocation, so the same input bytes + suite + seed +
package versions reproduce the same seal on any machine. Those fields remain
in the bundle under the file-integrity layer. History and rationale:
`docs/architecture.md` ("Audit notes").

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
