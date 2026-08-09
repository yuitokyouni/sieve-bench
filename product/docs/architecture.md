# Architecture notes

Design decisions worth explaining, including the ones we got wrong first.
Sieve asks simulation authors to expose their models to audit; the least it
can do is keep its own audit trail public.

## The two-layer seal

Every run produces two hashes with deliberately different scopes:

- **`bundle_hash` — scientific identity.** SHA-256 of the canonical bundle
  JSON with volatile fields nulled shape-preserving. It covers the data
  content hash, suite hash, claim, master seed and seed tree, package
  versions, all results, the profile and the findings. It excludes run IDs,
  timestamps, the artifact index, filesystem paths and the machine
  fingerprint. Contract: same input bytes + same suite + same seed + same
  package versions ⇒ same seal, on any machine. This is the hash to quote.
- **`bundle.sha256` — distribution integrity.** A sha256sum-compatible
  sidecar over the written `evidence_bundle.json` bytes, which include the
  artifact index (per-file hashes of the report, results, observations,
  manifest). It pins the run directory as shipped. `sha256sum -c
  bundle.sha256` verifies it with no Sieve installed.

`sieve verify` checks both layers plus every artifact hash and returns
mismatches as results, not exceptions. Threat model: editing any artifact or
the bundle file trips the file layer; editing scientific content trips both;
regenerating an entire self-consistent run directory is always possible — but
then the scientific seal no longer matches the one that was quoted, which is
the point of quoting it.

## Audit notes: two seal-scope corrections during M1

The seal format was corrected twice before v0.1.0, both times because our own
regression tests (not a user) caught the leak. Kept on record deliberately.

1. **Artifact hashes reintroduced volatile IDs.** During M1 we found that
   the artifact index — file hashes of `manifest.json` and the HTML report,
   which legitimately contain the run ID and timestamps — was accidentally
   reintroducing volatile run identifiers into the supposedly deterministic
   seal (and the report was rendered before sealing, so it displayed an empty
   hash). We changed the format to separate scientific identity from
   distribution integrity: the artifact index moved outside `bundle_hash` and
   under the file-integrity sidecar, sealing now happens before the report is
   rendered, and regression tests cover both layers
   (`test_deterministic_rerun`, `test_verify_detects_{artifact,bundle}_tamper`).

2. **The machine fingerprint made the seal machine-local.** While preparing
   the independent-reproduction protocol we found the seal also covered the
   platform string, local filesystem paths and the CLI invocation — so a
   third party rerunning the same bytes on their own machine could never
   reproduce the quoted hash. Those fields stay in the bundle (they are
   provenance) but moved outside the seal; data identity inside the seal is
   the content hash, never a path. Regression tests:
   `test_seal_ignores_machine_local_facts`, `test_seal_is_path_independent`,
   `test_seal_still_pins_data_and_seed`.

The residual reproducibility caveat is honest and documented: floating-point
results can legitimately change across major dependency versions (that is a
different computation, and the seal *should* differ). The reproduction
contract therefore names package versions — see `docs/reproduce.md`.

## Why the reference ships as derived statistics

The suite contains 124 per-window statistic values with calendar-block
labels, not raw index series (which are not redistributable). This is what
makes the golden path fully offline. Each shipped distribution carries the
research commit it was frozen at and per-index SHA-256 hashes of the
`(timestamp, close)` source pairs, so anyone who fetches the same window can
verify they derived from the same source without Sieve ever distributing it.

## Why there is no score

Evidence is claim-scoped: a simulation that is excellent for volatility
texture and useless for leverage response is not "0.7 good" — it is exactly
that profile, and any scalar would erase the information a decision needs.
The absence of aggregation is enforced by tests that scan both the schemas
and the source text (`tests/unit/test_no_score.py`). This is also why
different simulation kinds (daily returns vs limit-order-book dynamics) get
different suites rather than adapters into one universal test.
