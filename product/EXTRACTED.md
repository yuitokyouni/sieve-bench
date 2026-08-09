# This directory has been extracted

The Sieve product now lives at **https://github.com/yuitokyouni/sieve**
(canonical branch `main`), extracted from this directory at commit
`de13a8f` via `git subtree split -P product` (split head `2f17dc1`,
released as v0.1.0). Full product history was preserved.

From that point on:

- **Product development happens in `yuitokyouni/sieve`.** This directory is
  frozen as the historical source and is not synchronized.
- **This repository (sieve-bench) remains the research provenance**: the
  metrics, baselines, reference statistics and calibrated inference shipped
  by the product were migrated verbatim from the research code here, and the
  product's golden regression tests pin that parity bit-for-bit.
