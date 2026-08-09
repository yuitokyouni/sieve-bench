# Reproduce the example result

This is the protocol we ask independent reproducers to run. It takes about
five minutes, uses no network after install, and ends with two checks: the
statuses and the scientific seal.

## Environment

Any OS. Python ≥ 3.11. Results are exact given the same package versions;
the frozen example below was produced with:

| package | version |
|---|---|
| python | 3.11.x |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pydantic | 2.13.4 |
| polars | 1.43.2 |

Different major versions will still reproduce the statuses; the byte-exact
`bundle_hash` is only guaranteed under matching numerical stack versions
(a different floating-point stack is a different computation, and the seal is
supposed to notice).

## Steps

```
git clone https://github.com/yuitokyouni/sieve
cd sieve
pip install -e .            # or: uv pip install -e .

# 1. integrity of the shipped example, without running anything
sieve verify docs/example-run

# 2. rerun the golden path from scratch (offline, ~5 s)
sieve test examples/csv_returns --suite financial-daily@1.0 \
      --claim descriptive-market-dynamics --out /tmp/sieve-runs

# 3. verify your own run
sieve verify /tmp/sieve-runs/<run-id>
```

## Expected results

The validation profile printed by step 2 (and in
`report/index.html`):

| dimension | status |
|---|---|
| marginal_distribution | PASS |
| tail_behavior | **FAIL** |
| return_dependence | PASS |
| volatility_dynamics | PASS |
| leverage_asymmetry | **FAIL** |
| multiscale_behavior | NOT_TESTED |
| drift_nonstationarity | PASS |
| regime_response | NOT_TESTED |
| intervention_validity | NOT_TESTED |
| reproducibility_provenance | WARN |

The two FAILs are the point of the example: the input is a GARCH(1,1)-t
sample, and `leverage_asymmetry` is exactly the mechanism GARCH does not
contain; the left-tail FAIL shows a single parameter set not spanning the
cross-index reference spread.

The scientific seal in `evidence_bundle.json` (`bundle_hash`) and printed by
`sieve test`:

```
31bda432c319adfcc40a25e137f50dff15446513c886dfc85cbcf61138dbedf5
```

The suite hash recorded in the same bundle (`suite.suite_hash`):

```
7808f705d610cfde4621d4ea5b46dced75aaa3816558d64b6e9051d24891df61
```

Your `run_id`, timestamps and machine fingerprint will differ — they are
outside the seal by design (`docs/architecture.md`).

## If something does not match

That is a result, and we want to hear it: please open an issue with your
`results.json`, `manifest.json` (it contains the environment fingerprint)
and OS. A reproduction failure under matching package versions would be a
bug in Sieve's determinism contract, which is exactly what this protocol
exists to catch.
