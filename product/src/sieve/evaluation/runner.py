"""The golden path: input → windows → tests → profile → findings → bundle.

Semantics (documented, tested):

- The input series is cut into **non-overlapping** windows of the suite's
  window length. Fewer than 5 windows ⇒ every statistical test is
  INSUFFICIENT (that is an answer, not an error).
- Each metric compares input-window values against the suite's shipped
  reference distribution (124 real windows with calendar blocks) using the
  design-preserving block bootstrap null; p-values are adjusted by the
  suite-declared family procedure; the decision line is the suite-declared
  calibrated alpha.
- Baseline context is attached to every metric: which shipped baselines this
  metric fails to separate from the reference (p ≥ alpha at the same line).
  A metric that "passes" the input while failing to separate garch_t proves
  little, and the report says so on the row itself.
- Nothing aggregates across dimensions. Ever.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import numpy as np

from sieve import __version__
from sieve.adapters.csv import load_input
from sieve.core.enums import Prespecification, Status
from sieve.core.hashing import sha256_file
from sieve.core.models import ArtifactRef, EvidenceBundle, RunManifest, TestResult
from sieve.evaluation.findings import build_findings
from sieve.evaluation.profile import build_profile
from sieve.inference.blockboot import block_boot_test
from sieve.inference.ks import ks_stat
from sieve.inference.multiplicity import ADJUSTMENTS
from sieve.metrics import registry as metric_registry
from sieve.provenance.bundle import seal, write
from sieve.provenance.environment import environment_fingerprint, make_rngs
from sieve.reporting.html import render_report
from sieve.suites.loader import load as load_suite

MIN_WINDOWS = 5


def cut_windows(r: np.ndarray, window: int) -> list[np.ndarray]:
    return [r[i:i + window] for i in range(0, len(r) - window + 1, window)]


def run_test(input_path: str | Path, suite_ref: str, claim_id: str,
             out_root: str | Path = ".sieve/runs",
             master_seed: int = 20260802) -> Path:
    suite = load_suite(suite_ref)
    claim = suite.claim(claim_id)
    r, model, dataset = load_input(input_path)

    rngs, seed_tree = make_rngs(master_seed)
    run_id = uuid.uuid4().hex[:12]
    run_dir = Path(out_root) / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "report").mkdir(parents=True, exist_ok=True)

    ref = suite.reference_stats()
    base = suite.baseline_stats()
    window = int(suite.manifest.reference.get("window_length", 1000))
    alpha = float(suite.manifest.inference.get("alpha", 0.01))
    n_draw = int(suite.manifest.inference.get("n_draw", 2000))
    adjust_name = str(suite.manifest.inference.get("multiple_testing", "holm"))
    adjust = ADJUSTMENTS[adjust_name]

    wins = cut_windows(r, window)
    ref_blocks = np.array([w["block"] for w in ref["windows"]])

    # ---------------- observations: input window x metric ------------------
    metric_refs = suite.manifest.metrics
    obs: dict[str, list[float]] = {}
    for mref in metric_refs:
        mid = mref.partition("@")[0]
        obs[mid] = [metric_registry.compute(mref, w) for w in wins]
    _write_observations(run_dir, wins, obs)

    # ---------------- per-metric test vs reference --------------------------
    results: list[TestResult] = []
    raw_p: list[float] = []
    baseline_context: dict[str, list[str]] = {}
    for mref in metric_refs:
        mid = mref.partition("@")[0]
        _, spec, _dim = metric_registry.resolve(mref)
        ref_vals = np.asarray(ref["values"][mid], float)
        in_vals = np.asarray(obs[mid], float)

        # which shipped baselines does this metric fail to separate?
        blind = []
        for bref in suite.manifest.baselines:
            bid = bref.partition("@")[0]
            bvals = np.asarray(base["values"][bid][mid], float)
            _, bp, _, _ = block_boot_test(ref_vals, ref_blocks, bvals,
                                          ks_stat, rngs["baselines"], n_draw)
            if not np.isfinite(bp) or bp >= alpha:
                blind.append(bid)
        baseline_context[mid] = blind

        caveats = [
            "decision line alpha=%.3g is the calibrated nominal level whose "
            "true size is 3-5%% under the reference dependence structure "
            "(research selftest)" % alpha,
        ]
        if blind:
            caveats.append(
                "at the same decision line this metric does NOT separate the "
                "reference from: " + ", ".join(blind)
                + " — agreement with the reference on this metric is weak "
                  "evidence for the stated claim")
        if spec.prespecification is Prespecification.POST_HOC:
            caveats.append(
                "this diagnostic was added post hoc during the research phase "
                "and should be read as a discovered failure mode, not as "
                "pre-registered confirmatory evidence")

        if len(wins) < MIN_WINDOWS or np.isfinite(in_vals).sum() < MIN_WINDOWS:
            results.append(TestResult(
                test_id=f"{mid}::vs-reference", metric_ref=mref,
                statistic_name="ks", statistic_value=None, p_value=None,
                n_reference=len(ref_vals), n_simulation=len(wins),
                status=Status.INSUFFICIENT,
                prespecification=spec.prespecification,
                caveats=caveats + [
                    f"only {len(wins)} non-overlapping windows of length "
                    f"{window} available; need >= {MIN_WINDOWS}"]))
            raw_p.append(float("nan"))
            continue

        k, p, _, n_blocks = block_boot_test(ref_vals, ref_blocks, in_vals,
                                            ks_stat, rngs["resampling"], n_draw)
        results.append(TestResult(
            test_id=f"{mid}::vs-reference", metric_ref=mref,
            statistic_name="ks",
            statistic_value=None if not np.isfinite(k) else k,
            p_value=None if not np.isfinite(p) else p,
            effect_size=None if not np.isfinite(k) else k,
            n_reference=len(ref_vals), n_simulation=len(wins),
            status=Status.PASS,          # provisional; set after adjustment
            prespecification=spec.prespecification,
            caveats=caveats))
        raw_p.append(p)

    adj = adjust(raw_p)
    for res, ap in zip(results, adj):
        if res.status is Status.INSUFFICIENT:
            continue
        res.adjusted_p_value = None if not np.isfinite(ap) else float(ap)
        if res.p_value is None:
            res.status = Status.INSUFFICIENT
        elif ap < alpha:
            res.status = Status.FAIL
        else:
            res.status = Status.PASS
            res.caveats.append(
                "PASS means 'not separated from the reference at the "
                "calibrated line', not 'validated'; power against modest "
                "differences is limited by ~6 independent calendar blocks")

    profile = build_profile(results, suite.manifest.metrics, model)
    findings = build_findings(results, baseline_context, claim)

    manifest = RunManifest(
        run_id=run_id, created_at=dt.datetime.now(dt.timezone.utc),
        sieve_version=__version__,
        command=f"sieve test {input_path} --suite {suite_ref} --claim {claim_id}",
        master_seed=master_seed, seed_tree=seed_tree,
        environment=environment_fingerprint(),
        input_path=str(input_path),
        input_hash=dataset.content_hash)

    limitations = [
        "reference and baseline distributions are frozen derived statistics "
        "from sieve-bench@" + str(ref.get("commit", "unknown"))
        + "; raw reference series are not shipped",
        "multiscale behavior, regime response and intervention validity are "
        "not tested by this suite version",
        "the input series is treated as one stationary sample; no regime "
        "segmentation is applied",
    ]

    bundle = EvidenceBundle(
        bundle_id=uuid.uuid4(),
        created_at=dt.datetime.now(dt.timezone.utc),
        run_manifest=manifest, model=model, dataset=dataset, claim=claim,
        suite=suite.manifest, profile=profile, results=results,
        findings=findings, limitations=limitations, artifact_index=[])

    # Seal first: bundle_hash is the deterministic scientific seal and the
    # report displays it. The artifact index is filled afterwards and is
    # covered by the bundle.sha256 sidecar instead (see provenance.bundle).
    seal(bundle)

    _write_json(run_dir / "manifest.json", json.loads(manifest.model_dump_json()))
    _write_json(run_dir / "results.json",
                [json.loads(x.model_dump_json()) for x in results])
    _write_json(run_dir / "findings.json",
                [json.loads(x.model_dump_json()) for x in findings])
    _write_json(run_dir / "artifacts" / "baseline_context.json", baseline_context)
    render_report(run_dir / "report" / "index.html", bundle, baseline_context)

    for rel in ("manifest.json", "observations.parquet", "results.json",
                "findings.json", "artifacts/baseline_context.json",
                "report/index.html"):
        p = run_dir / rel
        kind = ("report" if rel.startswith("report") else
                "table" if rel.endswith((".parquet", ".json")) else "other")
        bundle.artifact_index.append(
            ArtifactRef(path=rel, sha256=sha256_file(p), kind=kind))

    write(bundle, run_dir)
    return run_dir


def _write_observations(run_dir: Path, wins, obs: dict[str, list[float]]) -> None:
    import polars as pl

    rows = {"window": list(range(len(wins))),
            "n_obs": [len(w) for w in wins]}
    rows.update(obs)
    pl.DataFrame(rows).write_parquet(run_dir / "observations.parquet")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True,
                               ensure_ascii=False) + "\n")
