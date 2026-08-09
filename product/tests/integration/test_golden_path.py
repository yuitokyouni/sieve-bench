"""M1 exit criterion (spec §16, §22): the offline golden path produces a
complete, deterministic, verifiable run directory — and INSUFFICIENT /
NOT_TESTED / POST_HOC are visible, first-class outcomes."""

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from sieve.evaluation.runner import run_test
from sieve.provenance.bundle import load, verify

PRODUCT = Path(__file__).resolve().parents[2]
EXAMPLE = PRODUCT / "examples" / "csv_returns"
SUITE, CLAIM = "financial-daily@1.0", "descriptive-market-dynamics"

EXPECTED_FILES = ("manifest.json", "observations.parquet", "results.json",
                  "findings.json", "evidence_bundle.json", "bundle.sha256",
                  "report/index.html", "artifacts/baseline_context.json")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    return run_test(EXAMPLE, SUITE, CLAIM,
                    out_root=tmp_path_factory.mktemp("runs"))


@pytest.fixture(scope="module")
def rerun_dir(tmp_path_factory) -> Path:
    return run_test(EXAMPLE, SUITE, CLAIM,
                    out_root=tmp_path_factory.mktemp("runs2"))


def test_all_outputs_exist(run_dir):
    for rel in EXPECTED_FILES:
        assert (run_dir / rel).exists(), rel


def test_bundle_verifies_intact(run_dir):
    assert verify(run_dir) == []


def test_seal_is_path_independent(run_dir, tmp_path):
    """The same input content in a different location yields the same
    scientific seal — the reproduction contract for third parties."""
    elsewhere = tmp_path / "moved" / "input"
    shutil.copytree(EXAMPLE, elsewhere)
    other = run_test(elsewhere, SUITE, CLAIM, out_root=tmp_path / "runs")
    b1 = load(run_dir / "evidence_bundle.json")
    b2 = load(other / "evidence_bundle.json")
    assert b1.run_manifest.input_path != b2.run_manifest.input_path
    assert b1.bundle_hash == b2.bundle_hash


def test_deterministic_rerun(run_dir, rerun_dir):
    b1, b2 = load(run_dir / "evidence_bundle.json"), \
             load(rerun_dir / "evidence_bundle.json")
    assert b1.bundle_hash == b2.bundle_hash
    assert (run_dir / "results.json").read_bytes() == \
           (rerun_dir / "results.json").read_bytes()
    assert (run_dir / "findings.json").read_bytes() == \
           (rerun_dir / "findings.json").read_bytes()
    f1 = pl.read_parquet(run_dir / "observations.parquet")
    f2 = pl.read_parquet(rerun_dir / "observations.parquet")
    assert f1.equals(f2)
    # volatile identifiers do differ — that is the point of excluding them
    assert b1.run_manifest.run_id != b2.run_manifest.run_id


def test_verify_detects_artifact_tamper(run_dir, tmp_path):
    tampered = tmp_path / "t1"
    shutil.copytree(run_dir, tampered)
    p = tampered / "results.json"
    body = json.loads(p.read_text())
    body[0]["p_value"] = 0.5
    p.write_text(json.dumps(body))
    problems = verify(tampered)
    assert any("results.json" in x for x in problems)


def test_verify_detects_bundle_tamper(run_dir, tmp_path):
    tampered = tmp_path / "t2"
    shutil.copytree(run_dir, tampered)
    p = tampered / "evidence_bundle.json"
    body = json.loads(p.read_text())
    for res in body["results"]:
        res["status"] = "PASS"
    p.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")))
    problems = verify(tampered)
    assert any("bundle_hash mismatch" in x for x in problems)
    assert any("sidecar" in x for x in problems)


def test_seed_lineage_recorded(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    names = [n["name"] for n in manifest["seed_tree"]]
    assert names == ["windows", "resampling", "baselines", "report"]
    assert all("spawn_key" in n for n in manifest["seed_tree"])
    assert manifest["master_seed"] == 20260802


def test_profile_has_first_class_not_tested(run_dir):
    bundle = load(run_dir / "evidence_bundle.json")
    by_dim = {d.dimension: d for d in bundle.profile.dimensions}
    assert len(by_dim) == 10
    for dim in ("multiscale_behavior", "regime_response",
                "intervention_validity"):
        assert by_dim[dim].status.value == "NOT_TESTED"
        assert by_dim[dim].note
    # example manifest omits git_commit/code_uri → provenance dimension warns
    assert by_dim["reproducibility_provenance"].status.value == "WARN"


def test_post_hoc_disclosure_reaches_results_and_report(run_dir):
    results = json.loads((run_dir / "results.json").read_text())
    by_metric = {r["metric_ref"].split("@")[0]: r for r in results}
    for mid in ("variance_ratio_20", "drift"):
        assert by_metric[mid]["prespecification"] == "POST_HOC"
    assert "POST HOC" in (run_dir / "report" / "index.html").read_text()


def test_report_is_human_readable_and_scoreless(run_dir):
    html = (run_dir / "report" / "index.html").read_text()
    bundle = load(run_dir / "evidence_bundle.json")
    assert "Claim under evaluation" in html
    assert "Validation profile" in html
    assert bundle.bundle_hash[:16] in html
    assert "No aggregate score exists" in html
    for mid in ("excess_kurtosis", "leverage", "drift"):
        assert mid in html


def test_baseline_context_attached_to_passes(run_dir):
    """A PASS on a metric that cannot separate garch_t must say so."""
    results = json.loads((run_dir / "results.json").read_text())
    ctx = json.loads(
        (run_dir / "artifacts" / "baseline_context.json").read_text())
    passing_blind = [r for r in results if r["status"] == "PASS"
                     and ctx[r["metric_ref"].split("@")[0]]]
    assert passing_blind, "example should have at least one weak PASS"
    for r in passing_blind:
        assert any("does NOT separate" in c for c in r["caveats"])


def test_insufficient_on_short_input(tmp_path):
    src = (EXAMPLE / "returns.csv").read_text().splitlines()
    short = tmp_path / "returns.csv"
    short.write_text("\n".join(src[:1201]) + "\n")     # 1200 returns → 1 window
    run_dir = run_test(short, SUITE, CLAIM, out_root=tmp_path / "runs")
    results = json.loads((run_dir / "results.json").read_text())
    assert results
    assert all(r["status"] == "INSUFFICIENT" for r in results)
    assert all(r["p_value"] is None for r in results)
    assert verify(run_dir) == []       # an INSUFFICIENT run still seals


def test_offline_no_network_modules_in_product():
    for f in (PRODUCT / "src").rglob("*.py"):
        text = f.read_text()
        for token in ("import requests", "import urllib", "import http.client",
                      "import socket", "urlopen("):
            assert token not in text, (f, token)
