"""M0 exit criterion: canonical serialization round-trips and the seal is
deterministic across volatile IDs (spec §16)."""

import datetime as dt
import uuid

import pytest

from sieve.core.models import (
    ClaimSpec,
    DatasetManifest,
    EvidenceBundle,
    ModelManifest,
    RunManifest,
    TestSuiteManifest,
    ValidationProfile,
)
from sieve.core.serialization import (
    canonical_bytes,
    hashable_bytes,
    null_out_excluded,
    to_jsonable,
)
from sieve.provenance.bundle import load, seal, verify, write


def mini_bundle(run_id="r1", note=None) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id=uuid.uuid4(),
        created_at=dt.datetime.now(dt.timezone.utc),
        run_manifest=RunManifest(
            run_id=run_id,
            created_at=dt.datetime.now(dt.timezone.utc),
            sieve_version="0.1.0", command="sieve test x", master_seed=1,
            input_path="x.csv", input_hash="00"),
        model=ModelManifest(model_id="m", model_version="1",
                            display_name="M", adapter_id="csv@1", notes=note),
        dataset=DatasetManifest(dataset_id="d"),
        claim=ClaimSpec(claim_id="c", version="1", statement="s",
                        use_case="u"),
        suite=TestSuiteManifest(suite_id="s", version="1",
                                claim_types=["descriptive"]),
        profile=ValidationProfile.not_tested(),
        results=[], findings=[])


def test_canonical_bytes_sorted_compact_newline():
    body = canonical_bytes({"b": 1, "a": [1.5, None]})
    assert body == b'{"a":[1.5,null],"b":1}\n'


def test_canonical_bytes_rejects_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        canonical_bytes({"x": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="non-finite"):
        canonical_bytes({"x": {"y": float("inf")}})


def test_null_out_excluded_nulls_shape_preserving():
    data = to_jsonable(mini_bundle())
    out = null_out_excluded(data)
    assert out["bundle_id"] is None
    assert out["created_at"] is None
    assert out["bundle_hash"] is None
    assert out["artifact_index"] is None
    assert out["run_manifest"]["run_id"] is None
    assert out["run_manifest"]["created_at"] is None
    assert out["run_manifest"]["command"] is None
    assert out["run_manifest"]["input_path"] is None
    assert out["run_manifest"]["environment"] is None
    assert out["dataset"]["source_uri"] is None
    # untouched fields survive, original is not mutated
    assert out["model"]["model_id"] == "m"
    assert out["run_manifest"]["master_seed"] == 1
    assert out["run_manifest"]["input_hash"] == "00"
    assert data["bundle_id"] is not None


def test_seal_ignores_volatile_ids():
    a, b = mini_bundle(run_id="run-A"), mini_bundle(run_id="run-B")
    assert a.bundle_id != b.bundle_id
    assert hashable_bytes(a) == hashable_bytes(b)
    assert seal(a).bundle_hash == seal(b).bundle_hash


def test_seal_ignores_machine_local_facts():
    """Same content on another machine or path ⇒ same scientific seal."""
    a, b = mini_bundle(), mini_bundle()
    b.run_manifest.command = "sieve test /somewhere/else/x.csv --suite s"
    b.run_manifest.input_path = "/somewhere/else/x.csv"
    b.run_manifest.environment = {"python": "3.12.1",
                                  "platform": "Windows-11-AMD64"}
    b.dataset.source_uri = "C:\\data\\x.csv"
    assert hashable_bytes(a) == hashable_bytes(b)


def test_seal_still_pins_data_and_seed():
    a, b, c = mini_bundle(), mini_bundle(), mini_bundle()
    b.run_manifest.input_hash = "ff"          # different data content
    c.run_manifest.master_seed = 2            # different seed
    assert hashable_bytes(a) != hashable_bytes(b)
    assert hashable_bytes(a) != hashable_bytes(c)


def test_seal_changes_with_content():
    a, b = mini_bundle(), mini_bundle(note="different content")
    assert seal(a).bundle_hash != seal(b).bundle_hash


def test_write_load_verify_roundtrip(tmp_path):
    bundle = seal(mini_bundle())
    write(bundle, tmp_path)
    assert (tmp_path / "evidence_bundle.json").exists()
    assert (tmp_path / "bundle.sha256").exists()
    reloaded = load(tmp_path / "evidence_bundle.json")
    assert reloaded.bundle_hash == bundle.bundle_hash
    assert reloaded.model.model_id == "m"
    assert verify(tmp_path) == []


def test_verify_flags_missing_sidecar(tmp_path):
    write(seal(mini_bundle()), tmp_path)
    (tmp_path / "bundle.sha256").unlink()
    assert any("sidecar" in p for p in verify(tmp_path))


def test_verify_flags_edited_bundle(tmp_path):
    write(seal(mini_bundle()), tmp_path)
    p = tmp_path / "evidence_bundle.json"
    p.write_text(p.read_text().replace('"display_name":"M"',
                                       '"display_name":"Q"'))
    problems = verify(tmp_path)
    assert any("bundle_hash mismatch" in x for x in problems)
    assert any("sidecar" in x for x in problems)
