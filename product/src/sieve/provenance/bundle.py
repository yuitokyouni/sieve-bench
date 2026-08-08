"""Assemble, seal, write and verify EvidenceBundles.

Two integrity layers with different scopes, both checked by ``verify``:

1. **Scientific seal** — ``bundle_hash``: SHA-256 of the canonical bundle with
   volatile fields (IDs, timestamps, artifact index) nulled. Deterministic:
   same input + suite + seed → same seal, byte for byte. This is the hash to
   quote in a paper or PR; it pins what was measured, not when or where.
2. **File integrity** — ``bundle.sha256``: sha256sum-compatible sidecar over
   the written ``evidence_bundle.json`` bytes (which include the artifact
   index), plus per-artifact hashes inside that index. This pins the run
   directory as shipped; ``sha256sum -c bundle.sha256`` also works.

Threat model: modifying any artifact, the index, or the bundle file trips
layer 2; modifying scientific content trips both. An adversary who regenerates
the entire run directory can of course produce a self-consistent layer 2 —
but then the layer-1 seal no longer matches the one that was quoted.

A verification failure never raises: it returns the list of mismatches,
because "this bundle was modified" is a result, not an exception (spec §8).
"""

from __future__ import annotations

import json
from pathlib import Path

from sieve.core.hashing import sha256_bytes, sha256_file
from sieve.core.models import EvidenceBundle
from sieve.core.serialization import canonical_bytes, hashable_bytes, to_jsonable


def seal(bundle: EvidenceBundle) -> EvidenceBundle:
    """Compute the deterministic scientific seal. Call before rendering the
    report (the report displays the seal) and before filling artifact_index
    (the index is outside the seal by design)."""
    bundle.bundle_hash = sha256_bytes(hashable_bytes(bundle))
    return bundle


def write(bundle: EvidenceBundle, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "evidence_bundle.json"
    body = canonical_bytes(to_jsonable(bundle))
    path.write_bytes(body)
    (run_dir / "bundle.sha256").write_text(
        f"{sha256_bytes(body)}  evidence_bundle.json\n")
    return path


def load(path: str | Path) -> EvidenceBundle:
    return EvidenceBundle.model_validate(json.loads(Path(path).read_text()))


def verify(run_dir_or_bundle: str | Path) -> list[str]:
    """Return a list of problems; empty list = intact."""
    p = Path(run_dir_or_bundle)
    if p.is_dir():
        bundle_path, base = p / "evidence_bundle.json", p
    else:
        bundle_path, base = p, p.parent
    problems: list[str] = []
    if not bundle_path.exists():
        return [f"missing {bundle_path}"]
    try:
        bundle = load(bundle_path)
    except Exception as e:                      # malformed is a verify failure
        return [f"unparseable bundle: {e}"]

    recomputed = sha256_bytes(hashable_bytes(bundle))
    if recomputed != bundle.bundle_hash:
        problems.append(
            f"bundle_hash mismatch: recorded {bundle.bundle_hash[:16]}…, "
            f"recomputed {recomputed[:16]}…")

    sidecar = base / "bundle.sha256"
    if sidecar.exists():
        recorded = sidecar.read_text().split()[0]
        if recorded != sha256_file(bundle_path):
            problems.append(
                "bundle.sha256 sidecar disagrees with evidence_bundle.json "
                "file bytes")
    else:
        problems.append("missing bundle.sha256 sidecar")

    for ref in bundle.artifact_index:
        ap = base / ref.path
        if not ap.exists():
            problems.append(f"missing artifact {ref.path}")
        elif sha256_file(ap) != ref.sha256:
            problems.append(f"artifact modified: {ref.path}")
    return problems
