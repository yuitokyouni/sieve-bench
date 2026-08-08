"""Suite loading. A suite is versioned data + config; immutable once published.

``suite_hash`` covers the manifest YAML and every data file the suite ships,
so two runs claiming ``financial-daily@1.0.0`` either used byte-identical
suite content or their bundles say otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sieve.core.hashing import sha256_bytes, sha256_file
from sieve.core.models import ClaimSpec, TestSuiteManifest

_SEARCH: list[Path] = []


def add_search_path(p: str | Path) -> None:
    p = Path(p)
    if p not in _SEARCH:
        _SEARCH.insert(0, p)


def default_search_paths() -> list[Path]:
    here = Path(__file__).resolve()
    # repo layout: product/src/sieve/suites/loader.py → product/suites
    # installed wheel: suites are force-included at sieve/_suites
    builtin = here.parents[3] / "suites"
    installed = here.parents[1] / "_suites"
    out: list[Path] = []
    for p in _SEARCH + [builtin, installed, Path.cwd() / "suites"]:
        if p not in out:
            out.append(p)
    return out


def _find(suite_ref: str) -> Path:
    suite_id, _, version = suite_ref.partition("@")
    for root in default_search_paths():
        base = root / suite_id
        if not base.is_dir():
            continue
        if version:
            cand = base / version
            if (cand / "suite.yaml").exists():
                return cand
            # allow major.minor → pick exact dirs starting with it
            matches = sorted(d for d in base.iterdir()
                             if d.is_dir() and d.name.startswith(version)
                             and (d / "suite.yaml").exists())
            if matches:
                return matches[-1]
        else:
            matches = sorted(d for d in base.iterdir()
                             if d.is_dir() and (d / "suite.yaml").exists())
            if matches:
                return matches[-1]
    raise FileNotFoundError(f"suite not found: {suite_ref}")


class LoadedSuite:
    def __init__(self, path: Path):
        self.path = path
        raw = (path / "suite.yaml").read_bytes()
        cfg = yaml.safe_load(raw)
        data_hashes = {}
        for f in sorted(path.iterdir()):
            if f.is_file() and f.name != "suite.yaml":
                data_hashes[f.name] = sha256_file(f)
        if (path / "claims").is_dir():
            for f in sorted((path / "claims").glob("*.yaml")):
                data_hashes[f"claims/{f.name}"] = sha256_file(f)
        body = sha256_bytes(raw + json.dumps(data_hashes, sort_keys=True).encode())
        self.manifest = TestSuiteManifest(
            suite_id=cfg["suite_id"], version=str(cfg["version"]),
            claim_types=list(cfg.get("claim_types", [])),
            reference=cfg.get("reference", {}),
            metrics=list(cfg.get("metrics", [])),
            baselines=list(cfg.get("baselines", [])),
            inference=cfg.get("inference", {}),
            suite_hash=body)
        self._cfg = cfg

    # ---- shipped data -----------------------------------------------------
    def reference_stats(self) -> dict:
        return json.loads((self.path / "reference_stats.json").read_text())

    def baseline_stats(self) -> dict:
        return json.loads((self.path / "baseline_stats.json").read_text())

    def claim(self, claim_id: str) -> ClaimSpec:
        p = self.path / "claims" / f"{claim_id}.yaml"
        if not p.exists():
            known = [f.stem for f in (self.path / "claims").glob("*.yaml")]
            raise FileNotFoundError(
                f"claim '{claim_id}' not in suite (known: {known})")
        return ClaimSpec.model_validate(yaml.safe_load(p.read_text()))


def load(suite_ref: str) -> LoadedSuite:
    return LoadedSuite(_find(suite_ref))


def list_suites() -> list[str]:
    out = []
    for root in default_search_paths():
        if not root.is_dir():
            continue
        for sid in sorted(root.iterdir()):
            if sid.is_dir():
                for v in sorted(sid.iterdir()):
                    if (v / "suite.yaml").exists():
                        out.append(f"{sid.name}@{v.name}")
    return sorted(set(out))
