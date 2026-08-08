"""Tier-0 file adapter: ``returns.csv`` (+ optional ``manifest.yaml``).

Contract (spec §6.1): a CSV with columns ``timestamp,return``. Anything the
manifest does not state is recorded as absent — provenance gaps are reported,
never silently filled.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from sieve.core.hashing import sha256_file, sha256_params
from sieve.core.models import DatasetManifest, ModelManifest


class InputError(ValueError):
    """User-facing input problem (CLI exit class: invalid input)."""


def read_returns(path: str | Path) -> tuple[np.ndarray, list[str]]:
    path = Path(path)
    if not path.exists():
        raise InputError(f"input not found: {path}")
    ts, vals = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise InputError("empty CSV")
        cols = [c.strip().lower() for c in header]
        if "return" not in cols:
            raise InputError("returns.csv must have a 'return' column "
                             f"(got {header})")
        ri = cols.index("return")
        ti = cols.index("timestamp") if "timestamp" in cols else None
        for ln, row in enumerate(reader, start=2):
            if not row or all(not c.strip() for c in row):
                continue
            try:
                vals.append(float(row[ri]))
            except (ValueError, IndexError) as e:
                raise InputError(f"line {ln}: bad return value {row!r}") from e
            ts.append(row[ti] if ti is not None and ti < len(row) else "")
    r = np.asarray(vals, dtype=float)
    if len(r) < 50:
        raise InputError(f"only {len(r)} returns; need at least 50")
    if not np.isfinite(r).all():
        raise InputError("non-finite values in return column")
    return r, ts


def load_input(
        input_path: str | Path
) -> tuple[np.ndarray, ModelManifest, DatasetManifest]:
    input_path = Path(input_path)
    csv_path = (input_path / "returns.csv" if input_path.is_dir()
                else input_path)
    r, _ = read_returns(csv_path)

    meta: dict = {}
    mpath = (input_path if input_path.is_dir() else input_path.parent) / "manifest.yaml"
    if mpath.exists():
        meta = yaml.safe_load(mpath.read_text()) or {}

    params = dict(meta.get("parameters", {}))
    model = ModelManifest(
        model_id=str(meta.get("model_id", csv_path.stem)),
        model_version=str(meta.get("model_version", "unversioned")),
        display_name=str(meta.get("display_name", meta.get("model_id", csv_path.stem))),
        model_family=meta.get("model_family"),
        adapter_id="csv@1",
        code_uri=meta.get("code_uri"),
        git_commit=meta.get("git_commit"),
        parameters=params,
        parameters_hash=sha256_params(params),
        authors=list(meta.get("authors", [])),
        license=meta.get("license"),
        notes=meta.get("notes"))
    dataset = DatasetManifest(
        dataset_id=str(meta.get("dataset_id", f"input:{csv_path.name}")),
        source_uri=str(csv_path),
        frequency=str(meta.get("frequency", "daily")),
        content_hash=sha256_file(csv_path))
    return r, model, dataset
