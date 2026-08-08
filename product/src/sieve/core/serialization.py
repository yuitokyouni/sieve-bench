"""Canonical serialization: one byte stream per logical bundle.

The hash is only meaningful if serialization is deterministic. Rules:

- JSON with sorted keys, no insignificant whitespace, UTF-8, ``\n`` line end;
- floats via ``repr`` round-trip (Python floats serialize shortest-repr, which
  is deterministic for a given value across platforms we target);
- volatile fields (``HASH_EXCLUDED_PATHS``) are *replaced by null* — not
  removed — before hashing, so the hashed shape is stable;
- NaN/Inf are forbidden in bundles (they are not JSON); statistics that can
  produce them must map to ``null`` + an INSUFFICIENT/WARN status upstream.
"""

from __future__ import annotations

import json
import math
from typing import Any

from sieve.core.models import HASH_EXCLUDED_PATHS, EvidenceBundle


def _reject_non_finite(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite float at {path}; encode as null upstream")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reject_non_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _reject_non_finite(v, f"{path}[{i}]")


def to_jsonable(bundle: EvidenceBundle) -> dict:
    return json.loads(bundle.model_dump_json())


def canonical_bytes(data: dict) -> bytes:
    _reject_non_finite(data)
    return (json.dumps(data, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def null_out_excluded(data: dict) -> dict:
    """Return a copy with volatile fields set to null (shape preserved)."""
    import copy

    out = copy.deepcopy(data)
    for path in HASH_EXCLUDED_PATHS:
        node = out
        for key in path[:-1]:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, dict) and path[-1] in node:
            node[path[-1]] = None
    return out


def hashable_bytes(bundle: EvidenceBundle) -> bytes:
    return canonical_bytes(null_out_excluded(to_jsonable(bundle)))
