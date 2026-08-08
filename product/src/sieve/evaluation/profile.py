"""Fold test results into the ten-dimension ValidationProfile.

Rules (never an average):
  any FAIL → FAIL;  else any INSUFFICIENT → INSUFFICIENT;
  else any WARN → WARN;  else PASS.  No member tests → NOT_TESTED.
Reproducibility/provenance is scored from manifest completeness, not statistics.
"""

from __future__ import annotations

from sieve.core.enums import DIMENSIONS, Status
from sieve.core.models import (
    DimensionStatus,
    ModelManifest,
    TestResult,
    ValidationProfile,
)
from sieve.metrics import registry as metric_registry


def _fold(statuses: list[Status]) -> Status:
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.INSUFFICIENT in statuses:
        return Status.INSUFFICIENT
    if Status.WARN in statuses:
        return Status.WARN
    return Status.PASS


def build_profile(results: list[TestResult], metric_refs: list[str],
                  model: ModelManifest) -> ValidationProfile:
    by_dim: dict[str, list[TestResult]] = {}
    for mref in metric_refs:
        _, _, dim = metric_registry.resolve(mref)
        by_dim.setdefault(dim, [])
    for res in results:
        _, _, dim = metric_registry.resolve(res.metric_ref)
        by_dim.setdefault(dim, []).append(res)

    dims: list[DimensionStatus] = []
    for d in DIMENSIONS:
        if d == "reproducibility_provenance":
            missing = [f for f, v in (("git_commit", model.git_commit),
                                      ("code_uri", model.code_uri),
                                      ("model_version",
                                       None if model.model_version == "unversioned"
                                       else model.model_version))
                       if not v]
            dims.append(DimensionStatus(
                dimension=d,
                status=Status.PASS if not missing else Status.WARN,
                note=None if not missing
                else "input manifest missing: " + ", ".join(missing)))
            continue
        members = by_dim.get(d, [])
        if not members:
            dims.append(DimensionStatus(
                dimension=d, status=Status.NOT_TESTED,
                note="no metric in this suite version tests this dimension"))
            continue
        dims.append(DimensionStatus(
            dimension=d, status=_fold([m.status for m in members]),
            test_refs=[m.test_id for m in members]))
    return ValidationProfile(dimensions=dims)
