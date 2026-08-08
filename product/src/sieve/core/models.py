"""Versioned schemas for every durable artifact (spec §4).

Design rules, enforced by tests:

- No field anywhere aggregates evidence into a single model-wide score.
- Everything that reaches a report must be reconstructible from these objects.
- ``EvidenceBundle`` is the product artifact; its canonical serialization
  (``sieve.core.serialization``) is what gets hashed and verified.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from sieve.core.enums import DIMENSIONS, Prespecification, Severity, Status

SCHEMA_VERSION = "0.1.0"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransformSpec(_Model):
    name: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ModelManifest(_Model):
    schema_version: str = SCHEMA_VERSION
    model_id: str
    model_version: str
    display_name: str
    model_family: str | None = None
    adapter_id: str
    code_uri: str | None = None
    git_commit: str | None = None
    container_digest: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    parameters_hash: str = ""
    authors: list[str] = Field(default_factory=list)
    license: str | None = None
    notes: str | None = None


class DatasetManifest(_Model):
    schema_version: str = SCHEMA_VERSION
    dataset_id: str
    source_uri: str | None = None
    symbols: list[str] = Field(default_factory=list)
    start: datetime | None = None
    end: datetime | None = None
    frequency: str = "daily"
    timezone: str | None = None
    transforms: list[TransformSpec] = Field(default_factory=list)
    content_hash: str = ""
    license: str | None = None


class ClaimSpec(_Model):
    claim_id: str
    version: str
    statement: str
    use_case: str
    scope: dict[str, JsonValue] = Field(default_factory=dict)
    required_dimensions: list[str] = Field(default_factory=list)
    optional_dimensions: list[str] = Field(default_factory=list)
    decision_policy: str | None = None


class MetricSpec(_Model):
    metric_id: str
    version: str
    display_name: str
    function_path: str
    input_contract: str
    scale_invariant: bool
    intended_signal: str
    known_blind_spots: list[str] = Field(default_factory=list)
    prespecification: Prespecification = Prespecification.UNKNOWN
    references: list[str] = Field(default_factory=list)


class BaselineSpec(_Model):
    baseline_id: str
    version: str
    description: str
    mechanisms_present: list[str] = Field(default_factory=list)
    mechanisms_absent: list[str] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    calibration_ref: str | None = None


class TestResult(_Model):
    __test__ = False          # a schema, not a pytest collection target

    test_id: str
    metric_ref: str
    baseline_ref: str | None = None
    statistic_name: str
    statistic_value: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None
    effect_size: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_reference: int = 0
    n_simulation: int = 0
    status: Status
    prespecification: Prespecification = Prespecification.UNKNOWN
    caveats: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class FailureFinding(_Model):
    finding_id: str
    code: str
    severity: Severity
    title: str
    observation: str
    interpretation: str
    claim_impact: str
    evidence_refs: list[str] = Field(default_factory=list)
    questions_for_author: list[str] = Field(default_factory=list)


class DimensionStatus(_Model):
    dimension: str
    status: Status
    test_refs: list[str] = Field(default_factory=list)
    note: str | None = None


class ValidationProfile(_Model):
    """Ten dimensions, each with a status. Never averaged (spec §4.8)."""

    dimensions: list[DimensionStatus]

    def as_dict(self) -> dict[str, Status]:
        return {d.dimension: d.status for d in self.dimensions}

    @classmethod
    def not_tested(cls) -> "ValidationProfile":
        return cls(dimensions=[DimensionStatus(dimension=d, status=Status.NOT_TESTED)
                               for d in DIMENSIONS])


class TestSuiteManifest(_Model):
    __test__ = False          # a schema, not a pytest collection target

    suite_id: str
    version: str
    claim_types: list[str]
    reference: dict[str, JsonValue] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    inference: dict[str, JsonValue] = Field(default_factory=dict)
    suite_hash: str = ""


class ArtifactRef(_Model):
    path: str            # relative to the run directory
    sha256: str
    kind: str            # "table" | "report" | "figure" | "input" | "other"


class SeedNode(_Model):
    name: str
    entropy: int
    spawn_key: list[int] = Field(default_factory=list)


class RunManifest(_Model):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: datetime
    sieve_version: str
    command: str
    master_seed: int
    seed_tree: list[SeedNode] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    input_path: str
    input_hash: str


class AuthorResponse(_Model):
    author_name: str
    received_at: datetime
    response: str
    reproduction_confirmed: bool | None = None
    correction_refs: list[str] = Field(default_factory=list)


class EvidenceBundle(_Model):
    schema_version: str = SCHEMA_VERSION
    bundle_id: UUID
    created_at: datetime
    run_manifest: RunManifest
    model: ModelManifest
    dataset: DatasetManifest
    claim: ClaimSpec
    suite: TestSuiteManifest
    profile: ValidationProfile
    results: list[TestResult]
    findings: list[FailureFinding]
    limitations: list[str] = Field(default_factory=list)
    artifact_index: list[ArtifactRef] = Field(default_factory=list)
    author_responses: list[AuthorResponse] = Field(default_factory=list)
    bundle_hash: str = ""


# Fields excluded from the canonical hash: they vary between otherwise
# identical runs (spec §16 "except explicitly excluded timestamps/IDs").
# ``artifact_index`` is excluded because manifest.json and report/index.html
# legitimately contain the excluded run_id/created_at (and the report embeds
# bundle_hash itself); their file hashes would smuggle the volatile IDs back
# into the seal. The artifact index is instead covered by the sidecar hash of
# the written bundle file (see sieve.provenance.bundle).
HASH_EXCLUDED_PATHS = (
    ("bundle_hash",),
    ("bundle_id",),
    ("created_at",),
    ("run_manifest", "run_id"),
    ("run_manifest", "created_at"),
    ("artifact_index",),
)
