"""Closed vocabularies. Kept tiny on purpose: every addition is a schema change."""

from enum import Enum


class Status(str, Enum):
    """Result of one test, or of one evidence dimension.

    There is deliberately no ordering and no numeric mapping: statuses do not
    average. ``NOT_TESTED`` and ``INSUFFICIENT`` are first-class outcomes, not
    error states (spec §2.2, §16).
    """

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    NOT_TESTED = "NOT_TESTED"
    INSUFFICIENT = "INSUFFICIENT"


class Prespecification(str, Enum):
    """Whether a diagnostic existed before the evidence it is judging.

    ``POST_HOC`` is not a defect — it is disclosure. A metric added after
    observing a failure is a discovered failure mode, not confirmatory
    evidence, and the report must say so (spec §5.5).
    """

    PRE_SPECIFIED = "PRE_SPECIFIED"
    POST_HOC = "POST_HOC"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# The ten evidence dimensions of the financial validation profile (spec §4.8).
DIMENSIONS = (
    "marginal_distribution",
    "tail_behavior",
    "return_dependence",
    "volatility_dynamics",
    "leverage_asymmetry",
    "multiscale_behavior",
    "drift_nonstationarity",
    "regime_response",
    "intervention_validity",
    "reproducibility_provenance",
)
