"""Turn FAILed tests into findings a model author can act on.

Severity maps from the KS effect size only (never to a model-wide verdict):
KS >= 0.5 HIGH, >= 0.3 MEDIUM, else LOW. The author questions follow the
external-audit package convention (spec §19): concrete, no pitch.
"""

from __future__ import annotations

from sieve.core.enums import Severity, Status
from sieve.core.models import ClaimSpec, FailureFinding, TestResult
from sieve.metrics import registry as metric_registry


def _severity(ks: float | None) -> Severity:
    if ks is None:
        return Severity.INFO
    if ks >= 0.5:
        return Severity.HIGH
    if ks >= 0.3:
        return Severity.MEDIUM
    return Severity.LOW


def build_findings(results: list[TestResult],
                   baseline_context: dict[str, list[str]],
                   claim: ClaimSpec) -> list[FailureFinding]:
    out: list[FailureFinding] = []
    for res in results:
        if res.status is not Status.FAIL:
            continue
        mid = res.metric_ref.partition("@")[0]
        _, spec, dim = metric_registry.resolve(res.metric_ref)
        blind = baseline_context.get(mid, [])
        interp = (f"The simulated distribution of '{spec.display_name}' is "
                  f"separated from the empirical reference "
                  f"(KS={res.statistic_value:.2f}, adjusted "
                  f"p={res.adjusted_p_value:.4g} at the calibrated line).")
        if blind:
            interp += (" Note this metric also fails to separate the reference "
                       "from " + ", ".join(blind) + "; treat its pass/fail as "
                       "one dimension of evidence, not a verdict.")
        out.append(FailureFinding(
            finding_id=f"F-{mid}",
            code=f"DIVERGES_{mid.upper()}",
            severity=_severity(res.statistic_value),
            title=f"{spec.display_name} diverges from the empirical reference",
            observation=(f"Across {res.n_simulation} non-overlapping simulated "
                         f"windows vs {res.n_reference} reference windows, "
                         f"KS={res.statistic_value:.3f}."),
            interpretation=interp,
            claim_impact=(f"Weakens support for claim '{claim.claim_id}' on the "
                          f"dimension '{dim}'. Other dimensions are unaffected "
                          "by this finding."),
            evidence_refs=[res.test_id, "observations.parquet",
                           "artifacts/baseline_context.json"],
            questions_for_author=[
                "Did we extract the model output correctly (frequency, units, "
                "log vs simple returns)?",
                f"Is matching the empirical distribution of "
                f"'{spec.display_name}' within the intended use of the model?",
                "Is there a configuration of the model under which this "
                "dimension is expected to match, and should we evaluate that "
                "configuration instead?"]))
    return out
