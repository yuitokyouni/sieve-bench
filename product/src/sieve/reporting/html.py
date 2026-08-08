"""Local HTML report. Section order is fixed by spec §9.1:
claim → scope/reference → profile matrix → critical findings →
what was not tested → provenance. A global score never appears because
none exists anywhere in the system."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from sieve.core.branding import PRODUCT_NAME, REPORT_FOOTER
from sieve.core.enums import Status
from sieve.core.models import EvidenceBundle

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html"]))


def render_report(out_path: str | Path, bundle: EvidenceBundle,
                  baseline_context: dict[str, list[str]]) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl = _env.get_template("report.html.j2")
    not_tested = [d for d in bundle.profile.dimensions
                  if d.status is Status.NOT_TESTED]
    insufficient = [d for d in bundle.profile.dimensions
                    if d.status is Status.INSUFFICIENT]
    out_path.write_text(tpl.render(
        b=bundle, product=PRODUCT_NAME, footer=REPORT_FOOTER,
        baseline_context=baseline_context,
        not_tested=not_tested, insufficient=insufficient,
        Status=Status))
    return out_path
