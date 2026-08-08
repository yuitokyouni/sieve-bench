"""``sieve`` command-line interface.

Exit-code contract (spec §8: evaluation outcomes are not process errors):

  0  the command ran; statuses (PASS/FAIL/WARN/NOT_TESTED/INSUFFICIENT) are
     results, never exit codes
  2  invalid input (bad CSV, unknown claim, malformed manifest)
  3  missing dependency or missing suite/artifact on disk
  4  ``sieve verify`` found the bundle or its artifacts modified
  1  internal error (a bug in sieve; please report)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from sieve import __version__
from sieve.core.branding import PRODUCT_NAME, PRODUCT_TAGLINE

app = typer.Typer(
    name="sieve", add_completion=False, no_args_is_help=True,
    help=f"{PRODUCT_NAME} — {PRODUCT_TAGLINE}\n\n"
         "Runs entirely offline: nothing is uploaded, no network is used.")
suites_app = typer.Typer(no_args_is_help=True, help="Inspect installed suites.")
metrics_app = typer.Typer(no_args_is_help=True, help="Inspect registered metrics.")
baselines_app = typer.Typer(no_args_is_help=True, help="Inspect registered baselines.")
schemas_app = typer.Typer(no_args_is_help=True, help="Export JSON schemas.")
app.add_typer(suites_app, name="suites")
app.add_typer(metrics_app, name="metrics")
app.add_typer(baselines_app, name="baselines")
app.add_typer(schemas_app, name="schemas")

EXIT_INPUT, EXIT_MISSING, EXIT_TAMPER = 2, 3, 4


def _fail(code: int, msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


@app.callback(invoke_without_command=True)
def _root(version: bool = typer.Option(False, "--version",
                                       help="Print version and exit.")):
    if version:
        typer.echo(f"sieve {__version__}")
        raise typer.Exit(0)


@app.command()
def doctor():
    """Check the local environment; print what would block an evaluation."""
    problems = 0
    typer.echo(f"sieve {__version__} on python "
               f"{sys.version.split()[0]} ({sys.executable})")
    for mod in ("numpy", "scipy", "pydantic", "yaml", "jinja2", "polars",
                "typer"):
        try:
            m = __import__(mod)
            typer.echo(f"  dep {mod:<10} {getattr(m, '__version__', '?')}")
        except ImportError:
            typer.secho(f"  dep {mod:<10} MISSING", fg=typer.colors.RED)
            problems += 1
    from sieve.suites.loader import default_search_paths, list_suites
    for p in default_search_paths():
        typer.echo(f"  suite path {p}{'' if p.is_dir() else '  (absent)'}")
    found = list_suites()
    if found:
        for s in found:
            typer.echo(f"  suite {s}")
    else:
        typer.secho("  no suites found", fg=typer.colors.RED)
        problems += 1
    typer.echo("  network: not used by any command (evaluation is local)")
    if problems:
        _fail(EXIT_MISSING, f"{problems} problem(s) above")
    typer.echo("ok")


@app.command()
def test(
    input_path: Path = typer.Argument(..., help="returns.csv or a directory "
                                      "containing returns.csv (+ manifest.yaml)"),
    suite: str = typer.Option(..., "--suite", help="suite ref, e.g. "
                              "financial-daily@1.0"),
    claim: str = typer.Option(..., "--claim", help="claim id declared by the "
                              "suite, e.g. descriptive-market-dynamics"),
    out: Path = typer.Option(Path(".sieve/runs"), "--out",
                             help="root directory for run outputs"),
    seed: int = typer.Option(20260802, "--seed", help="master seed; the full "
                             "seed tree is recorded in the run manifest"),
):
    """Evaluate a simulated return series against a claim, offline.

    Writes manifest.json, observations.parquet, results.json, findings.json,
    evidence_bundle.json, bundle.sha256 and report/index.html into a new run
    directory. A model FAIL is a result, not an error: exit code stays 0.
    """
    from sieve.adapters.csv import InputError

    try:
        from sieve.evaluation.runner import run_test
        run_dir = run_test(input_path, suite, claim, out_root=out,
                           master_seed=seed)
    except InputError as e:
        _fail(EXIT_INPUT, str(e))
    except FileNotFoundError as e:
        _fail(EXIT_MISSING, str(e))

    from sieve.provenance.bundle import load
    bundle = load(run_dir / "evidence_bundle.json")
    typer.echo(f"run {bundle.run_manifest.run_id} → {run_dir}")
    typer.echo(f"claim: {bundle.claim.claim_id} — {bundle.claim.statement}")
    for d in bundle.profile.dimensions:
        mark = {"PASS": typer.colors.GREEN, "FAIL": typer.colors.RED,
                "WARN": typer.colors.YELLOW}.get(d.status.value)
        typer.secho(f"  {d.status.value:<12} {d.dimension}", fg=mark)
    typer.echo(f"findings: {len(bundle.findings)}  "
               f"bundle {bundle.bundle_hash[:16]}…")
    typer.echo(f"report: {run_dir / 'report' / 'index.html'}")
    typer.echo("statuses are per-dimension evidence; no aggregate score exists")


@app.command()
def verify(run_dir: Path = typer.Argument(..., help="run directory or "
                                          "evidence_bundle.json")):
    """Recompute the bundle hash and every artifact hash; report mismatches.

    Exit 0 = intact, 4 = modified, 3 = missing.
    """
    if not run_dir.exists():
        _fail(EXIT_MISSING, f"not found: {run_dir}")
    from sieve.provenance.bundle import verify as _verify
    problems = _verify(run_dir)
    if not problems:
        typer.secho("intact: bundle hash and all artifact hashes match",
                    fg=typer.colors.GREEN)
        return
    for p in problems:
        typer.secho(f"  {p}", fg=typer.colors.RED)
    _fail(EXIT_TAMPER, f"{len(problems)} problem(s)")


@app.command()
def report(run_dir: Path = typer.Argument(..., help="existing run directory")):
    """Re-render report/index.html from the stored evidence bundle."""
    bundle_path = run_dir / "evidence_bundle.json"
    if not bundle_path.exists():
        _fail(EXIT_MISSING, f"no evidence_bundle.json in {run_dir}")
    from sieve.provenance.bundle import load
    from sieve.reporting.html import render_report
    bundle = load(bundle_path)
    ctx_path = run_dir / "artifacts" / "baseline_context.json"
    ctx = json.loads(ctx_path.read_text()) if ctx_path.exists() else {}
    out = render_report(run_dir / "report" / "index.html", bundle, ctx)
    typer.echo(str(out))


@suites_app.command("list")
def suites_list():
    """List installed suites (id@version) and where they were found."""
    from sieve.suites.loader import list_suites
    for s in list_suites():
        typer.echo(s)


@suites_app.command("show")
def suites_show(suite: str = typer.Argument(..., help="e.g. financial-daily@1.0")):
    """Print a suite's manifest, claims and hash."""
    from sieve.suites.loader import load
    try:
        s = load(suite)
    except FileNotFoundError as e:
        _fail(EXIT_MISSING, str(e))
    typer.echo(json.dumps(json.loads(s.manifest.model_dump_json()), indent=2))
    claims = sorted(p.stem for p in (s.path / "claims").glob("*.yaml"))
    typer.echo("claims: " + ", ".join(claims))


@metrics_app.command("list")
def metrics_list():
    """List registered metrics with dimension and pre-specification."""
    from sieve.metrics.registry import _ENTRIES
    for mid, (_, spec, dim) in _ENTRIES.items():
        typer.echo(f"{mid}@{spec.version:<3} {dim:<24} "
                   f"{spec.prespecification.value:<14} {spec.display_name}")


@metrics_app.command("show")
def metrics_show(metric: str = typer.Argument(...)):
    """Print a metric's full spec, including known blind spots."""
    from sieve.metrics.registry import resolve
    try:
        _, spec, dim = resolve(metric)
    except KeyError as e:
        _fail(EXIT_MISSING, str(e))
    body = json.loads(spec.model_dump_json())
    body["dimension"] = dim
    typer.echo(json.dumps(body, indent=2))


@baselines_app.command("list")
def baselines_list():
    """List baselines with declared present/absent mechanisms."""
    from sieve.baselines.registry import all_specs
    for spec in all_specs():
        typer.echo(f"{spec.baseline_id}@{spec.version}: {spec.description}")
        typer.echo(f"    present: {', '.join(spec.mechanisms_present) or '—'}")
        typer.echo(f"    absent:  {', '.join(spec.mechanisms_absent) or '—'}")


@schemas_app.command("export")
def schemas_export(out: Path = typer.Option(Path("schemas"), "--out")):
    """Write the JSON Schema of every durable artifact to *.schema.json."""
    from sieve.core import models
    out.mkdir(parents=True, exist_ok=True)
    for name in ("EvidenceBundle", "ModelManifest", "DatasetManifest",
                 "ClaimSpec", "MetricSpec", "BaselineSpec", "TestResult",
                 "FailureFinding", "ValidationProfile", "TestSuiteManifest",
                 "RunManifest"):
        cls = getattr(models, name)
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(cls.model_json_schema(), indent=2,
                                   sort_keys=True) + "\n")
        typer.echo(str(path))


def main():
    app()


if __name__ == "__main__":
    main()
