"""CLI surface and the exit-code contract (spec §8): evaluation outcomes are
results; only broken input, missing pieces, tampering or bugs are errors."""

from typer.testing import CliRunner

from sieve.cli.app import app

runner = CliRunner()


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert "sieve 0.1.0" in res.output


def test_doctor_ok():
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    assert "financial-daily@1.0.0" in res.output
    assert "network: not used" in res.output


def test_suites_list():
    res = runner.invoke(app, ["suites", "list"])
    assert res.exit_code == 0
    assert "financial-daily@1.0.0" in res.output


def test_suites_show_includes_hash_and_claims():
    res = runner.invoke(app, ["suites", "show", "financial-daily@1.0"])
    assert res.exit_code == 0
    assert "suite_hash" in res.output
    assert "descriptive-market-dynamics" in res.output


def test_metrics_list_and_show():
    res = runner.invoke(app, ["metrics", "list"])
    assert res.exit_code == 0
    assert res.output.count("@1") == 8
    res = runner.invoke(app, ["metrics", "show", "leverage"])
    assert res.exit_code == 0
    assert "known_blind_spots" in res.output


def test_baselines_list_shows_mechanisms():
    res = runner.invoke(app, ["baselines", "list"])
    assert res.exit_code == 0
    assert "present:" in res.output and "absent:" in res.output
    assert "garch_t" in res.output


def test_missing_input_exits_2():
    res = runner.invoke(app, ["test", "/no/such/file.csv", "--suite",
                              "financial-daily@1.0", "--claim",
                              "descriptive-market-dynamics"])
    assert res.exit_code == 2


def test_unknown_suite_exits_3(tmp_path):
    csv = tmp_path / "returns.csv"
    csv.write_text("timestamp,return\n" +
                   "\n".join(f"2020-01-{i%28+1:02d},0.{i%7}" for i in range(60)))
    res = runner.invoke(app, ["test", str(csv), "--suite", "no-such@9",
                              "--claim", "c"])
    assert res.exit_code == 3


def test_unknown_claim_exits_3(tmp_path):
    csv = tmp_path / "returns.csv"
    csv.write_text("timestamp,return\n" +
                   "\n".join(f"2020-01-{i%28+1:02d},0.{i%7}" for i in range(60)))
    res = runner.invoke(app, ["test", str(csv), "--suite",
                              "financial-daily@1.0", "--claim", "no-such"])
    assert res.exit_code == 3


def test_malformed_csv_exits_2(tmp_path):
    csv = tmp_path / "returns.csv"
    csv.write_text("timestamp,return\n2020-01-01,not_a_number\n")
    res = runner.invoke(app, ["test", str(csv), "--suite",
                              "financial-daily@1.0", "--claim",
                              "descriptive-market-dynamics"])
    assert res.exit_code == 2


def test_verify_missing_dir_exits_3(tmp_path):
    res = runner.invoke(app, ["verify", str(tmp_path / "nope")])
    assert res.exit_code == 3


def test_schemas_export(tmp_path):
    res = runner.invoke(app, ["schemas", "export", "--out", str(tmp_path)])
    assert res.exit_code == 0
    names = {p.name for p in tmp_path.glob("*.schema.json")}
    assert "EvidenceBundle.schema.json" in names
    assert len(names) == 11
