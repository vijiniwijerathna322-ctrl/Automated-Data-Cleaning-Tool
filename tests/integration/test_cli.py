import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from datacleaner.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_no_args_shows_help() -> None:
    assert "clean" in runner.invoke(app, []).output


def test_clean_end_to_end(tmp_path: Path, sample_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "clean",
            str(sample_path),
            "-o",
            str(out),
            "--target",
            "Exam_Score",
            "--feature-source",
            "Exam_Score",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Duplicates removed: 1" in result.stdout

    cleaned = pd.read_csv(out / "cleaned.csv")
    ml = pd.read_csv(out / "ml_ready.csv")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))

    assert len(cleaned) == 18
    assert "Student_ID" not in ml.columns
    assert not any(c.startswith("Exam_Score_Category") for c in ml.columns)
    assert report["missing_fixed"] == 4


@pytest.mark.parametrize("fmt", ["xlsx", "parquet"])
def test_clean_other_output_formats(tmp_path: Path, sample_path: Path, fmt: str) -> None:
    out = tmp_path / "o"
    result = runner.invoke(app, ["clean", str(sample_path), "-o", str(out), "-f", fmt])
    assert result.exit_code == 0, result.output
    assert (out / f"cleaned.{fmt}").is_file()


def test_clean_with_config_file(tmp_path: Path, sample_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("handle_missing: false\n", encoding="utf-8")
    out = tmp_path / "o"

    result = runner.invoke(app, ["clean", str(sample_path), "-o", str(out), "-c", str(cfg)])
    assert result.exit_code == 0, result.output
    assert pd.read_csv(out / "cleaned.csv").isna().sum().sum() > 0


def test_clean_prints_warnings(tmp_path: Path) -> None:
    src = tmp_path / "t.csv"
    src.write_text("t\na\nb\nc\nd\n", encoding="utf-8")
    result = runner.invoke(
        app, ["clean", str(src), "-o", str(tmp_path / "o"), "--feature-source", "t"]
    )
    assert result.exit_code == 0
    assert "not numeric" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["clean", "nope.csv"],
        ["clean", "{sample}", "--target", "nope"],
        ["clean", "{sample}", "-c", "missing.yaml"],
        ["profile", "nope.csv"],
    ],
)
def test_errors_exit_nonzero(args: list[str], sample_path: Path, tmp_path: Path) -> None:
    argv = [a.replace("{sample}", str(sample_path)) for a in args]

    if argv[0] == "clean":
        argv += ["-o", str(tmp_path / "o")]

    result = runner.invoke(app, argv)
    assert result.exit_code == 1
    assert "error:" in result.output


def test_profile(sample_path: Path) -> None:
    result = runner.invoke(app, ["profile", str(sample_path)])
    assert result.exit_code == 0
    assert "19 rows x 7 columns" in result.stdout
    assert "Student_ID" in result.stdout


def test_init_config_roundtrip(tmp_path: Path, sample_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    assert runner.invoke(app, ["init-config", str(cfg)]).exit_code == 0
    assert runner.invoke(app, ["init-config", str(cfg)]).exit_code == 1
    assert runner.invoke(app, ["init-config", str(cfg), "--force"]).exit_code == 0

    out = tmp_path / "o"
    result = runner.invoke(app, ["clean", str(sample_path), "-o", str(out), "-c", str(cfg)])
    assert result.exit_code == 0, result.output


def test_verbose_flag(tmp_path: Path, sample_path: Path) -> None:
    result = runner.invoke(app, ["-v", "clean", str(sample_path), "-o", str(tmp_path / "o")])
    assert result.exit_code == 0
