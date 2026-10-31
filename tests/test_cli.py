"""Tests for robustness.cli.

The real YOLO model and COCO128 are replaced with the fake detector and the
synthetic samples by patching ``cli.make_detector`` and ``cli.load_samples``.
"""

import json
import subprocess
import sys

import pytest

from robustness import cli
from robustness.perturbations import PERTURBATIONS


@pytest.fixture
def fake_backend(monkeypatch, synthetic_samples, fake_detector):
    """Make the CLI use the fake detector and synthetic samples.

    Returns the list of (weights, confidence) the CLI asked a detector for.
    """
    created = []

    def make_detector(weights, confidence):
        created.append((weights, confidence))
        return fake_detector

    def load_samples(data_dir, limit):
        return synthetic_samples[:limit]

    monkeypatch.setattr(cli, "make_detector", make_detector)
    monkeypatch.setattr(cli, "load_samples", load_samples)
    return created


@pytest.fixture
def results_file(fake_backend, tmp_path):
    """A results file produced by `robustness run` with a small grid."""
    path = tmp_path / "results.json"
    code = cli.main(
        [
            "run",
            "--perturbations",
            "gaussian_blur,brightness_decrease",
            "--severities",
            "1,3",
            "--output",
            str(path),
            "--no-progress",
        ]
    )
    assert code == 0
    return path


# --- Argument parsing ----------------------------------------------------------


def test_run_defaults():
    args = cli.build_parser().parse_args(["run"])
    assert args.perturbations == list(PERTURBATIONS)
    assert args.severities == [1, 2, 3, 4, 5]
    assert args.iou_threshold == 0.5
    assert args.confidence == 0.25
    assert args.limit is None
    assert args.seed == 0


def test_check_defaults():
    args = cli.build_parser().parse_args(["check"])
    assert args.threshold == 0.05
    assert str(args.baseline).replace("\\", "/") == "baselines/baseline.json"


@pytest.mark.parametrize(
    ("argv", "attribute", "expected"),
    [
        (["run", "--perturbations", "all"], "perturbations", list(PERTURBATIONS)),
        (
            ["run", "--perturbations", "glare, motion_blur"],
            "perturbations",
            ["glare", "motion_blur"],
        ),
        (["run", "--severities", "5,1"], "severities", [5, 1]),
        (["run", "--limit", "10"], "limit", 10),
        (["run", "--iou-threshold", "0.75"], "iou_threshold", 0.75),
        (["check", "--threshold", "0"], "threshold", 0.0),
        (["report", "--examples", "0"], "examples", 0),
    ],
)
def test_options_are_parsed(argv, attribute, expected):
    assert getattr(cli.build_parser().parse_args(argv), attribute) == expected


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "required"),
        (["frobnicate"], "invalid choice"),
        (["run", "--perturbations", "fog"], "unknown perturbation"),
        (["run", "--perturbations", ","], "unknown perturbation"),
        (["run", "--severities", "0"], "between 1 and 5"),
        (["run", "--severities", "1,6"], "between 1 and 5"),
        (["run", "--severities", "high"], "must be an integer"),
        (["run", "--severities", ","], "at least one severity"),
        (["run", "--iou-threshold", "0"], "must be in (0.0, 1.0]"),
        (["run", "--confidence", "abc"], "expected a number"),
        (["run", "--limit", "0"], "must be >= 1"),
        (["run", "--limit", "ten"], "expected an integer"),
        (["check", "--threshold", "1.5"], "must be in [0.0, 1.0]"),
        (["report", "--examples", "-1"], "must be >= 0"),
    ],
)
def test_invalid_arguments_exit_with_code_2(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- run -----------------------------------------------------------------------


def test_run_writes_results(fake_backend, tmp_path, capsys):
    path = tmp_path / "results" / "results.json"
    argv = ["run", "--perturbations", "gaussian_blur,brightness_decrease", "--severities", "1,3"]
    assert cli.main([*argv, "--output", str(path), "--no-progress"]) == 0

    data = json.loads(path.read_text())
    assert data["metadata"]["dataset"] == "coco128"
    assert data["metadata"]["severities"] == [1, 3]
    assert data["metadata"]["confidence_threshold"] == 0.25
    assert set(data["results"]) == {"clean", "gaussian_blur", "brightness_decrease"}
    assert fake_backend == [("yolov8n.pt", 0.25)]  # detector created once
    output = capsys.readouterr().out
    assert "Saved results to" in output
    assert "Worst condition: brightness_decrease/3" in output


def test_run_limit_and_options(fake_backend, tmp_path):
    path = tmp_path / "r.json"
    argv = ["run", "--limit", "2", "--seed", "9", "--perturbations", "glare", "--severities", "2"]
    argv += ["--confidence", "0.4", "--output", str(path), "--no-progress"]
    assert cli.main(argv) == 0
    meta = json.loads(path.read_text())["metadata"]
    assert meta["num_images"] == 2
    assert meta["seed"] == 9
    assert fake_backend == [("yolov8n.pt", 0.4)]


# --- report --------------------------------------------------------------------


def test_report_without_examples(results_file, tmp_path):
    out = tmp_path / "docs"
    assert (
        cli.main(
            ["report", "--results", str(results_file), "--output-dir", str(out), "--examples", "0"]
        )
        == 0
    )
    assert (out / "results.md").exists()
    assert (out / "robustness_curves.png").exists()
    assert not (out / "examples").exists()


def test_report_with_examples(results_file, tmp_path):
    out = tmp_path / "docs"
    argv = ["report", "--results", str(results_file), "--output-dir", str(out), "--examples", "2"]
    assert cli.main(argv) == 0
    assert len(list((out / "examples").glob("example_*.png"))) == 2


def test_report_missing_results_file_returns_2(tmp_path, capsys):
    assert cli.main(["report", "--results", str(tmp_path / "missing.json")]) == 2
    assert "error: results file not found" in capsys.readouterr().err


# --- baseline ------------------------------------------------------------------


def test_baseline_copies_results(results_file, tmp_path, capsys):
    baseline = tmp_path / "baselines" / "baseline.json"
    argv = ["baseline", "--results", str(results_file), "--baseline", str(baseline)]
    assert cli.main(argv) == 0
    assert baseline.read_text() == results_file.read_text()
    assert "Saved baseline" in capsys.readouterr().out

    assert cli.main(argv) == 0
    assert "Replaced baseline" in capsys.readouterr().out


def test_baseline_rejects_invalid_results_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    baseline = tmp_path / "baseline.json"
    assert cli.main(["baseline", "--results", str(bad), "--baseline", str(baseline)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not baseline.exists()  # a broken file never becomes the baseline


# --- check ---------------------------------------------------------------------


def test_check_passes_against_own_baseline(results_file, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    cli.main(["baseline", "--results", str(results_file), "--baseline", str(baseline)])
    assert cli.main(["check", "--results", str(results_file), "--baseline", str(baseline)]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


def test_check_fails_on_regression(results_file, tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    cli.main(["baseline", "--results", str(results_file), "--baseline", str(baseline)])

    # Simulate a worse model: clean recall drops from 1.0 to 0.6.
    data = json.loads(results_file.read_text())
    data["results"]["clean"]["0"]["overall"]["recall"] = 0.6
    worse = tmp_path / "worse.json"
    worse.write_text(json.dumps(data))

    argv = ["check", "--results", str(worse), "--baseline", str(baseline)]
    assert cli.main(argv) == 1
    assert "FAIL clean/0" in capsys.readouterr().out
    assert cli.main([*argv, "--threshold", "0.5"]) == 0  # within a looser threshold


def test_check_missing_baseline_returns_2(results_file, tmp_path, capsys):
    argv = ["check", "--results", str(results_file), "--baseline", str(tmp_path / "none.json")]
    assert cli.main(argv) == 2
    assert "error: results file not found" in capsys.readouterr().err


# --- Entry point ---------------------------------------------------------------


def test_python_dash_m_entry_point():
    completed = subprocess.run(
        [sys.executable, "-m", "robustness", "--help"], capture_output=True, text=True
    )
    assert completed.returncode == 0
    for command in ("run", "report", "baseline", "check"):
        assert command in completed.stdout
