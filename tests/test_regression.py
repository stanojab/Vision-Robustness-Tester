"""Tests for robustness.regression."""

import json

import pytest

from robustness.regression import check_regression, compare_results


def status_of(report, condition, metric):
    return next(c.status for c in report.checks if (c.condition, c.metric) == (condition, metric))


# --- Threshold logic -----------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "current", "threshold", "expected"),
    [
        (0.80, 0.74, 0.05, "regression"),  # dropped 0.06 > 0.05
        (0.80, 0.75, 0.05, "pass"),  # dropped exactly 0.05: allowed (despite float error)
        (0.80, 0.76, 0.05, "pass"),  # small drop within tolerance
        (0.80, 0.80, 0.05, "pass"),  # unchanged
        (0.80, 0.95, 0.05, "pass"),  # improvement is never a failure
        (0.50, 0.49, 0.0, "regression"),  # threshold 0: any drop fails
        (0.50, 0.50, 0.0, "pass"),
        (0.90, 0.70, 0.25, "pass"),  # custom, looser threshold
    ],
)
def test_threshold(make_results, baseline, current, threshold, expected):
    report = compare_results(
        make_results({"clean/0": {"recall": current}}),
        make_results({"clean/0": {"recall": baseline}}),
        threshold=threshold,
    )
    assert status_of(report, "clean/0", "recall") == expected
    assert report.passed is (expected == "pass")


def test_float_rounding_does_not_cause_false_regression():
    # The reason for the epsilon in the comparison: in floating point,
    # 0.80 - 0.75 is slightly MORE than 0.05.
    assert 0.80 - 0.75 > 0.05


@pytest.mark.parametrize("metric", ["precision", "recall", "f1"])
def test_every_tracked_metric_is_checked(make_results, metric):
    report = compare_results(
        make_results({"glare/3": {metric: 0.5}}),
        make_results({"glare/3": {metric: 0.9}}),
    )
    assert not report.passed
    assert [(c.condition, c.metric) for c in report.failures] == [("glare/3", metric)]


def test_identical_results_pass(fake_results):
    report = compare_results(fake_results, fake_results)
    assert report.passed
    assert report.failures == []
    assert {c.status for c in report.checks} == {"pass"}
    # 1 clean + 3 perturbations x 3 severities = 10 conditions, 3 metrics each
    assert len(report.checks) == 30


def test_per_class_changes_are_not_tracked(make_results):
    baseline = make_results({"clean/0": {}})
    current = make_results({"clean/0": {}})
    baseline["results"]["clean"]["0"]["per_class"] = {"3": {"recall": 1.0}}
    current["results"]["clean"]["0"]["per_class"] = {"3": {"recall": 0.0}}
    assert compare_results(current, baseline).passed


def test_negative_threshold_raises(make_results):
    results = make_results({"clean/0": {}})
    with pytest.raises(ValueError, match="threshold"):
        compare_results(results, results, threshold=-0.1)


# --- Missing and extra keys ----------------------------------------------------


def test_condition_missing_from_current_fails(make_results):
    baseline = make_results({"clean/0": {}, "glare/5": {}})
    current = make_results({"clean/0": {}})
    report = compare_results(current, baseline)
    assert not report.passed
    assert {c.condition for c in report.failures} == {"glare/5"}
    assert {c.status for c in report.failures} == {"missing"}


def test_metric_missing_from_current_fails(make_results):
    baseline = make_results({"clean/0": {}})
    current = make_results({"clean/0": {}})
    del current["results"]["clean"]["0"]["overall"]["f1"]
    report = compare_results(current, baseline)
    assert status_of(report, "clean/0", "f1") == "missing"
    assert not report.passed


def test_new_condition_is_reported_but_does_not_fail(make_results):
    baseline = make_results({"clean/0": {}})
    current = make_results({"clean/0": {}, "motion_blur/2": {"recall": 0.1}})
    report = compare_results(current, baseline)
    assert report.passed
    assert status_of(report, "motion_blur/2", "recall") == "new"
    assert "New conditions not in the baseline" in report.summary()


def test_empty_baseline_passes_with_warning(make_results):
    report = compare_results(make_results({"clean/0": {}}), {"results": {}})
    assert report.passed
    assert any("no metrics" in warning for warning in report.warnings)


def test_condition_without_overall_section_is_missing(make_results):
    baseline = make_results({"clean/0": {}})
    current = {"results": {"clean": {"0": {}}}}  # malformed: no "overall"
    report = compare_results(current, baseline)
    assert {c.status for c in report.checks} == {"missing"}


# --- Settings mismatch ---------------------------------------------------------


def test_different_settings_produce_warning_but_still_compare(make_results):
    baseline = make_results({"clean/0": {}}, num_images=128)
    current = make_results({"clean/0": {}}, num_images=5)
    report = compare_results(current, baseline)
    assert report.passed
    assert len(report.warnings) == 1
    assert "num_images" in report.warnings[0]
    assert "WARNING: setting 'num_images' differs" in report.summary()


def test_matching_settings_produce_no_warning(make_results):
    results = make_results({"clean/0": {}}, iou_threshold=0.5)
    assert compare_results(results, results).warnings == []


# --- Summary text --------------------------------------------------------------


def test_summary_lists_regressions_and_result(make_results):
    report = compare_results(
        make_results({"clean/0": {}, "gaussian_blur/3": {"recall": 0.74}}),
        make_results({"clean/0": {}, "gaussian_blur/3": {"recall": 0.80}, "glare/1": {}}),
    )
    summary = report.summary()
    assert "5 passed, 1 regressed, 3 missing, 0 new" in summary
    assert "FAIL gaussian_blur/3" in summary
    assert "0.800 -> 0.740 (-0.060)" in summary
    assert "Missing from the current results" in summary
    assert summary.endswith("RESULT: FAIL")


def test_summary_mentions_improvements(make_results):
    report = compare_results(
        make_results({"clean/0": {"recall": 0.9}}), make_results({"clean/0": {"recall": 0.5}})
    )
    assert "1 metric(s) improved" in report.summary()
    assert report.summary().endswith("RESULT: PASS")


def test_delta_is_none_when_a_value_is_missing(make_results):
    report = compare_results(make_results({}), make_results({"clean/0": {}}))
    assert all(c.delta is None for c in report.checks)


# --- File-based check with exit codes ------------------------------------------


@pytest.fixture
def write_json(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path

    return write


def test_check_regression_exit_code_0_on_pass(make_results, write_json, capsys):
    results = write_json("results.json", make_results({"clean/0": {"recall": 0.8}}))
    baseline = write_json("baseline.json", make_results({"clean/0": {"recall": 0.8}}))
    assert check_regression(results, baseline) == 0
    assert "RESULT: PASS" in capsys.readouterr().out


def test_check_regression_exit_code_1_on_regression(make_results, write_json, capsys):
    results = write_json("results.json", make_results({"clean/0": {"recall": 0.5}}))
    baseline = write_json("baseline.json", make_results({"clean/0": {"recall": 0.8}}))
    assert check_regression(results, baseline) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out


def test_check_regression_custom_threshold(make_results, write_json):
    results = write_json("results.json", make_results({"clean/0": {"recall": 0.5}}))
    baseline = write_json("baseline.json", make_results({"clean/0": {"recall": 0.8}}))
    assert check_regression(results, baseline, threshold=0.3) == 0


def test_check_regression_missing_baseline_raises(make_results, write_json, tmp_path):
    results = write_json("results.json", make_results({"clean/0": {}}))
    with pytest.raises(FileNotFoundError, match="results file not found"):
        check_regression(results, tmp_path / "no_baseline.json")
