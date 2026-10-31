"""Tests for robustness.report."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from robustness.dataset import Sample
from robustness.metrics import Box
from robustness.report import (
    BOX_COLOR_BGR,
    _series_style,
    detailed_table,
    draw_boxes,
    format_metric,
    markdown_report,
    perturbation_names,
    pivot_table,
    plot_robustness_curves,
    save_example_grid,
    save_markdown_report,
    select_examples,
)

# --- Formatting ----------------------------------------------------------------


def overall(tp, fp, fn, precision=0.0, recall=0.0, f1=0.0):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


@pytest.mark.parametrize(
    ("values", "metric", "expected"),
    [
        (overall(3, 1, 2, precision=0.75), "precision", "0.750"),
        (overall(0, 0, 4), "precision", "n/a"),  # no predictions
        (overall(0, 4, 0), "precision", "0.000"),  # predictions, all wrong: really 0
        (overall(0, 4, 0), "recall", "n/a"),  # no ground truth
        (overall(0, 0, 0), "f1", "n/a"),  # nothing at all
        (overall(0, 0, 4), "f1", "0.000"),
    ],
)
def test_format_metric(values, metric, expected):
    assert format_metric(values, metric) == expected


# --- Tables --------------------------------------------------------------------


def test_perturbation_names_follow_registry_order(make_results):
    # JSON order is "glare" first, but the registry puts gaussian_blur before glare.
    results = make_results({"clean/0": {}, "glare/1": {}, "gaussian_blur/1": {}})
    assert perturbation_names(results) == ["gaussian_blur", "glare"]


def test_pivot_table(make_results):
    results = make_results(
        {
            "clean/0": {"recall": 0.9},
            "gaussian_blur/1": {"recall": 0.8},
            "gaussian_blur/3": {"recall": 0.5},
            "glare/1": {"recall": 0.85},  # no severity 3 for glare
        }
    )
    lines = pivot_table(results, "recall").splitlines()
    assert lines[0] == "| Perturbation | 0 (clean) | 1 | 3 |"
    assert lines[1] == "| --- | ---: | ---: | ---: |"
    assert lines[2] == "| gaussian_blur | 0.900 | 0.800 | 0.500 |"
    assert lines[3] == "| glare | 0.900 | 0.850 | - |"


def test_detailed_table_has_one_row_per_condition(fake_results):
    lines = detailed_table(fake_results).splitlines()
    assert lines[0] == "| Perturbation | Severity | Precision | Recall | F1 | TP | FP | FN |"
    assert len(lines) == 2 + 1 + 3 * 3  # header + separator + clean + 3x3 conditions
    assert lines[2] == "| clean | 0 | 1.000 | 1.000 | 1.000 | 5 | 0 | 0 |"


def test_markdown_report_sections(fake_results):
    report = markdown_report(fake_results)
    for expected in [
        "# Robustness report",
        "- **Model:** fake-threshold",
        "- **Dataset:** synthetic",
        "- **Images:** 4",
        "## Recall by severity",
        "## F1 by severity",
        "## All metrics",
    ]:
        assert expected in report
    assert "Confidence threshold" not in report  # not in this run's metadata: omitted


def test_save_markdown_report(fake_results, tmp_path):
    path = save_markdown_report(fake_results, tmp_path / "docs" / "results.md")
    assert path.read_text(encoding="utf-8") == markdown_report(fake_results)


# --- Chart ---------------------------------------------------------------------


def test_plot_robustness_curves_writes_png(fake_results, tmp_path):
    path = plot_robustness_curves(fake_results, tmp_path / "docs" / "chart.png")
    image = cv2.imread(str(path))
    assert image is not None  # a real, readable PNG
    assert image.shape[0] > 100 and image.shape[1] > 100


def test_series_style_is_fixed_per_perturbation():
    # Colour follows the perturbation, not its position in a particular run.
    assert _series_style("glare") == _series_style("glare")
    assert _series_style("glare") != _series_style("gaussian_blur")


# --- Example images ------------------------------------------------------------


def test_draw_boxes_draws_on_a_copy():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    drawn = draw_boxes(image, [Box(20, 30, 60, 80, class_id=0, confidence=0.9)], {0: "person"})
    assert image.max() == 0  # input untouched
    assert tuple(drawn[55, 20]) == BOX_COLOR_BGR  # left edge of the box
    assert tuple(drawn[55, 40]) == (0, 0, 0)  # inside the box is not filled


def test_draw_boxes_handles_box_at_top_edge_and_unknown_class():
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    drawn = draw_boxes(image, [Box(0, 0, 40, 30, class_id=99, confidence=0.5)])
    assert drawn.shape == image.shape
    assert drawn.any()


def _sample(objects: int) -> Sample:
    boxes = [Box(0, 0, 1, 1, class_id=0)] * objects
    return Sample(Path(f"{objects}.jpg"), np.zeros((4, 4, 3), np.uint8), boxes)


def test_select_examples_prefers_busy_images():
    samples = [_sample(n) for n in (0, 5, 1, 3, 4)]
    assert select_examples(samples, 2) == [1, 3]  # first two with >= 3 objects


def test_select_examples_fills_up_with_other_images():
    samples = [_sample(n) for n in (0, 5, 1)]
    assert select_examples(samples, 3) == [1, 0, 2]
    assert select_examples(samples, 10) == [1, 0, 2]  # never more than available


def test_save_example_grid(fake_detector, synthetic_samples, tmp_path):
    path = save_example_grid(
        fake_detector,
        synthetic_samples[0],
        image_index=0,
        perturbations=["gaussian_blur", "glare", "brightness_decrease"],
        severity=3,
        path=tmp_path / "examples" / "grid.png",
        class_names={0: "rectangle"},
    )
    assert cv2.imread(str(path)) is not None
    assert fake_detector.calls == 4  # clean + 3 perturbations
