"""Tests for robustness.runner, using the fake detector from conftest.py."""

import json
import sys
import types
from datetime import datetime

import numpy as np
import pytest

from robustness.metrics import Box
from robustness.runner import (
    CLEAN,
    YoloDetector,
    boxes_from_arrays,
    image_rng,
    load_results,
    run_evaluation,
    save_results,
)


def evaluate(detector, samples, **kwargs):
    """run_evaluation with the progress bar off and a small default setup."""
    kwargs.setdefault("perturbations", ["brightness_decrease", "gaussian_blur"])
    kwargs.setdefault("severities", [1, 3, 5])
    return run_evaluation(detector, samples, show_progress=False, **kwargs)


def overall(results, perturbation, severity):
    return results["results"][perturbation][str(severity)]["overall"]


# --- End-to-end with the fake detector -----------------------------------------


def test_results_contain_every_condition(fake_detector, synthetic_samples):
    results = evaluate(fake_detector, synthetic_samples)
    assert set(results["results"]) == {CLEAN, "brightness_decrease", "gaussian_blur"}
    assert list(results["results"][CLEAN]) == ["0"]
    assert list(results["results"]["gaussian_blur"]) == ["1", "3", "5"]
    condition = results["results"]["gaussian_blur"]["3"]
    assert set(condition) == {"overall", "per_class"}
    assert set(condition["overall"]) == {"tp", "fp", "fn", "precision", "recall", "f1"}


def test_clean_images_are_detected_perfectly(fake_detector, synthetic_samples):
    # Sanity check of the test setup itself: the fake finds all 5 rectangles exactly.
    clean = overall(evaluate(fake_detector, synthetic_samples), CLEAN, 0)
    assert (clean["tp"], clean["fp"], clean["fn"]) == (5, 0, 0)
    assert clean["recall"] == 1.0


def test_darkening_reduces_recall(fake_detector, synthetic_samples):
    results = evaluate(fake_detector, synthetic_samples)
    # Objects are 230 bright; the fake's threshold is 128.
    # Severity 1 (x0.8 = 184) is still detected, severity 3 (x0.45 = 103) is not.
    assert overall(results, "brightness_decrease", 1)["recall"] == 1.0
    assert overall(results, "brightness_decrease", 3)["recall"] == 0.0
    assert overall(results, "brightness_decrease", 3)["fn"] == 5


def test_detector_called_once_per_image_and_condition(fake_detector, synthetic_samples):
    evaluate(fake_detector, synthetic_samples, perturbations=["glare", "motion_blur"])
    # 4 images x (1 clean + 2 perturbations x 3 severities)
    assert fake_detector.calls == 4 * (1 + 2 * 3)


def test_iou_threshold_is_used(synthetic_samples):
    class ShiftedDetector:
        """Returns every true box shifted 5 px right: IoU 0.6 for the first object."""

        name = "shifted"

        def __init__(self, samples):
            self.boxes = {id(s.image): s.ground_truth for s in samples}

        def predict(self, image):
            return [Box(b.x1 + 5, b.y1, b.x2 + 5, b.y2, b.class_id) for b in self.boxes[id(image)]]

    samples = synthetic_samples[:1]
    loose = evaluate(ShiftedDetector(samples), samples, perturbations=[], iou_threshold=0.5)
    strict = evaluate(ShiftedDetector(samples), samples, perturbations=[], iou_threshold=0.7)
    # Object 1: 20x16 box, IoU (15*16) / (2*320 - 240) = 0.6; object 2: 20x26, IoU ~0.6
    assert overall(loose, CLEAN, 0)["tp"] == 2
    assert overall(strict, CLEAN, 0)["tp"] == 0


# --- Reproducibility -----------------------------------------------------------


def test_same_seed_gives_identical_results(fake_detector, synthetic_samples):
    kwargs = {"perturbations": ["gaussian_noise", "glare"], "seed": 7}
    first = evaluate(fake_detector, synthetic_samples, **kwargs)
    second = evaluate(fake_detector, synthetic_samples, **kwargs)
    assert first["results"] == second["results"]


def test_results_do_not_depend_on_which_other_perturbations_run(fake_detector, synthetic_samples):
    alone = evaluate(fake_detector, synthetic_samples, perturbations=["glare"])
    together = evaluate(fake_detector, synthetic_samples, perturbations=["gaussian_noise", "glare"])
    assert alone["results"]["glare"] == together["results"]["glare"]


def test_image_rng_is_stable_and_independent():
    def draw(*args):
        return image_rng(*args).integers(0, 1_000_000, size=3).tolist()

    assert draw(0, "glare", 3) == draw(0, "glare", 3)
    assert draw(0, "glare", 3) != draw(1, "glare", 3)  # seed matters
    assert draw(0, "glare", 3) != draw(0, "glare", 4)  # each image gets its own noise
    assert draw(0, "glare", 3) != draw(0, "gaussian_noise", 3)


# --- Metadata ------------------------------------------------------------------


def test_metadata(fake_detector, synthetic_samples):
    results = evaluate(
        fake_detector,
        synthetic_samples,
        severities=[3, 1, 3],
        iou_threshold=0.6,
        seed=42,
        extra_metadata={"dataset": "synthetic"},
    )
    meta = results["metadata"]
    assert meta["model"] == "fake-threshold"
    assert meta["num_images"] == 4
    assert meta["perturbations"] == ["brightness_decrease", "gaussian_blur"]
    assert meta["severities"] == [1, 3]  # sorted, duplicates removed
    assert meta["iou_threshold"] == 0.6
    assert meta["seed"] == 42
    assert meta["dataset"] == "synthetic"
    assert meta["duration_seconds"] >= 0
    assert datetime.fromisoformat(meta["created_at"]).tzinfo is not None


def test_results_are_json_serializable(fake_detector, synthetic_samples):
    results = evaluate(fake_detector, synthetic_samples)
    assert json.loads(json.dumps(results)) == results


def test_progress_bar_is_shown(fake_detector, synthetic_samples, capsys):
    run_evaluation(fake_detector, synthetic_samples, perturbations=[], show_progress=True)
    assert "Evaluating" in capsys.readouterr().err


# --- Validation (fails before any inference) -----------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"perturbations": ["fog"]}, "unknown perturbation"),
        ({"severities": [0]}, "severities must be between 1 and 5"),
        ({"severities": [2, 6]}, "severities must be between 1 and 5"),
    ],
    ids=["unknown-perturbation", "severity-0", "severity-6"],
)
def test_invalid_settings_fail_fast(fake_detector, synthetic_samples, kwargs, message):
    with pytest.raises(ValueError, match=message):
        evaluate(fake_detector, synthetic_samples, **kwargs)
    assert fake_detector.calls == 0  # no time wasted on inference


def test_no_samples_raises(fake_detector):
    with pytest.raises(ValueError, match="no samples"):
        evaluate(fake_detector, [])


# --- Saving and loading --------------------------------------------------------


def test_save_and_load_round_trip(fake_detector, synthetic_samples, tmp_path):
    results = evaluate(fake_detector, synthetic_samples)
    path = save_results(results, tmp_path / "nested" / "dir" / "results.json")
    assert path.exists()
    assert load_results(path) == results


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="results file not found"):
        load_results(tmp_path / "missing.json")


@pytest.mark.parametrize("content", ['{"metadata": {}}', "[1, 2, 3]"], ids=["no-results", "list"])
def test_load_non_results_file_raises(tmp_path, content):
    path = tmp_path / "other.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="not a results file"):
        load_results(path)


# --- YOLO adapter (without loading the real model) ------------------------------


def test_boxes_from_arrays():
    boxes = boxes_from_arrays(
        np.array([[1.5, 2.0, 10.0, 20.0], [0, 0, 5, 5]], dtype=np.float32),
        np.array([3.0, 0.0], dtype=np.float32),
        np.array([0.9, 0.4], dtype=np.float32),
    )
    assert boxes[0] == Box(1.5, 2.0, 10.0, 20.0, class_id=3, confidence=pytest.approx(0.9))
    assert isinstance(boxes[0].class_id, int)
    assert isinstance(boxes[0].x1, float)  # plain Python types, safe for JSON
    assert boxes[1].class_id == 0


def test_boxes_from_arrays_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        boxes_from_arrays(np.zeros((2, 4)), np.zeros(1), np.zeros(2))


class _FakeTensor:
    """Mimics the tiny part of a torch tensor the adapter uses: .cpu().numpy()."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_yolo_detector_adapter(monkeypatch):
    """YoloDetector converts Ultralytics output to Boxes and passes its settings through."""
    calls = {}

    class FakeYOLO:
        names = {0: "person", 1: "bicycle"}

        def __init__(self, weights):
            calls["weights"] = weights

        def predict(self, image, **kwargs):
            calls["predict_kwargs"] = kwargs
            boxes = types.SimpleNamespace(
                xyxy=_FakeTensor([[10, 20, 30, 40]]),
                cls=_FakeTensor([1]),
                conf=_FakeTensor([0.75]),
            )
            return [types.SimpleNamespace(boxes=boxes)]

    # Swap in a fake `ultralytics` module so no real model is downloaded or loaded.
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    detector = YoloDetector("weights/yolov8s.pt", confidence=0.4, device="cpu")
    boxes = detector.predict(np.zeros((50, 50, 3), dtype=np.uint8))

    assert detector.name == "yolov8s"
    assert detector.class_names == {0: "person", 1: "bicycle"}
    assert calls["weights"] == "weights/yolov8s.pt"
    assert calls["predict_kwargs"] == {"conf": 0.4, "device": "cpu", "verbose": False}
    assert boxes == [Box(10.0, 20.0, 30.0, 40.0, class_id=1, confidence=0.75)]
