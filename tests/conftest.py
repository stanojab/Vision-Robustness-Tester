"""Shared pytest fixtures.

All test data is synthetic and created in memory: tests never download the
dataset or the model, and never need a GPU.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from robustness.dataset import Sample
from robustness.metrics import Box
from robustness.runner import run_evaluation


def _synthetic_scene(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Build a deterministic 'natural-looking' uint8 image.

    A smooth gradient plus a few solid rectangles (edges, like object outlines)
    plus mild noise (texture). Mid-range values on average, so both darkening
    and brightening have room to change it.
    """
    rng = np.random.default_rng(seed)
    height, width = shape[:2]
    gradient = np.linspace(40, 200, width, dtype=np.float32)[np.newaxis, :].repeat(height, axis=0)
    image = gradient.copy()
    image[height // 4 : height // 2, width // 5 : width // 2] = 230.0
    image[height // 2 : 3 * height // 4, width // 2 : 4 * width // 5] = 25.0
    image += rng.normal(0, 8, size=image.shape)
    if len(shape) == 3:
        # Slightly different value per channel so the image is not grey.
        image = np.stack([image, image * 0.9, image * 1.1], axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture
def scene_image() -> np.ndarray:
    """A 64x96 BGR synthetic scene."""
    return _synthetic_scene((64, 96, 3))


@pytest.fixture
def make_image() -> Callable[..., np.ndarray]:
    """Factory fixture: ``make_image(shape, seed=0)`` builds a synthetic scene."""
    return _synthetic_scene


@pytest.fixture
def texture_image() -> np.ndarray:
    """A 64x64 BGR image of uniform random noise: maximum fine detail.

    Useful for checking detail-removing perturbations (blur, low resolution),
    because on pure texture every bit of smoothing measurably lowers variance.
    """
    return np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


# --- Fake detector and synthetic dataset ---------------------------------------


class FakeDetector:
    """A tiny, deterministic stand-in for YOLO.

    It "detects" bright rectangles on a dark background by thresholding the image
    and taking the bounding box of each bright region, all as class 0. Because it
    really looks at the pixels, perturbations affect it in a predictable way:
    darkening pushes objects below the threshold, so recall drops.
    """

    name = "fake-threshold"

    def __init__(self, threshold: int = 128, min_area: int = 20) -> None:
        self.threshold = threshold
        self.min_area = min_area  # ignore specks, e.g. single noisy pixels
        self.calls = 0

    def predict(self, image: np.ndarray) -> list[Box]:
        self.calls += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < self.min_area:
                continue
            confidence = float(gray[y : y + h, x : x + w].mean()) / 255
            boxes.append(Box(x, y, x + w, y + h, class_id=0, confidence=confidence))
        return boxes


# Rectangles (x1, y1, x2, y2) drawn as white objects in each synthetic image.
# The last image is intentionally empty (an image with no objects).
SYNTHETIC_OBJECTS = [
    [(8, 8, 28, 24), (36, 30, 56, 56)],
    [(20, 10, 44, 34)],
    [(4, 40, 20, 60), (30, 4, 60, 20)],
    [],
]


def _synthetic_sample(index: int, rects: list[tuple[int, int, int, int]]) -> Sample:
    image = np.full((64, 64, 3), 30, dtype=np.uint8)  # dark background
    for x1, y1, x2, y2 in rects:
        image[y1:y2, x1:x2] = 230
    ground_truth = [Box(x1, y1, x2, y2, class_id=0) for x1, y1, x2, y2 in rects]
    return Sample(path=Path(f"synthetic_{index}.png"), image=image, ground_truth=ground_truth)


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def synthetic_samples() -> list[Sample]:
    """Four 64x64 images with white rectangles as objects (one image has none)."""
    return [_synthetic_sample(i, rects) for i, rects in enumerate(SYNTHETIC_OBJECTS)]


@pytest.fixture
def yolo_dataset_dir(tmp_path: Path) -> Path:
    """A small YOLO-format dataset on disk, laid out like COCO128.

    - img_0.jpg: two objects (classes 0 and 2)
    - img_1.png: one object, label file has blank lines
    - img_2.jpg: no label file at all (no objects)
    - notes.txt: not an image, must be ignored
    """
    images = tmp_path / "images" / "train2017"
    labels = tmp_path / "labels" / "train2017"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    blank = np.zeros((100, 200, 3), dtype=np.uint8)  # height 100, width 200
    cv2.imwrite(str(images / "img_0.jpg"), blank)
    cv2.imwrite(str(images / "img_1.png"), blank)
    cv2.imwrite(str(images / "img_2.jpg"), blank)
    (images / "notes.txt").write_text("not an image")

    (labels / "img_0.txt").write_text("0 0.5 0.5 0.5 0.5\n2 0.25 0.75 0.1 0.2\n")
    (labels / "img_1.txt").write_text("\n1 0.5 0.5 1.0 1.0\n\n")
    return tmp_path


# --- Results files -------------------------------------------------------------


def build_results(conditions: dict[str, dict[str, float]], **metadata: Any) -> dict[str, Any]:
    """Build a results dict in the runner's format with chosen metric values.

    ``conditions`` maps ``"perturbation/severity"`` to metric overrides, e.g.
    ``{"clean/0": {"recall": 0.8}}``. Unspecified metrics default to a perfect
    score, so each test only spells out the numbers it is about.
    """
    results: dict[str, dict[str, Any]] = {}
    for condition, metrics in conditions.items():
        name, severity = condition.split("/")
        overall = {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        overall.update(metrics)
        results.setdefault(name, {})[severity] = {"overall": overall, "per_class": {}}
    return {
        "metadata": {"model": "fake", "num_images": 10, "seed": 0, **metadata},
        "results": results,
    }


@pytest.fixture
def make_results() -> Callable[..., dict[str, Any]]:
    """Factory fixture for hand-built results (see ``build_results``)."""
    return build_results


@pytest.fixture
def fake_results(fake_detector: FakeDetector, synthetic_samples: list[Sample]) -> dict[str, Any]:
    """Results from a real (fast) run of the fake detector on the synthetic samples."""
    return run_evaluation(
        fake_detector,
        synthetic_samples,
        perturbations=["brightness_decrease", "gaussian_blur", "glare"],
        severities=[1, 3, 5],
        extra_metadata={"dataset": "synthetic"},
        show_progress=False,
    )
