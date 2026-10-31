"""Run a detector on clean and perturbed images and collect the metrics.

The model is hidden behind the small :class:`Detector` protocol, so the runner
works with any object that has a ``name`` and a ``predict(image)`` method. The
real implementation wraps YOLOv8; tests use a fast, deterministic fake.
"""

import json
import time
import zlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from tqdm import tqdm

from robustness import __version__
from robustness.dataset import Sample
from robustness.metrics import DEFAULT_IOU_THRESHOLD, Box, compute_metrics
from robustness.perturbations import MAX_SEVERITY, PERTURBATIONS, apply_perturbation

CLEAN = "clean"  # name used in the results for the unperturbed images
DEFAULT_SEVERITIES = tuple(range(1, MAX_SEVERITY + 1))
DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25  # Ultralytics' own default prediction threshold


class Detector(Protocol):
    """Anything that can detect objects in an image."""

    name: str

    def predict(self, image: np.ndarray) -> list[Box]:
        """Return the detected boxes (pixel coordinates, class id, confidence)."""
        ...


def boxes_from_arrays(xyxy: np.ndarray, classes: np.ndarray, confidences: np.ndarray) -> list[Box]:
    """Convert parallel arrays of corners, class ids and confidences into ``Box`` objects."""
    return [
        Box(float(x1), float(y1), float(x2), float(y2), class_id=int(cls), confidence=float(conf))
        for (x1, y1, x2, y2), cls, conf in zip(xyxy, classes, confidences, strict=True)
    ]


class YoloDetector:
    """A pretrained Ultralytics YOLO model behind the ``Detector`` interface."""

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "cpu",
    ) -> None:
        # Imported here rather than at the top of the file: ultralytics pulls in
        # PyTorch, which is slow to import and never needed by the tests.
        from ultralytics import YOLO

        self.model = YOLO(weights)  # downloads the weights on first use
        self.name = Path(weights).stem
        self.confidence = confidence
        self.device = device
        self.class_names: dict[int, str] = dict(self.model.names)

    def predict(self, image: np.ndarray) -> list[Box]:
        result = self.model.predict(image, conf=self.confidence, device=self.device, verbose=False)
        boxes = result[0].boxes
        return boxes_from_arrays(
            boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()
        )


def _image_rng(seed: int, perturbation: str, image_index: int) -> np.random.Generator:
    """Random generator for one (perturbation, image) pair.

    Seeding per pair (instead of one generator for the whole run) means the noise
    added to image 17 is identical whether you run 5 images or 128, one
    perturbation or all of them. Severity is deliberately not part of the seed,
    so e.g. the glare spot stays in the same place and only grows with severity.
    ``zlib.crc32`` turns the name into a stable number (Python's ``hash()`` of a
    string changes between processes).
    """
    return np.random.default_rng([seed, zlib.crc32(perturbation.encode()), image_index])


def _validate_settings(
    samples: Sequence[Sample], perturbations: Sequence[str], severities: Sequence[int]
) -> None:
    """Fail fast, before any slow inference, if the run settings are invalid."""
    if not samples:
        raise ValueError("no samples to evaluate")
    unknown = [name for name in perturbations if name not in PERTURBATIONS]
    if unknown:
        available = ", ".join(sorted(PERTURBATIONS))
        raise ValueError(f"unknown perturbation(s): {', '.join(unknown)}. Available: {available}")
    invalid = [s for s in severities if not 1 <= s <= MAX_SEVERITY]
    if invalid:
        raise ValueError(
            f"severities must be between 1 and {MAX_SEVERITY} "
            f"(the clean images are always evaluated as severity 0), got {invalid}"
        )


def run_evaluation(
    detector: Detector,
    samples: Sequence[Sample],
    perturbations: Sequence[str] = tuple(PERTURBATIONS),
    severities: Sequence[int] = DEFAULT_SEVERITIES,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    seed: int = 0,
    extra_metadata: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Evaluate ``detector`` on the clean images and on every perturbation/severity.

    Args:
        detector: The model to evaluate (loaded once by the caller).
        samples: Images with ground truth.
        perturbations: Names from the perturbation registry.
        severities: Severity levels to test for each perturbation (1-5).
        iou_threshold: Minimum IoU for a detection to count as correct.
        seed: Seed for the random perturbations, for reproducible runs.
        extra_metadata: Extra settings to record, e.g. dataset name, class names.
        show_progress: Show a progress bar.

    Returns:
        A JSON-serializable dict with ``metadata`` and ``results``, where
        ``results[perturbation][str(severity)]`` holds the metrics of that
        condition and ``results["clean"]["0"]`` the unperturbed baseline.
    """
    _validate_settings(samples, perturbations, severities)
    perturbations, severities = list(perturbations), sorted(set(severities))

    conditions = [(CLEAN, 0)] + [(name, s) for name in perturbations for s in severities]
    progress = tqdm(
        total=len(conditions) * len(samples),
        desc="Evaluating",
        unit="img",
        disable=not show_progress,
    )

    start = time.perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for name, severity in conditions:
        progress.set_postfix_str(f"{name} s{severity}")
        image_pairs = []  # (predictions, ground truth) per image
        for index, sample in enumerate(samples):
            image = sample.image
            if name != CLEAN:
                image = apply_perturbation(name, image, severity, _image_rng(seed, name, index))
            image_pairs.append((detector.predict(image), sample.ground_truth))
            progress.update()
        metrics = compute_metrics(image_pairs, iou_threshold)
        results.setdefault(name, {})[str(severity)] = metrics.to_dict()
    progress.close()

    metadata = {
        "model": detector.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_version": __version__,
        "num_images": len(samples),
        "perturbations": perturbations,
        "severities": severities,
        "iou_threshold": iou_threshold,
        "seed": seed,
        "duration_seconds": round(time.perf_counter() - start, 1),
        **(extra_metadata or {}),
    }
    return {"metadata": metadata, "results": results}


def save_results(results: dict[str, Any], path: Path) -> Path:
    """Write results to a JSON file, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def load_results(path: Path) -> dict[str, Any]:
    """Read a results JSON file written by :func:`save_results`.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a results file (missing ``results`` section).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"{path} is not a results file (missing 'results' section)")
    return data
