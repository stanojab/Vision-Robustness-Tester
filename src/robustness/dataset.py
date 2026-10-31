"""Download and load a YOLO-format dataset (COCO128 by default).

A YOLO dataset has one label file per image, with the same file name stem::

    coco128/images/train2017/000000000009.jpg
    coco128/labels/train2017/000000000009.txt

Each label line is ``class x_center y_center width height`` with coordinates
normalized to 0-1. Images without a label file contain no objects.
"""

import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robustness.metrics import Box, yolo_to_box

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
DEFAULT_DATA_DIR = Path("data")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    """One image with its ground-truth boxes in pixel coordinates."""

    path: Path
    image: np.ndarray  # BGR uint8, shape (H, W, 3)
    ground_truth: list[Box]


def download_coco128(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download and extract COCO128 into ``data_dir`` unless it is already there.

    Returns:
        The dataset root, ``data_dir / "coco128"``.
    """
    root = Path(data_dir) / "coco128"
    if (root / "images").is_dir():
        return root

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading COCO128 to {data_dir} ...")
    # Download and extract into a temporary folder first, then move the result
    # into place. An interrupted download can then never leave behind a
    # half-extracted dataset that the check above would mistake for a complete one.
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        zip_path = Path(tmp) / "coco128.zip"
        urllib.request.urlretrieve(COCO128_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp)  # zipfile refuses paths that escape `tmp`
        shutil.move(str(Path(tmp) / "coco128"), root)
    return root


def load_labels(label_path: Path, image_width: int, image_height: int) -> list[Box]:
    """Read a YOLO label file and convert every line to a pixel-coordinate ``Box``.

    A missing label file means the image has no objects. Blank lines are skipped.

    Raises:
        ValueError: if a line cannot be parsed, naming the file and line number.
    """
    if not label_path.exists():
        return []

    boxes = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
            boxes.append(yolo_to_box(values, image_width, image_height))
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: invalid label line: {error}") from error
    return boxes


def load_yolo_dataset(images_dir: Path, labels_dir: Path, limit: int | None = None) -> list[Sample]:
    """Load images and their ground-truth boxes, sorted by file name.

    Args:
        images_dir: Folder containing the images.
        labels_dir: Folder containing one ``<image stem>.txt`` label file per image.
        limit: Load only the first ``limit`` images (for quick runs). None loads all.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")

    # Sorted so that "the first N images" is the same on every machine.
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"no images found in {images_dir}")

    samples = []
    for image_path in image_paths[:limit]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        height, width = image.shape[:2]
        ground_truth = load_labels(labels_dir / f"{image_path.stem}.txt", width, height)
        samples.append(Sample(path=image_path, image=image, ground_truth=ground_truth))
    return samples


def load_coco128(data_dir: Path = DEFAULT_DATA_DIR, limit: int | None = None) -> list[Sample]:
    """Download COCO128 if needed and load it."""
    root = download_coco128(data_dir)
    return load_yolo_dataset(root / "images" / "train2017", root / "labels" / "train2017", limit)
