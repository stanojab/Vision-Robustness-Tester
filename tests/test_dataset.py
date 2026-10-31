"""Tests for robustness.dataset (no network: the download is faked)."""

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from robustness import dataset
from robustness.dataset import download_coco128, load_coco128, load_labels, load_yolo_dataset
from robustness.metrics import Box

# --- Label files ---------------------------------------------------------------


def test_load_labels_converts_to_pixel_boxes(tmp_path):
    label_file = tmp_path / "labels.txt"
    label_file.write_text("0 0.5 0.5 0.5 0.5\n2 0.25 0.75 0.1 0.2\n")
    boxes = load_labels(label_file, image_width=200, image_height=100)
    assert boxes == [
        Box(50.0, 25.0, 150.0, 75.0, class_id=0),
        # cx=50, cy=75, w=20, h=20
        Box(pytest.approx(40.0), 65.0, 60.0, 85.0, class_id=2),
    ]


def test_missing_label_file_means_no_objects(tmp_path):
    assert load_labels(tmp_path / "does_not_exist.txt", 100, 100) == []


def test_blank_lines_are_skipped(tmp_path):
    label_file = tmp_path / "labels.txt"
    label_file.write_text("\n\n1 0.5 0.5 1.0 1.0\n   \n")
    assert len(load_labels(label_file, 100, 100)) == 1


@pytest.mark.parametrize(
    "bad_line",
    ["0 0.5 abc 0.1 0.1", "0 0.5 0.5 0.1", "0 0.5 0.5 -0.1 0.1"],
    ids=["not-a-number", "too-few-values", "negative-width"],
)
def test_invalid_label_line_reports_file_and_line(tmp_path, bad_line):
    label_file = tmp_path / "labels.txt"
    label_file.write_text(f"0 0.5 0.5 0.1 0.1\n{bad_line}\n")
    with pytest.raises(ValueError, match=r"labels\.txt:2: invalid label line"):
        load_labels(label_file, 100, 100)


# --- Loading a dataset folder --------------------------------------------------


def _load(root: Path, limit: int | None = None):
    return load_yolo_dataset(root / "images" / "train2017", root / "labels" / "train2017", limit)


def test_load_yolo_dataset_reads_images_and_labels(yolo_dataset_dir):
    samples = _load(yolo_dataset_dir)

    # Sorted by name; notes.txt is ignored because it is not an image.
    assert [s.path.name for s in samples] == ["img_0.jpg", "img_1.png", "img_2.jpg"]
    for sample in samples:
        assert sample.image.shape == (100, 200, 3)
        assert sample.image.dtype == np.uint8

    assert [b.class_id for b in samples[0].ground_truth] == [0, 2]
    assert samples[0].ground_truth[0].xyxy == (50.0, 25.0, 150.0, 75.0)
    assert samples[1].ground_truth == [Box(0.0, 0.0, 200.0, 100.0, class_id=1)]
    assert samples[2].ground_truth == []  # no label file -> no objects


def test_limit_loads_only_first_images(yolo_dataset_dir):
    samples = _load(yolo_dataset_dir, limit=2)
    assert [s.path.name for s in samples] == ["img_0.jpg", "img_1.png"]


def test_limit_larger_than_dataset_loads_everything(yolo_dataset_dir):
    assert len(_load(yolo_dataset_dir, limit=100)) == 3


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_raises(yolo_dataset_dir, limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        _load(yolo_dataset_dir, limit=limit)


def test_missing_images_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="images directory not found"):
        load_yolo_dataset(tmp_path / "nope", tmp_path / "labels")


def test_empty_images_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no images found"):
        load_yolo_dataset(tmp_path, tmp_path)


def test_corrupt_image_raises(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"this is not a jpeg")
    with pytest.raises(ValueError, match="could not read image"):
        load_yolo_dataset(tmp_path, tmp_path)


# --- Download (faked) ----------------------------------------------------------


def _fake_coco128_zip() -> bytes:
    """A zip with the same layout as the real COCO128 archive, but one tiny image."""
    ok, jpeg = cv2.imencode(".jpg", np.zeros((10, 20, 3), dtype=np.uint8))
    assert ok
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("coco128/images/train2017/000000000001.jpg", jpeg.tobytes())
        archive.writestr("coco128/labels/train2017/000000000001.txt", "0 0.5 0.5 0.5 0.5\n")
    return buffer.getvalue()


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the real HTTP download; records every URL that was 'downloaded'."""
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append(url)
        Path(filename).write_bytes(_fake_coco128_zip())

    monkeypatch.setattr(dataset.urllib.request, "urlretrieve", fake_urlretrieve)
    return calls


def test_download_extracts_dataset(tmp_path, fake_download):
    root = download_coco128(tmp_path)
    assert root == tmp_path / "coco128"
    assert (root / "images" / "train2017" / "000000000001.jpg").exists()
    assert (root / "labels" / "train2017" / "000000000001.txt").exists()
    assert fake_download == [dataset.COCO128_URL]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["coco128"]  # temp files cleaned up


def test_download_is_skipped_when_dataset_exists(tmp_path, fake_download):
    download_coco128(tmp_path)
    download_coco128(tmp_path)
    assert len(fake_download) == 1


def test_failed_download_leaves_no_partial_dataset(tmp_path, monkeypatch):
    def failing_urlretrieve(url, filename):
        Path(filename).write_bytes(b"partial")
        raise OSError("connection lost")

    monkeypatch.setattr(dataset.urllib.request, "urlretrieve", failing_urlretrieve)
    with pytest.raises(OSError, match="connection lost"):
        download_coco128(tmp_path)
    # Nothing left behind that the next run could mistake for a complete dataset.
    assert list(tmp_path.iterdir()) == []


def test_load_coco128_downloads_and_loads(tmp_path, fake_download):
    samples = load_coco128(tmp_path)
    assert len(samples) == 1
    assert samples[0].ground_truth == [Box(5.0, 2.5, 15.0, 7.5, class_id=0)]
