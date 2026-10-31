"""Tests for robustness.metrics.

Expected values are worked out by hand in the comments, so each test also
documents how the metric is defined.
"""

import json

import pytest

from robustness.metrics import (
    Box,
    Counts,
    compute_metrics,
    iou,
    match_detections,
    yolo_to_box,
)


def box(x1: float, y1: float, x2: float, y2: float, cls: int = 0, conf: float = 1.0) -> Box:
    """Short helper to keep the test cases readable."""
    return Box(x1, y1, x2, y2, class_id=cls, confidence=conf)


# --- IoU -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("box_a", "box_b", "expected"),
    [
        ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),  # identical
        ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),  # disjoint
        ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),  # touching along an edge
        ((0, 0, 10, 10), (10, 10, 20, 20), 0.0),  # touching at a corner
        # Overlap is the 5x5 square (5,5)-(10,10): 25 / (100 + 100 - 25) = 1/7
        ((0, 0, 10, 10), (5, 5, 15, 15), 25 / 175),
        # Overlap 2x2 = 4, each area 8: 4 / (8 + 8 - 4) = 1/3
        ((0, 0, 4, 2), (2, 0, 6, 2), 1 / 3),
        # Small box fully inside a big one: 4 / 100
        ((0, 0, 10, 10), (2, 2, 4, 4), 0.04),
        # Non-integer coordinates: overlap 0.5x1 = 0.5, areas 1 and 1 -> 0.5 / 1.5
        ((0.0, 0.0, 1.0, 1.0), (0.5, 0.0, 1.5, 1.0), 1 / 3),
    ],
    ids=["identical", "disjoint", "edge", "corner", "partial", "partial-2", "contained", "float"],
)
def test_iou_known_values(box_a, box_b, expected):
    assert iou(box_a, box_b) == pytest.approx(expected)
    assert iou(box_b, box_a) == pytest.approx(expected)  # IoU is symmetric


@pytest.mark.parametrize(
    ("box_a", "box_b"),
    [
        ((0, 0, 10, 10), (5, 0, 5, 10)),  # zero-width line inside a box
        ((0, 0, 10, 10), (3, 3, 3, 3)),  # single point inside a box
        ((5, 5, 5, 5), (5, 5, 5, 5)),  # two identical points: union is zero
        ((0, 0, 0, 10), (0, 0, 0, 10)),  # two identical lines: union is zero
    ],
    ids=["line-in-box", "point-in-box", "two-points", "two-lines"],
)
def test_iou_zero_area_boxes_return_zero(box_a, box_b):
    assert iou(box_a, box_b) == 0.0


def test_iou_accepts_box_xyxy():
    assert iou(box(0, 0, 10, 10).xyxy, box(0, 0, 10, 10).xyxy) == 1.0


@pytest.mark.parametrize(
    "bad_box", [(10, 0, 0, 10), (0, 10, 10, 0)], ids=["x-swapped", "y-swapped"]
)
def test_iou_rejects_swapped_corners(bad_box):
    with pytest.raises(ValueError, match="invalid box"):
        iou(bad_box, (0, 0, 10, 10))


# --- YOLO label conversion -----------------------------------------------------


def test_yolo_to_box_centered_box():
    # Centre (0.5, 0.5) of a 200x100 image is (100, 50); half-size is (50, 25).
    result = yolo_to_box((0, 0.5, 0.5, 0.5, 0.5), image_width=200, image_height=100)
    assert result == Box(50.0, 25.0, 150.0, 75.0, class_id=0)


def test_yolo_to_box_full_image_and_class_id():
    result = yolo_to_box((3, 0.5, 0.5, 1.0, 1.0), image_width=640, image_height=480)
    assert result.xyxy == (0.0, 0.0, 640.0, 480.0)
    assert result.class_id == 3
    assert isinstance(result.class_id, int)


def test_yolo_to_box_off_center():
    # cx=0.25*400=100, cy=0.75*200=150, w=0.1*400=40, h=0.2*200=40
    result = yolo_to_box((7.0, 0.25, 0.75, 0.1, 0.2), image_width=400, image_height=200)
    assert result.xyxy == pytest.approx((80.0, 130.0, 120.0, 170.0))
    assert result.class_id == 7  # float class from a label file becomes int


def test_yolo_to_box_clips_to_image():
    # Left edge would be at (0.05 - 0.1) * 100 = -5 px, so it is clipped to 0.
    result = yolo_to_box((0, 0.05, 0.5, 0.2, 0.2), image_width=100, image_height=100)
    assert result.xyxy == pytest.approx((0.0, 40.0, 15.0, 60.0))


@pytest.mark.parametrize(
    ("label", "width", "height", "message"),
    [
        ((0, 0.5, 0.5, 0.1), 100, 100, "5 values"),
        ((0, 0.5, 0.5, 0.1, 0.1, 0.9), 100, 100, "5 values"),
        ((0, 0.5, 0.5, -0.1, 0.1), 100, 100, "non-negative"),
        ((0, 0.5, 0.5, 0.1, 0.1), 0, 100, "image size"),
    ],
    ids=["too-short", "too-long", "negative-width", "zero-image-width"],
)
def test_yolo_to_box_invalid_input(label, width, height, message):
    with pytest.raises(ValueError, match=message):
        yolo_to_box(label, width, height)


# --- Matching ------------------------------------------------------------------


def test_match_perfect_prediction():
    result = match_detections([box(0, 0, 10, 10)], [box(0, 0, 10, 10)])
    assert result.matches == [(0, 0, 1.0)]
    assert result.false_positives == []
    assert result.false_negatives == []


def test_match_wrong_class_is_fp_and_fn():
    # Perfect location, wrong class: a car predicted where a person is.
    result = match_detections([box(0, 0, 10, 10, cls=2)], [box(0, 0, 10, 10, cls=0)])
    assert result.matches == []
    assert result.false_positives == [0]
    assert result.false_negatives == [0]


def test_match_low_iou_is_fp_and_fn():
    # IoU 1/7 < 0.5: right class, but too far off.
    result = match_detections([box(5, 5, 15, 15)], [box(0, 0, 10, 10)])
    assert result.matches == []
    assert result.false_positives == [0]
    assert result.false_negatives == [0]


def test_match_iou_exactly_at_threshold_counts():
    # Prediction covers the top half of the GT: 50 / 100 = 0.5 exactly (>= 0.5).
    result = match_detections([box(0, 0, 10, 5)], [box(0, 0, 10, 10)], iou_threshold=0.5)
    assert len(result.matches) == 1


@pytest.mark.parametrize(("threshold", "matched"), [(0.1, True), (0.3, True), (0.5, False)])
def test_match_threshold_is_configurable(threshold, matched):
    # IoU of this pair is 1/3.
    result = match_detections([box(0, 0, 4, 2)], [box(2, 0, 6, 2)], iou_threshold=threshold)
    assert (len(result.matches) == 1) is matched


def test_duplicate_predictions_only_one_true_positive():
    # Two detections of the same object: the more confident one wins, the other is a FP.
    predictions = [box(0, 0, 10, 10, conf=0.6), box(1, 1, 10, 10, conf=0.9)]
    result = match_detections(predictions, [box(0, 0, 10, 10)])
    assert [m[0] for m in result.matches] == [1]  # index of the 0.9-confidence box
    assert result.false_positives == [0]
    assert result.false_negatives == []


def test_ground_truth_never_matched_twice():
    predictions = [box(0, 0, 10, 10, conf=c) for c in (0.9, 0.8, 0.7)]
    result = match_detections(predictions, [box(0, 0, 10, 10)])
    assert len(result.matches) == 1
    assert sorted(result.false_positives) == [1, 2]
    matched_gt = [gt for _, gt, _ in result.matches]
    assert len(matched_gt) == len(set(matched_gt))


def test_higher_confidence_gets_first_pick_even_with_lower_iou():
    # Greedy by confidence, not by IoU: the 0.9 box (IoU 0.8) takes the GT,
    # leaving the better-placed but less confident box (IoU 1.0) as a FP.
    precise_but_unsure = box(0, 0, 10, 10, conf=0.3)
    sure_but_sloppy = box(0, 0, 10, 8, conf=0.9)
    result = match_detections([precise_but_unsure, sure_but_sloppy], [box(0, 0, 10, 10)])
    assert result.matches == [(1, 0, pytest.approx(0.8))]
    assert result.false_positives == [0]


def test_prediction_matches_ground_truth_with_highest_iou():
    gts = [box(0, 0, 10, 10), box(2, 0, 12, 10)]
    # Prediction (2,0)-(12,10) is identical to the second GT (IoU 1.0 vs 0.67).
    result = match_detections([box(2, 0, 12, 10)], gts)
    assert result.matches == [(0, 1, 1.0)]
    assert result.false_negatives == [0]


def test_second_prediction_falls_back_to_next_free_ground_truth():
    # Two people standing side by side, two predictions. Both predictions overlap
    # GT 0 best, but GT 0 is taken by the more confident one, so the second
    # prediction matches GT 1 (IoU 70/130 = 0.54 >= 0.5) instead of becoming a FP.
    gts = [box(0, 0, 10, 10), box(4, 0, 14, 10)]
    predictions = [box(0, 0, 10, 10, conf=0.9), box(1, 0, 11, 10, conf=0.8)]
    result = match_detections(predictions, gts)
    assert sorted((p, g) for p, g, _ in result.matches) == [(0, 0), (1, 1)]
    assert result.false_positives == []
    assert result.false_negatives == []


def test_match_respects_class_with_multiple_objects():
    gts = [box(0, 0, 10, 10, cls=0), box(0, 0, 10, 10, cls=1)]  # overlapping person + bike
    predictions = [box(0, 0, 10, 10, cls=1, conf=0.9), box(0, 0, 10, 10, cls=0, conf=0.8)]
    result = match_detections(predictions, gts)
    assert sorted((p, g) for p, g, _ in result.matches) == [(0, 1), (1, 0)]


@pytest.mark.parametrize(
    ("predictions", "ground_truth", "fp", "fn"),
    [
        ([], [box(0, 0, 10, 10), box(20, 20, 30, 30)], [], [0, 1]),  # no predictions
        ([box(0, 0, 10, 10), box(20, 20, 30, 30)], [], [0, 1], []),  # no ground truth
        ([], [], [], []),  # both empty
    ],
    ids=["no-predictions", "no-ground-truth", "both-empty"],
)
def test_match_empty_inputs(predictions, ground_truth, fp, fn):
    result = match_detections(predictions, ground_truth)
    assert result.matches == []
    assert result.false_positives == fp
    assert result.false_negatives == fn


def test_zero_area_prediction_never_matches():
    result = match_detections([box(5, 0, 5, 10)], [box(0, 0, 10, 10)])
    assert result.false_positives == [0]
    assert result.false_negatives == [0]


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_match_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([], [], iou_threshold=threshold)


# --- Counts: precision / recall / F1 -------------------------------------------


def test_counts_ratios_hand_calculated():
    counts = Counts(tp=3, fp=1, fn=2)
    assert counts.precision == pytest.approx(3 / 4)
    assert counts.recall == pytest.approx(3 / 5)
    # F1 = 2PR / (P + R) = 2 * 0.75 * 0.6 / 1.35 = 2/3
    assert counts.f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("counts", "precision", "recall", "f1"),
    [
        (Counts(tp=0, fp=0, fn=0), 0.0, 0.0, 0.0),  # both empty: nothing to measure
        (Counts(tp=0, fp=4, fn=0), 0.0, 0.0, 0.0),  # no ground truth, only false alarms
        (Counts(tp=0, fp=0, fn=4), 0.0, 0.0, 0.0),  # no predictions at all
        (Counts(tp=4, fp=0, fn=0), 1.0, 1.0, 1.0),  # perfect
        (Counts(tp=2, fp=0, fn=2), 1.0, 0.5, 2 / 3),  # precise but misses half
        (Counts(tp=2, fp=2, fn=0), 0.5, 1.0, 2 / 3),  # finds all but half are wrong
    ],
    ids=["both-empty", "no-ground-truth", "no-predictions", "perfect", "half-recall", "half-prec"],
)
def test_counts_edge_cases(counts, precision, recall, f1):
    assert counts.precision == pytest.approx(precision)
    assert counts.recall == pytest.approx(recall)
    assert counts.f1 == pytest.approx(f1)


def test_counts_addition():
    assert Counts(1, 2, 3) + Counts(10, 20, 30) == Counts(11, 22, 33)


def test_counts_to_dict():
    assert Counts(tp=1, fp=1, fn=0).to_dict() == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2 / 3),
    }


# --- compute_metrics over several images ---------------------------------------


def test_compute_metrics_overall_and_per_class():
    images = [
        # Image 1: person found correctly; bicycle predicted as class 2 (car).
        (
            [box(0, 0, 10, 10, cls=0), box(20, 20, 30, 30, cls=2)],
            [box(0, 0, 10, 10, cls=0), box(20, 20, 30, 30, cls=1)],
        ),
        # Image 2: no objects, but a false person detection.
        ([box(0, 0, 5, 5, cls=0)], []),
        # Image 3: a bicycle the model missed.
        ([], [box(0, 0, 10, 10, cls=1)]),
    ]
    result = compute_metrics(images)

    # Overall: TP=1, FP=2 (car, false person), FN=2 (two bicycles)
    assert result.overall == Counts(tp=1, fp=2, fn=2)
    assert result.overall.precision == pytest.approx(1 / 3)
    assert result.overall.recall == pytest.approx(1 / 3)

    assert result.per_class == {
        0: Counts(tp=1, fp=1, fn=0),
        1: Counts(tp=0, fp=0, fn=2),
        2: Counts(tp=0, fp=1, fn=0),
    }
    assert list(result.per_class) == [0, 1, 2]  # sorted by class id


def test_compute_metrics_matches_within_each_image_only():
    # Same coordinates, but in different images: they must not match each other.
    images = [([box(0, 0, 10, 10)], []), ([], [box(0, 0, 10, 10)])]
    assert compute_metrics(images).overall == Counts(tp=0, fp=1, fn=1)


def test_compute_metrics_passes_iou_threshold():
    images = [([box(0, 0, 4, 2)], [box(2, 0, 6, 2)])]  # IoU 1/3
    assert compute_metrics(images, iou_threshold=0.5).overall.tp == 0
    assert compute_metrics(images, iou_threshold=0.3).overall.tp == 1


def test_compute_metrics_no_images():
    result = compute_metrics([])
    assert result.overall == Counts()
    assert result.per_class == {}


def test_evaluation_result_to_dict_is_json_serializable():
    result = compute_metrics([([box(0, 0, 10, 10, cls=5)], [box(0, 0, 10, 10, cls=5)])])
    data = json.loads(json.dumps(result.to_dict()))
    assert data["overall"]["tp"] == 1
    assert data["per_class"]["5"]["f1"] == 1.0  # class ids become string keys in JSON
