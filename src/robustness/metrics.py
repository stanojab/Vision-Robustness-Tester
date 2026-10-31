"""Detection metrics: IoU, YOLO label conversion, box matching, precision/recall.

Boxes are in absolute pixel coordinates ``(x1, y1, x2, y2)``: ``(x1, y1)`` is the
top-left corner and ``(x2, y2)`` the bottom-right corner.

Edge-case convention: whenever a ratio would divide by zero (precision with no
predictions, recall with no ground truth, IoU of two zero-area boxes), the
result is 0.0. This matches scikit-learn's ``zero_division=0`` default. The raw
TP/FP/FN counts are always kept next to the ratios, so a caller can still tell
"undefined because there was no data" apart from "bad".
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_IOU_THRESHOLD = 0.5

BoxCoords = Sequence[float]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class Box:
    """A labelled box in pixel coordinates.

    Used for both predictions and ground truth; ground truth keeps the default
    confidence of 1.0.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """The box corners as an ``(x1, y1, x2, y2)`` tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or 0.0 when the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def box_area(box: BoxCoords) -> float:
    """Area of an ``(x1, y1, x2, y2)`` box. Zero-area boxes (lines, points) are allowed.

    Raises:
        ValueError: if the corners are swapped (x2 < x1 or y2 < y1), which means
            the box was built incorrectly upstream.
    """
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError(f"invalid box {tuple(box)}: expected x1 <= x2 and y1 <= y2")
    return (x2 - x1) * (y2 - y1)


def iou(box_a: BoxCoords, box_b: BoxCoords) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes, in [0, 1].

    Boxes that only touch along an edge or corner have zero intersection, so
    their IoU is 0.0. Two zero-area boxes have a zero union; that returns 0.0
    because a degenerate box can never be a valid match.
    """
    area_a = box_area(box_a)
    area_b = box_area(box_b)

    # The intersection is the rectangle between the inner edges of both boxes;
    # if the boxes don't overlap, its width or height would be negative -> 0.
    inter_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = inter_width * inter_height

    union = area_a + area_b - intersection
    return safe_divide(intersection, union)


def yolo_to_box(label: Sequence[float], image_width: int, image_height: int) -> Box:
    """Convert one YOLO label to a pixel-coordinate ground-truth ``Box``.

    Args:
        label: ``(class_id, x_center, y_center, width, height)`` with the last four
            values normalized to 0-1 relative to the image size.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    The result is clipped to the image, because rounding in label files can put
    a box edge a fraction of a pixel outside it.
    """
    if len(label) != 5:
        raise ValueError(f"YOLO label must have 5 values (class cx cy w h), got {len(label)}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
    class_id, x_center, y_center, width, height = label
    if width < 0 or height < 0:
        raise ValueError(f"YOLO label width/height must be non-negative, got {width}, {height}")

    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return Box(
        x1=min(max(x1, 0.0), image_width),
        y1=min(max(y1, 0.0), image_height),
        x2=min(max(x2, 0.0), image_width),
        y2=min(max(y2, 0.0), image_height),
        class_id=int(class_id),
    )


@dataclass
class MatchResult:
    """Outcome of matching one image's predictions to its ground truth.

    All indices refer to positions in the original ``predictions`` and
    ``ground_truth`` lists passed to :func:`match_detections`.
    """

    matches: list[tuple[int, int, float]]  # (prediction index, ground-truth index, IoU)
    false_positives: list[int]  # prediction indices that matched nothing
    false_negatives: list[int]  # ground-truth indices that no prediction matched


def _check_iou_threshold(iou_threshold: float) -> None:
    # A threshold of 0 would let any same-class prediction "match", even with no overlap.
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")


def match_detections(
    predictions: Sequence[Box],
    ground_truth: Sequence[Box],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> MatchResult:
    """Greedily match predictions to ground-truth boxes of the same class.

    Predictions are processed from highest to lowest confidence. Each one is
    matched to the still-unmatched ground-truth box of the same class with the
    highest IoU, if that IoU is >= ``iou_threshold``; otherwise it is a false
    positive. Each ground-truth box can be matched at most once, so duplicate
    detections of one object count as false positives. Ground-truth boxes left
    unmatched at the end are false negatives.
    """
    _check_iou_threshold(iou_threshold)

    # Highest confidence first; Python's sort is stable, so ties keep input order.
    order = sorted(range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True)

    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []

    for pred_index in order:
        prediction = predictions[pred_index]
        best_gt_index, best_iou = None, 0.0
        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt or gt.class_id != prediction.class_id:
                continue
            overlap = iou(prediction.xyxy, gt.xyxy)
            if overlap > best_iou:
                best_gt_index, best_iou = gt_index, overlap

        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_index)
            matches.append((pred_index, best_gt_index, best_iou))
        else:
            false_positives.append(pred_index)

    false_negatives = [i for i in range(len(ground_truth)) if i not in matched_gt]
    return MatchResult(matches, false_positives, false_negatives)


@dataclass(frozen=True)
class Counts:
    """True positive, false positive and false negative counts, plus derived ratios."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        """Fraction of predictions that were correct: TP / (TP + FP)."""
        return safe_divide(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Fraction of ground-truth objects that were found: TP / (TP + FN)."""
        return safe_divide(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall.

        Computed as 2TP / (2TP + FP + FN), which is algebraically the same as
        2PR / (P + R) but has a single denominator to guard against zero.
        """
        return safe_divide(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    def to_dict(self) -> dict[str, float]:
        """Counts and ratios as a JSON-friendly dict."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class EvaluationResult:
    """Metrics over a whole dataset: overall and per class id."""

    overall: Counts
    per_class: dict[int, Counts]

    def to_dict(self) -> dict:
        """JSON-friendly dict. Class ids become strings because JSON keys must be strings."""
        return {
            "overall": self.overall.to_dict(),
            "per_class": {str(cls): counts.to_dict() for cls, counts in self.per_class.items()},
        }


def compute_metrics(
    images: Iterable[tuple[Sequence[Box], Sequence[Box]]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> EvaluationResult:
    """Compute TP/FP/FN, precision, recall and F1 over a set of images.

    Args:
        images: One ``(predictions, ground_truth)`` pair per image. Matching is
            done per image, because a prediction can only match objects in the
            same image.
        iou_threshold: Minimum IoU for a prediction to count as a match.

    Counts are summed over all images first and the ratios computed once at the
    end (micro-averaging), so every object counts equally and images without any
    objects or predictions can't cause a division by zero.
    """
    per_class: defaultdict[int, Counts] = defaultdict(Counts)
    for predictions, ground_truth in images:
        result = match_detections(predictions, ground_truth, iou_threshold)
        for pred_index, _, _ in result.matches:
            per_class[predictions[pred_index].class_id] += Counts(tp=1)
        for pred_index in result.false_positives:
            per_class[predictions[pred_index].class_id] += Counts(fp=1)
        for gt_index in result.false_negatives:
            per_class[ground_truth[gt_index].class_id] += Counts(fn=1)

    overall = sum(per_class.values(), Counts())
    return EvaluationResult(overall=overall, per_class=dict(sorted(per_class.items())))
