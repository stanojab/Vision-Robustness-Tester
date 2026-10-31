"""Turn a results JSON into a Markdown report, a chart and example images.

Uses matplotlib's object-oriented ``Figure`` API instead of ``pyplot``: no global
plotting state and no GUI backend, so it works the same on a laptop, in Docker
and in CI.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.figure import Figure

from robustness.dataset import Sample
from robustness.metrics import Box
from robustness.perturbations import PERTURBATIONS, apply_perturbation
from robustness.runner import CLEAN, Detector, image_rng

# Chart colours: one fixed colour (and marker) per perturbation, assigned by its
# position in the registry, so a perturbation keeps its colour even when a run
# only includes some of them. The palette was checked for colour-blind safety.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
SERIES_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BOX_COLOR_BGR = (214, 120, 42)  # the blue above, in OpenCV's BGR order
OUTLINE_BGR = (20, 20, 20)

TABLE_METRICS = ("precision", "recall", "f1")


# --- Markdown ------------------------------------------------------------------


def format_metric(overall: dict[str, float], metric: str) -> str:
    """Format one metric with 3 decimals, or ``n/a`` when it is undefined.

    Precision is undefined without predictions (TP + FP = 0) and recall without
    ground truth (TP + FN = 0). The metrics module stores those as 0.0; here we
    show ``n/a`` so a reader doesn't mistake "no data" for "terrible model".
    """
    tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
    undefined = {
        "precision": tp + fp == 0,
        "recall": tp + fn == 0,
        "f1": tp + fp + fn == 0,
    }
    if undefined.get(metric, False):
        return "n/a"
    return f"{overall[metric]:.3f}"


def perturbation_names(results: dict[str, Any]) -> list[str]:
    """Perturbations present in the results, in registry order."""
    present = [name for name in results["results"] if name != CLEAN]
    order = list(PERTURBATIONS)
    return sorted(present, key=lambda name: order.index(name) if name in order else len(order))


def _severities(results: dict[str, Any], name: str) -> list[int]:
    return sorted(int(severity) for severity in results["results"][name])


def _overall(results: dict[str, Any], name: str, severity: int) -> dict[str, float]:
    return results["results"][name][str(severity)]["overall"]


def pivot_table(results: dict[str, Any], metric: str) -> str:
    """Markdown table with one row per perturbation and one column per severity.

    The "0 (clean)" column repeats the clean score on every row, so each row
    reads left to right as "how the metric degrades for this perturbation".
    """
    names = perturbation_names(results)
    severities = sorted({s for name in names for s in _severities(results, name)})
    clean = _overall(results, CLEAN, 0)

    header = ["Perturbation", "0 (clean)"] + [str(s) for s in severities]
    lines = [_row(header), _row(["---"] + ["---:"] * (len(header) - 1))]
    for name in names:
        cells = [name, format_metric(clean, metric)]
        for severity in severities:
            if str(severity) in results["results"][name]:
                cells.append(format_metric(_overall(results, name, severity), metric))
            else:
                cells.append("-")
        lines.append(_row(cells))
    return "\n".join(lines)


def detailed_table(results: dict[str, Any]) -> str:
    """Markdown table with every condition and all metrics and counts."""
    header = ["Perturbation", "Severity", "Precision", "Recall", "F1", "TP", "FP", "FN"]
    lines = [_row(header), _row(["---", "---:"] + ["---:"] * (len(header) - 2))]
    conditions = [(CLEAN, 0)] + [
        (name, s) for name in perturbation_names(results) for s in _severities(results, name)
    ]
    for name, severity in conditions:
        overall = _overall(results, name, severity)
        metrics = [format_metric(overall, m) for m in TABLE_METRICS]
        counts = [str(overall[c]) for c in ("tp", "fp", "fn")]
        lines.append(_row([name, str(severity), *metrics, *counts]))
    return "\n".join(lines)


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def markdown_report(results: dict[str, Any]) -> str:
    """Full Markdown report: run settings, recall and F1 summaries, all details."""
    meta = results.get("metadata", {})
    settings = [
        ("Model", meta.get("model")),
        ("Dataset", meta.get("dataset")),
        ("Images", meta.get("num_images")),
        ("IoU threshold", meta.get("iou_threshold")),
        ("Confidence threshold", meta.get("confidence_threshold")),
        ("Seed", meta.get("seed")),
        ("Created", meta.get("created_at")),
    ]
    settings_lines = [f"- **{label}:** {value}" for label, value in settings if value is not None]
    sections = [
        "# Robustness report",
        "\n".join(settings_lines),
        "## Recall by severity",
        pivot_table(results, "recall"),
        "## F1 by severity",
        pivot_table(results, "f1"),
        "## All metrics",
        detailed_table(results),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def save_markdown_report(results: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(results), encoding="utf-8")
    return path


# --- Chart ---------------------------------------------------------------------


def _series_style(name: str) -> tuple[str, str]:
    """Colour and marker for a perturbation, fixed by its registry position."""
    index = list(PERTURBATIONS).index(name) if name in PERTURBATIONS else 0
    return SERIES_COLORS[index % len(SERIES_COLORS)], SERIES_MARKERS[index % len(SERIES_MARKERS)]


def plot_robustness_curves(
    results: dict[str, Any], path: Path, metrics: Sequence[str] = ("recall", "f1")
) -> Path:
    """Save a PNG with one panel per metric: metric vs severity, one line per perturbation.

    Every line starts at severity 0 with the clean score, so the panels show how
    far each perturbation pulls the model down from its clean performance.
    Both panels share the same 0-1 y-axis so they can be compared directly.
    """
    names = perturbation_names(results)
    clean = _overall(results, CLEAN, 0)
    titles = {"precision": "Precision", "recall": "Recall", "f1": "F1 score"}

    fig = Figure(figsize=(11, 5), dpi=150, facecolor=SURFACE)
    axes = fig.subplots(1, len(metrics), sharey=True, squeeze=False)[0]
    for ax, metric in zip(axes, metrics, strict=True):
        ax.set_facecolor(SURFACE)
        for name in names:
            severities = _severities(results, name)
            xs = [0, *severities]
            ys = [clean[metric]] + [_overall(results, name, s)[metric] for s in severities]
            color, marker = _series_style(name)
            ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=5, label=name)

        ax.set_title(titles.get(metric, metric), color=TEXT_PRIMARY, fontsize=12, loc="left")
        ax.set_xlabel("Severity (0 = clean)", color=TEXT_SECONDARY)
        ax.set_xticks(range(0, 6))
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    meta = results.get("metadata", {})
    title = f"{meta.get('model', 'model')} robustness"
    if meta.get("dataset") and meta.get("num_images"):
        title += f" on {meta['dataset']} ({meta['num_images']} images)"
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=14, x=0.01, ha="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        labelcolor=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))  # leave room for the legend and title

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path


# --- Example images ------------------------------------------------------------


def draw_boxes(
    image: np.ndarray, boxes: Sequence[Box], class_names: dict[int, str] | None = None
) -> np.ndarray:
    """Return a copy of ``image`` with the boxes and their labels drawn on it."""
    out = image.copy()
    height, width = out.shape[:2]
    thickness = max(2, round(min(height, width) / 160))  # scale lines with image size
    font_scale = thickness * 0.22
    for box in boxes:
        top_left = (round(box.x1), round(box.y1))
        bottom_right = (round(box.x2), round(box.y2))
        # A dark outline under the coloured box keeps it visible on any background.
        cv2.rectangle(out, top_left, bottom_right, OUTLINE_BGR, thickness + 2)
        cv2.rectangle(out, top_left, bottom_right, BOX_COLOR_BGR, thickness)

        name = (class_names or {}).get(box.class_id, str(box.class_id))
        label = f"{name} {box.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Put the label above the box, or inside it if the box touches the top edge.
        text_y = top_left[1] - 4 if top_left[1] - text_h - 4 > 0 else top_left[1] + text_h + 4
        cv2.rectangle(
            out,
            (top_left[0], text_y - text_h - baseline),
            (top_left[0] + text_w, text_y + baseline),
            BOX_COLOR_BGR,
            cv2.FILLED,
        )
        cv2.putText(
            out,
            label,
            (top_left[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def select_examples(samples: Sequence[Sample], count: int, min_objects: int = 3) -> list[int]:
    """Pick the indices of ``count`` example images.

    Prefers the first images with at least ``min_objects`` ground-truth objects
    (more interesting to look at), then fills up with the remaining images in order.
    """
    busy = [i for i, s in enumerate(samples) if len(s.ground_truth) >= min_objects]
    rest = [i for i in range(len(samples)) if i not in busy]
    return (busy + rest)[:count]


def save_example_grid(
    detector: Detector,
    sample: Sample,
    image_index: int,
    perturbations: Sequence[str],
    severity: int,
    path: Path,
    seed: int = 0,
    class_names: dict[int, str] | None = None,
) -> Path:
    """Save a grid: the clean image plus each perturbation, with predicted boxes drawn.

    ``image_index`` and ``seed`` must match the evaluation run, so the random
    perturbations (noise, glare) look exactly like the images that were scored.
    """
    panels = [("clean", sample.image)]
    for name in perturbations:
        rng = image_rng(seed, name, image_index)
        panels.append(
            (f"{name} (severity {severity})", apply_perturbation(name, sample.image, severity, rng))
        )

    columns = 3
    rows = math.ceil(len(panels) / columns)
    fig = Figure(figsize=(4 * columns, 3.5 * rows), dpi=120, facecolor=SURFACE)
    axes = fig.subplots(rows, columns, squeeze=False).flatten()
    for ax, (title, image) in zip(axes, panels, strict=False):
        boxes = detector.predict(image)
        drawn = draw_boxes(image, boxes, class_names)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        count = f"{len(boxes)} detection{'s' if len(boxes) != 1 else ''}"
        ax.set_title(f"{title}\n{count}", color=TEXT_PRIMARY, fontsize=10)
    for ax in axes:
        ax.axis("off")  # also hides unused panels in the last row

    fig.suptitle(
        f"{sample.path.name} - {len(sample.ground_truth)} ground-truth objects",
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path
