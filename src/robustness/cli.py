"""Command-line interface.

Subcommands::

    robustness run       evaluate the model on clean + perturbed images, save results JSON
    robustness report    generate the Markdown table, chart and example images
    robustness baseline  save a results file as the new baseline
    robustness check     compare a results file to the baseline (exit code 1 on regression)

Exit codes: 0 success, 1 regression found (``check``), 2 usage or input error.
"""

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from robustness.dataset import DEFAULT_DATA_DIR, Sample, load_coco128
from robustness.metrics import DEFAULT_IOU_THRESHOLD
from robustness.perturbations import MAX_SEVERITY, PERTURBATIONS
from robustness.regression import DEFAULT_BASELINE_PATH, DEFAULT_THRESHOLD, check_regression
from robustness.report import (
    perturbation_names,
    plot_robustness_curves,
    save_example_grid,
    save_markdown_report,
    select_examples,
)
from robustness.runner import (
    CLEAN,
    DEFAULT_CONFIDENCE,
    DEFAULT_SEVERITIES,
    DEFAULT_WEIGHTS,
    Detector,
    YoloDetector,
    load_results,
    run_evaluation,
    save_results,
)

DEFAULT_RESULTS_PATH = Path("results/results.json")
DEFAULT_DOCS_DIR = Path("docs")


# The two slow, external dependencies sit behind these functions so the CLI
# tests can replace them with the fake detector and synthetic samples.
def make_detector(weights: str, confidence: float) -> Detector:
    return YoloDetector(weights, confidence)


def load_samples(data_dir: Path, limit: int | None) -> list[Sample]:
    return load_coco128(data_dir, limit)


# --- Argument types (argparse turns ArgumentTypeError into a clean usage error) ---


def perturbation_list(text: str) -> list[str]:
    """Parse ``"all"`` or a comma-separated list of perturbation names."""
    if text.strip() == "all":
        return list(PERTURBATIONS)
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in PERTURBATIONS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown perturbation(s): {', '.join(unknown) or text!r}. "
            f"Choose from: all, {', '.join(PERTURBATIONS)}"
        )
    return names


def severity(text: str) -> int:
    """Parse one severity level, 1-5."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"severity must be an integer, got {text!r}") from None
    if not 1 <= value <= MAX_SEVERITY:
        raise argparse.ArgumentTypeError(f"severity must be between 1 and {MAX_SEVERITY}")
    return value


def severity_list(text: str) -> list[int]:
    """Parse a comma-separated list of severities, e.g. ``"1,3,5"``."""
    parts = [part for part in text.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one severity, e.g. 1,3,5")
    return [severity(part) for part in parts]


def _float_in_range(text: str, low: float, high: float, include_low: bool) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    above_low = value >= low if include_low else value > low
    if not (above_low and value <= high):
        bracket = "[" if include_low else "("
        raise argparse.ArgumentTypeError(f"must be in {bracket}{low}, {high}], got {value}")
    return value


def iou_value(text: str) -> float:
    return _float_in_range(text, 0.0, 1.0, include_low=False)


def fraction(text: str) -> float:
    return _float_in_range(text, 0.0, 1.0, include_low=True)


def _int_at_least(text: str, minimum: int) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < minimum:
        raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
    return value


def positive_int(text: str) -> int:
    return _int_at_least(text, 1)


def non_negative_int(text: str) -> int:
    return _int_at_least(text, 0)


# --- Commands ------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    samples = load_samples(args.data_dir, args.limit)
    detector = make_detector(args.weights, args.confidence)
    class_names = getattr(detector, "class_names", None)
    results = run_evaluation(
        detector,
        samples,
        perturbations=args.perturbations,
        severities=args.severities,
        iou_threshold=args.iou_threshold,
        seed=args.seed,
        extra_metadata={
            "dataset": "coco128",
            "weights": args.weights,
            "confidence_threshold": args.confidence,
            "class_names": {str(k): v for k, v in (class_names or {}).items()},
        },
        show_progress=not args.no_progress,
    )
    path = save_results(results, args.output)

    clean = results["results"][CLEAN]["0"]["overall"]
    conditions = [
        (f"{name}/{sev}", values["overall"]["f1"])
        for name, by_severity in results["results"].items()
        if name != CLEAN
        for sev, values in by_severity.items()
    ]
    print(f"\nSaved results to {path}")
    print(
        f"Clean: precision {clean['precision']:.3f}, "
        f"recall {clean['recall']:.3f}, F1 {clean['f1']:.3f}"
    )
    if conditions:
        worst_name, worst_f1 = min(conditions, key=lambda item: item[1])
        print(f"Worst condition: {worst_name} (F1 {worst_f1:.3f})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    output_dir = Path(args.output_dir)
    print(f"Wrote {save_markdown_report(results, output_dir / 'results.md')}")
    print(f"Wrote {plot_robustness_curves(results, output_dir / 'robustness_curves.png')}")

    if args.examples > 0:
        meta = results.get("metadata", {})
        weights = args.weights or meta.get("weights", DEFAULT_WEIGHTS)
        detector = make_detector(weights, meta.get("confidence_threshold", DEFAULT_CONFIDENCE))
        class_names = {int(k): v for k, v in meta.get("class_names", {}).items()}
        samples = load_samples(args.data_dir, meta.get("num_images"))
        for index in select_examples(samples, args.examples):
            sample = samples[index]
            path = save_example_grid(
                detector,
                sample,
                image_index=index,
                perturbations=perturbation_names(results),
                severity=args.example_severity,
                path=output_dir / "examples" / f"example_{sample.path.stem}.png",
                seed=meta.get("seed", 0),
                class_names=class_names,
            )
            print(f"Wrote {path}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    results = load_results(args.results)  # validates the file before it becomes the baseline
    baseline = Path(args.baseline)
    replaced = baseline.exists()
    baseline.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.results, baseline)

    meta = results.get("metadata", {})
    action = "Replaced" if replaced else "Saved"
    print(
        f"{action} baseline {baseline} from {args.results} "
        f"(model: {meta.get('model')}, images: {meta.get('num_images')}, "
        f"created: {meta.get('created_at')})"
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    return check_regression(args.results, args.baseline, args.threshold)


# --- Parser --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robustness",
        description="Measure object detection robustness to image degradation "
        "and detect regressions against a baseline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    defaults = argparse.ArgumentDefaultsHelpFormatter

    run = sub.add_parser(
        "run", help="evaluate the model and save results", formatter_class=defaults
    )
    run.add_argument(
        "--perturbations",
        type=perturbation_list,
        default=list(PERTURBATIONS),
        help="'all' or a comma-separated list, e.g. gaussian_blur,glare",
    )
    run.add_argument(
        "--severities",
        type=severity_list,
        default=list(DEFAULT_SEVERITIES),
        help="comma-separated severities 1-5 (clean images are always included)",
    )
    run.add_argument("--iou-threshold", type=iou_value, default=DEFAULT_IOU_THRESHOLD)
    run.add_argument(
        "--confidence",
        type=fraction,
        default=DEFAULT_CONFIDENCE,
        help="detection confidence threshold",
    )
    run.add_argument("--limit", type=positive_int, default=None, help="only use the first N images")
    run.add_argument("--seed", type=int, default=0, help="seed for random perturbations")
    run.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLO weights file or name")
    run.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    run.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    run.add_argument("--no-progress", action="store_true", help="hide the progress bar")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser(
        "report", help="generate table, chart and examples", formatter_class=defaults
    )
    report.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_DOCS_DIR)
    report.add_argument(
        "--examples",
        type=non_negative_int,
        default=2,
        help="number of example images (needs the model and dataset; 0 to skip)",
    )
    report.add_argument("--example-severity", type=severity, default=3)
    report.add_argument(
        "--weights", default=None, help="weights for examples (default: from results)"
    )
    report.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    report.set_defaults(func=cmd_report)

    baseline = sub.add_parser(
        "baseline", help="save results as the new baseline", formatter_class=defaults
    )
    baseline.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    baseline.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    baseline.set_defaults(func=cmd_baseline)

    check = sub.add_parser(
        "check", help="compare results to the baseline", formatter_class=defaults
    )
    check.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    check.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    check.add_argument(
        "--threshold", type=fraction, default=DEFAULT_THRESHOLD, help="maximum allowed metric drop"
    )
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        # Expected user-facing problems (missing file, bad JSON, invalid settings):
        # a one-line message instead of a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
