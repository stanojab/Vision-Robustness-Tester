"""Compare a results file to a saved baseline and flag performance regressions.

For every condition (clean, and each perturbation/severity) the overall
precision, recall and F1 of the current run are compared to the baseline. A
metric *regresses* when it drops by more than ``threshold`` in absolute terms:
with the default 0.05, a recall going from 0.80 to 0.74 fails (drop 0.06),
0.80 to 0.75 passes (drop exactly 0.05).

Only the overall metrics are tracked. Per-class metrics on a small dataset are
too noisy: a class with two instances swings by 0.5 when one detection flips.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from robustness.runner import load_results

DEFAULT_THRESHOLD = 0.05
DEFAULT_BASELINE_PATH = Path("baselines/baseline.json")
TRACKED_METRICS = ("precision", "recall", "f1")

# Settings that must match for the comparison to be apples-to-apples.
COMPARABLE_SETTINGS = (
    "model",
    "dataset",
    "num_images",
    "iou_threshold",
    "confidence_threshold",
    "seed",
)

# Floating-point slack, so that a drop of exactly the threshold
# (e.g. 0.80 - 0.75 = 0.05000000000000004) is not reported as a regression.
_EPSILON = 1e-9

Status = Literal["pass", "regression", "missing", "new"]


@dataclass
class MetricCheck:
    """The comparison of one metric for one condition."""

    condition: str  # e.g. "gaussian_blur/3" or "clean/0"
    metric: str
    baseline: float | None
    current: float | None
    status: Status

    @property
    def delta(self) -> float | None:
        """current - baseline (negative means the metric got worse)."""
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline


@dataclass
class RegressionReport:
    """All metric checks of a comparison, plus warnings about mismatched settings."""

    threshold: float
    checks: list[MetricCheck]
    warnings: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[MetricCheck]:
        """Checks that make the comparison fail: regressions and missing metrics."""
        return [c for c in self.checks if c.status in ("regression", "missing")]

    @property
    def passed(self) -> bool:
        return not self.failures

    def count(self, status: Status) -> int:
        return sum(1 for c in self.checks if c.status == status)

    def summary(self) -> str:
        """Human-readable summary for the terminal / CI log."""
        lines = [f"Regression check (threshold: max drop {self.threshold:.3f})"]
        lines += [f"WARNING: {warning}" for warning in self.warnings]
        lines.append(
            f"{self.count('pass')} passed, {self.count('regression')} regressed, "
            f"{self.count('missing')} missing, {self.count('new')} new"
        )

        regressions = [c for c in self.checks if c.status == "regression"]
        if regressions:
            lines.append("\nRegressions:")
            for c in sorted(regressions, key=lambda c: c.delta):
                lines.append(
                    f"  FAIL {c.condition:<24} {c.metric:<9} "
                    f"{c.baseline:.3f} -> {c.current:.3f} ({c.delta:+.3f})"
                )
        missing = [c for c in self.checks if c.status == "missing"]
        if missing:
            lines.append("\nMissing from the current results (present in the baseline):")
            lines += [f"  FAIL {c.condition:<24} {c.metric}" for c in missing]
        new = [c for c in self.checks if c.status == "new"]
        if new:
            conditions = sorted({c.condition for c in new})
            lines.append(
                f"\nNew conditions not in the baseline (not checked): {', '.join(conditions)}"
            )

        improved = [c for c in self.checks if c.status == "pass" and c.delta > self.threshold]
        if improved:
            lines.append(f"\n{len(improved)} metric(s) improved by more than the threshold.")

        lines.append("\nRESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _flatten(results: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Map (condition, metric) -> value, e.g. ("gaussian_blur/3", "recall") -> 0.71."""
    values = {}
    for perturbation, severities in results.get("results", {}).items():
        for severity, metrics in severities.items():
            overall = metrics.get("overall", {})
            for metric in TRACKED_METRICS:
                if metric in overall:
                    values[(f"{perturbation}/{severity}", metric)] = overall[metric]
    return values


def _settings_warnings(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    current_meta, baseline_meta = current.get("metadata", {}), baseline.get("metadata", {})
    warnings = []
    for key in COMPARABLE_SETTINGS:
        if key in current_meta and key in baseline_meta and current_meta[key] != baseline_meta[key]:
            warnings.append(
                f"setting '{key}' differs (baseline: {baseline_meta[key]}, "
                f"current: {current_meta[key]}); the comparison may not be meaningful"
            )
    return warnings


def compare_results(
    current: dict[str, Any], baseline: dict[str, Any], threshold: float = DEFAULT_THRESHOLD
) -> RegressionReport:
    """Compare every tracked metric of ``current`` against ``baseline``.

    - A metric that dropped by more than ``threshold`` is a regression.
    - A metric in the baseline but not in the current results is ``missing``,
      which also fails: a run that silently skipped tests must not pass.
    - A metric only in the current results is ``new``; it is reported but not
      checked, because there is nothing to compare it to.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    current_values, baseline_values = _flatten(current), _flatten(baseline)
    checks = []
    for key in sorted(baseline_values.keys() | current_values.keys()):
        condition, metric = key
        old, new = baseline_values.get(key), current_values.get(key)
        if old is None:
            status: Status = "new"
        elif new is None:
            status = "missing"
        elif old - new > threshold + _EPSILON:
            status = "regression"
        else:
            status = "pass"
        checks.append(MetricCheck(condition, metric, old, new, status))

    warnings = _settings_warnings(current, baseline)
    if not baseline_values:
        warnings.append("the baseline contains no metrics, so nothing was checked")
    return RegressionReport(threshold, checks, warnings)


def check_regression(
    results_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """Compare two files, print the summary and return the exit code (0 pass, 1 regression)."""
    report = compare_results(load_results(results_path), load_results(baseline_path), threshold)
    print(f"Current:  {results_path}\nBaseline: {baseline_path}")
    print(report.summary())
    return 0 if report.passed else 1
