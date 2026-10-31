#!/usr/bin/env bash
# Run the linter, the formatting check and the test suite with coverage.
# Used locally, inside Docker and in CI, so all three run exactly the same checks.
# Stops at the first failing step and exits non-zero.
#
# Usage: bash scripts/run_tests.sh [extra pytest arguments]
# Set PYTHON to choose the interpreter, e.g. PYTHON=.venv/Scripts/python on Windows.
set -euo pipefail

PYTHON="${PYTHON:-python}"
CORE_MODULES="*/robustness/perturbations.py,*/robustness/metrics.py,*/robustness/regression.py"
CORE_MIN_COVERAGE=85

# Run from the repository root, wherever the script is called from.
cd "$(dirname "$0")/.."

echo "==> Lint (ruff check)"
"$PYTHON" -m ruff check .

echo "==> Formatting (ruff format --check)"
"$PYTHON" -m ruff format --check .

echo "==> Tests with coverage"
"$PYTHON" -m pytest --cov=robustness --cov-report=term-missing --cov-report=xml "$@"

echo "==> Coverage of core modules (minimum ${CORE_MIN_COVERAGE}%)"
"$PYTHON" -m coverage report --include="$CORE_MODULES" --fail-under="$CORE_MIN_COVERAGE"

echo "All checks passed."
