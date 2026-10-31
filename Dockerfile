# CPU-only image that can run both the test suite and the CLI.
#
#   docker build -t vision-robustness-tester .
#   docker run --rm vision-robustness-tester bash scripts/run_tests.sh
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#       vision-robustness-tester robustness run --limit 10

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. CPU-only PyTorch first, from the PyTorch CPU index. The default PyPI wheels
#    bundle CUDA libraries (several GB) that are useless without a GPU. This is
#    the biggest layer, so it comes before anything that changes often.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. The project with its dev tools (pytest, ruff) so the image can run the tests.
#    ultralytics depends on opencv-python (the GUI build), which needs libGL and
#    other system libraries missing from slim images. We swap it for the headless
#    build: same `cv2` module, no GUI dependencies, smaller image. The two share
#    files, so headless is reinstalled after removing the GUI build.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[dev]" \
    && pip uninstall -y opencv-python \
    && pip install --no-deps --force-reinstall opencv-python-headless

COPY tests ./tests
COPY scripts ./scripts
COPY baselines ./baselines

# Run as an unprivileged user; it owns /app so downloads (dataset, weights)
# and results can be written there.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

CMD ["robustness", "--help"]
