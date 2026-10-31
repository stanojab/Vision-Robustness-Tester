"""Shared pytest fixtures.

All test data is synthetic and created in memory: tests never download the
dataset or the model, and never need a GPU.
"""

from collections.abc import Callable

import numpy as np
import pytest


def _synthetic_scene(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """Build a deterministic 'natural-looking' uint8 image.

    A smooth gradient plus a few solid rectangles (edges, like object outlines)
    plus mild noise (texture). Mid-range values on average, so both darkening
    and brightening have room to change it.
    """
    rng = np.random.default_rng(seed)
    height, width = shape[:2]
    gradient = np.linspace(40, 200, width, dtype=np.float32)[np.newaxis, :].repeat(height, axis=0)
    image = gradient.copy()
    image[height // 4 : height // 2, width // 5 : width // 2] = 230.0
    image[height // 2 : 3 * height // 4, width // 2 : 4 * width // 5] = 25.0
    image += rng.normal(0, 8, size=image.shape)
    if len(shape) == 3:
        # Slightly different value per channel so the image is not grey.
        image = np.stack([image, image * 0.9, image * 1.1], axis=-1)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture
def scene_image() -> np.ndarray:
    """A 64x96 BGR synthetic scene."""
    return _synthetic_scene((64, 96, 3))


@pytest.fixture
def make_image() -> Callable[..., np.ndarray]:
    """Factory fixture: ``make_image(shape, seed=0)`` builds a synthetic scene."""
    return _synthetic_scene


@pytest.fixture
def texture_image() -> np.ndarray:
    """A 64x64 BGR image of uniform random noise: maximum fine detail.

    Useful for checking detail-removing perturbations (blur, low resolution),
    because on pure texture every bit of smoothing measurably lowers variance.
    """
    return np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
