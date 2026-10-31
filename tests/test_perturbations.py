"""Tests for robustness.perturbations."""

from itertools import pairwise

import numpy as np
import pytest

from robustness.perturbations import (
    GAUSSIAN_NOISE_SIGMA,
    MAX_SEVERITY,
    PERTURBATIONS,
    apply_perturbation,
)

ALL_NAMES = sorted(PERTURBATIONS)
RANDOM_NAMES = ["gaussian_noise", "glare"]
DETERMINISTIC_NAMES = [name for name in ALL_NAMES if name not in RANDOM_NAMES]
ALL_SEVERITIES = list(range(MAX_SEVERITY + 1))


def strictly_increasing(values: list[float]) -> bool:
    return all(a < b for a, b in pairwise(values))


def strictly_decreasing(values: list[float]) -> bool:
    return all(a > b for a, b in pairwise(values))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two images."""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


# --- Registry -----------------------------------------------------------------


def test_registry_contains_all_required_perturbations():
    assert set(PERTURBATIONS) == {
        "brightness_decrease",
        "brightness_increase",
        "gaussian_blur",
        "motion_blur",
        "gaussian_noise",
        "low_resolution",
        "jpeg_compression",
        "glare",
    }


def test_apply_perturbation_matches_direct_call(scene_image):
    direct = PERTURBATIONS["gaussian_noise"](scene_image, 3, 42)
    via_name = apply_perturbation("gaussian_noise", scene_image, 3, 42)
    np.testing.assert_array_equal(direct, via_name)


def test_apply_perturbation_unknown_name_lists_available(scene_image):
    with pytest.raises(ValueError, match="Unknown perturbation 'fog'.*gaussian_blur"):
        apply_perturbation("fog", scene_image, 1)


# --- Contract shared by every perturbation -------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_severity_zero_returns_unchanged_copy(name, scene_image):
    result = PERTURBATIONS[name](scene_image, 0, rng=0)
    np.testing.assert_array_equal(result, scene_image)
    # A copy, so callers can safely modify the result without touching the input.
    assert result is not scene_image


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("severity", ALL_SEVERITIES)
@pytest.mark.parametrize(
    "shape",
    [(64, 96, 3), (33, 47, 3), (40, 40)],
    ids=["color", "odd-size", "grayscale"],
)
def test_preserves_shape_and_dtype(name, severity, shape, make_image):
    image = make_image(shape)
    result = PERTURBATIONS[name](image, severity, rng=0)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("name", ALL_NAMES)
def test_does_not_modify_input(name, scene_image):
    original = scene_image.copy()
    PERTURBATIONS[name](scene_image, MAX_SEVERITY, rng=0)
    np.testing.assert_array_equal(scene_image, original)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_positive_severity_changes_the_image(name, scene_image):
    for severity in range(1, MAX_SEVERITY + 1):
        result = PERTURBATIONS[name](scene_image, severity, rng=0)
        assert not np.array_equal(result, scene_image), f"severity {severity} had no effect"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_higher_severity_changes_image_more(name, scene_image):
    """Mean absolute difference from the original strictly grows with severity."""
    diffs = [
        mean_abs_diff(PERTURBATIONS[name](scene_image, severity, rng=0), scene_image)
        for severity in ALL_SEVERITIES
    ]
    assert diffs[0] == 0.0
    assert strictly_increasing(diffs), diffs


# --- Reproducibility -----------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
def test_same_seed_gives_same_output(name, scene_image):
    first = PERTURBATIONS[name](scene_image, 3, rng=123)
    second = PERTURBATIONS[name](scene_image, 3, rng=123)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_same_generator_state_gives_same_output(name, scene_image):
    first = PERTURBATIONS[name](scene_image, 3, rng=np.random.default_rng(7))
    second = PERTURBATIONS[name](scene_image, 3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("name", RANDOM_NAMES)
def test_different_seeds_give_different_output(name, scene_image):
    first = PERTURBATIONS[name](scene_image, 3, rng=1)
    second = PERTURBATIONS[name](scene_image, 3, rng=2)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("name", DETERMINISTIC_NAMES)
def test_deterministic_perturbations_ignore_seed(name, scene_image):
    first = PERTURBATIONS[name](scene_image, 3, rng=1)
    second = PERTURBATIONS[name](scene_image, 3, rng=2)
    np.testing.assert_array_equal(first, second)


# --- Input validation ----------------------------------------------------------


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("severity", [-1, 6, 100])
def test_out_of_range_severity_raises_value_error(name, severity, scene_image):
    with pytest.raises(ValueError, match="severity must be between 0 and 5"):
        PERTURBATIONS[name](scene_image, severity)


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("severity", [2.5, 3.0, "3", None, True])
def test_non_integer_severity_raises_type_error(name, severity, scene_image):
    with pytest.raises(TypeError, match="severity must be an int"):
        PERTURBATIONS[name](scene_image, severity)


def test_numpy_integer_severity_is_accepted(scene_image):
    result = PERTURBATIONS["gaussian_blur"](scene_image, np.int64(2))
    assert result.shape == scene_image.shape


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (np.zeros((8, 8, 3), dtype=np.float32), TypeError),  # wrong dtype
        ([[0, 1], [2, 3]], TypeError),  # not an array
        (np.zeros((8, 8, 4), dtype=np.uint8), ValueError),  # 4 channels
        (np.zeros((8,), dtype=np.uint8), ValueError),  # 1-D
    ],
    ids=["float32", "list", "4-channel", "1-d"],
)
def test_invalid_image_raises(image, error):
    with pytest.raises(error, match="image must"):
        PERTURBATIONS["gaussian_blur"](image, 1)


# --- Behaviour of individual perturbations -------------------------------------


@pytest.mark.parametrize("name", ["gaussian_blur", "motion_blur", "low_resolution"])
def test_blur_like_perturbations_reduce_variance(name, texture_image):
    """Blurring averages neighbouring pixels, so pixel variance must keep dropping."""
    variances = [
        float(np.var(PERTURBATIONS[name](texture_image, severity))) for severity in ALL_SEVERITIES
    ]
    assert strictly_decreasing(variances), variances


def test_brightness_decrease_lowers_mean(scene_image):
    means = [
        float(PERTURBATIONS["brightness_decrease"](scene_image, s).mean()) for s in ALL_SEVERITIES
    ]
    assert strictly_decreasing(means), means


@pytest.mark.parametrize("name", ["brightness_increase", "glare"])
def test_brightening_perturbations_raise_mean(name, scene_image):
    means = [float(PERTURBATIONS[name](scene_image, s, rng=0).mean()) for s in ALL_SEVERITIES]
    assert strictly_increasing(means), means


def test_gaussian_noise_has_expected_standard_deviation():
    # Mid-grey image so the noise is not clipped at 0 or 255.
    image = np.full((200, 200), 128, dtype=np.uint8)
    severity = 2
    noisy = PERTURBATIONS["gaussian_noise"](image, severity, rng=0)
    residual = noisy.astype(np.float64) - image
    assert residual.std() == pytest.approx(GAUSSIAN_NOISE_SIGMA[severity], rel=0.05)
    assert abs(residual.mean()) < 0.5  # zero-mean noise


def test_motion_blur_is_horizontal():
    """A horizontal stripe is unaffected by a horizontal smear; a vertical one is smeared."""
    horizontal = np.zeros((41, 41), dtype=np.uint8)
    horizontal[20, :] = 255
    vertical = horizontal.T.copy()

    np.testing.assert_array_equal(PERTURBATIONS["motion_blur"](horizontal, 3), horizontal)
    smeared = PERTURBATIONS["motion_blur"](vertical, 3)
    assert smeared[20, 20] < 255  # the line got dimmer...
    assert smeared[20, 15] > 0  # ...and spread sideways


def test_low_resolution_removes_fine_detail():
    # A 1-pixel checkerboard is the finest possible detail. At severity 3 (scale 0.25)
    # each 4x4 block is averaged to the same grey, so the pattern disappears.
    checkerboard = (np.indices((40, 40)).sum(axis=0) % 2 * 255).astype(np.uint8)
    result = PERTURBATIONS["low_resolution"](checkerboard, 3)
    assert result.std() < 0.1 * checkerboard.std()


def test_glare_creates_saturated_spot_but_keeps_rest_of_image():
    dark = np.full((100, 100, 3), 20, dtype=np.uint8)
    result = PERTURBATIONS["glare"](dark, 1, rng=0)
    assert result.max() == 255  # the centre of the spot is fully overexposed
    assert result.min() == 20  # a small spot leaves far-away pixels untouched


def test_jpeg_encoding_failure_raises_clear_error(monkeypatch, scene_image):
    # Simulate OpenCV failing to encode, which otherwise would surface as a
    # confusing error further down when decoding a None buffer.
    monkeypatch.setattr("robustness.perturbations.cv2.imencode", lambda *args: (False, None))
    with pytest.raises(RuntimeError, match="JPEG encoding failed"):
        PERTURBATIONS["jpeg_compression"](scene_image, 1)
