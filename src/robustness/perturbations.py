"""Image perturbations that simulate real-world image degradation.

Every perturbation has the same signature::

    perturb(image, severity, rng=None) -> image

- ``image``: uint8 array of shape (H, W) or (H, W, 3), BGR as loaded by OpenCV.
- ``severity``: int from 0 (no change) to 5 (strongest).
- ``rng``: a seed or ``numpy.random.Generator``. Only perturbations with a random
  component use it; the same seed always gives the same output.

The output always has the same shape and dtype as the input, and the input is
never modified in place.

To add a new perturbation: write a function decorated with ``@_perturbation``
and add it to the ``PERTURBATIONS`` registry at the bottom of this file.
"""

import functools
from collections.abc import Callable

import cv2
import numpy as np

MAX_SEVERITY = 5

RngLike = np.random.Generator | int | None
Perturbation = Callable[..., np.ndarray]

# Strength of each perturbation per severity level. The tuple index is the
# severity, so index 0 is never used (severity 0 always returns the input).
BRIGHTNESS_DECREASE_FACTOR = (1.0, 0.8, 0.6, 0.45, 0.3, 0.15)  # pixel multiplier
BRIGHTNESS_INCREASE_OFFSET = (0, 30, 60, 90, 120, 150)  # added to every pixel
GAUSSIAN_BLUR_SIGMA = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0)  # Gaussian kernel std (px)
MOTION_BLUR_LENGTH = (1, 5, 9, 15, 21, 31)  # length of the smear (px)
GAUSSIAN_NOISE_SIGMA = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0)  # noise std (0-255 scale)
LOW_RESOLUTION_SCALE = (1.0, 0.6, 0.4, 0.25, 0.15, 0.1)  # downscale factor
JPEG_QUALITY = (100, 50, 30, 15, 8, 4)  # OpenCV JPEG quality (1-100)
GLARE_RADIUS_FRACTION = (0.0, 0.15, 0.25, 0.35, 0.45, 0.6)  # of the shorter image side


def _validate(image: np.ndarray, severity: int) -> None:
    """Raise a clear error if the image or severity is invalid."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        got = image.dtype if isinstance(image, np.ndarray) else type(image).__name__
        raise TypeError(f"image must be a uint8 numpy array, got {got}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3)):
        raise ValueError(f"image must have shape (H, W) or (H, W, 3), got {image.shape}")
    # bool is a subclass of int in Python, so reject it explicitly:
    # otherwise severity=True would silently mean severity=1.
    if isinstance(severity, bool) or not isinstance(severity, int | np.integer):
        raise TypeError(
            f"severity must be an int between 0 and {MAX_SEVERITY}, got {type(severity).__name__}"
        )
    if not 0 <= severity <= MAX_SEVERITY:
        raise ValueError(f"severity must be between 0 and {MAX_SEVERITY}, got {severity}")


def _perturbation(
    func: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
) -> Perturbation:
    """Wrap a perturbation so all of them share the same contract.

    The wrapper validates the inputs, returns an unchanged copy for severity 0,
    and turns ``rng`` (seed, Generator or None) into a Generator. The wrapped
    function therefore only has to implement severities 1-5.
    """

    @functools.wraps(func)
    def wrapper(image: np.ndarray, severity: int, rng: RngLike = None) -> np.ndarray:
        _validate(image, severity)
        if severity == 0:
            return image.copy()
        return func(image, int(severity), np.random.default_rng(rng))

    return wrapper


def _to_uint8(values: np.ndarray) -> np.ndarray:
    """Round and clip float values back to the valid uint8 range."""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


@_perturbation
def brightness_decrease(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Darken the image (underexposure, low light) by scaling all pixel values down."""
    return _to_uint8(image.astype(np.float32) * BRIGHTNESS_DECREASE_FACTOR[severity])


@_perturbation
def brightness_increase(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Brighten the image (overexposure) by adding a constant; bright areas wash out to white."""
    return _to_uint8(image.astype(np.float32) + BRIGHTNESS_INCREASE_OFFSET[severity])


@_perturbation
def gaussian_blur(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Blur the image uniformly (out-of-focus camera)."""
    # ksize=(0, 0) lets OpenCV pick a kernel size large enough for the sigma.
    return cv2.GaussianBlur(image, (0, 0), sigmaX=GAUSSIAN_BLUR_SIGMA[severity])


@_perturbation
def motion_blur(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Smear the image horizontally (camera panning or a fast-moving object).

    Each output pixel is the average of ``length`` neighbouring pixels in its row,
    which is what a camera sees when the scene moves sideways during exposure.
    """
    length = MOTION_BLUR_LENGTH[severity]
    kernel = np.full((1, length), 1.0 / length, dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


@_perturbation
def gaussian_noise(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Add random Gaussian noise (sensor noise, e.g. high ISO in low light)."""
    noise = rng.normal(0.0, GAUSSIAN_NOISE_SIGMA[severity], size=image.shape)
    return _to_uint8(image.astype(np.float32) + noise)


@_perturbation
def low_resolution(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Downscale then upscale back to the original size (low-resolution camera).

    INTER_AREA averages pixels when shrinking, so fine detail is lost for good.
    Upscaling uses bilinear interpolation because that is what a detection
    pipeline does when it resizes a small camera frame up to the model input size.
    """
    height, width = image.shape[:2]
    scale = LOW_RESOLUTION_SCALE[severity]
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


@_perturbation
def jpeg_compression(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Encode and decode as JPEG at decreasing quality (compression artifacts)."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY[severity]])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    # IMREAD_UNCHANGED keeps grayscale images grayscale.
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)


@_perturbation
def glare(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Add an overexposed bright spot (sun or headlight glare) at a random position.

    The spot is a Gaussian-shaped mask that is fully white at its centre and fades
    out smoothly. Higher severity makes the spot larger.
    """
    height, width = image.shape[:2]
    center_y = rng.integers(0, height)
    center_x = rng.integers(0, width)
    sigma = GLARE_RADIUS_FRACTION[severity] * min(height, width)

    yy, xx = np.ogrid[:height, :width]
    squared_distance = (yy - center_y) ** 2 + (xx - center_x) ** 2
    mask = np.exp(-squared_distance / (2 * sigma**2))
    if image.ndim == 3:
        mask = mask[..., np.newaxis]  # broadcast the same mask over all channels

    # Blend each pixel towards white by the mask weight (1.0 at the centre).
    return _to_uint8(image * (1.0 - mask) + 255.0 * mask)


# Registry of all perturbations, keyed by the name used in the CLI and results.
PERTURBATIONS: dict[str, Perturbation] = {
    "brightness_decrease": brightness_decrease,
    "brightness_increase": brightness_increase,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "low_resolution": low_resolution,
    "jpeg_compression": jpeg_compression,
    "glare": glare,
}


def apply_perturbation(
    name: str, image: np.ndarray, severity: int, rng: RngLike = None
) -> np.ndarray:
    """Apply a perturbation by its registry name.

    Raises:
        ValueError: if ``name`` is not a registered perturbation.
    """
    try:
        perturb = PERTURBATIONS[name]
    except KeyError:
        available = ", ".join(sorted(PERTURBATIONS))
        raise ValueError(f"Unknown perturbation {name!r}. Available: {available}") from None
    return perturb(image, severity, rng)
