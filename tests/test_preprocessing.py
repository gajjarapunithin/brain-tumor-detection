"""Unit tests for preprocessing utilities."""

import numpy as np

from src.data.preprocessing import normalize_intensity, resize_image, skull_strip


def test_normalize_intensity_range():
    result = normalize_intensity(np.array([[0, 10], [20, 30]], dtype=np.uint8))
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_resize_image_shape():
    result = resize_image(np.zeros((10, 20), dtype=np.uint8), 8)
    assert result.shape == (8, 8)


def test_skull_strip_is_explicit_noop():
    image = np.ones((4, 4), dtype=np.float32)
    assert np.array_equal(skull_strip(image), image)
