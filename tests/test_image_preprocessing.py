import pytest
from PIL import Image

from src.features.image_preprocessing import (
    ResizePadConfig,
    ResizePadTransform,
    pad_to_width,
    resize_keep_aspect,
)


def test_resize_keep_aspect_uses_fixed_height() -> None:
    image = Image.new("RGB", (200, 50), color="white")

    resized = resize_keep_aspect(image, target_height=100)

    assert resized.size == (400, 100)


def test_resize_keep_aspect_respects_max_width() -> None:
    image = Image.new("RGB", (1000, 50), color="white")

    resized = resize_keep_aspect(image, target_height=100, max_width=512)

    assert resized.size == (512, 100)


def test_pad_to_width_left_aligns_by_default() -> None:
    image = Image.new("RGB", (10, 4), color=(0, 0, 0))

    padded = pad_to_width(image, target_width=16, pad_value=255)

    assert padded.size == (16, 4)
    assert padded.getpixel((0, 0)) == (0, 0, 0)
    assert padded.getpixel((15, 0)) == (255, 255, 255)


def test_resize_pad_transform_returns_configured_shape() -> None:
    image = Image.new("RGB", (200, 50), color="white")
    transform = ResizePadTransform(
        ResizePadConfig(target_height=64, max_width=256, pad_value=255)
    )

    transformed = transform(image)

    assert transformed.size == (256, 64)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        ResizePadConfig(target_height=0)
