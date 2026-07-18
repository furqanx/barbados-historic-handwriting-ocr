"""Image preprocessing utilities for line-level handwriting recognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageOps


PadAlign = Literal["left", "center", "right"]


@dataclass(frozen=True)
class ResizePadConfig:
    """Configuration for resizing line images while preserving aspect ratio."""

    target_height: int = 96
    max_width: int = 1536
    pad_value: int = 255
    align: PadAlign = "left"
    resample: int = Image.Resampling.BILINEAR

    def __post_init__(self) -> None:
        if self.target_height <= 0:
            raise ValueError("target_height must be positive.")
        if self.max_width <= 0:
            raise ValueError("max_width must be positive.")
        if not 0 <= self.pad_value <= 255:
            raise ValueError("pad_value must be in [0, 255].")
        if self.align not in {"left", "center", "right"}:
            raise ValueError("align must be one of: left, center, right.")


@dataclass(frozen=True)
class ResizeKeepAspectConfig:
    """Configuration for dynamic-width line image resizing."""

    target_height: int = 96
    max_width: int | None = 2048
    autocontrast_cutoff: int | None = None
    resample: int = Image.Resampling.BILINEAR

    def __post_init__(self) -> None:
        if self.target_height <= 0:
            raise ValueError("target_height must be positive.")
        if self.max_width is not None and self.max_width <= 0:
            raise ValueError("max_width must be positive when provided.")
        if self.autocontrast_cutoff is not None and self.autocontrast_cutoff < 0:
            raise ValueError("autocontrast_cutoff must be non-negative when provided.")


def resize_keep_aspect(
    image: Image.Image,
    *,
    target_height: int,
    max_width: int | None = None,
    resample: int = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Resize an image to a fixed height while preserving aspect ratio."""

    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    scale = target_height / height
    new_width = max(1, int(round(width * scale)))
    if max_width is not None:
        new_width = min(new_width, max_width)
    return image.resize((new_width, target_height), resample=resample)


def pad_to_width(
    image: Image.Image,
    *,
    target_width: int,
    pad_value: int = 255,
    align: PadAlign = "left",
) -> Image.Image:
    """Pad or crop an image to a fixed width."""

    if target_width <= 0:
        raise ValueError("target_width must be positive.")
    if align not in {"left", "center", "right"}:
        raise ValueError("align must be one of: left, center, right.")

    image = image.convert("RGB")
    width, height = image.size

    if width == target_width:
        return image

    if width > target_width:
        if align == "left":
            left = 0
        elif align == "center":
            left = (width - target_width) // 2
        else:
            left = width - target_width
        return image.crop((left, 0, left + target_width, height))

    canvas = Image.new("RGB", (target_width, height), color=(pad_value,) * 3)
    if align == "left":
        left = 0
    elif align == "center":
        left = (target_width - width) // 2
    else:
        left = target_width - width
    canvas.paste(image, (left, 0))
    return canvas


@dataclass(frozen=True)
class ResizePadTransform:
    """Callable image transform for OCR line images."""

    config: ResizePadConfig = field(default_factory=ResizePadConfig)

    def __call__(self, image: Image.Image) -> Image.Image:
        resized = resize_keep_aspect(
            image,
            target_height=self.config.target_height,
            max_width=self.config.max_width,
            resample=self.config.resample,
        )
        return pad_to_width(
            resized,
            target_width=self.config.max_width,
            pad_value=self.config.pad_value,
            align=self.config.align,
        )


@dataclass(frozen=True)
class ResizeKeepAspectTransform:
    """Resize a line image to fixed height and keep dynamic width."""

    config: ResizeKeepAspectConfig = field(default_factory=ResizeKeepAspectConfig)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.config.autocontrast_cutoff is not None:
            image = autocontrast(image, cutoff=self.config.autocontrast_cutoff)
        return resize_keep_aspect(
            image,
            target_height=self.config.target_height,
            max_width=self.config.max_width,
            resample=self.config.resample,
        )


def autocontrast(image: Image.Image, cutoff: int = 1) -> Image.Image:
    """Apply light autocontrast while preserving RGB mode."""

    return ImageOps.autocontrast(image.convert("RGB"), cutoff=cutoff)
