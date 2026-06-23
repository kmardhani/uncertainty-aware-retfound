"""Small image preprocessing utilities for retinal image experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from PIL import Image

ImageTransform = Callable[[Image.Image], Image.Image]


def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an integer or tuple size into a width-height pair."""

    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"Image size must be positive. Got: {size}")
        return (size, size)

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive. Got: {size}")

    return (width, height)


@dataclass(frozen=True)
class EnsureRGB:
    """Convert an image to RGB mode."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


@dataclass(frozen=True)
class ResizeImage:
    """Resize an image to a fixed size using bilinear resampling."""

    size: int | tuple[int, int]

    def __call__(self, image: Image.Image) -> Image.Image:
        target_size = _normalize_size(self.size)
        return image.resize(target_size, resample=Image.Resampling.BILINEAR)


@dataclass(frozen=True)
class CenterCropImage:
    """Crop an image around its center."""

    size: int | tuple[int, int]

    def __call__(self, image: Image.Image) -> Image.Image:
        crop_width, crop_height = _normalize_size(self.size)
        image_width, image_height = image.size

        if crop_width > image_width or crop_height > image_height:
            raise ValueError(
                "Center crop size must not exceed image size. "
                f"Requested {(crop_width, crop_height)} for image {image.size}."
            )

        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height

        return image.crop((left, top, right, bottom))


@dataclass(frozen=True)
class ComposeTransforms:
    """Apply a sequence of image transforms in order."""

    transforms: Sequence[ImageTransform]

    def __call__(self, image: Image.Image) -> Image.Image:
        output = image
        for transform in self.transforms:
            output = transform(output)
        return output


def build_transform_from_config(config: dict[str, Any]) -> ImageTransform:
    """Build a simple image transform pipeline from a small config mapping."""

    transforms: list[ImageTransform] = []

    if config.get("ensure_rgb", False):
        transforms.append(EnsureRGB())

    if "resize" in config and config["resize"] is not None:
        transforms.append(ResizeImage(config["resize"]))

    if "center_crop" in config and config["center_crop"] is not None:
        transforms.append(CenterCropImage(config["center_crop"]))

    return ComposeTransforms(transforms)
