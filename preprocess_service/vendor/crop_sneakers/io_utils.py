from __future__ import annotations

from pathlib import Path
import re

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_input_images(input_path: Path, output_dir: Path) -> list[Path]:
    if input_path.is_file():
        if not is_image_file(input_path):
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    input_root = input_path.resolve()
    output_root = output_dir.resolve()
    skip_output_subtree = output_root.is_relative_to(input_root)

    images: list[Path] = []
    for path in sorted(input_path.rglob("*")):
        if not is_image_file(path):
            continue
        if skip_output_subtree and path.resolve().is_relative_to(output_root):
            continue
        images.append(path)
    return images


def slugify_stem(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return slug or "image"
