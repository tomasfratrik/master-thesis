import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)

    return (values - minimum) / (maximum - minimum)


def upsample_heatmap(
    coarse_heatmap: np.ndarray,
    image_size: tuple[int, int],
    blur_radius: float,
) -> np.ndarray:
    heatmap_image = Image.fromarray((coarse_heatmap * 255.0).astype(np.uint8), mode="L")
    upsampled_image = heatmap_image.resize(image_size, resample=Image.Resampling.BICUBIC)

    if blur_radius > 0:
        upsampled_image = upsampled_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    upsampled_values = np.asarray(upsampled_image, dtype=np.float32) / 255.0
    return normalize_map(upsampled_values)


def heatmap_rgb(normalized_heatmap: np.ndarray) -> np.ndarray:
    channel_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    red_values = np.array([0, 0, 0, 255, 255], dtype=np.float32)
    green_values = np.array([0, 128, 255, 255, 0], dtype=np.float32)
    blue_values = np.array([0, 255, 255, 0, 0], dtype=np.float32)

    flattened = normalized_heatmap.reshape(-1)
    red = np.interp(flattened, channel_positions, red_values)
    green = np.interp(flattened, channel_positions, green_values)
    blue = np.interp(flattened, channel_positions, blue_values)

    rgb = np.stack([red, green, blue], axis=-1)
    return rgb.reshape(normalized_heatmap.shape[0], normalized_heatmap.shape[1], 3).astype(np.uint8)


def heatmap_rgba(normalized_heatmap: np.ndarray, overlay_alpha: float) -> Image.Image:
    rgb = heatmap_rgb(normalized_heatmap)
    alpha_scale = max(0.0, min(float(overlay_alpha), 1.0))
    alpha = (normalized_heatmap * 255.0 * alpha_scale).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def overlay_image(image: Image.Image, heatmap: Image.Image) -> Image.Image:
    return Image.alpha_composite(image.convert("RGBA"), heatmap)


def output_directory(base_dir: Path, image_path: Path, mode: str, tag: str | None) -> Path:
    folder_name = tag or f"{image_path.stem}_{mode}"
    target = base_dir / folder_name
    suffix = 1
    while target.exists():
        target = base_dir / f"{folder_name}_{suffix}"
        suffix += 1

    target.mkdir(parents=True, exist_ok=True)
    return target


def save_outputs(
    image: Image.Image,
    coarse_heatmap: np.ndarray,
    full_heatmap: np.ndarray,
    output_dir: Path,
    report: dict[str, object],
    *,
    overlay_alpha: float,
) -> None:
    original_path = output_dir / "original.png"
    heatmap_path = output_dir / "heatmap.png"
    overlay_path = output_dir / "overlay.png"
    coarse_values_path = output_dir / "heatmap_coarse.npy"
    full_values_path = output_dir / "heatmap.npy"
    report_path = output_dir / "report.json"

    rendered_heatmap = heatmap_rgba(full_heatmap, overlay_alpha)

    image.save(original_path)
    rendered_heatmap.save(heatmap_path)
    overlay_image(image, rendered_heatmap).save(overlay_path)
    np.save(coarse_values_path, coarse_heatmap.astype(np.float32))
    np.save(full_values_path, full_heatmap.astype(np.float32))
    report_path.write_text(json.dumps(report, indent=2))


def save_channel_grid(
    image: Image.Image,
    channel_grid: Image.Image,
    output_dir: Path,
    report: dict[str, object],
) -> None:
    original_path = output_dir / "original.png"
    grid_path = output_dir / "feature_channels.png"
    report_path = output_dir / "report.json"

    image.save(original_path)
    channel_grid.save(grid_path)
    report_path.write_text(json.dumps(report, indent=2))


def render_channel_grid(
    layer_maps: list[tuple[str, list[np.ndarray]]],
    *,
    tile_size: int = 160,
    label_height: int = 30,
    padding: int = 10,
) -> Image.Image:
    if not layer_maps:
        raise ValueError("No feature channels were captured.")

    max_columns = max(len(channels) for _, channels in layer_maps)
    width = max_columns * tile_size + padding * 2
    row_height = label_height + tile_size + padding
    height = len(layer_maps) * row_height + padding
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = padding
    for layer_name, channels in layer_maps:
        draw.text((padding, y), layer_name, fill=(30, 30, 30))
        y += label_height

        x = padding
        for channel_map in channels:
            normalized = normalize_map(channel_map)
            tile_rgb = heatmap_rgb(normalized)
            tile = Image.fromarray(tile_rgb, mode="RGB").resize(
                (tile_size, tile_size),
                resample=Image.Resampling.BICUBIC,
            )
            canvas.paste(tile, (x, y))
            x += tile_size

        y += tile_size + padding

    return canvas
