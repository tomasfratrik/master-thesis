from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import clamp_box


def crop_detections(
    image: Any,
    detections: list[dict[str, Any]],
    output_dir: Path,
    padding: int,
    padding_ratio: float,
    crop_prefix: str = "shoe",
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    saved: list[dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = clamp_box(
            *det["box_xyxy"],
            width=width,
            height=height,
            padding=padding,
            padding_ratio=padding_ratio,
        )
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        crop_path = output_dir / f"{crop_prefix}_{idx:03d}.jpg"
        crop.save(crop_path, quality=95)

        saved.append(
            {
                "file": crop_path.name,
                "label": det["label"],
                "score": det["score"],
                "members": int(det.get("members", 1)),
                "box_xyxy": [x1, y1, x2, y2],
            }
        )
    return saved
