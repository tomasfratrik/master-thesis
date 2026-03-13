from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cropping import crop_detections
from .detection import finalize_detections
from .model import Image, detect_raw_boxes


def process_image(
    input_image: Path,
    output_dir: Path,
    processor: Any,
    model: Any,
    device: str,
    labels: str,
    box_threshold: float,
    text_threshold: float,
    nms_threshold: float,
    same_shoe_dedup: bool,
    same_shoe_center_ratio: float,
    same_shoe_iou_threshold: float,
    same_shoe_area_ratio: float,
    final_nms_threshold: float,
    group_pairs: bool,
    prefer_pair_labels: bool,
    pair_overlap_threshold: float,
    pair_gap_ratio: float,
    padding: int,
    padding_ratio: float,
    save_metadata: bool,
    crop_prefix: str = "shoe",
) -> list[dict[str, Any]]:
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_image).convert("RGB")

    raw_detections = detect_raw_boxes(
        image=image,
        processor=processor,
        model=model,
        device=device,
        labels=labels,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    merged_detections = finalize_detections(
        raw_detections=raw_detections,
        nms_threshold=nms_threshold,
        same_shoe_dedup=same_shoe_dedup,
        same_shoe_center_ratio=same_shoe_center_ratio,
        same_shoe_iou_threshold=same_shoe_iou_threshold,
        same_shoe_area_ratio=same_shoe_area_ratio,
        final_nms_threshold=final_nms_threshold,
        group_pairs=group_pairs,
        prefer_pair_labels=prefer_pair_labels,
        pair_overlap_threshold=pair_overlap_threshold,
        pair_gap_ratio=pair_gap_ratio,
    )

    saved = crop_detections(
        image=image,
        detections=merged_detections,
        output_dir=output_dir,
        padding=padding,
        padding_ratio=padding_ratio,
        crop_prefix=crop_prefix,
    )

    if save_metadata:
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    return saved
