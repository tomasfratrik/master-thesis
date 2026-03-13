from __future__ import annotations

from .cli import main, parse_args
from .cropping import crop_detections
from .detection import (
    dedup_same_shoe_detections,
    finalize_detections,
    group_nearby_detections,
    is_pair_label,
    is_same_shoe_detection,
    nms,
    split_pair_and_single_detections,
    suppress_single_overlaps_with_pairs,
)
from .geometry import box_iou, boxes_overlap, clamp_box, expand_box
from .io_utils import IMAGE_EXTENSIONS, collect_input_images, is_image_file, slugify_stem
from .model import (
    detect_raw_boxes,
    detect_raw_boxes_batch,
    ensure_dependencies,
    fallback_to_cpu,
    format_device_banner,
    is_cuda_oom_error,
    load_detector,
    parse_device_index,
    resolve_batch_size,
)
from .pipeline import process_image

__all__ = [
    "IMAGE_EXTENSIONS",
    "box_iou",
    "boxes_overlap",
    "clamp_box",
    "collect_input_images",
    "crop_detections",
    "dedup_same_shoe_detections",
    "detect_raw_boxes",
    "detect_raw_boxes_batch",
    "ensure_dependencies",
    "expand_box",
    "fallback_to_cpu",
    "finalize_detections",
    "format_device_banner",
    "group_nearby_detections",
    "is_cuda_oom_error",
    "is_image_file",
    "is_pair_label",
    "is_same_shoe_detection",
    "load_detector",
    "main",
    "nms",
    "parse_args",
    "parse_device_index",
    "process_image",
    "resolve_batch_size",
    "slugify_stem",
    "split_pair_and_single_detections",
    "suppress_single_overlaps_with_pairs",
]
