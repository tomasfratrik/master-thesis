from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cropping import crop_detections
from .detection import finalize_detections
from .io_utils import collect_input_images, slugify_stem
from .model import (
    Image,
    detect_raw_boxes_batch,
    fallback_to_cpu,
    is_cuda_oom_error,
    load_detector,
    resolve_batch_size,
    torch,
)
from .pipeline import process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect shoes/sneakers and save each detected crop."
    )
    parser.add_argument(
        "input_image",
        type=Path,
        help="Path to a source image or a directory containing images.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where cropped images will be written.",
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face model id for zero-shot object detection.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model only from local Hugging Face cache (no network).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Batch size for directory input. "
            "Use 0 for auto (adaptive on CUDA by VRAM, cpu=1)."
        ),
    )
    parser.add_argument(
        "--dir-output-layout",
        choices=("flat", "grouped"),
        default="flat",
        help=(
            "Directory-input output layout: 'flat' puts all crops in one folder, "
            "'grouped' keeps per-image subfolders (default: flat)."
        ),
    )
    parser.add_argument(
        "--show-load-progress",
        action="store_true",
        help="Show Transformers loading progress bars.",
    )
    parser.add_argument(
        "--labels",
        default="shoe . sneaker .",
        help=(
            "Dot-separated text prompts used for detection. "
            "Example: 'shoe . sneaker . footwear .'."
        ),
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for candidate boxes (default: 0.35).",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.2,
        help="Text alignment threshold (default: 0.2).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Base padding in pixels around each crop (default: 8).",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.12,
        help=(
            "Extra padding as fraction of box size. "
            "Example 0.12 adds ~12%% of width/height on each side (default: 0.12)."
        ),
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.35,
        help="IoU threshold for non-maximum suppression (default: 0.35).",
    )
    parser.add_argument(
        "--no-same-shoe-dedup",
        action="store_true",
        help="Disable near-duplicate suppression for the same shoe.",
    )
    parser.add_argument(
        "--same-shoe-center-ratio",
        type=float,
        default=0.35,
        help=(
            "Center-distance ratio for same-shoe dedup. Lower is stricter "
            "(default: 0.35)."
        ),
    )
    parser.add_argument(
        "--same-shoe-iou-threshold",
        type=float,
        default=0.05,
        help="Minimum IoU to consider two boxes as the same shoe (default: 0.05).",
    )
    parser.add_argument(
        "--same-shoe-area-ratio",
        type=float,
        default=0.55,
        help=(
            "Minimum area-ratio (smaller/larger) for same-shoe dedup "
            "(default: 0.55)."
        ),
    )
    parser.add_argument(
        "--final-nms-threshold",
        type=float,
        default=0.0,
        help=(
            "Second NMS pass after pair grouping to suppress duplicate product crops "
            "(default: disabled, set >0 to enable)."
        ),
    )
    parser.add_argument(
        "--pair-gap-ratio",
        type=float,
        default=0.6,
        help=(
            "Grouping distance for nearby shoes as a fraction of box size. "
            "Higher merges detections more aggressively (default: 0.6)."
        ),
    )
    parser.add_argument(
        "--no-group-pairs",
        action="store_true",
        help="Disable grouping nearby shoes into one crop.",
    )
    parser.add_argument(
        "--prefer-pair-labels",
        action="store_true",
        help="Prefer pair-labeled detections and suppress overlapping single-shoe boxes.",
    )
    parser.add_argument(
        "--pair-overlap-threshold",
        type=float,
        default=0.15,
        help=(
            "If a shoe-level box overlaps a pair-labeled box by this IoU or more, "
            "the shoe-level box is dropped (default: 0.15)."
        ),
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save detected boxes and scores to metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_image

    input_images = collect_input_images(input_path, args.output_dir)
    if not input_images:
        raise SystemExit(f"No supported images found in: {input_path}")

    processor, model, device = load_detector(
        args.model_id,
        local_files_only=args.local_files_only,
        show_load_progress=args.show_load_progress,
    )

    if input_path.is_file():
        try:
            detections = process_image(
                input_image=input_images[0],
                output_dir=args.output_dir,
                processor=processor,
                model=model,
                device=device,
                labels=args.labels,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_threshold=args.nms_threshold,
                same_shoe_dedup=not args.no_same_shoe_dedup,
                same_shoe_center_ratio=args.same_shoe_center_ratio,
                same_shoe_iou_threshold=args.same_shoe_iou_threshold,
                same_shoe_area_ratio=args.same_shoe_area_ratio,
                final_nms_threshold=args.final_nms_threshold,
                group_pairs=not args.no_group_pairs,
                prefer_pair_labels=args.prefer_pair_labels,
                pair_overlap_threshold=args.pair_overlap_threshold,
                pair_gap_ratio=args.pair_gap_ratio,
                padding=args.padding,
                padding_ratio=args.padding_ratio,
                save_metadata=args.save_metadata,
                crop_prefix="shoe",
            )
        except Exception as exc:
            if not is_cuda_oom_error(exc, device):
                raise
            model, device = fallback_to_cpu(model)
            detections = process_image(
                input_image=input_images[0],
                output_dir=args.output_dir,
                processor=processor,
                model=model,
                device=device,
                labels=args.labels,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_threshold=args.nms_threshold,
                same_shoe_dedup=not args.no_same_shoe_dedup,
                same_shoe_center_ratio=args.same_shoe_center_ratio,
                same_shoe_iou_threshold=args.same_shoe_iou_threshold,
                same_shoe_area_ratio=args.same_shoe_area_ratio,
                final_nms_threshold=args.final_nms_threshold,
                group_pairs=not args.no_group_pairs,
                prefer_pair_labels=args.prefer_pair_labels,
                pair_overlap_threshold=args.pair_overlap_threshold,
                pair_gap_ratio=args.pair_gap_ratio,
                padding=args.padding,
                padding_ratio=args.padding_ratio,
                save_metadata=args.save_metadata,
                crop_prefix="shoe",
            )
        print(f"Detected and saved {len(detections)} shoe crops in {args.output_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_path.resolve()
    total_crops = 0
    global_metadata: list[dict[str, Any]] = []
    batch_size = resolve_batch_size(args.batch_size, device)
    if args.batch_size == 0:
        print(f"Auto batch size: {batch_size}")
    if device == "cpu" and len(input_images) >= 50:
        print(
            "Warning: running on CPU for many images can be slow. "
            "Use CUDA and a larger --batch-size when available."
        )

    offset = 0
    while offset < len(input_images):
        batch_paths = input_images[offset : offset + batch_size]
        batch_images: list[Any] = []
        for path in batch_paths:
            with Image.open(path) as loaded_image:
                batch_images.append(loaded_image.convert("RGB"))

        try:
            try:
                batch_raw = detect_raw_boxes_batch(
                    images=batch_images,
                    processor=processor,
                    model=model,
                    device=device,
                    labels=args.labels,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
            except Exception as exc:
                if not is_cuda_oom_error(exc, device):
                    raise

                if batch_size > 1:
                    print(
                        f"Warning: CUDA out of memory with batch-size {batch_size}. "
                        "Retrying this batch as single-image inference."
                    )
                    batch_size = 1
                    print("Adjusted batch size to 1 for remaining batches.")
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_raw = []
                    for image in batch_images:
                        try:
                            raw_single = detect_raw_boxes_batch(
                                images=[image],
                                processor=processor,
                                model=model,
                                device=device,
                                labels=args.labels,
                                box_threshold=args.box_threshold,
                                text_threshold=args.text_threshold,
                            )[0]
                        except Exception as single_exc:
                            if not is_cuda_oom_error(single_exc, device):
                                raise
                            model, device = fallback_to_cpu(model)
                            batch_size = resolve_batch_size(args.batch_size, device)
                            raw_single = detect_raw_boxes_batch(
                                images=[image],
                                processor=processor,
                                model=model,
                                device=device,
                                labels=args.labels,
                                box_threshold=args.box_threshold,
                                text_threshold=args.text_threshold,
                            )[0]
                        batch_raw.append(raw_single)
                else:
                    model, device = fallback_to_cpu(model)
                    batch_size = resolve_batch_size(args.batch_size, device)
                    batch_raw = detect_raw_boxes_batch(
                        images=batch_images,
                        processor=processor,
                        model=model,
                        device=device,
                        labels=args.labels,
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                    )

            for idx_in_batch, (image_path, image, raw_detections) in enumerate(
                zip(batch_paths, batch_images, batch_raw),
                start=1,
            ):
                merged_detections = finalize_detections(
                    raw_detections=raw_detections,
                    nms_threshold=args.nms_threshold,
                    same_shoe_dedup=not args.no_same_shoe_dedup,
                    same_shoe_center_ratio=args.same_shoe_center_ratio,
                    same_shoe_iou_threshold=args.same_shoe_iou_threshold,
                    same_shoe_area_ratio=args.same_shoe_area_ratio,
                    final_nms_threshold=args.final_nms_threshold,
                    group_pairs=not args.no_group_pairs,
                    prefer_pair_labels=args.prefer_pair_labels,
                    pair_overlap_threshold=args.pair_overlap_threshold,
                    pair_gap_ratio=args.pair_gap_ratio,
                )

                if args.dir_output_layout == "grouped":
                    rel_parent = image_path.resolve().parent.relative_to(input_root)
                    image_output_dir = args.output_dir / rel_parent / image_path.stem
                    crop_prefix = "shoe"
                else:
                    rel_stem = image_path.resolve().relative_to(input_root).with_suffix("")
                    image_output_dir = args.output_dir
                    crop_prefix = f"{slugify_stem(str(rel_stem).replace('/', '__'))}__shoe"

                saved = crop_detections(
                    image=image,
                    detections=merged_detections,
                    output_dir=image_output_dir,
                    padding=args.padding,
                    padding_ratio=args.padding_ratio,
                    crop_prefix=crop_prefix,
                )
                total_crops += len(saved)

                if args.save_metadata:
                    if args.dir_output_layout == "grouped":
                        metadata_path = image_output_dir / "metadata.json"
                        metadata_path.write_text(
                            json.dumps(saved, indent=2), encoding="utf-8"
                        )
                    else:
                        source_rel = str(image_path.resolve().relative_to(input_root))
                        for item in saved:
                            global_metadata.append(
                                {
                                    **item,
                                    "source_image": source_rel,
                                }
                            )

                global_idx = offset + idx_in_batch
                print(
                    f"[{global_idx}/{len(input_images)}] {image_path} "
                    f"({len(saved)} crops)"
                )
        finally:
            for image in batch_images:
                image.close()

        offset += len(batch_paths)

    if args.save_metadata and args.dir_output_layout == "flat":
        metadata_path = args.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(global_metadata, indent=2), encoding="utf-8")

    print(
        f"Processed {len(input_images)} images and saved {total_crops} shoe crops in {args.output_dir}"
    )
