from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.preprocess_service.config import CropSneakersRuntimeConfig, RuntimeOptions
from backend.app.preprocess_service.pipeline import PreprocessPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sneaker crop preprocessing step for one image and save all detected crops."
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where crop images and metadata will be written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input image stem.",
    )
    parser.add_argument(
        "--return-original-if-empty",
        action="store_true",
        help="Save the original image as a fallback when no sneaker crop is detected.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=None,
        help="Fixed crop padding in pixels. Uses pipeline default when omitted.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=None,
        help="Relative crop padding based on detected box size. Uses pipeline default when omitted.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help='Detector labels, for example "shoe . sneaker .". Uses pipeline default when omitted.',
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
        help="Object detection box threshold. Uses pipeline default when omitted.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=None,
        help="Object detection text threshold. Uses pipeline default when omitted.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the detection model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> RuntimeOptions:
    crop_config = CropSneakersRuntimeConfig(
        return_original_if_empty=args.return_original_if_empty,
        padding=args.padding,
        padding_ratio=args.padding_ratio,
        labels=args.labels,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        local_files_only=True if args.local_files_only else None,
    )
    return RuntimeOptions(
        enabled_steps=["crop_sneakers"],
        include_images=True,
        crop_sneakers=crop_config,
    )


def save_outputs(result: dict[str, Any], input_path: Path, output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = result.get("outputs") or []
    for index, output in enumerate(outputs, start=1):
        output_path = output_dir / f"{prefix}_crop_{index}.jpg"
        output["image"].save(output_path, format="JPEG", quality=95)
        print(output_path)

    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata = {
        "input": str(input_path),
        "output_count": len(outputs),
        "applied_steps": result.get("applied_steps", []),
        "skipped_steps": result.get("skipped_steps", []),
        "metadata": result.get("metadata", {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata_path)

    return len(outputs)


def main() -> int:
    args = parse_args()
    input_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input image does not exist: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input path is not a file: {input_path}")

    pipeline = PreprocessPipeline()
    result = pipeline.process_single_image(
        filename=input_path.name,
        image_bytes=input_path.read_bytes(),
        runtime=build_runtime(args),
    )

    prefix = args.prefix or input_path.stem
    output_count = save_outputs(
        result=result,
        input_path=input_path,
        output_dir=output_dir,
        prefix=prefix,
    )
    if output_count == 0:
        print("No crop detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
