from __future__ import annotations

import inspect
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers.utils import logging as transformers_logging
except ImportError:
    AutoModelForZeroShotObjectDetection = None
    AutoProcessor = None
    transformers_logging = None


def ensure_dependencies() -> None:
    if (
        Image is None
        or torch is None
        or AutoModelForZeroShotObjectDetection is None
        or AutoProcessor is None
        or transformers_logging is None
    ):
        raise SystemExit("Missing dependencies. Run: pip install -r requirements.txt")


def parse_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", maxsplit=1)[1])
    except ValueError:
        return 0


def resolve_batch_size(requested_batch_size: int, device: str) -> int:
    if requested_batch_size > 0:
        return requested_batch_size
    if not device.startswith("cuda") or torch is None or not torch.cuda.is_available():
        return 1

    try:
        device_index = parse_device_index(device)
        if device_index >= torch.cuda.device_count():
            return 1
        total_vram_gib = (
            torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        )
    except RuntimeError:
        return 1

    if total_vram_gib < 5:
        return 1
    if total_vram_gib < 8:
        return 2
    if total_vram_gib < 12:
        return 4
    return 8


def format_device_banner(device: str) -> str:
    if device.startswith("cuda") and torch is not None:
        try:
            device_index = parse_device_index(device)
            if torch.cuda.is_available() and torch.cuda.device_count() > device_index:
                device_name = torch.cuda.get_device_name(device_index)
                return f"Using device: {device} ({device_name})"
        except (ValueError, RuntimeError):
            pass
    return f"Using device: {device}"


def is_cuda_oom_error(error: BaseException, device: str) -> bool:
    return (
        device.startswith("cuda")
        and torch is not None
        and isinstance(error, torch.OutOfMemoryError)
    )


def fallback_to_cpu(model: Any) -> tuple[Any, str]:
    print("Warning: CUDA out of memory. Falling back to CPU for remaining work.")
    moved_model = model.to("cpu")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(format_device_banner("cpu"))
    return moved_model, "cpu"


def load_detector(
    model_id: str,
    device: str | None = None,
    local_files_only: bool = False,
    show_load_progress: bool = False,
) -> tuple[Any, Any, str]:
    ensure_dependencies()
    if show_load_progress:
        transformers_logging.enable_progress_bar()
    else:
        transformers_logging.disable_progress_bar()

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(format_device_banner(resolved_device))
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    ).to(resolved_device)
    model.eval()
    return processor, model, resolved_device


def detect_raw_boxes(
    image: Any,
    processor: Any,
    model: Any,
    device: str,
    labels: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    return detect_raw_boxes_batch(
        images=[image],
        processor=processor,
        model=model,
        device=device,
        labels=labels,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]


def detect_raw_boxes_batch(
    images: list[Any],
    processor: Any,
    model: Any,
    device: str,
    labels: str,
    box_threshold: float,
    text_threshold: float,
) -> list[list[dict[str, Any]]]:
    if not images:
        return []

    batched_labels = [labels] * len(images)
    inputs = processor(images=images, text=batched_labels, return_tensors="pt").to(device)

    with torch.inference_mode():
        with_scores = model(**inputs)

    post_process_kwargs: dict[str, Any] = {
        "text_threshold": text_threshold,
        "target_sizes": [image.size[::-1] for image in images],
    }
    post_process_sig = inspect.signature(
        processor.post_process_grounded_object_detection
    )
    if "box_threshold" in post_process_sig.parameters:
        post_process_kwargs["box_threshold"] = box_threshold
    elif "threshold" in post_process_sig.parameters:
        post_process_kwargs["threshold"] = box_threshold
    else:
        raise RuntimeError(
            "Unsupported transformers version: expected 'box_threshold' or "
            "'threshold' in GroundingDinoProcessor.post_process_grounded_object_detection."
        )

    batch_results = processor.post_process_grounded_object_detection(
        with_scores,
        inputs.input_ids,
        **post_process_kwargs,
    )

    all_raw: list[list[dict[str, Any]]] = []
    for results in batch_results:
        if "text_labels" in results:
            labels_out = results["text_labels"]
        else:
            labels_out = results["labels"]

        raw_detections: list[dict[str, Any]] = []
        for box, score, label in zip(results["boxes"], results["scores"], labels_out):
            xmin, ymin, xmax, ymax = box.tolist()
            raw_detections.append(
                {
                    "label": str(label),
                    "score": float(score),
                    "box_xyxy": (float(xmin), float(ymin), float(xmax), float(ymax)),
                }
            )
        all_raw.append(raw_detections)
    return all_raw
