import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .preprocess_service.config import RuntimeOptions
from .preprocess_service.pipeline import PreprocessPipeline


@dataclass
class PreparedImage:
    original_filename: str
    mime_type: str
    image_bytes: bytes
    source: str


@dataclass
class PreprocessOutcome:
    images: list[PreparedImage]
    warnings: list[dict[str, str]]


pipeline = PreprocessPipeline()


def _image_to_bytes(image: Any, image_format: str) -> bytes:
    working = image.convert("RGB")
    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": image_format}
    if image_format in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = 95
    working.save(buffer, **save_kwargs)
    return buffer.getvalue()


def preprocess_uploads(
    uploads: list[tuple[str, bytes, str]],
) -> PreprocessOutcome:
    if not uploads:
        return PreprocessOutcome(images=[], warnings=[])

    prepared: list[PreparedImage] = []
    warnings: list[dict[str, str]] = []
    for filename, payload, mime_type in uploads:
        runtime = RuntimeOptions()
        try:
            result = pipeline.process_single_image(
                filename=filename,
                image_bytes=payload,
                runtime=runtime,
            )
        except BaseException as error:
            # If crop-model dependencies are unavailable or the preprocessing step
            # fails unexpectedly, fall back to the rest of the pipeline rather than
            # failing the whole request.
            result = pipeline.process_single_image(
                filename=filename,
                image_bytes=payload,
                runtime=RuntimeOptions(disabled_steps=["crop_sneakers"]),
            )
            warnings.append(
                {
                    "code": "preprocess_crop_skipped",
                    "filename": filename,
                    "message": (
                        "Sneaker crop step failed, so the original image was used after "
                        f"fallback. ({type(error).__name__})"
                    ),
                }
            )

        crop_runs = (result.get("metadata") or {}).get("crop_sneakers") or []
        for crop_meta in crop_runs:
            fallback = crop_meta.get("device_fallback")
            if fallback:
                warnings.append(
                    {
                        "code": "preprocess_cuda_oom_cpu_retry",
                        "filename": filename,
                        "message": (
                            "GPU ran out of memory during sneaker cropping. "
                            f"Retried on {fallback['to']}."
                        ),
                    }
                )
            if crop_meta.get("used_original"):
                warnings.append(
                    {
                        "code": "preprocess_used_original",
                        "filename": filename,
                        "message": (
                            "No sneaker crop was detected, so the original image was analyzed."
                        ),
                    }
                )

        outputs = result.get("outputs") or []
        if not outputs:
            prepared.append(
                PreparedImage(
                    original_filename=filename,
                    image_bytes=payload,
                    mime_type=mime_type,
                    source="original",
                )
            )
            continue

        first_output = outputs[0]
        image = first_output["image"]
        image_format = str(first_output.get("format") or "JPEG")
        prepared.append(
            PreparedImage(
                original_filename=first_output.get("name") or filename,
                image_bytes=_image_to_bytes(image, image_format),
                mime_type=f"image/{image_format.lower()}",
                source="preprocessed",
            )
        )

    return PreprocessOutcome(images=prepared, warnings=warnings)
