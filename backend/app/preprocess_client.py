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
) -> list[PreparedImage]:
    if not uploads:
        return []

    prepared: list[PreparedImage] = []
    for filename, payload, mime_type in uploads:
        runtime = RuntimeOptions()
        try:
            result = pipeline.process_single_image(
                filename=filename,
                image_bytes=payload,
                runtime=runtime,
            )
        except BaseException:
            # If crop-model dependencies are unavailable, fall back to the rest of the
            # preprocessing pipeline rather than failing the whole request.
            result = pipeline.process_single_image(
                filename=filename,
                image_bytes=payload,
                runtime=RuntimeOptions(disabled_steps=["crop_sneakers"]),
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

    return prepared
