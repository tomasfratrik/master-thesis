from __future__ import annotations

import base64
import io
import json
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .config import RuntimeOptions
from .pipeline import PreprocessPipeline

app = FastAPI(title="Sneaker Preprocess Service")
pipeline = PreprocessPipeline()


def _normalize_output_format(value: str | None) -> str:
    normalized = str(value or "JPEG").strip().upper()
    if normalized == "JPG":
        normalized = "JPEG"
    if normalized not in {"JPEG", "PNG", "WEBP"}:
        return "JPEG"
    return normalized


def _mime_type_for_format(image_format: str) -> str:
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }[image_format]


def _parse_runtime_options(raw_options: str | None) -> RuntimeOptions:
    if raw_options is None or not raw_options.strip():
        return RuntimeOptions()
    try:
        parsed = json.loads(raw_options)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in 'options': {exc}")

    try:
        return RuntimeOptions.model_validate(parsed)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())


def _image_to_bytes(image: Any, image_format: str) -> bytes:
    image_format = _normalize_output_format(image_format)
    if image_format == "JPEG":
        working = image.convert("RGB")
    else:
        working = image.convert("RGB")

    buffer = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": image_format}
    if image_format in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = 95
    working.save(buffer, **save_kwargs)
    return buffer.getvalue()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/preprocess")
async def preprocess(
    files: list[UploadFile] = File(...),
    options: str | None = Form(default=None),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    runtime = _parse_runtime_options(options)
    request_id = str(uuid.uuid4())
    results: list[dict[str, Any]] = []

    for file in files:
        filename = file.filename or "upload"
        payload = await file.read()
        if not payload:
            results.append(
                {"filename": filename, "status": "error", "error": "Empty file payload."}
            )
            continue

        try:
            result = pipeline.process_single_image(
                filename=filename,
                image_bytes=payload,
                runtime=runtime,
            )
        except Exception as exc:
            results.append(
                {
                    "filename": filename,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        serialized_outputs: list[dict[str, Any]] = []
        for output in result["outputs"]:
            image = output["image"]
            image_format = _normalize_output_format(output.get("format"))
            item = {
                "name": output["name"],
                "width": int(image.width),
                "height": int(image.height),
                "mime_type": _mime_type_for_format(image_format),
                "format": image_format,
            }
            if runtime.include_images:
                encoded = base64.b64encode(_image_to_bytes(image, image_format)).decode(
                    "ascii"
                )
                item["image_base64"] = encoded
            serialized_outputs.append(item)

        results.append(
            {
                "filename": filename,
                "status": "ok",
                "outputs": serialized_outputs,
                "metadata": result["metadata"],
                "applied_steps": result["applied_steps"],
                "skipped_steps": result["skipped_steps"],
            }
        )

    return JSONResponse(
        {
            "request_id": request_id,
            "count": len(results),
            "results": results,
        }
    )
