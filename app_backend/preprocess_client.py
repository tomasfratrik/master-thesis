import base64
import json
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib import request

from .config import PREPROCESS_TIMEOUT_SECONDS, PREPROCESS_URL


@dataclass
class PreparedImage:
    original_filename: str
    mime_type: str
    image_bytes: bytes
    source: str


def _encode_multipart(
    files: list[tuple[str, str, bytes, str]],
    fields: list[tuple[str, str]],
) -> tuple[bytes, str]:
    boundary = f"----SneakerBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for field_name, filename, payload, mime_type in files:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(payload)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def preprocess_uploads(
    uploads: list[tuple[str, bytes, str]],
) -> list[PreparedImage]:
    if not uploads:
        return []

    if not PREPROCESS_URL:
        return [
            PreparedImage(
                original_filename=filename,
                image_bytes=payload,
                mime_type=mime_type,
                source="original",
            )
            for filename, payload, mime_type in uploads
        ]

    fields = [("options", json.dumps({"include_images": True}))]
    files = [
        ("files", filename, payload, mime_type)
        for filename, payload, mime_type in uploads
    ]
    body, boundary = _encode_multipart(files=files, fields=fields)
    req = request.Request(
        PREPROCESS_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with request.urlopen(req, timeout=PREPROCESS_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    prepared: list[PreparedImage] = []
    for result in payload.get("results", []):
        outputs = result.get("outputs") or []
        if result.get("status") != "ok" or not outputs:
            continue
        first_output = outputs[0]
        image_base64 = first_output.get("image_base64")
        if not image_base64:
            continue

        mime_type = first_output.get("mime_type") or mimetypes.guess_type(
            result.get("filename") or ""
        )[0] or "application/octet-stream"
        prepared.append(
            PreparedImage(
                original_filename=result.get("filename") or "upload",
                image_bytes=base64.b64decode(image_base64),
                mime_type=mime_type,
                source="preprocessed",
            )
        )

    return prepared
