import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import PREVIEWS_DIR
from .finetuned_classifier_service import FineTunedSneakerClassifier

from .auth import authenticate_user, create_session, create_user, get_current_user
from .config import APP_MEDIA_URL_PREFIX, MODEL_CHECKPOINT, PREVIEW_DIR, UPLOADS_DIR
from .db import get_connection, init_db, utc_now
from .preprocess_client import preprocess_uploads


app = FastAPI(title="Sneaker Catalog App Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(APP_MEDIA_URL_PREFIX, StaticFiles(directory=str(UPLOADS_DIR)), name="app-media")
app.mount("/previews", StaticFiles(directory=str(PREVIEW_DIR)), name="previews")

init_db()

classifier: FineTunedSneakerClassifier | None = None
classifier_error: str | None = None
if MODEL_CHECKPOINT:
    try:
        classifier = FineTunedSneakerClassifier(checkpoint_path=MODEL_CHECKPOINT)
    except Exception as error:  # pragma: no cover - startup fallback
        classifier_error = str(error)
else:
    classifier_error = "SNEAKER_MODEL_CHECKPOINT is not set."


def _catalog_or_404(catalog_id: str, user_id: str) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM catalogs WHERE id = ? AND user_id = ?",
            (catalog_id, user_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Catalog not found.")
    return dict(row)


def _store_file(item_id: str, role: str, filename: str, payload: bytes) -> str:
    item_dir = UPLOADS_DIR / item_id / role
    item_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "upload.bin"
    target = item_dir / f"{uuid.uuid4().hex}_{safe_name}"
    target.write_bytes(payload)
    return str(target.relative_to(UPLOADS_DIR).as_posix())


def _item_images(item_id: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT image_role, original_filename, mime_type, stored_path
            FROM item_images
            WHERE item_id = ?
            ORDER BY created_at ASC
            """,
            (item_id,),
        ).fetchall()

    return [
        {
            "image_role": row["image_role"],
            "original_filename": row["original_filename"],
            "mime_type": row["mime_type"],
            "url": f"{APP_MEDIA_URL_PREFIX}/{row['stored_path']}",
        }
        for row in rows
    ]


def _serialize_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "catalog_id": row["catalog_id"],
        "title": row["title"],
        "notes": row["notes"],
        "predicted_class_name": row["predicted_class_name"],
        "predicted_label": row["predicted_label"],
        "predicted_score": row["predicted_score"],
        "margin_vs_second": row["margin_vs_second"],
        "query_image_count": row["query_image_count"],
        "aggregation": row["aggregation"],
        "created_at": row["created_at"],
        "images": _item_images(row["id"]),
        "prediction": json.loads(row["raw_prediction_json"]) if row["raw_prediction_json"] else None,
    }


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "predict_ready": classifier is not None,
            "predict_error": classifier_error,
            "checkpoint_path": MODEL_CHECKPOINT,
            "previews_dir": str(PREVIEWS_DIR),
            "preprocess_mode": "local",
        }
    )


@app.post("/auth/register")
async def register(payload: dict[str, str]) -> JSONResponse:
    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    user = create_user(email=email, password=password)
    token = create_session(user["id"])
    return JSONResponse({"user": user, "token": token})


@app.post("/auth/login")
async def login(payload: dict[str, str]) -> JSONResponse:
    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""
    user = authenticate_user(email=email, password=password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_session(user["id"])
    return JSONResponse({"user": {"id": user["id"], "email": user["email"]}, "token": token})


@app.get("/me")
async def me(current_user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    return JSONResponse({"id": current_user["id"], "email": current_user["email"]})


@app.get("/catalogs")
async def list_catalogs(current_user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, name, description, created_at
            FROM catalogs
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (current_user["id"],),
        ).fetchall()

    return JSONResponse({"catalogs": [dict(row) for row in rows]})


@app.post("/catalogs")
async def create_catalog(
    payload: dict[str, str],
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    name = (payload.get("name") or "").strip()
    description = payload.get("description")
    if not name:
        raise HTTPException(status_code=400, detail="Catalog name is required.")

    catalog_id = str(uuid.uuid4())
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO catalogs (id, user_id, name, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (catalog_id, current_user["id"], name, description, utc_now()),
        )

    return JSONResponse(
        {
            "id": catalog_id,
            "name": name,
            "description": description,
        }
    )


@app.get("/catalogs/{catalog_id}/items")
async def list_catalog_items(
    catalog_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    _catalog_or_404(catalog_id=catalog_id, user_id=current_user["id"])

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM catalog_items
            WHERE catalog_id = ?
            ORDER BY created_at DESC
            """,
            (catalog_id,),
        ).fetchall()

    return JSONResponse({"items": [_serialize_item(dict(row)) for row in rows]})


@app.get("/catalogs/{catalog_id}/items/{item_id}")
async def get_catalog_item(
    catalog_id: str,
    item_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    _catalog_or_404(catalog_id=catalog_id, user_id=current_user["id"])

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM catalog_items
            WHERE id = ? AND catalog_id = ?
            """,
            (item_id, catalog_id),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Catalog item not found.")

    return JSONResponse(_serialize_item(dict(row)))


@app.post("/catalogs/{catalog_id}/analyze-item")
async def analyze_catalog_item(
    catalog_id: str,
    files: list[UploadFile] = File(...),
    title: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    aggregation: Literal["embedding_mean", "logit_mean", "prob_mean"] = Query("logit_mean"),
    top_k: int = Query(5, ge=1, le=50),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    if classifier is None:
        raise HTTPException(status_code=503, detail=f"Predictor unavailable: {classifier_error}")
    _catalog_or_404(catalog_id=catalog_id, user_id=current_user["id"])

    uploads: list[tuple[str, bytes, str]] = []
    for file in files:
        payload = await file.read()
        if not payload:
            continue
        mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
        uploads.append((file.filename or "upload", payload, mime_type))

    if not uploads:
        raise HTTPException(status_code=400, detail="At least one non-empty file is required.")

    prepared_images = preprocess_uploads(uploads)
    if not prepared_images:
        raise HTTPException(status_code=422, detail="Preprocess step produced no usable images.")

    prediction = classifier.predict_image_bytes_batch(
        [item.image_bytes for item in prepared_images],
        k=top_k,
        aggregation=aggregation,
    )
    prediction["query_filenames"] = [filename for filename, _, _ in uploads]

    item_id = str(uuid.uuid4())
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO catalog_items (
                id, catalog_id, title, notes,
                predicted_class_name, predicted_label, predicted_score,
                margin_vs_second, query_image_count, aggregation,
                raw_prediction_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                catalog_id,
                title,
                notes,
                prediction["class_name"],
                prediction["label"],
                prediction["score"],
                prediction["margin_vs_second"],
                prediction["query_image_count"],
                prediction["aggregation"],
                json.dumps(prediction),
                utc_now(),
            ),
        )

        for filename, payload, mime_type in uploads:
            stored_path = _store_file(item_id=item_id, role="original", filename=filename, payload=payload)
            connection.execute(
                """
                INSERT INTO item_images (id, item_id, image_role, original_filename, mime_type, stored_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    item_id,
                    "original",
                    filename,
                    mime_type,
                    stored_path,
                    utc_now(),
                ),
            )

        for index, prepared in enumerate(prepared_images, start=1):
            filename = f"processed_{index:02d}_{Path(prepared.original_filename).name}"
            stored_path = _store_file(
                item_id=item_id,
                role="processed",
                filename=filename,
                payload=prepared.image_bytes,
            )
            connection.execute(
                """
                INSERT INTO item_images (id, item_id, image_role, original_filename, mime_type, stored_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    item_id,
                    prepared.source,
                    filename,
                    prepared.mime_type,
                    stored_path,
                    utc_now(),
                ),
            )

        row = connection.execute(
            "SELECT * FROM catalog_items WHERE id = ?",
            (item_id,),
        ).fetchone()

    return JSONResponse(_serialize_item(dict(row)))
