import base64
import json
import mimetypes
import uuid
from pathlib import Path
from typing import cast
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import PREVIEWS_DIR, TEST_SPLIT_ROOT, TRAIN_SPLIT_ROOT, VAL_SPLIT_ROOT
from .embedding_retrieval import CatalogEmbeddingRetrieval
from .finetuned_classifier_service import FineTunedSneakerClassifier

from .auth import (
    authenticate_user,
    create_session,
    create_user,
    ensure_demo_users,
    get_current_user,
    require_admin,
)
from .catalog_metadata import build_brand_prefix_map, format_class_label, normalize_class_name
from .config import APP_MEDIA_URL_PREFIX, MODEL_CHECKPOINT, PREVIEW_DIR, UPLOADS_DIR
from .db import get_connection, init_db, utc_now
from .preprocess_client import PreparedImage, prepare_uploads_without_preprocess, preprocess_uploads
from .prototype_catalog import ensure_catalog_embeddings, load_classifier_prototypes, upsert_prototype_class
from .training_jobs import (
    accept_training_job,
    create_training_job,
    get_active_checkpoint_path,
    get_training_job,
    initialize_training_jobs,
    list_training_jobs,
    start_training_job,
)


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
ensure_demo_users()
initialize_training_jobs()

classifier: FineTunedSneakerClassifier | None = None
classifier_error: str | None = None
retrieval_index: CatalogEmbeddingRetrieval | None = None
active_checkpoint = get_active_checkpoint_path()
if active_checkpoint:
    try:
        classifier = FineTunedSneakerClassifier(checkpoint_path=active_checkpoint)
        print("[startup] Preparing retrieval catalog embeddings...", flush=True)
        ensure_catalog_embeddings(classifier)
        retrieval_index = CatalogEmbeddingRetrieval(classifier)
        print(
            f"[startup] Retrieval index ready with {retrieval_index.catalog_size} classes "
            f"across {retrieval_index.row_count} embedding rows.",
            flush=True,
        )
    except Exception as error:  # pragma: no cover - startup fallback
        classifier_error = str(error)
else:
    classifier_error = "SNEAKER_MODEL_CHECKPOINT is not set."


def _reload_classifier(checkpoint_path: str) -> None:
    global classifier, classifier_error, retrieval_index
    classifier = FineTunedSneakerClassifier(checkpoint_path=Path(checkpoint_path))
    print("[startup] Refreshing retrieval catalog embeddings after classifier reload...", flush=True)
    ensure_catalog_embeddings(classifier)
    retrieval_index = CatalogEmbeddingRetrieval(classifier)
    print(
        f"[startup] Retrieval index ready with {retrieval_index.catalog_size} classes "
        f"across {retrieval_index.row_count} embedding rows.",
        flush=True,
    )
    classifier_error = None


def _refresh_retrieval_index() -> None:
    global retrieval_index
    if classifier is None:
        retrieval_index = None
        return
    print("[retrieval] Refreshing retrieval index...", flush=True)
    ensure_catalog_embeddings(classifier)
    if retrieval_index is None:
        retrieval_index = CatalogEmbeddingRetrieval(classifier)
    else:
        retrieval_index.refresh()
    print(
        f"[retrieval] Retrieval index refreshed with {retrieval_index.catalog_size} classes "
        f"across {retrieval_index.row_count} embedding rows.",
        flush=True,
    )


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


def _admin_user_counts() -> int:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM users WHERE role = 'admin'"
        ).fetchone()
    return int(row["count"]) if row is not None else 0


def _prepared_image_payload(prepared: PreparedImage) -> dict[str, str]:
    encoded = base64.b64encode(prepared.image_bytes).decode("ascii")
    return {
        "input_filename": prepared.input_filename,
        "filename": prepared.original_filename,
        "mime_type": prepared.mime_type,
        "source": prepared.source,
        "data_url": f"data:{prepared.mime_type};base64,{encoded}",
    }


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes, str]]:
    uploads: list[tuple[str, bytes, str]] = []
    for file in files:
        payload = await file.read()
        if not payload:
            continue
        mime_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
        uploads.append((file.filename or "upload", payload, mime_type))
    return uploads


def _prepare_prediction_payload(
    uploads: list[tuple[str, bytes, str]],
    *,
    top_k: int,
    mode: Literal["grouped", "per_image"],
    aggregation: Literal["embedding_mean", "logit_mean", "prob_mean"],
    prepared_images: list[PreparedImage] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if classifier is None:
        raise HTTPException(status_code=503, detail=f"Predictor unavailable: {classifier_error}")
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one non-empty file is required.")

    if prepared_images is None:
        preprocess_outcome = preprocess_uploads(uploads)
        prepared_images = preprocess_outcome.images
        warnings = preprocess_outcome.warnings
    else:
        warnings = warnings or []

    if not prepared_images:
        raise HTTPException(status_code=422, detail="Preprocess step produced no usable images.")

    if mode == "grouped":
        result = classifier.predict_image_bytes_batch(
            [item.image_bytes for item in prepared_images],
            k=top_k,
            aggregation=aggregation,
        )
        result["query_filenames"] = [filename for filename, _, _ in uploads]
        result["prepared_sources"] = [item.source for item in prepared_images]
        return {
            "mode": mode,
            "top_k": top_k,
            "query_image_count": len(uploads),
            "processed_image_count": len(prepared_images),
            "aggregation": aggregation,
            "warnings": warnings,
            "processed_images": [_prepared_image_payload(item) for item in prepared_images],
            "result": result,
        }

    results: list[dict[str, Any]] = []
    for prepared in prepared_images:
        prediction = classifier.predict_image_bytes(
            prepared.image_bytes,
            k=top_k,
            aggregation="embedding_mean",
        )
        prediction["query_filename"] = prepared.input_filename
        prediction["processed_filename"] = prepared.original_filename
        prediction["prepared_source"] = prepared.source
        prediction["processed_image"] = _prepared_image_payload(prepared)
        results.append(prediction)

    return {
        "mode": mode,
        "top_k": top_k,
        "query_image_count": len(uploads),
        "processed_image_count": len(prepared_images),
        "warnings": warnings,
        "results": results,
    }


def _prepare_retrieval_payload(
    uploads: list[tuple[str, bytes, str]],
    *,
    top_k: int,
    mode: Literal["grouped", "per_image"],
    prepared_images: list[PreparedImage] | None = None,
    warnings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if classifier is None or retrieval_index is None:
        raise HTTPException(status_code=503, detail=f"Retrieval index unavailable: {classifier_error}")
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one non-empty file is required.")

    if prepared_images is None:
        preprocess_outcome = preprocess_uploads(uploads)
        prepared_images = preprocess_outcome.images
        warnings = preprocess_outcome.warnings
    else:
        warnings = warnings or []

    if not prepared_images:
        raise HTTPException(status_code=422, detail="Preprocess step produced no usable images.")

    if mode == "grouped":
        result = retrieval_index.search_image_bytes_batch(
            [item.image_bytes for item in prepared_images],
            k=top_k,
        )
        result["query_filenames"] = [filename for filename, _, _ in uploads]
        result["prepared_sources"] = [item.source for item in prepared_images]
        return {
            "analysis_method": "retrieval",
            "mode": mode,
            "top_k": top_k,
            "query_image_count": len(uploads),
            "processed_image_count": len(prepared_images),
            "aggregation": "embedding_mean",
            "warnings": warnings,
            "processed_images": [_prepared_image_payload(item) for item in prepared_images],
            "result": result,
        }

    results: list[dict[str, Any]] = []
    for prepared in prepared_images:
        prediction = retrieval_index.search_image_bytes(
            prepared.image_bytes,
            k=top_k,
        )
        prediction["query_filename"] = prepared.input_filename
        prediction["processed_filename"] = prepared.original_filename
        prediction["prepared_source"] = prepared.source
        prediction["processed_image"] = _prepared_image_payload(prepared)
        results.append(prediction)

    return {
        "analysis_method": "retrieval",
        "mode": mode,
        "top_k": top_k,
        "query_image_count": len(uploads),
        "processed_image_count": len(prepared_images),
        "warnings": warnings,
        "results": results,
    }


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "predict_ready": classifier is not None,
            "predict_error": classifier_error,
            "checkpoint_path": get_active_checkpoint_path(),
            "previews_dir": str(PREVIEWS_DIR),
            "preprocess_mode": "local",
            "retrieval_ready": retrieval_index is not None,
            "retrieval_catalog_size": retrieval_index.catalog_size if retrieval_index is not None else 0,
        }
    )


@app.get("/supported-sneakers")
async def supported_sneakers() -> JSONResponse:
    if classifier is None:
        raise HTTPException(status_code=503, detail=f"Predictor unavailable: {classifier_error}")

    with get_connection() as connection:
        metadata_rows = connection.execute(
            """
            SELECT class_name, display_name, brand
            FROM catalog_products
            ORDER BY brand ASC, display_name ASC
            """
        ).fetchall()

    grouped: dict[str, list[dict[str, str]]] = {}
    items: list[dict[str, str]] = []
    if metadata_rows:
        for row in metadata_rows:
            brand = row["brand"] or "Other"
            item = {
                "class_name": row["class_name"],
                "label": row["display_name"],
                "brand": brand,
            }
            items.append(item)
            grouped.setdefault(brand, []).append(item)
    else:
        brand_map = build_brand_prefix_map(sorted(classifier.class_names))
        for class_name in sorted(classifier.class_names):
            brand = brand_map[class_name]
            item = {
                "class_name": class_name,
                "label": format_class_label(class_name),
                "brand": brand,
            }
            items.append(item)
            grouped.setdefault(brand, []).append(item)

    brands = [
        {"brand": brand, "count": len(grouped[brand])}
        for brand in sorted(grouped)
    ]

    return JSONResponse(
        {
            "count": len(items),
            "items": items,
            "brands": brands,
            "groups": {brand: grouped[brand] for brand in sorted(grouped)},
        }
    )


@app.post("/analyze")
async def analyze_public(
    files: list[UploadFile] = File(...),
    mode: Literal["grouped", "per_image"] = Query("grouped"),
    aggregation: Literal["embedding_mean", "logit_mean", "prob_mean"] = Query("logit_mean"),
    top_k: int = Query(5, ge=1, le=10),
) -> JSONResponse:
    uploads = await _read_uploads(files)
    payload = _prepare_prediction_payload(
        uploads,
        top_k=top_k,
        mode=mode,
        aggregation=aggregation,
    )
    payload["analysis_method"] = "classifier"
    return JSONResponse(payload)


@app.post("/analyze-retrieval")
async def analyze_retrieval(
    files: list[UploadFile] = File(...),
    mode: Literal["grouped", "per_image"] = Query("grouped"),
    top_k: int = Query(5, ge=1, le=10),
) -> JSONResponse:
    uploads = await _read_uploads(files)
    payload = _prepare_retrieval_payload(
        uploads,
        top_k=top_k,
        mode=mode,
    )
    return JSONResponse(payload)


@app.post("/auth/register")
async def register(payload: dict[str, str]) -> JSONResponse:
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    full_name = (payload.get("full_name") or "").strip() or None
    email = (payload.get("email") or "").strip() or None
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    user = create_user(username=username, password=password, full_name=full_name, email=email)
    token = create_session(user["id"])
    return JSONResponse({"user": user, "token": token})


@app.post("/auth/login")
async def login(payload: dict[str, str]) -> JSONResponse:
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    user = authenticate_user(username=username, password=password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_session(user["id"])
    return JSONResponse({"user": user, "token": token})


@app.get("/me")
async def me(current_user: dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    return JSONResponse(current_user)


@app.get("/admin/users")
async def admin_list_users(current_user: dict[str, Any] = Depends(require_admin)) -> JSONResponse:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                users.id,
                users.username,
                users.full_name,
                users.email,
                users.role,
                users.created_at,
                COUNT(DISTINCT catalogs.id) AS catalog_count,
                COUNT(DISTINCT catalog_items.id) AS item_count
            FROM users
            LEFT JOIN catalogs ON catalogs.user_id = users.id
            LEFT JOIN catalog_items ON catalog_items.catalog_id = catalogs.id
            GROUP BY users.id
            ORDER BY
                CASE users.role WHEN 'admin' THEN 0 ELSE 1 END,
                users.username ASC
            """
        ).fetchall()

    return JSONResponse(
        {
            "users": [
                {
                    **dict(row),
                    "is_current_user": row["id"] == current_user["id"],
                }
                for row in rows
            ]
        }
    )


@app.patch("/admin/users/{user_id}")
async def admin_update_user(
    user_id: str,
    payload: dict[str, str],
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    role = (payload.get("role") or "").strip().lower()
    if role not in {"user", "admin"}:
        raise HTTPException(status_code=400, detail="Role must be either 'user' or 'admin'.")

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, username, full_name, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found.")

        if row["role"] == "admin" and role != "admin" and _admin_user_counts() <= 1:
            raise HTTPException(status_code=400, detail="At least one admin account must remain.")

        connection.execute(
            "UPDATE users SET role = ? WHERE id = ?",
            (role, user_id),
        )

        updated = connection.execute(
            "SELECT id, username, full_name, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    return JSONResponse(dict(updated))


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(
    user_id: str,
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    if user_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own admin account.")

    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="User not found.")

        if row["role"] == "admin" and _admin_user_counts() <= 1:
            raise HTTPException(status_code=400, detail="At least one admin account must remain.")

        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))

    return JSONResponse({"deleted": True, "id": user_id})


@app.get("/admin/training-jobs")
async def admin_list_training_jobs(
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    return JSONResponse(
        {
            "active_checkpoint_path": get_active_checkpoint_path(),
            "jobs": list_training_jobs(),
        }
    )


@app.get("/admin/training-jobs/{job_id}")
async def admin_get_training_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    job = get_training_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return JSONResponse(job)


@app.post("/admin/training-jobs")
async def admin_create_training_job(
    brand: str = Form(...),
    display_name: str = Form(...),
    class_name: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    top_k: int = Form(5),
    required_topk_accuracy: float = Form(0.90),
    required_new_class_topk_accuracy: float = Form(0.90),
    train_files: list[UploadFile] | None = File(default=None),
    test_files: list[UploadFile] | None = File(default=None),
    preview_files: list[UploadFile] | None = File(default=None),
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    brand = brand.strip()
    display_name = display_name.strip()
    if not brand or not display_name:
        raise HTTPException(status_code=400, detail="Brand and display name are required.")

    resolved_class_name = normalize_class_name(class_name or f"{brand}_{display_name}")
    train_uploads = await _read_uploads(cast(list[UploadFile], train_files or []))
    test_uploads = await _read_uploads(cast(list[UploadFile], test_files or []))
    preview_uploads = await _read_uploads(cast(list[UploadFile], preview_files or []))

    try:
        job_id = create_training_job(
            created_by_user_id=current_user["id"],
            brand=brand,
            display_name=display_name,
            class_name=resolved_class_name,
            notes=notes,
            train_uploads=train_uploads,
            test_uploads=test_uploads,
            preview_uploads=preview_uploads,
            top_k=top_k,
            required_topk_accuracy=required_topk_accuracy,
            required_new_class_topk_accuracy=required_new_class_topk_accuracy,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    job = get_training_job(job_id)
    return JSONResponse(job, status_code=201)


@app.post("/admin/training-jobs/{job_id}/start")
async def admin_start_training_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    job = get_training_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")

    try:
        start_training_job(
            job_id=job_id,
            train_root=TRAIN_SPLIT_ROOT,
            val_root=VAL_SPLIT_ROOT if VAL_SPLIT_ROOT.exists() else None,
            test_root=TEST_SPLIT_ROOT,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    updated = get_training_job(job_id)
    return JSONResponse(updated or {"started": True, "id": job_id})


@app.post("/admin/training-jobs/{job_id}/accept")
async def admin_accept_training_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    try:
        job = accept_training_job(job_id)
        if job.get("activated_checkpoint_path"):
            _reload_classifier(job["activated_checkpoint_path"])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {error}") from error

    return JSONResponse(job)


@app.post("/admin/prototype-classes")
async def admin_create_prototype_class(
    brand: str = Form(...),
    display_name: str = Form(...),
    class_name: str | None = Form(default=None),
    notes: str | None = Form(default=None),
    top_k: int = Form(5),
    skip_preprocess: bool = Form(False),
    reference_files: list[UploadFile] | None = File(default=None),
    test_files: list[UploadFile] | None = File(default=None),
    preview_files: list[UploadFile] | None = File(default=None),
    current_user: dict[str, Any] = Depends(require_admin),
) -> JSONResponse:
    del current_user
    if classifier is None or retrieval_index is None:
        raise HTTPException(status_code=503, detail=f"Predictor unavailable: {classifier_error}")

    brand = brand.strip()
    display_name = display_name.strip()
    if not brand or not display_name:
        raise HTTPException(status_code=400, detail="Brand and display name are required.")

    resolved_class_name = normalize_class_name(class_name or f"{brand}_{display_name}")
    existing_metadata = [
        item["class_name"]
        for item in load_classifier_prototypes()
    ]
    if resolved_class_name in classifier.class_names or resolved_class_name in existing_metadata:
        raise HTTPException(status_code=409, detail="Class name already exists.")

    reference_uploads = await _read_uploads(cast(list[UploadFile], reference_files or []))
    test_uploads = await _read_uploads(cast(list[UploadFile], test_files or []))
    preview_uploads = await _read_uploads(cast(list[UploadFile], preview_files or []))

    if not reference_uploads:
        raise HTTPException(status_code=400, detail="At least one reference image is required.")
    if not preview_uploads:
        preview_uploads = list(reference_uploads)

    reference_preprocessed = (
        prepare_uploads_without_preprocess(reference_uploads)
        if skip_preprocess
        else preprocess_uploads(reference_uploads)
    )
    reference_images = reference_preprocessed.images
    if not reference_images:
        raise HTTPException(status_code=422, detail="Reference images produced no usable crops.")

    prototype_feature = classifier.build_prototype_from_image_bytes_batch(
        [item.image_bytes for item in reference_images]
    )
    temp_prototype = {
        "class_name": resolved_class_name,
        "label": display_name,
        "feature": prototype_feature,
        "candidate_type": "catalog_embedding",
        "brand": brand,
        "model": display_name,
    }

    evaluation_items: list[dict[str, Any]] = []
    top1_correct = 0
    topk_correct = 0
    processed_test_images = []
    evaluation_warnings = list(reference_preprocessed.warnings)
    if test_uploads:
        test_preprocessed = (
            prepare_uploads_without_preprocess(test_uploads)
            if skip_preprocess
            else preprocess_uploads(test_uploads)
        )
        evaluation_warnings.extend(test_preprocessed.warnings)
        processed_test_images = [_prepared_image_payload(item) for item in test_preprocessed.images]
        for item in test_preprocessed.images:
            prediction = retrieval_index.search_image_bytes(
                item.image_bytes,
                k=top_k,
                extra_entries=[temp_prototype],
            )
            top_classes = [candidate["class_name"] for candidate in prediction["top_k"]]
            is_top1 = prediction["class_name"] == resolved_class_name
            is_topk = resolved_class_name in top_classes
            top1_correct += int(is_top1)
            topk_correct += int(is_topk)
            evaluation_items.append(
                {
                    "input_filename": item.input_filename,
                    "processed_filename": item.original_filename,
                    "is_top1": is_top1,
                    "is_topk": is_topk,
                    "prediction": prediction,
                }
            )

    evaluation_summary = {
        "test_image_count": len(evaluation_items),
        "top1_accuracy": (top1_correct / len(evaluation_items)) if evaluation_items else None,
        f"top{top_k}_accuracy": (topk_correct / len(evaluation_items)) if evaluation_items else None,
    }

    product_id = upsert_prototype_class(
        class_name=resolved_class_name,
        display_name=display_name,
        brand=brand,
        notes=notes,
        prototype_embedding=prototype_feature.detach().cpu().tolist(),
        reference_uploads=reference_uploads,
        test_uploads=test_uploads,
        preview_uploads=preview_uploads,
        evaluation_summary=evaluation_summary,
    )
    _refresh_retrieval_index()
    return JSONResponse(
        {
            "saved": True,
            "product_id": product_id,
            "class_name": resolved_class_name,
            "display_name": display_name,
            "brand": brand,
            "activated_in_retrieval": True,
            "activated_in_default_matcher": False,
            "skip_preprocess": skip_preprocess,
            "reference_image_count": len(reference_images),
            "processed_reference_images": [_prepared_image_payload(item) for item in reference_images],
            "processed_test_images": processed_test_images,
            "warnings": evaluation_warnings,
            "evaluation": {
                "summary": evaluation_summary,
                "results": evaluation_items,
            },
        }
    )


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

    uploads = await _read_uploads(files)
    preprocess_outcome = preprocess_uploads(uploads)
    prepared_images = preprocess_outcome.images
    payload = _prepare_prediction_payload(
        uploads,
        top_k=top_k,
        mode="grouped",
        aggregation=aggregation,
        prepared_images=prepared_images,
        warnings=preprocess_outcome.warnings,
    )
    prediction = payload["result"]

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
