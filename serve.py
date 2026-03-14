import os
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import PREVIEWS_DIR
from backend.app.finetuned_classifier_service import FineTunedSneakerClassifier
from sneaker_service import SneakerLabelService

app = FastAPI(title="Sneaker Visual Search (Per-Class)")
app.mount("/previews", StaticFiles(directory=str(PREVIEWS_DIR)), name="previews")

# CORS for local Svelte dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load services once if their assets are available.
search_service = None
search_service_error = None
try:
    search_service = SneakerLabelService()
except Exception as error:  # pragma: no cover - startup fallback
    search_service_error = str(error)

checkpoint_classifier = None
checkpoint_classifier_error = None
checkpoint_path = os.getenv("SNEAKER_MODEL_CHECKPOINT")
if checkpoint_path:
    try:
        checkpoint_classifier = FineTunedSneakerClassifier(checkpoint_path=checkpoint_path)
    except Exception as error:  # pragma: no cover - startup fallback
        checkpoint_classifier_error = str(error)


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "search_ready": search_service is not None,
            "search_error": search_service_error,
            "predict_ready": checkpoint_classifier is not None,
            "predict_error": checkpoint_classifier_error,
            "checkpoint_path": checkpoint_path,
            "previews_dir": str(PREVIEWS_DIR),
        }
    )

@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = Query(10, ge=1, le=50),
):
    if search_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Search service is not configured: {search_service_error}",
        )

    b = await file.read()
    prediction = search_service.predict_image_bytes(b, k=k)
    return JSONResponse(prediction)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    k: int = Query(5, ge=1, le=50),
    aggregation: Literal["embedding_mean", "logit_mean", "prob_mean"] = Query("embedding_mean"),
):
    if checkpoint_classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Checkpoint classifier is not configured. "
                "Set SNEAKER_MODEL_CHECKPOINT before starting the API."
            ),
        )

    b = await file.read()
    prediction = checkpoint_classifier.predict_image_bytes(b, k=k, aggregation=aggregation)
    return JSONResponse(prediction)


@app.post("/predict-item")
async def predict_item(
    files: list[UploadFile] = File(...),
    k: int = Query(5, ge=1, le=50),
    aggregation: Literal["embedding_mean", "logit_mean", "prob_mean"] = Query("embedding_mean"),
):
    if checkpoint_classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Checkpoint classifier is not configured. "
                "Set SNEAKER_MODEL_CHECKPOINT before starting the API."
            ),
        )
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    payloads: list[bytes] = []
    filenames: list[str] = []
    for file in files:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail=f"Empty file payload: {file.filename}")
        payloads.append(payload)
        filenames.append(file.filename or "upload")

    prediction = checkpoint_classifier.predict_image_bytes_batch(
        payloads,
        k=k,
        aggregation=aggregation,
    )
    prediction["query_filenames"] = filenames
    return JSONResponse(prediction)
