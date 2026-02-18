# serve.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sneaker_service import SneakerLabelService

app = FastAPI(title="Sneaker Visual Search (Per-Class)")

# CORS for local Svelte dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load model + search assets once
service = SneakerLabelService()

@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = Query(10, ge=1, le=50),
):
    b = await file.read()
    prediction = service.predict_image_bytes(b, k=k)
    return JSONResponse(prediction)
