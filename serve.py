# serve.py
import io, json
from typing import List
import numpy as np
import torch, faiss
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    DEVICE, FAISS_INDEX, CLS_META_JSON
)
from model_loader import load_model

app = FastAPI(title="Sneaker Visual Search (Per-Class)")

# CORS for local Svelte dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load model once
model, preprocess = load_model()

# Load FAISS index + metadata
index = faiss.read_index(str(FAISS_INDEX))
with open(CLS_META_JSON) as f:
    CLS_META = json.load(f)

@torch.no_grad()
def encode_image_bytes(b: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    z = model.encode_image(x)
    z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype("float32")  # [1, D]

@app.post("/search")
async def search(
    file: UploadFile = File(...),
    k: int = Query(10, ge=1, le=50),
):
    b = await file.read()
    q = encode_image_bytes(b)  # [1, D]
    sims, idxs = index.search(q, k)  # inner product
    sims = sims[0].tolist()
    idxs = idxs[0].tolist()

    results = []
    for score, i in zip(sims, idxs):
        if i < 0:  # FAISS returns -1 for empty results in some indexes
            continue
        meta = CLS_META[i]
        results.append({
            "id": meta["class"],
            "preview": meta["rep_path"],
            "count": meta["count"],
            "score": float(score)
        })

    return JSONResponse({"results": results})
