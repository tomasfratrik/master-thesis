import io
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image

from config import CLS_META_JSON, DEVICE, FAISS_INDEX
from model_loader import load_model


class SneakerLabelService:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX,
        meta_path: Path = CLS_META_JSON,
        use_checkpoint: bool | None = None,
    ) -> None:
        self.model, self.preprocess = load_model(use_checkpoint=use_checkpoint)
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if self.index.ntotal != len(self.meta):
            raise ValueError(
                "FAISS index size and class metadata size do not match: "
                f"{self.index.ntotal} != {len(self.meta)}"
            )

    @staticmethod
    def _extract_label(meta: dict[str, Any]) -> str:
        return str(meta.get("label") or meta.get("class") or meta.get("id") or "unknown")

    @staticmethod
    def _extract_id(meta: dict[str, Any]) -> str:
        return str(meta.get("id") or meta.get("class") or meta.get("label") or "unknown")

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> np.ndarray:
        x = self.preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        z = self.model.encode_image(x)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy().astype("float32")

    def predict_image(self, image: Image.Image, k: int = 1) -> dict[str, Any]:
        k = max(1, min(int(k), len(self.meta)))
        q = self._encode_image(image)
        scores, indices = self.index.search(q, k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        top_k: list[dict[str, Any]] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            item = self.meta[idx]
            top_k.append({
                "id": self._extract_id(item),
                "label": self._extract_label(item),
                "score": float(score),
                "count": int(item.get("count", 0)),
                "preview": item.get("rep_path"),
            })

        if not top_k:
            raise RuntimeError("No results returned from FAISS index.")

        return {
            "label": top_k[0]["label"],
            "score": top_k[0]["score"],
            "top_k": top_k,
        }

    def predict_image_bytes(self, image_bytes: bytes, k: int = 1) -> dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_image(image=image, k=k)

    def predict_image_path(self, image_path: str | Path, k: int = 1) -> dict[str, Any]:
        image = Image.open(image_path)
        return self.predict_image(image=image, k=k)
