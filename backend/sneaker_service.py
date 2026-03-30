import io
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image

from backend.config import CLS_META_JSON, DEVICE, FAISS_INDEX, MODEL_CHECKPOINT, MODEL_USE_CHECKPOINT
from backend.model_loader import load_encoder


class SneakerLabelService:
    def __init__(
        self,
        index_path: Path = FAISS_INDEX,
        meta_path: Path = CLS_META_JSON,
        use_checkpoint: bool | None = None,
    ) -> None:
        self.use_checkpoint = MODEL_USE_CHECKPOINT if use_checkpoint is None else use_checkpoint
        self.encoder = load_encoder(use_checkpoint=use_checkpoint)
        self.model = self.encoder.model
        self.preprocess = self.encoder.preprocess
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        if self.index.ntotal != len(self.meta):
            raise ValueError(
                "FAISS index size and class metadata size do not match: "
                f"{self.index.ntotal} != {len(self.meta)}"
            )

        if self.use_checkpoint:
            self._validate_checkpoint_labels()

    @staticmethod
    def _normalize_label(value: str) -> str:
        return "".join(ch.lower() for ch in value if ch.isalnum())

    def _validate_checkpoint_labels(self) -> None:
        if MODEL_CHECKPOINT is None:
            raise ValueError(
                "Checkpoint retrieval requested but SNEAKER_MODEL_CHECKPOINT is not set."
            )

        checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu")
        checkpoint_classes = checkpoint.get("class_names")
        if not checkpoint_classes:
            raise ValueError(
                f"Checkpoint does not contain class_names: {MODEL_CHECKPOINT}"
            )

        index_classes = [self._extract_id(item) for item in self.meta]
        normalized_checkpoint = {self._normalize_label(name) for name in checkpoint_classes}
        normalized_index = {self._normalize_label(name) for name in index_classes}

        if normalized_checkpoint != normalized_index:
            checkpoint_only = sorted(set(checkpoint_classes) - set(index_classes))[:5]
            index_only = sorted(set(index_classes) - set(checkpoint_classes))[:5]
            raise ValueError(
                "Checkpoint class labels do not match the FAISS index metadata. "
                "This usually means the checkpoint was trained on a different dataset than "
                "the retrieval index. Use `python -m backend.predict_finetuned` for checkpoint "
                "classification, or rebuild the embeddings and index with "
                "`python -m backend.embded --use-checkpoint` followed by "
                "`python -m backend.index`. "
                f"Checkpoint-only examples: {checkpoint_only or 'none'}. "
                f"Index-only examples: {index_only or 'none'}."
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
        z = self.encoder.encode_image_tensors(x)
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
