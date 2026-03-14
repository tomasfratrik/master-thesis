import io
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import clip
import torch
from PIL import Image

from backend.config import DEVICE, MODEL_NAME, PREVIEWS_DIR, PREVIEW_LIMIT, PREVIEW_URL_PREFIX


def _format_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def _preview_urls(class_name: str, limit: int = PREVIEW_LIMIT) -> list[str]:
    preview_dir = PREVIEWS_DIR / class_name
    if not preview_dir.exists():
        return []

    urls: list[str] = []
    for path in sorted(preview_dir.iterdir()):
        if not path.is_file():
            continue
        urls.append(f"{PREVIEW_URL_PREFIX}/{quote(class_name)}/{quote(path.name)}")
        if len(urls) >= limit:
            break
    return urls


AggregationMode = Literal["embedding_mean", "logit_mean", "prob_mean"]


class FineTunedSneakerClassifier:
    def __init__(self, checkpoint_path: str | Path, device: torch.device = DEVICE) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device, jit=False)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        class_names = checkpoint.get("class_names")
        if not class_names:
            raise ValueError("Checkpoint does not contain class_names.")

        self.class_names: list[str] = list(class_names)
        self.class_prompts = [
            f"a photo of {_format_class_name(class_name)} sneakers"
            for class_name in self.class_names
        ]
        with torch.no_grad():
            self.text_tokens = clip.tokenize(self.class_prompts).to(self.device)
            text_features = self.model.encode_text(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features.float()

    @torch.no_grad()
    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required for prediction.")

        image_tensors = [
            self.preprocess(image.convert("RGB"))
            for image in images
        ]
        batch = torch.stack(image_tensors, dim=0).to(self.device)
        image_features = self.model.encode_image(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    @staticmethod
    def _aggregate_features(image_features: torch.Tensor) -> torch.Tensor:
        # Represent one sneaker item by averaging all view embeddings, then renormalizing.
        aggregated = image_features.mean(dim=0, keepdim=True)
        return aggregated / aggregated.norm(dim=-1, keepdim=True)

    @staticmethod
    def _validate_aggregation(aggregation: str) -> AggregationMode:
        valid = {"embedding_mean", "logit_mean", "prob_mean"}
        if aggregation not in valid:
            raise ValueError(
                f"Unsupported aggregation mode: {aggregation}. "
                f"Expected one of: {', '.join(sorted(valid))}."
            )
        return aggregation  # type: ignore[return-value]

    def _compute_logits(self, image_features: torch.Tensor) -> torch.Tensor:
        return 100.0 * image_features @ self.text_features.T

    def _build_prediction_result(
        self,
        probabilities: torch.Tensor,
        k: int,
        query_image_count: int,
        aggregation: AggregationMode,
    ) -> dict[str, Any]:
        k = max(1, min(int(k), len(self.class_names)))
        top_scores, top_indices = probabilities.topk(k)

        top_k: list[dict[str, Any]] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            class_name = self.class_names[index]
            top_k.append(
                {
                    "class_name": class_name,
                    "label": _format_class_name(class_name),
                    "prompt": self.class_prompts[index],
                    "score": float(score),
                    "preview_urls": _preview_urls(class_name),
                }
            )

        second_best_score = top_k[1]["score"] if len(top_k) > 1 else 0.0
        return {
            "label": top_k[0]["label"],
            "class_name": top_k[0]["class_name"],
            "score": top_k[0]["score"],
            "margin_vs_second": float(top_k[0]["score"] - second_best_score),
            "query_image_count": int(query_image_count),
            "aggregation": aggregation,
            "preview_urls": top_k[0]["preview_urls"],
            "top_k": top_k,
        }

    @torch.no_grad()
    def predict_images(
        self,
        images: list[Image.Image],
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
    ) -> dict[str, Any]:
        aggregation = self._validate_aggregation(aggregation)
        image_features = self._encode_images(images)
        if aggregation == "embedding_mean":
            aggregated_features = self._aggregate_features(image_features)
            probabilities = self._compute_logits(aggregated_features).softmax(dim=-1)[0]
        else:
            logits = self._compute_logits(image_features)
            if aggregation == "logit_mean":
                probabilities = logits.mean(dim=0).softmax(dim=-1)
            else:
                probabilities = logits.softmax(dim=-1).mean(dim=0)
        return self._build_prediction_result(
            probabilities=probabilities,
            k=k,
            query_image_count=len(images),
            aggregation=aggregation,
        )

    def predict_image(
        self,
        image: Image.Image,
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
    ) -> dict[str, Any]:
        return self.predict_images(images=[image], k=k, aggregation=aggregation)

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
    ) -> dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_image(image=image, k=k, aggregation=aggregation)

    def predict_image_bytes_batch(
        self,
        image_payloads: list[bytes],
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
    ) -> dict[str, Any]:
        images = [Image.open(io.BytesIO(payload)) for payload in image_payloads]
        return self.predict_images(images=images, k=k, aggregation=aggregation)

    def predict_image_path(
        self,
        image_path: str | Path,
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
    ) -> dict[str, Any]:
        image = Image.open(image_path)
        return self.predict_image(image=image, k=k, aggregation=aggregation)

    def predict_image_paths(
        self,
        image_paths: list[str | Path],
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
    ) -> dict[str, Any]:
        images = [Image.open(image_path) for image_path in image_paths]
        return self.predict_images(images=images, k=k, aggregation=aggregation)
