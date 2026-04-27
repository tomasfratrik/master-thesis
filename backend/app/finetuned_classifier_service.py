import io
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import torch
from PIL import Image

from .catalog_metadata import format_class_label
from backend.config import DEVICE, MODEL_NAME, PREVIEWS_DIR, PREVIEW_LIMIT, PREVIEW_URL_PREFIX
from backend.model_loader import load_encoder


def _format_class_name(class_name: str) -> str:
    return format_class_label(class_name)


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

        self.encoder = load_encoder(
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            checkpoint_map_location=self.device,
            checkpoint_strict=False,
        )
        self.model = self.encoder.model
        self.preprocess = self.encoder.preprocess
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        class_names = checkpoint.get("class_names")
        if not class_names:
            raise ValueError("Checkpoint does not contain class_names.")

        self.class_names: list[str] = list(class_names)
        self.class_prompts = [
            f"a photo of {_format_class_name(class_name)} sneakers"
            for class_name in self.class_names
        ]
        with torch.no_grad():
            self.text_tokens = self.encoder.tokenize_texts(self.class_prompts)
            text_features = self.encoder.encode_text_tokens(self.text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features.float()
        self.extra_prototypes: list[dict[str, Any]] = []

    @torch.no_grad()
    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required for prediction.")

        image_tensors = [
            self.preprocess(image.convert("RGB"))
            for image in images
        ]
        batch = torch.stack(image_tensors, dim=0).to(self.device)
        image_features = self.encoder.encode_image_tensors(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    @staticmethod
    def _aggregate_features(image_features: torch.Tensor) -> torch.Tensor:
        # Represent one sneaker item by averaging all view embeddings, then renormalizing.
        aggregated = image_features.mean(dim=0, keepdim=True)
        return aggregated / aggregated.norm(dim=-1, keepdim=True)

    @staticmethod
    def _chunked[T](items: list[T], size: int) -> list[list[T]]:
        return [items[index : index + size] for index in range(0, len(items), size)]

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

    def set_extra_prototypes(self, items: list[dict[str, Any]]) -> None:
        normalized: list[dict[str, Any]] = []
        for item in items:
            feature = item["feature"]
            if not torch.is_tensor(feature):
                feature = torch.tensor(feature, dtype=torch.float32)
            feature = feature.to(self.device).float()
            if feature.ndim > 1:
                feature = feature.squeeze(0)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            normalized.append(
                {
                    "class_name": item["class_name"],
                    "label": item.get("label") or _format_class_name(item["class_name"]),
                    "feature": feature,
                    "preview_urls": item.get("preview_urls") or _preview_urls(item["class_name"]),
                    "candidate_type": item.get("candidate_type", "prototype"),
                }
            )
        self.extra_prototypes = normalized

    def build_prototype_from_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required to build a prototype.")

        feature_chunks: list[torch.Tensor] = []
        for chunk in self._chunked(images, 16):
            feature_chunks.append(self._encode_images(chunk))
        image_features = torch.cat(feature_chunks, dim=0)
        aggregated = self._aggregate_features(image_features)
        return aggregated[0].detach().float()

    def build_prototype_from_image_bytes_batch(self, image_payloads: list[bytes]) -> torch.Tensor:
        images = [Image.open(io.BytesIO(payload)) for payload in image_payloads]
        return self.build_prototype_from_images(images)

    def build_prototype_from_image_paths(self, image_paths: list[str | Path]) -> torch.Tensor:
        if not image_paths:
            raise ValueError("At least one image path is required to build a prototype.")

        feature_chunks: list[torch.Tensor] = []
        for chunk in self._chunked([Path(path) for path in image_paths], 16):
            images: list[Image.Image] = []
            for image_path in chunk:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            feature_chunks.append(self._encode_images(images))
        image_features = torch.cat(feature_chunks, dim=0)
        aggregated = self._aggregate_features(image_features)
        return aggregated[0].detach().float()

    def _candidate_space(
        self,
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        metadata = [
            {
                "class_name": class_name,
                "label": _format_class_name(class_name),
                "prompt": prompt,
                "preview_urls": _preview_urls(class_name),
                "candidate_type": "checkpoint",
            }
            for class_name, prompt in zip(self.class_names, self.class_prompts)
        ]
        features = [self.text_features]

        combined_extras = list(self.extra_prototypes)
        if extra_prototypes:
            shadowed = {item["class_name"] for item in extra_prototypes}
            combined_extras = [
                item for item in combined_extras if item["class_name"] not in shadowed
            ] + extra_prototypes

        if combined_extras:
            prototype_matrix = torch.stack(
                [item["feature"] for item in combined_extras],
                dim=0,
            ).to(self.device)
            features.append(prototype_matrix)
            metadata.extend(
                [
                    {
                        "class_name": item["class_name"],
                        "label": item.get("label") or _format_class_name(item["class_name"]),
                        "prompt": None,
                        "preview_urls": item.get("preview_urls") or _preview_urls(item["class_name"]),
                        "candidate_type": item.get("candidate_type", "prototype"),
                    }
                    for item in combined_extras
                ]
            )

        return torch.cat(features, dim=0), metadata

    def _build_prediction_result(
        self,
        probabilities: torch.Tensor,
        candidate_metadata: list[dict[str, Any]],
        k: int,
        query_image_count: int,
        aggregation: AggregationMode,
    ) -> dict[str, Any]:
        k = max(1, min(int(k), len(candidate_metadata)))
        top_scores, top_indices = probabilities.topk(k)

        top_k: list[dict[str, Any]] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            candidate = candidate_metadata[index]
            top_k.append(
                {
                    "class_name": candidate["class_name"],
                    "label": candidate["label"],
                    "prompt": candidate["prompt"],
                    "score": float(score),
                    "preview_urls": candidate["preview_urls"],
                    "candidate_type": candidate["candidate_type"],
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
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        aggregation = self._validate_aggregation(aggregation)
        image_features = self._encode_images(images)
        candidate_features, candidate_metadata = self._candidate_space(extra_prototypes)
        if aggregation == "embedding_mean":
            aggregated_features = self._aggregate_features(image_features)
            probabilities = (100.0 * aggregated_features @ candidate_features.T).softmax(dim=-1)[0]
        else:
            logits = 100.0 * image_features @ candidate_features.T
            if aggregation == "logit_mean":
                probabilities = logits.mean(dim=0).softmax(dim=-1)
            else:
                probabilities = logits.softmax(dim=-1).mean(dim=0)
        return self._build_prediction_result(
            probabilities=probabilities,
            candidate_metadata=candidate_metadata,
            k=k,
            query_image_count=len(images),
            aggregation=aggregation,
        )

    def predict_image(
        self,
        image: Image.Image,
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.predict_images(
            images=[image],
            k=k,
            aggregation=aggregation,
            extra_prototypes=extra_prototypes,
        )

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_image(
            image=image,
            k=k,
            aggregation=aggregation,
            extra_prototypes=extra_prototypes,
        )

    def predict_image_bytes_batch(
        self,
        image_payloads: list[bytes],
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        images = [Image.open(io.BytesIO(payload)) for payload in image_payloads]
        return self.predict_images(
            images=images,
            k=k,
            aggregation=aggregation,
            extra_prototypes=extra_prototypes,
        )

    def predict_image_path(
        self,
        image_path: str | Path,
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        image = Image.open(image_path)
        return self.predict_image(
            image=image,
            k=k,
            aggregation=aggregation,
            extra_prototypes=extra_prototypes,
        )

    def predict_image_paths(
        self,
        image_paths: list[str | Path],
        k: int = 5,
        aggregation: AggregationMode = "embedding_mean",
        extra_prototypes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        images = [Image.open(image_path) for image_path in image_paths]
        return self.predict_images(
            images=images,
            k=k,
            aggregation=aggregation,
            extra_prototypes=extra_prototypes,
        )
