from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from backend.config import DEVICE
from backend.model_loader import VisionLanguageEncoder, load_encoder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def format_class_label(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def list_labeled_images(root: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append((image_path, class_name))
    return items


def load_checkpoint_class_names(checkpoint_path: Path) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"Checkpoint does not contain class_names: {checkpoint_path}")
    return list(class_names)


class EvaluationImageEncoder:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        backend: str | None = None,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: torch.device = DEVICE,
    ) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.encoder: VisionLanguageEncoder = load_encoder(
            backend=backend,
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            checkpoint_map_location=self.device,
            checkpoint_strict=False,
            use_checkpoint=False if self.checkpoint_path is None else None,
        )
        self.preprocess = self.encoder.preprocess
        self.backend = self.encoder.backend
        self.model_name = self.encoder.model_name
        self.pretrained = getattr(self.encoder, "pretrained", pretrained)

    @staticmethod
    def _chunked[T](items: list[T], size: int) -> list[list[T]]:
        return [items[index : index + size] for index in range(0, len(items), size)]

    @torch.no_grad()
    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required.")

        image_tensors = [self.preprocess(image.convert("RGB")) for image in images]
        batch = torch.stack(image_tensors, dim=0).to(self.device)
        image_features = self.encoder.encode_image_tensors(batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()

    @staticmethod
    def _aggregate_features(image_features: torch.Tensor) -> torch.Tensor:
        aggregated = image_features.mean(dim=0, keepdim=True)
        return aggregated / aggregated.norm(dim=-1, keepdim=True)

    def build_prototype_from_images(self, images: list[Image.Image]) -> torch.Tensor:
        feature_chunks: list[torch.Tensor] = []
        for chunk in self._chunked(images, 16):
            feature_chunks.append(self._encode_images(chunk))
        image_features = torch.cat(feature_chunks, dim=0)
        aggregated = self._aggregate_features(image_features)
        return aggregated[0].detach().float()

    def build_prototype_from_image_paths(self, image_paths: list[str | Path]) -> torch.Tensor:
        if not image_paths:
            raise ValueError("At least one image path is required.")

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

    def model_summary(self) -> dict[str, str | None]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
        }


class ZeroShotSneakerClassifier(EvaluationImageEncoder):
    def __init__(
        self,
        *,
        class_names: list[str],
        prompt_template: str = "a photo of {label} sneakers",
        checkpoint_path: str | Path | None = None,
        backend: str | None = None,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            backend=backend,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
        self.class_names = list(class_names)
        self.prompt_template = prompt_template
        self.class_prompts = [
            prompt_template.format(class_name=class_name, label=format_class_label(class_name))
            for class_name in self.class_names
        ]
        with torch.no_grad():
            text_tokens = self.encoder.tokenize_texts(self.class_prompts)
            text_features = self.encoder.encode_text_tokens(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features.float()

    @torch.no_grad()
    def predict_image_path(
        self,
        image_path: Path,
        *,
        top_k: int,
    ) -> tuple[list[dict[str, float | str]], float]:
        with Image.open(image_path) as image:
            image_features = self._encode_images([image.convert("RGB")])

        probabilities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[0]
        k = max(1, min(int(top_k), len(self.class_names)))
        top_scores, top_indices = probabilities.topk(k)

        top_predictions: list[dict[str, float | str]] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            class_name = self.class_names[index]
            top_predictions.append(
                {
                    "class_name": class_name,
                    "label": format_class_label(class_name),
                    "score": float(score),
                }
            )

        second_best_score = float(top_predictions[1]["score"]) if len(top_predictions) > 1 else 0.0
        margin = float(top_predictions[0]["score"]) - second_best_score
        return top_predictions, margin
