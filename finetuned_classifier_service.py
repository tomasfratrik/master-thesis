import io
from pathlib import Path
from typing import Any
from urllib.parse import quote

import clip
import torch
from PIL import Image

from config import DEVICE, MODEL_NAME, PREVIEWS_DIR, PREVIEW_LIMIT, PREVIEW_URL_PREFIX


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
    def predict_image(self, image: Image.Image, k: int = 5) -> dict[str, Any]:
        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()

        probabilities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)[0]
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

        return {
            "label": top_k[0]["label"],
            "class_name": top_k[0]["class_name"],
            "score": top_k[0]["score"],
            "preview_urls": top_k[0]["preview_urls"],
            "top_k": top_k,
        }

    def predict_image_bytes(self, image_bytes: bytes, k: int = 5) -> dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_image(image=image, k=k)

    def predict_image_path(self, image_path: str | Path, k: int = 5) -> dict[str, Any]:
        image = Image.open(image_path)
        return self.predict_image(image=image, k=k)
