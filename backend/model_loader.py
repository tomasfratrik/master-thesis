from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from backend.config import (
    DEVICE,
    MODEL_BACKEND,
    MODEL_CHECKPOINT,
    MODEL_NAME,
    MODEL_PRETRAINED,
    MODEL_USE_CHECKPOINT,
)

try:
    import clip as openai_clip
except ImportError:  # pragma: no cover - dependency environment dependent
    openai_clip = None

try:
    import open_clip
except ImportError:  # pragma: no cover - dependency environment dependent
    open_clip = None

try:
    from torchvision import models as torchvision_models
except ImportError:  # pragma: no cover - dependency environment dependent
    torchvision_models = None


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load checkpoint metadata from disk without validating its model family."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must deserialize to a dict: {path}")
    return checkpoint


def infer_checkpoint_backend(checkpoint: dict[str, Any]) -> str | None:
    """Infer the backend identifier stored in a training checkpoint."""
    backend = checkpoint.get("model_backend")
    if isinstance(backend, str) and backend:
        return backend

    model_name = checkpoint.get("model_name")
    if isinstance(model_name, str) and model_name in {"efficientnet_b0", "resnet50"}:
        return model_name

    family = checkpoint.get("model_family")
    if family == "image_classifier":
        return None

    return None


class VisionLanguageEncoder:
    backend: str
    model_name: str
    preprocess: Any
    model: Any
    supports_text = False
    supports_classifier_head = False

    def __init__(self, *, device: torch.device) -> None:
        self.device = device

    def load_checkpoint_weights(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=map_location or self.device)
        state_dict = checkpoint.get("model_state_dict")
        if not state_dict:
            raise ValueError(f"Checkpoint does not contain model_state_dict: {path}")
        self.model.load_state_dict(state_dict, strict=strict)
        self.model.eval()
        return checkpoint

    def tokenize_texts(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_image_features(
        self,
        image_batch: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError

    def encode_image_tensors(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self.encode_image_features(image_batch, normalize=True)

    def encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def classify_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_logits(
        self,
        image_batch: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(image_batch, text_tokens)

    def prepare_model_for_training(self) -> None:
        self.model.train()

    def prepare_optimizer_step(self) -> None:
        """Hook for backends that need weight dtype conversion around optimizer.step()."""


class OpenAIClipEncoder(VisionLanguageEncoder):
    supports_text = True

    def __init__(self, *, device: torch.device, model_name: str) -> None:
        if openai_clip is None:
            raise ImportError("openai CLIP is not installed.")
        super().__init__(device=device)
        self.backend = "clip"
        self.model_name = model_name
        self.model, self.preprocess = openai_clip.load(model_name, device=device, jit=False)
        self.model.eval()

    def tokenize_texts(self, texts: list[str]) -> torch.Tensor:
        return openai_clip.tokenize(texts).to(self.device)

    def encode_image_features(
        self,
        image_batch: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        image_features = self.model.encode_image(image_batch)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text_tokens)

    def prepare_optimizer_step(self) -> None:
        if self.device.type == "cuda":
            openai_clip.model.convert_weights(self.model)


class OpenClipEncoder(VisionLanguageEncoder):
    supports_text = True

    def __init__(self, *, device: torch.device, model_name: str, pretrained: str | None) -> None:
        if open_clip is None:
            raise ImportError(
                "open_clip_torch is not installed. Install it before using MODEL_BACKEND=open_clip."
            )
        super().__init__(device=device)
        self.backend = "open_clip"
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def tokenize_texts(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts).to(self.device)

    def encode_image_features(
        self,
        image_batch: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        image_features = self.model.encode_image(image_batch)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text_tokens)


class TorchvisionImageClassifierEncoder(VisionLanguageEncoder):
    supports_classifier_head = True

    def __init__(
        self,
        *,
        device: torch.device,
        model_name: str,
        num_classes: int | None = None,
    ) -> None:
        if torchvision_models is None:
            raise ImportError("torchvision is not installed.")
        if model_name not in {"efficientnet_b0", "resnet50"}:
            raise ValueError(f"Unsupported torchvision classifier model: {model_name}")

        super().__init__(device=device)
        self.backend = model_name
        self.model_name = model_name
        self.pretrained = "IMAGENET1K_V1" if model_name == "efficientnet_b0" else "IMAGENET1K_V2"

        if model_name == "efficientnet_b0":
            weights = torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.model = torchvision_models.efficientnet_b0(weights=weights)
            feature_dim = int(self.model.classifier[1].in_features)
            if num_classes is not None:
                self.model.classifier[1] = nn.Linear(feature_dim, num_classes)
            self.classifier_head = self.model.classifier
            self.feature_extractor = self.model.features
        else:
            weights = torchvision_models.ResNet50_Weights.IMAGENET1K_V2
            self.model = torchvision_models.resnet50(weights=weights)
            feature_dim = int(self.model.fc.in_features)
            if num_classes is not None:
                self.model.fc = nn.Linear(feature_dim, num_classes)
            self.classifier_head = self.model.fc
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_dim = feature_dim
        self.preprocess = weights.transforms()
        self.model.to(self.device)
        self.model.eval()

    def encode_image_features(
        self,
        image_batch: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        image_features = self.feature_extractor(image_batch)
        if self.model_name == "efficientnet_b0":
            image_features = self.model.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def classify_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(image_features)


def load_encoder(
    *,
    backend: str | None = None,
    model_name: str | None = None,
    pretrained: str | None = None,
    device: torch.device = DEVICE,
    checkpoint_path: str | Path | None = None,
    use_checkpoint: bool | None = None,
    checkpoint_map_location: str | torch.device | None = None,
    checkpoint_strict: bool = False,
) -> VisionLanguageEncoder:
    checkpoint_metadata: dict[str, Any] | None = None
    effective_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if effective_checkpoint is None:
        effective_use_checkpoint = MODEL_USE_CHECKPOINT if use_checkpoint is None else use_checkpoint
        if effective_use_checkpoint:
            if MODEL_CHECKPOINT is None:
                raise ValueError("MODEL_CHECKPOINT is None but MODEL_USE_CHECKPOINT is True.")
            effective_checkpoint = Path(MODEL_CHECKPOINT)

    inferred_backend = None
    if effective_checkpoint is not None:
        checkpoint_metadata = load_checkpoint_metadata(effective_checkpoint)
        inferred_backend = infer_checkpoint_backend(checkpoint_metadata)

    selected_backend = backend or inferred_backend or MODEL_BACKEND
    if model_name is not None:
        selected_model = model_name
    elif selected_backend in {"efficientnet_b0", "resnet50"}:
        selected_model = selected_backend
    else:
        selected_model = MODEL_NAME
    selected_pretrained = pretrained if pretrained is not None else MODEL_PRETRAINED

    if selected_backend == "clip":
        encoder: VisionLanguageEncoder = OpenAIClipEncoder(device=device, model_name=selected_model)
    elif selected_backend == "open_clip":
        encoder = OpenClipEncoder(
            device=device,
            model_name=selected_model,
            pretrained=selected_pretrained,
        )
    elif selected_backend in {"efficientnet_b0", "resnet50"}:
        class_names = None if checkpoint_metadata is None else checkpoint_metadata.get("class_names")
        num_classes = len(class_names) if class_names else None
        encoder = TorchvisionImageClassifierEncoder(
            device=device,
            model_name=selected_model,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unsupported MODEL_BACKEND: {selected_backend}")

    if effective_checkpoint is not None:
        encoder.load_checkpoint_weights(
            effective_checkpoint,
            map_location=checkpoint_map_location,
            strict=checkpoint_strict,
        )

    return encoder


def load_model(use_checkpoint: bool | None = None):
    encoder = load_encoder(use_checkpoint=use_checkpoint)
    return encoder.model, encoder.preprocess
