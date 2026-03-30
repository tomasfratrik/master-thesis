from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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


class VisionLanguageEncoder:
    backend: str
    model_name: str
    preprocess: Any
    model: Any

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

    def encode_image_tensors(self, image_batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
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

    def encode_image_tensors(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image_batch)

    def encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text_tokens)

    def prepare_optimizer_step(self) -> None:
        if self.device.type == "cuda":
            openai_clip.model.convert_weights(self.model)


class OpenClipEncoder(VisionLanguageEncoder):
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

    def encode_image_tensors(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image_batch)

    def encode_text_tokens(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text_tokens)


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
    selected_backend = backend or MODEL_BACKEND
    selected_model = model_name or MODEL_NAME
    selected_pretrained = pretrained if pretrained is not None else MODEL_PRETRAINED

    if selected_backend == "clip":
        encoder: VisionLanguageEncoder = OpenAIClipEncoder(device=device, model_name=selected_model)
    elif selected_backend == "open_clip":
        encoder = OpenClipEncoder(
            device=device,
            model_name=selected_model,
            pretrained=selected_pretrained,
        )
    else:
        raise ValueError(f"Unsupported MODEL_BACKEND: {selected_backend}")

    effective_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if effective_checkpoint is None:
        effective_use_checkpoint = MODEL_USE_CHECKPOINT if use_checkpoint is None else use_checkpoint
        if effective_use_checkpoint:
            if MODEL_CHECKPOINT is None:
                raise ValueError("MODEL_CHECKPOINT is None but MODEL_USE_CHECKPOINT is True.")
            effective_checkpoint = Path(MODEL_CHECKPOINT)

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
