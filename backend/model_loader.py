from pathlib import Path
import torch
import clip

from backend.config import DEVICE, MODEL_BACKEND, MODEL_NAME, MODEL_CHECKPOINT, MODEL_USE_CHECKPOINT


def _load_clip(use_checkpoint: bool):
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    if use_checkpoint:
        if MODEL_CHECKPOINT is None:
            raise ValueError("MODEL_CHECKPOINT is None but MODEL_USE_CHECKPOINT is True.")
        ckpt_path = Path(MODEL_CHECKPOINT)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, preprocess


def load_model(use_checkpoint: bool | None = None):
    if use_checkpoint is None:
        use_checkpoint = MODEL_USE_CHECKPOINT
    if MODEL_BACKEND == "clip":
        return _load_clip(use_checkpoint)
    raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")
