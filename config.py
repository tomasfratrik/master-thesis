import os
from pathlib import Path

import torch


def _default_dataset_root() -> Path:
    raw_root = REPO_ROOT / "dataset" / "sneakers"
    preprocessed_root = REPO_ROOT / "dataset" / "sneakers-preprocessed"
    return preprocessed_root if preprocessed_root.exists() else raw_root


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

# Dataset roots
RAW_DATASET_ROOT = Path(
    os.getenv("SNEAKER_RAW_DATASET_ROOT", REPO_ROOT / "dataset" / "sneakers")
)
PREPROCESSED_DATASET_ROOT = Path(
    os.getenv(
        "SNEAKER_PREPROCESSED_DATASET_ROOT",
        REPO_ROOT / "dataset" / "sneakers-preprocessed",
    )
)
DATASET_ROOT = Path(os.getenv("SNEAKER_DATASET_ROOT", _default_dataset_root()))
DATASET_SPLIT_ROOT = Path(
    os.getenv("SNEAKER_SPLIT_ROOT", REPO_ROOT / "dataset" / "sneakers-split")
)
TRAINING_DATA_ROOT = Path(
    os.getenv(
        "SNEAKER_TRAINING_DATA_ROOT",
        DATASET_SPLIT_ROOT if DATASET_SPLIT_ROOT.exists() else DATASET_ROOT,
    )
)
TRAIN_SPLIT_ROOT = Path(
    os.getenv("SNEAKER_TRAIN_ROOT", DATASET_SPLIT_ROOT / "train")
)
VAL_SPLIT_ROOT = Path(
    os.getenv("SNEAKER_VAL_ROOT", DATASET_SPLIT_ROOT / "val")
)
TEST_SPLIT_ROOT = Path(
    os.getenv("SNEAKER_TEST_ROOT", DATASET_SPLIT_ROOT / "test")
)

# Artifacts output
ARTIFACTS = PROJECT_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FINETUNE_OUTPUT_DIR = ARTIFACTS / "finetuned_models"
FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR = ARTIFACTS / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_URL_PREFIX = "/previews"
PREVIEW_LIMIT = _int_env("SNEAKER_PREVIEW_LIMIT", 10)

# Model
MODEL_BACKEND = "clip"
MODEL_NAME = "ViT-B/32"
# MODEL_NAME = "ViT-L/14"
MODEL_USE_CHECKPOINT = False
MODEL_CHECKPOINT = os.getenv("SNEAKER_MODEL_CHECKPOINT")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training (fine-tuning)
BATCH_SIZE = _int_env("SNEAKER_BATCH_SIZE", 32)
EPOCHS = _int_env("SNEAKER_EPOCHS", 30)
LEARNING_RATE = _float_env("SNEAKER_LEARNING_RATE", 1e-6)
WARMUP_EPOCHS = _int_env("SNEAKER_WARMUP_EPOCHS", 3)
WEIGHT_DECAY = _float_env("SNEAKER_WEIGHT_DECAY", 0.001)
DATALOADER_NUM_WORKERS = _int_env("SNEAKER_DATALOADER_WORKERS", 4)
BEST_CHECKPOINT_NAME = "clip_sneaker_best.pt"
FINAL_CHECKPOINT_NAME = "clip_sneaker_final.pt"
TRAINING_HISTORY_NAME = "training_history.json"
BEST_CHECKPOINT_PATH = FINETUNE_OUTPUT_DIR / BEST_CHECKPOINT_NAME
FINAL_CHECKPOINT_PATH = FINETUNE_OUTPUT_DIR / FINAL_CHECKPOINT_NAME
TRAINING_HISTORY_PATH = FINETUNE_OUTPUT_DIR / TRAINING_HISTORY_NAME

# Embedding files (per-image & per-class)
IMG_EMB_NPY = ARTIFACTS / "image_embeddings.npy"
IMG_META_JSON = ARTIFACTS / "image_meta.json"
CLS_EMB_NPY = ARTIFACTS / "class_embeddings.npy"
CLS_META_JSON = ARTIFACTS / "class_meta.json"
FAISS_INDEX = ARTIFACTS / "class_index.faiss"
