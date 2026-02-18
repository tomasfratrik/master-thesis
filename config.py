from pathlib import Path

import torch

# Adjust to your dataset root
DATASET_ROOT = Path("sneakers-dataset")

# Artifacts output
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
FINETUNE_OUTPUT_DIR = ARTIFACTS / "finetuned_models"
FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_BACKEND = "clip"
MODEL_NAME = "ViT-B/32"
# MODEL_NAME = "ViT-L/14"
MODEL_USE_CHECKPOINT = False
MODEL_CHECKPOINT = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training (fine-tuning)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-6
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 0.001
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
