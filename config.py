# config.py
from pathlib import Path
import torch

# Adjust to your dataset root
DATASET_ROOT = Path("sneakers-dataset")

# Artifacts output
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# Model
MODEL_BACKEND = "clip"
MODEL_NAME = "ViT-B/32"
# MODEL_NAME = "ViT-L/14"
# Optional weights checkpoint (can be any compatible state_dict)
MODEL_USE_CHECKPOINT = False
MODEL_CHECKPOINT = None  # e.g. Path("artifacts/models/your_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Embedding files (per-image & per-class)
IMG_EMB_NPY = ARTIFACTS / "image_embeddings.npy"      # shape [M, D]
IMG_META_JSON = ARTIFACTS / "image_meta.json"         # list of dicts
CLS_EMB_NPY = ARTIFACTS / "class_embeddings.npy"      # shape [C, D]
CLS_META_JSON = ARTIFACTS / "class_meta.json"         # list of dicts

# FAISS index file (per-class search first)
FAISS_INDEX = ARTIFACTS / "class_index.faiss"
