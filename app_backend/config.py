import os
from pathlib import Path

from config import PREVIEWS_DIR, PROJECT_DIR


APP_DATA_DIR = PROJECT_DIR / "app_data"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = APP_DATA_DIR / "app.sqlite3"
UPLOADS_DIR = APP_DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

APP_MEDIA_URL_PREFIX = "/app-media"
PREPROCESS_URL = os.getenv("SNEAKER_PREPROCESS_URL")
MODEL_CHECKPOINT = os.getenv("SNEAKER_MODEL_CHECKPOINT")
PREPROCESS_TIMEOUT_SECONDS = int(os.getenv("SNEAKER_PREPROCESS_TIMEOUT_SECONDS", "60"))

# Reuse preview assets from the inference repo so the frontend can stay on one backend origin.
PREVIEW_DIR = PREVIEWS_DIR
