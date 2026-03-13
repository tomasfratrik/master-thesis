import os
from pathlib import Path

from config import PREVIEWS_DIR, PROJECT_DIR


APP_DATA_DIR = PROJECT_DIR / "app_data"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = APP_DATA_DIR / "app.sqlite3"
UPLOADS_DIR = APP_DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

APP_MEDIA_URL_PREFIX = "/app-media"
MODEL_CHECKPOINT = os.getenv("SNEAKER_MODEL_CHECKPOINT")

# Reuse preview assets from the inference repo so the frontend can stay on one backend origin.
PREVIEW_DIR = PREVIEWS_DIR
