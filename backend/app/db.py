import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .config import DB_PATH


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def init_db(db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS catalogs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS catalog_items (
                id TEXT PRIMARY KEY,
                catalog_id TEXT NOT NULL,
                title TEXT,
                notes TEXT,
                predicted_class_name TEXT,
                predicted_label TEXT,
                predicted_score REAL,
                margin_vs_second REAL,
                query_image_count INTEGER NOT NULL DEFAULT 0,
                aggregation TEXT,
                raw_prediction_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS item_images (
                id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                image_role TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                mime_type TEXT,
                stored_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (item_id) REFERENCES catalog_items(id) ON DELETE CASCADE
            );
            """
        )


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
