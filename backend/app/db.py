"""SQLite schema initialization and connection helpers for the backend."""

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .config import DB_PATH


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """Add a column to an existing table when it is missing."""
    if column_name in _table_columns(connection, table_name):
        return
    connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")


def _unique_username(connection: sqlite3.Connection, preferred: str, user_id: str | None = None) -> str:
    base = (preferred or "user").strip().lower() or "user"
    candidate = base
    counter = 2
    while True:
        if user_id is None:
            row = connection.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (candidate,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT 1 FROM users WHERE username = ? AND id != ?",
                (candidate, user_id),
            ).fetchone()
        if row is None:
            return candidate
        candidate = f"{base}{counter}"
        counter += 1


def _ensure_user_profile_columns(connection: sqlite3.Connection) -> None:
    """Backfill username, full name, and role fields for existing users."""
    _ensure_column(
        connection,
        table_name="users",
        column_name="username",
        column_definition="username TEXT",
    )
    _ensure_column(
        connection,
        table_name="users",
        column_name="full_name",
        column_definition="full_name TEXT",
    )
    _ensure_column(
        connection,
        table_name="users",
        column_name="role",
        column_definition="role TEXT NOT NULL DEFAULT 'user'",
    )

    rows = connection.execute(
        "SELECT id, email, username, full_name, role FROM users ORDER BY created_at ASC"
    ).fetchall()
    for row in rows:
        user_id, email, username, full_name, role = row
        next_username = username
        if not next_username:
            base = (email or "user").split("@", 1)[0]
            next_username = _unique_username(connection, base, user_id=user_id)

        next_full_name = full_name or next_username
        next_role = role or "user"
        connection.execute(
            """
            UPDATE users
            SET username = ?, full_name = ?, role = ?
            WHERE id = ?
            """,
            (next_username, next_full_name, next_role, user_id),
        )


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the application schema and apply lightweight migrations."""
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                username TEXT UNIQUE,
                full_name TEXT,
                role TEXT NOT NULL DEFAULT 'user',
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
                matched_product_id TEXT,
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

            CREATE TABLE IF NOT EXISTS catalog_products (
                id TEXT PRIMARY KEY,
                catalog_id TEXT NOT NULL,
                class_name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                brand TEXT,
                model TEXT,
                price_eur REAL,
                last_updated TEXT,
                colorway TEXT,
                sku TEXT,
                retail_price REAL,
                currency TEXT,
                release_year INTEGER,
                release_date TEXT,
                gender TEXT,
                category TEXT,
                source TEXT,
                source_url TEXT,
                description TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (catalog_id) REFERENCES catalogs(id) ON DELETE CASCADE,
                UNIQUE (catalog_id, class_name)
            );

            CREATE TABLE IF NOT EXISTS catalog_product_images (
                id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                image_role TEXT,
                original_filename TEXT,
                mime_type TEXT,
                stored_path TEXT,
                source_url TEXT,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES catalog_products(id) ON DELETE CASCADE
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

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                created_by_user_id TEXT NOT NULL,
                brand TEXT NOT NULL,
                display_name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                notes TEXT,
                status TEXT NOT NULL,
                top_k INTEGER NOT NULL DEFAULT 5,
                required_topk_accuracy REAL NOT NULL DEFAULT 0.90,
                required_new_class_topk_accuracy REAL NOT NULL DEFAULT 0.90,
                output_dir TEXT,
                dataset_root TEXT,
                best_checkpoint_path TEXT,
                final_checkpoint_path TEXT,
                evaluation_report_path TEXT,
                overall_top1_accuracy REAL,
                overall_topk_accuracy REAL,
                new_class_top1_accuracy REAL,
                new_class_topk_accuracy REAL,
                gate_passed INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 0,
                accepted_at TEXT,
                activated_checkpoint_path TEXT,
                error_text TEXT,
                log_text TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (created_by_user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS training_job_files (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                split TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                mime_type TEXT,
                stored_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS catalog_product_prototypes (
                id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL UNIQUE,
                embedding_json TEXT NOT NULL,
                reference_image_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (product_id) REFERENCES catalog_products(id) ON DELETE CASCADE
            );

            """
        )

        # Additive migration for existing local databases created before the
        # product metadata schema existed.
        _ensure_user_profile_columns(connection)
        _ensure_column(
            connection,
            table_name="catalog_items",
            column_name="matched_product_id",
            column_definition="matched_product_id TEXT",
        )
        _ensure_column(
            connection,
            table_name="catalog_products",
            column_name="price_eur",
            column_definition="price_eur REAL",
        )
        _ensure_column(
            connection,
            table_name="catalog_products",
            column_name="last_updated",
            column_definition="last_updated TEXT",
        )
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_catalog_items_catalog_id
            ON catalog_items (catalog_id);

            CREATE INDEX IF NOT EXISTS idx_catalog_items_matched_product_id
            ON catalog_items (matched_product_id);

            CREATE INDEX IF NOT EXISTS idx_catalog_products_catalog_id
            ON catalog_products (catalog_id);

            CREATE INDEX IF NOT EXISTS idx_catalog_products_brand
            ON catalog_products (brand);

            CREATE INDEX IF NOT EXISTS idx_catalog_products_class_name
            ON catalog_products (class_name);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
            ON users (username);

            CREATE INDEX IF NOT EXISTS idx_catalog_product_images_product_id
            ON catalog_product_images (product_id);

            CREATE INDEX IF NOT EXISTS idx_item_images_item_id
            ON item_images (item_id);

            CREATE INDEX IF NOT EXISTS idx_training_jobs_status
            ON training_jobs (status);

            CREATE INDEX IF NOT EXISTS idx_training_jobs_created_by_user_id
            ON training_jobs (created_by_user_id);

            CREATE INDEX IF NOT EXISTS idx_training_jobs_class_name
            ON training_jobs (class_name);

            CREATE INDEX IF NOT EXISTS idx_training_job_files_job_id
            ON training_job_files (job_id);

            CREATE INDEX IF NOT EXISTS idx_catalog_product_prototypes_product_id
            ON catalog_product_prototypes (product_id);
            """
        )


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access and automatic commit."""
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
