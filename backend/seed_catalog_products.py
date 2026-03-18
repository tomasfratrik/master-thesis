import argparse
import secrets
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import torch

from backend.app.auth import hash_password
from backend.app.catalog_metadata import build_brand_prefix_map, format_class_label, infer_model_name
from backend.app.config import MODEL_CHECKPOINT
from backend.app.db import get_connection, init_db, utc_now


DEFAULT_SYSTEM_EMAIL = "system@sneaker-matcher.local"
DEFAULT_CATALOG_NAME = "Reference Catalog"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed catalog product metadata from a model checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=MODEL_CHECKPOINT,
        help="Path to a checkpoint that contains class_names.",
    )
    parser.add_argument(
        "--user-email",
        default=DEFAULT_SYSTEM_EMAIL,
        help="System user email that owns the reference catalog.",
    )
    parser.add_argument(
        "--catalog-name",
        default=DEFAULT_CATALOG_NAME,
        help="Catalog name for seeded reference products.",
    )
    return parser.parse_args()


def ensure_user(connection: sqlite3.Connection, email: str) -> str:
    row = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if row is not None:
        return row["id"]

    salt = secrets.token_hex(16)
    password_hash = hash_password("disabled", salt)
    user_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO users (id, email, password_hash, password_salt, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, email, password_hash, salt, utc_now()),
    )
    return user_id


def ensure_catalog(connection: sqlite3.Connection, user_id: str, catalog_name: str) -> str:
    row = connection.execute(
        "SELECT id FROM catalogs WHERE user_id = ? AND name = ?",
        (user_id, catalog_name),
    ).fetchone()
    if row is not None:
        return row["id"]

    catalog_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO catalogs (id, user_id, name, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            catalog_id,
            user_id,
            catalog_name,
            "System-generated reference catalog metadata seeded from checkpoint classes.",
            utc_now(),
        ),
    )
    return catalog_id


def load_class_names(checkpoint_path: Path) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint does not contain class_names.")
    return sorted(str(class_name) for class_name in class_names)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is None:
        raise SystemExit("Checkpoint path is required. Pass --checkpoint or set SNEAKER_MODEL_CHECKPOINT.")
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    init_db()
    class_names = load_class_names(checkpoint_path)
    brand_map = build_brand_prefix_map(class_names)
    last_updated = date.today().isoformat()

    with get_connection() as connection:
        user_id = ensure_user(connection, args.user_email.strip().lower())
        catalog_id = ensure_catalog(connection, user_id, args.catalog_name.strip())

        seeded = 0
        for class_name in class_names:
            brand = brand_map[class_name]
            display_name = format_class_label(class_name)
            model = infer_model_name(class_name, brand)
            now = utc_now()
            connection.execute(
                """
                INSERT INTO catalog_products (
                    id, catalog_id, class_name, display_name,
                    brand, model, price_eur, last_updated,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(catalog_id, class_name) DO UPDATE SET
                    display_name = excluded.display_name,
                    brand = excluded.brand,
                    model = excluded.model,
                    price_eur = excluded.price_eur,
                    last_updated = excluded.last_updated,
                    updated_at = excluded.updated_at
                """,
                (
                    str(uuid.uuid4()),
                    catalog_id,
                    class_name,
                    display_name,
                    brand,
                    model,
                    None,
                    last_updated,
                    now,
                    now,
                ),
            )
            seeded += 1

    print(
        f"Seeded {seeded} products into catalog '{args.catalog_name}' "
        f"for {args.user_email} from {checkpoint_path}."
    )


if __name__ == "__main__":
    main()
