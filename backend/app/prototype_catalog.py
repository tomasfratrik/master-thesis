from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from backend.config import PREVIEWS_DIR

from .catalog_metadata import infer_model_name
from .config import UPLOADS_DIR
from .db import get_connection, utc_now
from .training_jobs import ensure_reference_catalog


def _store_product_file(product_id: str, role: str, filename: str, payload: bytes) -> str:
    target_dir = UPLOADS_DIR / "catalog_products" / product_id / role
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "upload.bin"
    target = target_dir / f"{uuid.uuid4().hex}_{safe_name}"
    target.write_bytes(payload)
    return str(target.relative_to(UPLOADS_DIR).as_posix())


def save_preview_assets(class_name: str, preview_uploads: list[tuple[str, bytes, str]]) -> None:
    preview_dir = PREVIEWS_DIR / class_name
    preview_dir.mkdir(parents=True, exist_ok=True)
    for existing in preview_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    for index, (filename, payload, _mime_type) in enumerate(preview_uploads, start=1):
        suffix = Path(filename).suffix.lower() or ".jpg"
        target = preview_dir / f"{class_name}_preview_{index:02d}{suffix}"
        target.write_bytes(payload)


def upsert_prototype_class(
    *,
    class_name: str,
    display_name: str,
    brand: str,
    notes: str | None,
    prototype_embedding: list[float],
    reference_uploads: list[tuple[str, bytes, str]],
    test_uploads: list[tuple[str, bytes, str]],
    preview_uploads: list[tuple[str, bytes, str]],
    evaluation_summary: dict[str, Any] | None = None,
) -> str:
    catalog_id = ensure_reference_catalog()
    now = utc_now()
    metadata_json = json.dumps(
        {
            "notes": notes,
            "prototype_evaluation": evaluation_summary,
        }
    )

    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT id
            FROM catalog_products
            WHERE catalog_id = ? AND class_name = ?
            """,
            (catalog_id, class_name),
        ).fetchone()
        product_id = existing["id"] if existing is not None else str(uuid.uuid4())

        connection.execute(
            """
            INSERT INTO catalog_products (
                id, catalog_id, class_name, display_name, brand, model,
                price_eur, last_updated, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(catalog_id, class_name) DO UPDATE SET
                display_name = excluded.display_name,
                brand = excluded.brand,
                model = excluded.model,
                last_updated = excluded.last_updated,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                product_id,
                catalog_id,
                class_name,
                display_name,
                brand,
                infer_model_name(class_name, brand),
                None,
                now[:10],
                metadata_json,
                now,
                now,
            ),
        )

        connection.execute(
            "DELETE FROM catalog_product_images WHERE product_id = ?",
            (product_id,),
        )

        for role, uploads in (
            ("reference", reference_uploads),
            ("test", test_uploads),
            ("preview", preview_uploads),
        ):
            for index, (filename, payload, mime_type) in enumerate(uploads):
                stored_path = _store_product_file(product_id, role, filename, payload)
                connection.execute(
                    """
                    INSERT INTO catalog_product_images (
                        id, product_id, image_role, original_filename, mime_type,
                        stored_path, sort_order, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        product_id,
                        role,
                        filename,
                        mime_type,
                        stored_path,
                        index,
                        now,
                    ),
                )

        connection.execute(
            """
            INSERT INTO catalog_product_prototypes (
                id, product_id, embedding_json, reference_image_count, updated_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                embedding_json = excluded.embedding_json,
                reference_image_count = excluded.reference_image_count,
                updated_at = excluded.updated_at
            """,
            (
                str(uuid.uuid4()),
                product_id,
                json.dumps(prototype_embedding),
                len(reference_uploads),
                now,
                now,
            ),
        )

    save_preview_assets(class_name, preview_uploads)
    return product_id


def load_classifier_prototypes() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                catalog_products.class_name,
                catalog_products.display_name,
                catalog_product_prototypes.embedding_json,
                catalog_product_prototypes.reference_image_count
            FROM catalog_product_prototypes
            JOIN catalog_products ON catalog_products.id = catalog_product_prototypes.product_id
            ORDER BY catalog_products.display_name ASC
            """
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        items.append(
            {
                "class_name": row["class_name"],
                "label": row["display_name"],
                "feature": json.loads(row["embedding_json"]),
                "reference_image_count": row["reference_image_count"],
            }
        )
    return items
