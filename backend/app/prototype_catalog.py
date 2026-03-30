from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image
from backend.config import (
    PRECOMPUTED_CLASS_EMBEDDINGS,
    PRECOMPUTED_CLASS_METADATA,
    PRECOMPUTED_IMAGE_EMBEDDINGS,
    PRECOMPUTED_IMAGE_METADATA,
    PREVIEWS_DIR,
)
from backend.config import PREVIEW_URL_PREFIX, TRAIN_SPLIT_ROOT

from .catalog_metadata import format_class_label, infer_model_name
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


def _preview_urls(class_name: str) -> list[str]:
    preview_dir = PREVIEWS_DIR / class_name
    if not preview_dir.exists():
        return []

    urls: list[str] = []
    for path in sorted(preview_dir.iterdir()):
        if not path.is_file():
            continue
        urls.append(f"{PREVIEW_URL_PREFIX}/{quote(class_name)}/{quote(path.name)}")
    return urls


def _preview_images(class_name: str) -> list[Image.Image]:
    preview_dir = PREVIEWS_DIR / class_name
    if not preview_dir.exists():
        return []

    images: list[Image.Image] = []
    for path in sorted(preview_dir.iterdir()):
        if not path.is_file():
            continue
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _normalize_lookup_key(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _dataset_image_paths(class_name: str) -> list[Path]:
    class_dir = TRAIN_SPLIT_ROOT / class_name
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.iterdir() if path.is_file())


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


def ensure_catalog_embeddings(classifier: Any) -> int:
    if PRECOMPUTED_IMAGE_EMBEDDINGS is not None and PRECOMPUTED_IMAGE_METADATA is not None:
        print(
            f"[retrieval] Using precomputed image embeddings from {PRECOMPUTED_IMAGE_EMBEDDINGS.name} "
            f"and {PRECOMPUTED_IMAGE_METADATA.name}.",
            flush=True,
        )
        return 0

    if PRECOMPUTED_CLASS_EMBEDDINGS is not None and PRECOMPUTED_CLASS_METADATA is not None:
        print(
            f"[retrieval] Using precomputed catalog embeddings from {PRECOMPUTED_CLASS_EMBEDDINGS.name} "
            f"and {PRECOMPUTED_CLASS_METADATA.name}.",
            flush=True,
        )
        return 0

    now = utc_now()
    updated = 0

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                catalog_products.id,
                catalog_products.class_name,
                catalog_product_prototypes.reference_image_count
            FROM catalog_products
            LEFT JOIN catalog_product_prototypes
                ON catalog_product_prototypes.product_id = catalog_products.id
            ORDER BY catalog_products.display_name ASC
            """
        ).fetchall()
        total = len(rows)
        print(f"[retrieval] Checking catalog embeddings for {total} classes...", flush=True)

        for index, row in enumerate(rows, start=1):
            dataset_paths = _dataset_image_paths(row["class_name"])
            if dataset_paths:
                source_count = len(dataset_paths)
                if row["reference_image_count"] == source_count:
                    print(
                        f"[retrieval] {index}/{total} {row['class_name']}: up to date from train split ({source_count} images)",
                        flush=True,
                    )
                    continue
                print(
                    f"[retrieval] {index}/{total} {row['class_name']}: rebuilding from train split ({source_count} images)",
                    flush=True,
                )
                prototype_feature = classifier.build_prototype_from_image_paths(dataset_paths)
            else:
                preview_images = _preview_images(row["class_name"])
                if not preview_images:
                    print(
                        f"[retrieval] {index}/{total} {row['class_name']}: skipped, no train or preview images",
                        flush=True,
                    )
                    continue
                source_count = len(preview_images)
                if row["reference_image_count"] == source_count:
                    print(
                        f"[retrieval] {index}/{total} {row['class_name']}: up to date from previews ({source_count} images)",
                        flush=True,
                    )
                    continue
                print(
                    f"[retrieval] {index}/{total} {row['class_name']}: rebuilding from previews ({source_count} images)",
                    flush=True,
                )
                prototype_feature = classifier.build_prototype_from_images(preview_images)

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
                    row["id"],
                    json.dumps(prototype_feature.detach().cpu().tolist()),
                    source_count,
                    now,
                    now,
                ),
            )
            updated += 1
        print(f"[retrieval] Catalog embedding rebuild complete. Updated {updated}/{total} classes.", flush=True)

    return updated


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


def _catalog_product_map() -> dict[str, dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, class_name, display_name, brand, model
            FROM catalog_products
            ORDER BY display_name ASC
            """
        ).fetchall()
    return {
        row["class_name"]: {
            "product_id": row["id"],
            "class_name": row["class_name"],
            "label": row["display_name"] or format_class_label(row["class_name"]),
            "brand": row["brand"],
            "model": row["model"],
        }
        for row in rows
    }


def _catalog_class_alias_map(product_map: dict[str, dict[str, Any]]) -> dict[str, str]:
    aliases = {_normalize_lookup_key(class_name): class_name for class_name in product_map}
    if PREVIEWS_DIR.exists():
        for path in PREVIEWS_DIR.iterdir():
            if path.is_dir():
                aliases.setdefault(_normalize_lookup_key(path.name), path.name)
    return aliases


def _resolve_catalog_class_name(
    class_name: str,
    product_map: dict[str, dict[str, Any]],
    alias_map: dict[str, str],
) -> str:
    if class_name in product_map:
        return class_name
    return alias_map.get(_normalize_lookup_key(class_name), class_name)


def _metadata_class_name(meta: dict[str, Any], *, context: str) -> str:
    raw_class_name = meta.get("class_name")
    if not raw_class_name:
        raw_class_name = meta.get("class")

    if not raw_class_name:
        raise ValueError(f"{context} metadata is missing class_name/class.")

    return str(raw_class_name)


def _metadata_label(meta: dict[str, Any], product: dict[str, Any], class_name: str) -> str:
    if product.get("label"):
        return str(product["label"])

    if meta.get("label"):
        return str(meta["label"])

    return format_class_label(class_name)


def _metadata_reference_count(meta: dict[str, Any], *, fallback: int) -> int:
    if meta.get("image_count") is not None:
        return int(meta["image_count"])

    if meta.get("count") is not None:
        return int(meta["count"])

    return fallback


def _metadata_source_path(meta: dict[str, Any]) -> str | None:
    if meta.get("path"):
        return str(meta["path"])

    if meta.get("sample_image"):
        return str(meta["sample_image"])

    return None


def _load_precomputed_embedding_entries() -> list[dict[str, Any]]:
    if PRECOMPUTED_CLASS_EMBEDDINGS is None or PRECOMPUTED_CLASS_METADATA is None:
        return []

    embeddings = np.load(PRECOMPUTED_CLASS_EMBEDDINGS)
    metadata = json.loads(PRECOMPUTED_CLASS_METADATA.read_text())
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Precomputed class embeddings and metadata length mismatch: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    product_map = _catalog_product_map()
    alias_map = _catalog_class_alias_map(product_map)
    items: list[dict[str, Any]] = []
    for feature, meta in zip(embeddings, metadata):
        raw_class_name = _metadata_class_name(meta, context="Class embedding")
        class_name = _resolve_catalog_class_name(raw_class_name, product_map, alias_map)
        product = product_map.get(class_name, {})
        items.append(
            {
                "product_id": product.get("product_id"),
                "class_name": class_name,
                "label": _metadata_label(meta, product, class_name),
                "brand": product.get("brand"),
                "model": product.get("model"),
                "feature": feature.tolist(),
                "reference_image_count": _metadata_reference_count(meta, fallback=0),
                "preview_urls": _preview_urls(class_name),
                "candidate_type": "catalog_embedding",
            }
        )
    return items


def _load_precomputed_image_entries() -> list[dict[str, Any]]:
    if PRECOMPUTED_IMAGE_EMBEDDINGS is None or PRECOMPUTED_IMAGE_METADATA is None:
        return []

    embeddings = np.load(PRECOMPUTED_IMAGE_EMBEDDINGS)
    metadata = json.loads(PRECOMPUTED_IMAGE_METADATA.read_text())
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Precomputed image embeddings and metadata length mismatch: "
            f"{len(embeddings)} vs {len(metadata)}"
        )

    product_map = _catalog_product_map()
    alias_map = _catalog_class_alias_map(product_map)
    items: list[dict[str, Any]] = []
    for feature, meta in zip(embeddings, metadata):
        raw_class_name = _metadata_class_name(meta, context="Image embedding")
        class_name = _resolve_catalog_class_name(raw_class_name, product_map, alias_map)
        product = product_map.get(class_name, {})
        items.append(
            {
                "product_id": product.get("product_id"),
                "class_name": class_name,
                "label": _metadata_label(meta, product, class_name),
                "brand": product.get("brand"),
                "model": product.get("model"),
                "feature": feature.tolist(),
                "reference_image_count": _metadata_reference_count(meta, fallback=1),
                "preview_urls": _preview_urls(class_name),
                "candidate_type": "catalog_embedding",
                "source_path": _metadata_source_path(meta),
                "embedding_source": "precomputed_image",
            }
        )
    return items


def _load_saved_prototype_entries(exclude_class_names: set[str] | None = None) -> list[dict[str, Any]]:
    exclude_class_names = exclude_class_names or set()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                catalog_products.id,
                catalog_products.class_name,
                catalog_products.display_name,
                catalog_products.brand,
                catalog_products.model,
                catalog_product_prototypes.embedding_json,
                catalog_product_prototypes.reference_image_count
            FROM catalog_product_prototypes
            JOIN catalog_products ON catalog_products.id = catalog_product_prototypes.product_id
            ORDER BY catalog_products.display_name ASC
            """
        ).fetchall()

    entries: list[dict[str, Any]] = []
    for row in rows:
        if row["class_name"] in exclude_class_names:
            continue
        entries.append(
            {
                "product_id": row["id"],
                "class_name": row["class_name"],
                "label": row["display_name"] or format_class_label(row["class_name"]),
                "brand": row["brand"],
                "model": row["model"],
                "feature": json.loads(row["embedding_json"]),
                "reference_image_count": row["reference_image_count"],
                "preview_urls": _preview_urls(row["class_name"]),
                "candidate_type": "catalog_embedding",
            }
        )
    return entries


def load_catalog_embedding_entries() -> list[dict[str, Any]]:
    precomputed_entries = _load_precomputed_embedding_entries()
    seen = {entry["class_name"] for entry in precomputed_entries}
    return precomputed_entries + _load_saved_prototype_entries(seen)


def load_catalog_image_embedding_entries() -> list[dict[str, Any]]:
    return _load_precomputed_image_entries()
