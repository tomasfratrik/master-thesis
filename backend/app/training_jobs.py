from __future__ import annotations

"""
Training job persistence and execution helpers for admin fine-tuning workflows.

Builds job datasets, runs training/evaluation commands, and tracks acceptance state.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auth import hash_password
from .catalog_metadata import infer_model_name
from .config import MODEL_CHECKPOINT, TRAINING_JOBS_DIR, UPLOADS_DIR
from .db import get_connection, utc_now
from backend.model_loader import infer_checkpoint_backend, load_checkpoint_metadata


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SETTINGS_ACTIVE_CHECKPOINT = "active_checkpoint_path"
REFERENCE_CATALOG_NAME = "Reference Catalog"
REFERENCE_SYSTEM_EMAIL = "system@sneaker-matcher.local"
DEFAULT_ADMIN_TRAIN_BATCH_SIZE = int(os.getenv("SNEAKER_ADMIN_TRAIN_BATCH_SIZE", "8"))
DEFAULT_ADMIN_TRAIN_NUM_WORKERS = int(os.getenv("SNEAKER_ADMIN_TRAIN_NUM_WORKERS", "0"))

_RUNNING_THREADS: dict[str, threading.Thread] = {}
_THREAD_LOCK = threading.Lock()


@dataclass
class TrainingJobRecord:
    id: str
    brand: str
    display_name: str
    class_name: str
    status: str


def _job_root(job_id: str) -> Path:
    return TRAINING_JOBS_DIR / job_id


def _job_upload_root(job_id: str) -> Path:
    return _job_root(job_id) / "uploads"


def _job_dataset_root(job_id: str) -> Path:
    return _job_root(job_id) / "dataset"


def _job_output_root(job_id: str) -> Path:
    return _job_root(job_id) / "output"


def _job_eval_report_path(job_id: str) -> Path:
    return _job_root(job_id) / "eval_report.json"


def _training_module_name(active_checkpoint: str | None) -> str:
    if active_checkpoint:
        try:
            checkpoint_backend = infer_checkpoint_backend(load_checkpoint_metadata(active_checkpoint))
        except Exception:
            checkpoint_backend = None
        if checkpoint_backend == "efficientnet_b0":
            return "backend.finetune_efficientnet"

    backend_name = os.getenv("SNEAKER_MODEL_BACKEND", "clip")
    if backend_name == "efficientnet_b0":
        return "backend.finetune_efficientnet"
    return "backend.finetune_clip"


def _training_checkpoint_names(training_module: str) -> tuple[str, str]:
    if training_module == "backend.finetune_efficientnet":
        return "efficientnet_b0_sneaker_best.pt", "efficientnet_b0_sneaker_final.pt"
    return "clip_sneaker_best.pt", "clip_sneaker_final.pt"


def get_active_checkpoint_path() -> str | None:
    """Return the checkpoint currently marked as active for inference."""
    if MODEL_CHECKPOINT:
        return MODEL_CHECKPOINT

    with get_connection() as connection:
        row = connection.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            (SETTINGS_ACTIVE_CHECKPOINT,),
        ).fetchone()
    if row is not None and row["value"]:
        return row["value"]
    return None


def set_active_checkpoint_path(path: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (SETTINGS_ACTIVE_CHECKPOINT, path, utc_now()),
        )


def initialize_training_jobs() -> None:
    """Restore checkpoint settings and fail interrupted jobs on startup."""
    active_checkpoint = get_active_checkpoint_path()
    if active_checkpoint:
        set_active_checkpoint_path(active_checkpoint)

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE training_jobs
            SET status = 'failed',
                error_text = COALESCE(error_text, 'Training interrupted during backend restart.'),
                finished_at = COALESCE(finished_at, ?),
                updated_at = ?
            WHERE status IN ('queued', 'running')
            """,
            (utc_now(), utc_now()),
        )


def ensure_reference_catalog() -> str:
    """Create or return the system-owned reference catalog."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id FROM users WHERE email = ?",
            (REFERENCE_SYSTEM_EMAIL,),
        ).fetchone()
        if row is None:
            user_id = str(uuid.uuid4())
            password_salt = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO users (id, email, username, full_name, role, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    REFERENCE_SYSTEM_EMAIL,
                    "system",
                    "System User",
                    "admin",
                    hash_password("disabled", password_salt),
                    password_salt,
                    utc_now(),
                ),
            )
        else:
            user_id = row["id"]

        catalog_row = connection.execute(
            "SELECT id FROM catalogs WHERE user_id = ? AND name = ?",
            (user_id, REFERENCE_CATALOG_NAME),
        ).fetchone()
        if catalog_row is not None:
            return str(catalog_row["id"])

        catalog_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO catalogs (id, user_id, name, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                catalog_id,
                user_id,
                REFERENCE_CATALOG_NAME,
                "System-generated reference catalog metadata.",
                utc_now(),
            ),
        )
        return catalog_id


def _relative_media_path(path: Path) -> str:
    return str(path.relative_to(UPLOADS_DIR).as_posix())


def _store_job_file(job_id: str, split: str, filename: str, payload: bytes) -> str:
    target_dir = UPLOADS_DIR / "training_jobs" / job_id / split
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name or "upload.bin"
    target = target_dir / f"{uuid.uuid4().hex}_{safe_name}"
    target.write_bytes(payload)
    return _relative_media_path(target)


def create_training_job(
    *,
    created_by_user_id: str,
    brand: str,
    display_name: str,
    class_name: str,
    notes: str | None,
    train_uploads: list[tuple[str, bytes, str]],
    test_uploads: list[tuple[str, bytes, str]],
    preview_uploads: list[tuple[str, bytes, str]],
    top_k: int = 5,
    required_topk_accuracy: float = 0.90,
    required_new_class_topk_accuracy: float = 0.90,
) -> str:
    """Create a new admin training job and persist its uploaded files."""
    if not train_uploads:
        raise ValueError("At least one training image is required.")
    if not test_uploads:
        raise ValueError("At least one test image is required.")

    job_id = str(uuid.uuid4())
    now = utc_now()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO training_jobs (
                id, created_by_user_id, brand, display_name, class_name, notes, status,
                top_k, required_topk_accuracy, required_new_class_topk_accuracy,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                created_by_user_id,
                brand,
                display_name,
                class_name,
                notes,
                top_k,
                required_topk_accuracy,
                required_new_class_topk_accuracy,
                now,
                now,
            ),
        )

        for split, uploads in (
            ("train", train_uploads),
            ("test", test_uploads),
            ("preview", preview_uploads),
        ):
            for filename, payload, mime_type in uploads:
                stored_path = _store_job_file(job_id, split, filename, payload)
                connection.execute(
                    """
                    INSERT INTO training_job_files (
                        id, job_id, split, original_filename, mime_type, stored_path, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), job_id, split, filename, mime_type, stored_path, now),
                )

    return job_id


def _job_files(job_id: str) -> dict[str, list[dict[str, str]]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT split, original_filename, mime_type, stored_path
            FROM training_job_files
            WHERE job_id = ?
            ORDER BY created_at ASC
            """,
            (job_id,),
        ).fetchall()

    files: dict[str, list[dict[str, str]]] = {"train": [], "test": [], "preview": []}
    for row in rows:
        files.setdefault(row["split"], []).append(
            {
                "original_filename": row["original_filename"],
                "mime_type": row["mime_type"] or "application/octet-stream",
                "url": f"/app-media/{row['stored_path']}",
            }
        )
    return files


def list_training_jobs() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                jobs.*,
                users.username AS created_by_username,
                COALESCE(SUM(CASE WHEN files.split = 'train' THEN 1 ELSE 0 END), 0) AS train_count,
                COALESCE(SUM(CASE WHEN files.split = 'test' THEN 1 ELSE 0 END), 0) AS test_count,
                COALESCE(SUM(CASE WHEN files.split = 'preview' THEN 1 ELSE 0 END), 0) AS preview_count
            FROM training_jobs AS jobs
            LEFT JOIN users ON users.id = jobs.created_by_user_id
            LEFT JOIN training_job_files AS files ON files.job_id = jobs.id
            GROUP BY jobs.id, users.username
            ORDER BY jobs.created_at DESC
            """
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["gate_passed"] = bool(item["gate_passed"])
        item["accepted"] = bool(item["accepted"])
        item["files"] = _job_files(item["id"])
        items.append(item)
    return items


def get_training_job(job_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT jobs.*, users.username AS created_by_username
            FROM training_jobs AS jobs
            LEFT JOIN users ON users.id = jobs.created_by_user_id
            WHERE jobs.id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["gate_passed"] = bool(item["gate_passed"])
    item["accepted"] = bool(item["accepted"])
    item["files"] = _job_files(job_id)
    return item


def _update_job(job_id: str, **fields: Any) -> None:
    """Apply partial field updates to a stored training job."""
    if not fields:
        return
    fields["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    with get_connection() as connection:
        connection.execute(
            f"UPDATE training_jobs SET {assignments} WHERE id = ?",
            values,
        )


def _safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _mirror_split_root(source_root: Path | None, destination_root: Path) -> None:
    """Mirror an existing dataset split into a job dataset workspace."""
    if source_root is None or not source_root.exists():
        return

    for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            _safe_symlink(image_path, destination_root / class_dir.name / image_path.name)


def _copy_job_split_to_dataset(job_id: str, split: str, class_name: str, destination_root: Path) -> None:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT stored_path, original_filename
            FROM training_job_files
            WHERE job_id = ? AND split = ?
            ORDER BY created_at ASC
            """,
            (job_id, split),
        ).fetchall()

    class_root = destination_root / class_name
    class_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        source = UPLOADS_DIR / row["stored_path"]
        suffix = Path(row["original_filename"]).suffix.lower() or ".jpg"
        target = class_root / f"{class_name}_job_{job_id[:8]}_{index:03d}{suffix}"
        shutil.copy2(source, target)


def _build_dataset_for_job(
    *,
    job_id: str,
    class_name: str,
    train_root: Path,
    val_root: Path | None,
    test_root: Path,
) -> Path:
    """Build the merged dataset tree used for one training job run."""
    dataset_root = _job_dataset_root(job_id)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    merged_train = dataset_root / "train"
    merged_val = dataset_root / "val"
    merged_test = dataset_root / "test"

    _mirror_split_root(train_root, merged_train)
    _mirror_split_root(val_root, merged_val)
    _mirror_split_root(test_root, merged_test)

    _copy_job_split_to_dataset(job_id, "train", class_name, merged_train)
    _copy_job_split_to_dataset(job_id, "test", class_name, merged_test)
    return dataset_root


def _run_command_stream(
    job_id: str,
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
) -> int:
    """Run a subprocess and append its combined output to the job log."""
    _append_job_log(job_id, f"$ {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        if stripped:
            _append_job_log(job_id, stripped)

    return process.wait()


def _evaluate_job_gates(job: dict[str, Any], report: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Evaluate whether a completed job satisfies the acceptance gates."""
    summary = report.get("summary", {})
    per_class = report.get("per_class", {})
    class_summary = per_class.get(job["class_name"], {})

    overall_top1 = float(summary.get("top1_accuracy", 0.0))
    overall_topk = float(summary.get(f"top{job['top_k']}_accuracy", 0.0))
    new_class_top1 = float(class_summary.get("top1_accuracy", 0.0))
    new_class_topk = float(class_summary.get("topk_accuracy", 0.0))

    gate_passed = (
        overall_topk >= float(job["required_topk_accuracy"])
        and new_class_topk >= float(job["required_new_class_topk_accuracy"])
        and class_summary.get("total", 0) > 0
    )
    return gate_passed, {
        "overall_top1_accuracy": overall_top1,
        "overall_topk_accuracy": overall_topk,
        "new_class_top1_accuracy": new_class_top1,
        "new_class_topk_accuracy": new_class_topk,
    }


def _append_job_log(job_id: str, text: str) -> None:
    """Append one line of text to the persisted job log buffer."""
    if not text:
        return
    with get_connection() as connection:
        row = connection.execute(
            "SELECT log_text FROM training_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        current = row["log_text"] if row is not None and row["log_text"] else ""
        next_text = f"{current}\n{text}".strip() if current else text
        if len(next_text) > 40000:
            next_text = next_text[-40000:]
        connection.execute(
            "UPDATE training_jobs SET log_text = ?, updated_at = ? WHERE id = ?",
            (next_text, utc_now(), job_id),
        )


def _run_training_job(
    *,
    job_id: str,
    train_root: Path,
    val_root: Path | None,
    test_root: Path,
) -> None:
    """Execute the full train-and-evaluate lifecycle for one job."""
    job = get_training_job(job_id)
    if job is None:
        return

    _update_job(job_id, status="running", started_at=utc_now(), error_text=None)
    output_root = _job_output_root(job_id)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        dataset_root = _build_dataset_for_job(
            job_id=job_id,
            class_name=job["class_name"],
            train_root=train_root,
            val_root=val_root,
            test_root=test_root,
        )
        _update_job(job_id, dataset_root=str(dataset_root), output_dir=str(output_root))

        env = os.environ.copy()
        env["SNEAKER_FINETUNE_OUTPUT_DIR"] = str(output_root)
        active_checkpoint = get_active_checkpoint_path()
        if active_checkpoint:
            env["SNEAKER_MODEL_CHECKPOINT"] = active_checkpoint

        training_module = _training_module_name(active_checkpoint)
        train_command = [
            sys.executable,
            "-m",
            training_module,
            "--data-root",
            str(dataset_root),
            "--batch-size",
            str(DEFAULT_ADMIN_TRAIN_BATCH_SIZE),
            "--num-workers",
            str(DEFAULT_ADMIN_TRAIN_NUM_WORKERS),
        ]
        if active_checkpoint:
            train_command.extend(["--init-checkpoint", active_checkpoint])

        return_code = _run_command_stream(
            job_id,
            train_command,
            env=env,
            cwd=Path(__file__).resolve().parents[2],
        )
        if return_code != 0:
            raise RuntimeError("Training command failed.")

        best_checkpoint_name, final_checkpoint_name = _training_checkpoint_names(training_module)
        best_checkpoint = output_root / best_checkpoint_name
        final_checkpoint = output_root / final_checkpoint_name
        eval_report = _job_eval_report_path(job_id)

        eval_command = [
            sys.executable,
            "-m",
            "backend.evaluate_finetuned",
            "--checkpoint",
            str(best_checkpoint),
            "--test-root",
            str(dataset_root / "test"),
            "--top-k",
            str(job["top_k"]),
            "--output-json",
            str(eval_report),
        ]
        return_code = _run_command_stream(
            job_id,
            eval_command,
            env=env,
            cwd=Path(__file__).resolve().parents[2],
        )
        if return_code != 0:
            raise RuntimeError("Evaluation command failed.")

        report = json.loads(eval_report.read_text(encoding="utf-8"))
        gate_passed, metrics = _evaluate_job_gates(job, report)
        _update_job(
            job_id,
            status="completed",
            finished_at=utc_now(),
            best_checkpoint_path=str(best_checkpoint),
            final_checkpoint_path=str(final_checkpoint),
            evaluation_report_path=str(eval_report),
            gate_passed=int(gate_passed),
            **metrics,
        )
    except Exception as error:
        _update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            error_text=str(error),
        )
    finally:
        with _THREAD_LOCK:
            _RUNNING_THREADS.pop(job_id, None)


def start_training_job(
    *,
    job_id: str,
    train_root: Path,
    val_root: Path | None,
    test_root: Path,
) -> None:
    """Queue a background thread to run the requested training job."""
    job = get_training_job(job_id)
    if job is None:
        raise ValueError("Training job not found.")
    if job["status"] == "running":
        raise ValueError("Training job is already running.")

    with _THREAD_LOCK:
        running_thread = _RUNNING_THREADS.get(job_id)
        if running_thread is not None and running_thread.is_alive():
            raise ValueError("Training job is already running.")

        thread = threading.Thread(
            target=_run_training_job,
            kwargs={
                "job_id": job_id,
                "train_root": train_root,
                "val_root": val_root,
                "test_root": test_root,
            },
            daemon=True,
            name=f"training-job-{job_id}",
        )
        _RUNNING_THREADS[job_id] = thread
        _update_job(
            job_id,
            status="queued",
            error_text=None,
            log_text=None,
            gate_passed=0,
            accepted=0,
            accepted_at=None,
            activated_checkpoint_path=None,
            overall_top1_accuracy=None,
            overall_topk_accuracy=None,
            new_class_top1_accuracy=None,
            new_class_topk_accuracy=None,
            best_checkpoint_path=None,
            final_checkpoint_path=None,
            evaluation_report_path=None,
            finished_at=None,
        )
        thread.start()


def accept_training_job(job_id: str) -> dict[str, Any]:
    """Accept a completed training job and promote its checkpoint/assets."""
    job = get_training_job(job_id)
    if job is None:
        raise ValueError("Training job not found.")
    if job["status"] != "completed":
        raise ValueError("Only completed training jobs can be accepted.")
    if not job["best_checkpoint_path"]:
        raise ValueError("No trained checkpoint is available for this job.")

    preview_target_dir = Path(os.getenv("SNEAKER_PREVIEW_DIR", "")) if os.getenv("SNEAKER_PREVIEW_DIR") else None
    if preview_target_dir is None:
        from backend.config import PREVIEWS_DIR

        preview_target_dir = PREVIEWS_DIR

    target_dir = preview_target_dir / job["class_name"]
    target_dir.mkdir(parents=True, exist_ok=True)
    for existing in target_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    files = _job_files(job_id).get("preview", [])
    for index, file in enumerate(files, start=1):
        source = UPLOADS_DIR / file["url"].removeprefix("/app-media/")
        suffix = Path(file["original_filename"]).suffix.lower() or ".jpg"
        shutil.copy2(source, target_dir / f"{job['class_name']}_preview_{index:02d}{suffix}")

    catalog_id = ensure_reference_catalog()
    now = utc_now()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO catalog_products (
                id, catalog_id, class_name, display_name, brand, model,
                price_eur, last_updated, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(catalog_id, class_name) DO UPDATE SET
                display_name = excluded.display_name,
                brand = excluded.brand,
                model = excluded.model,
                last_updated = excluded.last_updated,
                updated_at = excluded.updated_at
            """,
            (
                str(uuid.uuid4()),
                catalog_id,
                job["class_name"],
                job["display_name"],
                job["brand"],
                infer_model_name(job["class_name"], job["brand"]),
                None,
                now[:10],
                now,
                now,
            ),
        )

    set_active_checkpoint_path(job["best_checkpoint_path"])
    _update_job(
        job_id,
        accepted=1,
        accepted_at=now,
        activated_checkpoint_path=job["best_checkpoint_path"],
    )
    return get_training_job(job_id) or job
