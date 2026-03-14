import argparse
import shutil
from pathlib import Path

from backend.config import DATASET_ROOT, PREVIEWS_DIR, TRAIN_SPLIT_ROOT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def default_source_root() -> Path:
    return TRAIN_SPLIT_ROOT if TRAIN_SPLIT_ROOT.exists() else DATASET_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed class preview folders from an existing sneaker dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_source_root(),
        help="Root containing one folder per class.",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=PREVIEWS_DIR,
        help="Output root for preview images.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="How many images to copy per class.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preview files.",
    )
    return parser.parse_args()


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    preview_root = Path(args.preview_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    class_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class directories found under {source_root}")

    copied = 0
    for class_dir in class_dirs:
        images = list_images(class_dir)
        if not images:
            continue

        target_dir = preview_root / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for index, source_path in enumerate(images[: args.per_class], start=1):
            target_path = target_dir / (
                f"{class_dir.name}_preview_{index:02d}{source_path.suffix.lower()}"
            )
            if target_path.exists() and not args.overwrite:
                continue
            shutil.copy2(source_path, target_path)
            copied += 1

    print(f"Seeded {copied} preview images into {preview_root}")


if __name__ == "__main__":
    main()
