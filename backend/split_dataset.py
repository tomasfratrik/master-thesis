import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
REPO_ROOT = PROJECT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split sneaker dataset into train/val/test folders."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "sneakers",
        help="Root folder containing class image folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dataset" / "sneakers-split",
        help="Output root folder for train/val/test splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How files are written into split folders.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into existing output root.",
    )
    return parser.parse_args()


def discover_class_dirs(root: Path) -> List[Path]:
    return sorted({
        p.parent
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    })


def list_images(class_dir: Path) -> List[Path]:
    return sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio}, {val_ratio}, {test_ratio})"
        )


def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n

    # Keep small classes represented in non-train splits if possible.
    if n >= 3:
        if val_n == 0:
            val_n = 1
            train_n -= 1
        if test_n == 0:
            test_n = 1
            train_n -= 1
    elif n == 2:
        if train_n == 2:
            train_n = 1
            val_n = 1
            test_n = 0

    if train_n < 0:
        train_n = 0
    if val_n < 0:
        val_n = 0
    if test_n < 0:
        test_n = 0
    if train_n + val_n + test_n != n:
        raise RuntimeError("Split count mismatch.")
    return train_n, val_n, test_n


def write_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        dst.hardlink_to(src)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    if args.output_root.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output root already exists: {args.output_root}. "
            "Use --allow-existing-output to proceed."
        )

    class_dirs = discover_class_dirs(args.input_root)
    if not class_dirs:
        raise RuntimeError(f"No class directories with images found under {args.input_root}")

    # Guard against duplicate class leaf names if nested structure is used.
    class_name_to_dir: Dict[str, Path] = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        existing = class_name_to_dir.get(class_name)
        if existing is not None and existing != class_dir:
            raise RuntimeError(
                "Duplicate class leaf directory name detected: "
                f"{class_name} in {existing} and {class_dir}. "
                "Rename one class directory to keep labels unique."
            )
        class_name_to_dir[class_name] = class_dir

    rng = random.Random(args.seed)
    summary = {"train": 0, "val": 0, "test": 0, "classes": 0}

    for class_name, class_dir in sorted(class_name_to_dir.items()):
        images = list_images(class_dir)
        if not images:
            continue
        rng.shuffle(images)

        train_n, val_n, test_n = split_counts(
            n=len(images),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

        splits = {
            "train": images[:train_n],
            "val": images[train_n:train_n + val_n],
            "test": images[train_n + val_n:train_n + val_n + test_n],
        }

        for split_name, split_images in splits.items():
            for src in split_images:
                dst = args.output_root / split_name / class_name / src.name
                write_file(src=src, dst=dst, mode=args.mode)
            summary[split_name] += len(split_images)

        summary["classes"] += 1
        print(
            f"{class_name}: total={len(images)} "
            f"train={train_n} val={val_n} test={test_n}"
        )

    print("\nSplit completed.")
    print(
        f"Classes={summary['classes']} "
        f"train={summary['train']} val={summary['val']} test={summary['test']}"
    )
    print(f"Output root: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
