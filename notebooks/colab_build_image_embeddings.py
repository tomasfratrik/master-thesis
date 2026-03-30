"""
Colab-friendly script for building per-image embeddings from the real split train dataset.

Intended usage:
1. Put your split dataset on Drive so you have:
   /content/drive/MyDrive/thesis/dataset/sneakers-split/train/<class_name>/*.jpg
2. Install dependencies in Colab:

   !pip install git+https://github.com/openai/CLIP.git
   !pip install torch torchvision pillow tqdm numpy

3. Run this script:

   !python colab_build_image_embeddings.py \
       --train-root /content/drive/MyDrive/thesis/dataset/sneakers-split/train \
       --output-dir /content/drive/MyDrive/thesis/master-thesis/catalog-image-embeddings \
       --max-images-per-class 20

Outputs:
- image_embeddings.npy   shape [num_images, embedding_dim]
- image_meta.json        metadata for each image row
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TypeVar

import clip
import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-image catalog embeddings from a train split.")
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="Path to split train root, e.g. .../dataset/sneakers-split/train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where .npy and .json outputs will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="ViT-B/32",
        help="OpenAI CLIP model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Image batch size for embedding.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional cap on how many train images to embed per class.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["first", "random"],
        default="first",
        help="How to choose images when max-images-per-class is set.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when sample-mode=random.",
    )
    return parser.parse_args()


def list_class_dirs(train_root: Path) -> list[Path]:
    return sorted(path for path in train_root.iterdir() if path.is_dir())


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def select_images(
    image_paths: list[Path],
    *,
    max_images_per_class: int | None,
    sample_mode: str,
    rng: random.Random,
) -> list[Path]:
    if max_images_per_class is None or len(image_paths) <= max_images_per_class:
        return image_paths

    if sample_mode == "random":
        return sorted(rng.sample(image_paths, max_images_per_class))

    return image_paths[:max_images_per_class]


def chunked(items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


@torch.no_grad()
def encode_paths(
    model: torch.nn.Module,
    preprocess,
    image_paths: list[Path],
    device: torch.device,
) -> torch.Tensor:
    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            tensors.append(preprocess(image.convert("RGB")))
    batch = torch.stack(tensors, dim=0).to(device)
    features = model.encode_image(batch)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.float().cpu()


def main() -> None:
    args = parse_args()
    train_root = args.train_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_root.exists():
        raise FileNotFoundError(f"Train root does not exist: {train_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model, preprocess = clip.load(args.model_name, device=device, jit=False)
    model.eval()
    rng = random.Random(args.random_seed)

    class_dirs = list_class_dirs(train_root)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {train_root}")

    image_embeddings: list[np.ndarray] = []
    image_metadata: list[dict[str, object]] = []

    print(f"Found {len(class_dirs)} class directories.", flush=True)
    for class_index, class_dir in enumerate(class_dirs, start=1):
        all_image_paths = list_images(class_dir)
        if not all_image_paths:
            print(f"[{class_index}/{len(class_dirs)}] {class_dir.name}: skipped (no images)", flush=True)
            continue
        image_paths = select_images(
            all_image_paths,
            max_images_per_class=args.max_images_per_class,
            sample_mode=args.sample_mode,
            rng=rng,
        )

        print(
            f"[{class_index}/{len(class_dirs)}] {class_dir.name}: embedding {len(image_paths)}/{len(all_image_paths)} train images",
            flush=True,
        )
        for batch_paths in chunked(image_paths, args.batch_size):
            features = encode_paths(model, preprocess, batch_paths, device).numpy().astype("float32")
            image_embeddings.append(features)
            image_metadata.extend(
                {
                    "class_name": class_dir.name,
                    "label": class_dir.name.replace("_", " ").title(),
                    "path": str(path),
                    "filename": path.name,
                    "image_count": len(image_paths),
                    "available_image_count": len(all_image_paths),
                }
                for path in batch_paths
            )

    if not image_embeddings:
        raise RuntimeError("No embeddings were created.")

    embedding_matrix = np.vstack(image_embeddings).astype("float32")
    with open(output_dir / "image_meta.json", "w", encoding="utf-8") as handle:
        json.dump(image_metadata, handle, indent=2)

    np.save(output_dir / "image_embeddings.npy", embedding_matrix)

    print(f"Saved embeddings: {output_dir / 'image_embeddings.npy'}", flush=True)
    print(f"Saved metadata: {output_dir / 'image_meta.json'}", flush=True)
    print(f"Embedding matrix shape: {embedding_matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
