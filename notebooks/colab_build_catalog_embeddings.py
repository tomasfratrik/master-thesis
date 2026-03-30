"""
Colab-friendly script for building one prototype embedding per sneaker class.

Intended usage:
1. Put your dataset on Drive so you have a folder like:
   /content/drive/MyDrive/master-thesis/dataset/sneakers-split/train/<class_name>/*.jpg
2. Install dependencies in Colab:

   !pip install git+https://github.com/openai/CLIP.git
   !pip install torch torchvision pillow tqdm numpy

3. Run this script:

   !python colab_build_catalog_embeddings.py \
       --train-root /content/drive/MyDrive/master-thesis/dataset/sneakers-split/train \
       --output-dir /content/drive/MyDrive/master-thesis/catalog-embeddings

Outputs:
- class_embeddings.npy    shape [num_classes, embedding_dim]
- class_metadata.json    metadata for each embedding row
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeVar

import clip
import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build catalog sneaker embeddings from a train split.")
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
    return parser.parse_args()


def list_class_dirs(train_root: Path) -> list[Path]:
    return sorted(path for path in train_root.iterdir() if path.is_dir())


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


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


@torch.no_grad()
def build_class_prototype(
    model: torch.nn.Module,
    preprocess,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("At least one image is required to build a class prototype.")

    feature_chunks: list[torch.Tensor] = []
    for batch_paths in chunked(image_paths, batch_size):
        feature_chunks.append(encode_paths(model, preprocess, batch_paths, device))

    features = torch.cat(feature_chunks, dim=0)
    prototype = features.mean(dim=0, keepdim=True)
    prototype = prototype / prototype.norm(dim=-1, keepdim=True)
    return prototype[0].numpy().astype("float32")


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

    class_dirs = list_class_dirs(train_root)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {train_root}")

    class_embeddings: list[np.ndarray] = []
    class_metadata: list[dict[str, object]] = []

    print(f"Found {len(class_dirs)} class directories.", flush=True)
    for index, class_dir in enumerate(class_dirs, start=1):
        image_paths = list_images(class_dir)
        if not image_paths:
            print(f"[{index}/{len(class_dirs)}] {class_dir.name}: skipped (no images)", flush=True)
            continue

        print(
            f"[{index}/{len(class_dirs)}] {class_dir.name}: embedding {len(image_paths)} images",
            flush=True,
        )
        prototype = build_class_prototype(
            model=model,
            preprocess=preprocess,
            image_paths=image_paths,
            device=device,
            batch_size=args.batch_size,
        )
        class_embeddings.append(prototype)
        class_metadata.append(
            {
                "class_name": class_dir.name,
                "label": class_dir.name.replace("_", " ").title(),
                "image_count": len(image_paths),
                "sample_image": str(image_paths[0]),
            }
        )

    if not class_embeddings:
        raise RuntimeError("No embeddings were created.")

    embedding_matrix = np.vstack(class_embeddings).astype("float32")
    with open(output_dir / "class_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(class_metadata, handle, indent=2)

    np.save(output_dir / "class_embeddings.npy", embedding_matrix)

    print(f"Saved embeddings: {output_dir / 'class_embeddings.npy'}", flush=True)
    print(f"Saved metadata: {output_dir / 'class_metadata.json'}", flush=True)
    print(f"Embedding matrix shape: {embedding_matrix.shape}", flush=True)


if __name__ == "__main__":
    main()
