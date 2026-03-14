# embed.py
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from backend.config import (
    CLS_EMB_NPY,
    CLS_META_JSON,
    DATASET_ROOT,
    DEVICE,
    IMG_EMB_NPY,
    IMG_META_JSON,
)
from backend.model_loader import load_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image and class embeddings for a sneaker dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root containing class folders with images.",
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Load checkpoint weights when generating embeddings.",
    )
    return parser.parse_args()


def discover_class_dirs(root: Path) -> List[Path]:
    return sorted({
        p.parent
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    })

def list_images(cls_dir: Path) -> List[Path]:
    return sorted([
        p for p in cls_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])

@torch.no_grad()
def encode_batch(model, preprocess, paths: List[Path]):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))
    x = torch.stack(imgs, dim=0).to(DEVICE)
    z = model.encode_image(x)
    z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype("float32")  # [B, D]

def main():
    args = parse_args()
    model, preprocess = load_model(use_checkpoint=True if args.use_checkpoint else None)

    classes = discover_class_dirs(args.dataset_root)
    image_meta: List[Dict] = []
    image_vecs: List[np.ndarray] = []
    class_to_rep_path: Dict[str, str] = {}

    # Guard against ambiguous class naming when nested folders share the same leaf name.
    class_name_to_path: Dict[str, Path] = {}
    for class_dir in classes:
        class_name = class_dir.name
        if class_name in class_name_to_path and class_name_to_path[class_name] != class_dir:
            raise RuntimeError(
                "Duplicate class leaf directory name detected: "
                f"{class_name} in {class_name_to_path[class_name]} and {class_dir}. "
                "Use unique class leaf folder names."
            )
        class_name_to_path[class_name] = class_dir

    # Per-image embeddings
    for cls_dir in classes:
        class_name = cls_dir.name
        imgs = list_images(cls_dir)
        if not imgs:
            continue
        class_to_rep_path.setdefault(class_name, str(imgs[0].as_posix()))

        # batch to avoid OOM
        B = 64
        for i in tqdm(range(0, len(imgs), B), desc=f"Embedding {cls_dir.name}"):
            batch_paths = imgs[i:i+B]
            feats = encode_batch(model, preprocess, batch_paths)  # [b, D]
            image_vecs.append(feats)
            image_meta.extend([
                {
                    "class": class_name,
                    "path": str(p.as_posix()),
                    "filename": p.name
                } for p in batch_paths
            ])

    if not image_meta:
        raise RuntimeError(f"No images found under dataset root: {args.dataset_root}")

    image_vecs = np.vstack(image_vecs)  # [M, D]
    np.save(IMG_EMB_NPY, image_vecs)
    with open(IMG_META_JSON, "w") as f:
        json.dump(image_meta, f)
    print(f"Saved per-image embeddings: {image_vecs.shape} -> {IMG_EMB_NPY}")
    print(f"Saved per-image metadata: {len(image_meta)} -> {IMG_META_JSON}")

    # Per-class aggregation (average multi-view)
    # Group by class and average all its image vectors
    class_to_indices: Dict[str, List[int]] = {}
    for idx, meta in enumerate(image_meta):
        class_to_indices.setdefault(meta["class"], []).append(idx)

    cls_names = sorted(class_to_indices.keys())
    cls_vecs = []
    cls_meta = []
    for cname in cls_names:
        idxs = class_to_indices[cname]
        v = image_vecs[idxs].mean(axis=0)
        v = v / np.linalg.norm(v)  # re-normalize
        cls_vecs.append(v.astype("float32"))
        cls_meta.append({
            "class": cname,
            "rep_path": class_to_rep_path[cname],
            "count": len(idxs)
        })

    cls_vecs = np.vstack(cls_vecs).astype("float32")
    np.save(CLS_EMB_NPY, cls_vecs)
    with open(CLS_META_JSON, "w") as f:
        json.dump(cls_meta, f)

    print(f"Saved per-class embeddings: {cls_vecs.shape} -> {CLS_EMB_NPY}")
    print(f"Saved per-class metadata: {len(cls_meta)} classes -> {CLS_META_JSON}")
    print(f"Embedded dataset root: {args.dataset_root}")

if __name__ == "__main__":
    main()
