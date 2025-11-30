# embed.py
import json, os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import clip

from config import (
    DATASET_ROOT, ARTIFACTS, MODEL_NAME, DEVICE,
    IMG_EMB_NPY, IMG_META_JSON, CLS_EMB_NPY, CLS_META_JSON
)

def list_classes(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])

def list_images(cls_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in exts])

def load_model():
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)
    model.eval()
    return model, preprocess

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
    model, preprocess = load_model()

    classes = list_classes(DATASET_ROOT)
    image_meta: List[Dict] = []
    image_vecs: List[np.ndarray] = []

    # Per-image embeddings
    for cls_dir in classes:
        imgs = list_images(cls_dir)
        if not imgs:
            continue

        # batch to avoid OOM
        B = 64
        for i in tqdm(range(0, len(imgs), B), desc=f"Embedding {cls_dir.name}"):
            batch_paths = imgs[i:i+B]
            feats = encode_batch(model, preprocess, batch_paths)  # [b, D]
            image_vecs.append(feats)
            image_meta.extend([
                {
                    "class": cls_dir.name,
                    "path": str(p.as_posix()),
                    "filename": p.name
                } for p in batch_paths
            ])

    if not image_meta:
        raise RuntimeError("No images found under dataset root.")

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
        # capture one representative path for thumbnail
        rep_path = next(DATASET_ROOT.joinpath(cname).glob("*"))
        cls_meta.append({
            "class": cname,
            "rep_path": str(rep_path.as_posix()),
            "count": len(idxs)
        })

    cls_vecs = np.vstack(cls_vecs).astype("float32")
    np.save(CLS_EMB_NPY, cls_vecs)
    with open(CLS_META_JSON, "w") as f:
        json.dump(cls_meta, f)

    print(f"Saved per-class embeddings: {cls_vecs.shape} -> {CLS_EMB_NPY}")
    print(f"Saved per-class metadata: {len(cls_meta)} classes -> {CLS_META_JSON}")

if __name__ == "__main__":
    main()

