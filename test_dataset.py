import torch
import clip
from finetune_clip import SneakerDataset
from config import DATASET_ROOT

# Quick dataset test
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

dataset = SneakerDataset(DATASET_ROOT, preprocess)

print(f"\nDataset size: {len(dataset)}")
print(f"\nFirst 10 samples:")
for i in range(min(10, len(dataset))):
    img, txt = dataset[i]
    print(f"{i+1}. Image shape: {img.shape}, Text: {dataset.texts[i]}")

print(f"\nUnique sneaker categories found:")
image_exts = {".jpg", ".jpeg", ".png", ".webp"}
class_dirs = sorted({
    p.parent.name
    for p in DATASET_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in image_exts
})
for i, name in enumerate(class_dirs, 1):
    print(f"{i}. {name}")
