import torch
import clip
from pathlib import Path
from finetune_clip_nike import NikeSneakerDataset

# Quick dataset test
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

dataset = NikeSneakerDataset("sneakers-dataset", preprocess)

print(f"\nDataset size: {len(dataset)}")
print(f"\nFirst 10 samples:")
for i in range(min(10, len(dataset))):
    img, txt = dataset[i]
    print(f"{i+1}. Image shape: {img.shape}, Text: {dataset.texts[i]}")

print(f"\nUnique Nike categories found:")
nike_dirs = sorted([d.name for d in Path("sneakers-dataset").iterdir()
                   if d.is_dir() and d.name.startswith('nike_')])
for i, name in enumerate(nike_dirs, 1):
    print(f"{i}. {name}")
