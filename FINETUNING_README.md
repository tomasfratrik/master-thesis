# CLIP Fine-tuning for Nike Sneakers

This project fine-tunes OpenAI's CLIP model specifically on Nike sneakers from the sneakers-dataset.

## Files

- `finetune_clip_nike.py` - Main training script
- `load_finetuned_clip.py` - Script to load and test fine-tuned models
- `test_nike_dataset.py` - Quick test to verify dataset loading

## Dataset

The training uses all Nike shoe images from the `sneakers-dataset/` directory, specifically:

- Nike Air Force 1 (High, Mid, Low)
- Nike Air Jordan (1, 3, 4, 11)
- Nike Air Max (1, 90, 95, 97, 270, Plus)
- Nike Air VaporMax (Flyknit, Plus)
- Nike Blazer Mid 77
- Nike Cortez
- Nike Dunk (High, Low)

Total: ~2000+ Nike shoe images across 20+ categories

## Requirements

```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## Usage

### 1. Test Dataset Loading

```bash
python test_nike_dataset.py
```

This will show you how many Nike images are found and sample text descriptions.

### 2. Train the Model

```bash
python finetune_clip_nike.py
```

**Configuration** (edit in `finetune_clip_nike.py`):
- `BATCH_SIZE = 16` - Adjust based on your GPU memory
- `EPOCH = 10` - Number of training epochs
- `LEARNING_RATE = 5e-5` - Learning rate for Adam optimizer
- `MODEL_NAME = "ViT-B/32"` - Base CLIP model

**Training Process**:
- Uses contrastive learning to align image-text pairs
- Automatically saves best model based on lowest loss
- Saves checkpoints every 5 epochs
- Outputs to `artifacts/finetuned_models/`

**Expected Output**:
```
Found 20 Nike shoe categories
Total Nike images: 2314
Epoch 1/10: 100%|██████████| 145/145 [02:30<00:00]
Average Loss: 2.4567
Saved best model to artifacts/finetuned_models/clip_nike_best.pt
```

### 3. Load and Test Fine-tuned Model

```bash
python load_finetuned_clip.py
```

This will:
- Load the best fine-tuned model
- Test it on a sample Nike Air Force 1 image
- Show probability scores for different shoe types

**Example Output**:
```
Test image: sneakers-dataset/nike_air_force_1_low/0001.jpg

Predictions:
a photo of Nike Air Force 1 Low sneakers: 87.32%
a photo of Nike Air Jordan 1 High sneakers: 8.15%
a photo of Adidas Ultraboost sneakers: 3.21%
a photo of Converse Chuck Taylor sneakers: 1.32%
```

## Using the Fine-tuned Model in Your Code

```python
from load_finetuned_clip import load_finetuned_model, test_model

# Load model
model, preprocess = load_finetuned_model("artifacts/finetuned_models/clip_nike_best.pt")

# Test on your image
image_path = "path/to/your/nike_shoe.jpg"
queries = [
    "a photo of Nike Air Force 1 Low sneakers",
    "a photo of Nike Dunk High sneakers"
]

probabilities = test_model(model, preprocess, image_path, queries)
```

## Training Tips

1. **GPU Memory Issues**: Reduce `BATCH_SIZE` if you get out-of-memory errors
2. **Faster Training**: Use `MODEL_NAME = "ViT-B/32"` (faster) instead of "ViT-L/14" (more accurate)
3. **Better Results**: Increase `EPOCH` to 20-30 for better fine-tuning
4. **Data Augmentation**: Modify `NikeSneakerDataset` to add augmentations

## Model Checkpoints

After training, you'll find:
- `clip_nike_best.pt` - Best model (lowest loss)
- `clip_nike_final.pt` - Final model after all epochs
- `clip_nike_epoch_5.pt`, `clip_nike_epoch_10.pt`, etc. - Periodic checkpoints
- `training_history.json` - Loss history for each epoch

## Implementation Details

Based on the CLIP fine-tuning approach:
- Contrastive loss on image-text pairs
- Adam optimizer with betas=(0.9, 0.98)
- Learning rate: 5e-5 (safe for fine-tuning)
- Weight decay: 0.2
- Mixed precision training (FP16) on GPU
- Text templates: "a photo of {class_name} sneakers"

## Next Steps

After fine-tuning, you can:
1. Use the model for Nike shoe classification
2. Generate embeddings for your Nike shoe images
3. Build a Nike shoe search engine
4. Compare with the base CLIP model to see improvement
