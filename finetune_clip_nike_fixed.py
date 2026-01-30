import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from pathlib import Path
from tqdm import tqdm
import json

# Configuration - FIXED HYPERPARAMETERS
BATCH_SIZE = 32  # Increased from 16
EPOCH = 30  # Increased from 10
LEARNING_RATE = 1e-6  # MUCH lower - was 5e-5 (50x reduction!)
WARMUP_EPOCHS = 3
MODEL_NAME = "ViT-B/32"
DATASET_ROOT = Path("sneakers-dataset")
OUTPUT_DIR = Path("artifacts/finetuned_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)

class NikeSneakerDataset(Dataset):
    """Dataset for Nike sneakers with image-text pairs"""

    def __init__(self, dataset_root, preprocess):
        self.dataset_root = Path(dataset_root)
        self.preprocess = preprocess
        self.image_paths = []
        self.texts = []

        # Collect all Nike shoe images
        nike_dirs = sorted([d for d in self.dataset_root.iterdir()
                           if d.is_dir() and d.name.startswith('nike_')])

        print(f"Found {len(nike_dirs)} Nike shoe categories")

        for nike_dir in nike_dirs:
            # Extract readable class name
            class_name = nike_dir.name.replace('_', ' ').replace('nike ', '').title()

            # Get all images in this directory
            image_files = list(nike_dir.glob('*.jpg')) + list(nike_dir.glob('*.png'))

            for img_path in image_files:
                self.image_paths.append(img_path)
                # Create descriptive text for the image
                text = f"a photo of {class_name} sneakers"
                self.texts.append(text)

        # Tokenize all texts at once
        self.tokenized_texts = clip.tokenize(self.texts)

        print(f"Total Nike images: {len(self.image_paths)}")
        print(f"Sample texts: {self.texts[:3]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = self.preprocess(Image.open(self.image_paths[idx]))
            text = self.tokenized_texts[idx]
            return image, text
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return a valid default item
            return self.preprocess(Image.new('RGB', (224, 224))), self.tokenized_texts[0]

def convert_models_to_fp32(model):
    """Convert model parameters to FP32 for optimizer step"""
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def train():
    # Create dataset and dataloader
    dataset = NikeSneakerDataset(DATASET_ROOT, preprocess)
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Setup model precision
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    # Loss functions
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # FIXED: Much lower learning rate and no weight decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.001  # Reduced from 0.2 to 0.001
    )

    # Learning rate scheduler with warmup
    def get_lr(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Training loop
    best_loss = float('inf')
    training_history = []

    print("\n" + "="*70)
    print("FIXED TRAINING CONFIGURATION:")
    print(f"  Learning Rate: {LEARNING_RATE} (was 5e-5 - now 50x lower!)")
    print(f"  Weight Decay: 0.001 (was 0.2 - now 200x lower!)")
    print(f"  Batch Size: {BATCH_SIZE} (was 16)")
    print(f"  Epochs: {EPOCH} (was 10)")
    print(f"  Warmup Epochs: {WARMUP_EPOCHS}")
    print("="*70 + "\n")

    for epoch in range(EPOCH):
        model.train()
        epoch_loss = 0
        batch_count = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCH}")

        for batch in progress_bar:
            optimizer.zero_grad()

            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Create ground truth labels
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            # Calculate loss
            total_loss = (loss_img(logits_per_image, ground_truth) +
                         loss_txt(logits_per_text, ground_truth)) / 2

            # Backward pass
            total_loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with FP32 conversion if on GPU
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            # Track metrics
            epoch_loss += total_loss.item()
            batch_count += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        # Step scheduler
        scheduler.step()

        # Calculate average loss for epoch
        avg_loss = epoch_loss / batch_count
        training_history.append({
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{EPOCH} - Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = OUTPUT_DIR / f"clip_nike_best_fixed.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model (loss: {avg_loss:.4f})")

        # Early stopping if loss plateaus
        if epoch > 10 and avg_loss > 2.5:
            print("\n⚠️  WARNING: Loss is still very high after 10 epochs.")
            print("    This suggests the model is not learning effectively.")
            print("    Consider checking:")
            print("    1. Dataset quality and labels")
            print("    2. Further reducing learning rate")
            print("    3. Checking for data loading issues")

    # Save final model
    final_path = OUTPUT_DIR / f"clip_nike_final_fixed.pt"
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_path)

    # Save training history
    history_path = OUTPUT_DIR / "training_history_fixed.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Final model saved to {final_path}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    print("="*70)
    print("CLIP Fine-tuning for Nike Sneakers (FIXED VERSION)")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCH}")
    print(f"Learning rate: {LEARNING_RATE} (FIXED - much lower!)")
    print(f"Weight decay: 0.001 (FIXED - much lower!)")
    print("="*70)

    train()
