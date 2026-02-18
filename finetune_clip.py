import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from pathlib import Path
from tqdm import tqdm
import json
from config import (
    DATASET_ROOT,
    BEST_CHECKPOINT_PATH,
    FINAL_CHECKPOINT_PATH,
    TRAINING_HISTORY_PATH,
    MODEL_NAME,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)

# Configuration (from config.py)
device = DEVICE
print(f"Using device: {device}")

# Load CLIP model
model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)

class SneakerDataset(Dataset):
    """Dataset for sneakers with image-text pairs.

    Discovers any directory that directly contains image files and uses that
    directory name as the class label.
    """

    def __init__(self, dataset_root, preprocess):
        self.dataset_root = Path(dataset_root)
        self.preprocess = preprocess
        self.image_paths = []
        self.texts = []

        # Discover class directories (any dir that directly contains images)
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        class_dirs = sorted({
            p.parent
            for p in self.dataset_root.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        })

        print(f"Found {len(class_dirs)} sneaker categories")

        for class_dir in class_dirs:
            # Extract readable class name from directory containing images
            class_name = class_dir.name.replace('_', ' ').title()

            # Get all images directly in this directory
            image_files = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_exts
            ]

            for img_path in image_files:
                self.image_paths.append(img_path)
                # Create descriptive text for the image
                text = f"a photo of {class_name} sneakers"
                self.texts.append(text)

        # Tokenize all texts at once
        self.tokenized_texts = clip.tokenize(self.texts)

        print(f"Total images: {len(self.image_paths)}")
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
    dataset = SneakerDataset(DATASET_ROOT, preprocess)
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Setup model precision
    if device.type == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    # Loss functions
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Optimizer configuration
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=WEIGHT_DECAY
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
    print("TRAINING CONFIGURATION:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Warmup Epochs: {WARMUP_EPOCHS}")
    print("="*70 + "\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

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
            if device.type == "cpu":
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

        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, BEST_CHECKPOINT_PATH)
            print(f"Saved best model (loss: {avg_loss:.4f})")

        # Early stopping if loss plateaus
        if epoch > 10 and avg_loss > 2.5:
            print("\nWARNING: Loss is still very high after 10 epochs.")
            print("    This suggests the model is not learning effectively.")
            print("    Consider checking:")
            print("    1. Dataset quality and labels")
            print("    2. Further reducing learning rate")
            print("    3. Checking for data loading issues")

    # Save final model
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, FINAL_CHECKPOINT_PATH)

    # Save training history
    with open(TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Final model saved to {FINAL_CHECKPOINT_PATH}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    print("="*70)
    print("CLIP Fine-tuning for Sneakers")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print("="*70)

    train()
