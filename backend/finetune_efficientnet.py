"""Fine-tune an EfficientNet-B0 image classifier on sneaker classes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backend.config import (
    DATALOADER_NUM_WORKERS,
    DEVICE,
    FINETUNE_OUTPUT_DIR,
    TEST_SPLIT_ROOT,
    TRAINING_DATA_ROOT,
    TRAIN_SPLIT_ROOT,
    VAL_SPLIT_ROOT,
    WARMUP_EPOCHS,
)
from backend.model_loader import load_encoder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BEST_CHECKPOINT_PATH = FINETUNE_OUTPUT_DIR / "efficientnet_b0_sneaker_best.pt"
FINAL_CHECKPOINT_PATH = FINETUNE_OUTPUT_DIR / "efficientnet_b0_sneaker_final.pt"
TRAINING_HISTORY_PATH = FINETUNE_OUTPUT_DIR / "efficientnet_b0_training_history.json"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
device = DEVICE


def best_checkpoint_path_for_epoch(epoch: int) -> Path:
    """Return the epoch-specific path for the currently best checkpoint."""
    return FINETUNE_OUTPUT_DIR / f"efficientnet_b0_sneaker_best_epoch_{epoch}.pt"


def checkpoint_path_for_epoch(epoch: int) -> Path:
    """Return a regular epoch checkpoint path for visual diagnostics."""
    return FINETUNE_OUTPUT_DIR / f"efficientnet_b0_sneaker_epoch_{epoch}.pt"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for EfficientNet sneaker fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B0 on sneaker classes with train/val/test splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TRAINING_DATA_ROOT,
        help=(
            "Dataset root. Can be a split root containing train/val/test folders, "
            "or a single class-folder dataset."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Explicit train split root. Overrides auto-detection.",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Explicit validation split root. Overrides auto-detection.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=None,
        help="Explicit test split root. Overrides auto-detection.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=DATALOADER_NUM_WORKERS)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional EfficientNet checkpoint to continue training from.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Save a regular epoch checkpoint every N epochs. "
            "Use 1 when you want underfit/best/overfit checkpoints for visualizations."
        ),
    )
    return parser.parse_args()


def discover_class_dirs(root: Path) -> list[Path]:
    """Return class directories that contain at least one image file."""
    return sorted(
        {
            image_path.parent
            for image_path in root.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS
        }
    )


def resolve_dataset_roots(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    """Resolve train, validation, and test roots from CLI arguments."""
    if args.train_root is not None:
        train_root = args.train_root
        val_root = args.val_root
        test_root = args.test_root
    else:
        data_root = args.data_root
        if (data_root / "train").is_dir():
            train_root = data_root / "train"
            val_root = data_root / "val" if (data_root / "val").is_dir() else None
            test_root = data_root / "test" if (data_root / "test").is_dir() else None
        else:
            train_root = data_root
            val_root = args.val_root
            test_root = args.test_root

    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    if val_root is not None and not val_root.exists():
        raise FileNotFoundError(f"Validation root not found: {val_root}")
    if test_root is not None and not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    return train_root, val_root, test_root


class SneakerDataset(Dataset):
    """Dataset for closed-set sneaker image classification."""

    def __init__(self, dataset_root: Path, preprocess, class_names: list[str] | None = None):
        self.dataset_root = Path(dataset_root)
        self.preprocess = preprocess
        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        class_dirs = discover_class_dirs(self.dataset_root)
        if not class_dirs:
            raise RuntimeError(f"No class directories with images found under {self.dataset_root}")

        discovered_names = sorted({class_dir.name for class_dir in class_dirs})
        self.class_names = class_names or discovered_names
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        unknown_classes: set[str] = set()
        for class_dir in class_dirs:
            class_name = class_dir.name
            label = self.class_to_idx.get(class_name)
            if label is None:
                unknown_classes.add(class_name)
                continue

            image_files = sorted(
                image_path
                for image_path in class_dir.iterdir()
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS
            )
            self.image_paths.extend(image_files)
            self.labels.extend([label] * len(image_files))

        if unknown_classes:
            raise RuntimeError(
                "Found classes outside the training label set: "
                + ", ".join(sorted(unknown_classes))
            )
        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.dataset_root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return self.preprocess(image), self.labels[idx]
        except Exception as error:
            raise RuntimeError(f"Failed to load image {self.image_paths[idx]}") from error


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Build a DataLoader with sensible defaults for local training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader | None) -> dict[str, float] | None:
    """Evaluate the classifier on a validation or test split."""
    if dataloader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    monitor_loss: float,
    train_dataset: SneakerDataset,
    train_root: Path,
    val_root: Path | None,
    test_root: Path | None,
) -> None:
    """Persist an EfficientNet checkpoint with loader metadata."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": monitor_loss,
            "class_names": train_dataset.class_names,
            "model_backend": "efficientnet_b0",
            "model_family": "image_classifier",
            "feature_dim": 1280,
            "train_root": str(train_root),
            "val_root": None if val_root is None else str(val_root),
            "test_root": None if test_root is None else str(test_root),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    """Run EfficientNet-B0 fine-tuning and save checkpoints/history."""
    train_root, val_root, test_root = resolve_dataset_roots(args)
    print(f"Using device: {device}")

    encoder = load_encoder(
        backend="efficientnet_b0",
        model_name="efficientnet_b0",
        device=device,
        use_checkpoint=False,
    )
    model = encoder.model
    preprocess = encoder.preprocess

    if args.init_checkpoint is not None:
        encoder.load_checkpoint_weights(
            args.init_checkpoint,
            map_location="cpu",
            strict=False,
        )
        print(f"Loaded init checkpoint from {args.init_checkpoint} via CPU deserialization")

    train_dataset = SneakerDataset(train_root, preprocess)
    val_dataset = (
        SneakerDataset(val_root, preprocess, class_names=train_dataset.class_names)
        if val_root is not None
        else None
    )
    test_dataset = (
        SneakerDataset(test_root, preprocess, class_names=train_dataset.class_names)
        if test_root is not None
        else None
    )

    train_dataloader = create_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = (
        create_dataloader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_dataset is not None
        else None
    )
    test_dataloader = (
        create_dataloader(
            test_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if test_dataset is not None
        else None
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    def get_lr(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    best_metric = float("inf")
    training_history: list[dict[str, float | int | None]] = []

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION:")
    print("  Model: efficientnet_b0")
    print(f"  Init checkpoint: {args.init_checkpoint}")
    print(f"  Train root: {train_root}")
    print(f"  Val root: {val_root}")
    print(f"  Test root: {test_root}")
    print(f"  Classes: {len(train_dataset.class_names)}")
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Val images: {len(val_dataset) if val_dataset is not None else 0}")
    print(f"  Test images: {len(test_dataset) if test_dataset is not None else 0}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Checkpoint every: {args.checkpoint_every}")
    print("=" * 70 + "\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, labels in progress_bar:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = len(images)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        scheduler.step()

        train_loss = epoch_loss / total_samples if total_samples else 0.0
        val_metrics = evaluate(model, val_dataloader)
        monitor_loss = train_loss if val_metrics is None else float(val_metrics["loss"])

        history_item = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": None if val_metrics is None else float(val_metrics["loss"]),
            "val_accuracy": None if val_metrics is None else float(val_metrics["accuracy"]),
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_history.append(history_item)

        status = [f"Epoch {epoch + 1}/{args.epochs}", f"train_loss={train_loss:.4f}"]
        if val_metrics is not None:
            status.append(f"val_loss={val_metrics['loss']:.4f}")
            status.append(f"val_acc={val_metrics['accuracy']:.4f}")
        print(" - ".join(status))

        if monitor_loss < best_metric:
            best_metric = monitor_loss
            best_epoch_path = best_checkpoint_path_for_epoch(epoch + 1)
            _save_checkpoint(
                best_epoch_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                monitor_loss=monitor_loss,
                train_dataset=train_dataset,
                train_root=train_root,
                val_root=val_root,
                test_root=test_root,
            )
            _save_checkpoint(
                BEST_CHECKPOINT_PATH,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                monitor_loss=monitor_loss,
                train_dataset=train_dataset,
                train_root=train_root,
                val_root=val_root,
                test_root=test_root,
            )
            print(f"Saved best model to {best_epoch_path} (loss={monitor_loss:.4f})")
            print(f"Updated stable best checkpoint alias at {BEST_CHECKPOINT_PATH}")

        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            epoch_path = checkpoint_path_for_epoch(epoch + 1)
            _save_checkpoint(
                epoch_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                monitor_loss=monitor_loss,
                train_dataset=train_dataset,
                train_root=train_root,
                val_root=val_root,
                test_root=test_root,
            )
            print(f"Saved epoch checkpoint to {epoch_path}")

    final_metrics = evaluate(model, test_dataloader)
    _save_checkpoint(
        FINAL_CHECKPOINT_PATH,
        epoch=args.epochs,
        model=model,
        optimizer=optimizer,
        monitor_loss=best_metric,
        train_dataset=train_dataset,
        train_root=train_root,
        val_root=val_root,
        test_root=test_root,
    )

    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(training_history, file, indent=2)

    print(f"\n{'=' * 70}")
    print("Training completed")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Final checkpoint: {FINAL_CHECKPOINT_PATH}")
    if final_metrics is not None:
        print(
            f"Test loss: {final_metrics['loss']:.4f} | "
            f"Test accuracy: {final_metrics['accuracy']:.4f}"
        )
    print(f"Training history: {TRAINING_HISTORY_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print("=" * 70)
    print("EfficientNet-B0 Fine-tuning for Sneakers")
    print("=" * 70)
    print(f"Suggested split root: {TRAINING_DATA_ROOT}")
    print(f"Default train split: {TRAIN_SPLIT_ROOT}")
    print(f"Default val split: {VAL_SPLIT_ROOT}")
    print(f"Default test split: {TEST_SPLIT_ROOT}")
    print("=" * 70)
    train(parse_args())
