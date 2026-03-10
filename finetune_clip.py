import argparse
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    DATALOADER_NUM_WORKERS,
    DEVICE,
    EPOCHS,
    FINAL_CHECKPOINT_PATH,
    LEARNING_RATE,
    MODEL_NAME,
    TEST_SPLIT_ROOT,
    TRAINING_DATA_ROOT,
    TRAINING_HISTORY_PATH,
    TRAIN_SPLIT_ROOT,
    VAL_SPLIT_ROOT,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
device = DEVICE
print(f"Using device: {device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune CLIP on sneaker classes with train/val/test splits."
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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--num-workers", type=int, default=DATALOADER_NUM_WORKERS)
    return parser.parse_args()


def discover_class_dirs(root: Path) -> list[Path]:
    return sorted(
        {
            p.parent
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        }
    )


def format_class_name(class_dir_name: str) -> str:
    return class_dir_name.replace("_", " ").title()


def resolve_dataset_roots(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
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
    """Dataset for sneaker class classification with CLIP text prompts."""

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
        self.class_prompts = [
            f"a photo of {format_class_name(class_name)} sneakers"
            for class_name in self.class_names
        ]

        unknown_classes: set[str] = set()
        for class_dir in class_dirs:
            class_name = class_dir.name
            label = self.class_to_idx.get(class_name)
            if label is None:
                unknown_classes.add(class_name)
                continue

            image_files = sorted(
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
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


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def convert_models_to_fp32(model) -> None:
    for parameter in model.parameters():
        parameter.data = parameter.data.float()
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.float()


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, class_tokens: torch.Tensor) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits_per_image, _ = model(images, class_tokens)
        loss = loss_fn(logits_per_image, labels)
        predictions = logits_per_image.argmax(dim=1)

        total_loss += loss.item() * len(images)
        total_correct += (predictions == labels).sum().item()
        total_samples += len(images)

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(args: argparse.Namespace) -> None:
    train_root, val_root, test_root = resolve_dataset_roots(args)

    model, preprocess = clip.load(MODEL_NAME, device=device, jit=False)
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = (
        create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if val_dataset is not None
        else None
    )
    test_dataloader = (
        create_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if test_dataset is not None
        else None
    )

    class_tokens = clip.tokenize(train_dataset.class_prompts).to(device)

    if device.type == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    def get_lr(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    best_metric = float("inf")
    training_history: list[dict[str, float | int | str | None]] = []

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION:")
    print(f"  Model: {MODEL_NAME}")
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

            logits_per_image, _ = model(images, class_tokens)
            loss = loss_fn(logits_per_image, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if device.type == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

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
        val_metrics = evaluate(model, val_dataloader, class_tokens) if val_dataloader else None
        monitor_loss = val_metrics["loss"] if val_metrics is not None else train_loss

        history_item = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_accuracy": None if val_metrics is None else val_metrics["accuracy"],
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
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": monitor_loss,
                    "class_names": train_dataset.class_names,
                    "train_root": str(train_root),
                    "val_root": None if val_root is None else str(val_root),
                    "test_root": None if test_root is None else str(test_root),
                },
                BEST_CHECKPOINT_PATH,
            )
            print(f"Saved best model to {BEST_CHECKPOINT_PATH} (loss={monitor_loss:.4f})")

    final_metrics = evaluate(model, test_dataloader, class_tokens) if test_dataloader else None

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_metric,
            "class_names": train_dataset.class_names,
            "train_root": str(train_root),
            "val_root": None if val_root is None else str(val_root),
            "test_root": None if test_root is None else str(test_root),
        },
        FINAL_CHECKPOINT_PATH,
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
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print("=" * 70)
    print("CLIP Fine-tuning for Sneakers")
    print("=" * 70)
    print(f"Suggested split root: {TRAINING_DATA_ROOT}")
    print(f"Default train split: {TRAIN_SPLIT_ROOT}")
    print(f"Default val split: {VAL_SPLIT_ROOT}")
    print(f"Default test split: {TEST_SPLIT_ROOT}")
    print("=" * 70)
    train(parse_args())
