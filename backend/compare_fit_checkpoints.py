"""Compare checkpoints on train/validation/test splits to diagnose fit quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from backend.app.finetuned_classifier_service import FineTunedSneakerClassifier
from backend.eval_model import list_labeled_images


BAR_COLORS = [
    (43, 111, 173),
    (214, 91, 52),
    (45, 150, 86),
    (120, 90, 160),
    (230, 160, 40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more checkpoints on train/val/test splits. "
            "This is useful for showing underfitting and overfitting."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint to evaluate. Repeat for epoch_1, best, final, etc.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label for a checkpoint. Repeat once per --checkpoint.",
    )
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--val-root", type=Path, default=None)
    parser.add_argument("--test-root", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split image limit for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "training_plots",
    )
    parser.add_argument("--prefix", default="fit_gap")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <output-dir>/<prefix>.json.",
    )
    return parser.parse_args()


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}
    return checkpoint


def resolve_split_roots(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Path]:
    configured = {
        "train": args.train_root,
        "val": args.val_root,
        "test": args.test_root,
    }
    metadata_keys = {
        "train": "train_root",
        "val": "val_root",
        "test": "test_root",
    }

    roots: dict[str, Path] = {}
    for split_name, configured_root in configured.items():
        root = configured_root
        if root is None:
            metadata_value = metadata.get(metadata_keys[split_name])
            if metadata_value:
                root = Path(str(metadata_value))

        if root is not None:
            roots[split_name] = root

    return roots


def predict_scores(
    classifier: FineTunedSneakerClassifier,
    image_path: Path,
    top_k: int,
) -> tuple[list[dict[str, float | str]], float]:
    result = classifier.predict_image_path(image_path, k=top_k, aggregation="embedding_mean")
    return result["top_k"], float(result["margin_vs_second"])


def evaluate_split(
    *,
    classifier: FineTunedSneakerClassifier,
    split_name: str,
    split_root: Path,
    top_k: int,
    limit: int | None,
) -> dict[str, float | int | str]:
    if not split_root.exists():
        raise FileNotFoundError(f"{split_name} root not found: {split_root}")

    dataset = list_labeled_images(split_root)
    if limit is not None:
        dataset = dataset[:limit]
    if not dataset:
        raise RuntimeError(f"No labeled images found under {split_root}")

    top1_correct = 0
    topk_correct = 0
    top1_scores: list[float] = []
    margins: list[float] = []

    for image_path, expected_class in tqdm(dataset, desc=f"{split_name}: {split_root.name}"):
        top_predictions, margin = predict_scores(classifier, image_path, top_k)
        predicted_class = str(top_predictions[0]["class_name"])
        topk_classes = [str(item["class_name"]) for item in top_predictions]

        if predicted_class == expected_class:
            top1_correct += 1
        if expected_class in topk_classes:
            topk_correct += 1

        top1_scores.append(float(top_predictions[0]["score"]))
        margins.append(margin)

    total = len(dataset)
    return {
        "split": split_name,
        "root": str(split_root),
        "images": total,
        "top1_accuracy": top1_correct / total,
        f"top{top_k}_accuracy": topk_correct / total,
        "mean_top1_score": mean(top1_scores),
        "mean_margin_vs_second": mean(margins),
        "errors": total - top1_correct,
    }


def draw_grouped_bars(
    *,
    report: dict[str, Any],
    output_path: Path,
    metric: str,
    title: str,
) -> None:
    runs = report["runs"]
    split_names = ["train", "val", "test"]
    width = 1100
    height = 650
    left = 90
    top = 80
    plot_width = 840
    plot_height = 430

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    text = (35, 35, 35)
    axis = (40, 40, 40)
    grid = (225, 225, 225)

    draw.text((left, 25), title, fill=text)
    draw.line((left, top, left, top + plot_height), fill=axis, width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill=axis, width=2)

    for tick in range(6):
        ratio = tick / 5
        y = top + round(ratio * plot_height)
        value = 1.0 - ratio
        draw.line((left, y, left + plot_width, y), fill=grid)
        draw.text((left - 45, y - 7), f"{value:.1f}", fill=text)

    group_width = plot_width / len(split_names)
    bar_gap = 8
    bar_width = max(12, int((group_width - 50) / max(1, len(runs)) - bar_gap))

    for split_index, split_name in enumerate(split_names):
        group_left = left + split_index * group_width
        label_x = int(group_left + group_width / 2 - 18)
        draw.text((label_x, top + plot_height + 18), split_name, fill=text)

        for run_index, run in enumerate(runs):
            split_result = run["splits"].get(split_name)
            if split_result is None:
                continue
            value = float(split_result[metric])
            x0 = int(group_left + 25 + run_index * (bar_width + bar_gap))
            x1 = x0 + bar_width
            y1 = top + plot_height
            y0 = y1 - round(value * plot_height)
            color = BAR_COLORS[run_index % len(BAR_COLORS)]
            draw.rectangle((x0, y0, x1, y1), fill=color)
            draw.text((x0, y0 - 16), f"{value:.2f}", fill=text)

    for index, run in enumerate(runs):
        item_y = 90 + index * 25
        color = BAR_COLORS[index % len(BAR_COLORS)]
        draw.rectangle((960, item_y + 4, 974, item_y + 18), fill=color)
        draw.text((980, item_y), run["label"], fill=text)

    notes = [
        "Interpretation:",
        "underfitting: low train accuracy and low val/test accuracy",
        "overfitting: high train accuracy but clearly lower val/test accuracy",
        "good fit: train and val/test are both high and close together",
    ]
    for index, note in enumerate(notes):
        draw.text((left, 560 + index * 20), note, fill=text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    if args.label and len(args.label) != len(args.checkpoint):
        raise ValueError("Use the same number of --label values as --checkpoint values.")

    runs: list[dict[str, Any]] = []
    for index, checkpoint_path in enumerate(args.checkpoint):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        label = args.label[index] if args.label else checkpoint_path.stem
        metadata = load_checkpoint_metadata(checkpoint_path)
        split_roots = resolve_split_roots(args, metadata)
        if not split_roots:
            raise ValueError(
                f"No split roots available for {checkpoint_path}. "
                "Pass --train-root, --val-root, or --test-root."
            )

        classifier = FineTunedSneakerClassifier(checkpoint_path=checkpoint_path)
        split_results: dict[str, Any] = {}
        for split_name in ["train", "val", "test"]:
            split_root = split_roots.get(split_name)
            if split_root is None:
                continue
            split_results[split_name] = evaluate_split(
                classifier=classifier,
                split_name=split_name,
                split_root=split_root,
                top_k=args.top_k,
                limit=args.limit,
            )

        runs.append(
            {
                "label": label,
                "checkpoint": str(checkpoint_path),
                "epoch": metadata.get("epoch"),
                "splits": split_results,
            }
        )

    report = {"top_k": args.top_k, "runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or args.output_dir / f"{args.prefix}.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plot_path = args.output_dir / f"{args.prefix}_top1_accuracy.png"
    draw_grouped_bars(
        report=report,
        output_path=plot_path,
        metric="top1_accuracy",
        title="Train / Validation / Test Accuracy by Checkpoint",
    )

    print(json.dumps(report, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
