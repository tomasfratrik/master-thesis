import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from tqdm import tqdm

from config import TEST_SPLIT_ROOT
from finetuned_classifier_service import FineTunedSneakerClassifier


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned sneaker checkpoint on a labeled test split."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint.")
    parser.add_argument(
        "--test-root",
        type=Path,
        default=TEST_SPLIT_ROOT,
        help="Root directory with one folder per class.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k accuracy to report.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full evaluation report as JSON.",
    )
    return parser.parse_args()


def list_labeled_images(root: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                items.append((image_path, class_name))
    return items


@torch.no_grad()
def predict_scores(
    classifier: FineTunedSneakerClassifier,
    image_path: Path,
    top_k: int,
) -> tuple[list[dict[str, float | str]], float]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = classifier.preprocess(image).unsqueeze(0).to(classifier.device)
    image_features = classifier.model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    probabilities = (100.0 * image_features @ classifier.text_features.T).softmax(dim=-1)[0]
    k = max(1, min(int(top_k), len(classifier.class_names)))
    top_scores, top_indices = probabilities.topk(k)

    top_predictions: list[dict[str, float | str]] = []
    for score, index in zip(top_scores.tolist(), top_indices.tolist()):
        class_name = classifier.class_names[index]
        top_predictions.append(
            {
                "class_name": class_name,
                "label": class_name.replace("_", " ").title(),
                "score": float(score),
            }
        )

    second_best_score = float(top_predictions[1]["score"]) if len(top_predictions) > 1 else 0.0
    margin = float(top_predictions[0]["score"]) - second_best_score
    return top_predictions, margin


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    test_root = Path(args.test_root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    dataset = list_labeled_images(test_root)
    if not dataset:
        raise RuntimeError(f"No labeled images found under {test_root}")
    if args.limit is not None:
        dataset = dataset[: args.limit]

    classifier = FineTunedSneakerClassifier(checkpoint_path=checkpoint_path)

    total = len(dataset)
    top1_correct = 0
    topk_correct = 0
    top1_scores: list[float] = []
    top1_margins: list[float] = []
    correct_margins: list[float] = []
    incorrect_margins: list[float] = []
    per_class: dict[str, dict[str, int]] = {}
    failures: list[dict[str, object]] = []

    for image_path, expected_class in tqdm(dataset, desc="Evaluating"):
        top_predictions, margin = predict_scores(
            classifier=classifier,
            image_path=image_path,
            top_k=args.top_k,
        )
        predicted_class = str(top_predictions[0]["class_name"])
        topk_classes = [str(item["class_name"]) for item in top_predictions]
        is_top1 = predicted_class == expected_class
        is_topk = expected_class in topk_classes

        stats = per_class.setdefault(expected_class, {"total": 0, "top1_correct": 0, "topk_correct": 0})
        stats["total"] += 1

        if is_top1:
            top1_correct += 1
            stats["top1_correct"] += 1
            correct_margins.append(margin)
        else:
            incorrect_margins.append(margin)
            failures.append(
                {
                    "image": str(image_path),
                    "expected_class": expected_class,
                    "predicted_class": predicted_class,
                    "top1_score": float(top_predictions[0]["score"]),
                    "second_best_score": float(top_predictions[1]["score"]) if len(top_predictions) > 1 else 0.0,
                    "margin_vs_second": margin,
                    "top_k": top_predictions,
                }
            )

        if is_topk:
            topk_correct += 1
            stats["topk_correct"] += 1

        top1_scores.append(float(top_predictions[0]["score"]))
        top1_margins.append(margin)

    per_class_summary = {
        class_name: {
            **stats,
            "top1_accuracy": stats["top1_correct"] / stats["total"],
            "topk_accuracy": stats["topk_correct"] / stats["total"],
        }
        for class_name, stats in sorted(per_class.items())
    }

    summary = {
        "checkpoint": str(checkpoint_path),
        "test_root": str(test_root),
        "images_evaluated": total,
        "classes_evaluated": len(per_class_summary),
        "top1_accuracy": top1_correct / total,
        f"top{args.top_k}_accuracy": topk_correct / total,
        "mean_top1_score": mean(top1_scores),
        "mean_margin_vs_second": mean(top1_margins),
        "mean_margin_when_correct": mean(correct_margins) if correct_margins else 0.0,
        "mean_margin_when_wrong": mean(incorrect_margins) if incorrect_margins else 0.0,
        "errors": len(failures),
    }

    report = {
        "summary": summary,
        "per_class": per_class_summary,
        "hardest_errors": sorted(
            failures,
            key=lambda item: (item["margin_vs_second"], -float(item["top1_score"])),
        )[:25],
    }

    print(json.dumps(summary, indent=2))

    if failures:
        print("\nTop errors by smallest margin:")
        for item in report["hardest_errors"][:10]:
            print(
                f"- expected={item['expected_class']} predicted={item['predicted_class']} "
                f"score={item['top1_score']:.4f} margin={item['margin_vs_second']:.4f} "
                f"path={item['image']}"
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
