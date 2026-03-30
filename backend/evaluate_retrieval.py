import argparse
import json
from pathlib import Path
from statistics import mean

from PIL import Image
from tqdm import tqdm

from backend.app.embedding_retrieval import CatalogEmbeddingRetrieval
from backend.config import TEST_SPLIT_ROOT
from backend.eval_model import EvaluationImageEncoder, list_labeled_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embedding-based sneaker retrieval on a labeled test split."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to fine-tuned checkpoint. Omit for base-model retrieval evaluation.",
    )
    parser.add_argument(
        "--backend",
        choices=["clip", "open_clip"],
        default=None,
        help="Optional encoder backend override. Falls back to config/env defaults.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional encoder model name override. Falls back to config/env defaults.",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional pretrained tag for open_clip. Falls back to config/env defaults.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=TEST_SPLIT_ROOT,
        help="Root directory with one folder per class.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k accuracy to report.")
    parser.add_argument(
        "--class-aggregation",
        choices=["max", "topn_mean"],
        default="max",
        help="How to aggregate multiple image matches into one class score.",
    )
    parser.add_argument(
        "--top-n-per-class",
        type=int,
        default=3,
        help="When using topn_mean, average this many best image matches per class.",
    )
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


def predict_scores(
    retrieval: CatalogEmbeddingRetrieval,
    image_path: Path,
    top_k: int,
) -> tuple[list[dict[str, object]], float]:
    with Image.open(image_path) as image:
        prediction = retrieval.search_images([image.convert("RGB")], k=top_k)

    top_predictions = prediction["top_k"]
    second_best_score = float(top_predictions[1]["score"]) if len(top_predictions) > 1 else 0.0
    margin = float(top_predictions[0]["score"]) - second_best_score
    return top_predictions, margin


def get_class_stats(
    per_class: dict[str, dict[str, int]],
    class_name: str,
) -> dict[str, int]:
    if class_name not in per_class:
        per_class[class_name] = {"total": 0, "top1_correct": 0, "topk_correct": 0}

    return per_class[class_name]


def second_best_score(top_predictions: list[dict[str, object]]) -> float:
    if len(top_predictions) < 2:
        return 0.0

    return float(top_predictions[1]["score"])


def build_per_class_summary(per_class: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}

    for class_name, stats in sorted(per_class.items()):
        summary[class_name] = {
            "total": stats["total"],
            "top1_correct": stats["top1_correct"],
            "topk_correct": stats["topk_correct"],
            "top1_accuracy": stats["top1_correct"] / stats["total"],
            "topk_accuracy": stats["topk_correct"] / stats["total"],
        }

    return summary


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    test_root = Path(args.test_root)

    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    dataset = list_labeled_images(test_root)
    if not dataset:
        raise RuntimeError(f"No labeled images found under {test_root}")
    if args.limit is not None:
        dataset = dataset[: args.limit]

    query_encoder = EvaluationImageEncoder(
        checkpoint_path=checkpoint_path,
        backend=args.backend,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )
    retrieval = CatalogEmbeddingRetrieval(
        query_encoder,
        class_aggregation=args.class_aggregation,
        top_n_per_class=args.top_n_per_class,
    )
    if not retrieval.entries:
        raise RuntimeError("No retrieval catalog entries are available.")

    total = len(dataset)
    top1_correct = 0
    topk_correct = 0
    top1_scores: list[float] = []
    top1_margins: list[float] = []
    correct_margins: list[float] = []
    incorrect_margins: list[float] = []
    per_class: dict[str, dict[str, int]] = {}
    failures: list[dict[str, object]] = []

    for image_path, expected_class in tqdm(dataset, desc="Evaluating retrieval"):
        top_predictions, margin = predict_scores(
            retrieval=retrieval,
            image_path=image_path,
            top_k=args.top_k,
        )
        predicted_class = str(top_predictions[0]["class_name"])
        topk_classes = [str(item["class_name"]) for item in top_predictions]
        is_top1 = predicted_class == expected_class
        is_topk = expected_class in topk_classes

        stats = get_class_stats(per_class, expected_class)
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
                    "second_best_score": second_best_score(top_predictions),
                    "margin_vs_second": margin,
                    "top_k": top_predictions,
                }
            )

        if is_topk:
            topk_correct += 1
            stats["topk_correct"] += 1

        top1_scores.append(float(top_predictions[0]["score"]))
        top1_margins.append(margin)

    per_class_summary = build_per_class_summary(per_class)

    mean_margin_when_correct = 0.0
    if correct_margins:
        mean_margin_when_correct = mean(correct_margins)

    mean_margin_when_wrong = 0.0
    if incorrect_margins:
        mean_margin_when_wrong = mean(incorrect_margins)

    summary = {
        **query_encoder.model_summary(),
        "test_root": str(test_root),
        "catalog_entries": retrieval.catalog_size,
        "catalog_rows": retrieval.row_count,
        "catalog_mode": retrieval.entry_mode,
        "class_aggregation": args.class_aggregation,
        "top_n_per_class": args.top_n_per_class,
        "images_evaluated": total,
        "classes_evaluated": len(per_class_summary),
        "top1_accuracy": top1_correct / total,
        f"top{args.top_k}_accuracy": topk_correct / total,
        "mean_top1_score": mean(top1_scores),
        "mean_margin_vs_second": mean(top1_margins),
        "mean_margin_when_correct": mean_margin_when_correct,
        "mean_margin_when_wrong": mean_margin_when_wrong,
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
