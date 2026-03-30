import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from tqdm import tqdm

from backend.config import (
    PRECOMPUTED_IMAGE_EMBEDDINGS,
    PRECOMPUTED_IMAGE_METADATA,
    TEST_SPLIT_ROOT,
)
from backend.eval_model import EvaluationImageEncoder, format_class_label, list_labeled_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-level embedding retrieval on a labeled test split."
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
        "--top-images-per-class",
        type=int,
        default=3,
        help="Number of best matching train-image similarities to average per class.",
    )
    parser.add_argument(
        "--image-embeddings",
        type=Path,
        default=PRECOMPUTED_IMAGE_EMBEDDINGS,
        help="Path to precomputed image_embeddings.npy",
    )
    parser.add_argument(
        "--image-metadata",
        type=Path,
        default=PRECOMPUTED_IMAGE_METADATA,
        help="Path to precomputed image_meta.json",
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


def load_catalog(image_embeddings_path: Path, image_metadata_path: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    if image_embeddings_path is None or not image_embeddings_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {image_embeddings_path}")
    if image_metadata_path is None or not image_metadata_path.exists():
        raise FileNotFoundError(f"Image metadata not found: {image_metadata_path}")

    embeddings = np.load(image_embeddings_path).astype("float32")
    metadata = json.loads(image_metadata_path.read_text())
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embedding/image metadata length mismatch: {len(embeddings)} vs {len(metadata)}"
        )
    return embeddings, metadata


def _metadata_class(meta: dict[str, object]) -> str:
    return str(meta.get("class_name") or meta.get("class"))


def _metadata_model_info(meta: dict[str, object]) -> dict[str, object]:
    return {
        "embedding_backend": meta.get("embedding_backend"),
        "embedding_model_name": meta.get("embedding_model_name"),
        "embedding_pretrained": meta.get("embedding_pretrained"),
        "embedding_checkpoint": meta.get("embedding_checkpoint"),
    }


def predict_scores(
    encoder: EvaluationImageEncoder,
    image_path: Path,
    image_embeddings: np.ndarray,
    image_metadata: list[dict[str, object]],
    *,
    top_k: int,
    top_images_per_class: int,
) -> tuple[list[dict[str, object]], float]:
    query_feature = encoder.build_prototype_from_image_paths([image_path]).cpu().numpy().astype("float32")

    image_scores = image_embeddings @ query_feature
    per_class_scores: dict[str, list[float]] = defaultdict(list)
    for score, meta in zip(image_scores.tolist(), image_metadata):
        per_class_scores[_metadata_class(meta)].append(float(score))

    ranked_classes: list[dict[str, object]] = []
    for class_name, scores in per_class_scores.items():
        top_scores = sorted(scores, reverse=True)[: max(1, top_images_per_class)]
        aggregated_score = float(sum(top_scores) / len(top_scores))
        ranked_classes.append(
            {
                "class_name": class_name,
                "label": format_class_label(class_name),
                "score": aggregated_score,
            }
        )

    ranked_classes.sort(key=lambda item: float(item["score"]), reverse=True)
    top_predictions = ranked_classes[: max(1, min(int(top_k), len(ranked_classes)))]
    second_best_score = float(top_predictions[1]["score"]) if len(top_predictions) > 1 else 0.0
    margin = float(top_predictions[0]["score"]) - second_best_score
    return top_predictions, margin


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

    image_embeddings, image_metadata = load_catalog(
        image_embeddings_path=Path(args.image_embeddings),
        image_metadata_path=Path(args.image_metadata),
    )
    encoder = EvaluationImageEncoder(
        checkpoint_path=checkpoint_path,
        backend=args.backend,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    total = len(dataset)
    top1_correct = 0
    topk_correct = 0
    top1_scores: list[float] = []
    top1_margins: list[float] = []
    correct_margins: list[float] = []
    incorrect_margins: list[float] = []
    per_class: dict[str, dict[str, int]] = {}
    failures: list[dict[str, object]] = []

    for image_path, expected_class in tqdm(dataset, desc="Evaluating image retrieval"):
        top_predictions, margin = predict_scores(
            encoder=encoder,
            image_path=image_path,
            image_embeddings=image_embeddings,
            image_metadata=image_metadata,
            top_k=args.top_k,
            top_images_per_class=args.top_images_per_class,
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
        **encoder.model_summary(),
        "test_root": str(test_root),
        "catalog_images": int(image_embeddings.shape[0]),
        "catalog_classes": len({_metadata_class(item) for item in image_metadata}),
        "catalog_embedding_model": _metadata_model_info(image_metadata[0]) if image_metadata else {},
        "top_images_per_class": args.top_images_per_class,
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
