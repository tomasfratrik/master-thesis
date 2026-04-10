import argparse
import json
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from backend.config import TEST_SPLIT_ROOT
from backend.eval_model import (
    ZeroShotSneakerClassifier,
    list_labeled_images,
    load_checkpoint_class_names,
)
from backend.tagging_store import load_tag_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP classification on a labeled test split."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to fine-tuned checkpoint. Omit for base-model zero-shot evaluation.",
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
        "--prompt-template",
        default="a photo of {label} sneakers",
        help="Prompt template used for zero-shot class prompts.",
    )
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
    parser.add_argument(
        "--tags-file",
        type=Path,
        default=None,
        help="Optional JSONL file with per-image tags keyed by relative path under test_root.",
    )
    return parser.parse_args()


def predict_scores(
    classifier: ZeroShotSneakerClassifier,
    image_path: Path,
    top_k: int,
) -> tuple[list[dict[str, float | str]], float]:
    return classifier.predict_image_path(image_path, top_k=top_k)


def get_bucket_stats(
    buckets: dict[str, dict[str, int]],
    bucket_name: str,
) -> dict[str, int]:
    if bucket_name not in buckets:
        buckets[bucket_name] = {"total": 0, "top1_correct": 0, "topk_correct": 0}

    return buckets[bucket_name]


def second_best_score(top_predictions: list[dict[str, object]]) -> float:
    if len(top_predictions) < 2:
        return 0.0

    return float(top_predictions[1]["score"])


def build_bucket_summary(buckets: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}

    for bucket_name, stats in sorted(buckets.items()):
        summary[bucket_name] = {
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
    tag_index = load_tag_index(args.tags_file)
    dataset_image_keys = {image_path.relative_to(test_root).as_posix() for image_path, _ in dataset}
    unmatched_tag_entries = sorted(set(tag_index) - dataset_image_keys)

    class_names = (
        load_checkpoint_class_names(checkpoint_path)
        if checkpoint_path is not None
        else sorted({class_name for _, class_name in dataset})
    )
    classifier = ZeroShotSneakerClassifier(
        class_names=class_names,
        prompt_template=args.prompt_template,
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
    per_tag: dict[str, dict[str, int]] = {}
    failures: list[dict[str, object]] = []
    failures_by_tag: dict[str, list[dict[str, object]]] = {}
    tagged_images = 0
    untagged_images = 0

    for image_path, expected_class in tqdm(dataset, desc="Evaluating"):
        top_predictions, margin = predict_scores(
            classifier=classifier,
            image_path=image_path,
            top_k=args.top_k,
        )
        image_key = image_path.relative_to(test_root).as_posix()
        image_tags = tag_index.get(image_key, [])
        predicted_class = str(top_predictions[0]["class_name"])
        topk_classes = [str(item["class_name"]) for item in top_predictions]
        is_top1 = predicted_class == expected_class
        is_topk = expected_class in topk_classes

        if image_tags:
            tagged_images += 1
        else:
            untagged_images += 1

        stats = get_bucket_stats(per_class, expected_class)
        stats["total"] += 1

        for tag in image_tags:
            tag_stats = get_bucket_stats(per_tag, tag)
            tag_stats["total"] += 1

        if is_top1:
            top1_correct += 1
            stats["top1_correct"] += 1
            for tag in image_tags:
                per_tag[tag]["top1_correct"] += 1
            correct_margins.append(margin)
        else:
            incorrect_margins.append(margin)
            failure = {
                "image": str(image_path),
                "image_key": image_key,
                "tags": image_tags,
                "expected_class": expected_class,
                "predicted_class": predicted_class,
                "top1_score": float(top_predictions[0]["score"]),
                "second_best_score": second_best_score(top_predictions),
                "margin_vs_second": margin,
                "top_k": top_predictions,
            }
            failures.append(failure)
            for tag in image_tags:
                failures_by_tag.setdefault(tag, []).append(failure)

        if is_topk:
            topk_correct += 1
            stats["topk_correct"] += 1
            for tag in image_tags:
                per_tag[tag]["topk_correct"] += 1

        top1_scores.append(float(top_predictions[0]["score"]))
        top1_margins.append(margin)

    per_class_summary = build_bucket_summary(per_class)
    per_tag_summary = build_bucket_summary(per_tag)

    class_source = "test_root"
    if checkpoint_path is not None:
        class_source = "checkpoint"

    mean_margin_when_correct = 0.0
    if correct_margins:
        mean_margin_when_correct = mean(correct_margins)

    mean_margin_when_wrong = 0.0
    if incorrect_margins:
        mean_margin_when_wrong = mean(incorrect_margins)

    summary = {
        **classifier.model_summary(),
        "test_root": str(test_root),
        "class_count": len(class_names),
        "class_source": class_source,
        "prompt_template": args.prompt_template,
        "tags_file": None if args.tags_file is None else str(args.tags_file),
        "tag_entries_loaded": len(tag_index),
        "tagged_images": tagged_images,
        "untagged_images": untagged_images,
        "unmatched_tag_entries": len(unmatched_tag_entries),
        "images_evaluated": total,
        "classes_evaluated": len(per_class_summary),
        "tags_evaluated": len(per_tag_summary),
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
        "per_tag": per_tag_summary,
        "unmatched_tag_entries": unmatched_tag_entries[:100],
        "hardest_errors_by_tag": {
            tag: sorted(
                tag_failures,
                key=lambda item: (item["margin_vs_second"], -float(item["top1_score"])),
            )[:10]
            for tag, tag_failures in sorted(failures_by_tag.items())
        },
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

    if per_tag_summary:
        print("\nPer-tag summary:")
        for tag, stats in sorted(
            per_tag_summary.items(),
            key=lambda item: item[1]["top1_accuracy"],
        ):
            print(
                f"- {tag}: total={stats['total']} "
                f"top1={stats['top1_accuracy']:.4f} "
                f"top{args.top_k}={stats['topk_accuracy']:.4f}"
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
        print(f"\nSaved report to {args.output_json}")


if __name__ == "__main__":
    main()
