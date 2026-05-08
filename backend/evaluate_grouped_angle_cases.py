"""Build and evaluate grouped multi-angle sneaker test cases."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image
from tqdm import tqdm

from backend.app.finetuned_classifier_service import FineTunedSneakerClassifier


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIEW_NAMES = ("front_view", "back_view", "side_view", "top_view")
AGGREGATIONS = ("embedding_mean", "logit_mean", "prob_mean")
DEFAULT_COMBINATIONS = {
    "front_back": ("front_view", "back_view"),
    "front_side": ("front_view", "side_view"),
    "front_top": ("front_view", "top_view"),
    "back_side": ("back_view", "side_view"),
    "back_top": ("back_view", "top_view"),
    "side_top": ("side_view", "top_view"),
    "back_back": ("back_view", "back_view"),
    "front_front": ("front_view", "front_view"),
    "side_side": ("side_view", "side_view"),
    "top_top": ("top_view", "top_view"),
    "front_back_top": ("front_view", "back_view", "top_view"),
    "front_back_side": ("front_view", "back_view", "side_view"),
    "front_side_top": ("front_view", "side_view", "top_view"),
    "back_side_top": ("back_view", "side_view", "top_view"),
    "all_views": ("front_view", "back_view", "side_view", "top_view"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped angle cases and evaluate all grouped aggregation modes."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        required=True,
        help="Dataset split root with class folders. Repeat for test and val.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Fine-tuned classifier checkpoint.",
    )
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path("artifacts") / "grouped_test_cases_mixed" / "generated_angle_cases",
        help="Output root for generated grouped test cases.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts") / "grouped_eval_angle_cases.json",
    )
    parser.add_argument(
        "--cases-per-class-combo",
        type=int,
        default=2,
        help="Maximum generated cases per class and angle combination.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images into case-root. By default the report references source images only.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing generated case-root before creating cases.",
    )
    return parser.parse_args()


def detect_view(path: Path) -> str | None:
    name = path.name.lower()
    for view_name in VIEW_NAMES:
        if view_name in name:
            return view_name
    return None


def collect_images(source_roots: list[Path]) -> dict[str, dict[str, list[Path]]]:
    by_class: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        for class_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                view_name = detect_view(image_path)
                if view_name is None:
                    continue
                by_class[class_dir.name][view_name].append(image_path)
    return by_class


def select_images_for_combo(
    view_images: dict[str, list[Path]],
    combination: tuple[str, ...],
    case_index: int,
) -> list[Path] | None:
    selected: list[Path] = []
    used_by_view: dict[str, int] = defaultdict(int)

    for view_name in combination:
        candidates = view_images.get(view_name, [])
        use_index = case_index + used_by_view[view_name]
        if use_index >= len(candidates):
            return None
        selected.append(candidates[use_index])
        used_by_view[view_name] += 1

    return selected


def copy_case_images(
    *,
    case_root: Path,
    class_name: str,
    case_id: str,
    image_paths: list[Path],
) -> list[str]:
    target_dir = case_root / class_name / case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    relative_paths: list[str] = []
    for image_path in image_paths:
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)
        relative_paths.append(target_path.relative_to(case_root).as_posix())
    return relative_paths


def build_cases(
    *,
    source_roots: list[Path],
    case_root: Path,
    cases_per_class_combo: int,
    copy_images: bool,
) -> list[dict[str, Any]]:
    collected = collect_images(source_roots)
    cases: list[dict[str, Any]] = []

    for class_name in sorted(collected):
        view_images = collected[class_name]
        for combination_name, combination in DEFAULT_COMBINATIONS.items():
            for case_number in range(cases_per_class_combo):
                image_paths = select_images_for_combo(view_images, combination, case_number)
                if image_paths is None:
                    continue

                case_id = f"{combination_name}_{case_number + 1:02d}"
                source_paths = [str(path) for path in image_paths]
                if copy_images:
                    case_images = copy_case_images(
                        case_root=case_root,
                        class_name=class_name,
                        case_id=case_id,
                        image_paths=image_paths,
                    )
                else:
                    case_images = source_paths

                cases.append(
                    {
                        "case_id": f"{class_name}/{case_id}",
                        "case_name": case_id,
                        "expected_class": class_name,
                        "combination": combination_name,
                        "views": list(combination),
                        "image_count": len(image_paths),
                        "images": case_images,
                        "source_images": source_paths,
                    }
                )

    return cases


def load_images(case: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in case["source_images"]:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def evaluate_cases(
    *,
    classifier: FineTunedSneakerClassifier,
    cases: list[dict[str, Any]],
    top_k: int,
) -> None:
    for case in tqdm(cases, desc="Grouped angle cases"):
        images = load_images(case)
        results: dict[str, Any] = {}
        for aggregation in AGGREGATIONS:
            result = classifier.predict_images(images, k=top_k, aggregation=aggregation)
            predicted_class = result["class_name"]
            topk_classes = [item["class_name"] for item in result["top_k"]]
            results[aggregation] = {
                "predicted_class": predicted_class,
                "predicted_label": result["label"],
                "top1_score": result["score"],
                "margin_vs_second": result["margin_vs_second"],
                "top1_correct": predicted_class == case["expected_class"],
                "topk_correct": case["expected_class"] in topk_classes,
                "query_image_count": result["query_image_count"],
                "aggregation": aggregation,
                "top_k": result["top_k"],
            }
        case["results"] = results


def summarize_aggregation(cases: list[dict[str, Any]], aggregation: str) -> dict[str, Any]:
    total = len(cases)
    top1_correct = sum(1 for case in cases if case["results"][aggregation]["top1_correct"])
    topk_correct = sum(1 for case in cases if case["results"][aggregation]["topk_correct"])
    scores = [float(case["results"][aggregation]["top1_score"]) for case in cases]
    margins = [float(case["results"][aggregation]["margin_vs_second"]) for case in cases]
    incorrect = [
        case["case_id"]
        for case in cases
        if not case["results"][aggregation]["top1_correct"]
    ]
    return {
        "cases_evaluated": total,
        "top1_correct": top1_correct,
        "top5_correct": topk_correct,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "top5_accuracy": topk_correct / total if total else 0.0,
        "mean_top1_score": mean(scores) if scores else 0.0,
        "mean_margin_vs_second": mean(margins) if margins else 0.0,
        "incorrect_case_ids": incorrect,
    }


def summarize_combination(cases: list[dict[str, Any]], aggregation: str) -> list[dict[str, Any]]:
    by_combination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_combination[case["combination"]].append(case)

    rows: list[dict[str, Any]] = []
    for combination, combination_cases in sorted(by_combination.items()):
        summary = summarize_aggregation(combination_cases, aggregation)
        summary["combination"] = combination
        summary["views"] = list(combination_cases[0]["views"])
        rows.append(summary)
    return rows


def summarize_class(cases: list[dict[str, Any]], aggregation: str) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_class[case["expected_class"]].append(case)

    rows: list[dict[str, Any]] = []
    for class_name, class_cases in sorted(by_class.items()):
        summary = summarize_aggregation(class_cases, aggregation)
        summary["class_name"] = class_name
        rows.append(summary)
    return rows


def best_and_worst(combination_summary: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sorted_best = sorted(
        combination_summary,
        key=lambda item: (
            -float(item["top1_accuracy"]),
            -float(item["mean_margin_vs_second"]),
            item["combination"],
        ),
    )
    sorted_worst = sorted(
        combination_summary,
        key=lambda item: (
            float(item["top1_accuracy"]),
            float(item["mean_margin_vs_second"]),
            item["combination"],
        ),
    )
    return {
        "best_combinations": sorted_best[:5],
        "worst_combinations": sorted_worst[:5],
    }


def write_manifest(case_root: Path, source_roots: list[Path], cases: list[dict[str, Any]]) -> None:
    manifest = {
        "source_roots": [str(path) for path in source_roots],
        "notes": [
            "Cases are generated from class folders and image filenames that include view names.",
            "Cases group images from the same class and view combination.",
            "They are not guaranteed to show the exact same physical sneaker pair.",
        ],
        "cases": [
            {
                "case_id": case["case_id"],
                "expected_class": case["expected_class"],
                "combination": case["combination"],
                "views": case["views"],
                "image_count": case["image_count"],
                "images": case["images"],
                "source_images": case["source_images"],
            }
            for case in cases
        ],
    }
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.overwrite and args.case_root.exists():
        shutil.rmtree(args.case_root)

    cases = build_cases(
        source_roots=args.source_root,
        case_root=args.case_root,
        cases_per_class_combo=args.cases_per_class_combo,
        copy_images=args.copy,
    )
    if not cases:
        raise RuntimeError("No grouped cases could be generated from the provided source roots.")

    write_manifest(args.case_root, args.source_root, cases)

    classifier = FineTunedSneakerClassifier(checkpoint_path=args.checkpoint)
    evaluate_cases(classifier=classifier, cases=cases, top_k=args.top_k)

    per_aggregation = {
        aggregation: summarize_aggregation(cases, aggregation)
        for aggregation in AGGREGATIONS
    }
    per_combination = {
        aggregation: summarize_combination(cases, aggregation)
        for aggregation in AGGREGATIONS
    }
    per_class = {
        aggregation: summarize_class(cases, aggregation)
        for aggregation in AGGREGATIONS
    }
    highlights = {
        aggregation: best_and_worst(per_combination[aggregation])
        for aggregation in AGGREGATIONS
    }

    report = {
        "summary": {
            **classifier.model_summary(),
            "grouped_test_root": str(args.case_root),
            "source_roots": [str(path) for path in args.source_root],
            "cases_evaluated": len(cases),
            "images_evaluated": sum(case["image_count"] for case in cases),
            "top_k": args.top_k,
            "aggregations": list(AGGREGATIONS),
        },
        "per_aggregation": per_aggregation,
        "per_combination": per_combination,
        "per_class": per_class,
        "highlights": highlights,
        "cases": cases,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print("\nPer aggregation:")
    print(json.dumps(per_aggregation, indent=2))
    print("\nBest/worst combinations:")
    print(json.dumps(highlights, indent=2))
    print(f"\nWrote manifest to {args.case_root / 'manifest.json'}")
    print(f"Wrote report to {args.output_json}")


if __name__ == "__main__":
    main()
