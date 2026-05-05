import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from backend.app.finetuned_classifier_service import FineTunedSneakerClassifier
from backend.eval_model import (
    EvaluationImageEncoder,
    ZeroShotSneakerClassifier,
    format_class_label,
    load_checkpoint_class_names,
)

from .rendering import output_directory, save_outputs
from .visualizations import (
    build_classifier_gradcam_map,
    build_classifier_feature_map,
    build_feature_map,
    build_gradcam_map,
    build_reference_target_feature,
    build_text_target_feature,
    load_rgb_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualization maps for CLIP-style image analysis."
    )
    parser.add_argument(
        "--mode",
        choices=["gradcam", "feature_map", "feature_channels"],
        required=True,
        help="Visualization mode to generate.",
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional fine-tuned checkpoint. Omit for base-model analysis.",
    )
    parser.add_argument(
        "--backend",
        choices=["clip", "open_clip"],
        default=None,
        help="Optional encoder backend override.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional encoder model name override.",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Optional pretrained tag for open_clip.",
    )
    parser.add_argument(
        "--prompt-text",
        default=None,
        help="Target prompt text for Grad-CAM similarity analysis.",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Target class name. Converted into a text prompt using the template.",
    )
    parser.add_argument(
        "--prompt-template",
        default="a photo of {label} sneakers",
        help="Prompt template used with --class-name.",
    )
    parser.add_argument(
        "--reference-image",
        nargs="*",
        default=None,
        help="One or more reference images used as the similarity target instead of text.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction and include top-k results in the report.",
    )
    parser.add_argument(
        "--predict-root",
        type=Path,
        default=None,
        help="Root directory containing one folder per class for zero-shot labels.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of predictions to include when --predict is used.",
    )
    parser.add_argument(
        "--block-index",
        type=int,
        default=-2,
        help="Transformer block index used for visualization. -2 is the default for ViT CLIP models.",
    )
    parser.add_argument(
        "--feature-reduction",
        choices=["norm", "mean", "max"],
        default="norm",
        help="Reduction used to convert patch embeddings into a scalar feature map.",
    )
    parser.add_argument(
        "--feature-channel-count",
        type=int,
        default=8,
        help="Number of strongest channels to render for each layer in feature_channels mode.",
    )
    parser.add_argument(
        "--visualization-blur-radius",
        type=float,
        default=4.0,
        help="Extra Gaussian blur radius applied after upsampling the heatmap.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.6,
        help="Overlay strength between 0 and 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "visualizations",
        help="Directory where heatmaps and the report are written.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional output subfolder name. Defaults to <image_stem>_<mode>.",
    )
    return parser.parse_args()


def resolve_prompt_text(args: argparse.Namespace) -> str | None:
    if args.prompt_text:
        return args.prompt_text

    if not args.class_name:
        return None

    label = format_class_label(args.class_name)
    return args.prompt_template.format(class_name=args.class_name, label=label)


def validate_target_args(args: argparse.Namespace, prompt_text: str | None) -> None:
    if args.mode != "gradcam":
        return

    has_prompt_target = prompt_text is not None
    has_reference_target = bool(args.reference_image)
    has_prediction_target = bool(args.predict)

    if has_prompt_target and has_reference_target:
        raise ValueError("Choose either a text target or reference images, not both.")

    if not has_prompt_target and not has_reference_target and not has_prediction_target:
        raise ValueError("Grad-CAM requires --prompt-text, --class-name, --reference-image, or --predict.")


def list_class_names_from_root(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(f"Prediction root not found: {root}")

    class_names: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            class_names.append(path.name)

    if not class_names:
        raise ValueError(f"No class directories found in prediction root: {root}")

    return class_names


def infer_prediction_root(image_path: Path) -> Path | None:
    parent = image_path.parent
    grandparent = parent.parent
    if not grandparent.exists():
        return None

    sibling_directories = [path for path in grandparent.iterdir() if path.is_dir()]
    if not sibling_directories:
        return None

    if parent.name not in {path.name for path in sibling_directories}:
        return None

    return grandparent


def resolve_prediction_class_names(
    *,
    checkpoint_path: str | Path | None,
    predict_root: Path | None,
    image_path: Path,
) -> tuple[list[str], str | None]:
    if predict_root is not None:
        class_names = list_class_names_from_root(predict_root)
        return class_names, str(predict_root)

    inferred_root = infer_prediction_root(image_path)
    if inferred_root is not None:
        class_names = list_class_names_from_root(inferred_root)
        return class_names, str(inferred_root)

    if checkpoint_path is not None:
        class_names = load_checkpoint_class_names(Path(checkpoint_path))
        return class_names, None

    raise ValueError(
        "Could not determine prediction classes. Pass --predict-root or use an image inside a "
        "class-folder dataset."
    )


def run_prediction(
    *,
    image_path: Path,
    checkpoint_path: str | Path | None,
    backend: str | None,
    model_name: str | None,
    pretrained: str | None,
    predict_root: Path | None,
    top_k: int,
) -> tuple[list[dict[str, float | str]], float, str | None, str]:
    if checkpoint_path is not None:
        classifier = FineTunedSneakerClassifier(checkpoint_path=checkpoint_path)
        result = classifier.predict_image_path(image_path, k=top_k)
        predictions = result["top_k"]
        margin = float(result["margin_vs_second"])
        return predictions, margin, None, "app_checkpoint_classifier"

    class_names, resolved_root = resolve_prediction_class_names(
        checkpoint_path=checkpoint_path,
        predict_root=predict_root,
        image_path=image_path,
    )
    classifier = ZeroShotSneakerClassifier(
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        backend=backend,
        model_name=model_name,
        pretrained=pretrained,
    )
    predictions, margin = classifier.predict_image_path(image_path, top_k=top_k)
    return predictions, margin, resolved_root, "zero_shot_classifier"


def class_index_for_name(class_names: list[str], class_name: str) -> int:
    try:
        return class_names.index(class_name)
    except ValueError as error:
        raise ValueError(f"Class {class_name} is not present in the loaded checkpoint.") from error


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    prompt_text = resolve_prompt_text(args)
    validate_target_args(args, prompt_text)

    prediction_results: list[dict[str, float | str]] | None = None
    prediction_margin: float | None = None
    prediction_root: str | None = None
    prediction_source: str | None = None
    predicted_class_name: str | None = None
    predicted_prompt_text: str | None = None

    if args.predict:
        prediction_results, prediction_margin, prediction_root, prediction_source = run_prediction(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            backend=args.backend,
            model_name=args.model_name,
            pretrained=args.pretrained,
            predict_root=args.predict_root,
            top_k=args.top_k,
        )
        predicted_class_name = str(prediction_results[0]["class_name"])
        predicted_prompt_text = args.prompt_template.format(
            class_name=predicted_class_name,
            label=format_class_label(predicted_class_name),
        )

        if args.mode == "gradcam" and prompt_text is None and not args.reference_image:
            prompt_text = predicted_prompt_text

    encoder = EvaluationImageEncoder(
        checkpoint_path=args.checkpoint,
        backend=args.backend,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    image = load_rgb_image(args.image)
    report: dict[str, Any] = {
        **encoder.model_summary(),
        "mode": args.mode,
        "image": str(args.image),
        "visualization_blur_radius": args.visualization_blur_radius,
        "overlay_alpha": args.overlay_alpha,
        "block_index_requested": args.block_index,
        "prediction_enabled": args.predict,
    }

    if args.mode == "gradcam":
        target_mode = "text"
        target_description: dict[str, object] = {}

        if encoder.encoder.supports_classifier_head:
            if args.reference_image:
                raise ValueError("Reference-image Grad-CAM is not supported for image-classifier backends.")

            if args.checkpoint is None:
                raise ValueError("Image-classifier Grad-CAM requires a checkpoint with class_names.")

            class_names = load_checkpoint_class_names(Path(args.checkpoint))
            target_class_name = args.class_name or predicted_class_name
            if target_class_name is None:
                raise ValueError(
                    "Image-classifier Grad-CAM requires --class-name or --predict to select a target class."
                )

            class_index = class_index_for_name(class_names, target_class_name)
            coarse_heatmap, full_heatmap, summary = build_classifier_gradcam_map(
                encoder,
                image,
                class_index=class_index,
                class_name=target_class_name,
                blur_radius=args.visualization_blur_radius,
            )
            target_mode = "classifier_logit"
            target_description["class_name"] = target_class_name
            target_description["class_index"] = class_index
            if predicted_class_name is not None and target_class_name == predicted_class_name:
                target_description["target_source"] = "predicted_top1"
        elif prompt_text is not None:
            target_feature = build_text_target_feature(encoder, prompt_text)
            target_description["prompt_text"] = prompt_text
            if predicted_class_name is not None and prompt_text == predicted_prompt_text:
                target_description["target_source"] = "predicted_top1"
            coarse_heatmap, full_heatmap, summary = build_gradcam_map(
                encoder,
                image,
                target_feature,
                block_index=args.block_index,
                blur_radius=args.visualization_blur_radius,
            )
        else:
            reference_paths = [Path(path) for path in args.reference_image]
            for path in reference_paths:
                if not path.exists():
                    raise FileNotFoundError(f"Reference image not found: {path}")
            target_feature = build_reference_target_feature(encoder, reference_paths)
            target_mode = "reference_images"
            target_description["reference_images"] = [str(path) for path in reference_paths]
            coarse_heatmap, full_heatmap, summary = build_gradcam_map(
                encoder,
                image,
                target_feature,
                block_index=args.block_index,
                blur_radius=args.visualization_blur_radius,
            )
        report["target_mode"] = target_mode
        report["target"] = target_description
    elif args.mode == "feature_map":
        if encoder.encoder.supports_classifier_head:
            coarse_heatmap, full_heatmap, summary = build_classifier_feature_map(
                encoder,
                image,
                blur_radius=args.visualization_blur_radius,
                reduction=args.feature_reduction,
            )
        else:
            coarse_heatmap, full_heatmap, summary = build_feature_map(
                encoder,
                image,
                block_index=args.block_index,
                blur_radius=args.visualization_blur_radius,
                reduction=args.feature_reduction,
            )
        report["feature_reduction"] = args.feature_reduction
    else:
        from .rendering import save_channel_grid
        from .visualizations import build_feature_channel_grid

        channel_grid, summary = build_feature_channel_grid(
            encoder,
            image,
            channel_count=args.feature_channel_count,
        )
        report.update(summary)
        report["feature_channel_count"] = args.feature_channel_count

        target_dir = output_directory(args.output_dir, args.image, args.mode, args.tag)
        save_channel_grid(image, channel_grid, target_dir, report)
        print(json.dumps(report, indent=2))
        print(f"\nSaved visualization outputs to {target_dir}")
        return

    report.update(summary)

    if prediction_results is not None:
        report["prediction"] = {
            "prediction_source": prediction_source,
            "prediction_root": prediction_root,
            "top_k": prediction_results,
            "margin_vs_second": prediction_margin,
        }

    target_dir = output_directory(args.output_dir, args.image, args.mode, args.tag)
    save_outputs(
        image,
        coarse_heatmap,
        full_heatmap,
        target_dir,
        report,
        overlay_alpha=args.overlay_alpha,
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved visualization outputs to {target_dir}")
