import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageFilter

from backend.eval_model import EvaluationImageEncoder, format_class_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM style heatmaps for CLIP-style image similarity."
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
        help="Target prompt text for similarity analysis.",
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
        "--block-index",
        type=int,
        default=-2,
        help="Transformer block index used for Grad-CAM. -2 is the default for ViT CLIP models.",
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
        default=Path("artifacts") / "gradcam",
        help="Directory where heatmaps and the report are written.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional output subfolder name. Defaults to the input image stem.",
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
    has_prompt_target = prompt_text is not None
    has_reference_target = bool(args.reference_image)

    if has_prompt_target and has_reference_target:
        raise ValueError("Choose either a text target or reference images, not both.")

    if not has_prompt_target and not has_reference_target:
        raise ValueError("Pass --prompt-text, --class-name, or --reference-image.")


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def normalize_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim > 1:
        feature = feature.squeeze(0)

    feature = feature.float()
    return feature / feature.norm(dim=-1, keepdim=True)


@torch.no_grad()
def build_text_target_feature(encoder: EvaluationImageEncoder, prompt_text: str) -> torch.Tensor:
    text_tokens = encoder.encoder.tokenize_texts([prompt_text])
    text_features = encoder.encoder.encode_text_tokens(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return normalize_feature(text_features[0]).to(encoder.device)


@torch.no_grad()
def build_reference_target_feature(
    encoder: EvaluationImageEncoder,
    reference_images: list[Path],
) -> torch.Tensor:
    feature = encoder.build_prototype_from_image_paths(reference_images)
    return normalize_feature(feature).to(encoder.device)


def resolve_visual_block(model: torch.nn.Module, block_index: int) -> tuple[torch.nn.Module, int, int]:
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("The loaded model does not expose a visual encoder.")

    transformer = getattr(visual, "transformer", None)
    if transformer is None:
        raise ValueError(
            "Grad-CAM is currently implemented only for transformer-based visual encoders."
        )

    blocks = getattr(transformer, "resblocks", None)
    if blocks is None:
        raise ValueError("The visual transformer does not expose residual attention blocks.")

    block_count = len(blocks)
    try:
        block = blocks[block_index]
    except IndexError as error:
        raise ValueError(
            f"Invalid --block-index {block_index}. Model has {block_count} transformer blocks."
        ) from error

    resolved_index = block_index
    if resolved_index < 0:
        resolved_index = block_count + resolved_index

    return block, resolved_index, block_count


class ActivationRecorder:
    def __init__(self, block: torch.nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.handle = block.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output
        output.retain_grad()

    def close(self) -> None:
        self.handle.remove()

    def patch_activations(self) -> torch.Tensor:
        if self.activations is None:
            raise RuntimeError("Grad-CAM activations were not captured.")

        token_features = self.activations.permute(1, 0, 2)[0]
        return token_features[1:]

    def patch_gradients(self) -> torch.Tensor:
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Grad-CAM gradients were not captured.")

        token_gradients = self.activations.grad.permute(1, 0, 2)[0]
        return token_gradients[1:]


def similarity_score(
    encoder: EvaluationImageEncoder,
    image_tensor: torch.Tensor,
    target_feature: torch.Tensor,
) -> torch.Tensor:
    image_features = encoder.encoder.encode_image_tensors(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_feature = image_features[0].float()
    target_vector = target_feature.to(image_feature.device).float()
    return 100.0 * torch.dot(image_feature, target_vector)


def grid_side_length(token_count: int) -> int:
    side_length = int(math.sqrt(token_count))
    if side_length * side_length != token_count:
        raise ValueError(
            "Grad-CAM expected a square patch grid but got "
            f"{token_count} patch tokens instead."
        )
    return side_length


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)

    return (values - minimum) / (maximum - minimum)


def upsample_heatmap(
    coarse_heatmap: np.ndarray,
    image_size: tuple[int, int],
    blur_radius: float,
) -> np.ndarray:
    heatmap_image = Image.fromarray((coarse_heatmap * 255.0).astype(np.uint8), mode="L")
    upsampled_image = heatmap_image.resize(image_size, resample=Image.Resampling.BICUBIC)

    if blur_radius > 0:
        upsampled_image = upsampled_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    upsampled_values = np.asarray(upsampled_image, dtype=np.float32) / 255.0
    return normalize_map(upsampled_values)


def heatmap_rgb(normalized_heatmap: np.ndarray) -> np.ndarray:
    channel_positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    red_values = np.array([0, 0, 0, 255, 255], dtype=np.float32)
    green_values = np.array([0, 128, 255, 255, 0], dtype=np.float32)
    blue_values = np.array([0, 255, 255, 0, 0], dtype=np.float32)

    flattened = normalized_heatmap.reshape(-1)
    red = np.interp(flattened, channel_positions, red_values)
    green = np.interp(flattened, channel_positions, green_values)
    blue = np.interp(flattened, channel_positions, blue_values)

    rgb = np.stack([red, green, blue], axis=-1)
    return rgb.reshape(normalized_heatmap.shape[0], normalized_heatmap.shape[1], 3).astype(np.uint8)


def heatmap_rgba(normalized_heatmap: np.ndarray, overlay_alpha: float) -> Image.Image:
    rgb = heatmap_rgb(normalized_heatmap)
    alpha_scale = max(0.0, min(float(overlay_alpha), 1.0))
    alpha = (normalized_heatmap * 255.0 * alpha_scale).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def overlay_image(image: Image.Image, heatmap: Image.Image) -> Image.Image:
    return Image.alpha_composite(image.convert("RGBA"), heatmap)


def build_gradcam_map(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    target_feature: torch.Tensor,
    *,
    block_index: int,
    blur_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    image_tensor = encoder.preprocess(image).unsqueeze(0).to(encoder.device)
    model = encoder.encoder.model
    model.eval()

    block, resolved_block_index, block_count = resolve_visual_block(model, block_index)
    recorder = ActivationRecorder(block)

    try:
        score = similarity_score(encoder, image_tensor, target_feature)
        model.zero_grad(set_to_none=True)
        score.backward()

        patch_activations = recorder.patch_activations()
        patch_gradients = recorder.patch_gradients()

        channel_weights = patch_gradients.mean(dim=0)
        token_scores = torch.sum(patch_activations * channel_weights.unsqueeze(0), dim=-1)
        token_scores = torch.relu(token_scores)

        side_length = grid_side_length(token_scores.shape[0])
        coarse_heatmap = token_scores.reshape(side_length, side_length).detach().cpu().numpy()
        coarse_heatmap = normalize_map(coarse_heatmap)
        full_heatmap = upsample_heatmap(coarse_heatmap, image.size, blur_radius)
    finally:
        recorder.close()

    summary = {
        "score": float(score.item()),
        "block_index": int(resolved_block_index),
        "block_count": int(block_count),
        "patch_rows": int(coarse_heatmap.shape[0]),
        "patch_columns": int(coarse_heatmap.shape[1]),
        "heatmap_max": float(full_heatmap.max()),
        "heatmap_mean": float(full_heatmap.mean()),
    }
    return coarse_heatmap, full_heatmap, summary


def output_directory(base_dir: Path, image_path: Path, tag: str | None) -> Path:
    folder_name = image_path.stem
    if tag:
        folder_name = tag

    target = base_dir / folder_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_outputs(
    image: Image.Image,
    coarse_heatmap: np.ndarray,
    full_heatmap: np.ndarray,
    output_dir: Path,
    report: dict[str, object],
    *,
    overlay_alpha: float,
) -> None:
    original_path = output_dir / "original.png"
    heatmap_path = output_dir / "heatmap.png"
    overlay_path = output_dir / "overlay.png"
    coarse_values_path = output_dir / "heatmap_coarse.npy"
    full_values_path = output_dir / "heatmap.npy"
    report_path = output_dir / "report.json"

    rendered_heatmap = heatmap_rgba(full_heatmap, overlay_alpha)

    image.save(original_path)
    rendered_heatmap.save(heatmap_path)
    overlay_image(image, rendered_heatmap).save(overlay_path)
    np.save(coarse_values_path, coarse_heatmap.astype(np.float32))
    np.save(full_values_path, full_heatmap.astype(np.float32))
    report_path.write_text(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    prompt_text = resolve_prompt_text(args)
    validate_target_args(args, prompt_text)

    encoder = EvaluationImageEncoder(
        checkpoint_path=args.checkpoint,
        backend=args.backend,
        model_name=args.model_name,
        pretrained=args.pretrained,
    )

    image = load_rgb_image(args.image)
    target_mode = "text"
    target_description: dict[str, object] = {}

    if prompt_text is not None:
        target_feature = build_text_target_feature(encoder, prompt_text)
        target_description["prompt_text"] = prompt_text
    else:
        reference_paths = [Path(path) for path in args.reference_image]
        for path in reference_paths:
            if not path.exists():
                raise FileNotFoundError(f"Reference image not found: {path}")
        target_feature = build_reference_target_feature(encoder, reference_paths)
        target_mode = "reference_images"
        target_description["reference_images"] = [str(path) for path in reference_paths]

    coarse_heatmap, full_heatmap, gradcam_summary = build_gradcam_map(
        encoder,
        image,
        target_feature,
        block_index=args.block_index,
        blur_radius=args.visualization_blur_radius,
    )

    target_dir = output_directory(args.output_dir, args.image, args.tag)
    report = {
        **encoder.model_summary(),
        "image": str(args.image),
        "target_mode": target_mode,
        "target": target_description,
        "visualization_blur_radius": args.visualization_blur_radius,
        "overlay_alpha": args.overlay_alpha,
        **gradcam_summary,
    }
    save_outputs(
        image,
        coarse_heatmap,
        full_heatmap,
        target_dir,
        report,
        overlay_alpha=args.overlay_alpha,
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved Grad-CAM outputs to {target_dir}")


if __name__ == "__main__":
    main()
