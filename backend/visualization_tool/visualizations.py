import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.eval_model import EvaluationImageEncoder

from .rendering import normalize_map, render_channel_grid, upsample_heatmap


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
            "Visualization is currently implemented only for transformer-based visual encoders."
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
    def __init__(self, block: torch.nn.Module, *, retain_gradients: bool) -> None:
        self.activations: torch.Tensor | None = None
        self.retain_gradients = retain_gradients
        self.handle = block.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output
        if self.retain_gradients:
            output.retain_grad()

    def close(self) -> None:
        self.handle.remove()

    def patch_activations(self) -> torch.Tensor:
        if self.activations is None:
            raise RuntimeError("Visualization activations were not captured.")

        token_features = self.activations.permute(1, 0, 2)[0]
        return token_features[1:]

    def patch_gradients(self) -> torch.Tensor:
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Gradients were not captured.")

        token_gradients = self.activations.grad.permute(1, 0, 2)[0]
        return token_gradients[1:]


class CnnActivationRecorder:
    def __init__(self, layer: torch.nn.Module, *, retain_gradients: bool) -> None:
        self.activations: torch.Tensor | None = None
        self.retain_gradients = retain_gradients
        self.handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output
        if self.retain_gradients:
            output.retain_grad()

    def close(self) -> None:
        self.handle.remove()

    def feature_activations(self) -> torch.Tensor:
        if self.activations is None:
            raise RuntimeError("CNN activations were not captured.")
        return self.activations[0]

    def feature_gradients(self) -> torch.Tensor:
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("CNN Grad-CAM gradients were not captured.")
        return self.activations.grad[0]


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


def classifier_score(
    encoder: EvaluationImageEncoder,
    image_tensor: torch.Tensor,
    class_index: int,
) -> torch.Tensor:
    logits = encoder.encoder.model(image_tensor)
    if logits.ndim != 2:
        raise ValueError(f"Expected classifier logits with shape [batch, classes], got {logits.shape}.")
    if class_index < 0 or class_index >= logits.shape[1]:
        raise ValueError(
            f"Class index {class_index} is outside classifier output range 0..{logits.shape[1] - 1}."
        )
    return logits[0, class_index]


def resolve_cnn_target_layer(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError("The loaded classifier does not expose a CNN feature extractor.")

    for index in range(len(features) - 1, -1, -1):
        layer = features[index]
        if any(isinstance(module, torch.nn.Conv2d) for module in layer.modules()):
            return layer, f"features.{index}"

    raise ValueError("Could not find a convolutional target layer for CNN Grad-CAM.")


def reduce_cnn_activations(feature_activations: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "norm":
        return torch.norm(feature_activations, dim=0)
    if reduction == "mean":
        return feature_activations.mean(dim=0)
    if reduction == "max":
        return feature_activations.max(dim=0).values
    raise ValueError(f"Unsupported feature reduction: {reduction}")


def strongest_channel_indices(activations: torch.Tensor, channel_count: int) -> torch.Tensor:
    if activations.ndim != 3:
        raise ValueError(f"Expected channel activations with shape [channels, height, width], got {activations.shape}.")

    count = max(1, min(int(channel_count), activations.shape[0]))
    scores = activations.abs().mean(dim=(1, 2))
    return scores.topk(count).indices


def cnn_feature_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    features = getattr(model, "features", None)
    if features is None:
        return []

    layers: list[tuple[str, torch.nn.Module]] = []
    for index, layer in enumerate(features):
        if any(isinstance(module, torch.nn.Conv2d) for module in layer.modules()):
            layers.append((f"features.{index}", layer))
    return layers


def transformer_feature_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    visual = getattr(model, "visual", None)
    transformer = getattr(visual, "transformer", None) if visual is not None else None
    blocks = getattr(transformer, "resblocks", None) if transformer is not None else None
    if blocks is None:
        return []

    return [(f"resblock.{index}", block) for index, block in enumerate(blocks)]


def grid_side_length(token_count: int) -> int:
    side_length = int(math.sqrt(token_count))
    if side_length * side_length != token_count:
        raise ValueError(
            "Visualization expected a square patch grid but got "
            f"{token_count} patch tokens instead."
        )
    return side_length


def reduce_patch_activations(patch_activations: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "norm":
        return torch.norm(patch_activations, dim=-1)
    if reduction == "mean":
        return patch_activations.mean(dim=-1)
    if reduction == "max":
        return patch_activations.max(dim=-1).values
    raise ValueError(f"Unsupported feature reduction: {reduction}")


def build_feature_map(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    *,
    block_index: int,
    blur_radius: float,
    reduction: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    image_tensor = encoder.preprocess(image).unsqueeze(0).to(encoder.device)
    model = encoder.encoder.model
    model.eval()

    block, resolved_block_index, block_count = resolve_visual_block(model, block_index)
    recorder = ActivationRecorder(block, retain_gradients=False)

    try:
        with torch.no_grad():
            encoder.encoder.encode_image_tensors(image_tensor)
        patch_activations = recorder.patch_activations()
        token_scores = reduce_patch_activations(patch_activations, reduction)
        side_length = grid_side_length(token_scores.shape[0])
        coarse_heatmap = token_scores.reshape(side_length, side_length).detach().cpu().numpy()
        coarse_heatmap = normalize_map(coarse_heatmap)
        full_heatmap = upsample_heatmap(coarse_heatmap, image.size, blur_radius)
    finally:
        recorder.close()

    summary = {
        "block_index": int(resolved_block_index),
        "block_count": int(block_count),
        "patch_rows": int(coarse_heatmap.shape[0]),
        "patch_columns": int(coarse_heatmap.shape[1]),
        "heatmap_max": float(full_heatmap.max()),
        "heatmap_mean": float(full_heatmap.mean()),
        "feature_reduction": reduction,
    }
    return coarse_heatmap, full_heatmap, summary


def build_classifier_feature_map(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    *,
    blur_radius: float,
    reduction: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    image_tensor = encoder.preprocess(image).unsqueeze(0).to(encoder.device)
    model = encoder.encoder.model
    model.eval()

    target_layer, target_layer_name = resolve_cnn_target_layer(model)
    recorder = CnnActivationRecorder(target_layer, retain_gradients=False)

    try:
        with torch.no_grad():
            model(image_tensor)
        feature_activations = recorder.feature_activations()
        coarse_tensor = reduce_cnn_activations(feature_activations, reduction)
        coarse_heatmap = coarse_tensor.detach().cpu().numpy()
        coarse_heatmap = normalize_map(coarse_heatmap)
        full_heatmap = upsample_heatmap(coarse_heatmap, image.size, blur_radius)
    finally:
        recorder.close()

    summary = {
        "target_layer": target_layer_name,
        "feature_rows": int(coarse_heatmap.shape[0]),
        "feature_columns": int(coarse_heatmap.shape[1]),
        "heatmap_max": float(full_heatmap.max()),
        "heatmap_mean": float(full_heatmap.mean()),
        "feature_reduction": reduction,
    }
    return coarse_heatmap, full_heatmap, summary


def build_classifier_feature_channel_grid(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    *,
    channel_count: int,
) -> tuple[Image.Image, dict[str, int | str]]:
    image_tensor = encoder.preprocess(image).unsqueeze(0).to(encoder.device)
    model = encoder.encoder.model
    model.eval()

    layers = cnn_feature_layers(model)
    if not layers:
        raise ValueError("The loaded classifier does not expose CNN feature layers.")

    recorders = [
        (name, CnnActivationRecorder(layer, retain_gradients=False))
        for name, layer in layers
    ]
    try:
        with torch.no_grad():
            model(image_tensor)

        layer_maps: list[tuple[str, list[np.ndarray]]] = []
        for layer_name, recorder in recorders:
            activations = recorder.feature_activations()
            indices = strongest_channel_indices(activations, channel_count)
            channel_maps = [
                activations[index].detach().cpu().numpy()
                for index in indices.tolist()
            ]
            layer_maps.append((layer_name, channel_maps))
    finally:
        for _, recorder in recorders:
            recorder.close()

    grid = render_channel_grid(layer_maps)
    summary = {
        "feature_channel_source": "cnn",
        "feature_layer_count": len(layer_maps),
        "channels_per_layer": min(max(1, int(channel_count)), max(1, max(len(maps) for _, maps in layer_maps))),
    }
    return grid, summary


def build_transformer_feature_channel_grid(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    *,
    channel_count: int,
) -> tuple[Image.Image, dict[str, int | str]]:
    image_tensor = encoder.preprocess(image).unsqueeze(0).to(encoder.device)
    model = encoder.encoder.model
    model.eval()

    blocks = transformer_feature_blocks(model)
    if not blocks:
        raise ValueError("The loaded model does not expose transformer feature blocks.")

    recorders = [
        (name, ActivationRecorder(block, retain_gradients=False))
        for name, block in blocks
    ]
    try:
        with torch.no_grad():
            encoder.encoder.encode_image_tensors(image_tensor)

        layer_maps: list[tuple[str, list[np.ndarray]]] = []
        for layer_name, recorder in recorders:
            patch_activations = recorder.patch_activations()
            side_length = grid_side_length(patch_activations.shape[0])
            channel_first = patch_activations.T.reshape(patch_activations.shape[1], side_length, side_length)
            indices = strongest_channel_indices(channel_first, channel_count)
            channel_maps = [
                channel_first[index].detach().cpu().numpy()
                for index in indices.tolist()
            ]
            layer_maps.append((layer_name, channel_maps))
    finally:
        for _, recorder in recorders:
            recorder.close()

    grid = render_channel_grid(layer_maps)
    summary = {
        "feature_channel_source": "transformer_tokens",
        "feature_layer_count": len(layer_maps),
        "channels_per_layer": min(max(1, int(channel_count)), max(1, max(len(maps) for _, maps in layer_maps))),
    }
    return grid, summary


def build_feature_channel_grid(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    *,
    channel_count: int,
) -> tuple[Image.Image, dict[str, int | str]]:
    if encoder.encoder.supports_classifier_head:
        return build_classifier_feature_channel_grid(
            encoder,
            image,
            channel_count=channel_count,
        )

    return build_transformer_feature_channel_grid(
        encoder,
        image,
        channel_count=channel_count,
    )


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
    recorder = ActivationRecorder(block, retain_gradients=True)

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


def build_classifier_gradcam_map(
    encoder: EvaluationImageEncoder,
    image: Image.Image,
    *,
    class_index: int,
    class_name: str,
    blur_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    image_tensor = encoder.preprocess(image).unsqueeze(0).to(encoder.device)
    model = encoder.encoder.model
    model.eval()

    target_layer, target_layer_name = resolve_cnn_target_layer(model)
    recorder = CnnActivationRecorder(target_layer, retain_gradients=True)

    try:
        score = classifier_score(encoder, image_tensor, class_index)
        model.zero_grad(set_to_none=True)
        score.backward()

        activations = recorder.feature_activations()
        gradients = recorder.feature_gradients()
        channel_weights = gradients.mean(dim=(1, 2))
        coarse_tensor = torch.sum(activations * channel_weights[:, None, None], dim=0)
        coarse_tensor = torch.relu(coarse_tensor)
        coarse_heatmap = coarse_tensor.detach().cpu().numpy()
        coarse_heatmap = normalize_map(coarse_heatmap)
        full_heatmap = upsample_heatmap(coarse_heatmap, image.size, blur_radius)
    finally:
        recorder.close()

    summary = {
        "score": float(score.item()),
        "target_layer": target_layer_name,
        "target_class_index": int(class_index),
        "target_class_name": class_name,
        "feature_rows": int(coarse_heatmap.shape[0]),
        "feature_columns": int(coarse_heatmap.shape[1]),
        "heatmap_max": float(full_heatmap.max()),
        "heatmap_mean": float(full_heatmap.mean()),
    }
    return coarse_heatmap, full_heatmap, summary
