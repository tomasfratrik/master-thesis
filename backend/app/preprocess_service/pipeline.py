from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import PreprocessConfig, RuntimeOptions
from .image_plugins import AVIF_PLUGIN_AVAILABLE
from .steps import CropSneakersStep, GrayscaleStep, NormalizeFormatStep, ResizeLimitStep


class PreprocessPipeline:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.normalize_format_step = NormalizeFormatStep(self.config.normalize_format)
        self.resize_limit_step = ResizeLimitStep(self.config.resize_limit)
        self.grayscale_step = GrayscaleStep(self.config.grayscale)
        self.crop_sneakers_step = CropSneakersStep(self.config.crop_sneakers)

    @staticmethod
    def _load_input_image(image_bytes: bytes, filename: str | None = None) -> Image.Image:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.copy()
        except UnidentifiedImageError as exc:
            suffix = Path(filename or "").suffix.lower()
            if suffix == ".avif" and not AVIF_PLUGIN_AVAILABLE:
                raise ValueError(
                    "AVIF decode support is not enabled. Install dependencies from "
                    "`requirements.txt` so `pillow-avif-plugin` is available."
                ) from exc
            raise ValueError("Unsupported or corrupted image input.") from exc

    @staticmethod
    def _step_enabled(
        runtime: RuntimeOptions,
        step_name: str,
        default_enabled: bool,
        runtime_enabled: bool | None,
    ) -> bool:
        if runtime.has_enabled_steps():
            return runtime.should_run_step(step_name)
        if not runtime.should_run_step(step_name):
            return False
        if runtime_enabled is not None:
            return runtime_enabled
        return default_enabled

    def process_single_image(
        self, filename: str, image_bytes: bytes, runtime: RuntimeOptions | None = None
    ) -> dict[str, Any]:
        runtime = runtime or RuntimeOptions()
        base_name = Path(filename or "upload").stem or "upload"
        image = self._load_input_image(image_bytes, filename=filename)

        artifacts: list[dict[str, Any]] = [
            {
                "image": image,
                "name": f"{base_name}.jpg",
                "format": None,
                "suffix": ".jpg",
            }
        ]
        step_metadata: dict[str, Any] = {}
        applied_steps: list[str] = []
        skipped_steps: list[str] = []

        normalize_runtime = runtime.normalize_format
        normalize_step_name = "normalize_format"
        normalize_enabled = self._step_enabled(
            runtime=runtime,
            step_name=normalize_step_name,
            default_enabled=self.config.normalize_format.enabled,
            runtime_enabled=normalize_runtime.enabled,
        )

        if normalize_enabled:
            normalized_artifacts: list[dict[str, Any]] = []
            normalize_runs: list[dict[str, Any]] = []
            for artifact in artifacts:
                normalized_image, normalize_meta, output_format, suffix = (
                    self.normalize_format_step.process(
                        artifact["image"], runtime=normalize_runtime
                    )
                )
                normalize_runs.append(normalize_meta)
                normalized_artifacts.append(
                    {
                        "image": normalized_image,
                        "name": f"{base_name}{suffix}",
                        "format": output_format,
                        "suffix": suffix,
                    }
                )

            if normalized_artifacts:
                artifacts = normalized_artifacts

            applied_steps.append(normalize_step_name)
            step_metadata[normalize_step_name] = normalize_runs
        else:
            skipped_steps.append(normalize_step_name)

        resize_runtime = runtime.resize_limit
        resize_step_name = "resize_limit"
        resize_enabled = self._step_enabled(
            runtime=runtime,
            step_name=resize_step_name,
            default_enabled=self.config.resize_limit.enabled,
            runtime_enabled=resize_runtime.enabled,
        )

        if resize_enabled:
            resized_artifacts: list[dict[str, Any]] = []
            resize_runs: list[dict[str, Any]] = []
            for artifact in artifacts:
                resized_image, resize_meta = self.resize_limit_step.process(
                    artifact["image"], runtime=resize_runtime
                )
                resize_runs.append(resize_meta)
                resized_artifacts.append({**artifact, "image": resized_image})

            if resized_artifacts:
                artifacts = resized_artifacts

            applied_steps.append(resize_step_name)
            step_metadata[resize_step_name] = resize_runs
        else:
            skipped_steps.append(resize_step_name)

        grayscale_runtime = runtime.grayscale
        grayscale_step_name = "grayscale"
        grayscale_enabled = self._step_enabled(
            runtime=runtime,
            step_name=grayscale_step_name,
            default_enabled=self.config.grayscale.enabled,
            runtime_enabled=grayscale_runtime.enabled,
        )

        if grayscale_enabled:
            grayscaled_artifacts: list[dict[str, Any]] = []
            grayscale_runs: list[dict[str, Any]] = []
            for artifact in artifacts:
                gray_image, gray_meta = self.grayscale_step.process(
                    artifact["image"], runtime=grayscale_runtime
                )
                grayscale_runs.append(gray_meta)
                grayscaled_artifacts.append({**artifact, "image": gray_image})

            if grayscaled_artifacts:
                artifacts = grayscaled_artifacts

            applied_steps.append(grayscale_step_name)
            step_metadata[grayscale_step_name] = grayscale_runs
        else:
            skipped_steps.append(grayscale_step_name)

        crop_runtime = runtime.crop_sneakers
        step_name = "crop_sneakers"
        crop_enabled = self._step_enabled(
            runtime=runtime,
            step_name=step_name,
            default_enabled=self.config.crop_sneakers.enabled,
            runtime_enabled=crop_runtime.enabled,
        )

        if crop_enabled:
            next_artifacts: list[dict[str, Any]] = []
            crop_step_runs: list[dict[str, Any]] = []
            for artifact_idx, artifact in enumerate(artifacts, start=1):
                crops, crop_meta = self.crop_sneakers_step.process(
                    artifact["image"], runtime=crop_runtime
                )
                crop_step_runs.append(crop_meta)
                for crop_idx, crop_image in enumerate(crops, start=1):
                    next_artifacts.append(
                        {
                            "image": crop_image,
                            "name": (
                                f"{base_name}_crop_{artifact_idx}_{crop_idx}"
                                f"{artifact['suffix']}"
                            ),
                            "format": artifact["format"],
                            "suffix": artifact["suffix"],
                        }
                    )

            if next_artifacts:
                artifacts = next_artifacts

            applied_steps.append(step_name)
            step_metadata[step_name] = crop_step_runs
        else:
            skipped_steps.append(step_name)

        if runtime.max_outputs_per_image is not None:
            artifacts = artifacts[: runtime.max_outputs_per_image]

        return {
            "filename": filename,
            "outputs": artifacts,
            "metadata": step_metadata,
            "applied_steps": applied_steps,
            "skipped_steps": skipped_steps,
        }
