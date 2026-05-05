from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .config import PreprocessConfig, RuntimeOptions
from .image_plugins import AVIF_PLUGIN_AVAILABLE
from .steps import CropSneakersStep, GrayscaleStep, NormalizeFormatStep, ResizeLimitStep


class PreprocessPipeline:
    KNOWN_STEPS = {
        "normalize_format",
        "resize_limit",
        "grayscale",
        "crop_sneakers",
    }

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

    @staticmethod
    def _normalize_step_order(step_order: list[str]) -> list[str]:
        normalized_order = [step.strip().lower() for step in step_order if step and step.strip()]
        unknown_steps = sorted(set(normalized_order) - PreprocessPipeline.KNOWN_STEPS)
        if unknown_steps:
            raise ValueError(f"Unknown preprocess step(s): {', '.join(unknown_steps)}")

        ordered_steps: list[str] = []
        for step in normalized_order:
            if step not in ordered_steps:
                ordered_steps.append(step)

        for step in PreprocessPipeline.KNOWN_STEPS:
            if step not in ordered_steps:
                ordered_steps.append(step)

        return ordered_steps

    def _runtime_step_order(self, runtime: RuntimeOptions) -> list[str]:
        if runtime.step_order:
            return self._normalize_step_order(runtime.step_order)
        return self._normalize_step_order(self.config.step_order)

    def _run_normalize_format(
        self,
        *,
        artifacts: list[dict[str, Any]],
        base_name: str,
        runtime: RuntimeOptions,
        step_metadata: dict[str, Any],
        applied_steps: list[str],
        skipped_steps: list[str],
    ) -> list[dict[str, Any]]:
        step_name = "normalize_format"
        step_runtime = runtime.normalize_format
        enabled = self._step_enabled(
            runtime=runtime,
            step_name=step_name,
            default_enabled=self.config.normalize_format.enabled,
            runtime_enabled=step_runtime.enabled,
        )

        if not enabled:
            skipped_steps.append(step_name)
            return artifacts

        next_artifacts: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for artifact in artifacts:
            normalized_image, normalize_meta, output_format, suffix = self.normalize_format_step.process(
                artifact["image"], runtime=step_runtime
            )
            runs.append(normalize_meta)
            next_artifacts.append(
                {
                    "image": normalized_image,
                    "name": f"{base_name}{suffix}",
                    "format": output_format,
                    "suffix": suffix,
                }
            )

        applied_steps.append(step_name)
        step_metadata[step_name] = runs
        return next_artifacts or artifacts

    def _run_resize_limit(
        self,
        *,
        artifacts: list[dict[str, Any]],
        runtime: RuntimeOptions,
        step_metadata: dict[str, Any],
        applied_steps: list[str],
        skipped_steps: list[str],
    ) -> list[dict[str, Any]]:
        step_name = "resize_limit"
        step_runtime = runtime.resize_limit
        enabled = self._step_enabled(
            runtime=runtime,
            step_name=step_name,
            default_enabled=self.config.resize_limit.enabled,
            runtime_enabled=step_runtime.enabled,
        )

        if not enabled:
            skipped_steps.append(step_name)
            return artifacts

        next_artifacts: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for artifact in artifacts:
            resized_image, resize_meta = self.resize_limit_step.process(
                artifact["image"], runtime=step_runtime
            )
            runs.append(resize_meta)
            next_artifacts.append({**artifact, "image": resized_image})

        applied_steps.append(step_name)
        step_metadata[step_name] = runs
        return next_artifacts or artifacts

    def _run_grayscale(
        self,
        *,
        artifacts: list[dict[str, Any]],
        runtime: RuntimeOptions,
        step_metadata: dict[str, Any],
        applied_steps: list[str],
        skipped_steps: list[str],
    ) -> list[dict[str, Any]]:
        step_name = "grayscale"
        step_runtime = runtime.grayscale
        enabled = self._step_enabled(
            runtime=runtime,
            step_name=step_name,
            default_enabled=self.config.grayscale.enabled,
            runtime_enabled=step_runtime.enabled,
        )

        if not enabled:
            skipped_steps.append(step_name)
            return artifacts

        next_artifacts: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for artifact in artifacts:
            gray_image, gray_meta = self.grayscale_step.process(
                artifact["image"], runtime=step_runtime
            )
            runs.append(gray_meta)
            next_artifacts.append({**artifact, "image": gray_image})

        applied_steps.append(step_name)
        step_metadata[step_name] = runs
        return next_artifacts or artifacts

    def _run_crop_sneakers(
        self,
        *,
        artifacts: list[dict[str, Any]],
        base_name: str,
        runtime: RuntimeOptions,
        step_metadata: dict[str, Any],
        applied_steps: list[str],
        skipped_steps: list[str],
    ) -> list[dict[str, Any]]:
        step_name = "crop_sneakers"
        step_runtime = runtime.crop_sneakers
        enabled = self._step_enabled(
            runtime=runtime,
            step_name=step_name,
            default_enabled=self.config.crop_sneakers.enabled,
            runtime_enabled=step_runtime.enabled,
        )

        if not enabled:
            skipped_steps.append(step_name)
            return artifacts

        next_artifacts: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for artifact_idx, artifact in enumerate(artifacts, start=1):
            crops, crop_meta = self.crop_sneakers_step.process(
                artifact["image"], runtime=step_runtime
            )
            runs.append(crop_meta)
            for crop_idx, crop_image in enumerate(crops, start=1):
                next_artifacts.append(
                    {
                        "image": crop_image,
                        "name": f"{base_name}_crop_{artifact_idx}_{crop_idx}{artifact['suffix']}",
                        "format": artifact["format"],
                        "suffix": artifact["suffix"],
                    }
                )

        applied_steps.append(step_name)
        step_metadata[step_name] = runs
        return next_artifacts or artifacts

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

        step_runners = {
            "normalize_format": self._run_normalize_format,
            "resize_limit": self._run_resize_limit,
            "grayscale": self._run_grayscale,
            "crop_sneakers": self._run_crop_sneakers,
        }

        for step_name in self._runtime_step_order(runtime):
            runner = step_runners[step_name]
            if step_name in {"normalize_format", "crop_sneakers"}:
                artifacts = runner(
                    artifacts=artifacts,
                    base_name=base_name,
                    runtime=runtime,
                    step_metadata=step_metadata,
                    applied_steps=applied_steps,
                    skipped_steps=skipped_steps,
                )
            else:
                artifacts = runner(
                    artifacts=artifacts,
                    runtime=runtime,
                    step_metadata=step_metadata,
                    applied_steps=applied_steps,
                    skipped_steps=skipped_steps,
                )

        if runtime.max_outputs_per_image is not None:
            artifacts = artifacts[: runtime.max_outputs_per_image]

        return {
            "filename": filename,
            "outputs": artifacts,
            "metadata": step_metadata,
            "applied_steps": applied_steps,
            "skipped_steps": skipped_steps,
        }
