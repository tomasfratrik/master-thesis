from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizeFormatConfig(BaseModel):
    enabled: bool = True
    target_format: str = "JPEG"
    jpeg_quality: int = 95
    background_rgb: tuple[int, int, int] = (255, 255, 255)


class NormalizeFormatRuntimeConfig(BaseModel):
    enabled: bool | None = None
    target_format: str | None = None
    jpeg_quality: int | None = None
    background_rgb: tuple[int, int, int] | None = None


class ResizeLimitConfig(BaseModel):
    enabled: bool = False
    max_long_side: int = 1600
    only_downscale: bool = True
    resample: str = "LANCZOS"


class ResizeLimitRuntimeConfig(BaseModel):
    enabled: bool | None = None
    max_long_side: int | None = None
    only_downscale: bool | None = None
    resample: str | None = None


class GrayscaleConfig(BaseModel):
    enabled: bool = True
    keep_rgb_output: bool = True


class GrayscaleRuntimeConfig(BaseModel):
    enabled: bool | None = None
    keep_rgb_output: bool | None = None


class CropSneakersConfig(BaseModel):
    enabled: bool = True
    model_id: str = "IDEA-Research/grounding-dino-base"
    local_files_only: bool = False
    labels: str = "shoe . sneaker ."
    box_threshold: float = 0.35
    text_threshold: float = 0.2
    nms_threshold: float = 0.35
    same_shoe_dedup: bool = True
    same_shoe_center_ratio: float = 0.35
    same_shoe_iou_threshold: float = 0.05
    same_shoe_area_ratio: float = 0.55
    final_nms_threshold: float = 0.0
    group_pairs: bool = True
    prefer_pair_labels: bool = False
    pair_overlap_threshold: float = 0.15
    pair_gap_ratio: float = 0.6
    padding: int = 8
    padding_ratio: float = 0.12
    return_original_if_empty: bool = True


class CropSneakersRuntimeConfig(BaseModel):
    enabled: bool | None = None
    local_files_only: bool | None = None
    labels: str | None = None
    box_threshold: float | None = None
    text_threshold: float | None = None
    nms_threshold: float | None = None
    same_shoe_dedup: bool | None = None
    same_shoe_center_ratio: float | None = None
    same_shoe_iou_threshold: float | None = None
    same_shoe_area_ratio: float | None = None
    final_nms_threshold: float | None = None
    group_pairs: bool | None = None
    prefer_pair_labels: bool | None = None
    pair_overlap_threshold: float | None = None
    pair_gap_ratio: float | None = None
    padding: int | None = None
    padding_ratio: float | None = None
    return_original_if_empty: bool | None = None


class RuntimeOptions(BaseModel):
    step_order: list[str] = Field(default_factory=list)
    enabled_steps: list[str] = Field(default_factory=list)
    disabled_steps: list[str] = Field(default_factory=list)
    include_images: bool = True
    max_outputs_per_image: int | None = Field(default=None, ge=1)
    normalize_format: NormalizeFormatRuntimeConfig = Field(
        default_factory=NormalizeFormatRuntimeConfig
    )
    resize_limit: ResizeLimitRuntimeConfig = Field(
        default_factory=ResizeLimitRuntimeConfig
    )
    grayscale: GrayscaleRuntimeConfig = Field(
        default_factory=GrayscaleRuntimeConfig
    )
    crop_sneakers: CropSneakersRuntimeConfig = Field(
        default_factory=CropSneakersRuntimeConfig
    )

    @staticmethod
    def _normalized(names: list[str]) -> set[str]:
        return {name.strip().lower() for name in names if name and name.strip()}

    def should_run_step(self, step_name: str) -> bool:
        step = step_name.lower()
        enabled = self._normalized(self.enabled_steps)
        if enabled:
            return step in enabled
        disabled = self._normalized(self.disabled_steps)
        return step not in disabled

    def has_enabled_steps(self) -> bool:
        return bool(self._normalized(self.enabled_steps))


class PreprocessConfig(BaseModel):
    step_order: list[str] = Field(
        default_factory=lambda: [
            "normalize_format",
            "crop_sneakers",
            "resize_limit",
            "grayscale",
        ]
    )
    normalize_format: NormalizeFormatConfig = Field(
        default_factory=NormalizeFormatConfig
    )
    resize_limit: ResizeLimitConfig = Field(default_factory=ResizeLimitConfig)
    grayscale: GrayscaleConfig = Field(default_factory=GrayscaleConfig)
    crop_sneakers: CropSneakersConfig = Field(default_factory=CropSneakersConfig)
