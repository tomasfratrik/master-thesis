from __future__ import annotations

import io

from PIL import Image

from ..config import NormalizeFormatConfig, NormalizeFormatRuntimeConfig

SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class NormalizeFormatStep:
    step_name = "normalize_format"

    def __init__(self, config: NormalizeFormatConfig) -> None:
        self.config = config

    @staticmethod
    def _resolve(base_value: object, override_value: object) -> object:
        return base_value if override_value is None else override_value

    @staticmethod
    def _normalize_target_format(value: str) -> str:
        normalized = str(value).strip().upper()
        if normalized == "JPG":
            normalized = "JPEG"
        if normalized not in SUPPORTED_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_FORMATS))
            raise ValueError(f"Unsupported target format '{value}'. Allowed: {allowed}")
        return normalized

    @staticmethod
    def _flatten_to_rgb(
        image: Image.Image, background_rgb: tuple[int, int, int]
    ) -> Image.Image:
        has_alpha = image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        )
        if has_alpha:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, background_rgb + (255,))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")

    def process(
        self, image: Image.Image, runtime: NormalizeFormatRuntimeConfig
    ) -> tuple[Image.Image, dict[str, object], str, str]:
        target_format = self._normalize_target_format(
            str(self._resolve(self.config.target_format, runtime.target_format))
        )
        background_rgb = tuple(
            int(v)
            for v in self._resolve(self.config.background_rgb, runtime.background_rgb)
        )
        jpeg_quality = int(
            self._resolve(self.config.jpeg_quality, runtime.jpeg_quality)
        )

        normalized = self._flatten_to_rgb(image, background_rgb=background_rgb)

        buffer = io.BytesIO()
        save_kwargs: dict[str, object] = {"format": target_format}
        if target_format in {"JPEG", "WEBP"}:
            save_kwargs["quality"] = jpeg_quality
        normalized.save(buffer, **save_kwargs)
        buffer.seek(0)

        with Image.open(buffer) as reopened:
            materialized = reopened.convert("RGB")

        return (
            materialized,
            {
                "source_mode": image.mode,
                "output_mode": materialized.mode,
                "target_format": target_format,
                "size": [materialized.width, materialized.height],
            },
            target_format,
            SUPPORTED_FORMATS[target_format],
        )

