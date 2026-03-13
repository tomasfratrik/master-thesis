from __future__ import annotations

from PIL import Image

from ..config import ResizeLimitConfig, ResizeLimitRuntimeConfig


class ResizeLimitStep:
    step_name = "resize_limit"

    _RESAMPLE_MAP = {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS,
    }

    def __init__(self, config: ResizeLimitConfig) -> None:
        self.config = config

    @staticmethod
    def _resolve(base_value: object, override_value: object) -> object:
        return base_value if override_value is None else override_value

    def process(
        self, image: Image.Image, runtime: ResizeLimitRuntimeConfig
    ) -> tuple[Image.Image, dict[str, object]]:
        max_long_side = int(
            self._resolve(self.config.max_long_side, runtime.max_long_side)
        )
        if max_long_side <= 0:
            raise ValueError("resize_limit.max_long_side must be greater than 0.")

        only_downscale = bool(
            self._resolve(self.config.only_downscale, runtime.only_downscale)
        )
        resample_name = str(
            self._resolve(self.config.resample, runtime.resample)
        ).strip().upper()
        if resample_name not in self._RESAMPLE_MAP:
            allowed = ", ".join(sorted(self._RESAMPLE_MAP))
            raise ValueError(
                f"Unsupported resize_limit.resample '{resample_name}'. Allowed: {allowed}"
            )

        width, height = image.size
        current_long_side = max(width, height)
        should_resize = current_long_side != max_long_side
        if only_downscale and current_long_side <= max_long_side:
            should_resize = False

        if not should_resize:
            return image.copy(), {
                "resized": False,
                "source_size": [width, height],
                "output_size": [width, height],
                "max_long_side": max_long_side,
                "resample": resample_name,
            }

        scale = max_long_side / float(current_long_side)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), self._RESAMPLE_MAP[resample_name])
        return resized, {
            "resized": True,
            "source_size": [width, height],
            "output_size": [new_width, new_height],
            "max_long_side": max_long_side,
            "scale": scale,
            "resample": resample_name,
        }

