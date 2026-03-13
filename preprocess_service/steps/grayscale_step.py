from __future__ import annotations

from PIL import Image

from ..config import GrayscaleConfig, GrayscaleRuntimeConfig


class GrayscaleStep:
    step_name = "grayscale"

    def __init__(self, config: GrayscaleConfig) -> None:
        self.config = config

    @staticmethod
    def _resolve(base_value: object, override_value: object) -> object:
        return base_value if override_value is None else override_value

    def process(
        self, image: Image.Image, runtime: GrayscaleRuntimeConfig
    ) -> tuple[Image.Image, dict[str, object]]:
        keep_rgb_output = bool(
            self._resolve(self.config.keep_rgb_output, runtime.keep_rgb_output)
        )

        gray = image.convert("L")
        output = gray.convert("RGB") if keep_rgb_output else gray
        return output, {
            "source_mode": image.mode,
            "output_mode": output.mode,
            "keep_rgb_output": keep_rgb_output,
        }

