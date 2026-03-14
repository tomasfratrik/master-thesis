from __future__ import annotations

from threading import Lock
from typing import Any

from PIL import Image

from ..config import CropSneakersConfig, CropSneakersRuntimeConfig
from ..vendor.crop_sneakers.detection import finalize_detections
from ..vendor.crop_sneakers.geometry import clamp_box
from ..vendor.crop_sneakers.model import detect_raw_boxes, load_detector


class CropSneakersStep:
    step_name = "crop_sneakers"

    def __init__(self, config: CropSneakersConfig) -> None:
        self.config = config
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._load_lock = Lock()

    @staticmethod
    def _resolve(base_value: Any, override_value: Any) -> Any:
        return base_value if override_value is None else override_value

    def _ensure_loaded(self, runtime: CropSneakersRuntimeConfig) -> None:
        if self._processor is not None and self._model is not None and self._device is not None:
            return
        with self._load_lock:
            if (
                self._processor is not None
                and self._model is not None
                and self._device is not None
            ):
                return
            local_files_only = self._resolve(
                self.config.local_files_only, runtime.local_files_only
            )
            self._processor, self._model, self._device = load_detector(
                self.config.model_id,
                local_files_only=local_files_only,
            )

    def process(
        self, image: Image.Image, runtime: CropSneakersRuntimeConfig
    ) -> tuple[list[Image.Image], dict[str, Any]]:
        self._ensure_loaded(runtime)
        if self._processor is None or self._model is None or self._device is None:
            raise RuntimeError("CropSneakersStep was not initialized correctly.")

        raw_detections = detect_raw_boxes(
            image=image,
            processor=self._processor,
            model=self._model,
            device=self._device,
            labels=self._resolve(self.config.labels, runtime.labels),
            box_threshold=self._resolve(self.config.box_threshold, runtime.box_threshold),
            text_threshold=self._resolve(
                self.config.text_threshold, runtime.text_threshold
            ),
        )

        detections = finalize_detections(
            raw_detections=raw_detections,
            nms_threshold=self._resolve(self.config.nms_threshold, runtime.nms_threshold),
            same_shoe_dedup=self._resolve(
                self.config.same_shoe_dedup, runtime.same_shoe_dedup
            ),
            same_shoe_center_ratio=self._resolve(
                self.config.same_shoe_center_ratio, runtime.same_shoe_center_ratio
            ),
            same_shoe_iou_threshold=self._resolve(
                self.config.same_shoe_iou_threshold, runtime.same_shoe_iou_threshold
            ),
            same_shoe_area_ratio=self._resolve(
                self.config.same_shoe_area_ratio, runtime.same_shoe_area_ratio
            ),
            final_nms_threshold=self._resolve(
                self.config.final_nms_threshold, runtime.final_nms_threshold
            ),
            group_pairs=self._resolve(self.config.group_pairs, runtime.group_pairs),
            prefer_pair_labels=self._resolve(
                self.config.prefer_pair_labels, runtime.prefer_pair_labels
            ),
            pair_overlap_threshold=self._resolve(
                self.config.pair_overlap_threshold, runtime.pair_overlap_threshold
            ),
            pair_gap_ratio=self._resolve(
                self.config.pair_gap_ratio, runtime.pair_gap_ratio
            ),
        )

        padding = int(self._resolve(self.config.padding, runtime.padding))
        padding_ratio = float(
            self._resolve(self.config.padding_ratio, runtime.padding_ratio)
        )

        width, height = image.size
        crops: list[Image.Image] = []
        crop_metadata: list[dict[str, Any]] = []
        for detection in detections:
            x1, y1, x2, y2 = clamp_box(
                *detection["box_xyxy"],
                width=width,
                height=height,
                padding=padding,
                padding_ratio=padding_ratio,
            )
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            crop_metadata.append(
                {
                    "label": str(detection["label"]),
                    "score": float(detection["score"]),
                    "members": int(detection.get("members", 1)),
                    "box_xyxy": [x1, y1, x2, y2],
                }
            )

        used_original = False
        return_original_if_empty = self._resolve(
            self.config.return_original_if_empty, runtime.return_original_if_empty
        )
        if not crops and return_original_if_empty:
            used_original = True
            crops = [image.copy()]

        return crops, {
            "detections": crop_metadata,
            "count": len(crop_metadata),
            "used_original": used_original,
            "device": self._device,
        }
