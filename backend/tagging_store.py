from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

DEFAULT_TAGS = [
    "production_quality",
    "low_quality",
    "clean_background",
    "busy_background",
    "multiple_sneakers",
    "obstructed_or_crop",
    "blur",
    "side_view",
    "top_view",
    "angled_view",
    "harsh_light",
    "worn_on_foot",
]

DEFAULT_PRESETS = {
    "prod_clean": [
        "production_quality",
        "clean_background",
    ],
    "worn_photo": [
        "worn_on_foot",
    ],
    "low_quality_scene": [
        "low_quality",
        "busy_background",
    ],
    "hard_case": [
        "obstructed_or_crop",
        "multiple_sneakers",
    ],
}

REMOVED_TAGS = {
    "single_sneaker",
    "product_photo",
    "real_world_photo",
    "low_light",
}

TAG_RENAMES = {
    "occluded": "obstructed_or_crop",
    "cropped": "obstructed_or_crop",
}


def normalize_relative_path(path: str | Path) -> str:
    return Path(path).as_posix()


def normalize_tags(tags: set[str]) -> set[str]:
    normalized: set[str] = set()
    for tag in tags:
        clean_tag = str(tag).strip()
        if not clean_tag:
            continue
        if clean_tag in REMOVED_TAGS:
            continue
        clean_tag = TAG_RENAMES.get(clean_tag, clean_tag)
        normalized.add(clean_tag)
    return normalized


def discover_images(root: Path) -> list[str]:
    images: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        images.append(path.relative_to(root).as_posix())
    return images


@dataclass
class TagRecord:
    image: str
    tags: list[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "image": self.image,
                "tags": self.tags,
            },
            ensure_ascii=True,
        )


def load_tag_index(tags_file: str | Path | None) -> dict[str, list[str]]:
    if tags_file is None:
        return {}

    path = Path(tags_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Tags file not found: {path}")

    entries: dict[str, set[str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        image = normalize_relative_path(payload["image"])
        tags = normalize_tags({str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()})
        if not tags:
            entries.pop(image, None)
            continue
        entries[image] = tags

    return {image: sorted(tags) for image, tags in entries.items()}


class ImageTagStore:
    def __init__(
        self,
        *,
        image_root: str | Path,
        tags_file: str | Path | None = None,
        default_tags: list[str] | None = None,
        presets: dict[str, list[str]] | None = None,
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"Image root is not a directory: {self.image_root}")

        self.tags_file = (
            Path(tags_file).expanduser().resolve()
            if tags_file is not None
            else self.image_root / "image_tags.jsonl"
        )
        self.default_tags = list(default_tags or DEFAULT_TAGS)
        self.presets = dict(presets or DEFAULT_PRESETS)
        self.image_paths = discover_images(self.image_root)
        self._entries: dict[str, set[str]] = {}
        self.load()

    def load(self) -> None:
        self._entries = {}
        if not self.tags_file.exists():
            return

        for raw_line in self.tags_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image = normalize_relative_path(payload["image"])
            tags = normalize_tags(
                {str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()}
            )
            if not tags:
                self._entries.pop(image, None)
                continue
            self._entries[image] = tags

    def save(self) -> None:
        self.tags_file.parent.mkdir(parents=True, exist_ok=True)
        records: list[str] = []

        for image in sorted(self._entries):
            tags = sorted(self._entries[image])
            if not tags:
                continue
            records.append(TagRecord(image=image, tags=tags).to_json())

        content = "\n".join(records)
        if content:
            content += "\n"
        self.tags_file.write_text(content, encoding="utf-8")

    def image_count(self) -> int:
        return len(self.image_paths)

    def known_tags(self) -> list[str]:
        extras = {
            tag
            for tags in self._entries.values()
            for tag in tags
        }
        ordered = list(self.default_tags)
        ordered.extend(sorted(extras - set(self.default_tags)))
        return ordered

    def tags_for(self, image: str) -> list[str]:
        return sorted(self._entries.get(normalize_relative_path(image), set()))

    def tag_set_for(self, image: str) -> set[str]:
        return set(self._entries.get(normalize_relative_path(image), set()))

    def set_tags(self, image: str, tags: list[str] | set[str]) -> None:
        normalized_image = normalize_relative_path(image)
        normalized_tags = normalize_tags({str(tag).strip() for tag in tags if str(tag).strip()})
        if normalized_tags:
            self._entries[normalized_image] = normalized_tags
        else:
            self._entries.pop(normalized_image, None)

    def add_tags(self, image: str, tags: list[str] | set[str]) -> None:
        next_tags = self.tag_set_for(image)
        next_tags.update({str(tag).strip() for tag in tags if str(tag).strip()})
        self.set_tags(image, next_tags)

    def remove_tags(self, image: str, tags: list[str] | set[str]) -> None:
        next_tags = self.tag_set_for(image)
        next_tags.difference_update({str(tag).strip() for tag in tags if str(tag).strip()})
        self.set_tags(image, next_tags)

    def toggle_tag(self, image: str, tag: str, enabled: bool) -> None:
        if enabled:
            self.add_tags(image, [tag])
        else:
            self.remove_tags(image, [tag])

    def apply_preset(self, image: str, preset_name: str) -> None:
        tags = self.presets.get(preset_name)
        if not tags:
            raise KeyError(f"Unknown preset: {preset_name}")
        self.add_tags(image, tags)

    def filtered_images(
        self,
        *,
        query: str = "",
        show_only_untagged: bool = False,
    ) -> list[str]:
        needle = query.strip().lower()
        filtered: list[str] = []

        for image in self.image_paths:
            tags = self._entries.get(image, set())
            if show_only_untagged and tags:
                continue
            if needle and needle not in image.lower():
                continue
            filtered.append(image)

        return filtered

    def delete_image(self, image: str) -> None:
        normalized_image = normalize_relative_path(image)
        target = self.image_root / normalized_image
        if not target.exists():
            raise FileNotFoundError(f"Image not found: {target}")

        target.unlink()
        self._entries.pop(normalized_image, None)
        self.image_paths = [item for item in self.image_paths if item != normalized_image]
