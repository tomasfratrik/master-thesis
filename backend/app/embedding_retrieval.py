from __future__ import annotations

"""
Catalog embedding retrieval for classifier-compatible sneaker search.

Builds a searchable embedding index and returns class-level ranked matches.
"""

import io
from collections import OrderedDict
from typing import Any

import torch
from PIL import Image

from .prototype_catalog import load_catalog_embedding_entries, load_catalog_image_embedding_entries


ClassAggregationMode = str


class CatalogEmbeddingRetrieval:
    def __init__(
        self,
        classifier: Any,
        *,
        class_aggregation: ClassAggregationMode = "max",
        top_n_per_class: int = 3,
    ) -> None:
        self.classifier = classifier
        self.device = classifier.device
        self.class_aggregation = class_aggregation
        self.top_n_per_class = max(1, int(top_n_per_class))
        self.entries: list[dict[str, Any]] = []
        self.entry_mode = "class"
        self.feature_matrix = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.class_groups: list[dict[str, Any]] = []
        self.refresh()

    @staticmethod
    def _normalize_feature(feature: Any, device: torch.device) -> torch.Tensor:
        """Convert a stored embedding into a normalized device tensor."""
        if not torch.is_tensor(feature):
            feature = torch.tensor(feature, dtype=torch.float32)
        feature = feature.to(device).float()
        if feature.ndim > 1:
            feature = feature.squeeze(0)
        return feature / feature.norm(dim=-1, keepdim=True)

    def _group_rows(self, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], torch.Tensor]:
        """Group embedding rows by class while keeping per-row retrieval data."""
        normalized_rows: list[dict[str, Any]] = []
        feature_rows: list[torch.Tensor] = []

        grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for row_index, item in enumerate(entries):
            feature = self._normalize_feature(item["feature"], self.device)
            feature_rows.append(feature)
            normalized_rows.append(
                {
                    "product_id": item.get("product_id"),
                    "class_name": item["class_name"],
                    "label": item["label"],
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "preview_urls": item.get("preview_urls", []),
                    "candidate_type": item.get("candidate_type", "catalog_embedding"),
                    "source_path": item.get("source_path"),
                }
            )

            class_name = item["class_name"]
            if class_name not in grouped:
                grouped[class_name] = {
                    "product_id": item.get("product_id"),
                    "class_name": class_name,
                    "label": item["label"],
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "preview_urls": item.get("preview_urls", []),
                    "candidate_type": item.get("candidate_type", "catalog_embedding"),
                    "row_indices": [],
                }
            grouped[class_name]["row_indices"].append(row_index)

        class_groups: list[dict[str, Any]] = []
        for item in grouped.values():
            class_groups.append(
                {
                    "product_id": item.get("product_id"),
                    "class_name": item["class_name"],
                    "label": item["label"],
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "preview_urls": item.get("preview_urls", []),
                    "candidate_type": item.get("candidate_type", "catalog_embedding"),
                    "row_indices": torch.tensor(
                        item["row_indices"],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    "row_count": len(item["row_indices"]),
                }
            )

        if feature_rows:
            feature_matrix = torch.stack(feature_rows, dim=0).to(self.device)
        else:
            feature_matrix = torch.empty((0, 1), dtype=torch.float32, device=self.device)

        self.class_groups = class_groups
        return normalized_rows, feature_matrix

    def refresh(self) -> int:
        """Reload catalog embedding entries from disk and the database."""
        image_entries = load_catalog_image_embedding_entries()
        if image_entries:
            self.entry_mode = "image"
            seen = {item["class_name"] for item in image_entries}
            entries = image_entries + [
                item for item in load_catalog_embedding_entries()
                if item["class_name"] not in seen
            ]
        else:
            self.entry_mode = "class"
            entries = load_catalog_embedding_entries()

        self.entries, self.feature_matrix = self._group_rows(entries)
        return len(self.class_groups)

    @property
    def catalog_size(self) -> int:
        return len(self.class_groups)

    @property
    def row_count(self) -> int:
        return int(self.feature_matrix.shape[0]) if self.feature_matrix.ndim == 2 else 0

    def _candidate_space(
        self,
        extra_entries: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Assemble retrieval candidates from catalog rows and optional extras."""
        metadata = list(self.class_groups)
        if self.feature_matrix.numel() == 0:
            features: list[torch.Tensor] = []
        else:
            features = [self.feature_matrix]

        row_offset = self.feature_matrix.shape[0]
        if extra_entries:
            extra_features: list[torch.Tensor] = []
            extra_metadata: list[dict[str, Any]] = []
            for item in extra_entries:
                feature = self._normalize_feature(item["feature"], self.device)
                extra_features.append(feature)
                extra_metadata.append(
                    {
                        "product_id": item.get("product_id"),
                        "class_name": item["class_name"],
                        "label": item["label"],
                        "brand": item.get("brand"),
                        "model": item.get("model"),
                        "preview_urls": item.get("preview_urls", []),
                        "candidate_type": item.get("candidate_type", "catalog_embedding"),
                        "row_indices": torch.tensor([row_offset], dtype=torch.long, device=self.device),
                        "row_count": 1,
                    }
                )
                row_offset += 1
            if extra_features:
                features.append(torch.stack(extra_features, dim=0).to(self.device))
                metadata.extend(extra_metadata)

        if not features:
            raise ValueError("No catalog embeddings are available.")

        return torch.cat(features, dim=0), metadata

    def _query_feature(self, images: list[Image.Image]) -> torch.Tensor:
        return self.classifier.build_prototype_from_images(images).to(self.device)

    def _aggregate_class_score(self, row_scores: torch.Tensor) -> torch.Tensor:
        """Reduce per-image scores into one score for a class."""
        if self.class_aggregation == "max":
            return row_scores.max()
        top_n = min(self.top_n_per_class, int(row_scores.shape[0]))
        if top_n == 1:
            return row_scores.max()
        top_scores = row_scores.topk(top_n).values
        return top_scores.mean()

    def _build_result(
        self,
        class_scores: torch.Tensor,
        candidate_metadata: list[dict[str, Any]],
        *,
        k: int,
        query_image_count: int,
    ) -> dict[str, Any]:
        """Convert retrieval scores into the public ranked response payload."""
        k = max(1, min(int(k), len(candidate_metadata)))
        probabilities = class_scores.softmax(dim=-1)
        top_scores, top_indices = probabilities.topk(k)

        top_k: list[dict[str, Any]] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            candidate = candidate_metadata[index]
            top_k.append(
                {
                    "product_id": candidate.get("product_id"),
                    "class_name": candidate["class_name"],
                    "label": candidate["label"],
                    "score": float(score),
                    "preview_urls": candidate.get("preview_urls", []),
                    "candidate_type": candidate.get("candidate_type", "catalog_embedding"),
                    "brand": candidate.get("brand"),
                    "model": candidate.get("model"),
                    "reference_image_count": candidate.get("row_count", 1),
                }
            )

        second_best_score = top_k[1]["score"] if len(top_k) > 1 else 0.0
        return {
            "product_id": top_k[0].get("product_id"),
            "label": top_k[0]["label"],
            "class_name": top_k[0]["class_name"],
            "score": top_k[0]["score"],
            "margin_vs_second": float(top_k[0]["score"] - second_best_score),
            "query_image_count": int(query_image_count),
            "aggregation": f"image_retrieval_{self.class_aggregation}",
            "preview_urls": top_k[0]["preview_urls"],
            "candidate_type": top_k[0]["candidate_type"],
            "catalog_mode": self.entry_mode,
            "top_k": top_k,
        }

    def search_images(
        self,
        images: list[Image.Image],
        *,
        k: int = 5,
        extra_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Search the catalog with one or more query images."""
        candidate_features, candidate_metadata = self._candidate_space(extra_entries)
        query_feature = self._query_feature(images)
        row_scores = (100.0 * query_feature @ candidate_features.T).squeeze(0)

        class_scores: list[torch.Tensor] = []
        for candidate in candidate_metadata:
            per_class_scores = row_scores.index_select(0, candidate["row_indices"])
            class_scores.append(self._aggregate_class_score(per_class_scores))
        stacked_class_scores = torch.stack(class_scores, dim=0)

        return self._build_result(
            class_scores=stacked_class_scores,
            candidate_metadata=candidate_metadata,
            k=k,
            query_image_count=len(images),
        )

    def search_image_bytes(
        self,
        image_bytes: bytes,
        *,
        k: int = 5,
        extra_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self.search_images(
                [image.convert("RGB")],
                k=k,
                extra_entries=extra_entries,
            )

    def search_image_bytes_batch(
        self,
        image_payloads: list[bytes],
        *,
        k: int = 5,
        extra_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        images: list[Image.Image] = []
        for payload in image_payloads:
            with Image.open(io.BytesIO(payload)) as image:
                images.append(image.convert("RGB"))
        return self.search_images(images, k=k, extra_entries=extra_entries)
