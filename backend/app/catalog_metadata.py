from __future__ import annotations

"""Sneaker catalog metadata helpers for labels, brands, and class names."""

import re


def format_class_label(class_name: str) -> str:
    label = class_name.replace("_", " ").title()
    if label.startswith("Air Jordan "):
        return f"Nike {label}"
    return label


def tokenize_class_name(class_name: str) -> list[str]:
    return [token for token in class_name.split("_") if token]


def is_model_like_token(token: str) -> bool:
    return any(character.isdigit() for character in token)


def build_brand_prefix_map(class_names: list[str]) -> dict[str, str]:
    """Infer brand labels from checkpoint or catalog class names."""
    tokenized = {class_name: tokenize_class_name(class_name) for class_name in class_names}
    prefix_counts: dict[tuple[str, ...], int] = {}
    next_token_sets: dict[tuple[str, ...], set[str]] = {}

    for tokens in tokenized.values():
        for index in range(1, len(tokens) + 1):
            prefix = tuple(tokens[:index])
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            if index < len(tokens):
                next_token_sets.setdefault(prefix, set()).add(tokens[index])

    brands: dict[str, str] = {}
    for class_name, tokens in tokenized.items():
        if len(tokens) >= 2 and tokens[0].lower() == "air" and tokens[1].lower() == "jordan":
            brands[class_name] = "Nike"
            continue
        if (
            len(tokens) >= 3
            and tokens[0].lower() == "nike"
            and tokens[1].lower() == "air"
            and tokens[2].lower() == "jordan"
        ):
            brands[class_name] = "Nike"
            continue

        brand_tokens = [tokens[0]]
        max_brand_tokens = min(3, len(tokens))

        while len(brand_tokens) < max_brand_tokens:
            prefix = tuple(brand_tokens)
            next_tokens = next_token_sets.get(prefix, set())
            if len(next_tokens) != 1:
                break

            next_token = tokens[len(brand_tokens)]
            if is_model_like_token(next_token):
                break

            if prefix_counts.get(prefix, 0) == 1 and len(brand_tokens) >= 2:
                break

            brand_tokens.append(next_token)

        brands[class_name] = " ".join(brand_tokens).title()

    return brands


def infer_model_name(class_name: str, brand: str) -> str:
    display_name = format_class_label(class_name)
    if display_name.startswith(f"{brand} "):
        return display_name[len(brand) + 1 :]
    return display_name


def normalize_class_name(value: str) -> str:
    """Normalize arbitrary text into the stored catalog class-name format."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Class name is required.")
    return normalized
