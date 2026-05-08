"""Render underfitting/overfitting diagnostics from a training history JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/validation loss, validation accuracy, and fit diagnostics."
    )
    parser.add_argument("history", type=Path, help="Path to training_history.json.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to artifacts/training_plots/<history_stem>_diagnostics.png.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "training_plots",
        help="Directory used when --output is not provided.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix used when --output is not provided.",
    )
    parser.add_argument(
        "--loss-only",
        action="store_true",
        help="Render only the train/validation loss panel.",
    )
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Training history must be a non-empty JSON list: {path}")
    return data


def series(history: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for index, item in enumerate(history, start=1):
        value = item.get(key)
        if value is None:
            continue
        points.append((int(item.get("epoch", index)), float(value)))
    return points


def best_epoch_by_val_loss(history: list[dict[str, Any]]) -> tuple[int, float] | None:
    values = series(history, "val_loss")
    if not values:
        return None
    return min(values, key=lambda item: item[1])


def best_epoch_by_val_accuracy(history: list[dict[str, Any]]) -> tuple[int, float] | None:
    values = series(history, "val_accuracy") or series(history, "val_acc") or series(history, "accuracy")
    if not values:
        return None
    return max(values, key=lambda item: item[1])


def value_range(points: list[tuple[int, float]]) -> tuple[float, float]:
    values = [value for _, value in points]
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = abs(minimum) * 0.1 or 1.0
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.08
    return minimum - padding, maximum + padding


def draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
    title: str,
    fixed_labels: bool = False,
) -> None:
    axis = (40, 40, 40)
    grid = (225, 225, 225)
    text = (35, 35, 35)

    draw.text((left, top - 28), title, fill=text)
    draw.line((left, top, left, top + height), fill=axis, width=2)
    draw.line((left, top + height, left + width, top + height), fill=axis, width=2)

    for tick in range(6):
        ratio = tick / 5
        y = top + round(ratio * height)
        value = value_max - ratio * (value_max - value_min)
        draw.line((left, y, left + width, y), fill=grid, width=1)
        label = f"{value:.1f}" if fixed_labels else f"{value:.3g}"
        draw.text((left - 68, y - 7), label, fill=text)

    if epoch_max == epoch_min:
        x_ticks = [epoch_min]
    else:
        x_ticks = sorted({round(epoch_min + index * (epoch_max - epoch_min) / 5) for index in range(6)})

    for epoch in x_ticks:
        x, _ = point_to_pixel(
            epoch=epoch,
            value=value_min,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=value_min,
            value_max=value_max,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        draw.line((x, top + height, x, top + height + 5), fill=axis)
        draw.text((x - 8, top + height + 10), str(epoch), fill=text)

    draw.text((left + width // 2 - 20, top + height + 30), "epoch", fill=text)


def point_to_pixel(
    *,
    epoch: int,
    value: float,
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    x_ratio = 0.0 if epoch_max == epoch_min else (epoch - epoch_min) / (epoch_max - epoch_min)
    y_ratio = (value - value_min) / (value_max - value_min)
    return left + round(x_ratio * width), top + height - round(y_ratio * height)


def draw_line_series(
    draw: ImageDraw.ImageDraw,
    *,
    points: list[tuple[int, float]],
    color: tuple[int, int, int],
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    pixels = [
        point_to_pixel(
            epoch=epoch,
            value=value,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=value_min,
            value_max=value_max,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        for epoch, value in points
    ]
    if len(pixels) >= 2:
        draw.line(pixels, fill=color, width=3)
    for x, y in pixels:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def draw_vertical_marker(
    draw: ImageDraw.ImageDraw,
    *,
    epoch: int,
    label: str,
    color: tuple[int, int, int],
    epoch_min: int,
    epoch_max: int,
    left: int,
    top: int,
    width: int,
    height: int,
    label_y_offset: int,
) -> None:
    x, _ = point_to_pixel(
        epoch=epoch,
        value=0.0,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        value_min=0.0,
        value_max=1.0,
        left=left,
        top=top,
        width=width,
        height=height,
    )
    draw.line((x, top, x, top + height), fill=color, width=2)
    draw.text((x + 6, top + label_y_offset), label, fill=color)


def loss_at(points: list[tuple[int, float]], epoch: int) -> float | None:
    by_epoch = dict(points)
    return by_epoch.get(epoch)


def build_notes(
    *,
    train_loss: list[tuple[int, float]],
    val_loss: list[tuple[int, float]],
    val_accuracy: list[tuple[int, float]],
    best_loss: tuple[int, float] | None,
    best_accuracy: tuple[int, float] | None,
) -> list[str]:
    first_epoch = train_loss[0][0]
    final_epoch = train_loss[-1][0]
    first_train = train_loss[0][1]
    first_val = val_loss[0][1]
    final_train = train_loss[-1][1]
    final_val = val_loss[-1][1]
    final_gap = final_val - final_train

    best_train_loss = None if best_loss is None else loss_at(train_loss, best_loss[0])
    has_strong_overfit_signal = (
        best_loss is not None
        and best_train_loss is not None
        and best_loss[0] < final_epoch
        and final_val > best_loss[1]
        and final_train <= best_train_loss
    )
    if best_loss is None:
        overfit_note = "overfit candidate: unavailable without val_loss"
    elif has_strong_overfit_signal:
        overfit_note = f"overfit candidate: after epoch {best_loss[0]}, val_loss worsens while train_loss keeps falling"
    elif best_loss[0] < final_epoch and final_val > best_loss[1]:
        overfit_note = f"overfit candidate: after epoch {best_loss[0]}, validation loss gets worse"
    else:
        overfit_note = "overfit candidate: not strong in this run"

    accuracy_note = "best val accuracy: unavailable"
    if best_accuracy is not None:
        accuracy_note = f"best val accuracy: epoch {best_accuracy[0]} ({best_accuracy[1]:.4f})"

    best_loss_note = "best val loss: unavailable"
    if best_loss is not None:
        best_loss_note = f"best val loss: epoch {best_loss[0]} ({best_loss[1]:.4f})"

    final_accuracy_note = "final val accuracy: unavailable"
    if val_accuracy:
        final_accuracy_note = f"final val accuracy: {val_accuracy[-1][1]:.4f}"

    return [
        "How to read this:",
        f"underfit checkpoint candidate: epoch {first_epoch}, before the model has learned enough",
        best_loss_note,
        overfit_note,
        f"initial loss: train={first_train:.4f}, val={first_val:.4f}",
        f"final loss: train={final_train:.4f}, val={final_val:.4f}, gap={final_gap:.4f}",
        accuracy_note,
        final_accuracy_note,
    ]


def render(
    history: list[dict[str, Any]],
    output_path: Path,
    *,
    loss_only: bool = False,
) -> None:
    train_loss = series(history, "train_loss")
    val_loss = series(history, "val_loss")
    val_accuracy = series(history, "val_accuracy") or series(history, "val_acc") or series(history, "accuracy")
    if not train_loss or not val_loss:
        raise ValueError("Diagnostics require train_loss and val_loss series.")

    all_loss_points = train_loss + val_loss
    epoch_values = [epoch for epoch, _ in all_loss_points]
    epoch_min = min(epoch_values)
    epoch_max = max(epoch_values)
    loss_min, loss_max = value_range(all_loss_points)
    loss_min = max(0.0, loss_min)
    best_loss = best_epoch_by_val_loss(history)
    best_accuracy = best_epoch_by_val_accuracy(history)

    width = 1100
    height = 520 if loss_only else 840
    left = 90
    loss_top = 80
    accuracy_top = 435
    plot_width = 780
    loss_height = 310 if loss_only else 260
    accuracy_height = 180
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    text = (35, 35, 35)
    train_color = (43, 111, 173)
    val_color = (214, 91, 52)
    best_color = (45, 150, 86)
    final_color = (120, 90, 160)
    accuracy_color = (230, 160, 40)

    draw.text((left, 25), "Training Diagnostics: Underfitting and Overfitting", fill=text)
    draw_axes(
        draw,
        left=left,
        top=loss_top,
        width=plot_width,
        height=loss_height,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        value_min=loss_min,
        value_max=loss_max,
        title="Loss curves",
    )

    draw_line_series(
        draw,
        points=train_loss,
        color=train_color,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        value_min=loss_min,
        value_max=loss_max,
        left=left,
        top=loss_top,
        width=plot_width,
        height=loss_height,
    )
    draw_line_series(
        draw,
        points=val_loss,
        color=val_color,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        value_min=loss_min,
        value_max=loss_max,
        left=left,
        top=loss_top,
        width=plot_width,
        height=loss_height,
    )

    if best_loss is not None:
        draw_vertical_marker(
            draw,
            epoch=best_loss[0],
            label=f"best val loss {best_loss[0]}",
            color=best_color,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            left=left,
            top=loss_top,
            width=plot_width,
            height=loss_height,
            label_y_offset=8,
        )

    draw_vertical_marker(
        draw,
        epoch=epoch_max,
        label=f"final {epoch_max}",
        color=final_color,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        left=left,
        top=loss_top,
        width=plot_width,
        height=loss_height,
        label_y_offset=28,
    )

    if val_accuracy and not loss_only:
        draw_axes(
            draw,
            left=left,
            top=accuracy_top,
            width=plot_width,
            height=accuracy_height,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=0.0,
            value_max=1.0,
            title="Validation accuracy",
            fixed_labels=True,
        )
        draw_line_series(
            draw,
            points=val_accuracy,
            color=accuracy_color,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=0.0,
            value_max=1.0,
            left=left,
            top=accuracy_top,
            width=plot_width,
            height=accuracy_height,
        )
        if best_accuracy is not None:
            draw_vertical_marker(
                draw,
                epoch=best_accuracy[0],
                label=f"best acc {best_accuracy[0]}",
                color=best_color,
                epoch_min=epoch_min,
                epoch_max=epoch_max,
                left=left,
                top=accuracy_top,
                width=plot_width,
                height=accuracy_height,
                label_y_offset=8,
            )

    legend_items = [
        ("train_loss", train_color),
        ("val_loss", val_color),
        ("best validation", best_color),
        ("final checkpoint", final_color),
    ]
    if not loss_only:
        legend_items.insert(2, ("val_accuracy", accuracy_color))

    for index, (label, color) in enumerate(legend_items):
        item_y = 90 + index * 25
        draw.rectangle((910, item_y + 4, 924, item_y + 18), fill=color)
        draw.text((930, item_y), label, fill=text)

    if not loss_only:
        notes = build_notes(
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            best_loss=best_loss,
            best_accuracy=best_accuracy,
        )
        for index, note in enumerate(notes):
            draw.text((90, 660 + index * 18), note, fill=text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        prefix = args.prefix or args.history.stem
        output = args.output_dir / f"{prefix}_diagnostics.png"
    render(load_history(args.history), output, loss_only=args.loss_only)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
