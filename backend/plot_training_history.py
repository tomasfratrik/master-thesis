"""Render training-history graphs from fine-tuning JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PALETTE = [
    (43, 111, 173),
    (214, 91, 52),
    (45, 150, 86),
    (120, 90, 160),
    (230, 160, 40),
    (80, 160, 180),
]


class HistoryRun:
    def __init__(self, *, name: str, path: Path, history: list[dict[str, Any]]) -> None:
        self.name = name
        self.path = path
        self.history = history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss, accuracy, and learning-rate curves from training_history.json."
    )
    parser.add_argument(
        "history",
        type=Path,
        nargs="+",
        help="Path to a training history JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "training_plots",
        help="Directory where PNG plots are written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix. Defaults to the history file stem.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help=(
            "Optional label for a history file. Repeat once per input history. "
            "Defaults to each history file stem."
        ),
    )
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Training history must be a non-empty JSON list: {path}")
    return data


def numeric_series(history: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for index, item in enumerate(history, start=1):
        value = item.get(key)
        if value is None:
            continue
        epoch = int(item.get("epoch", index))
        points.append((epoch, float(value)))
    return points


def first_available_key(history: list[dict[str, Any]], keys: list[str]) -> str | None:
    for key in keys:
        if numeric_series(history, key):
            return key
    return None


def value_range(series_items: list[list[tuple[int, float]]]) -> tuple[float, float]:
    values = [value for series in series_items for _, value in series]
    if not values:
        raise ValueError("No values available to plot.")

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = abs(minimum) * 0.1 or 1.0
        return minimum - padding, maximum + padding

    padding = (maximum - minimum) * 0.08
    return minimum - padding, maximum + padding


def epoch_range(histories: list[list[dict[str, Any]]]) -> tuple[int, int]:
    epochs = [
        int(item.get("epoch", index))
        for history in histories
        for index, item in enumerate(history, start=1)
    ]
    return min(epochs), max(epochs)


def point_to_pixel(
    *,
    epoch: int,
    value: float,
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
) -> tuple[int, int]:
    if epoch_max == epoch_min:
        x_ratio = 0.0
    else:
        x_ratio = (epoch - epoch_min) / (epoch_max - epoch_min)
    y_ratio = (value - value_min) / (value_max - value_min)
    x = plot_left + round(x_ratio * plot_width)
    y = plot_top + plot_height - round(y_ratio * plot_height)
    return x, y


def draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    title: str,
    width: int,
    height: int,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
) -> None:
    axis_color = (40, 40, 40)
    grid_color = (225, 225, 225)
    text_color = (35, 35, 35)

    draw.text((plot_left, 18), title, fill=text_color)
    draw.line((plot_left, plot_top, plot_left, plot_top + plot_height), fill=axis_color, width=2)
    draw.line(
        (plot_left, plot_top + plot_height, plot_left + plot_width, plot_top + plot_height),
        fill=axis_color,
        width=2,
    )

    for tick in range(6):
        ratio = tick / 5
        y = plot_top + round(ratio * plot_height)
        value = value_max - ratio * (value_max - value_min)
        draw.line((plot_left, y, plot_left + plot_width, y), fill=grid_color, width=1)
        draw.text((10, y - 7), f"{value:.3g}", fill=text_color)

    if epoch_max == epoch_min:
        x_ticks = [epoch_min]
    else:
        x_ticks = sorted({round(epoch_min + i * (epoch_max - epoch_min) / 5) for i in range(6)})

    for epoch in x_ticks:
        x, _ = point_to_pixel(
            epoch=epoch,
            value=value_min,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=value_min,
            value_max=value_max,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        draw.line((x, plot_top + plot_height, x, plot_top + plot_height + 5), fill=axis_color)
        draw.text((x - 8, height - 28), str(epoch), fill=text_color)

    draw.text((plot_left + plot_width // 2 - 20, height - 28), "epoch", fill=text_color)


def draw_series(
    draw: ImageDraw.ImageDraw,
    *,
    series: list[tuple[int, float]],
    color: tuple[int, int, int],
    marker: str,
    epoch_min: int,
    epoch_max: int,
    value_min: float,
    value_max: float,
    plot_left: int,
    plot_top: int,
    plot_width: int,
    plot_height: int,
) -> None:
    points = [
        point_to_pixel(
            epoch=epoch,
            value=value,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=value_min,
            value_max=value_max,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        for epoch, value in series
    ]
    if len(points) >= 2:
        draw.line(points, fill=color, width=3)
    for x, y in points:
        if marker == "square":
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)
        else:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    labels: list[tuple[str, tuple[int, int, int]]],
    x: int,
    y: int,
) -> None:
    for index, (label, color) in enumerate(labels):
        item_y = y + index * 20
        draw.rectangle((x, item_y + 4, x + 14, item_y + 14), fill=color)
        draw.text((x + 20, item_y), label, fill=(35, 35, 35))


def render_plot(
    *,
    title: str,
    runs: list[HistoryRun],
    series_getter,
    output_path: Path,
    fixed_value_range: tuple[float, float] | None = None,
) -> None:
    width = 960
    height = 560
    plot_left = 80
    plot_top = 58
    plot_width = 760
    plot_height = 420

    plot_series: list[tuple[str, list[tuple[int, float]], tuple[int, int, int], str]] = []
    for run_index, run in enumerate(runs):
        run_series = series_getter(run)
        for series_index, (label, series) in enumerate(run_series):
            if not series:
                continue
            color = PALETTE[(run_index + series_index) % len(PALETTE)]
            marker = "square" if series_index % 2 else "circle"
            plot_series.append((label, series, color, marker))

    if not plot_series:
        raise ValueError(f"No plottable series for {title}.")

    if fixed_value_range is None:
        value_min, value_max = value_range([series for _, series, _, _ in plot_series])
    else:
        value_min, value_max = fixed_value_range
    epoch_min, epoch_max = epoch_range([run.history for run in runs])

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_axes(
        draw,
        title=title,
        width=width,
        height=height,
        plot_left=plot_left,
        plot_top=plot_top,
        plot_width=plot_width,
        plot_height=plot_height,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        value_min=value_min,
        value_max=value_max,
    )

    legend_items: list[tuple[str, tuple[int, int, int]]] = []
    for label, series, color, marker in plot_series:
        draw_series(
            draw,
            series=series,
            color=color,
            marker=marker,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            value_min=value_min,
            value_max=value_max,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        legend_items.append((label, color))

    draw_legend(draw, labels=legend_items, x=855, y=70)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_runs(paths: list[Path], labels: list[str]) -> list[HistoryRun]:
    if labels and len(labels) != len(paths):
        raise ValueError("--label must be provided once per input history file.")

    runs: list[HistoryRun] = []
    for index, path in enumerate(paths):
        name = labels[index] if labels else path.stem
        runs.append(HistoryRun(name=name, path=path, history=load_history(path)))
    return runs


def loss_series(run: HistoryRun) -> list[tuple[str, list[tuple[int, float]]]]:
    if len(run.history) == 0:
        return []

    if len(run.name) > 20:
        prefix = run.name[:20] + "..."
    else:
        prefix = run.name
    return [
        (f"{prefix} train_loss", numeric_series(run.history, "train_loss")),
        (f"{prefix} val_loss", numeric_series(run.history, "val_loss")),
    ]


def accuracy_series(run: HistoryRun) -> list[tuple[str, list[tuple[int, float]]]]:
    accuracy_key = first_available_key(run.history, ["val_accuracy", "val_acc", "accuracy"])
    if accuracy_key is None:
        return []
    return [(run.name, numeric_series(run.history, accuracy_key))]


def lr_series(run: HistoryRun) -> list[tuple[str, list[tuple[int, float]]]]:
    return [(run.name, numeric_series(run.history, "lr"))]


def main() -> None:
    args = parse_args()
    runs = build_runs(args.history, args.label)
    prefix = args.prefix or ("comparison" if len(runs) > 1 else runs[0].path.stem)

    loss_path = args.output_dir / f"{prefix}_loss.png"
    accuracy_path = args.output_dir / f"{prefix}_accuracy.png"
    lr_path = args.output_dir / f"{prefix}_learning_rate.png"

    render_plot(
        title="Training and Validation Loss",
        runs=runs,
        series_getter=loss_series,
        output_path=loss_path,
    )
    render_plot(
        title="Validation Accuracy",
        runs=runs,
        series_getter=accuracy_series,
        output_path=accuracy_path,
        fixed_value_range=(0.0, 1.0),
    )
    render_plot(
        title="Learning Rate",
        runs=runs,
        series_getter=lr_series,
        output_path=lr_path,
    )

    for run in runs:
        epochs = [int(item.get("epoch", index)) for index, item in enumerate(run.history, start=1)]
        keys = sorted({key for item in run.history for key in item})
        accuracy_key = first_available_key(run.history, ["val_accuracy", "val_acc", "accuracy"])
        print(
            f"Loaded {len(run.history)} history records from {run.path} "
            f"as {run.name} (epochs {min(epochs)}..{max(epochs)})."
        )
        print(f"Available keys: {', '.join(keys)}")
        print(f"Accuracy key: {accuracy_key}")
    print(f"Wrote {loss_path}")
    print(f"Wrote {accuracy_path}")
    print(f"Wrote {lr_path}")


if __name__ == "__main__":
    main()
