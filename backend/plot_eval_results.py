"""Render comparison plots from evaluation JSON reports."""

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


class EvalRun:
    def __init__(self, *, name: str, path: Path, data: dict[str, Any]) -> None:
        self.name = name
        self.path = path
        self.data = data

    @property
    def summary(self) -> dict[str, Any]:
        return self.data.get("summary", {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot summary, per-tag, and per-class metrics from evaluation JSON files."
    )
    parser.add_argument("report", nargs="+", type=Path, help="Evaluation JSON report(s).")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for a report. Repeat once per report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "eval_plots",
        help="Directory where PNG plots are written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to comparison or the report stem.",
    )
    parser.add_argument(
        "--metric",
        choices=["top1_accuracy", "topk_accuracy"],
        default="top1_accuracy",
        help="Metric used for per-tag and per-class plots.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only render the summary plot. Useful when comparing old reports without tags.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Evaluation report must be a JSON object: {path}")
    return data


def build_runs(paths: list[Path], labels: list[str]) -> list[EvalRun]:
    if labels and len(labels) != len(paths):
        raise ValueError("--label must be provided once per input report.")

    runs: list[EvalRun] = []
    for index, path in enumerate(paths):
        name = labels[index] if labels else path.stem
        runs.append(EvalRun(name=name, path=path, data=load_report(path)))
    return runs


def short_label(label: str, max_len: int = 24) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def wrap_label(label: str, max_line_len: int = 14, max_lines: int = 3) -> list[str]:
    words = label.replace("_", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_line_len:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word

    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = short_label(lines[-1], max_line_len)
    return lines or [short_label(label, max_line_len)]


def draw_legend(
    draw: ImageDraw.ImageDraw,
    runs: list[EvalRun],
    *,
    x: int,
    y: int,
) -> None:
    for index, run in enumerate(runs):
        color = PALETTE[index % len(PALETTE)]
        item_y = y + index * 20
        draw.rectangle((x, item_y + 4, x + 14, item_y + 14), fill=color)
        draw.text((x + 20, item_y), short_label(run.name, 30), fill=(35, 35, 35))


def draw_grouped_bar_chart(
    *,
    title: str,
    categories: list[str],
    runs: list[EvalRun],
    values_by_run: list[list[float]],
    output_path: Path,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    width = max(1000, 170 + len(categories) * 74)
    height = 720
    plot_left = 90
    plot_top = 58
    plot_width = width - 250
    plot_height = 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

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
        value = y_max - ratio * (y_max - y_min)
        draw.line((plot_left, y, plot_left + plot_width, y), fill=grid_color, width=1)
        draw.text((18, y - 7), f"{value:.1f}", fill=text_color)

    group_width = plot_width / max(1, len(categories))
    bar_gap = 3
    bar_width = max(5, int((group_width - 12) / max(1, len(runs))))

    for category_index, category in enumerate(categories):
        group_x = plot_left + category_index * group_width
        for run_index, values in enumerate(values_by_run):
            if category_index >= len(values):
                continue
            value = values[category_index]
            clamped = max(y_min, min(y_max, value))
            ratio = (clamped - y_min) / (y_max - y_min)
            bar_height = round(ratio * plot_height)
            x0 = int(group_x + 6 + run_index * (bar_width + bar_gap))
            x1 = x0 + bar_width
            y0 = plot_top + plot_height - bar_height
            y1 = plot_top + plot_height
            color = PALETTE[run_index % len(PALETTE)]
            draw.rectangle((x0, y0, x1, y1), fill=color)

        label_lines = wrap_label(category, max_line_len=13, max_lines=3)
        label_x = int(group_x + 4)
        for line_index, line in enumerate(label_lines):
            draw.text(
                (label_x, plot_top + plot_height + 12 + line_index * 15),
                line,
                fill=text_color,
            )

    draw_legend(draw, runs, x=width - 145, y=70)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def summary_plot(runs: list[EvalRun], output_path: Path) -> None:
    categories = ["top1_accuracy", "mean_margin_vs_second"]
    values_by_run = [
        [float(run.summary.get(category, 0.0) or 0.0) for category in categories]
        for run in runs
    ]
    draw_grouped_bar_chart(
        title="Evaluation Summary",
        categories=categories,
        runs=runs,
        values_by_run=values_by_run,
        output_path=output_path,
    )


def sorted_keys_for_section(runs: list[EvalRun], section: str) -> list[str]:
    keys: set[str] = set()
    totals: dict[str, int] = {}
    for run in runs:
        data = run.data.get(section, {})
        if not isinstance(data, dict):
            continue
        keys.update(data.keys())
        for key, item in data.items():
            if isinstance(item, dict):
                totals[key] = max(totals.get(key, 0), int(item.get("total", 0) or 0))
    return sorted(keys, key=lambda key: (-totals.get(key, 0), key))


def has_section_data(runs: list[EvalRun], section: str) -> bool:
    return any(isinstance(run.data.get(section), dict) and bool(run.data.get(section)) for run in runs)


def section_plot(
    *,
    runs: list[EvalRun],
    section: str,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    categories = sorted_keys_for_section(runs, section)
    values_by_run: list[list[float]] = []
    for run in runs:
        section_data = run.data.get(section, {})
        values: list[float] = []
        for category in categories:
            item = section_data.get(category, {}) if isinstance(section_data, dict) else {}
            values.append(float(item.get(metric, 0.0) or 0.0) if isinstance(item, dict) else 0.0)
        values_by_run.append(values)

    draw_grouped_bar_chart(
        title=title,
        categories=categories,
        runs=runs,
        values_by_run=values_by_run,
        output_path=output_path,
    )


def main() -> None:
    args = parse_args()
    runs = build_runs(args.report, args.label)
    prefix = args.prefix or ("comparison" if len(runs) > 1 else runs[0].path.stem)

    summary_path = args.output_dir / f"{prefix}_summary.png"
    tag_path = args.output_dir / f"{prefix}_per_tag_{args.metric}.png"
    class_path = args.output_dir / f"{prefix}_per_class_{args.metric}.png"

    summary_plot(runs, summary_path)
    wrote_paths = [summary_path]
    if not args.summary_only and has_section_data(runs, "per_tag"):
        section_plot(
            runs=runs,
            section="per_tag",
            metric=args.metric,
            title=f"Per-Tag {args.metric}",
            output_path=tag_path,
        )
        wrote_paths.append(tag_path)
    if not args.summary_only and has_section_data(runs, "per_class"):
        section_plot(
            runs=runs,
            section="per_class",
            metric=args.metric,
            title=f"Per-Class {args.metric}",
            output_path=class_path,
        )
        wrote_paths.append(class_path)

    for run in runs:
        summary = run.summary
        print(
            f"Loaded {run.path} as {run.name}: "
            f"images={summary.get('images_evaluated')}, "
            f"top1={summary.get('top1_accuracy')}, "
            f"top5={summary.get('top5_accuracy')}"
        )
    for path in wrote_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
