from __future__ import annotations


def clamp_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    padding: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    box_w = max(1.0, xmax - xmin)
    box_h = max(1.0, ymax - ymin)
    extra_x = int(round(box_w * padding_ratio))
    extra_y = int(round(box_h * padding_ratio))

    x1 = max(0, int(round(xmin)) - padding - extra_x)
    y1 = max(0, int(round(ymin)) - padding - extra_y)
    x2 = min(width, int(round(xmax)) + padding + extra_x)
    y2 = min(height, int(round(ymax)) + padding + extra_y)
    return x1, y1, x2, y2


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def expand_box(
    box: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    padx = w * ratio
    pady = h * ratio
    return (x1 - padx, y1 - pady, x2 + padx, y2 + pady)
