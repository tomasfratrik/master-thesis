from __future__ import annotations

from functools import lru_cache
from typing import Any

from .geometry import box_iou, boxes_overlap, expand_box


def nms(
    detections: list[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda x: x["score"], reverse=True):
        current_box = det["box_xyxy"]
        if all(box_iou(current_box, kept_det["box_xyxy"]) < iou_threshold for kept_det in kept):
            kept.append(det)
    return kept


def is_same_shoe_detection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    center_ratio: float,
    iou_threshold: float,
    area_ratio_threshold: float,
) -> bool:
    if box_iou(a, b) < iou_threshold:
        return False

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    area_a = aw * ah
    area_b = bw * bh
    area_ratio = min(area_a, area_b) / max(area_a, area_b)
    if area_ratio < area_ratio_threshold:
        return False

    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    size_scale = min((aw + ah) / 2.0, (bw + bh) / 2.0)
    return center_dist <= center_ratio * size_scale


def dedup_same_shoe_detections(
    detections: list[dict[str, Any]],
    center_ratio: float,
    iou_threshold: float,
    area_ratio_threshold: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda x: x["score"], reverse=True):
        box = det["box_xyxy"]
        if all(
            not is_same_shoe_detection(
                box,
                kept_det["box_xyxy"],
                center_ratio=center_ratio,
                iou_threshold=iou_threshold,
                area_ratio_threshold=area_ratio_threshold,
            )
            for kept_det in kept
        ):
            kept.append(det)
    return kept


def group_nearby_detections(
    detections: list[dict[str, Any]],
    gap_ratio: float,
) -> list[dict[str, Any]]:
    if len(detections) <= 1:
        return detections

    def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    n = len(detections)
    valid_edge: list[list[bool]] = [[False] * n for _ in range(n)]
    edge_cost: list[list[float]] = [[0.0] * n for _ in range(n)]
    candidate_edges: list[tuple[float, float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            box_i = detections[i]["box_xyxy"]
            box_j = detections[j]["box_xyxy"]
            expanded_i = expand_box(box_i, gap_ratio)
            expanded_j = expand_box(box_j, gap_ratio)
            if not boxes_overlap(expanded_i, expanded_j):
                continue

            dist = center_distance(box_i, box_j)
            iou = box_iou(box_i, box_j)
            valid_edge[i][j] = True
            valid_edge[j][i] = True
            edge_cost[i][j] = dist
            edge_cost[j][i] = dist
            candidate_edges.append((dist, -iou, i, j))

    chosen_pairs: list[tuple[int, int]] = []
    if n <= 18:

        @lru_cache(maxsize=None)
        def solve(mask: int) -> tuple[int, float]:
            first = -1
            for idx in range(n):
                if not (mask & (1 << idx)):
                    first = idx
                    break
            if first == -1:
                return (0, 0.0)

            best_pairs, best_cost = solve(mask | (1 << first))

            for j in range(first + 1, n):
                if mask & (1 << j):
                    continue
                if not valid_edge[first][j]:
                    continue
                sub_pairs, sub_cost = solve(mask | (1 << first) | (1 << j))
                cand_pairs = sub_pairs + 1
                cand_cost = sub_cost + edge_cost[first][j]
                if cand_pairs > best_pairs or (
                    cand_pairs == best_pairs and cand_cost < best_cost
                ):
                    best_pairs, best_cost = cand_pairs, cand_cost
            return best_pairs, best_cost

        def reconstruct(mask: int) -> None:
            first = -1
            for idx in range(n):
                if not (mask & (1 << idx)):
                    first = idx
                    break
            if first == -1:
                return

            best_pairs, best_cost = solve(mask)

            skip_pairs, skip_cost = solve(mask | (1 << first))
            if skip_pairs == best_pairs and abs(skip_cost - best_cost) <= 1e-9:
                reconstruct(mask | (1 << first))
                return

            for j in range(first + 1, n):
                if mask & (1 << j):
                    continue
                if not valid_edge[first][j]:
                    continue
                sub_pairs, sub_cost = solve(mask | (1 << first) | (1 << j))
                cand_pairs = sub_pairs + 1
                cand_cost = sub_cost + edge_cost[first][j]
                if cand_pairs == best_pairs and abs(cand_cost - best_cost) <= 1e-9:
                    chosen_pairs.append((first, j))
                    reconstruct(mask | (1 << first) | (1 << j))
                    return

            reconstruct(mask | (1 << first))

        reconstruct(0)
    else:
        candidate_edges.sort(key=lambda x: (x[0], x[1]))
        matched: set[int] = set()
        for _, _, i, j in candidate_edges:
            if i in matched or j in matched:
                continue
            matched.add(i)
            matched.add(j)
            chosen_pairs.append((i, j))

    matched = {idx for pair in chosen_pairs for idx in pair}

    merged: list[dict[str, Any]] = []

    for i, j in chosen_pairs:
        members = [detections[i], detections[j]]
        xs1 = [m["box_xyxy"][0] for m in members]
        ys1 = [m["box_xyxy"][1] for m in members]
        xs2 = [m["box_xyxy"][2] for m in members]
        ys2 = [m["box_xyxy"][3] for m in members]
        merged.append(
            {
                "label": "shoe_pair",
                "score": max(m["score"] for m in members),
                "box_xyxy": (min(xs1), min(ys1), max(xs2), max(ys2)),
                "members": 2,
            }
        )

    for idx, det in enumerate(detections):
        if idx in matched:
            continue
        members = [det]
        xs1 = [m["box_xyxy"][0] for m in members]
        ys1 = [m["box_xyxy"][1] for m in members]
        xs2 = [m["box_xyxy"][2] for m in members]
        ys2 = [m["box_xyxy"][3] for m in members]
        merged.append(
            {
                "label": "shoe_pair" if len(members) > 1 else members[0]["label"],
                "score": max(m["score"] for m in members),
                "box_xyxy": (min(xs1), min(ys1), max(xs2), max(ys2)),
                "members": len(members),
            }
        )

    return sorted(merged, key=lambda x: x["score"], reverse=True)


def is_pair_label(label: str) -> bool:
    return "pair" in label.lower()


def split_pair_and_single_detections(
    detections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_detections: list[dict[str, Any]] = []
    single_detections: list[dict[str, Any]] = []
    for det in detections:
        if is_pair_label(str(det.get("label", ""))):
            pair_detections.append(
                {
                    **det,
                    "label": "shoe_pair",
                    "members": int(det.get("members", 2)),
                }
            )
        else:
            single_detections.append(det)
    return pair_detections, single_detections


def suppress_single_overlaps_with_pairs(
    pair_detections: list[dict[str, Any]],
    single_detections: list[dict[str, Any]],
    overlap_threshold: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for det in single_detections:
        overlaps_pair = any(
            box_iou(det["box_xyxy"], pair_det["box_xyxy"]) >= overlap_threshold
            for pair_det in pair_detections
        )
        if not overlaps_pair:
            kept.append(det)
    return kept


def finalize_detections(
    raw_detections: list[dict[str, Any]],
    nms_threshold: float,
    same_shoe_dedup: bool,
    same_shoe_center_ratio: float,
    same_shoe_iou_threshold: float,
    same_shoe_area_ratio: float,
    final_nms_threshold: float,
    group_pairs: bool,
    prefer_pair_labels: bool,
    pair_overlap_threshold: float,
    pair_gap_ratio: float,
) -> list[dict[str, Any]]:
    filtered = nms(raw_detections, nms_threshold)
    if same_shoe_dedup:
        filtered = dedup_same_shoe_detections(
            filtered,
            center_ratio=same_shoe_center_ratio,
            iou_threshold=same_shoe_iou_threshold,
            area_ratio_threshold=same_shoe_area_ratio,
        )

    if not group_pairs:
        merged_detections = filtered
    else:
        if prefer_pair_labels:
            pair_detections, single_detections = split_pair_and_single_detections(filtered)
            single_detections = suppress_single_overlaps_with_pairs(
                pair_detections,
                single_detections,
                overlap_threshold=pair_overlap_threshold,
            )
            grouped_single_detections = group_nearby_detections(
                single_detections, pair_gap_ratio
            )
            merged_detections = pair_detections + grouped_single_detections
        else:
            _, single_detections = split_pair_and_single_detections(filtered)
            base_detections = single_detections if single_detections else filtered
            merged_detections = group_nearby_detections(base_detections, pair_gap_ratio)

    if final_nms_threshold > 0:
        merged_detections = nms(merged_detections, final_nms_threshold)
    return merged_detections
