#!/usr/bin/env python3
"""Extract the road-platform and supply-enclosure walls from ``map.webp``.

The arena texture contains many decorative seams, so a completely unconstrained
contour finder also selects floor tiles and team-colour paint.  This extractor
uses Canny/Hough inside small, named search bands around the physical walls.  It
therefore snaps the geometry to the pixels in the source map while preserving
the four user-confirmed road passages as explicit openings.

Only the blue side is detected directly.  The caller mirrors the result by 180
degrees for the red side, matching the physical field symmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np


Orientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class WallSegment:
    wall_id: str
    category: str
    start_px: tuple[float, float]
    end_px: tuple[float, float]
    thickness_px: float = 18.0


@dataclass(frozen=True)
class RoadEnclosure:
    region_px: tuple[tuple[float, float], ...]
    walls: tuple[WallSegment, ...]
    detected_lines: dict[str, tuple[tuple[float, float], tuple[float, float]]]


def _canny(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 7, 28, 28)
    return cv2.Canny(filtered, 32, 96)


def _axis_coordinate(
    edges: np.ndarray,
    orientation: Orientation,
    expected: float,
    search: tuple[int, int, int, int],
) -> float:
    """Return the Hough-supported axis nearest an expected physical wall."""
    x1, y1, x2, y2 = search
    crop = edges[y1:y2, x1:x2]
    lines = cv2.HoughLinesP(
        crop, 1, np.pi / 360, threshold=24, minLineLength=45, maxLineGap=18,
    )
    candidates: list[tuple[float, float]] = []
    if lines is not None:
        for lx1, ly1, lx2, ly2 in lines.reshape(-1, 4):
            dx, dy = float(lx2 - lx1), float(ly2 - ly1)
            length = math.hypot(dx, dy)
            if orientation == "horizontal" and abs(dy) <= max(2.0, abs(dx) * 0.08):
                coordinate = y1 + (ly1 + ly2) / 2
            elif orientation == "vertical" and abs(dx) <= max(2.0, abs(dy) * 0.08):
                coordinate = x1 + (lx1 + lx2) / 2
            else:
                continue
            candidates.append((coordinate, length))
    if not candidates:
        return float(expected)
    candidates.sort(key=lambda item: abs(item[0] - expected) - min(item[1], 240) * 0.003)
    close = [item for item in candidates if abs(item[0] - candidates[0][0]) <= 7]
    return float(sum(value * weight for value, weight in close) / sum(weight for _, weight in close))


def _angled_line(
    edges: np.ndarray,
    expected_start: tuple[float, float],
    expected_end: tuple[float, float],
    search: tuple[int, int, int, int],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Snap a slanted wall to the best matching Hough segment."""
    x1, y1, x2, y2 = search
    crop = edges[y1:y2, x1:x2]
    lines = cv2.HoughLinesP(
        crop, 1, np.pi / 720, threshold=24, minLineLength=55, maxLineGap=22,
    )
    expected_angle = math.atan2(
        expected_end[1] - expected_start[1], expected_end[0] - expected_start[0],
    )
    expected_mid = (
        (expected_start[0] + expected_end[0]) / 2,
        (expected_start[1] + expected_end[1]) / 2,
    )
    best: tuple[float, tuple[float, float], tuple[float, float]] | None = None
    if lines is not None:
        for raw in lines.reshape(-1, 4):
            start = (float(raw[0] + x1), float(raw[1] + y1))
            end = (float(raw[2] + x1), float(raw[3] + y1))
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            angle_error = abs(math.atan2(math.sin(angle - expected_angle), math.cos(angle - expected_angle)))
            angle_error = min(angle_error, abs(math.pi - angle_error))
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            midpoint_error = math.hypot(midpoint[0] - expected_mid[0], midpoint[1] - expected_mid[1])
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            score = angle_error * 130 + midpoint_error - min(length, 180) * 0.12
            if best is None or score < best[0]:
                best = (score, start, end)
    if best is None:
        return expected_start, expected_end

    # Hough endpoints often stop at paint/texture gaps.  Keep their fitted line
    # but project the semantic corner anchors onto it so the enclosure closes.
    _, start, end = best
    direction = np.asarray(end) - np.asarray(start)
    direction /= max(float(np.linalg.norm(direction)), 1e-6)
    origin = np.asarray(start)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        projected = origin + direction * float(np.dot(np.asarray(point) - origin, direction))
        return float(projected[0]), float(projected[1])

    return project(expected_start), project(expected_end)


def extract_blue_enclosures(map_path: Path) -> RoadEnclosure:
    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(map_path)
    if image.shape[1] != 2337 or image.shape[0] != 1283:
        raise ValueError(f"unexpected map dimensions: {image.shape[1]}x{image.shape[0]}")
    edges = _canny(image)

    left_x = _axis_coordinate(edges, "vertical", 1138, (1122, 24, 1154, 336))
    right_x = _axis_coordinate(edges, "vertical", 2026, (2011, 24, 2036, 360))
    top_y = _axis_coordinate(edges, "horizontal", 36, (1125, 25, 2034, 48))
    centre_y = _axis_coordinate(edges, "horizontal", 210, (1485, 194, 1895, 224))
    bottom_y = _axis_coordinate(edges, "horizontal", 351, (1738, 335, 2034, 365))
    right_slope = _angled_line(edges, (1677, centre_y), (1751, bottom_y), (1658, 190, 1770, 363))

    # Supply-zone fence axes are exceptionally clear in the original image.
    supply_top_y = _axis_coordinate(edges, "horizontal", 40, (2008, 30, 2150, 49))
    supply_mid_y = _axis_coordinate(edges, "horizontal", 182, (2008, 172, 2150, 191))
    supply_low_y = _axis_coordinate(edges, "horizontal", 271, (2008, 260, 2220, 281))

    right_slope_start, right_slope_end = right_slope
    region = (
        (left_x, top_y), (right_x, top_y), (right_x, bottom_y),
        right_slope_end, right_slope_start, (left_x, centre_y),
    )

    # Gate openings use the user-confirmed B1/B2/B3 tolerance widths.  A small
    # extra margin prevents a 0.35 m route grid node from grazing a wall corner.
    b1_top, b1_bottom = 84 - 43, 84 + 43
    b2_left, b2_right = 1549 - 38, 1549 + 38
    b3_left, b3_right = 1642 - 38, 1642 + 38
    centre_end = max(right_slope_start[0], 1677)

    walls = (
        WallSegment("blue_road_left_lower", "road_fence", (left_x, b1_bottom), (left_x, centre_y)),
        WallSegment("blue_road_lower_left", "road_fence", (left_x, centre_y), (b2_left, centre_y)),
        WallSegment("blue_road_gate_separator_23", "road_fence", (b2_right, centre_y), (b3_left, centre_y)),
        WallSegment("blue_road_right_slope", "road_fence", right_slope_start, right_slope_end),
        WallSegment("blue_road_lower_right", "road_fence", right_slope_end, (right_x, bottom_y)),
        WallSegment("blue_road_outer_right", "road_fence", (right_x, top_y), (right_x, bottom_y)),
        # This inner rail is visible between the B3 structure and the rough-road section.
        WallSegment("blue_road_inner_rail", "road_fence", (centre_end, centre_y), (1889, centre_y), 15.0),
        WallSegment("blue_supply_middle_fence", "supply_fence", (2037, supply_mid_y), (2142, supply_mid_y), 14.0),
        WallSegment("blue_supply_lower_fence", "supply_fence", (2015, supply_low_y), (2211, supply_low_y), 14.0),
    )
    detected = {
        "road_left": ((left_x, top_y), (left_x, centre_y)),
        "road_right": ((right_x, top_y), (right_x, bottom_y)),
        "road_right_slope": right_slope,
        "supply_top": ((2016, supply_top_y), (2142, supply_top_y)),
        "supply_middle": ((2037, supply_mid_y), (2142, supply_mid_y)),
        "supply_lower": ((2015, supply_low_y), (2211, supply_low_y)),
    }
    return RoadEnclosure(region_px=region, walls=walls, detected_lines=detected)


def segment_polygon(segment: WallSegment) -> tuple[tuple[float, float], ...]:
    """Convert a wall centre line into a closed, thick collision polygon."""
    sx, sy = segment.start_px
    ex, ey = segment.end_px
    length = math.hypot(ex - sx, ey - sy)
    if length < 1e-6:
        return ()
    nx = -(ey - sy) / length * segment.thickness_px / 2
    ny = (ex - sx) / length * segment.thickness_px / 2
    return ((sx + nx, sy + ny), (ex + nx, ey + ny), (ex - nx, ey - ny), (sx - nx, sy - ny))


def draw_debug_overlay(map_path: Path, output_path: Path) -> None:
    data = extract_blue_enclosures(map_path)
    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    overlay = image.copy()
    region = np.asarray(data.region_px, dtype=np.int32)
    cv2.fillPoly(overlay, [region], (150, 80, 20))
    image = cv2.addWeighted(overlay, 0.20, image, 0.80, 0)
    cv2.polylines(image, [region], True, (255, 210, 40), 3, cv2.LINE_AA)
    colours = {"road_fence": (0, 220, 255), "supply_fence": (255, 120, 40), "supply_structure": (180, 80, 255)}
    for wall in data.walls:
        polygon = np.asarray(segment_polygon(wall), dtype=np.int32)
        cv2.fillPoly(image, [polygon], colours[wall.category])
        midpoint = (
            round((wall.start_px[0] + wall.end_px[0]) / 2),
            round((wall.start_px[1] + wall.end_px[1]) / 2),
        )
        cv2.putText(image, wall.wall_id.removeprefix("blue_"), midpoint, cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    draw_debug_overlay(root / "docs" / "assets" / "map.webp", root / "analysis" / "outputs" / "road_enclosure_cv.png")
