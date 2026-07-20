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

    # These are physical walls, transcribed from the user's yellow correction.
    # Each guide is fitted back to an edge in the clean map; coloured field
    # paint is deliberately not treated as collision geometry.  Keeping the
    # individual segments also preserves every doorway/opening in the trace.
    # Registered from the 714x394 correction image to map.webp with a
    # 508-inlier SIFT/RANSAC homography.  Guides describe only the yellow
    # strokes; no unmarked interior structures are inferred as walls.
    wall_guides = (
        ("blue_road_top", "road_fence", (1188, 40), (2026, 40), 10.0),
        ("blue_road_outer_right", "road_fence", (2026, 40), (2026, 350), 10.0),
        ("blue_road_bottom_right", "road_fence", (1757, 350), (2026, 350), 10.0),
        ("blue_road_left_upper", "road_fence", (1102, 120), (1275, 120), 10.0),
        ("blue_road_left_lower", "road_fence", (1108, 190), (1235, 190), 10.0),
        ("blue_road_gate_post", "road_fence", (1504, 193), (1504, 226), 8.0),
        ("blue_road_structure_post", "road_fence", (1594, 186), (1594, 226), 8.0),
        ("blue_road_inner_rail", "road_fence", (1680, 203), (1896, 203), 8.0),
        ("blue_supply_outer_left", "supply_fence", (2218, 36), (2218, 273), 8.0),
        ("blue_supply_outer_bottom", "supply_fence", (2218, 273), (2308, 273), 8.0),
        ("blue_fortress_outer", "fortress_wall", (2278, 513), (2278, 786), 10.0),
        ("blue_base_top", "base_wall", (2055, 855), (2334, 855), 10.0),
        ("blue_base_left", "base_wall", (2055, 855), (2055, 1278), 10.0),
    )

    fitted_walls: list[WallSegment] = []
    for wall_id, category, start, end, thickness in wall_guides:
        margin = 28
        search = (
            max(0, int(min(start[0], end[0]) - margin)),
            max(0, int(min(start[1], end[1]) - margin)),
            min(image.shape[1], int(max(start[0], end[0]) + margin + 1)),
            min(image.shape[0], int(max(start[1], end[1]) + margin + 1)),
        )
        dx, dy = end[0] - start[0], end[1] - start[1]
        # The field walls are manufactured as an orthogonal layout.  OpenCV is
        # used only to locate the physical axis; it must not introduce the
        # small slopes seen in paint, shadows or perspective-resampling noise.
        if abs(dx) >= abs(dy) * 2:
            axis = _axis_coordinate(edges, "horizontal", (start[1] + end[1]) / 2, search)
            fitted_start = (float(start[0]), axis)
            fitted_end = (float(end[0]), axis)
        elif abs(dy) >= abs(dx) * 2:
            axis = _axis_coordinate(edges, "vertical", (start[0] + end[0]) / 2, search)
            fitted_start = (axis, float(start[1]))
            fitted_end = (axis, float(end[1]))
        else:
            fitted_start, fitted_end = _angled_line(edges, start, end, search)
        # A short decorative edge can be perpendicular to the guide and project
        # both semantic endpoints onto one pixel.  Such a degenerate fit is not
        # evidence for moving/removing the wall, so retain the user guide.
        expected_length = math.dist(start, end)
        if math.dist(fitted_start, fitted_end) < expected_length * 0.55:
            fitted_start, fitted_end = start, end
        fitted_walls.append(
            WallSegment(wall_id, category, fitted_start, fitted_end, thickness)
        )
    walls = tuple(fitted_walls)
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
    colours = {
        "road_fence": (0, 220, 255),
        "supply_fence": (255, 120, 40),
        "fortress_wall": (40, 220, 255),
        "base_wall": (40, 180, 255),
    }
    for wall in data.walls:
        polygon = np.asarray(segment_polygon(wall), dtype=np.int32)
        cv2.fillPoly(image, [polygon], colours[wall.category])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    draw_debug_overlay(root / "docs" / "assets" / "map.webp", root / "analysis" / "outputs" / "road_enclosure_cv.png")
