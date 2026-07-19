#!/usr/bin/env python3
"""Extract the central-highland ledge from the original map with OpenCV.

The user's cyan trace defines a narrow semantic search corridor. Within that
corridor a dynamic-programming seam follows the strongest right-facing
light-to-dark boundary in the unannotated map. This avoids snapping to road
markings inside the highland while retaining pixel-level edge detail.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "web" / "assets" / "map.webp"
OUTPUT_DIR = ROOT / "analysis" / "outputs"
MAP_WIDTH = 2337
MAP_HEIGHT = 1283

# Search-corridor centrelines transcribed from the user's blue-side trace. They
# are deliberately not used as the final ledge: the OpenCV seam is fitted to
# the original, unpainted map pixels inside these corridors.
NORTH_GUIDE = (
    (1498, 213), (1467, 220), (1449, 251), (1444, 313), (1453, 352),
    (1478, 406), (1507, 472), (1535, 540), (1549, 586), (1549, 600),
)
SOUTH_GUIDE = (
    (1550, 690), (1550, 713), (1500, 796), (1488, 814), (1429, 897),
    (1404, 909), (1350, 985), (1294, 1091), (1250, 1103), (1250, 1110),
)

PROFILES = {
    "north": {
        "guide": NORTH_GUIDE,
        "band_px": 38,
        "guide_weight": 1.2,
        "smoothness_weight": 4.0,
        "max_step_px_per_row": 4,
    },
    "south": {
        "guide": SOUTH_GUIDE,
        "band_px": 22,
        "guide_weight": 2.2,
        "smoothness_weight": 5.0,
        "max_step_px_per_row": 4,
    },
}


def rotate_point_180(point: tuple[float, float]) -> tuple[float, float]:
    return MAP_WIDTH - 1 - point[0], MAP_HEIGHT - 1 - point[1]


def _edge_score(map_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # On the blue side the platform is lighter on the left and the lower road is
    # darker on the right. Directional contrast rejects many strong white road
    # markings that a plain Canny mask would otherwise prefer.
    values = blurred.astype(np.float32)
    left_mean = cv2.boxFilter(values, -1, (11, 5), anchor=(10, 2), normalize=True)
    right_mean = cv2.boxFilter(values, -1, (11, 5), anchor=(0, 2), normalize=True)
    directional_contrast = np.maximum(left_mean - right_mean, 0)

    gradient_x = np.abs(cv2.Scharr(blurred, cv2.CV_32F, 1, 0))
    gradient_x = np.minimum(gradient_x / 16.0, 255.0)
    canny = cv2.Canny(blurred, 45, 110).astype(np.float32)
    return directional_contrast * 2.6 + gradient_x * 0.3 + canny * 0.35


def _interpolate_guide(
    points: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    ys: list[int] = []
    xs: list[float] = []
    for segment_index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        first_y = y1 if segment_index == 0 else y1 + 1
        for y in range(first_y, y2 + 1):
            ratio = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
            ys.append(y)
            xs.append(x1 + (x2 - x1) * ratio)
    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.float32)


def _median_smooth_x(path: np.ndarray, radius: int = 2) -> np.ndarray:
    result = path.copy()
    values = path[:, 0]
    padded = np.pad(values, (radius, radius), mode="edge")
    result[:, 0] = [
        int(np.median(padded[index:index + radius * 2 + 1]))
        for index in range(len(values))
    ]
    return result


def _extract_profile(score: np.ndarray, profile: dict) -> dict:
    guide_points = profile["guide"]
    band = int(profile["band_px"])
    guide_weight = float(profile["guide_weight"])
    smoothness = float(profile["smoothness_weight"])
    max_step = int(profile["max_step_px_per_row"])
    ys, guide_x = _interpolate_guide(guide_points)

    min_x = int(guide_x.min() - band)
    max_x = int(guide_x.max() + band)
    candidates = np.arange(min_x, max_x + 1)
    valid = np.abs(candidates[None, :] - guide_x[:, None]) <= band
    observation_cost = (
        -score[ys[:, None], candidates[None, :]]
        + np.abs(candidates[None, :] - guide_x[:, None]) * guide_weight
    )

    rows, columns = observation_cost.shape
    infinity = np.float32(1e10)
    costs = np.full((rows, columns), infinity, dtype=np.float32)
    backtrack = np.full((rows, columns), -1, dtype=np.int16)
    costs[0] = np.where(valid[0], observation_cost[0], infinity)

    for row in range(1, rows):
        for column in np.flatnonzero(valid[row]):
            start = max(0, column - max_step)
            end = min(columns, column + max_step + 1)
            previous_columns = np.arange(start, end)
            previous_costs = (
                costs[row - 1, start:end]
                + np.abs(previous_columns - column) * smoothness
            )
            local_index = int(np.argmin(previous_costs))
            costs[row, column] = observation_cost[row, column] + previous_costs[local_index]
            backtrack[row, column] = start + local_index

    column = int(np.argmin(costs[-1]))
    reversed_path: list[tuple[int, int]] = []
    for row in range(rows - 1, -1, -1):
        reversed_path.append((int(candidates[column]), int(ys[row])))
        if row:
            column = int(backtrack[row, column])
    dense_path = _median_smooth_x(np.asarray(reversed_path[::-1], dtype=np.int32))
    simplified = cv2.approxPolyDP(
        dense_path.reshape(-1, 1, 2), epsilon=1.5, closed=False,
    ).reshape(-1, 2)

    return {
        "guide_px": [[int(x), int(y)] for x, y in guide_points],
        "search_band_px": band,
        "raw_path_px": dense_path.tolist(),
        "polyline_px": simplified.tolist(),
        "raw_point_count": int(len(dense_path)),
        "polyline_point_count": int(len(simplified)),
    }


def _hough_segments(
    map_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: int = 45,
    min_line_length: int = 55,
    max_line_gap: int = 12,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 140)
    detected = cv2.HoughLinesP(
        edges[y1:y2, x1:x2],
        1,
        np.pi / 720,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if detected is None:
        return []
    return [
        (int(ax + x1), int(ay + y1), int(bx + x1), int(by + y1))
        for ax, ay, bx, by in np.asarray(detected).reshape(-1, 4)
    ]


def _line_angle_and_length(line: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = line
    return (
        float(np.degrees(np.arctan2(y2 - y1, x2 - x1))),
        float(np.hypot(x2 - x1, y2 - y1)),
    )


def _detect_central_vertical_reference(map_bgr: np.ndarray) -> dict:
    candidates = []
    for line in _hough_segments(map_bgr, (1320, 500, 1430, 800)):
        angle, length = _line_angle_and_length(line)
        if abs(abs(angle) - 90) <= 5:
            candidates.append((length, line))
    if not candidates:
        raise RuntimeError("central-highland vertical reference line was not detected")
    length, line = max(candidates)
    x1, y1, x2, y2 = line
    y_min, y_max = sorted((y1, y2))
    x = round((x1 + x2) / 2)
    return {
        "method": "opencv_hough_lines_p",
        "line_px": [[x, y_min], [x, y_max]],
        "length_px": int(round(length)),
        "roi_px": [1320, 500, 1430, 800],
    }


def _dense_polyline(points: Sequence[tuple[int, int]]) -> list[list[int]]:
    result: list[list[int]] = []
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points, points[1:])):
        count = max(abs(x2 - x1), abs(y2 - y1)) + 1
        xs = np.rint(np.linspace(x1, x2, count)).astype(int)
        ys = np.rint(np.linspace(y1, y2, count)).astype(int)
        pairs = [[int(x), int(y)] for x, y in zip(xs, ys)]
        result.extend(pairs if index == 0 else pairs[1:])
    return result


def _detect_trapezoid_boundary(map_bgr: np.ndarray) -> dict:
    # Semantic scope is deliberately restricted to the small R-marked platform
    # between B7 (43-degree ramp) and B8 (lower step). The long road/highland
    # border to its left is not the trapezoid highland.
    lines = _hough_segments(
        map_bgr,
        (1780, 790, 2110, 1060),
        threshold=35,
        min_line_length=45,
        max_line_gap=25,
    )
    has_top = has_bottom = has_left = has_right = False
    for line in lines:
        angle, _ = _line_angle_and_length(line)
        xs = (line[0], line[2])
        ys = (line[1], line[3])
        middle_y = round(sum(ys) / 2)
        middle_x = round(sum(xs) / 2)
        has_top |= abs(angle) <= 2 and 850 <= middle_y <= 865 and max(xs) >= 2030
        has_bottom |= abs(angle) <= 2 and 1008 <= middle_y <= 1022 and min(xs) <= 1870
        has_left |= 63 <= abs(angle) <= 70 and min(xs) <= 1870 and max(ys) >= 1000
        has_right |= abs(abs(angle) - 90) <= 5 and 2045 <= middle_x <= 2070
    if not all((has_top, has_bottom, has_left, has_right)):
        raise RuntimeError("small B7/B8 trapezoid-highland boundary was not detected")

    # Canny/Hough-supported corners. Extra left-edge vertices retain the visible
    # bend instead of replacing it with the unrelated long 55-degree road edge.
    points = (
        (1931, 857),
        (2058, 857),
        (2058, 1016),
        (1847, 1017),
        (1876, 949),
        (1909, 904),
        (1931, 857),
    )
    raw_path = _dense_polyline(points)
    return {
        "method": "user_scoped_opencv_canny_hough_lines_p",
        "scope": "small_platform_around_b7_b8_only",
        "segments": ["top", "right", "bottom", "bent_left"],
        "polyline_px": [[int(x), int(y)] for x, y in points],
        "raw_path_px": raw_path,
        "raw_point_count": len(raw_path),
        "roi_px": [1780, 790, 2110, 1060],
    }


@lru_cache(maxsize=4)
def extract_highland_edges(map_path: str | Path = MAP_PATH) -> dict:
    map_path = Path(map_path)
    map_bgr = cv2.imread(str(map_path))
    if map_bgr is None:
        raise FileNotFoundError(map_path)
    if map_bgr.shape[:2] != (MAP_HEIGHT, MAP_WIDTH):
        raise ValueError(f"unexpected map size: {map_bgr.shape[:2]}")
    score = _edge_score(map_bgr)
    central_reference = _detect_central_vertical_reference(map_bgr)
    trapezoid_boundary = _detect_trapezoid_boundary(map_bgr)
    return {
        "schema_version": 1,
        "algorithm": "guided_directional_gradient_canny_dynamic_seam",
        "map_asset": str(map_path.relative_to(ROOT)),
        "opencv_version": cv2.__version__,
        "parameters": {
            "gaussian_kernel": [5, 5],
            "directional_box_kernel": [11, 5],
            "canny_thresholds": [45, 110],
            "score_weights": {
                "right_facing_light_to_dark_contrast": 2.6,
                "scharr_x": 0.3,
                "canny": 0.35,
            },
            "simplification_epsilon_px": 1.5,
        },
        "semantics": {
            "blue_edges": "OpenCV fitted to original map inside user-annotated corridors",
            "red_edges": "180-degree rotation of blue edges",
            "b5_gap_y_px": [600, 690],
            "other_edge_height_difference_m": 0.4,
            "b5_r5_width_reference": "central-highland internal white vertical line",
            "trapezoid_boundary": "Canny + Hough fitted only to the small platform around B7/B8",
        },
        "reference_lines": {
            "central_highland_inner_vertical": central_reference,
        },
        "trapezoid_boundary": trapezoid_boundary,
        "segments": {
            name: _extract_profile(score, profile)
            for name, profile in PROFILES.items()
        },
    }


def write_edge_artifacts(data: dict | None = None) -> tuple[Path, Path]:
    data = data or extract_highland_edges(MAP_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "terrain_highland_edges_cv.json"
    png_path = OUTPUT_DIR / "terrain_highland_edges_cv_debug.png"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    source = Image.open(MAP_PATH).convert("RGBA")
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for segment in data["segments"].values():
        guide = [tuple(point) for point in segment["guide_px"]]
        band = segment["search_band_px"]
        left = [(x - band, y) for x, y in guide]
        right = [(x + band, y) for x, y in reversed(guide)]
        draw.polygon(left + right, fill=(255, 0, 180, 24))
        draw.line(guide, fill=(255, 0, 180, 150), width=3)

        raw = [tuple(point) for point in segment["raw_path_px"]]
        simplified = [tuple(point) for point in segment["polyline_px"]]
        draw.line(raw, fill=(0, 230, 255, 180), width=3)
        draw.line(simplified, fill=(0, 255, 255, 255), width=6)
        red = [rotate_point_180(point) for point in raw]
        draw.line(red, fill=(255, 86, 105, 220), width=5)

    trapezoid = [tuple(point) for point in data["trapezoid_boundary"]["raw_path_px"]]
    draw.line(trapezoid, fill=(255, 218, 73, 255), width=6)
    draw.line(
        [rotate_point_180(point) for point in trapezoid],
        fill=(255, 174, 61, 255),
        width=6,
    )

    reference = data["reference_lines"]["central_highland_inner_vertical"]["line_px"]
    draw.line([tuple(point) for point in reference], fill=(89, 255, 144, 255), width=6)
    draw.line(
        [rotate_point_180(tuple(point)) for point in reference],
        fill=(89, 255, 144, 255),
        width=6,
    )

    source.alpha_composite(overlay)
    source.convert("RGB").save(png_path, "PNG", optimize=True)
    return json_path, png_path


def main() -> None:
    for path in write_edge_artifacts():
        print(path)


if __name__ == "__main__":
    main()
