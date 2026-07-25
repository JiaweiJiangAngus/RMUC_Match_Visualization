#!/usr/bin/env python3
"""Build high-resolution position-density grids for all 96 regional teams."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "docs" / "data" / "catalog.json"
DEFAULT_GAMES = ROOT / "docs" / "data" / "games"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "heatmaps"
FIELD_WIDTH = 28.0
FIELD_HEIGHT = 15.0
CELL_SIZE_METRES = 0.1
GRID_WIDTH = round(FIELD_WIDTH / CELL_SIZE_METRES)
GRID_HEIGHT = round(FIELD_HEIGHT / CELL_SIZE_METRES)
GAUSSIAN_SIGMA_METRES = 0.22
WINDOW_SECONDS = 15
MOBILE_ROLES = {"英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"}
ROW = {"type": 1, "side": 2, "hp": 3, "x": 5, "y": 6}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sparse_grid(values: Counter) -> str:
    return ",".join(f"{index}:{values[index]}" for index in sorted(values))


def main() -> None:
    args = options()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    school_region: dict[str, str] = {}
    for region, matches in catalog["matches"].items():
        for match in matches:
            for school in (match["red"], match["blue"]):
                previous = school_region.setdefault(school, region)
                if previous != region:
                    raise ValueError(f"{school} appears in both {previous} and {region}")
    if len(school_region) != 96:
        raise ValueError(f"expected 96 schools, found {len(school_region)}")

    grids = {
        school: {
            "red": Counter(),
            "blue": Counter(),
        }
        for school in school_region
    }
    window_grids = {
        school: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
        for school in school_region
    }
    games = defaultdict(int)
    samples = defaultdict(int)
    max_window = 0

    for path in sorted(args.games_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            game = json.load(handle)
        info = game["info"]
        side_school = {"红": info["red"], "蓝": info["blue"]}
        games[info["red"]] += 1
        games[info["blue"]] += 1
        max_window = max(
            max_window,
            (int(info["duration"]) - 1) // WINDOW_SECONDS,
        )
        for second, robots in game.get("frames", {}).items():
            window = int(float(second)) // WINDOW_SECONDS
            for robot in robots:
                if robot[ROW["type"]] not in MOBILE_ROLES:
                    continue
                if float(robot[ROW["hp"]] or 0) <= 0:
                    continue
                x, y = robot[ROW["x"]], robot[ROW["y"]]
                if x is None or y is None:
                    continue
                x = max(0.0, min(FIELD_WIDTH, float(x)))
                y = max(0.0, min(FIELD_HEIGHT, float(y)))
                grid_x = min(GRID_WIDTH - 1, int(x / CELL_SIZE_METRES))
                grid_y = min(GRID_HEIGHT - 1, int(y / CELL_SIZE_METRES))
                school = side_school[robot[ROW["side"]]]
                side = "red" if robot[ROW["side"]] == "红" else "blue"
                grid_index = grid_y * GRID_WIDTH + grid_x
                grids[school][side][grid_index] += 1
                window_grids[school][window][side][grid_index] += 1
                samples[school] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_count = max_window + 1
    schools = []
    for index, school in enumerate(sorted(school_region)):
        filename = f"{index:03d}.json"
        windows = []
        for window in range(window_count):
            red = window_grids[school][window]["red"]
            blue = window_grids[school][window]["blue"]
            windows.append({
                "start": window * WINDOW_SECONDS,
                "end": (window + 1) * WINDOW_SECONDS,
                "samples": sum(red.values()) + sum(blue.values()),
                "red": sparse_grid(red),
                "blue": sparse_grid(blue),
            })
        payload = {
            "school": school,
            "region": school_region[school],
            "games": games[school],
            "samples": samples[school],
            "red": sparse_grid(grids[school]["red"]),
            "blue": sparse_grid(grids[school]["blue"]),
            "windows": windows,
        }
        (args.output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        schools.append({
            "school": school,
            "region": school_region[school],
            "file": filename,
            "games": games[school],
            "samples": samples[school],
        })

    config = {
        "schema_version": 1,
        "kind": "rmuc_2026_team_position_heatmaps",
        "grid_width": GRID_WIDTH,
        "grid_height": GRID_HEIGHT,
        "cell_size_metres": CELL_SIZE_METRES,
        "gaussian_sigma_metres": GAUSSIAN_SIGMA_METRES,
        "window_seconds": WINDOW_SECONDS,
        "window_count": window_count,
        "x_range": [0, FIELD_WIDTH],
        "y_range": [0, FIELD_HEIGHT],
        "regions": catalog["regions"],
        "schools": schools,
        "source": "613局区域赛1Hz存活机器人位置；0.1m等距学校级红蓝方密度场；按比赛时间聚合为15秒切片；显示端以σ=0.22m二维高斯核表达单点位置误差；基地和前哨站不计入",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"wrote {len(schools)} team heatmaps from {sum(games.values()) // 2} games "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
