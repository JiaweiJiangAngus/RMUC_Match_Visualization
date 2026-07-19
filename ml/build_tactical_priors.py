#!/usr/bin/env python3
"""Build team/role/phase destination priors for operator inference."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .tactical_inference import GROUND_TYPES, TacticalMap
except ImportError:
    from tactical_inference import GROUND_TYPES, TacticalMap

try:
    from analysis.team_style_report import TEAMS
except ModuleNotFoundError:
    import sys

    ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
    from analysis.team_style_report import TEAMS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "team_tactical_priors.json"
HORIZONS = (5, 10, 15)
GRID_METRES = 0.25
MIN_SAMPLES = 5


def time_phase(second: float) -> str:
    if second < 60:
        return "opening"
    if second < 330:
        return "middle"
    return "endgame"


class QuantizedZones:
    """Cache semantic zones on a 0.25 m grid to keep the DB pass cheap."""

    def __init__(self, tactical_map: TacticalMap) -> None:
        self.map = tactical_map
        self.cache: dict[tuple[int, int, str], str] = {}

    def get(self, x: float, y: float, side: str) -> str:
        key = (round(x / GRID_METRES), round(y / GRID_METRES), side)
        if key not in self.cache:
            qx = min(28.0, max(0.0, key[0] * GRID_METRES))
            qy = min(15.0, max(0.0, key[1] * GRID_METRES))
            self.cache[key] = self.map.zone(qx, qy, side)
        return self.cache[key]


def build(db_path: Path) -> dict:
    schools = tuple(entry.school for entry in TEAMS)
    placeholders = ",".join("?" for _ in schools)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = connection.execute(
        f"""
        SELECT game_id,robot_id,时刻秒,学校名,机器人类型,阵营,x,y,当前血量
        FROM timeseries
        WHERE 学校名 IN ({placeholders})
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
          AND x BETWEEN 0 AND 28 AND y BETWEEN 0 AND 15
        ORDER BY game_id,robot_id,时刻秒
        """,
        schools,
    )
    zones = QuantizedZones(TacticalMap())
    counts: dict[tuple[str, str, str, str, int], Counter] = defaultdict(Counter)
    current_key = None
    current_meta = None
    points: list[tuple[int, float, float, float]] = []
    sampled_tracks = 0
    sampled_states = 0

    def flush() -> None:
        nonlocal sampled_tracks, sampled_states, points
        if current_meta is None or not points:
            points = []
            return
        school, role, side = current_meta
        by_second = {second: (x, y, hp) for second, x, y, hp in points}
        sampled_tracks += 1
        for second, x, y, hp in points:
            if second % 2 or hp <= 0:
                continue
            current_zone = zones.get(x, y, side)
            phase = time_phase(second)
            used = False
            for horizon in HORIZONS:
                future = by_second.get(second + horizon)
                if future is None or future[2] <= 0:
                    continue
                destination = zones.get(future[0], future[1], side)
                counts[(school, role, current_zone, phase, horizon)][destination] += 1
                used = True
            sampled_states += int(used)
        points = []

    for game_id, robot_id, second, school, role, side, x, y, hp in cursor:
        key = (int(game_id), int(robot_id))
        if key != current_key:
            flush()
            current_key = key
            current_meta = (str(school), str(role), str(side))
        points.append((round(float(second)), float(x), float(y), float(hp or 0)))
    flush()
    connection.close()

    records = []
    for (school, role, current_zone, phase, horizon), destinations in sorted(counts.items()):
        total = sum(destinations.values())
        if total < MIN_SAMPLES:
            continue
        records.append(
            {
                "school": school,
                "role": role,
                "current_zone": current_zone,
                "phase": phase,
                "horizon": horizon,
                "samples": total,
                "destinations": dict(destinations.most_common()),
            }
        )
    return {
        "schema_version": 1,
        "description": "team/role/current-zone/time-phase destination frequency prior",
        "horizons": HORIZONS,
        "grid_metres": GRID_METRES,
        "minimum_samples": MIN_SAMPLES,
        "schools": len(schools),
        "roles": len(GROUND_TYPES),
        "sampled_tracks": sampled_tracks,
        "sampled_states": sampled_states,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args.db)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sampled_tracks": payload["sampled_tracks"],
                "sampled_states": payload["sampled_states"],
                "records": len(payload["records"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
