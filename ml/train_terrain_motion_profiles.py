#!/usr/bin/env python3
"""Learn stateful terrain-motion priors from labelled referee trajectories.

Official ``飞坡`` events anchor the fly-ramp launch sequence.  Complete
side-to-side B3/R3 trajectory crossings train road-step ascent and descent
angles separately.  Physical gate geometry and pass/fail capability remain
separate hard labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import terrain_crossing_points as terrain  # noqa: E402
from analysis.team_terrain_capabilities import (  # noqa: E402
    TrackPoint,
    build_gates,
    gate_crossings,
    local_coordinates,
)
from analysis.team_style_report import GROUND_TYPES, TEAMS  # noqa: E402


DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
DEFAULT_OUTPUT = ROOT / "analysis" / "outputs" / "team_terrain_motion_profiles.json"
ALIGN_SPEED_MPS = 0.9
STOP_SPEED_MPS = 0.35
MIN_TEAM_ROLE_SAMPLES = 5
FLY_RAMP_CRUISE_MULTIPLIER = 1.12
ROAD_STEP_STRAIGHT_ANGLE_DEG = 20.0


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rounded(value: float, digits: int = 4) -> int | float:
    result = round(float(value), digits)
    return int(result) if result.is_integer() else result


def fly_ramp_event_window(connection: sqlite3.Connection, event: sqlite3.Row) -> dict | None:
    points = list(connection.execute(
        """
        SELECT 时刻秒,x,y
        FROM timeseries
        WHERE game_id=? AND robot_id=? AND 时刻秒 BETWEEN ? AND ?
          AND x IS NOT NULL AND y IS NOT NULL
        ORDER BY 时刻秒
        """,
        (event["game_id"], event["robot_id"], event["时刻秒"] - 7, event["时刻秒"] + 2),
    ))
    by_offset: dict[int, list[float]] = defaultdict(list)
    for left, right in zip(points, points[1:]):
        delta = float(right["时刻秒"]) - float(left["时刻秒"])
        if not 0.5 <= delta <= 1.6:
            continue
        speed = math.hypot(float(right["x"]) - float(left["x"]), float(right["y"]) - float(left["y"])) / delta
        if not 0 <= speed <= 3.5:
            continue
        offset = round(float(right["时刻秒"]) - float(event["时刻秒"]))
        by_offset[offset].append(speed)
    pre = [speed for offset in range(-5, 0) for speed in by_offset.get(offset, ())]
    if not pre or not any(by_offset.get(offset) for offset in (-2, -1, 0)):
        return None
    minimum = min(pre)
    # The referee event is emitted at the ramp crossing.  The final one-second
    # track segment ending at it is therefore a useful, observable
    # approximation of the minimum straight launch run.  Keep this quantity
    # separate from the hard centre-line constraint used by the router.
    runup_samples = by_offset.get(0) or by_offset.get(-1) or []
    runup_distance = median(runup_samples) if runup_samples else 0
    return {
        "minimum_pre_speed": minimum,
        "slow_seconds": sum(speed < ALIGN_SPEED_MPS for speed in pre),
        "aligned": minimum < ALIGN_SPEED_MPS,
        "stopped": minimum < STOP_SPEED_MPS,
        "runup_distance_m": runup_distance,
        "speed_by_offset": {offset: median(values) for offset, values in by_offset.items() if values},
    }


def motion_profile(windows: list[dict], fallback: dict | None = None) -> dict:
    if len(windows) < MIN_TEAM_ROLE_SAMPLES and fallback is not None:
        return {**fallback, "samples": len(windows), "source_scope": "global_fallback"}
    launch_values = [
        window["speed_by_offset"][0]
        for window in windows if 0 in window["speed_by_offset"]
    ]
    launch_speed = median(launch_values) if launch_values else 2.0
    acceleration = []
    previous = 0.1
    for offset in (-2, -1, 0):
        values = [
            window["speed_by_offset"][offset]
            for window in windows if offset in window["speed_by_offset"]
        ]
        observed = median(values) if values else launch_speed
        multiplier = clamp(observed / max(0.1, launch_speed) * FLY_RAMP_CRUISE_MULTIPLIER, 0.1, 1.25)
        previous = max(previous, multiplier)
        acceleration.append(rounded(previous, 3))
    aligned = [window for window in windows if window["aligned"]]
    align_seconds = median([window["slow_seconds"] for window in aligned]) if aligned else 1
    minimum_speed = median([window["minimum_pre_speed"] for window in aligned]) if aligned else 0.25
    runup_values = [window["runup_distance_m"] for window in windows if window["runup_distance_m"] > 0]
    learned_runup = median(runup_values) if runup_values else 1.4
    return {
        "samples": len(windows),
        "source_scope": "team_role" if fallback is not None else "global",
        "alignment_probability": rounded(sum(window["aligned"] for window in windows) / max(1, len(windows))),
        "full_stop_probability": rounded(sum(window["stopped"] for window in windows) / max(1, len(windows))),
        "alignment_seconds": int(clamp(round(align_seconds), 1, 3)),
        "alignment_multiplier": rounded(clamp(minimum_speed / max(0.1, launch_speed), 0.05, 0.55), 3),
        "straight_runup_m": rounded(clamp(learned_runup, 1.0, 2.8), 3),
        "centerline_required": True,
        "acceleration_multipliers": acceleration,
        "cruise_multiplier": FLY_RAMP_CRUISE_MULTIPLIER,
        "observed_launch_speed_mps": rounded(launch_speed, 3),
    }


def road_step_crossing_window(points: list[TrackPoint], gate, crossing: dict) -> dict | None:
    end = min(points, key=lambda point: abs(point.second - crossing["second"]))
    origin_side = int(crossing["direction"].split("->")[0])
    side_threshold = gate.along_span * 0.32
    candidates = []
    for point in points:
        if not crossing["second"] - gate.max_seconds <= point.second < crossing["second"]:
            continue
        along, lateral = local_coordinates(point, gate)
        side = (
            -1 if along <= gate.along_center - side_threshold
            else 1 if along >= gate.along_center + side_threshold
            else 0
        )
        if (
            side == origin_side
            and gate.lateral_min - 0.28 <= lateral <= gate.lateral_max + 0.28
        ):
            candidates.append(point)
    if not candidates:
        return None
    start = max(candidates, key=lambda point: point.second)
    lateral = abs(end.x - start.x)
    longitudinal = abs(end.y - start.y)
    if longitudinal < 0.4:
        return None
    deviation = math.degrees(math.atan2(lateral, longitudinal))
    up_direction = "-1->+1" if gate.side == "blue" else "+1->-1"
    return {
        "lateral_drift_m": lateral,
        "longitudinal_travel_m": longitudinal,
        "deviation_deg": deviation,
        "straight": deviation <= ROAD_STEP_STRAIGHT_ANGLE_DEG,
        "direction": "up" if crossing["direction"] == up_direction else "down",
    }


def trajectory_road_step_windows(
    connection: sqlite3.Connection,
) -> tuple[dict[tuple[str, str], list[dict]], list[dict]]:
    gates = [gate for gate in build_gates(terrain.build_features()) if gate.ability == "road_step"]
    schools = [entry.school for entry in TEAMS]
    placeholders = ",".join("?" for _ in schools)
    gate_clauses = []
    parameters: list = list(schools)
    for gate in gates:
        gate_clauses.append("(y BETWEEN ? AND ? AND x BETWEEN ? AND ?)")
        parameters.extend([
            gate.along_min - 0.65, gate.along_max + 0.65,
            gate.lateral_min - 0.28, gate.lateral_max + 0.28,
        ])
    cursor = connection.execute(
        f"""
        SELECT game_id,robot_id,学校名,机器人类型,时刻秒,x,y,z,当前血量
        FROM timeseries
        WHERE 学校名 IN ({placeholders})
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
          AND x IS NOT NULL AND y IS NOT NULL
          AND ({" OR ".join(gate_clauses)})
        ORDER BY game_id,robot_id,时刻秒
        """,
        parameters,
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    all_windows: list[dict] = []
    current_key: tuple[int, int] | None = None
    current_school = ""
    current_role = ""
    points: list[TrackPoint] = []

    def flush() -> None:
        nonlocal points
        if not points:
            return
        for gate in gates:
            for crossing in gate_crossings(points, gate):
                window = road_step_crossing_window(points, gate, crossing)
                if not window:
                    continue
                grouped[(current_school, current_role)].append(window)
                all_windows.append(window)
        points = []

    for row in cursor:
        key = (int(row["game_id"]), int(row["robot_id"]))
        if key != current_key:
            flush()
            current_key = key
            current_school = str(row["学校名"])
            current_role = str(row["机器人类型"])
        points.append(TrackPoint(
            second=float(row["时刻秒"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=None if row["z"] is None else float(row["z"]),
            hp=None if row["当前血量"] is None else float(row["当前血量"]),
        ))
    flush()
    return grouped, all_windows


def road_step_profile(windows: list[dict], fallback: dict | None = None) -> dict:
    if len(windows) < MIN_TEAM_ROLE_SAMPLES and fallback is not None:
        return {**fallback, "samples": len(windows), "source_scope": "global_fallback"}
    return {
        "samples": len(windows),
        "source_scope": "team_role" if fallback is not None else "global",
        "straight_crossing_probability": rounded(
            sum(window["straight"] for window in windows) / max(1, len(windows)),
        ),
        "straight_angle_threshold_deg": ROAD_STEP_STRAIGHT_ANGLE_DEG,
        "median_deviation_deg": rounded(median([window["deviation_deg"] for window in windows]), 2) if windows else 0,
        "median_lateral_drift_m": rounded(median([window["lateral_drift_m"] for window in windows]), 3) if windows else 0,
        "route_alignment_enabled": (
            sum(window["straight"] for window in windows) / max(1, len(windows)) >= 0.5
        ),
    }


def directional_road_step_profile(windows: list[dict], fallback: dict | None = None) -> dict:
    result = road_step_profile(windows, fallback)
    result["directions"] = {
        direction: road_step_profile(
            [window for window in windows if window["direction"] == direction],
            fallback["directions"][direction] if fallback else None,
        )
        for direction in ("up", "down")
    }
    return result


def main() -> None:
    options = args()
    connection = sqlite3.connect(f"file:{options.db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    global_windows: list[dict] = []
    rows = connection.execute(
        """
        SELECT DISTINCT game_id,robot_id,时刻秒,学校名,机器人类型
        FROM events
        WHERE 类别='飞坡' AND robot_id IS NOT NULL
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
        ORDER BY game_id,robot_id,时刻秒
        """
    )
    for event in rows:
        window = fly_ramp_event_window(connection, event)
        if not window:
            continue
        grouped[(event["学校名"], event["机器人类型"])].append(window)
        global_windows.append(window)
    step_grouped, global_step_windows = trajectory_road_step_windows(connection)
    connection.close()

    global_profile = motion_profile(global_windows)
    global_step_profile = directional_road_step_profile(global_step_windows)
    teams = {}
    for entry in TEAMS:
        teams[entry.school] = {
            role: {
                "fly_ramp": motion_profile(grouped[(entry.school, role)], global_profile),
                "road_step": directional_road_step_profile(
                    step_grouped[(entry.school, role)], global_step_profile,
                ),
            }
            for role in GROUND_TYPES
        }
    payload = {
        "schema_version": 3,
        "kind": "terrain_motion_priors",
        "method": {
            "label": "裁判事件类别=飞坡",
            "window_seconds": [-5, 0],
            "alignment_speed_threshold_mps": ALIGN_SPEED_MPS,
            "full_stop_speed_threshold_mps": STOP_SPEED_MPS,
            "minimum_team_role_samples": MIN_TEAM_ROLE_SAMPLES,
            "road_step_label": "B3/R3 轨迹完整从一侧穿越到另一侧",
            "road_step_directions": ["up", "down"],
            "road_step_straight_angle_threshold_deg": ROAD_STEP_STRAIGHT_ANGLE_DEG,
            "fly_ramp_runup_label": "终点为飞坡事件时刻的最后一个 1 Hz 位移段距离",
            "note": "几何边界与通行资格不在此模型学习；此文件学习飞坡直线助跑/对位/停顿/加速，以及二级台阶跨越的横向偏移。",
        },
        "global": {"fly_ramp": global_profile, "road_step": global_step_profile},
        "teams": teams,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {len(global_windows)} fly-ramp and {len(global_step_windows)} "
        f"road-step windows to {options.output}"
    )


if __name__ == "__main__":
    main()
