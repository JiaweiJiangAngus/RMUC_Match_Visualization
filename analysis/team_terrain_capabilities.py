#!/usr/bin/env python3
"""Infer terrain capabilities for every ground role of the 44 advancing teams.

Official gain events are treated as direct observations. Terrain types without
an event label use continuous trajectory crossings and therefore receive only
probabilistic evidence levels. "Not observed" is explicitly not a negative
capability label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

try:
    from team_style_report import GROUND_TYPES, TEAMS
    import terrain_crossing_points as terrain
except ModuleNotFoundError:
    from analysis.team_style_report import GROUND_TYPES, TEAMS
    from analysis import terrain_crossing_points as terrain


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
OUTPUT_DIR = ROOT / "analysis" / "outputs"
DEFAULT_MANUAL_LABELS = ROOT / "analysis" / "manual_terrain_capabilities.csv"

ABILITY_ORDER = (
    "central_highland_step",
    "road_step",
    "fly_ramp",
    "rough_road",
    "road_tunnel",
    "highland_tunnel",
    "slope_43",
    "trapezoid_highland_step",
    "central_highland_400mm_jump",
)
ABILITY_ZH = {
    "central_highland_step": "中央高地台阶",
    "road_step": "公路台阶",
    "fly_ramp": "飞坡",
    "rough_road": "起伏路段",
    "road_tunnel": "公路隧道",
    "highland_tunnel": "高地隧道",
    "slope_43": "43°坡",
    "trapezoid_highland_step": "梯形高地台阶",
    "central_highland_400mm_jump": "400 mm 高差跳跃",
}
ABILITY_SHORT = {
    "central_highland_step": "高地台阶",
    "road_step": "公路台阶",
    "fly_ramp": "飞坡",
    "rough_road": "起伏路",
    "road_tunnel": "公路隧道",
    "highland_tunnel": "高地隧道",
    "slope_43": "43°坡",
    "trapezoid_highland_step": "梯高台阶",
    "central_highland_400mm_jump": "400mm跳",
}
OFFICIAL_EVENT_TO_ABILITY = {
    "过中央高地": "central_highland_step",
    "台阶跨越": "road_step",
    "飞坡": "fly_ramp",
}
GATE_CATEGORY_TO_ABILITY = {
    "central_highland_step": "central_highland_step",
    "road_step": "road_step",
    "fly_ramp": "fly_ramp",
    "rough_road": "rough_road",
    "road_tunnel": "road_tunnel",
    "highland_tunnel": "highland_tunnel",
    "slope_43": "slope_43",
    "trapezoid_highland_step": "trapezoid_highland_step",
}
HORIZONTAL_AXIS = {
    "central_highland_step", "fly_ramp", "rough_road", "highland_tunnel",
}
MAX_CROSSING_SECONDS = {
    "central_highland_step": 8.0,
    "road_step": 7.0,
    "fly_ramp": 12.0,
    "rough_road": 12.0,
    "road_tunnel": 8.0,
    "highland_tunnel": 10.0,
    "slope_43": 12.0,
    "trapezoid_highland_step": 10.0,
}
STATUS_ORDER = ("人工确认", "已证实", "较强迹象", "可能具备", "弱迹象", "未观察到", "无样本")


@dataclass(frozen=True)
class TrackPoint:
    second: float
    x: float
    y: float
    z: float | None
    hp: float | None


@dataclass(frozen=True)
class Gate:
    ability: str
    feature_id: str
    side: str
    axis: str
    along_min: float
    along_max: float
    lateral_min: float
    lateral_max: float
    max_seconds: float

    @property
    def along_center(self) -> float:
        return (self.along_min + self.along_max) / 2

    @property
    def along_span(self) -> float:
        return self.along_max - self.along_min


@dataclass
class Evidence:
    manual_confirmations: list[dict] = field(default_factory=list)
    official_events: int = 0
    official_games: set[int] = field(default_factory=set)
    trajectory_crossings: int = 0
    trajectory_games: set[int] = field(default_factory=set)
    trajectory_direction_counts: Counter = field(default_factory=Counter)
    trajectory_direction_games: dict[str, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    examples: list[dict] = field(default_factory=list)

    def add_manual(self, source: str, note: str) -> None:
        item = {"source": source, "note": note}
        self.manual_confirmations.append(item)
        if len(self.examples) < 4:
            self.examples.append({"source": "manual_confirmation", **item})

    def add_official(self, game_id: int, second: float) -> None:
        self.official_events += 1
        self.official_games.add(game_id)
        if len(self.examples) < 4:
            self.examples.append({"source": "official_event", "game_id": game_id, "second": second})

    def add_trajectory(
        self,
        game_id: int,
        second: float,
        feature_id: str,
        detail: dict | None = None,
    ) -> None:
        self.trajectory_crossings += 1
        self.trajectory_games.add(game_id)
        traversal = str((detail or {}).get("traversal", ""))
        if traversal:
            self.trajectory_direction_counts[traversal] += 1
            self.trajectory_direction_games[traversal].add(game_id)
        if len(self.examples) < 4:
            example = {
                "source": "trajectory",
                "game_id": game_id,
                "second": round(second, 3),
                "feature_id": feature_id,
            }
            if detail:
                example.update(detail)
            self.examples.append(example)


@dataclass
class RoleSample:
    games: set[int] = field(default_factory=set)
    alive_seconds: int = 0
    valid_points: int = 0


def build_gates(features: Sequence[terrain.Feature]) -> tuple[Gate, ...]:
    gates = []
    for feature in features:
        if feature.kind != "crossing_gate" or feature.category not in GATE_CATEGORY_TO_ABILITY:
            continue
        points = [terrain.map_to_field(x, y) for x, y in feature.map_geometry_px]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        axis = "x" if feature.category in HORIZONTAL_AXIS else "y"
        if axis == "x":
            along_min, along_max = min(xs), max(xs)
            lateral_min, lateral_max = min(ys), max(ys)
        else:
            along_min, along_max = min(ys), max(ys)
            lateral_min, lateral_max = min(xs), max(xs)
        gates.append(
            Gate(
                ability=GATE_CATEGORY_TO_ABILITY[feature.category],
                feature_id=feature.feature_id,
                side=feature.side,
                axis=axis,
                along_min=along_min,
                along_max=along_max,
                lateral_min=lateral_min,
                lateral_max=lateral_max,
                max_seconds=MAX_CROSSING_SECONDS[feature.category],
            )
        )
    return tuple(gates)


def local_coordinates(point: TrackPoint, gate: Gate) -> tuple[float, float]:
    return (point.x, point.y) if gate.axis == "x" else (point.y, point.x)


def gate_crossings(points: Sequence[TrackPoint], gate: Gate) -> list[dict]:
    """Detect complete side-to-side passages through an axis-aligned gate."""
    result: list[dict] = []
    origin_side: int | None = None
    origin_time = 0.0
    entered = False
    previous: TrackPoint | None = None
    last_record = -1e9
    lateral_margin = 0.28
    approach_margin = 0.65
    side_threshold = gate.along_span * 0.32

    def reset() -> None:
        nonlocal origin_side, origin_time, entered
        origin_side = None
        origin_time = 0.0
        entered = False

    for point in points:
        if point.hp is not None and point.hp <= 0:
            reset()
            previous = point
            continue
        if previous is not None:
            time_gap = point.second - previous.second
            distance = math.hypot(point.x - previous.x, point.y - previous.y)
            if time_gap <= 0 or time_gap > 1.6 or distance > 5.0:
                reset()

        along, lateral = local_coordinates(point, gate)
        lateral_ok = gate.lateral_min - lateral_margin <= lateral <= gate.lateral_max + lateral_margin
        approach_ok = gate.along_min - approach_margin <= along <= gate.along_max + approach_margin
        if not lateral_ok or not approach_ok:
            reset()
            previous = point
            continue

        side = 0
        if along <= gate.along_center - side_threshold:
            side = -1
        elif along >= gate.along_center + side_threshold:
            side = 1
        if gate.along_min <= along <= gate.along_max:
            entered = True

        if previous is not None:
            previous_along, previous_lateral = local_coordinates(previous, gate)
            previous_lateral_ok = (
                gate.lateral_min - lateral_margin
                <= previous_lateral
                <= gate.lateral_max + lateral_margin
            )
            if previous_lateral_ok and (
                (previous_along <= gate.along_min and along >= gate.along_max)
                or (previous_along >= gate.along_max and along <= gate.along_min)
            ):
                entered = True

        if origin_side is None:
            if side:
                origin_side = side
                origin_time = point.second
        elif point.second - origin_time > gate.max_seconds:
            reset()
            if side:
                origin_side = side
                origin_time = point.second
        elif side == -origin_side and entered:
            if point.second - last_record >= 3.0:
                result.append(
                    {
                        "second": point.second,
                        "feature_id": gate.feature_id,
                        "direction": f"{origin_side:+d}->{side:+d}",
                    }
                )
                last_record = point.second
            origin_side = side
            origin_time = point.second
            entered = False
        previous = point
    return result


def central_highland_jump_ascents(
    points: Sequence[TrackPoint],
    features: Sequence[terrain.Feature],
) -> list[dict]:
    """Detect stable low-to-high ascents across a non-entrance 400 mm ledge.

    A single polygon-side flip is not enough: UWB positions can oscillate around
    the edge, especially beside the highland tunnel.  Require two consecutive
    low-side points, two consecutive high-side points, a meaningful movement
    across the edge, and a persistent median Z increase.  The three designed
    entrances are excluded geometrically.
    """
    central_region = next(
        feature for feature in features if feature.feature_id == "central_highland_region"
    )
    ledges = [feature for feature in features if feature.kind == "conditional_ledge"]
    designed_entrances = [
        feature
        for feature in features
        if feature.kind == "crossing_gate"
        and feature.category in {
            "central_highland_step",
            "road_tunnel",
            "highland_tunnel",
        }
    ]
    region_polygon = central_region.map_geometry_px
    result: list[dict] = []
    last_record = -1e9
    tunnel_exclusion_radius_m = {
        "highland_tunnel": 2.8,
        "road_tunnel": 2.2,
    }

    def inside(point: TrackPoint) -> bool:
        return terrain.point_in_polygon(*terrain.field_to_map(point.x, point.y), region_polygon)

    for index in range(2, len(points) - 1):
        previous, current = points[index - 1], points[index]
        time_gap = current.second - previous.second
        if not 0.5 <= time_gap <= 1.6:
            continue
        if (previous.hp is not None and previous.hp <= 0) or (current.hp is not None and current.hp <= 0):
            continue
        distance = math.hypot(current.x - previous.x, current.y - previous.y)
        # Sub-0.55 m flips are dominated by edge jitter in stationary sentries;
        # jumps observed at 1 Hz move meaningfully across the interface.
        if not 0.55 <= distance <= 2.8:
            continue
        before = points[index - 2:index]
        after = points[index:index + 2]
        local_window = (*before, *after)
        if any(
            not 0.5 <= right.second - left.second <= 1.6
            or math.hypot(right.x - left.x, right.y - left.y) > 2.8
            for left, right in zip(local_window, local_window[1:])
        ):
            continue
        if not all(not inside(point) for point in before):
            continue
        if not all(inside(point) for point in after):
            continue
        if any(point.z is None for point in (*before, *after)):
            continue
        z_values = [float(point.z) for point in (*before, *after) if point.z is not None]
        if any(not -2.0 <= value <= 3.0 for value in z_values):
            continue
        before_z = median(float(point.z) for point in before if point.z is not None)
        after_z = median(float(point.z) for point in after if point.z is not None)
        height_gain = after_z - before_z
        if not 0.20 <= height_gain <= 1.20:
            continue

        start_px = terrain.field_to_map(previous.x, previous.y)
        end_px = terrain.field_to_map(current.x, current.y)
        crossed = next(
            (ledge for ledge in ledges if terrain.segment_hits_feature(start_px, end_px, ledge)),
            None,
        )
        if crossed is None:
            continue
        if any(
            terrain.segment_hits_feature(start_px, end_px, entrance)
            for entrance in designed_entrances
        ):
            continue
        midpoint = ((previous.x + current.x) / 2, (previous.y + current.y) / 2)
        if any(
            entrance.category in tunnel_exclusion_radius_m
            and math.hypot(
                midpoint[0] - terrain.map_to_field(*entrance.center_map_px)[0],
                midpoint[1] - terrain.map_to_field(*entrance.center_map_px)[1],
            )
            <= tunnel_exclusion_radius_m[entrance.category]
            for entrance in designed_entrances
        ):
            continue

        persistence = points[index:min(len(points), index + 4)]
        inside_points = [point for point in persistence if inside(point) and point.z is not None]
        if len(inside_points) < 3:
            continue
        if median(float(point.z) for point in inside_points) < before_z + 0.18:
            continue
        if current.second - last_record < 5.0:
            continue
        result.append(
            {
                "second": current.second,
                "feature_id": crossed.feature_id,
                "height_gain_m": round(height_gain, 3),
                "distance_m": round(distance, 3),
                "before_z_median": round(before_z, 3),
                "after_z_median": round(after_z, 3),
            }
        )
        last_record = current.second
    return result


def evidence_status(evidence: Evidence, sample_games: int) -> tuple[str, float, str]:
    if evidence.manual_confirmations:
        return "人工确认", 1.0, "positive_confirmed"
    if evidence.official_events:
        confidence = min(0.99, 0.93 + 0.015 * min(len(evidence.official_games), 4))
        return "已证实", round(confidence, 3), "positive_observed"
    crossings = evidence.trajectory_crossings
    games = len(evidence.trajectory_games)
    if crossings >= 3 and games >= 2:
        return "较强迹象", 0.80, "positive_probable"
    if crossings >= 2:
        return "可能具备", 0.65, "positive_probable"
    if crossings == 1:
        return "弱迹象", 0.40, "weak_unconfirmed"
    if sample_games:
        return "未观察到", 0.10, "unlabeled_not_negative"
    return "无样本", 0.0, "no_sample"


def load_manual_confirmations(
    path: Path,
    evidence: dict[tuple[str, str, str], Evidence],
) -> int:
    if not path.exists():
        return 0
    valid_schools = {entry.school for entry in TEAMS}
    count = 0
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            school = row["school"].strip()
            role = row["role"].strip()
            ability = row["ability"].strip()
            label = row["label"].strip()
            key = (school, role, ability)
            if school not in valid_schools:
                raise ValueError(f"unknown school in manual labels: {school}")
            if role not in GROUND_TYPES:
                raise ValueError(f"unknown role in manual labels: {role}")
            if ability not in ABILITY_ORDER:
                raise ValueError(f"unknown ability in manual labels: {ability}")
            if label != "confirmed":
                raise ValueError(f"unsupported manual label {label!r}; expected 'confirmed'")
            if key in seen:
                raise ValueError(f"duplicate manual label: {key}")
            seen.add(key)
            evidence[key].add_manual(row.get("source", "user"), row.get("note", ""))
            count += 1
    return count


def load_official_events(
    connection: sqlite3.Connection,
    schools: Sequence[str],
    evidence: dict[tuple[str, str, str], Evidence],
) -> dict[tuple[int, int, str], list[float]]:
    placeholders = ",".join("?" for _ in schools)
    rows = connection.execute(
        f"""
        SELECT game_id,robot_id,时刻秒,学校名,机器人类型,类别
        FROM events
        WHERE 学校名 IN ({placeholders})
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
          AND 类别 IN ('过中央高地','台阶跨越','飞坡')
        """,
        schools,
    )
    official_by_robot: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for game_id, robot_id, second, school, role, category in rows:
        ability = OFFICIAL_EVENT_TO_ABILITY[category]
        evidence[(school, role, ability)].add_official(int(game_id), float(second))
        official_by_robot[(int(game_id), int(robot_id), ability)].append(float(second))
    return official_by_robot


def analyze_tracks(
    connection: sqlite3.Connection,
    schools: Sequence[str],
    features: Sequence[terrain.Feature],
    evidence: dict[tuple[str, str, str], Evidence],
    samples: dict[tuple[str, str], RoleSample],
) -> tuple[dict, Counter]:
    gates = build_gates(features)
    placeholders = ",".join("?" for _ in schools)
    cursor = connection.execute(
        f"""
        SELECT game_id,robot_id,时刻秒,学校名,机器人类型,x,y,z,当前血量
        FROM timeseries
        WHERE 学校名 IN ({placeholders})
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
          AND x IS NOT NULL AND y IS NOT NULL
        ORDER BY game_id,robot_id,时刻秒
        """,
        schools,
    )

    detector_counts: Counter = Counter()
    track_crossings_by_robot: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    current_key: tuple[int, int] | None = None
    current_school = ""
    current_role = ""
    points: list[TrackPoint] = []

    def flush() -> None:
        nonlocal points
        if current_key is None or not points:
            points = []
            return
        game_id, robot_id = current_key
        valid = [
            point
            for point in points
            if 0 <= point.x <= 28 and 0 <= point.y <= 15
        ]
        if not valid:
            points = []
            return
        sample = samples[(current_school, current_role)]
        sample.games.add(game_id)
        sample.valid_points += len(valid)
        sample.alive_seconds += sum(point.hp is None or point.hp > 0 for point in valid)

        for gate in gates:
            for crossing in gate_crossings(valid, gate):
                detail = {"direction": crossing["direction"]}
                if gate.ability == "fly_ramp":
                    forward = "+1->-1" if gate.side == "blue" else "-1->+1"
                    detail["traversal"] = (
                        "forward" if crossing["direction"] == forward else "reverse"
                    )
                ability_evidence = evidence[(current_school, current_role, gate.ability)]
                ability_evidence.add_trajectory(
                    game_id,
                    crossing["second"],
                    gate.feature_id,
                    detail,
                )
                track_crossings_by_robot[(game_id, robot_id, gate.ability)].append(crossing["second"])
                detector_counts[gate.ability] += 1

        for ascent in central_highland_jump_ascents(valid, features):
            ability = "central_highland_400mm_jump"
            evidence[(current_school, current_role, ability)].add_trajectory(
                game_id,
                ascent["second"],
                ascent["feature_id"],
                {
                    "height_gain_m": ascent["height_gain_m"],
                    "distance_m": ascent["distance_m"],
                    "before_z_median": ascent["before_z_median"],
                    "after_z_median": ascent["after_z_median"],
                },
            )
            track_crossings_by_robot[(game_id, robot_id, ability)].append(ascent["second"])
            detector_counts[ability] += 1
        points = []

    for row in cursor:
        game_id, robot_id, second, school, role, x, y, z, hp = row
        key = (int(game_id), int(robot_id))
        if key != current_key:
            flush()
            current_key = key
            current_school = str(school)
            current_role = str(role)
        points.append(
            TrackPoint(
                second=float(second),
                x=float(x),
                y=float(y),
                z=None if z is None else float(z),
                hp=None if hp is None else float(hp),
            )
        )
    flush()
    return track_crossings_by_robot, detector_counts


def official_detector_validation(
    official: dict[tuple[int, int, str], list[float]],
    detected: dict[tuple[int, int, str], list[float]],
    tolerance_seconds: float = 15.0,
) -> dict:
    by_ability = defaultdict(lambda: {"events": 0, "matched": 0})
    for key, seconds in official.items():
        ability = key[2]
        candidates = detected.get(key, ())
        for second in seconds:
            by_ability[ability]["events"] += 1
            if any(abs(second - candidate) <= tolerance_seconds for candidate in candidates):
                by_ability[ability]["matched"] += 1
    return {
        ability: {
            **values,
            "match_rate": round(values["matched"] / values["events"], 4) if values["events"] else 0.0,
            "tolerance_seconds": tolerance_seconds,
        }
        for ability, values in by_ability.items()
    }


def build_rows(
    evidence: dict[tuple[str, str, str], Evidence],
    samples: dict[tuple[str, str], RoleSample],
) -> tuple[list[dict], list[dict], dict]:
    wide_rows = []
    long_rows = []
    summary_counts = {ability: Counter() for ability in ABILITY_ORDER}
    for entry in TEAMS:
        for role in GROUND_TYPES:
            sample = samples[(entry.school, role)]
            wide = {
                "stage": entry.stage,
                "region": entry.region,
                "school": entry.school,
                "team": entry.team,
                "role": role,
                "sample_games": len(sample.games),
                "alive_seconds": sample.alive_seconds,
                "valid_points": sample.valid_points,
            }
            for ability in ABILITY_ORDER:
                item = evidence[(entry.school, role, ability)]
                status, confidence, training_label = evidence_status(item, len(sample.games))
                summary_counts[ability][status] += 1
                prefix = ability
                wide[f"{prefix}_status"] = status
                wide[f"{prefix}_confidence"] = confidence
                wide[f"{prefix}_manual_confirmed"] = bool(item.manual_confirmations)
                wide[f"{prefix}_official_events"] = item.official_events
                wide[f"{prefix}_official_games"] = len(item.official_games)
                wide[f"{prefix}_trajectory_crossings"] = item.trajectory_crossings
                wide[f"{prefix}_trajectory_games"] = len(item.trajectory_games)
                if ability == "fly_ramp":
                    wide[f"{prefix}_forward_crossings"] = item.trajectory_direction_counts["forward"]
                    wide[f"{prefix}_forward_games"] = len(item.trajectory_direction_games["forward"])
                    wide[f"{prefix}_reverse_crossings"] = item.trajectory_direction_counts["reverse"]
                    wide[f"{prefix}_reverse_games"] = len(item.trajectory_direction_games["reverse"])
                long_rows.append(
                    {
                        "stage": entry.stage,
                        "region": entry.region,
                        "school": entry.school,
                        "team": entry.team,
                        "role": role,
                        "sample_games": len(sample.games),
                        "alive_seconds": sample.alive_seconds,
                        "ability": ability,
                        "ability_zh": ABILITY_ZH[ability],
                        "status": status,
                        "confidence": confidence,
                        "training_label": training_label,
                        "manual_confirmed": bool(item.manual_confirmations),
                        "manual_confirmations": item.manual_confirmations,
                        "official_events": item.official_events,
                        "official_games": len(item.official_games),
                        "trajectory_crossings": item.trajectory_crossings,
                        "trajectory_games": len(item.trajectory_games),
                        "trajectory_directions": {
                            direction: {
                                "crossings": count,
                                "games": len(item.trajectory_direction_games[direction]),
                            }
                            for direction, count in item.trajectory_direction_counts.items()
                        },
                        "evidence_examples": item.examples,
                    }
                )
            wide_rows.append(wide)
    summary = {
        "teams": len(TEAMS),
        "ground_roles": len(GROUND_TYPES),
        "team_role_rows": len(wide_rows),
        "ability_rows": len(long_rows),
        "status_counts": {
            ability: {status: summary_counts[ability][status] for status in STATUS_ORDER}
            for ability in ABILITY_ORDER
        },
    }
    return wide_rows, long_rows, summary


def compact_status(row: dict) -> str:
    status = row["status"]
    if status == "人工确认":
        return "人工确认"
    if status == "已证实":
        return f"证{row['official_events']}"
    if status == "较强迹象":
        return f"强{row['trajectory_crossings']}"
    if status == "可能具备":
        return f"可{row['trajectory_crossings']}"
    if status == "弱迹象":
        return "弱1"
    if status == "无样本":
        return "无样本"
    return "—"


def write_outputs(
    wide_rows: list[dict],
    long_rows: list[dict],
    summary: dict,
) -> tuple[Path, ...]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide_csv = OUTPUT_DIR / "team_ground_terrain_capabilities_44x5.csv"
    long_csv = OUTPUT_DIR / "team_ground_terrain_capabilities_long.csv"
    json_path = OUTPUT_DIR / "team_ground_terrain_capabilities.json"
    summary_path = OUTPUT_DIR / "team_ground_terrain_capabilities_summary.json"
    md_path = OUTPUT_DIR / "team_ground_terrain_capabilities.md"

    with wide_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(wide_rows[0]))
        writer.writeheader()
        writer.writerows(wide_rows)
    long_serialized = []
    for row in long_rows:
        item = dict(row)
        item["manual_confirmations"] = json.dumps(item["manual_confirmations"], ensure_ascii=False)
        item["trajectory_directions"] = json.dumps(item["trajectory_directions"], ensure_ascii=False)
        item["evidence_examples"] = json.dumps(item["evidence_examples"], ensure_ascii=False)
        long_serialized.append(item)
    with long_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(long_serialized[0]))
        writer.writeheader()
        writer.writerows(long_serialized)

    teams_payload = []
    by_team_role = {(row["school"], row["role"]): row for row in wide_rows}
    by_long = defaultdict(list)
    for row in long_rows:
        by_long[(row["school"], row["role"])].append(row)
    for entry in TEAMS:
        teams_payload.append(
            {
                "stage": entry.stage,
                "region": entry.region,
                "school": entry.school,
                "team": entry.team,
                "robots": [
                    {
                        "role": role,
                        "sample_games": by_team_role[(entry.school, role)]["sample_games"],
                        "alive_seconds": by_team_role[(entry.school, role)]["alive_seconds"],
                        "capabilities": by_long[(entry.school, role)],
                    }
                    for role in GROUND_TYPES
                ],
            }
        )
    payload = {
        "schema_version": 2,
        "status": "evidence_graded_not_observed_is_not_negative",
        "method": summary["method"],
        "summary": {key: value for key, value in summary.items() if key != "method"},
        "teams": teams_payload,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    long_index = {(row["school"], row["role"], row["ability"]): row for row in long_rows}
    lines = [
        "# 44 支队伍×5 个地面兵种的地形能力证据",
        "",
        "本报告中“未观察到”不等于“不具备”，不应直接当作训练负标签。",
        "",
        "- `证N`：有 N 次官方地形增益事件，记为已证实。",
        "- `人工确认`：用户提供的明确能力标签，优先级高于轨迹推断，但保留原始轨迹证据。",
        "- `强N`：轨迹至少 3 次且横跨至少 2 局。",
        "- `可N`：轨迹至少 2 次，可能具备。",
        "- `弱1`：仅 1 次轨迹迹象；`—`：有样本但未观察到；`无样本`：数据中未出场。",
        "",
        "## 总体证据分布",
        "",
        "| 能力 | 人工确认 | 已证实 | 较强迹象 | 可能具备 | 弱迹象 | 未观察到 | 无样本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ability in ABILITY_ORDER:
        counts = summary["status_counts"][ability]
        lines.append(
            f"| {ABILITY_ZH[ability]} | "
            + " | ".join(str(counts[status]) for status in STATUS_ORDER)
            + " |"
        )
    lines.extend(["", "## 逐队逐兵种", ""])
    header = "| 兵种 | 样本局 | " + " | ".join(ABILITY_SHORT[ability] for ability in ABILITY_ORDER) + " |"
    separator = "| --- | ---: | " + " | ".join("---" for _ in ABILITY_ORDER) + " |"
    for entry in TEAMS:
        lines.extend([f"### {entry.school}（{entry.team}）· {entry.stage}", "", header, separator])
        for role in GROUND_TYPES:
            sample_games = by_team_role[(entry.school, role)]["sample_games"]
            cells = [
                compact_status(long_index[(entry.school, role, ability)])
                for ability in ABILITY_ORDER
            ]
            lines.append(f"| {role} | {sample_games} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "## 400 mm 跳跃口径",
            "",
            "只统计从中央高地外侧低位进入高地、轨迹与非 B5/R5 的 400 mm 边缘相交、Z 高度同步上升且后续仍位于高地的过程。下高地、复活跳点和单点定位抖动不算跳跃能力。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return wide_csv, long_csv, json_path, summary_path, md_path


def analyze(
    db_path: Path,
    manual_labels_path: Path = DEFAULT_MANUAL_LABELS,
) -> tuple[list[dict], list[dict], dict]:
    schools = tuple(entry.school for entry in TEAMS)
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    known = {
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT 学校名 FROM timeseries WHERE 学校名 IS NOT NULL"
        )
    }
    missing = sorted(set(schools) - known)
    if missing:
        raise RuntimeError(f"schools missing from database: {missing}")

    evidence = defaultdict(Evidence)
    samples = defaultdict(RoleSample)
    features = terrain.build_features()
    manual_count = load_manual_confirmations(manual_labels_path, evidence)
    official = load_official_events(connection, schools, evidence)
    detected, detector_counts = analyze_tracks(
        connection, schools, features, evidence, samples,
    )
    validation = official_detector_validation(official, detected)
    connection.close()

    wide_rows, long_rows, summary = build_rows(evidence, samples)
    summary["method"] = {
        "manual_confirmations": {
            "path": str(manual_labels_path),
            "count": manual_count,
            "priority": "higher_than_official_and_trajectory_evidence",
        },
        "official_event_mapping": OFFICIAL_EVENT_TO_ABILITY,
        "trajectory_gate_detector": {
            "complete_side_to_side_passage": True,
            "max_time_seconds": MAX_CROSSING_SECONDS,
            "lateral_margin_m": 0.28,
            "approach_margin_m": 0.65,
            "position_interval_expected_seconds": 1.0,
            "fly_ramp_direction": {
                "blue_forward": "+1->-1 (field right to left)",
                "red_forward": "-1->+1 (field left to right)",
                "reverse_capability_minimum": "at least 2 reverse crossings in at least 2 games",
            },
        },
        "jump_400mm": {
            "direction": "outside_low_to_central_highland_inside_high",
            "minimum_median_z_gain_m": 0.20,
            "xy_step_range_m": [0.55, 2.8],
            "requires_consecutive_outside_points": 2,
            "requires_consecutive_inside_points": 3,
            "excludes_designed_entrances": [
                "central_highland_step",
                "road_tunnel",
                "highland_tunnel",
            ],
            "tunnel_exclusion_radius_m": {
                "highland_tunnel": 2.8,
                "road_tunnel": 2.2,
            },
        },
        "detector_crossing_counts": dict(detector_counts),
        "official_event_detector_validation": validation,
        "caveat": "trajectory-only evidence is probabilistic; not observed is not a negative capability label",
    }
    return wide_rows, long_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--manual-labels", type=Path, default=DEFAULT_MANUAL_LABELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wide_rows, long_rows, summary = analyze(args.db, args.manual_labels)
    if len(wide_rows) != 44 * 5 or len(long_rows) != 44 * 5 * len(ABILITY_ORDER):
        raise RuntimeError("unexpected output row count")
    for path in write_outputs(wide_rows, long_rows, summary):
        print(path)


if __name__ == "__main__":
    main()
