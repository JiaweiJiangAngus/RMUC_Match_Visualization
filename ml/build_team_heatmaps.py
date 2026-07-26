#!/usr/bin/env python3
"""Build five high-resolution tactical heatmap modes for all 96 teams."""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "docs" / "data" / "catalog.json"
DEFAULT_GAMES = ROOT / "docs" / "data" / "games"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "heatmaps"
DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
FIELD_WIDTH = 28.0
FIELD_HEIGHT = 15.0
CELL_SIZE_METRES = 0.1
GRID_WIDTH = round(FIELD_WIDTH / CELL_SIZE_METRES)
GRID_HEIGHT = round(FIELD_HEIGHT / CELL_SIZE_METRES)
GAUSSIAN_SIGMA_METRES = 0.22
WINDOW_SECONDS = 15
MOBILE_ROLES = ("英雄", "工程", "步兵3", "步兵4", "哨兵", "空中")
MOBILE_ROLE_SET = set(MOBILE_ROLES)
ROW = {"type": 1, "side": 2, "hp": 3, "x": 5, "y": 6}


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="regional SQLite dataset containing per-projectile firing events",
    )
    return parser.parse_args()


def sparse_grid(values: Counter) -> str:
    return ",".join(f"{index}:{values[index]}" for index in sorted(values))


def serialise_windows(values, window_count: int) -> list[dict]:
    windows = []
    for window in range(window_count):
        red = values[window]["red"]
        blue = values[window]["blue"]
        windows.append({
            "start": window * WINDOW_SECONDS,
            "end": (window + 1) * WINDOW_SECONDS,
            "samples": sum(red.values()) + sum(blue.values()),
            "red": sparse_grid(red),
            "blue": sparse_grid(blue),
        })
    return windows


def serialise_series(grids, windows, window_count: int) -> dict:
    red = grids["red"]
    blue = grids["blue"]
    return {
        "samples": sum(red.values()) + sum(blue.values()),
        "red": sparse_grid(red),
        "blue": sparse_grid(blue),
        "windows": serialise_windows(windows, window_count),
    }


def aggregate_series(
    school_grids,
    school_windows,
    school_role_grids,
    school_role_windows,
    window_count: int,
) -> dict:
    """Merge every school's sparse source counters before serialisation."""
    grids = {"red": Counter(), "blue": Counter()}
    windows = defaultdict(lambda: {"red": Counter(), "blue": Counter()})
    for school_grid in school_grids.values():
        for side in ("red", "blue"):
            grids[side].update(school_grid[side])
    for school_window in school_windows.values():
        for window in range(window_count):
            for side in ("red", "blue"):
                windows[window][side].update(school_window[window][side])

    payload = serialise_series(grids, windows, window_count)
    payload["roles"] = {}
    for role in MOBILE_ROLES:
        role_grids = {"red": Counter(), "blue": Counter()}
        role_windows = defaultdict(lambda: {"red": Counter(), "blue": Counter()})
        for school_roles in school_role_grids.values():
            for side in ("red", "blue"):
                role_grids[side].update(school_roles[role][side])
        for school_roles in school_role_windows.values():
            for window in range(window_count):
                for side in ("red", "blue"):
                    role_windows[window][side].update(
                        school_roles[role][window][side]
                    )
        payload["roles"][role] = serialise_series(
            role_grids,
            role_windows,
            window_count,
        )
    return payload


def load_firing_events(db_path: Path, schools: set[str]) -> dict[int, list[dict]]:
    """Load one aggregate row per shooter/second, weighted by projectiles fired."""
    if not db_path.exists():
        raise FileNotFoundError(f"regional dataset not found: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    events: dict[int, list[dict]] = defaultdict(list)
    try:
        for row in connection.execute(
            """
            SELECT game_id,CAST(时刻秒 AS INT) second,robot_id,
                   机器人类型 role,阵营 side,学校名 school,类别 calibre,
                   COUNT(*) shots
            FROM events
            WHERE 事件类型='发弹'
              AND 类别 IN ('17mm','42mm')
              AND 机器人类型 IN ('英雄','步兵3','步兵4','哨兵','空中')
            GROUP BY game_id,second,robot_id,role,side,school,calibre
            """
        ):
            if row["school"] in schools:
                events[int(row["game_id"])].append(dict(row))
    finally:
        connection.close()
    return events


def load_hit_events(db_path: Path, schools: set[str]) -> dict[int, list[dict]]:
    """Load mobile-robot hit events, retaining damage category and victim id."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    events: dict[int, list[dict]] = defaultdict(list)
    try:
        for row in connection.execute(
            """
            SELECT game_id,CAST(时刻秒 AS INT) second,robot_id,
                   机器人类型 role,阵营 side,学校名 school,类别 category,
                   COUNT(*) hits
            FROM events
            WHERE 事件类型='受击'
              AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵','空中')
            GROUP BY game_id,second,robot_id,role,side,school,category
            """
        ):
            if row["school"] in schools:
                events[int(row["game_id"])].append(dict(row))
    finally:
        connection.close()
    return events


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
    firing_events = load_firing_events(args.db, set(school_region))
    hit_events = load_hit_events(args.db, set(school_region))
    hit_category_counts = Counter()
    for events in hit_events.values():
        for event in events:
            hit_category_counts[event["category"]] += int(event["hits"])
    combat_hit_seconds = {
        (game_id, int(event["second"]), int(event["robot_id"]))
        for game_id, events in hit_events.items()
        for event in events
        if event["category"] in {"17mm", "42mm"}
    }

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
    role_grids = {
        school: {
            role: {"red": Counter(), "blue": Counter()}
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    role_window_grids = {
        school: {
            role: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    firing_grids = {
        school: {"red": Counter(), "blue": Counter()}
        for school in school_region
    }
    firing_window_grids = {
        school: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
        for school in school_region
    }
    firing_role_grids = {
        school: {
            role: {"red": Counter(), "blue": Counter()}
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    firing_role_window_grids = {
        school: {
            role: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    death_grids = {
        school: {"red": Counter(), "blue": Counter()}
        for school in school_region
    }
    death_window_grids = {
        school: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
        for school in school_region
    }
    death_role_grids = {
        school: {
            role: {"red": Counter(), "blue": Counter()}
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    death_role_window_grids = {
        school: {
            role: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    hit_grids = {
        school: {"red": Counter(), "blue": Counter()}
        for school in school_region
    }
    hit_window_grids = {
        school: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
        for school in school_region
    }
    hit_role_grids = {
        school: {
            role: {"red": Counter(), "blue": Counter()}
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    hit_role_window_grids = {
        school: {
            role: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    kill_grids = {
        school: {"red": Counter(), "blue": Counter()}
        for school in school_region
    }
    kill_window_grids = {
        school: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
        for school in school_region
    }
    kill_role_grids = {
        school: {
            role: {"red": Counter(), "blue": Counter()}
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    kill_role_window_grids = {
        school: {
            role: defaultdict(lambda: {"red": Counter(), "blue": Counter()})
            for role in MOBILE_ROLES
        }
        for school in school_region
    }
    games = defaultdict(int)
    samples = defaultdict(int)
    firing_samples = defaultdict(int)
    firing_groups = defaultdict(int)
    death_samples = defaultdict(int)
    missing_death_positions = 0
    hit_samples = defaultdict(int)
    hit_groups = defaultdict(int)
    exact_hit_groups = 0
    adjacent_hit_groups = 0
    missing_hit_groups = 0
    kill_samples = defaultdict(int)
    exact_second_kills = 0
    previous_second_kills = 0
    excluded_noncombat_deaths = 0
    exact_firing_groups = 0
    adjacent_firing_groups = 0
    missing_firing_groups = 0
    processed_game_ids: set[int] = set()
    max_window = 0

    for path in sorted(args.games_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            game = json.load(handle)
        info = game["info"]
        game_id = int(info["game_id"])
        processed_game_ids.add(game_id)
        side_school = {"红": info["red"], "蓝": info["blue"]}
        games[info["red"]] += 1
        games[info["blue"]] += 1
        max_window = max(
            max_window,
            (int(info["duration"]) - 1) // WINDOW_SECONDS,
        )
        frame_robots: dict[int, dict[int, list]] = {}
        previous_hp: dict[int, float] = {}
        frame_items = sorted(
            game.get("frames", {}).items(),
            key=lambda item: float(item[0]),
        )
        for second, robots in frame_items:
            second_index = int(float(second))
            window = second_index // WINDOW_SECONDS
            frame_robots[second_index] = {int(robot[0]): robot for robot in robots}
            for robot in robots:
                role = robot[ROW["type"]]
                if role not in MOBILE_ROLE_SET:
                    continue
                robot_id = int(robot[0])
                current_hp = float(robot[ROW["hp"]] or 0)
                previous_robot_hp = previous_hp.get(robot_id)
                previous_hp[robot_id] = current_hp
                x, y = robot[ROW["x"]], robot[ROW["y"]]
                school = side_school[robot[ROW["side"]]]
                side = "red" if robot[ROW["side"]] == "红" else "blue"
                if (
                    previous_robot_hp is not None
                    and previous_robot_hp > 0
                    and current_hp <= 0
                ):
                    kill_evidence_offset = None
                    if (game_id, second_index, robot_id) in combat_hit_seconds:
                        kill_evidence_offset = 0
                        exact_second_kills += 1
                    elif (game_id, second_index - 1, robot_id) in combat_hit_seconds:
                        kill_evidence_offset = -1
                        previous_second_kills += 1
                    else:
                        excluded_noncombat_deaths += 1
                    if x is None or y is None:
                        missing_death_positions += 1
                    else:
                        death_x = max(0.0, min(FIELD_WIDTH, float(x)))
                        death_y = max(0.0, min(FIELD_HEIGHT, float(y)))
                        death_grid_x = min(
                            GRID_WIDTH - 1,
                            int(death_x / CELL_SIZE_METRES),
                        )
                        death_grid_y = min(
                            GRID_HEIGHT - 1,
                            int(death_y / CELL_SIZE_METRES),
                        )
                        death_grid_index = death_grid_y * GRID_WIDTH + death_grid_x
                        death_grids[school][side][death_grid_index] += 1
                        death_window_grids[school][window][side][death_grid_index] += 1
                        death_role_grids[school][role][side][death_grid_index] += 1
                        death_role_window_grids[school][role][window][side][death_grid_index] += 1
                        death_samples[school] += 1
                        if kill_evidence_offset is not None:
                            killer_side_name = "蓝" if robot[ROW["side"]] == "红" else "红"
                            killer_side = "blue" if killer_side_name == "蓝" else "red"
                            killer_school = side_school[killer_side_name]
                            kill_grids[killer_school][killer_side][death_grid_index] += 1
                            kill_window_grids[killer_school][window][killer_side][death_grid_index] += 1
                            kill_role_grids[killer_school][role][killer_side][death_grid_index] += 1
                            kill_role_window_grids[killer_school][role][window][killer_side][death_grid_index] += 1
                            kill_samples[killer_school] += 1
                if current_hp <= 0:
                    continue
                if x is None or y is None:
                    continue
                x = max(0.0, min(FIELD_WIDTH, float(x)))
                y = max(0.0, min(FIELD_HEIGHT, float(y)))
                grid_x = min(GRID_WIDTH - 1, int(x / CELL_SIZE_METRES))
                grid_y = min(GRID_HEIGHT - 1, int(y / CELL_SIZE_METRES))
                grid_index = grid_y * GRID_WIDTH + grid_x
                grids[school][side][grid_index] += 1
                window_grids[school][window][side][grid_index] += 1
                role_grids[school][role][side][grid_index] += 1
                role_window_grids[school][role][window][side][grid_index] += 1
                samples[school] += 1

        for event in firing_events.get(game_id, ()):
            school = event["school"]
            role = event["role"]
            side = "red" if event["side"] == "红" else "blue"
            second = int(event["second"])
            robot = None
            matched_offset = None
            for offset in (0, -1, 1):
                robot = frame_robots.get(second + offset, {}).get(int(event["robot_id"]))
                if robot is not None:
                    matched_offset = offset
                    break
            if (
                robot is None
                or role not in MOBILE_ROLE_SET
                or school != side_school.get(event["side"])
                or robot[ROW["type"]] != role
                or robot[ROW["side"]] != event["side"]
            ):
                missing_firing_groups += 1
                continue
            x, y = robot[ROW["x"]], robot[ROW["y"]]
            if x is None or y is None:
                missing_firing_groups += 1
                continue
            if matched_offset == 0:
                exact_firing_groups += 1
            else:
                adjacent_firing_groups += 1
            x = max(0.0, min(FIELD_WIDTH, float(x)))
            y = max(0.0, min(FIELD_HEIGHT, float(y)))
            grid_x = min(GRID_WIDTH - 1, int(x / CELL_SIZE_METRES))
            grid_y = min(GRID_HEIGHT - 1, int(y / CELL_SIZE_METRES))
            grid_index = grid_y * GRID_WIDTH + grid_x
            window = second // WINDOW_SECONDS
            shots = int(event["shots"])
            firing_grids[school][side][grid_index] += shots
            firing_window_grids[school][window][side][grid_index] += shots
            firing_role_grids[school][role][side][grid_index] += shots
            firing_role_window_grids[school][role][window][side][grid_index] += shots
            firing_samples[school] += shots
            firing_groups[school] += 1

        for event in hit_events.get(game_id, ()):
            school = event["school"]
            role = event["role"]
            side = "red" if event["side"] == "红" else "blue"
            second = int(event["second"])
            robot = None
            matched_offset = None
            for offset in (0, -1, 1):
                robot = frame_robots.get(second + offset, {}).get(int(event["robot_id"]))
                if robot is not None:
                    matched_offset = offset
                    break
            if (
                robot is None
                or role not in MOBILE_ROLE_SET
                or school != side_school.get(event["side"])
                or robot[ROW["type"]] != role
                or robot[ROW["side"]] != event["side"]
            ):
                missing_hit_groups += 1
                continue
            x, y = robot[ROW["x"]], robot[ROW["y"]]
            if x is None or y is None:
                missing_hit_groups += 1
                continue
            if matched_offset == 0:
                exact_hit_groups += 1
            else:
                adjacent_hit_groups += 1
            x = max(0.0, min(FIELD_WIDTH, float(x)))
            y = max(0.0, min(FIELD_HEIGHT, float(y)))
            grid_x = min(GRID_WIDTH - 1, int(x / CELL_SIZE_METRES))
            grid_y = min(GRID_HEIGHT - 1, int(y / CELL_SIZE_METRES))
            grid_index = grid_y * GRID_WIDTH + grid_x
            window = second // WINDOW_SECONDS
            hits = int(event["hits"])
            hit_grids[school][side][grid_index] += hits
            hit_window_grids[school][window][side][grid_index] += hits
            hit_role_grids[school][role][side][grid_index] += hits
            hit_role_window_grids[school][role][window][side][grid_index] += hits
            hit_samples[school] += hits
            hit_groups[school] += 1

    missing_firing_groups += sum(
        len(events)
        for game_id, events in firing_events.items()
        if game_id not in processed_game_ids
    )
    missing_hit_groups += sum(
        len(events)
        for game_id, events in hit_events.items()
        if game_id not in processed_game_ids
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_count = max_window + 1
    schools = []
    for index, school in enumerate(sorted(school_region)):
        filename = f"{index:03d}.json"
        roles = {}
        for role in MOBILE_ROLES:
            roles[role] = serialise_series(
                role_grids[school][role],
                role_window_grids[school][role],
                window_count,
            )
        firing_roles = {}
        for role in MOBILE_ROLES:
            firing_roles[role] = serialise_series(
                firing_role_grids[school][role],
                firing_role_window_grids[school][role],
                window_count,
            )
        firing_payload = serialise_series(
            firing_grids[school],
            firing_window_grids[school],
            window_count,
        )
        firing_payload["groups"] = firing_groups[school]
        firing_payload["roles"] = firing_roles
        death_roles = {}
        for role in MOBILE_ROLES:
            death_roles[role] = serialise_series(
                death_role_grids[school][role],
                death_role_window_grids[school][role],
                window_count,
            )
        death_payload = serialise_series(
            death_grids[school],
            death_window_grids[school],
            window_count,
        )
        death_payload["roles"] = death_roles
        hit_roles = {}
        for role in MOBILE_ROLES:
            hit_roles[role] = serialise_series(
                hit_role_grids[school][role],
                hit_role_window_grids[school][role],
                window_count,
            )
        hit_payload = serialise_series(
            hit_grids[school],
            hit_window_grids[school],
            window_count,
        )
        hit_payload["groups"] = hit_groups[school]
        hit_payload["roles"] = hit_roles
        kill_roles = {}
        for role in MOBILE_ROLES:
            kill_roles[role] = serialise_series(
                kill_role_grids[school][role],
                kill_role_window_grids[school][role],
                window_count,
            )
        kill_payload = serialise_series(
            kill_grids[school],
            kill_window_grids[school],
            window_count,
        )
        kill_payload["roles"] = kill_roles
        payload = {
            "school": school,
            "region": school_region[school],
            "games": games[school],
            "samples": samples[school],
            "red": sparse_grid(grids[school]["red"]),
            "blue": sparse_grid(grids[school]["blue"]),
            "windows": serialise_windows(window_grids[school], window_count),
            "roles": roles,
            "shots": firing_payload,
            "deaths": death_payload,
            "hits": hit_payload,
            "kills": kill_payload,
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
            "shots": firing_samples[school],
            "deaths": death_samples[school],
            "hits": hit_samples[school],
            "kills": kill_samples[school],
        })

    total_games = sum(games.values()) // 2
    aggregate_payload = aggregate_series(
        grids,
        window_grids,
        role_grids,
        role_window_grids,
        window_count,
    )
    aggregate_payload.update({
        "school": "全部学校（96队）",
        "region": "全部赛区",
        "games": total_games,
        "team_count": len(schools),
    })
    aggregate_payload["shots"] = aggregate_series(
        firing_grids,
        firing_window_grids,
        firing_role_grids,
        firing_role_window_grids,
        window_count,
    )
    aggregate_payload["shots"]["groups"] = sum(firing_groups.values())
    aggregate_payload["deaths"] = aggregate_series(
        death_grids,
        death_window_grids,
        death_role_grids,
        death_role_window_grids,
        window_count,
    )
    aggregate_payload["hits"] = aggregate_series(
        hit_grids,
        hit_window_grids,
        hit_role_grids,
        hit_role_window_grids,
        window_count,
    )
    aggregate_payload["hits"]["groups"] = sum(hit_groups.values())
    aggregate_payload["kills"] = aggregate_series(
        kill_grids,
        kill_window_grids,
        kill_role_grids,
        kill_role_window_grids,
        window_count,
    )
    aggregate_filename = "all.json"
    (args.output_dir / aggregate_filename).write_text(
        json.dumps(
            aggregate_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    aggregate_entry = {
        "school": aggregate_payload["school"],
        "region": aggregate_payload["region"],
        "file": aggregate_filename,
        "games": total_games,
        "team_count": len(schools),
        "samples": aggregate_payload["samples"],
        "shots": aggregate_payload["shots"]["samples"],
        "deaths": aggregate_payload["deaths"]["samples"],
        "hits": aggregate_payload["hits"]["samples"],
        "kills": aggregate_payload["kills"]["samples"],
    }

    config = {
        "schema_version": 5,
        "kind": "rmuc_2026_team_five_mode_tactical_heatmaps",
        "modes": ["position", "shots", "hits", "kills", "deaths"],
        "grid_width": GRID_WIDTH,
        "grid_height": GRID_HEIGHT,
        "cell_size_metres": CELL_SIZE_METRES,
        "gaussian_sigma_metres": GAUSSIAN_SIGMA_METRES,
        "window_seconds": WINDOW_SECONDS,
        "window_count": window_count,
        "roles": list(MOBILE_ROLES),
        "x_range": [0, FIELD_WIDTH],
        "y_range": [0, FIELD_HEIGHT],
        "regions": catalog["regions"],
        "schools": schools,
        "aggregate": aggregate_entry,
        "firing_event_groups": exact_firing_groups + adjacent_firing_groups,
        "firing_projectiles": sum(firing_samples.values()),
        "firing_position_matching": {
            "exact_groups": exact_firing_groups,
            "adjacent_1s_groups": adjacent_firing_groups,
            "missing_groups": missing_firing_groups,
        },
        "death_events": sum(death_samples.values()),
        "death_position_detection": {
            "method": "同一robot_id相邻轨迹由HP>0转为HP=0时记录归零秒坐标",
            "missing_positions": missing_death_positions,
        },
        "hit_events": sum(hit_samples.values()),
        "hit_event_groups": exact_hit_groups + adjacent_hit_groups,
        "hit_categories": dict(sorted(hit_category_counts.items())),
        "hit_position_matching": {
            "exact_groups": exact_hit_groups,
            "adjacent_1s_groups": adjacent_hit_groups,
            "missing_groups": missing_hit_groups,
        },
        "kill_events": sum(kill_samples.values()),
        "kill_position_detection": {
            "weapon_hit_same_second": exact_second_kills,
            "weapon_hit_previous_1s": previous_second_kills,
            "excluded_noncombat_deaths": excluded_noncombat_deaths,
            "method": "敌方robot_id血量归零，且归零秒或前1秒存在17mm/42mm受击事件；位置取敌方归零坐标，归属取对手学校",
            "limitation": "裁判数据不提供具体击杀者，兵种筛选表示被击杀兵种",
        },
        "source": "613局区域赛1Hz机器人位置、血量、逐发发弹与受击事件；发弹和受击按同局同秒robot_id映射坐标，缺帧时只取相邻1秒；阵亡按血量由正数首次转为0去重；击杀仅认归零秒或前1秒存在17mm/42mm受击证据的敌方归零，并归到对手学校；0.1m等距学校级红蓝方密度场；支持六兵种、15秒切片与统一红蓝视角；基地和前哨站不计入",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"wrote {len(schools)} team heatmaps and all-school aggregate from "
        f"{total_games} games "
        f"with {sum(firing_samples.values())} projectiles, "
        f"{sum(hit_samples.values())} hits, {sum(kill_samples.values())} kills and "
        f"{sum(death_samples.values())} deaths to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
