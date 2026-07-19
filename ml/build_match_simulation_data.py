#!/usr/bin/env python3
"""Build compact empirical distributions for the browser matchup simulator.

The output is deliberately descriptive rather than an official-rules engine.
Each 15-second bin retains a team's observed structure/mobile damage, dart
windows, rune/buff events, terrain actions and fortress occupancy proxy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.team_style_report import TEAMS  # noqa: E402


DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
DEFAULT_PROFILES = ROOT / "analysis" / "outputs" / "team_profiles_44.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "models" / "match_simulation.json"

BIN_SECONDS = 15
BIN_COUNT = 28
COMBAT_CATEGORIES = {"17mm", "42mm", "飞镖"}
MOBILE_TYPES = {"英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"}
GROUND_TYPES = ("英雄", "工程", "步兵3", "步兵4", "哨兵")

# Per-bin compact layout.
BASE, OUTPOST, MOBILE, D17, D42, DART, DART_HITS, DART_GATES, FORT_OWN, FORT_ENEMY, BUFF, TERRAIN = range(12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def empty_game(game_id: int, opponent: str, won: bool, duration: int) -> dict:
    return {
        "id": game_id,
        "opponent": opponent,
        "won": won,
        "duration": duration,
        "bins": [[0.0] * 12 for _ in range(BIN_COUNT)],
        "targets": Counter(),
        "received": Counter(),
    }


def bin_index(second: float | None) -> int:
    return max(0, min(BIN_COUNT - 1, int(float(second or 0) // BIN_SECONDS)))


def rounded_number(value: float) -> int | float:
    rounded = float(round(float(value), 3))
    return int(rounded) if rounded.is_integer() else rounded


def main() -> None:
    args = parse_args()
    entries = {entry.school: entry for entry in TEAMS}
    schools = set(entries)
    profiles = {
        item["school"]: item
        for item in json.loads(args.profiles.read_text(encoding="utf-8"))
    }
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row

    matches = {
        int(row["game_id"]): row
        for row in connection.execute(
            "SELECT game_id,红方学校,蓝方学校,胜方,时长秒 FROM matches"
        )
    }
    games: dict[str, dict[int, dict]] = {school: {} for school in schools}
    for game_id, match in matches.items():
        red, blue = match["红方学校"], match["蓝方学校"]
        if red in schools:
            games[red][game_id] = empty_game(game_id, blue, match["胜方"] == "红", int(match["时长秒"] or 420))
        if blue in schools:
            games[blue][game_id] = empty_game(game_id, red, match["胜方"] == "蓝", int(match["时长秒"] or 420))

    event_sql = """
        SELECT game_id,时刻秒,事件类型,机器人类型,阵营,学校名,目标类型,类别,数值
        FROM events
        WHERE 事件类型 IN ('受击','飞镖命中','飞镖闸门开','增益')
        ORDER BY game_id,时刻秒
    """
    unique_tactical_events: set[tuple] = set()
    for event in connection.execute(event_sql):
        game_id = int(event["game_id"])
        match = matches.get(game_id)
        if match is None:
            continue
        second = float(event["时刻秒"] or 0)
        index = bin_index(second)
        event_type = event["事件类型"]
        school = event["学校名"]
        if event_type == "受击":
            victim = school
            if victim == match["红方学校"]:
                attacker = match["蓝方学校"]
            elif victim == match["蓝方学校"]:
                attacker = match["红方学校"]
            else:
                continue
            category = event["类别"] or "未知"
            if category not in COMBAT_CATEGORIES:
                continue
            damage = abs(float(event["数值"] or 0))
            victim_type = event["机器人类型"] or "未知"
            if attacker in games and game_id in games[attacker]:
                game = games[attacker][game_id]
                if victim_type == "基地":
                    game["bins"][index][BASE] += damage
                elif victim_type == "前哨站":
                    game["bins"][index][OUTPOST] += damage
                elif victim_type in MOBILE_TYPES:
                    game["bins"][index][MOBILE] += damage
                game["bins"][index][{"17mm": D17, "42mm": D42, "飞镖": DART}[category]] += damage
                game["targets"][victim_type] += damage
            if victim in games and game_id in games[victim]:
                group = "基地" if victim_type == "基地" else "前哨站" if victim_type == "前哨站" else "机器人"
                games[victim][game_id]["received"][group] += damage
            continue

        if school not in games or game_id not in games[school]:
            continue
        game = games[school][game_id]
        if event_type == "飞镖命中":
            game["bins"][index][DART_HITS] += 1
        elif event_type == "飞镖闸门开":
            game["bins"][index][DART_GATES] += 1
        elif event_type == "增益":
            category = event["类别"] or ""
            key = (game_id, school, int(second), category)
            if key in unique_tactical_events:
                continue
            unique_tactical_events.add(key)
            if category in {"小能量机关增益", "大能量机关增益"}:
                game["bins"][index][BUFF] += 1
            elif category in {"过中央高地", "台阶跨越", "飞坡"}:
                game["bins"][index][TERRAIN] += 1

    # Fortress centres are the two hexagonal structures marked on the map.
    # Canonical red-side coordinates: own (6.65, 7.5), enemy (21.35, 7.5).
    placeholders = ",".join("?" for _ in schools)
    occupancy_sql = f"""
        WITH ground AS (
          SELECT game_id,学校名,CAST(时刻秒 / {BIN_SECONDS} AS INT) AS bin,
                 CAST(时刻秒 AS INT) AS sec,
                 CASE WHEN 阵营='蓝' THEN 28.0-x ELSE x END AS cx,
                 CASE WHEN 阵营='蓝' THEN 15.0-y ELSE y END AS cy
          FROM timeseries
          WHERE 学校名 IN ({placeholders})
            AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
            AND 当前血量>0 AND x IS NOT NULL AND y IS NOT NULL
        )
        SELECT game_id,学校名,bin,
          COUNT(DISTINCT CASE WHEN (cx-6.65)*(cx-6.65)+(cy-7.5)*(cy-7.5)<=1.69 THEN sec END) AS own_sec,
          COUNT(DISTINCT CASE WHEN (cx-21.35)*(cx-21.35)+(cy-7.5)*(cy-7.5)<=1.69 THEN sec END) AS enemy_sec
        FROM ground
        WHERE bin BETWEEN 0 AND {BIN_COUNT - 1}
        GROUP BY game_id,学校名,bin
    """
    for row in connection.execute(occupancy_sql, tuple(schools)):
        school, game_id, index = row["学校名"], int(row["game_id"]), int(row["bin"])
        if school not in games or game_id not in games[school]:
            continue
        games[school][game_id]["bins"][index][FORT_OWN] = min(BIN_SECONDS, int(row["own_sec"] or 0))
        games[school][game_id]["bins"][index][FORT_ENEMY] = min(BIN_SECONDS, int(row["enemy_sec"] or 0))
    connection.close()

    teams = {}
    all_received = []
    for school in sorted(schools):
        game_rows = list(games[school].values())
        target_totals: Counter = Counter()
        received_totals: Counter = Counter()
        category_totals = Counter()
        totals = Counter()
        for game in game_rows:
            target_totals.update(game["targets"])
            received_totals.update(game["received"])
            for values in game["bins"]:
                totals["base"] += values[BASE]
                totals["outpost"] += values[OUTPOST]
                totals["mobile"] += values[MOBILE]
                totals["dart_hits"] += values[DART_HITS]
                totals["dart_gates"] += values[DART_GATES]
                totals["fort_own"] += values[FORT_OWN]
                totals["fort_enemy"] += values[FORT_ENEMY]
                totals["buff"] += values[BUFF]
                totals["terrain"] += values[TERRAIN]
                category_totals["17mm"] += values[D17]
                category_totals["42mm"] += values[D42]
                category_totals["飞镖"] += values[DART]
        count = max(1, len(game_rows))
        received_per_game = sum(received_totals.values()) / count
        all_received.append(received_per_game)
        entry = entries[school]
        teams[school] = {
            "school": school,
            "team": entry.team,
            "stage": entry.stage,
            "region": entry.region,
            "games": [
                {
                    "id": game["id"], "opponent": game["opponent"], "won": game["won"],
                    "duration": game["duration"],
                    "bins": [[rounded_number(value) for value in values] for values in game["bins"]],
                }
                for game in game_rows
            ],
            "aggregate": {
                "games": len(game_rows),
                "wins": sum(game["won"] for game in game_rows),
                "win_rate": round(sum(game["won"] for game in game_rows) / count, 4),
                "damage_per_game": round((totals["base"] + totals["outpost"] + totals["mobile"]) / count, 2),
                "base_damage_per_game": round(totals["base"] / count, 2),
                "outpost_damage_per_game": round(totals["outpost"] / count, 2),
                "mobile_damage_per_game": round(totals["mobile"] / count, 2),
                "received_per_game": round(received_per_game, 2),
                "damage_by_category": {key: round(value / count, 2) for key, value in category_totals.items()},
                "damage_by_target": {key: round(value / count, 2) for key, value in target_totals.items()},
                "dart_hits_per_game": round(totals["dart_hits"] / count, 3),
                "dart_gates_per_game": round(totals["dart_gates"] / count, 3),
                "fortress_own_seconds_per_game": round(totals["fort_own"] / count, 2),
                "fortress_enemy_seconds_per_game": round(totals["fort_enemy"] / count, 2),
                "buff_windows_per_game": round(totals["buff"] / count, 3),
                "terrain_actions_per_game": round(totals["terrain"] / count, 3),
                "style": profiles.get(school, {}).get("style", "样本不足"),
            },
        }

    payload = {
        "schema_version": 1,
        "kind": "empirical_15s_matchup_simulation_inputs",
        "bin_seconds": BIN_SECONDS,
        "bin_count": BIN_COUNT,
        "bin_schema": [
            "base_damage", "outpost_damage", "mobile_damage", "damage_17mm",
            "damage_42mm", "damage_dart", "dart_hits", "dart_gate_opens",
            "own_fortress_seconds", "enemy_fortress_seconds", "buff_windows", "terrain_actions",
        ],
        "fortress_proxy": {
            "canonical_centres_m": {"own": [6.65, 7.5], "enemy": [21.35, 7.5]},
            "radius_m": 1.3,
            "meaning": "seconds with at least one living ground robot inside the fortress radius",
        },
        "global": {
            "mean_received_damage_per_game": round(sum(all_received) / max(1, len(all_received)), 3),
            "limitations": [
                "cross-region matchups are counterfactual and have no direct head-to-head samples",
                "fortress control is inferred from position occupancy rather than a referee control event",
                "simulation score is an operator-facing tactical score, not the official referee score",
            ],
        },
        "teams": teams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(teams)} teams to {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
