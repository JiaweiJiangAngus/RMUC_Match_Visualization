#!/usr/bin/env python3
"""Build per-team/per-role parameters for the agent-based browser simulator.

The referee export has exact tracks, HP, heat and fired-projectile counters but
does not expose remaining ammunition or the attacker of a hit.  This builder
therefore keeps movement/firing parameters empirical and estimates team weapon
accuracy as detected hits divided by projectiles fired.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.team_style_report import TEAMS  # noqa: E402


DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
DEFAULT_MACRO = ROOT / "docs" / "data" / "models" / "match_simulation.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "models" / "full_simulation.json"
ROLES = ("英雄", "工程", "步兵3", "步兵4", "哨兵", "空中")
PHASES = 7
CORE_UNLOCK_SECONDS = (0, 60, 120, 180)
CORE_FIRST_INCOME_PER_10 = (50, 25, 25, 50)
# Total recurring core income after first completing Lv.1-Lv.4.  A little
# tolerance is needed because referee rows can land one second either side of
# the automatic-income tick.
CORE_INCOME_THRESHOLDS = (45, 68, 92, 138)
AUTOMATIC_INCOME_TICKS = {
    60: 50, 61: 50, 120: 50, 121: 50, 180: 50, 181: 50,
    240: 50, 241: 50, 300: 50, 301: 50, 360: 150, 361: 150,
}

HERO_ARCHETYPES = {
    "melee": {
        "label": "近战优先",
        "hp_by_level": [260, 300, 330, 360, 400, 430, 460, 500, 530, 600],
    },
    "ranged": {
        "label": "远程优先",
        "hp_by_level": [200, 220, 240, 260, 280, 300, 320, 340, 360, 400],
    },
}

ROLE_DEFAULTS = {
    "英雄": {"hp": 200, "weapon": "42mm", "range": 12.0, "speed": 1.55, "heat_limit": 200, "cooling": 20, "magazine": 12},
    "工程": {"hp": 250, "weapon": None, "range": 0, "speed": 1.45, "heat_limit": 0, "cooling": 0, "magazine": 0},
    "步兵3": {"hp": 200, "weapon": "17mm", "range": 8.0, "speed": 1.75, "heat_limit": 200, "cooling": 20, "magazine": 100},
    "步兵4": {"hp": 200, "weapon": "17mm", "range": 8.0, "speed": 1.75, "heat_limit": 200, "cooling": 20, "magazine": 100},
    "哨兵": {"hp": 400, "weapon": "17mm", "range": 9.0, "speed": 1.65, "heat_limit": 260, "cooling": 30, "magazine": 200},
    "空中": {"hp": 100, "weapon": "17mm", "range": 10.0, "speed": 3.0, "heat_limit": 300, "cooling": 35, "magazine": 250},
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--macro", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rounded(value: float, digits: int = 3) -> int | float:
    result = float(round(float(value), digits))
    return int(result) if result.is_integer() else result


def percentile(values: list[float], ratio: float) -> float | None:
    """Small dependency-free linear percentile helper."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def infer_regional_economy(db: sqlite3.Connection, schools: tuple[str, ...]) -> dict[str, dict]:
    """Infer team initial coins and technology-core timing from total coins.

    The export has no explicit technology-core event.  Total team coins never
    decrease when ammunition or revives are purchased, so first-completion
    income can be separated from the rulebook's automatic income.  Core income
    is periodic every ten seconds; a rolling ten-second increase is therefore
    robust to each core having a different payout phase.
    """
    placeholders = ",".join("?" for _ in schools)
    sequences: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for row in db.execute(
        f"""
        SELECT game_id,学校名,CAST(时刻秒 AS INT) second,MAX(队伍总金币) total_coins
        FROM timeseries
        WHERE 学校名 IN ({placeholders}) AND 队伍总金币 IS NOT NULL
        GROUP BY game_id,学校名,second
        ORDER BY game_id,学校名,second
        """,
        schools,
    ):
        sequences[(int(row["game_id"]), row["学校名"])].append(
            (int(row["second"]), int(round(row["total_coins"])))
        )

    samples: dict[str, list[dict]] = defaultdict(list)
    for (_, school), values in sequences.items():
        if not values:
            continue
        totals = {second: total for second, total in values}
        first_second, initial_coins = values[0]
        last_total = initial_coins
        residual_income = [0] * 421
        total_by_minute = []
        for second in range(first_second + 1, 421):
            current_total = totals.get(second, last_total)
            delta = max(0, current_total - last_total)
            last_total = current_total
            # Region telemetry alternates between the exact minute and the
            # following second.  Removing the scheduled amount on both ticks
            # may hide one coincident core payout, but only for one ten-second
            # window; the persistence check below prevents a false level.
            residual_income[second] = max(0, delta - AUTOMATIC_INCOME_TICKS.get(second, 0))
        for checkpoint in range(0, 421, 60):
            observed = [total for second, total in values if second <= max(1, checkpoint)]
            total_by_minute.append(observed[-1] if observed else initial_coins)

        rolling_income = [0] * 421
        for second in range(1, 420):
            rolling_income[second] = sum(residual_income[max(0, second - 9): second + 1])

        completions: list[int] = []
        for unlock, threshold in zip(CORE_UNLOCK_SECONDS, CORE_INCOME_THRESHOLDS):
            start = max(unlock, completions[-1] + 1 if completions else unlock)
            completion = None
            for second in range(start, 420):
                # Require the newly reached rate to persist.  This rejects an
                # automatic-income row that happened to share the other tick.
                future = rolling_income[second:min(420, second + 8)]
                if rolling_income[second] >= threshold and sum(value >= threshold for value in future) >= 6:
                    completion = second
                    break
            if completion is None:
                break
            completions.append(completion)

        samples[school].append({
            "initial_coins": initial_coins,
            "completions": completions,
            "peak_core_income_per_10": max(rolling_income),
            "total_coins_by_minute": total_by_minute,
        })

    priors = {}
    for school in schools:
        team_samples = samples.get(school, [])
        game_count = len(team_samples)
        completion_stats = []
        reach_rates = []
        for index in range(4):
            times = [sample["completions"][index] for sample in team_samples if len(sample["completions"]) > index]
            reach_rates.append(rounded(len(times) / max(1, game_count), 3))
            completion_stats.append({
                "samples": len(times),
                "p25": round(percentile(times, 0.25)) if times else None,
                "median": round(median(times)) if times else None,
                "p75": round(percentile(times, 0.75)) if times else None,
            })
        coin_curve = []
        for index, checkpoint in enumerate(range(0, 421, 60)):
            values = [sample["total_coins_by_minute"][index] for sample in team_samples]
            coin_curve.append([checkpoint, round(median(values)) if values else 400])
        initial_values = [sample["initial_coins"] for sample in team_samples]
        peak_values = [sample["peak_core_income_per_10"] for sample in team_samples]
        priors[school] = {
            "source": "RMUC 2026 区域赛队伍总金币遥测",
            "games": game_count,
            "regional_initial_coins": round(median(initial_values) / 25) * 25 if initial_values else 400,
            "regional_total_coins_by_minute": coin_curve,
            "core_reach_rate": reach_rates,
            "core_completion_seconds": completion_stats,
            "regional_peak_core_income_per_10": round(median(peak_values)) if peak_values else 0,
        }
    return priors


def is_uav_home(point: tuple[float, float] | list[float]) -> bool:
    """Canonical-red停机坪范围；与地图左右方的180°旋转保持一致。"""
    return point[0] < 3.2 and point[1] > 11.0


def build_uav_navigation(db: sqlite3.Connection, schools: tuple[str, ...]) -> dict[str, dict]:
    """Build state-separated UAV goals and five-second transition priors.

    Raw dwell counts mix the helipad with airborne positions.  Here every game
    contributes equal total weight, helipad runs are kept as an explicit state,
    and tactical movement is conditioned on the previous airborne cell.
    """
    placeholders = ",".join("?" for _ in schools)
    tracks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in db.execute(
        f"""
        SELECT 学校名,game_id,阵营,时刻秒,x,y
        FROM timeseries
        WHERE 学校名 IN ({placeholders}) AND 机器人类型='空中'
          AND x BETWEEN 0.05 AND 27.95 AND y BETWEEN 0.05 AND 14.95
        ORDER BY 学校名,game_id,时刻秒
        """,
        schools,
    ):
        x = 28.0 - float(row["x"]) if row["阵营"] == "蓝" else float(row["x"])
        y = 15.0 - float(row["y"]) if row["阵营"] == "蓝" else float(row["y"])
        tracks[(row["学校名"], int(row["game_id"]))].append({
            "second": float(row["时刻秒"]),
            "x": x,
            "y": y,
            "phase": min(PHASES - 1, int(float(row["时刻秒"]) / 60)),
            "home": is_uav_home((x, y)),
        })

    team_goal_weights: dict[tuple[str, int], Counter] = defaultdict(Counter)
    team_transition_weights: dict[tuple[str, int], Counter] = defaultdict(Counter)
    first_takeoffs: dict[str, list[float]] = defaultdict(list)
    airborne_runs: dict[str, list[float]] = defaultdict(list)
    parked_runs: dict[str, list[float]] = defaultdict(list)
    home_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    sample_counts: Counter = Counter()

    for (school, _), points in tracks.items():
        if not points:
            continue
        sample_counts[school] += len(points)
        for point in points:
            if point["home"]:
                home_points[school].append((point["x"], point["y"]))

        first_airborne = next((point["second"] for point in points if not point["home"]), None)
        if first_airborne is not None:
            first_takeoffs[school].append(first_airborne)

        run_state = points[0]["home"]
        run_start = points[0]["second"]
        previous_second = points[0]["second"]
        for point in points[1:]:
            gap = point["second"] - previous_second
            if point["home"] != run_state or gap > 1.6:
                duration = max(1.0, previous_second - run_start + 1)
                (parked_runs if run_state else airborne_runs)[school].append(duration)
                run_state = point["home"]
                run_start = point["second"]
            previous_second = point["second"]
        duration = max(1.0, previous_second - run_start + 1)
        (parked_runs if run_state else airborne_runs)[school].append(duration)

        # Normalise each game independently so a single long sortie does not
        # dominate a team's tactical probability mass.
        game_goals: dict[int, Counter] = defaultdict(Counter)
        for point in points:
            if point["home"]:
                continue
            cell = (round(point["x"] * 2) / 2, round(point["y"] * 2) / 2)
            # 边界外样本可能因 0.5 m 量化落回停机坪内，将这类
            # 量化伪影剔除，避免空中目标再次混入停机状态。
            if is_uav_home(cell):
                continue
            game_goals[point["phase"]][cell] += 1
        for phase, counts in game_goals.items():
            total = sum(counts.values()) or 1
            for cell, count in counts.items():
                team_goal_weights[(school, phase)][cell] += 1000 * count / total

        # Five-second transitions retain route continuity without shipping raw
        # one-second trajectories to the browser.
        game_transitions: dict[int, Counter] = defaultdict(Counter)
        right = 0
        for left, start in enumerate(points):
            if start["home"]:
                continue
            right = max(right, left + 1)
            while right < len(points) and points[right]["second"] - start["second"] < 4.0:
                right += 1
            if right >= len(points):
                break
            end = points[right]
            delta = end["second"] - start["second"]
            if delta > 6.0 or end["home"]:
                continue
            if any(point["home"] for point in points[left:right + 1]):
                continue
            distance = ((end["x"] - start["x"]) ** 2 + (end["y"] - start["y"]) ** 2) ** 0.5
            if distance > 18.0:
                continue
            source = (round(start["x"] * 2) / 2, round(start["y"] * 2) / 2)
            target = (round(end["x"] * 2) / 2, round(end["y"] * 2) / 2)
            if is_uav_home(source) or is_uav_home(target):
                continue
            game_transitions[start["phase"]][(*source, *target)] += 1
        for phase, counts in game_transitions.items():
            total = sum(counts.values()) or 1
            for transition, count in counts.items():
                team_transition_weights[(school, phase)][transition] += 1000 * count / total

    payload = {}
    for school in schools:
        goals_by_minute = []
        transitions_by_minute = []
        previous_goals = [[13.5, 12.5, 1]]
        for phase in range(PHASES):
            goals = [
                [rounded(cell[0]), rounded(cell[1]), max(1, round(weight))]
                for cell, weight in team_goal_weights[(school, phase)].most_common(24)
            ] or previous_goals
            goals_by_minute.append(goals)
            previous_goals = goals
            transitions_by_minute.append([
                [rounded(edge[0]), rounded(edge[1]), rounded(edge[2]), rounded(edge[3]), max(1, round(weight))]
                for edge, weight in team_transition_weights[(school, phase)].most_common(120)
            ])
        home = home_points.get(school, [])
        home_position = [
            rounded(median(point[0] for point in home)),
            rounded(median(point[1] for point in home)),
        ] if home else [1.5, 13.5]
        takeoff_values = first_takeoffs.get(school, [])
        air_values = [value for value in airborne_runs.get(school, []) if value >= 5]
        park_values = [value for value in parked_runs.get(school, []) if value >= 3]
        payload[school] = {
            "home": home_position,
            "airborne_goals_by_minute": goals_by_minute,
            "transitions_by_minute": transitions_by_minute,
            "first_takeoff_second": round(median(takeoff_values)) if takeoff_values else 420,
            "median_airborne_run_seconds": int(clamp(median(air_values) if air_values else 90, 20, 210)),
            "median_parked_run_seconds": int(clamp(median(park_values) if park_values else 30, 8, 120)),
            "samples": sample_counts[school],
            "source": "区域赛无人机轨迹；停机坪与空中状态分离，每局等权，5秒条件转移",
        }
    return payload


def main() -> None:
    options = args()
    schools = tuple(entry.school for entry in TEAMS)
    entries = {entry.school: entry for entry in TEAMS}
    placeholders = ",".join("?" for _ in schools)
    macro = json.loads(options.macro.read_text(encoding="utf-8"))
    db = sqlite3.connect(f"file:{options.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    economy_priors = infer_regional_economy(db, schools)
    uav_navigation = build_uav_navigation(db, schools)

    games = defaultdict(int)
    for row in db.execute("SELECT 红方学校,蓝方学校 FROM matches"):
        if row["红方学校"] in entries:
            games[row["红方学校"]] += 1
        if row["蓝方学校"] in entries:
            games[row["蓝方学校"]] += 1

    radar_counters: dict[str, int] = defaultdict(int)
    uav_counters_received: dict[str, int] = defaultdict(int)
    for row in db.execute(
        """
        SELECT e.学校名 target_school,m.红方学校 red_school,m.蓝方学校 blue_school
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='雷达反制UAV'
        """
    ):
        target = row["target_school"]
        if target in entries:
            uav_counters_received[target] += 1
        attacker = row["blue_school"] if target == row["red_school"] else row["red_school"]
        if attacker in entries:
            radar_counters[attacker] += 1

    shots: dict[tuple[str, str, str], int] = defaultdict(int)
    active_fire_seconds: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in db.execute(
        f"""
        SELECT 学校名,机器人类型,类别,COUNT(*) shots,
               COUNT(DISTINCT CAST(game_id AS TEXT)||':'||CAST(时刻秒 AS INT)) active_seconds
        FROM events
        WHERE 学校名 IN ({placeholders}) AND 事件类型='发弹'
          AND 机器人类型 IN ('英雄','步兵3','步兵4','哨兵','空中')
          AND 类别 IN ('17mm','42mm')
        GROUP BY 学校名,机器人类型,类别
        """,
        schools,
    ):
        key = (row["学校名"], row["机器人类型"], row["类别"])
        shots[key] = int(row["shots"] or 0)
        active_fire_seconds[key] = int(row["active_seconds"] or 0)

    dealt: dict[tuple[str, str], float] = defaultdict(float)
    for row in db.execute(
        f"""
        SELECT CASE
                 WHEN e.学校名=m.红方学校 THEN m.蓝方学校
                 WHEN e.学校名=m.蓝方学校 THEN m.红方学校
               END attacker,
               e.类别 category,SUM(ABS(e.数值)) damage
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='受击' AND e.类别 IN ('17mm','42mm')
        GROUP BY attacker,e.类别
        """
    ):
        if row["attacker"] in entries:
            dealt[(row["attacker"], row["category"])] = float(row["damage"] or 0)

    # Team/role position distributions in canonical red perspective.  A 0.5 m
    # grid is detailed enough for tactical goals without shipping raw tracks.
    goals: dict[tuple[str, str, int], list[list[float]]] = defaultdict(list)
    position_rows = db.execute(
        f"""
        WITH canonical AS (
          SELECT 学校名,机器人类型,
                 MIN({PHASES - 1},CAST(时刻秒/60 AS INT)) phase,
                 ROUND((CASE WHEN 阵营='蓝' THEN 28.0-x ELSE x END)*2)/2.0 qx,
                 ROUND((CASE WHEN 阵营='蓝' THEN 15.0-y ELSE y END)*2)/2.0 qy,
                 COUNT(*) samples
          FROM timeseries
          WHERE 学校名 IN ({placeholders})
            AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵','空中')
            AND 当前血量>0 AND x BETWEEN 0.05 AND 27.95 AND y BETWEEN 0.05 AND 14.95
          GROUP BY 学校名,机器人类型,phase,qx,qy
        ), ranked AS (
          SELECT *,ROW_NUMBER() OVER (
            PARTITION BY 学校名,机器人类型,phase ORDER BY samples DESC
          ) rank
          FROM canonical
        )
        SELECT * FROM ranked WHERE rank<=14
        ORDER BY 学校名,机器人类型,phase,rank
        """,
        schools,
    )
    for row in position_rows:
        goals[(row["学校名"], row["机器人类型"], int(row["phase"]))].append(
            [rounded(row["qx"]), rounded(row["qy"]), int(row["samples"])]
        )

    spawns: dict[tuple[str, str], list[float]] = {}
    for row in db.execute(
        f"""
        SELECT 学校名,机器人类型,
               AVG(CASE WHEN 阵营='蓝' THEN 28.0-x ELSE x END) x,
               AVG(CASE WHEN 阵营='蓝' THEN 15.0-y ELSE y END) y
        FROM timeseries
        WHERE 学校名 IN ({placeholders})
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵','空中')
          AND 时刻秒 BETWEEN 1 AND 3 AND x BETWEEN 0.05 AND 27.95 AND y BETWEEN 0.05 AND 14.95
        GROUP BY 学校名,机器人类型
        """,
        schools,
    ):
        spawns[(row["学校名"], row["机器人类型"])] = [rounded(row["x"]), rounded(row["y"])]

    hp_by_phase: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0] * PHASES)
    hp_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * PHASES)
    for row in db.execute(
        f"""
        SELECT 学校名,机器人类型,MIN({PHASES - 1},CAST(时刻秒/60 AS INT)) phase,
               AVG(最大血量) hp,COUNT(*) samples
        FROM timeseries
        WHERE 学校名 IN ({placeholders})
          AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵','空中')
          AND 最大血量>0
        GROUP BY 学校名,机器人类型,phase
        """,
        schools,
    ):
        key = (row["学校名"], row["机器人类型"])
        phase = int(row["phase"])
        hp_by_phase[key][phase] = float(row["hp"])
        hp_counts[key][phase] = int(row["samples"])

    # Mean observed moving speed.  Teleports, respawns and localization jumps
    # above 3.5 m/s are excluded.
    speeds: dict[tuple[str, str], float] = {}
    speed_query = f"""
        WITH track AS (
          SELECT 学校名,机器人类型,game_id,robot_id,时刻秒,x,y,
                 LAG(时刻秒) OVER w prev_t,LAG(x) OVER w prev_x,LAG(y) OVER w prev_y
          FROM timeseries
          WHERE 学校名 IN ({placeholders})
            AND 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵','空中')
            AND 当前血量>0 AND x BETWEEN 0.05 AND 27.95 AND y BETWEEN 0.05 AND 14.95
          WINDOW w AS (PARTITION BY game_id,robot_id ORDER BY 时刻秒)
        ), delta AS (
          SELECT 学校名,机器人类型,
                 SQRT((x-prev_x)*(x-prev_x)+(y-prev_y)*(y-prev_y))/(时刻秒-prev_t) speed
          FROM track WHERE 时刻秒-prev_t BETWEEN 0.5 AND 1.5
        )
        SELECT 学校名,机器人类型,AVG(speed) speed
        FROM delta WHERE speed BETWEEN 0.08 AND 3.5
        GROUP BY 学校名,机器人类型
    """
    for row in db.execute(speed_query, schools):
        speeds[(row["学校名"], row["机器人类型"])] = float(row["speed"] or 0)

    teams = {}
    for school in schools:
        entry = entries[school]
        team_shots = {
            category: sum(value for (name, _, cat), value in shots.items() if name == school and cat == category)
            for category in ("17mm", "42mm")
        }
        accuracy = {
            "17mm": clamp(dealt[(school, "17mm")] / 20 / max(1, team_shots["17mm"]), 0.025, 0.62),
            "42mm": clamp(dealt[(school, "42mm")] / 200 / max(1, team_shots["42mm"]), 0.025, 0.72),
        }
        role_payload = {}
        for role in ROLES:
            default = ROLE_DEFAULTS[role]
            weapon = default["weapon"]
            key = (school, role)
            phase_hp = []
            previous = default["hp"]
            for phase in range(PHASES):
                observed = hp_by_phase[key][phase]
                if hp_counts[key][phase] >= 10 and observed:
                    previous = max(previous, round(observed / 25) * 25)
                phase_hp.append(int(previous))
            level_by_minute = None
            if role == "英雄":
                ranged_hp = HERO_ARCHETYPES["ranged"]["hp_by_level"]
                level_by_minute = [
                    min(range(len(ranged_hp)), key=lambda index: abs(ranged_hp[index] - hp)) + 1
                    for hp in phase_hp
                ]
                level_by_minute[0] = 1
                for index in range(1, len(level_by_minute)):
                    level_by_minute[index] = max(level_by_minute[index - 1], level_by_minute[index])
                phase_hp = [ranged_hp[level - 1] for level in level_by_minute]
            empirical_speed = speeds.get(key, default["speed"])
            # Mean speed understates traversal speed because tracks include
            # aiming/holding; blend it with the role physical default.
            travel_speed = clamp(default["speed"] * 0.65 + empirical_speed * 0.9, 0.8, 3.6)
            role_shots = shots[(school, role, weapon)] if weapon else 0
            active = active_fire_seconds[(school, role, weapon)] if weapon else 0
            role_goals = []
            uav_profile = uav_navigation[school] if role == "空中" else None
            fallback = uav_profile["home"] if uav_profile else spawns.get(key, [2.4, 7.5])
            last = [[*fallback, 1]]
            for phase in range(PHASES):
                values = (uav_profile["airborne_goals_by_minute"][phase] if uav_profile
                          else goals.get((school, role, phase))) or last
                role_goals.append(values)
                last = values
            role_payload[role] = {
                # 无人机初始点必须是独立识别的停机坪，不再用含起飞后样本的
                # 通用“出生点中位数”，否则少数对局会从空中开始。
                "spawn": fallback if uav_profile else spawns.get(key, fallback),
                "speed_mps": rounded(travel_speed),
                "hp_by_minute": phase_hp,
                "weapon": weapon,
                "range_m": default["range"],
                "heat_limit": default["heat_limit"],
                "cooling_per_second": default["cooling"],
                "magazine": default["magazine"],
                "shots_per_game": rounded(role_shots / max(1, games[school]), 2),
                "burst_per_active_second": rounded(role_shots / max(1, active), 2),
                "goals_by_minute": role_goals,
            }
            if role == "英雄":
                role_payload[role].update({
                    "hero_archetype_default": "ranged",
                    "level_by_minute": level_by_minute,
                })
            elif role == "空中":
                role_payload[role]["uav_navigation"] = uav_profile
        aggregate = macro["teams"].get(school, {}).get("aggregate", {})
        teams[school] = {
            "team": entry.team,
            "stage": entry.stage,
            "region": entry.region,
            "games": games[school],
            "accuracy": {key: rounded(value, 4) for key, value in accuracy.items()},
            "dart_hits_per_game": aggregate.get("dart_hits_per_game", 0),
            "dart_gates_per_game": aggregate.get("dart_gates_per_game", 0),
            "radar_counters_per_game": rounded(radar_counters[school] / max(1, games[school]), 3),
            "uav_counters_received_per_game": rounded(uav_counters_received[school] / max(1, games[school]), 3),
            "economy_prior": economy_priors[school],
            "style": aggregate.get("style", "常规阵地运营"),
            "roles": role_payload,
        }
    db.close()

    payload = {
        "schema_version": 5,
        "kind": "agent_based_rmuc_2026_simulation_parameters",
        "ruleset": {
            "competition": "RoboMaster 2026 机甲大师超级对抗赛",
            "version": "V2.1.0",
            "released": "2026-07-16",
            "source_document": "RoboMaster 2026 机甲大师超级对抗赛比赛规则手册V2.1.0（20260716）.pdf",
        },
        "tick_seconds": 1,
        "duration_seconds": 420,
        "field_m": [28, 15],
        "roles": list(ROLES),
        "training_feature_schema": {
            "static": {
                "hero_archetype": {"type": "categorical", "values": ["ranged", "melee"], "default": "ranged"},
                "ruleset_version": {"type": "categorical", "value": "V2.1.0"},
            },
            "per_second": {
                "robot_level": {"type": "integer", "range": [1, 10]},
                "respawn_mode": {"type": "categorical", "values": ["none", "reading", "timed", "buyback"]},
                "respawn_progress": {"type": "number", "minimum": 0},
                "respawn_required": {"type": "number", "minimum": 0},
                "immediate_buyback_count": {"type": "integer", "minimum": 0},
                "radar_uav_counter_count": {"type": "integer", "range": [0, 5]},
                "radar_countered_seconds": {"type": "number", "range": [0, 45]},
                "uav_counter_buyout_count": {"type": "integer", "minimum": 0},
                "team_coins": {"type": "number", "minimum": 0},
                "technology_core_level": {"type": "integer", "range": [0, 4]},
                "technology_core_income_per_10": {"type": "number", "minimum": 0},
                "technology_core_next_level": {"type": ["integer", "null"], "range": [1, 4]},
                "uav_flight_state": {"type": "categorical", "values": ["parked", "airborne", "returning"]},
                "uav_support_active": {"type": "boolean"},
                "uav_support_seconds": {"type": "number", "minimum": 0},
                "uav_radar_weapon_locked": {"type": "boolean"},
                "terrain_action": {"type": ["categorical", "null"]},
                "terrain_speed_multiplier": {"type": "number", "range": [0.1, 1.25]},
            },
            "decision_labels": {
                "respawn_choice": ["timed_in_place", "immediate_buyback"],
                "uav_counter_choice": ["wait_45_seconds", "double_cost_buyout"],
            },
        },
        "structures": {
            "red": {"base": [2.66, 7.5], "outpost": [11.0, 3.25], "fortress": [6.65, 7.5]},
            "blue": {"base": [25.34, 7.5], "outpost": [17.0, 11.75], "fortress": [21.35, 7.5]},
        },
        "service_zones": {
            "red": {
                "supply": {"center": [1.8, 1.55], "radius": [1.65, 1.3], "ammo": True, "heal": True, "label": "补给区"},
                "base": {"center": [2.66, 7.5], "radius": [1.35, 1.35], "ammo": True, "heal": False, "label": "基地区"},
                "outpost": {"center": [11.0, 3.25], "radius": [1.15, 1.0], "ammo": True, "heal": False, "label": "前哨站下"},
            },
            "blue": {
                "supply": {"center": [26.2, 13.45], "radius": [1.65, 1.3], "ammo": True, "heal": True, "label": "补给区"},
                "base": {"center": [25.34, 7.5], "radius": [1.35, 1.35], "ammo": True, "heal": False, "label": "基地区"},
                "outpost": {"center": [17.0, 11.75], "radius": [1.15, 1.0], "ammo": True, "heal": False, "label": "前哨站下"},
            },
        },
        "assembly_zones": {
            "red": {
                "center": [13.0, 8.5], "radius": [1.25, 1.1], "label": "红方装配区",
                "entry_outside": [9.0, 7.56], "entry_inside": [9.78, 7.56],
            },
            "blue": {
                "center": [15.0, 6.5], "radius": [1.25, 1.1], "label": "蓝方装配区",
                "entry_outside": [19.0, 7.45], "entry_inside": [18.22, 7.45],
            },
        },
        "rules": {
            "initial_coins": 400,
            "initial_allowed_ammo": {"英雄": 0, "步兵3": 0, "步兵4": 0, "哨兵": 300, "空中": 750},
            "automatic_income": [[61, 50], [121, 50], [181, 50], [241, 50], [301, 50], [361, 150]],
            "base_hp": 5000,
            "outpost_hp": 1500,
            "damage": {"17mm": 20, "42mm": 200, "dart": 400},
            "heat_per_shot": {"17mm": 10, "42mm": 100},
            "hero_archetypes": HERO_ARCHETYPES,
            "hero_default_archetype": "ranged",
            "heal_ratio_per_second": 0.1,
            "late_heal_ratio_per_second": 0.25,
            "late_heal_start_second": 240,
            "out_of_combat_seconds": 6,
            "respawn": {
                "read_base": 10,
                "elapsed_seconds_divisor": 10,
                "buyback_read_penalty": 20,
                "normal_progress_per_second": 1,
                "fast_progress_per_second": 4,
                "fast_base_hp_below": 2000,
                "timed_hp_ratio": 0.1,
                "timed_invulnerable_seconds": 30,
                "minimum_invulnerable_after_zone_seconds": 10,
                "buyback_hp_ratio": 1.0,
                "buyback_weak_seconds": 3,
                "buyback_invulnerable_seconds": 3,
                "buyback_chassis_boost_seconds": 4,
                "buyback_minute_cost": 80,
                "buyback_level_cost": 20,
            },
            "radar_uav_counter": {
                "max_uses": 5,
                "lock_seconds": 45,
                "buyout_from_use": 4,
                "buyout_cost_multiplier": 2,
            },
            "uav_support": {
                "initial_seconds": 30,
                "periodic_seconds": 20,
                "periodic_interval_seconds": 60,
                "paid_cost_per_second": 1,
                "maximum_ammunition": 750,
                "ordinary_damage": False,
                "healing_and_respawn": False,
                "can_occupy_buff_points": False,
            },
            "technology_core": {
                "maximum_level": 4,
                "unlock_seconds": list(CORE_UNLOCK_SECONDS),
                "income_interval_seconds": 10,
                "first_income_per_10": list(CORE_FIRST_INCOME_PER_10),
                "repeat_income_per_10": [5, 10, 15, 0],
                "defense_ratio_by_level": [0, 0, 0.25, 0.5],
                "robot_level_cap_by_level": [5, 7, 10, 10],
                "level_four_base_hp_gain": 2000,
                "level_four_timeout_seconds": 45,
            },
            "ammo_cost": {"17mm": {"coins": 10, "rounds": 10}, "42mm": {"coins": 10, "rounds": 1}},
        },
        "limitations": [
            "remaining ammunition and explicit supply completion are not present in the referee export",
            "17mm attacker identity is estimated because hit events only identify the victim",
            "goal points are empirical 0.5 m position-density modes and are routed through team-role terrain capabilities",
            "UAV helipad and airborne samples are separated; airborne goals are game-normalized and connected by empirical five-second transitions rather than independent dwell-point sampling",
            "terrain traversal changes speed for both ascent and descent using configurable coarse priors; these priors model alignment and landing time and should be replaced by team-role distributions when enough national telemetry is available",
            "national matches always start at 400 coins; regional initial-coin ratings are retained only as descriptive telemetry and are not used by the simulator",
            "technology-core completion priors are inferred from persistent ten-second increases in regional total-coins telemetry because the export has no explicit assembly event",
            "V2.1.0 hero archetype and buyback choices were not present in regional telemetry; those features are rule-conditioned simulation inputs until national-match samples are ingested",
        ],
        "teams": teams,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(teams)} teams to {options.output} ({options.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
