#!/usr/bin/env python3
"""Build evidence-backed style profiles for the 2026 RMUC advancing teams.

The report deliberately distinguishes "not observed" from "cannot traverse".
Terrain evidence is a conservative proxy: a ground robot must remain for at
least three consecutive seconds on the elevated part of the central highland,
using both XY location and type-relative Z height.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "outputs"

FIELD_W = 28.0
FIELD_H = 15.0
MOBILE_TYPES = ("英雄", "工程", "步兵3", "步兵4", "哨兵", "空中")
GROUND_TYPES = ("英雄", "工程", "步兵3", "步兵4", "哨兵")

# Initial semantic approximation from the official top-down map.  Z separates
# robots on top of the highland from robots passing beneath/alongside it.
CENTRAL_HIGHLAND = (
    (9.4, 4.2),
    (10.8, 2.6),
    (16.0, 2.6),
    (18.6, 4.7),
    (18.6, 10.2),
    (17.0, 12.1),
    (11.4, 12.1),
    (9.4, 10.1),
)


@dataclass(frozen=True)
class TeamEntry:
    school: str
    team: str
    stage: str
    region: str


DIRECT = (
    TeamEntry("华南农业大学", "Taurus", "全国赛直通", "南部赛区"),
    TeamEntry("五邑大学", "IMCA", "全国赛直通", "南部赛区"),
    TeamEntry("广州城市理工学院", "野狼", "全国赛直通", "南部赛区"),
    TeamEntry("东莞理工学院", "ACE", "全国赛直通", "南部赛区"),
    TeamEntry("电子科技大学中山学院", "RoboBraver", "全国赛直通", "南部赛区"),
    TeamEntry("华南理工大学", "华南虎", "全国赛直通", "南部赛区"),
    TeamEntry("上海交通大学", "交龙", "全国赛直通", "南部赛区"),
    TeamEntry("仲恺农业工程学院", "奇点", "全国赛直通", "南部赛区"),
    TeamEntry("深圳大学", "RobotPilots", "全国赛直通", "南部赛区"),
    TeamEntry("武汉工程大学", "Nautilus", "全国赛直通", "南部赛区"),
    TeamEntry("中国石油大学（华东）", "RPS", "全国赛直通", "东部赛区"),
    TeamEntry("山东科技大学", "SmartRobot", "全国赛直通", "东部赛区"),
    TeamEntry("合肥工业大学（宣城校区）", "WDR", "全国赛直通", "东部赛区"),
    TeamEntry("浙江大学", "Hello World", "全国赛直通", "东部赛区"),
    TeamEntry("北京理工大学（珠海）", "毅恒", "全国赛直通", "东部赛区"),
    TeamEntry("哈尔滨工业大学（深圳）", "南工骁鹰", "全国赛直通", "东部赛区"),
    TeamEntry("华东理工大学", "起源", "全国赛直通", "东部赛区"),
    TeamEntry("南京理工大学", "Alliance", "全国赛直通", "东部赛区"),
    TeamEntry("东北大学", "TDT", "全国赛直通", "北部赛区"),
    TeamEntry("哈尔滨工业大学", "I Hiter", "全国赛直通", "北部赛区"),
    TeamEntry("哈尔滨工业大学（威海）", "HERO", "全国赛直通", "北部赛区"),
    TeamEntry("西北工业大学", "WMJ", "全国赛直通", "北部赛区"),
    TeamEntry("北京理工大学", "追梦", "全国赛直通", "北部赛区"),
    TeamEntry("复旦大学", "星云EGA", "全国赛直通", "北部赛区"),
    TeamEntry("同济大学", "SuperPower", "全国赛直通", "北部赛区"),
    TeamEntry("西安交通大学", "笃行", "全国赛直通", "北部赛区"),
    TeamEntry("大连理工大学", "凌BUG", "全国赛直通", "北部赛区"),
    TeamEntry("长安大学", "VGD", "全国赛直通", "北部赛区"),
)

REVIVAL = (
    TeamEntry("广东工业大学", "DynamicX", "复活赛", "南部赛区"),
    TeamEntry("桂林电子科技大学", "Evolution", "复活赛", "南部赛区"),
    TeamEntry("华中科技大学", "狼牙", "复活赛", "南部赛区"),
    TeamEntry("南华大学", "MA", "复活赛", "南部赛区"),
    TeamEntry("厦门大学嘉庚学院", "TCR", "复活赛", "南部赛区"),
    TeamEntry("深圳技术大学", "悍匠", "复活赛", "南部赛区"),
    TeamEntry("江南大学霞客湾校区", "SHARK", "复活赛", "东部赛区"),
    TeamEntry("南方科技大学", "ARTINX", "复活赛", "东部赛区"),
    TeamEntry("南京航空航天大学金城学院", "Born of Fire", "复活赛", "东部赛区"),
    TeamEntry("太原科技大学", "NewMaker", "复活赛", "东部赛区"),
    TeamEntry("中国科学技术大学", "RoboWalker", "复活赛", "东部赛区"),
    TeamEntry("中国矿业大学", "CUBOT", "复活赛", "东部赛区"),
    TeamEntry("山东理工大学", "齐奇", "复活赛", "北部赛区"),
    TeamEntry("沈阳理工大学", "Ambition", "复活赛", "北部赛区"),
    TeamEntry("西安电子科技大学", "IRobot", "复活赛", "北部赛区"),
    TeamEntry("应急管理大学", "风暴", "复活赛", "北部赛区"),
)

TEAMS = DIRECT + REVIVAL
ENTRY_BY_SCHOOL = {entry.school: entry for entry in TEAMS}


@dataclass
class TeamStats:
    games: int = 0
    wins: int = 0
    base_damage: Counter = field(default_factory=Counter)
    base_damage_received: Counter = field(default_factory=Counter)
    outpost_damage: Counter = field(default_factory=Counter)
    base_damage_games: set[int] = field(default_factory=set)
    base_first_hit_seconds: list[float] = field(default_factory=list)
    shots: Counter = field(default_factory=Counter)
    position_count: int = 0
    enemy_half_count: int = 0
    deep_count: int = 0
    early_enemy_half_count: int = 0
    early_position_count: int = 0
    x_sum: float = 0.0
    distance_m: float = 0.0
    role_count: Counter = field(default_factory=Counter)
    role_x_sum: Counter = field(default_factory=Counter)
    role_enemy_half: Counter = field(default_factory=Counter)
    role_deep: Counter = field(default_factory=Counter)
    role_distance: Counter = field(default_factory=Counter)
    terrain_runs: Counter = field(default_factory=Counter)
    terrain_seconds: Counter = field(default_factory=Counter)
    terrain_games: set[int] = field(default_factory=set)


def point_in_polygon(x: float, y: float, polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    previous = len(polygon) - 1
    for current, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[previous]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        previous = current
    return inside


def canonical_xy(side: str, x: float, y: float) -> tuple[float, float]:
    if side == "蓝":
        return FIELD_W - x, FIELD_H - y
    return x, y


def opponent_school(match: sqlite3.Row, school: str) -> str | None:
    if school == match["红方学校"]:
        return match["蓝方学校"]
    if school == match["蓝方学校"]:
        return match["红方学校"]
    return None


def safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def style_labels(rows: list[dict]) -> None:
    enemy_values = [row["enemy_half_rate"] for row in rows]
    deep_values = [row["deep_rate"] for row in rows]
    terrain_values = [row["terrain_seconds_per_game"] for row in rows]
    sentry_values = [row["sentry_mean_x"] for row in rows]
    outpost_values = [row["outpost_damage_per_game"] for row in rows]
    base_values = [row["base_damage_per_game"] for row in rows]
    cuts = {
        "enemy_low": percentile(enemy_values, 0.30),
        "enemy_high": percentile(enemy_values, 0.70),
        "deep_high": percentile(deep_values, 0.70),
        "terrain_high": percentile(terrain_values, 0.70),
        "sentry_high": percentile(sentry_values, 0.70),
        "outpost_high": percentile(outpost_values, 0.70),
        "base_mid": percentile(base_values, 0.50),
    }

    for row in rows:
        labels: list[str] = []
        if row["deep_rate"] >= cuts["deep_high"]:
            labels.append("多兵种深压")
        elif row["enemy_half_rate"] >= cuts["enemy_high"]:
            labels.append("中线前压")
        elif row["enemy_half_rate"] <= cuts["enemy_low"]:
            labels.append("稳守反击")

        if row["sentry_mean_x"] >= cuts["sentry_high"]:
            labels.append("哨兵前压")
        if row["outpost_damage_per_game"] >= cuts["outpost_high"]:
            labels.append("重前哨压制")
        if row["terrain_seconds_per_game"] >= cuts["terrain_high"]:
            labels.append("高地转点活跃")

        if row["base_damage_per_game"] >= cuts["base_mid"] and row["base_damage_total"]:
            shares = {
                "17mm持续磨基": row["base_17_share"],
                "英雄42mm炮击": row["base_42_share"],
                "飞镖磨基": row["base_dart_share"],
            }
            base_label = max(shares, key=shares.get)
            if shares[base_label] >= 0.45:
                labels.append(base_label)
            else:
                labels.append("混合磨基")
        row["style"] = "、".join(labels[:4]) or "均衡/样本不足"


def analyze(db_path: Path) -> tuple[list[dict], dict]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    schools = set(ENTRY_BY_SCHOOL)
    stats = {school: TeamStats() for school in schools}
    matches = {
        int(row["game_id"]): row
        for row in connection.execute(
            "SELECT game_id,赛区,红方学校,蓝方学校,胜方,时长秒 FROM matches"
        )
    }

    for game_id, match in matches.items():
        for side_name, winner_name in (("红方学校", "红"), ("蓝方学校", "蓝")):
            school = match[side_name]
            if school not in schools:
                continue
            stats[school].games += 1
            if match["胜方"] == winner_name:
                stats[school].wins += 1

    first_hit_by_game_team: dict[tuple[int, str], float] = {}
    event_sql = """
        SELECT game_id,时刻秒,事件类型,机器人类型,学校名,目标类型,类别,数值
        FROM events
        WHERE 事件类型 IN ('受击','发弹','飞镖命中')
        ORDER BY game_id,时刻秒
    """
    for event in connection.execute(event_sql):
        game_id = int(event["game_id"])
        match = matches.get(game_id)
        if match is None:
            continue
        event_type = event["事件类型"]
        school = event["学校名"]
        if event_type == "发弹":
            if school in schools:
                stats[school].shots[event["机器人类型"] or "未知"] += 1
            continue
        if event_type != "受击" or school is None:
            continue
        attacker = opponent_school(match, school)
        if attacker is None:
            continue
        victim_type = event["机器人类型"]
        category = event["类别"] or "未知"
        damage = abs(float(event["数值"] or 0.0))
        if victim_type == "基地":
            if attacker in schools:
                team_stats = stats[attacker]
                team_stats.base_damage[category] += damage
                team_stats.base_damage_games.add(game_id)
                key = (game_id, attacker)
                first_hit_by_game_team.setdefault(key, float(event["时刻秒"] or 0.0))
            if school in schools:
                stats[school].base_damage_received[category] += damage
        elif victim_type == "前哨站" and attacker in schools:
            stats[attacker].outpost_damage[category] += damage

    for (_game_id, school), second in first_hit_by_game_team.items():
        stats[school].base_first_hit_seconds.append(second)

    # A flat-ground type baseline learned from the first 20 seconds of all games.
    z_samples: dict[str, list[float]] = defaultdict(list)
    for robot_type, z in connection.execute(
        """
        SELECT 机器人类型,z FROM timeseries
        WHERE 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵')
          AND 时刻秒<=20 AND z BETWEEN -2 AND 2
        """
    ):
        z_samples[robot_type].append(float(z))
    z_baseline = {robot_type: median(values) for robot_type, values in z_samples.items()}

    last_position: dict[tuple[int, str, int], tuple[float, float, float, str]] = {}
    terrain_run: dict[tuple[int, str, int], tuple[int, str]] = {}

    def finish_run(key: tuple[int, str, int]) -> None:
        run = terrain_run.pop(key, None)
        if run is None:
            return
        seconds, robot_type = run
        if seconds >= 3:
            game_id, school, _robot_id = key
            stats[school].terrain_runs[robot_type] += 1
            stats[school].terrain_seconds[robot_type] += seconds
            stats[school].terrain_games.add(game_id)

    timeseries_sql = """
        SELECT game_id,时刻秒,robot_id,机器人类型,阵营,学校名,x,y,z
        FROM timeseries
        WHERE 机器人类型 IN ('英雄','工程','步兵3','步兵4','哨兵','空中')
        ORDER BY game_id,学校名,robot_id,时刻秒
    """
    previous_game = None
    for state in connection.execute(timeseries_sql):
        game_id = int(state["game_id"])
        if previous_game is not None and game_id != previous_game:
            for key in [key for key in terrain_run if key[0] == previous_game]:
                finish_run(key)
        previous_game = game_id
        school = state["学校名"]
        if school not in schools:
            continue
        x, y = state["x"], state["y"]
        if x is None or y is None or not (0 <= x <= FIELD_W and 0 <= y <= FIELD_H):
            continue
        x, y = canonical_xy(state["阵营"], float(x), float(y))
        robot_type = state["机器人类型"]
        second = float(state["时刻秒"])
        team_stats = stats[school]
        team_stats.position_count += 1
        team_stats.x_sum += x
        team_stats.role_count[robot_type] += 1
        team_stats.role_x_sum[robot_type] += x
        if x >= FIELD_W / 2:
            team_stats.enemy_half_count += 1
            team_stats.role_enemy_half[robot_type] += 1
        if x >= 21.0:
            team_stats.deep_count += 1
            team_stats.role_deep[robot_type] += 1
        if second <= 90:
            team_stats.early_position_count += 1
            if x >= FIELD_W / 2:
                team_stats.early_enemy_half_count += 1

        key = (game_id, school, int(state["robot_id"]))
        previous = last_position.get(key)
        if previous is not None and second - previous[0] == 1:
            distance = math.hypot(x - previous[1], y - previous[2])
            if distance <= 8.0:
                team_stats.distance_m += distance
                team_stats.role_distance[robot_type] += distance
        last_position[key] = (second, x, y, robot_type)

        elevated = False
        if robot_type in GROUND_TYPES and state["z"] is not None:
            z = float(state["z"])
            elevated = (
                -2.0 <= z <= 2.0
                and point_in_polygon(x, y, CENTRAL_HIGHLAND)
                and z >= z_baseline[robot_type] + 0.25
            )
        if elevated:
            run_seconds, _ = terrain_run.get(key, (0, robot_type))
            terrain_run[key] = (run_seconds + 1, robot_type)
        else:
            finish_run(key)

    for key in list(terrain_run):
        finish_run(key)
    connection.close()

    rows: list[dict] = []
    for entry in TEAMS:
        team_stats = stats[entry.school]
        games = max(1, team_stats.games)
        base_17 = float(team_stats.base_damage["17mm"])
        base_42 = float(team_stats.base_damage["42mm"])
        base_dart = float(team_stats.base_damage["飞镖"])
        base_total = base_17 + base_42 + base_dart
        terrain_runs_total = sum(team_stats.terrain_runs.values())
        terrain_roles = [
            robot_type
            for robot_type in GROUND_TYPES
            if team_stats.terrain_seconds[robot_type] >= 30
        ]
        terrain_seconds_total = sum(team_stats.terrain_seconds.values())
        if len(team_stats.terrain_games) >= 2 and terrain_roles:
            terrain_status = "中央高地已观察"
        elif terrain_runs_total:
            terrain_status = "中央高地有迹象"
        else:
            terrain_status = "样本中未观察到"
        role_mean_x = {
            robot_type: safe_rate(
                team_stats.role_x_sum[robot_type], team_stats.role_count[robot_type]
            )
            for robot_type in MOBILE_TYPES
        }
        row = {
            "stage": entry.stage,
            "region": entry.region,
            "school": entry.school,
            "team": entry.team,
            "games": team_stats.games,
            "wins": team_stats.wins,
            "win_rate": safe_rate(team_stats.wins, team_stats.games),
            "base_damage_total": base_total,
            "base_damage_per_game": base_total / games,
            "base_17_share": safe_rate(base_17, base_total),
            "base_42_share": safe_rate(base_42, base_total),
            "base_dart_share": safe_rate(base_dart, base_total),
            "base_damage_game_rate": safe_rate(len(team_stats.base_damage_games), team_stats.games),
            "median_first_base_hit_s": (
                median(team_stats.base_first_hit_seconds)
                if team_stats.base_first_hit_seconds
                else None
            ),
            "base_damage_received_per_game": sum(team_stats.base_damage_received.values()) / games,
            "outpost_damage_per_game": sum(team_stats.outpost_damage.values()) / games,
            "shots_per_game": sum(team_stats.shots.values()) / games,
            "hero_shots_per_game": team_stats.shots["英雄"] / games,
            "infantry_shots_per_game": (
                team_stats.shots["步兵3"] + team_stats.shots["步兵4"]
            )
            / games,
            "sentry_shots_per_game": team_stats.shots["哨兵"] / games,
            "enemy_half_rate": safe_rate(team_stats.enemy_half_count, team_stats.position_count),
            "deep_rate": safe_rate(team_stats.deep_count, team_stats.position_count),
            "early_enemy_half_rate": safe_rate(
                team_stats.early_enemy_half_count, team_stats.early_position_count
            ),
            "mean_x": safe_rate(team_stats.x_sum, team_stats.position_count),
            "distance_per_game_m": team_stats.distance_m / games,
            "sentry_mean_x": role_mean_x["哨兵"],
            "hero_mean_x": role_mean_x["英雄"],
            "engineer_mean_x": role_mean_x["工程"],
            "terrain_status": terrain_status,
            "terrain_game_rate": safe_rate(len(team_stats.terrain_games), team_stats.games),
            "terrain_games": len(team_stats.terrain_games),
            "terrain_runs": terrain_runs_total,
            "terrain_seconds_per_game": terrain_seconds_total / games,
            "terrain_roles": "、".join(terrain_roles) if terrain_roles else "—",
        }
        rows.append(row)

    style_labels(rows)
    summary = build_summary(rows, z_baseline)
    return rows, summary


def top_rows(rows: list[dict], key: str, count: int = 8) -> list[dict]:
    return [
        {
            "school": row["school"],
            "team": row["team"],
            "stage": row["stage"],
            key: round(float(row[key]), 4),
        }
        for row in sorted(rows, key=lambda row: row[key], reverse=True)[:count]
    ]


def build_summary(rows: list[dict], z_baseline: dict[str, float]) -> dict:
    direct = [row for row in rows if row["stage"] == "全国赛直通"]
    revival = [row for row in rows if row["stage"] == "复活赛"]
    confirmed = [row for row in rows if row["terrain_status"] == "中央高地已观察"]
    def cohort_metrics(cohort: list[dict]) -> dict:
        games = sum(row["games"] for row in cohort)
        base_damage = sum(row["base_damage_total"] for row in cohort)
        category_damage = {
            "17mm": sum(row["base_damage_total"] * row["base_17_share"] for row in cohort),
            "42mm": sum(row["base_damage_total"] * row["base_42_share"] for row in cohort),
            "飞镖": sum(row["base_damage_total"] * row["base_dart_share"] for row in cohort),
        }
        return {
            "teams": len(cohort),
            "games": games,
            "base_damage_per_game": round(safe_rate(base_damage, games), 4),
            "base_damage_mix": {
                key: round(safe_rate(value, base_damage), 4)
                for key, value in category_damage.items()
            },
            "mean_enemy_half_rate": round(float(np.mean([row["enemy_half_rate"] for row in cohort])), 4),
            "mean_deep_rate": round(float(np.mean([row["deep_rate"] for row in cohort])), 4),
            "outpost_damage_per_game": round(
                safe_rate(sum(row["outpost_damage_per_game"] * row["games"] for row in cohort), games),
                4,
            ),
        }

    return {
        "methodology": {
            "team_count": len(rows),
            "direct_count": len(direct),
            "revival_count": len(revival),
            "central_highland_polygon_m": CENTRAL_HIGHLAND,
            "z_flat_baseline_by_type": z_baseline,
            "terrain_confirmation": "中央高地范围内、比兵种开局平地Z基线高0.25m、连续至少3秒；至少2局且某兵种累计高位活动30秒，记为中央高地已观察",
            "limitation": "样本中未观察到不等于机器人不能跨越；17mm基地伤害不能由裁判数据唯一归因到步兵/哨兵/空中。",
        },
        "cohorts": {
            "direct_win_rate": round(float(np.mean([row["win_rate"] for row in direct])), 4),
            "revival_win_rate": round(float(np.mean([row["win_rate"] for row in revival])), 4),
            "confirmed_terrain_teams": len(confirmed),
            "direct": cohort_metrics(direct),
            "revival": cohort_metrics(revival),
        },
        "leaders": {
            "base_damage_per_game": top_rows(rows, "base_damage_per_game"),
            "outpost_damage_per_game": top_rows(rows, "outpost_damage_per_game"),
            "deep_rate": top_rows(rows, "deep_rate"),
            "terrain_seconds_per_game": top_rows(rows, "terrain_seconds_per_game"),
            "sentry_mean_x": top_rows(rows, "sentry_mean_x"),
            "base_dart_share": top_rows(
                [row for row in rows if row["base_damage_total"] >= 500],
                "base_dart_share",
            ),
            "base_42_share": top_rows(
                [row for row in rows if row["base_damage_total"] >= 500],
                "base_42_share",
            ),
        },
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(rows: list[dict], summary: dict) -> str:
    lines = [
        "# RMUC 2026 全国赛直通与复活赛队伍打法初析",
        "",
        "本报告覆盖官方名单中的 28 支全国赛直通队和 16 支复活赛队，仅使用当前 SQLite 中的区域赛对局。打法标签是可复算的数据画像，不是对队伍能力的绝对定性。",
        "",
        "## 数据与规则来源",
        "",
        "- 队伍名单：[RMUC 2026 机甲大师超级对抗赛区域赛获奖名单（官方）](https://www.robomaster.com/zh-CN/resource/pages/announcement/1919)。",
        "- 名额口径：[全国赛 28 支、复活赛 16 支名额分配（官方）](https://www.robomaster.com/zh-CN/resource/pages/announcement/1910)。",
        "- 场地构成：[2026 超级对抗赛比赛场地最终版公示（官方社区）](https://bbs.robomaster.com/article/1884488?source=9)。",
        "- 原始对局数据：本项目 SQLite；本报告未引入公开视频主观打分。",
        "",
        "## 口径",
        "",
        "- 进攻方向统一为己方 `x=0` 到敌方 `x=28`；蓝方轨迹旋转 180° 后统计。",
        "- 基地磨血按基地实际受击事件拆为 17 mm、42 mm、飞镖；只能可靠归因到进攻队和弹种。",
        "- `x≥14` 视为进入敌方半场，`x≥21` 作为深压代理。",
        "- 地形证据使用中央高地 XY 多边形和 Z 高度联合判断。`样本中未观察到` 不等于不能跨越。",
        "",
        "## 总览",
        "",
        f"- 直通队平均局胜率：{percent(summary['cohorts']['direct_win_rate'])}",
        f"- 复活赛队平均局胜率：{percent(summary['cohorts']['revival_win_rate'])}",
        f"- 中央高地连续高位活动达到保守观察标准：{summary['cohorts']['confirmed_terrain_teams']}/44 队（仅表示至少一种地面兵种实际出现过，不代表所有车型都能跨越所有地形）。",
        f"- 直通队加权基地伤害：{summary['cohorts']['direct']['base_damage_per_game']:.0f}/局；复活赛队：{summary['cohorts']['revival']['base_damage_per_game']:.0f}/局。",
        f"- 直通队基地伤害结构（17/42/镖）：{percent(summary['cohorts']['direct']['base_damage_mix']['17mm'])}/{percent(summary['cohorts']['direct']['base_damage_mix']['42mm'])}/{percent(summary['cohorts']['direct']['base_damage_mix']['飞镖'])}；复活赛队为 {percent(summary['cohorts']['revival']['base_damage_mix']['17mm'])}/{percent(summary['cohorts']['revival']['base_damage_mix']['42mm'])}/{percent(summary['cohorts']['revival']['base_damage_mix']['飞镖'])}。",
        "",
        "## 逐队画像",
        "",
        "| 阶段 | 赛区 | 队伍 | 局数 | 胜率 | 主流打法代理 | 基地伤害/局 | 17/42/镖 | 敌半场 | 深压 | 地形证据 | 证据兵种 |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        damage_mix = "/".join(
            percent(row[key])
            for key in ("base_17_share", "base_42_share", "base_dart_share")
        )
        lines.append(
            "| {stage} | {region} | {school}·{team} | {games} | {win_rate} | "
            "{style} | {base:.0f} | {mix} | {enemy} | {deep} | {terrain} | {roles} |".format(
                stage=row["stage"],
                region=row["region"].removesuffix("赛区"),
                school=row["school"],
                team=row["team"],
                games=row["games"],
                win_rate=percent(row["win_rate"]),
                style=row["style"],
                base=row["base_damage_per_game"],
                mix=damage_mix,
                enemy=percent(row["enemy_half_rate"]),
                deep=percent(row["deep_rate"]),
                terrain=row["terrain_status"],
                roles=row["terrain_roles"],
            )
        )

    lines.extend(["", "## 指标领先队伍", ""])
    titles = {
        "base_damage_per_game": "基地伤害/局",
        "outpost_damage_per_game": "前哨伤害/局",
        "deep_rate": "深压占比",
        "terrain_seconds_per_game": "中央高地高位活动秒数/局",
        "sentry_mean_x": "哨兵平均推进纵深",
        "base_dart_share": "飞镖磨基占比（累计基地伤害≥500）",
        "base_42_share": "42 mm磨基占比（累计基地伤害≥500）",
    }
    for key, title in titles.items():
        values = summary["leaders"][key]
        formatted = "、".join(
            f"{item['school']}·{item['team']} ({item[key]:.3f})" for item in values
        )
        lines.append(f"- {title}：{formatted}")

    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "- 没有操作手指令、目标点和完整雷达视野，因此这里描述的是实际行为风格，不是队伍的全部战术意图。",
            "- 17 mm 基地受击记录不带攻击者 robot_id，不能仅凭裁判事件唯一拆成步兵、哨兵或空中机器人贡献。",
            "- 当前地形仅保守确认中央高地；飞坡、公路、隧道、梯形高地需要下一版语义多边形与通道方向标注。",
            "- 每队只有 6–22 局，极端对手、红蓝方和淘汰赛策略会影响均值。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "team_profiles_44.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "team_profiles_44.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "team_profiles_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "team_profiles_44.md").write_text(
        render_markdown(rows, summary), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db.is_file():
        raise SystemExit(f"database not found: {args.db}")
    rows, summary = analyze(args.db.resolve())
    write_outputs(rows, summary, args.output_dir.resolve())
    print(f"analyzed {len(rows)} teams")
    print(f"report: {args.output_dir / 'team_profiles_44.md'}")
    print(f"csv: {args.output_dir / 'team_profiles_44.csv'}")


if __name__ == "__main__":
    main()
