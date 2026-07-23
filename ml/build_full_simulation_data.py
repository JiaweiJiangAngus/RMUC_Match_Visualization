#!/usr/bin/env python3
"""Build per-team/per-role parameters for the agent-based browser simulator.

The referee export has exact tracks, HP, heat and fired-projectile counters but
does not expose remaining ammunition or the attacker of a hit.  This builder
therefore keeps movement/firing parameters empirical and estimates team weapon
accuracy as detected hits divided by projectiles fired.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
DEFAULT_GAMES_DIR = ROOT / "docs" / "data" / "games"
DEFAULT_BEHAVIOR_LABELS = ROOT / "analysis" / "manual_team_behavior_labels.csv"
ROLES = ("英雄", "工程", "步兵3", "步兵4", "哨兵", "空中")
GROUND_ROLES = ROLES[:-1]
PHASES = 7
TARGET_PHASES = 14
BASE_DAMAGE_BIN_SECONDS = 15
BASE_DAMAGE_BINS = 420 // BASE_DAMAGE_BIN_SECONDS
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
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES_DIR)
    parser.add_argument("--behavior-labels", type=Path, default=DEFAULT_BEHAVIOR_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rounded(value: float, digits: int = 3) -> int | float:
    result = float(round(float(value), digits))
    return int(result) if result.is_integer() else result


def load_behavior_labels(path: Path) -> dict[tuple[str, str, str], dict]:
    """Load explicit supervised labels without hiding team names in the engine."""
    labels = {}
    if not path.exists():
        return labels
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["school"].strip(), row["role"].strip(), row["feature"].strip())
            labels[key] = {
                "value": row["value"].strip(),
                "label": row["label"].strip(),
                "source": row["source"].strip(),
                "note": row["note"].strip(),
            }
    return labels


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


def inside_ellipse(point: tuple[float, float], center: tuple[float, float], radius: tuple[float, float]) -> bool:
    dx = (point[0] - center[0]) / radius[0]
    dy = (point[1] - center[1]) / radius[1]
    return dx * dx + dy * dy <= 1


def stationary_service_weight(
    point: tuple[float, float],
    previous: tuple[float, float] | None,
    following: tuple[float, float] | None,
) -> float:
    """Down-weight stationary service dwell without deleting real base defence."""
    neighbours = [candidate for candidate in (previous, following) if candidate is not None]
    if not neighbours or max(((point[0] - value[0]) ** 2 + (point[1] - value[1]) ** 2) ** 0.5 for value in neighbours) > 0.4:
        return 1.0
    if inside_ellipse(point, (1.8, 1.55), (1.65, 1.3)):
        return 0.08
    if inside_ellipse(point, (2.66, 7.5), (1.35, 1.35)):
        return 0.3
    return 1.0


def build_ground_navigation(
    games_dir: Path,
    schools: tuple[str, ...],
) -> tuple[dict[tuple[str, str, int], list[list[float]]], dict[tuple[str, str, int], list[list[float]]]]:
    """Build per-game-balanced ground goals and five-second movement transitions."""
    allowed = set(schools)
    goal_weights: dict[tuple[str, str, int], Counter] = defaultdict(Counter)
    transition_weights: dict[tuple[str, str, int], Counter] = defaultdict(Counter)

    for path in sorted(games_dir.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            game = json.load(handle)
        info = game.get("info", {})
        side_school = {"红": info.get("red"), "蓝": info.get("blue")}
        tracks: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
        for second_text, rows in game.get("frames", {}).items():
            second = int(float(second_text))
            for row in rows:
                if len(row) < 7 or row[1] not in GROUND_ROLES or float(row[3] or 0) <= 0:
                    continue
                school = side_school.get(row[2])
                if school not in allowed or row[5] is None or row[6] is None:
                    continue
                x, y = float(row[5]), float(row[6])
                if row[2] == "蓝":
                    x, y = 28.0 - x, 15.0 - y
                if not (0.05 <= x <= 27.95 and 0.05 <= y <= 14.95):
                    continue
                tracks[(school, row[1])].append((second, x, y))

        for (school, role), values in tracks.items():
            values.sort()
            by_second = {second: (x, y) for second, x, y in values}
            game_goals: dict[int, Counter] = defaultdict(Counter)
            game_transitions: dict[int, Counter] = defaultdict(Counter)
            for index, (second, x, y) in enumerate(values):
                phase = min(PHASES - 1, second // 60)
                point = (x, y)
                previous = (values[index - 1][1], values[index - 1][2]) if index and second - values[index - 1][0] <= 2 else None
                following = (values[index + 1][1], values[index + 1][2]) if index + 1 < len(values) and values[index + 1][0] - second <= 2 else None
                cell = (round(x * 2) / 2, round(y * 2) / 2)
                game_goals[phase][cell] += stationary_service_weight(point, previous, following)

                target = by_second.get(second + 5)
                if target is None:
                    continue
                distance = ((target[0] - x) ** 2 + (target[1] - y) ** 2) ** 0.5
                if not (0.25 <= distance <= 18.0):
                    continue
                target_cell = (round(target[0] * 2) / 2, round(target[1] * 2) / 2)
                game_transitions[phase][(*cell, *target_cell)] += 1

            for phase, counts in game_goals.items():
                total = sum(counts.values()) or 1
                for cell, count in counts.items():
                    goal_weights[(school, role, phase)][cell] += 1000 * count / total
            for phase, counts in game_transitions.items():
                total = sum(counts.values()) or 1
                for edge, count in counts.items():
                    transition_weights[(school, role, phase)][edge] += 1000 * count / total

    goals = {
        key: [[rounded(cell[0]), rounded(cell[1]), max(1, round(weight))] for cell, weight in counts.most_common(28)]
        for key, counts in goal_weights.items()
    }
    transitions = {
        key: [
            [rounded(edge[0]), rounded(edge[1]), rounded(edge[2]), rounded(edge[3]), max(1, round(weight))]
            for edge, weight in counts.most_common(96)
        ]
        for key, counts in transition_weights.items()
    }
    return goals, transitions


def build_target_priors(
    db: sqlite3.Connection,
    schools: tuple[str, ...],
    behavior_labels: dict[tuple[str, str, str], dict],
) -> tuple[dict[str, list[dict]], dict[str, list[int]], dict[str, list[dict]], dict[str, dict]]:
    """Learn target preference and attribute structure hits to firing roles.

    Hit events identify the victim but not the attacker.  Shot events do carry
    the attacking robot id and role, so structure damage is distributed across
    the roles that fired the matching calibre in the same referee second.  The
    result is deliberately labelled as an attribution instead of ground truth.
    """
    allowed = set(schools)
    shot_roles: dict[tuple[int, int, str, str], Counter] = defaultdict(Counter)
    for row in db.execute(
        """
        SELECT game_id,CAST(时刻秒 AS INT) second,学校名,类别,机器人类型,COUNT(*) shots
        FROM events
        WHERE 事件类型='发弹' AND 类别 IN ('17mm','42mm')
          AND 机器人类型 IN ('英雄','步兵3','步兵4','哨兵','空中')
        GROUP BY game_id,second,学校名,类别,机器人类型
        """
    ):
        if row["学校名"] in allowed:
            shot_roles[(int(row["game_id"]), int(row["second"]), row["学校名"], row["类别"])][row["机器人类型"]] = int(row["shots"])

    records = []
    outpost_damage: Counter = Counter()
    outpost_role_damage: Counter = Counter()
    outpost_role_opening_damage: Counter = Counter()
    outpost_role_games: dict[tuple[str, str], set[int]] = defaultdict(set)
    outpost_attack_games: dict[str, set[int]] = defaultdict(set)
    outpost_first_hit_at: dict[tuple[str, int], float] = {}
    outpost_destroyed_at: dict[tuple[str, int], float] = {}
    for row in db.execute(
        """
        SELECT e.game_id,e.时刻秒,e.机器人类型 victim_type,e.类别,ABS(e.数值) damage,
               CASE WHEN e.学校名=m.红方学校 THEN m.蓝方学校 ELSE m.红方学校 END attacker
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='受击' AND e.类别 IN ('17mm','42mm')
        ORDER BY e.game_id,e.时刻秒
        """
    ):
        attacker = row["attacker"]
        if attacker not in allowed:
            continue
        game_id = int(row["game_id"])
        second = float(row["时刻秒"] or 0)
        target = "outpost" if row["victim_type"] == "前哨站" else "base" if row["victim_type"] == "基地" else "robot"
        damage = float(row["damage"] or 0)
        records.append((attacker, game_id, second, target, damage))
        if target == "outpost":
            key = (attacker, game_id)
            outpost_first_hit_at.setdefault(key, second)
            outpost_damage[key] += damage
            outpost_attack_games[attacker].add(game_id)
            firing = shot_roles.get((game_id, int(second), attacker, row["类别"]), Counter())
            fired = sum(firing.values())
            if fired:
                for role, count in firing.items():
                    attributed = damage * count / fired
                    outpost_role_damage[(attacker, role)] += attributed
                    if second < 60:
                        outpost_role_opening_damage[(attacker, role)] += attributed
                    outpost_role_games[(attacker, role)].add(game_id)
            if outpost_damage[key] >= 1500 and key not in outpost_destroyed_at:
                outpost_destroyed_at[key] = second

    per_game: dict[tuple[str, int, int, str], Counter] = defaultdict(Counter)
    event_samples: Counter = Counter()
    for school, game_id, second, target, damage in records:
        phase = min(TARGET_PHASES - 1, int(second // 30))
        destroyed_at = outpost_destroyed_at.get((school, game_id))
        outpost_state = "alive" if destroyed_at is None or second <= destroyed_at else "down"
        per_game[(school, game_id, phase, outpost_state)][target] += damage
        event_samples[(school, phase, outpost_state)] += 1

    team_weights: dict[tuple[str, int, str], Counter] = defaultdict(Counter)
    global_weights: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for (school, _, phase, outpost_state), counts in per_game.items():
        total = sum(counts.values()) or 1
        for target, damage in counts.items():
            value = 1000 * damage / total
            team_weights[(school, phase, outpost_state)][target] += value
            global_weights[(phase, outpost_state)][target] += value

    def smoothed_prior(school: str, phase: int, outpost_state: str) -> dict:
        team = team_weights[(school, phase, outpost_state)]
        global_values = global_weights[(phase, outpost_state)]
        if not global_values:
            global_values = Counter({"robot": 45, "outpost": 55}) if outpost_state == "alive" else Counter({"robot": 75, "base": 25})
        global_total = sum(global_values.values()) or 1
        smoothed = Counter(team)
        for target in ("robot", "outpost", "base"):
            smoothed[target] += 250 * global_values[target] / global_total
        total = sum(smoothed.values()) or 1
        return {
            "robot": rounded(smoothed["robot"] / total, 4),
            "outpost": rounded(smoothed["outpost"] / total, 4),
            "base": rounded(smoothed["base"] / total, 4),
            "samples": int(event_samples[(school, phase, outpost_state)]),
        }

    payload: dict[str, list[dict]] = {}
    for school in schools:
        phases = []
        for phase in range(TARGET_PHASES):
            phases.append({
                "outpost_alive": smoothed_prior(school, phase, "alive"),
                "outpost_down": smoothed_prior(school, phase, "down"),
            })
        payload[school] = phases
    destroy_seconds: dict[str, list[int]] = {school: [] for school in schools}
    for (school, _), second in outpost_destroyed_at.items():
        destroy_seconds[school].append(round(second))
    for values in destroy_seconds.values():
        values.sort()
    attack_windows: dict[str, list[dict]] = {school: [] for school in schools}
    for (school, game_id), first_hit in outpost_first_hit_at.items():
        destroyed = outpost_destroyed_at.get((school, game_id))
        attack_windows[school].append({
            "first_hit_second": round(first_hit),
            "destroy_second": round(destroyed) if destroyed is not None else None,
        })
    for values in attack_windows.values():
        values.sort(key=lambda item: (item["first_hit_second"], item["destroy_second"] or 999))

    role_evidence: dict[str, dict] = {}
    for school in schools:
        total = sum(outpost_role_damage[(school, role)] for role in ROLES) or 1
        opening_total = sum(outpost_role_opening_damage[(school, role)] for role in ROLES) or 1
        combat_roles = tuple(role for role in ROLES if role != "工程")
        maximum_share = max((outpost_role_damage[(school, role)] / total for role in combat_roles), default=0) or 1
        attacked_games = max(1, len(outpost_attack_games[school]))
        roles = {}
        for role in ROLES:
            damage = outpost_role_damage[(school, role)]
            share = damage / total
            opening_share = outpost_role_opening_damage[(school, role)] / opening_total
            role_games = len(outpost_role_games[(school, role)])
            # A visible, pre-planned assault is reserved for combat roles with
            # meaningful repeated evidence.  UAVs are valid attackers; only the
            # engineer is excluded from a weapon objective.
            primary = role in combat_roles and role_games >= 2 and (
                share >= 0.10 or opening_share >= 0.10
            )
            manual = behavior_labels.get((school, role, "outpost_assault_role"))
            if manual:
                primary = manual["value"].lower() in {"1", "true", "yes", "positive"}
            # Manual supervision confirms/rejects eligibility; how often the
            # role is committed still comes from its observed game rate.
            commitment_probability = clamp(0.2 + role_games / attacked_games * 0.75, 0.2, 0.92)
            roles[role] = {
                "attributed_damage": rounded(damage),
                "share": rounded(share, 4),
                "opening_share": rounded(opening_share, 4),
                "games": role_games,
                "game_rate": rounded(role_games / attacked_games, 4),
                "outpost_preference": rounded(clamp(share / maximum_share, 0.02, 1), 4),
                "primary_assault_role": primary,
                "commitment_probability": rounded(commitment_probability, 4),
                "manual_label": manual,
            }
        role_evidence[school] = {
            "method": "同局同秒同口径发弹角色按发弹数分摊前哨受击伤害",
            "matched_damage": rounded(total),
            "attack_games": len(outpost_attack_games[school]),
            "roles": roles,
        }
    return payload, destroy_seconds, attack_windows, role_evidence


def build_base_damage_timing(
    db: sqlite3.Connection,
    schools: tuple[str, ...],
    game_counts: dict[str, int],
) -> dict[str, dict]:
    """Learn when each team deals base damage, separated by damage source.

    This is an empirical event-time model rather than scripted damage.  The
    browser uses the direct-fire intensity only to weight legal in-range target
    selection, while the dart intensity adjusts only legal dart windows.
    """
    allowed = set(schools)
    events: dict[str, list[tuple[str, int, float, float, str]]] = {
        "all_sources": [],
        "direct_fire": [],
        "dart": [],
    }
    for row in db.execute(
        """
        SELECT e.game_id,e.时刻秒,e.类别,ABS(e.数值) damage,
               CASE
                 WHEN e.学校名=m.红方学校 THEN m.蓝方学校
                 WHEN e.学校名=m.蓝方学校 THEN m.红方学校
               END attacker
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='受击' AND e.机器人类型='基地'
          AND e.类别 IN ('17mm','42mm','飞镖')
        ORDER BY e.game_id,e.时刻秒
        """
    ):
        school = row["attacker"]
        if school not in allowed:
            continue
        second = clamp(float(row["时刻秒"] or 0), 0, 419.999)
        item = (
            school,
            int(row["game_id"]),
            second,
            float(row["damage"] or 0),
            row["类别"],
        )
        events["all_sources"].append(item)
        events["dart" if row["类别"] == "飞镖" else "direct_fire"].append(item)

    total_games = max(1, sum(int(game_counts.get(school, 0)) for school in schools))
    payload: dict[str, dict] = {
        school: {
            "source": "区域赛基地受击事件；按对手学校反推攻击方，15 秒窗按参赛局数归一化",
            "bin_seconds": BASE_DAMAGE_BIN_SECONDS,
            "games": int(game_counts.get(school, 0)),
        }
        for school in schools
    }

    for source, source_events in events.items():
        global_games_by_bin = [set() for _ in range(BASE_DAMAGE_BINS)]
        for school, game_id, second, _, _ in source_events:
            bin_index = min(BASE_DAMAGE_BINS - 1, int(second // BASE_DAMAGE_BIN_SECONDS))
            global_games_by_bin[bin_index].add((school, game_id))
        global_rates = [len(values) / total_games for values in global_games_by_bin]
        global_mean_rate = max(1e-6, sum(global_rates) / BASE_DAMAGE_BINS)

        by_school: dict[str, list[tuple[int, float, float, str]]] = defaultdict(list)
        for school, game_id, second, damage, category in source_events:
            by_school[school].append((game_id, second, damage, category))

        for school in schools:
            team_games = max(1, int(game_counts.get(school, 0)))
            first_by_game: dict[int, float] = {}
            games_by_bin = [set() for _ in range(BASE_DAMAGE_BINS)]
            damage_by_bin = [0.0] * BASE_DAMAGE_BINS
            hits_by_bin = [0] * BASE_DAMAGE_BINS
            category_damage: Counter = Counter()
            for game_id, second, damage, category in by_school.get(school, []):
                first_by_game[game_id] = min(first_by_game.get(game_id, second), second)
                bin_index = min(BASE_DAMAGE_BINS - 1, int(second // BASE_DAMAGE_BIN_SECONDS))
                games_by_bin[bin_index].add(game_id)
                damage_by_bin[bin_index] += damage
                hits_by_bin[bin_index] += 1
                category_damage[category] += damage
            total_damage = sum(damage_by_bin)
            first_seconds = list(first_by_game.values())
            bins = []
            for bin_index in range(BASE_DAMAGE_BINS):
                observed_games = len(games_by_bin[bin_index])
                observed_rate = observed_games / team_games
                # Two global-equivalent games keep sparse teams distinct
                # without making a single observed hit a deterministic script.
                smoothed_rate = (
                    observed_games + 2 * global_rates[bin_index]
                ) / (team_games + 2)
                bins.append({
                    "start_second": bin_index * BASE_DAMAGE_BIN_SECONDS,
                    "end_second": (bin_index + 1) * BASE_DAMAGE_BIN_SECONDS - 1,
                    "attack_games": observed_games,
                    "attack_game_rate": rounded(observed_rate, 4),
                    "hit_events": hits_by_bin[bin_index],
                    "damage": rounded(damage_by_bin[bin_index]),
                    "damage_share": rounded(
                        damage_by_bin[bin_index] / total_damage if total_damage else 0,
                        4,
                    ),
                    "relative_intensity": rounded(
                        clamp(smoothed_rate / global_mean_rate, 0.12, 4.0),
                        4,
                    ),
                })
            payload[school][source] = {
                "attack_games": len(first_by_game),
                "attack_game_rate": rounded(len(first_by_game) / team_games, 4),
                "first_damage_second": {
                    "samples": len(first_seconds),
                    "p25": rounded(percentile(first_seconds, 0.25)) if first_seconds else None,
                    "median": rounded(percentile(first_seconds, 0.5)) if first_seconds else None,
                    "p75": rounded(percentile(first_seconds, 0.75)) if first_seconds else None,
                },
                "total_damage": rounded(total_damage),
                "damage_share_by_weapon": {
                    category: rounded(value / total_damage, 4)
                    for category, value in sorted(category_damage.items())
                } if total_damage else {},
                "bins_15s": bins,
            }
    return payload


def infer_hero_archetype_priors(
    db: sqlite3.Connection,
    schools: tuple[str, ...],
    behavior_labels: dict[tuple[str, str, str], dict],
) -> dict[str, dict]:
    """Infer an automatic national archetype prior from regional hero posture.

    The new V2.1 archetype choice did not exist in regional telemetry.  We do
    not claim to observe that choice; instead, close structure firing and deep
    forward firing positions produce a transparent melee-priority default.
    """
    allowed = set(schools)
    structure_distances: dict[str, list[float]] = defaultdict(list)
    forward_positions: dict[str, list[float]] = defaultdict(list)
    for row in db.execute(
        """
        SELECT DISTINCT e.学校名,e.阵营,e.game_id,CAST(e.时刻秒 AS INT) second,t.x,t.y
        FROM events e JOIN timeseries t
          ON t.game_id=e.game_id AND t.robot_id=e.robot_id
         AND CAST(t.时刻秒 AS INT)=CAST(e.时刻秒 AS INT) AND t.学校名=e.学校名
        WHERE e.事件类型='发弹' AND e.机器人类型='英雄' AND e.类别='42mm'
          AND t.x BETWEEN 0.05 AND 27.95 AND t.y BETWEEN 0.05 AND 14.95
        """
    ):
        school = row["学校名"]
        if school not in allowed:
            continue
        x, y = float(row["x"]), float(row["y"])
        if row["阵营"] == "红":
            enemy_outpost, enemy_base, forward = (17.0, 11.75), (25.34, 7.5), x
        else:
            enemy_outpost, enemy_base, forward = (11.0, 3.25), (2.66, 7.5), 28.0 - x
        structure_distances[school].append(min(
            ((x - enemy_outpost[0]) ** 2 + (y - enemy_outpost[1]) ** 2) ** 0.5,
            ((x - enemy_base[0]) ** 2 + (y - enemy_base[1]) ** 2) ** 0.5,
        ))
        forward_positions[school].append(forward)

    result = {}
    for school in schools:
        distances = structure_distances.get(school, [])
        positions = forward_positions.get(school, [])
        median_distance = median(distances) if distances else 8.0
        median_forward = median(positions) if positions else 10.0
        archetype = "melee" if median_distance <= 6.0 or median_forward >= 14.0 else "ranged"
        archetype_label = behavior_labels.get((school, "英雄", "hero_archetype_default"))
        if archetype_label and archetype_label["value"] in HERO_ARCHETYPES:
            archetype = archetype_label["value"]
        inferred_style = "long_range" if median_distance >= 8.0 else "close_pressure" if median_distance <= 5.5 else "flexible"
        style_label = behavior_labels.get((school, "英雄", "engagement_style"))
        engagement_style = style_label["value"] if style_label else inferred_style
        result[school] = {
            "archetype": archetype,
            "source": archetype_label["source"] if archetype_label else "区域赛英雄42mm发弹位置的近结构距离与前压深度画像推断；非国赛实选标签",
            "manual_label": archetype_label,
            "firing_seconds": len(distances),
            "median_enemy_structure_distance_m": rounded(median_distance),
            "median_canonical_forward_x_m": rounded(median_forward),
            "engagement_style": engagement_style,
            "engagement_style_label": style_label,
            "preferred_range_m": rounded(clamp(median_distance * 0.92, 3.0, 11.4)),
        }
    return result


def build_dart_mode_priors(db: sqlite3.Connection, schools: tuple[str, ...]) -> dict[str, list[dict]]:
    """Return V2.1 legal base-dart modes weighted by observed team hits."""
    allowed_damage = {200: "fixed", 300: "random_fixed", 625: "random_moving", 1000: "terminal_moving"}
    counts: dict[str, Counter] = defaultdict(Counter)
    global_counts = Counter()
    for row in db.execute(
        """
        SELECT 学校名,CAST(数值 AS INT) damage,COUNT(*) hits
        FROM events
        WHERE 事件类型='飞镖命中' AND 目标类型='基地'
        GROUP BY 学校名,damage
        """
    ):
        damage = int(row["damage"] or 0)
        if damage not in allowed_damage:
            continue
        hits = int(row["hits"] or 0)
        global_counts[damage] += hits
        if row["学校名"] in schools:
            counts[row["学校名"]][damage] += hits
    # Regional data predates the 1000-point terminal target. Keep a small legal
    # fallback mass so teams without a base hit are not locked to an old mode.
    fallback = Counter(global_counts)
    fallback[1000] += 1
    result = {}
    for school in schools:
        values = counts[school] or fallback
        result[school] = [
            {"damage": damage, "mode": allowed_damage[damage], "weight": int(weight)}
            for damage, weight in sorted(values.items()) if weight > 0
        ]
    return result


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
    behavior_labels = load_behavior_labels(options.behavior_labels)
    db = sqlite3.connect(f"file:{options.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    economy_priors = infer_regional_economy(db, schools)
    uav_navigation = build_uav_navigation(db, schools)
    hero_archetype_priors = infer_hero_archetype_priors(db, schools, behavior_labels)
    dart_mode_priors = build_dart_mode_priors(db, schools)

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

    hit_events: Counter = Counter()
    damage_values: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for row in db.execute(
        f"""
        SELECT CASE
                 WHEN e.学校名=m.红方学校 THEN m.蓝方学校
                 WHEN e.学校名=m.蓝方学校 THEN m.红方学校
               END attacker,
               e.类别 category,e.机器人类型 victim,ABS(e.数值) damage,COUNT(*) hits
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='受击' AND e.类别 IN ('17mm','42mm')
        GROUP BY attacker,e.类别,e.机器人类型,damage
        """
    ):
        if row["attacker"] in entries:
            school, category = row["attacker"], row["category"]
            hits = int(row["hits"] or 0)
            hit_events[(school, category)] += hits
            target = "base" if row["victim"] == "基地" else "outpost" if row["victim"] == "前哨站" else "robot"
            damage_values[(school, category, target)][rounded(row["damage"])] += hits

    # Ground robots retain five-second conditional movement rather than
    # independently teleporting their intent between minute-level dwell modes.
    # Each game has equal total weight and stationary service dwell is reduced.
    goals, ground_transitions = build_ground_navigation(options.games_dir, schools)
    target_priors, outpost_destroy_seconds, outpost_attack_windows, outpost_attack_roles = build_target_priors(
        db, schools, behavior_labels,
    )
    base_damage_timing = build_base_damage_timing(db, schools, games)

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
            "17mm": clamp(hit_events[(school, "17mm")] / max(1, team_shots["17mm"]), 0.025, 0.62),
            "42mm": clamp(hit_events[(school, "42mm")] / max(1, team_shots["42mm"]), 0.025, 0.72),
        }
        accuracy_models = {
            category: {
                "distribution": "match_uniform_then_per_shot_bernoulli",
                "mean_probability": rounded(accuracy[category], 4),
                "match_multiplier_range": [0.78, 1.22],
                "per_shot_random": True,
                "shots": team_shots[category],
                "detected_hit_events": int(hit_events[(school, category)]),
                "source": "对手受击事件数/本队发弹数；每局先采样基础概率，每发再伯努利抽样",
            }
            for category in ("17mm", "42mm")
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
                "accuracy_model": accuracy_models.get(weapon),
                "goals_by_minute": role_goals,
            }
            if role in GROUND_ROLES:
                role_payload[role]["transitions_by_minute"] = [
                    ground_transitions.get((school, role, phase), [])
                    for phase in range(PHASES)
                ]
            if role == "英雄":
                damage_by_target = {}
                for target in ("robot", "outpost", "base"):
                    distribution = damage_values[(school, "42mm", target)]
                    # Values below 200 are already reduced by defense/remaining
                    # HP and must not be learned again as raw projectile damage.
                    raw_modes = Counter({damage: count for damage, count in distribution.items() if damage in {200, 300}})
                    mode_damage = raw_modes.most_common(1)[0][0] if raw_modes else 200
                    manual_damage = behavior_labels.get((school, role, f"{target}_damage_per_hit"))
                    if manual_damage:
                        mode_damage = float(manual_damage["value"])
                    damage_by_target[target] = {
                        "mode_damage": rounded(mode_damage),
                        "distribution": [
                            {"damage": damage, "weight": count}
                            for damage, count in distribution.most_common(12)
                        ],
                        "manual_label": manual_damage,
                        "source": "区域赛对手受击原始合法值 200/300 主模态；排除残血与防御减伤值",
                    }
                role_payload[role].update({
                    "hero_archetype_default": hero_archetype_priors[school]["archetype"],
                    "hero_archetype_evidence": hero_archetype_priors[school],
                    "engagement_profile": {
                        "style": hero_archetype_priors[school]["engagement_style"],
                        "preferred_range_m": hero_archetype_priors[school]["preferred_range_m"],
                        "source": hero_archetype_priors[school]["engagement_style_label"]
                        or "英雄42mm发弹位置与敌方结构距离",
                    },
                    "damage_per_hit_by_target": damage_by_target,
                    "level_by_minute": level_by_minute,
                })
            elif role == "空中":
                role_payload[role]["uav_navigation"] = uav_profile
        aggregate = macro["teams"].get(school, {}).get("aggregate", {})
        hero_profile = role_payload["英雄"]
        assault_roles = outpost_attack_roles[school]["roles"]
        behavior_profile = {
            "source": "44 队区域赛逐校统计；人工标签只覆盖数据无显式字段的已确认行为",
            "style": aggregate.get("style", "常规阵地运营"),
            "hero": {
                "archetype": hero_profile["hero_archetype_default"],
                "engagement_style": hero_profile["engagement_profile"]["style"],
                "preferred_range_m": hero_profile["engagement_profile"]["preferred_range_m"],
                "accuracy_42mm": accuracy_models["42mm"]["mean_probability"],
                "shots_42mm": accuracy_models["42mm"]["shots"],
                "base_damage_per_hit": hero_profile["damage_per_hit_by_target"]["base"]["mode_damage"],
            },
            "outpost": {
                "opening_target_probability": target_priors[school][0]["outpost_alive"]["outpost"],
                "primary_roles": [
                    role for role in ROLES
                    if assault_roles[role]["primary_assault_role"]
                ],
                "role_commitment_probability": {
                    role: assault_roles[role]["commitment_probability"]
                    for role in ROLES if role != "工程"
                },
                "uav_attributed_share": assault_roles["空中"]["share"],
                "uav_commitment_probability": assault_roles["空中"]["commitment_probability"],
            },
            "base": {
                "direct_fire_game_rate": base_damage_timing[school]["direct_fire"]["attack_game_rate"],
                "first_direct_damage_second": base_damage_timing[school]["direct_fire"]["first_damage_second"],
                "all_source_game_rate": base_damage_timing[school]["all_sources"]["attack_game_rate"],
                "first_any_damage_second": base_damage_timing[school]["all_sources"]["first_damage_second"],
            },
            "movement": {
                "speed_mps_by_role": {
                    role: role_payload[role]["speed_mps"] for role in ROLES
                },
                "uav_first_takeoff_second": uav_navigation[school]["first_takeoff_second"],
                "uav_median_airborne_run_seconds": uav_navigation[school]["median_airborne_run_seconds"],
            },
            "evidence": {
                "games": games[school],
                "hero_firing_seconds": hero_archetype_priors[school]["firing_seconds"],
                "outpost_attack_games": outpost_attack_roles[school]["attack_games"],
                "base_damage_games": base_damage_timing[school]["all_sources"]["attack_games"],
                "uav_navigation_samples": uav_navigation[school]["samples"],
            },
        }
        teams[school] = {
            "team": entry.team,
            "stage": entry.stage,
            "region": entry.region,
            "games": games[school],
            "accuracy": {key: rounded(value, 4) for key, value in accuracy.items()},
            "accuracy_models": accuracy_models,
            "dart_hits_per_game": aggregate.get("dart_hits_per_game", 0),
            "dart_gates_per_game": aggregate.get("dart_gates_per_game", 0),
            "dart_base_modes": dart_mode_priors[school],
            "radar_counters_per_game": rounded(radar_counters[school] / max(1, games[school]), 3),
            "uav_counters_received_per_game": rounded(uav_counters_received[school] / max(1, games[school]), 3),
            "economy_prior": economy_priors[school],
            "target_prior_by_30s": target_priors[school],
            "base_damage_timing": base_damage_timing[school],
            "outpost_destroy_seconds": outpost_destroy_seconds[school],
            "outpost_attack_windows": outpost_attack_windows[school],
            "outpost_attack_roles": outpost_attack_roles[school],
            "style": aggregate.get("style", "常规阵地运营"),
            "behavior_profile": behavior_profile,
            "roles": role_payload,
        }
    db.close()

    payload = {
        "schema_version": 12,
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
        "team_behavior_coverage": {
            "team_count": len(teams),
            "school_specific": True,
            "manual_labels_are_overrides_only": True,
            "dimensions": [
                "hero_archetype_and_engagement_range",
                "weapon_accuracy_and_target_damage",
                "outpost_role_attribution_and_commitment",
                "team_base_damage_timing_by_source",
                "ground_and_uav_movement",
                "terrain_capability_and_motion_in_navigation_model",
            ],
        },
        "training_feature_schema": {
            "static": {
                "hero_archetype": {"type": "categorical", "values": ["ranged", "melee"], "default": "team_profile"},
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
                "terrain_speed_multiplier": {"type": "number", "range": [0, 1.25]},
                "sampled_weapon_accuracy": {"type": "number", "range": [0.018, 0.9]},
                "assembly_protected": {"type": "boolean"},
                "assembly_invulnerable_seconds": {"type": "integer", "range": [0, 180]},
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
                "supply": {"shape": "rectangle", "center": [2.0, 1.65], "radius": [1.85, 1.45], "target_inset_ratio": 0.82, "ammo": True, "heal": True, "label": "补给区整块区域"},
                "base": {"shape": "ellipse", "center": [2.66, 7.5], "radius": [1.8, 1.55], "target_inset_ratio": 0.82, "target_inner_radius_ratio": 0.3, "ammo": True, "heal": False, "label": "基地区"},
                "outpost": {"shape": "half_ellipse", "center": [11.0, 3.25], "radius": [1.55, 1.35], "direction": [0.958778, 0.284157], "target_inset_ratio": 0.82, "ammo": True, "heal": False, "label": "前哨对方侧半圆"},
            },
            "blue": {
                "supply": {"shape": "rectangle", "center": [26.0, 13.35], "radius": [1.85, 1.45], "target_inset_ratio": 0.82, "ammo": True, "heal": True, "label": "补给区整块区域"},
                "base": {"shape": "ellipse", "center": [25.34, 7.5], "radius": [1.8, 1.55], "target_inset_ratio": 0.82, "target_inner_radius_ratio": 0.3, "ammo": True, "heal": False, "label": "基地区"},
                "outpost": {"shape": "half_ellipse", "center": [17.0, 11.75], "radius": [1.55, 1.35], "direction": [-0.958778, -0.284157], "target_inset_ratio": 0.82, "ammo": True, "heal": False, "label": "前哨对方侧半圆"},
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
            "damage": {"17mm": 20, "42mm": 200, "base_top_17mm": 5, "outpost_dart": 750},
            "base_armor": {
                "enemy_fortress_unlock_second": 180,
                "enemy_outpost_must_be_down": True,
                "capture_seconds": 20,
                "capture_grace_seconds": 3,
            },
            "dart_base_damage_modes": {
                "fixed": 200,
                "random_fixed": 300,
                "random_moving": 625,
                "terminal_moving": 1000,
            },
            "heat_per_shot": {"17mm": 10, "42mm": 100},
            "hero_archetypes": HERO_ARCHETYPES,
            "hero_default_archetype": "team_profile",
            "heal_ratio_per_second": 0.1,
            "late_heal_ratio_per_second": 0.25,
            "late_heal_start_second": 240,
            "out_of_combat_seconds": 6,
            "engineer_assembly_invulnerability_seconds": 180,
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
            "structure-hit attacker roles are attributed from same-game, same-second, same-calibre shot events; simultaneous shooters share damage in proportion to shots",
            "ground goals are per-game-balanced 0.5 m position modes with stationary service dwell down-weighted; five-second transitions preserve local tactical continuity",
            "team target priorities are learned in 30-second phases; visible outpost assignments additionally require repeated role-level shot attribution evidence",
            "base damage timing is learned per team in 15-second windows and separated into direct-fire and dart sources; it weights legal attacks but never applies scripted damage",
            "UAV helipad and airborne samples are separated; airborne goals are game-normalized and connected by empirical five-second transitions rather than independent dwell-point sampling",
            "fly-ramp alignment/stop/acceleration comes from official event windows; complete B3/R3 trajectories learn ascent/descent angles separately per school-role, with documented direction-specific global fallbacks",
            "navigation hard-blocks the user-annotated elevated regions and 16 terrain gates, but a full CAD-derived occupancy mask for every static wall and structure is not yet available",
            "national matches always start at 400 coins; regional initial-coin ratings are retained only as descriptive telemetry and are not used by the simulator",
            "technology-core completion priors are inferred from persistent ten-second increases in regional total-coins telemetry because the export has no explicit assembly event",
            "V2.1.0 hero archetype was not present in regional telemetry; automatic defaults are explicitly marked inferences from hero firing posture and remain user-overridable",
        ],
        "teams": teams,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(teams)} teams to {options.output} ({options.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
