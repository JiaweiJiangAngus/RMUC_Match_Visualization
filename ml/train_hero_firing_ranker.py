#!/usr/bin/env python3
"""Train a context-conditioned ranker for observed hero 42 mm firing anchors.

This model does not invent near/long-range class labels.  For every held-out
series it receives one position the hero actually chose for its next 42 mm
shot and three other positions that the same team really used in that match
phase.  Cross-entropy trains the chosen anchor to outrank those alternatives.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .train_trajectory import load_group_splits
except ImportError:
    from train_trajectory import load_group_splits  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.team_style_report import TEAMS  # noqa: E402


DEFAULT_DB = ROOT.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
DEFAULT_DATA_DIR = ROOT / "docs" / "data" / "games"
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "hero_firing_ranker.json"
GROUND_ROLES = ("英雄", "工程", "步兵3", "步兵4", "哨兵")
OFFSETS = (3, 5, 10)
CANDIDATES = 4
FIELD_DIAGONAL = math.hypot(28, 15)
STRUCTURE_POSITIONS = {
    "红": {
        "own_base": (2.66, 7.5), "own_outpost": (11.0, 3.25),
        "enemy_base": (25.34, 7.5), "enemy_outpost": (17.0, 11.75),
    },
    "蓝": {
        "own_base": (25.34, 7.5), "own_outpost": (17.0, 11.75),
        "enemy_base": (2.66, 7.5), "enemy_outpost": (11.0, 3.25),
    },
}
FEATURE_NAMES = (
    "elapsed_ratio",
    "remaining_ratio",
    "self_hp_ratio",
    "self_hp_loss_5s",
    "own_base_hp_ratio",
    "own_outpost_hp_ratio",
    "enemy_base_hp_ratio",
    "enemy_outpost_hp_ratio",
    "own_ground_alive_ratio",
    "enemy_ground_alive_ratio",
    "own_ground_hp_ratio",
    "enemy_ground_hp_ratio",
    "ground_hp_advantage",
    "candidate_x_ratio",
    "candidate_y_ratio",
    "candidate_own_base_distance_ratio",
    "candidate_own_outpost_distance_ratio",
    "candidate_enemy_base_distance_ratio",
    "candidate_enemy_outpost_distance_ratio",
    "candidate_nearest_enemy_distance_ratio",
    "candidate_train_heatmap_prior",
    "team_coins_ratio",
    "self_vulnerable",
)
ROW = {
    "type": 1, "side": 2, "hp": 3, "max_hp": 4,
    "x": 5, "y": 6, "coins": 10, "vulnerable": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--split-seed", type=int, default=7803)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    return parser.parse_args()


def opponent(side: str) -> str:
    return "蓝" if side == "红" else "红"


def frame_index(rows: list[list]) -> dict[tuple[str, str], list]:
    return {(row[ROW["side"]], row[ROW["type"]]): row for row in rows}


def hp_ratio(row: list | None) -> float:
    if not row:
        return 0.0
    return max(
        0.0,
        min(1.25, float(row[ROW["hp"]] or 0) / max(1.0, float(row[ROW["max_hp"]] or 1))),
    )


def canonical(point: tuple[float, float], side: str) -> tuple[float, float]:
    return point if side == "红" else (28 - point[0], 15 - point[1])


def distance_ratio(start: tuple[float, float], end: tuple[float, float]) -> float:
    return min(1.0, math.dist(start, end) / FIELD_DIAGONAL)


def ground_summary(
    frame: dict[tuple[str, str], list],
    side: str,
) -> tuple[list[list], float, float]:
    rows = [frame.get((side, role)) for role in GROUND_ROLES]
    present = [row for row in rows if row]
    alive = [row for row in present if float(row[ROW["hp"]] or 0) > 0]
    hp = sum(max(0.0, float(row[ROW["hp"]] or 0)) for row in present)
    maximum = sum(max(1.0, float(row[ROW["max_hp"]] or 1)) for row in present)
    return alive, len(alive) / len(GROUND_ROLES), min(1.25, hp / max(1.0, maximum))


def contextual_candidate_features(
    frames: dict[int, dict[tuple[str, str], list]],
    second: int,
    side: str,
    candidate_canonical: tuple[float, float],
) -> list[float] | None:
    frame = frames.get(second)
    if not frame:
        return None
    hero = frame.get((side, "英雄"))
    if not hero or hero[ROW["x"]] is None or hero[ROW["y"]] is None or float(hero[ROW["hp"]] or 0) <= 0:
        return None
    enemy = opponent(side)
    previous = frames.get(max(0, second - 5), {}).get((side, "英雄"))
    current_hp = hp_ratio(hero)
    previous_hp = hp_ratio(previous) if previous else current_hp
    own_alive, own_alive_ratio, own_hp = ground_summary(frame, side)
    enemy_alive, enemy_alive_ratio, enemy_hp = ground_summary(frame, enemy)
    candidate_world = canonical(candidate_canonical, side)
    nearest_enemy = min(
        (
            distance_ratio(
                candidate_world,
                (float(row[ROW["x"]]), float(row[ROW["y"]])),
            )
            for row in enemy_alive
            if row[ROW["x"]] is not None and row[ROW["y"]] is not None
        ),
        default=1.0,
    )
    positions = STRUCTURE_POSITIONS[side]
    own_base = frame.get((side, "基地"))
    own_outpost = frame.get((side, "前哨站"))
    enemy_base = frame.get((enemy, "基地"))
    enemy_outpost = frame.get((enemy, "前哨站"))
    return [
        min(1.0, second / 420),
        max(0.0, 1 - second / 420),
        current_hp,
        max(0.0, min(1.0, previous_hp - current_hp)),
        hp_ratio(own_base),
        hp_ratio(own_outpost),
        hp_ratio(enemy_base),
        hp_ratio(enemy_outpost),
        own_alive_ratio,
        enemy_alive_ratio,
        own_hp,
        enemy_hp,
        max(-1.0, min(1.0, own_hp - enemy_hp)),
        candidate_canonical[0] / 28,
        candidate_canonical[1] / 15,
        distance_ratio(candidate_world, positions["own_base"]),
        distance_ratio(candidate_world, positions["own_outpost"]),
        distance_ratio(candidate_world, positions["enemy_base"]),
        distance_ratio(candidate_world, positions["enemy_outpost"]),
        nearest_enemy,
        0.0,
        min(1.0, max(0.0, float(hero[ROW["coins"]] or 0)) / 2000),
        1.0 if hero[ROW["vulnerable"]] else 0.0,
    ]


def load_firing_seconds(db_path: Path) -> dict[int, dict[str, list[int]]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    values: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in connection.execute(
        """
        SELECT game_id,学校名,CAST(时刻秒 AS INT) second
        FROM events
        WHERE 事件类型='发弹' AND 机器人类型='英雄' AND 类别='42mm'
        GROUP BY game_id,学校名,second
        """
    ):
        values[int(row[0])][row[1]].append(int(row[2]))
    connection.close()
    return {
        game_id: {school: sorted(set(seconds)) for school, seconds in schools.items()}
        for game_id, schools in values.items()
    }


def game_payload(path: Path) -> tuple[dict, dict[int, dict[tuple[str, str], list]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        game = json.load(handle)
    return game["info"], {
        int(second): frame_index(rows)
        for second, rows in game.get("frames", {}).items()
    }


def anchor_pools(
    paths: list[Path],
    firing: dict[int, dict[str, list[int]]],
    teams: set[str],
) -> dict[tuple[str, int], list[tuple[float, float]]]:
    pools: dict[tuple[str, int], set[tuple[float, float]]] = defaultdict(set)
    for path in paths:
        game_id = int(path.name.removesuffix(".json.gz"))
        if game_id not in firing:
            continue
        info, frames = game_payload(path)
        side_by_school = {info["red"]: "红", info["blue"]: "蓝"}
        for school, seconds in firing[game_id].items():
            if school not in teams or school not in side_by_school:
                continue
            side = side_by_school[school]
            for second in seconds:
                hero = frames.get(second, {}).get((side, "英雄"))
                if not hero or hero[ROW["x"]] is None or hero[ROW["y"]] is None or float(hero[ROW["hp"]] or 0) <= 0:
                    continue
                point = canonical((float(hero[ROW["x"]]), float(hero[ROW["y"]])), side)
                pools[(school, min(6, second // 60))].add((
                    round(point[0], 2), round(point[1], 2),
                ))
    return {key: sorted(values) for key, values in pools.items()}


def candidate_set(
    school: str,
    phase: int,
    positive: tuple[float, float],
    pools: dict[tuple[str, int], list[tuple[float, float]]],
    all_phase: dict[int, list[tuple[float, float]]],
    random_source: random.Random,
) -> list[tuple[float, float]]:
    pool = pools.get((school, phase), [])
    if len(pool) < CANDIDATES:
        pool = [
            point
            for nearby in range(max(0, phase - 1), min(6, phase + 1) + 1)
            for point in pools.get((school, nearby), [])
        ]
    eligible = [point for point in pool if math.dist(point, positive) >= 0.9]
    if len(eligible) < CANDIDATES - 1:
        eligible.extend(
            point for point in all_phase.get(phase, [])
            if math.dist(point, positive) >= 0.9
        )
    unique = sorted(set(eligible), key=lambda point: math.dist(point, positive))
    if not unique:
        return []
    selected = [unique[0]]
    remaining = unique[1:]
    if remaining:
        selected.append(remaining[random_source.randrange(len(remaining))])
    if len(remaining) > 1:
        far_half = remaining[len(remaining) // 2:]
        selected.append(far_half[random_source.randrange(len(far_half))])
    while len(selected) < CANDIDATES - 1:
        selected.append(unique[random_source.randrange(len(unique))])
    return [positive, *selected[:CANDIDATES - 1]]


def build_groups(
    paths: list[Path],
    firing: dict[int, dict[str, list[int]]],
    pools: dict[tuple[str, int], list[tuple[float, float]]],
    teams: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Counter]:
    team_index = {school: index for index, school in enumerate(teams)}
    other_opponent = len(teams)
    all_phase: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (_school, phase), points in pools.items():
        all_phase[phase].extend(points)
    random_source = random.Random(seed)
    groups: list[list[list[float]]] = []
    team_ids: list[int] = []
    opponent_ids: list[int] = []
    counts: Counter = Counter()
    for path in paths:
        game_id = int(path.name.removesuffix(".json.gz"))
        if game_id not in firing:
            continue
        info, frames = game_payload(path)
        side_by_school = {info["red"]: "红", info["blue"]: "蓝"}
        opponent_by_school = {info["red"]: info["blue"], info["blue"]: info["red"]}
        for school, seconds in firing[game_id].items():
            opponent_school = opponent_by_school.get(school)
            if school not in team_index or school not in side_by_school:
                continue
            side = side_by_school[school]
            for firing_second in seconds:
                fired = frames.get(firing_second, {}).get((side, "英雄"))
                if not fired or fired[ROW["x"]] is None or fired[ROW["y"]] is None or float(fired[ROW["hp"]] or 0) <= 0:
                    continue
                positive = canonical((float(fired[ROW["x"]]), float(fired[ROW["y"]])), side)
                phase = min(6, firing_second // 60)
                candidates = candidate_set(
                    school, phase, positive, pools, all_phase, random_source,
                )
                if len(candidates) != CANDIDATES:
                    continue
                for offset in OFFSETS:
                    context_second = firing_second - offset
                    if context_second < 0:
                        continue
                    values = [
                        contextual_candidate_features(
                            frames, context_second, side, candidate,
                        )
                        for candidate in candidates
                    ]
                    if any(value is None for value in values):
                        continue
                    groups.append(values)  # type: ignore[arg-type]
                    team_ids.append(team_index[school])
                    opponent_ids.append(team_index.get(opponent_school, other_opponent))
                    counts[school] += 1
    return (
        np.asarray(groups, dtype=np.float32),
        np.asarray(team_ids, dtype=np.int64),
        np.asarray(opponent_ids, dtype=np.int64),
        counts,
    )


def encode(
    values: np.ndarray,
    team_ids: np.ndarray,
    opponent_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    team_count: int,
    opponent_count: int,
) -> np.ndarray:
    numeric = (values - mean) / std
    team_hot = np.eye(team_count, dtype=np.float32)[team_ids]
    opponent_hot = np.eye(opponent_count, dtype=np.float32)[opponent_ids]
    categorical = np.concatenate((team_hot, opponent_hot), axis=1)
    categorical = np.repeat(categorical[:, None, :], CANDIDATES, axis=1)
    return np.concatenate((numeric, categorical), axis=2)


def add_training_heatmap_prior(
    built: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, Counter]],
) -> None:
    elapsed_index = FEATURE_NAMES.index("elapsed_ratio")
    x_index = FEATURE_NAMES.index("candidate_x_ratio")
    y_index = FEATURE_NAMES.index("candidate_y_ratio")
    prior_index = FEATURE_NAMES.index("candidate_train_heatmap_prior")
    counts: Counter = Counter()
    maxima: Counter = Counter()
    for group, team_id in zip(built["train"][0], built["train"][1]):
        positive = group[0]
        phase = min(6, int(float(positive[elapsed_index]) * 420 // 60))
        key = (
            int(team_id), phase,
            round(float(positive[x_index]) * 28 * 2) / 2,
            round(float(positive[y_index]) * 15 * 2) / 2,
        )
        counts[key] += 1
    for (team_id, phase, _x, _y), count in counts.items():
        maxima[(team_id, phase)] = max(maxima[(team_id, phase)], count)
    for values, team_ids, _opponents, _counts in built.values():
        for group, team_id in zip(values, team_ids):
            phase = min(6, int(float(group[0, elapsed_index]) * 420 // 60))
            maximum = max(1, maxima[(int(team_id), phase)])
            for candidate in group:
                key = (
                    int(team_id), phase,
                    round(float(candidate[x_index]) * 28 * 2) / 2,
                    round(float(candidate[y_index]) * 15 * 2) / 2,
                )
                candidate[prior_index] = math.log1p(counts[key]) / math.log1p(maximum)


class HeroFiringRanker(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        prior_mean: float,
        prior_std: float,
        prior_epsilon: float = 0.02,
    ) -> None:
        super().__init__()
        self.prior_index = FEATURE_NAMES.index("candidate_train_heatmap_prior")
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.prior_epsilon = prior_epsilon
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        prior = (
            values[..., self.prior_index] * self.prior_std + self.prior_mean
        ).clamp_min(0)
        base = torch.log(prior + self.prior_epsilon)
        return base + self.score(self.trunk(values)).squeeze(-1)


def ranking_metrics(
    scores: torch.Tensor,
    team_ids: np.ndarray,
    teams: tuple[str, ...],
    raw_values: np.ndarray,
) -> dict:
    order = scores.argsort(dim=1, descending=True)
    top1 = float((order[:, 0] == 0).float().mean())
    ranks = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
    per_team = {}
    team_tensor = torch.from_numpy(team_ids).to(scores.device)
    for index, school in enumerate(teams):
        mask = team_tensor == index
        if mask.any():
            per_team[school] = {
                "groups": int(mask.sum()),
                "top1_accuracy": round(float((order[mask, 0] == 0).float().mean()), 4),
                "mean_reciprocal_rank": round(float((1 / ranks[mask].float()).mean()), 4),
            }
    heatmap_values = raw_values[:, :, FEATURE_NAMES.index("candidate_train_heatmap_prior")]
    heatmap_max = heatmap_values.max(axis=1, keepdims=True)
    heatmap_ties = np.maximum(1, (heatmap_values == heatmap_max).sum(axis=1))
    heatmap_expected_top1 = float(
        ((heatmap_values[:, 0] == heatmap_max[:, 0]) / heatmap_ties).mean()
    )
    return {
        "groups": len(scores),
        "top1_accuracy": round(top1, 4),
        "pairwise_accuracy": round(float((scores[:, :1] > scores[:, 1:]).float().mean()), 4),
        "mean_reciprocal_rank": round(float((1 / ranks.float()).mean()), 4),
        "random_top1_baseline": round(1 / CANDIDATES, 4),
        "training_heatmap_top1_baseline": round(heatmap_expected_top1, 4),
        "per_team": per_team,
    }


def rounded(tensor: torch.Tensor) -> list:
    return np.round(tensor.detach().cpu().numpy(), 7).tolist()


def main() -> None:
    options = parse_args()
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    teams = tuple(sorted(entry.school for entry in TEAMS))
    opponents = (*teams, "__OTHER__")
    team_set = set(teams)
    firing = load_firing_seconds(options.db)
    splits = load_group_splits(options.data_dir, options.split_seed)
    built = {}
    for index, split in enumerate(("train", "val", "test")):
        pools = anchor_pools(splits[split], firing, team_set)
        built[split] = build_groups(
            splits[split], firing, pools, teams, options.seed + index * 1009,
        )
        print(f"{split}: {len(built[split][0])} ranking groups")
    add_training_heatmap_prior(built)
    train_values = built["train"][0]
    mean = train_values.reshape(-1, len(FEATURE_NAMES)).mean(axis=0)
    std = np.maximum(
        train_values.reshape(-1, len(FEATURE_NAMES)).std(axis=0), 1e-4,
    )
    encoded = {
        split: encode(
            values[0], values[1], values[2], mean, std, len(teams), len(opponents),
        )
        for split, values in built.items()
    }
    loaders = {}
    for split in ("train", "val"):
        dataset = TensorDataset(
            torch.from_numpy(encoded[split]),
            torch.zeros(len(encoded[split]), dtype=torch.long),
        )
        loaders[split] = DataLoader(
            dataset, batch_size=options.batch_size, shuffle=split == "train",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_index = FEATURE_NAMES.index("candidate_train_heatmap_prior")
    model = HeroFiringRanker(
        encoded["train"].shape[2],
        options.hidden_size,
        float(mean[prior_index]),
        float(std[prior_index]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=2e-4,
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, options.epochs + 1):
        epoch_losses = {}
        for split in ("train", "val"):
            model.train(split == "train")
            total_loss = 0.0
            total_rows = 0
            for values, labels in loaders[split]:
                values, labels = values.to(device), labels.to(device)
                with torch.set_grad_enabled(split == "train"):
                    scores = model(values)
                    loss = nn.functional.cross_entropy(scores, labels)
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total_loss += float(loss.detach()) * len(values)
                total_rows += len(values)
            epoch_losses[split] = total_loss / max(1, total_rows)
        history.append({"epoch": epoch, **{key: round(value, 6) for key, value in epoch_losses.items()}})
        print(f"epoch {epoch:02d}: train={epoch_losses['train']:.5f} val={epoch_losses['val']:.5f}")
        if epoch_losses["val"] < best_loss - 1e-5:
            best_loss = epoch_losses["val"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= options.patience:
                break
    if best_state is None:
        raise RuntimeError("hero firing ranker did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics = {}
    with torch.no_grad():
        for split in ("val", "test"):
            scores = model(torch.from_numpy(encoded[split]).to(device))
            metrics[split] = ranking_metrics(
                scores, built[split][1], teams, built[split][0],
            )
    acceptance = {
        "validation_top1_beats_training_heatmap": (
            metrics["val"]["top1_accuracy"]
            > metrics["val"]["training_heatmap_top1_baseline"]
        ),
        "test_top1_beats_training_heatmap": (
            metrics["test"]["top1_accuracy"]
            > metrics["test"]["training_heatmap_top1_baseline"]
        ),
        "test_pairwise_accuracy_above_chance": (
            metrics["test"]["pairwise_accuracy"] > 0.5
        ),
        "test_mrr_above_random_four_way": (
            metrics["test"]["mean_reciprocal_rank"] > (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4
        ),
    }
    if not all(acceptance.values()):
        raise RuntimeError(
            "held-out firing-ranker acceptance gate failed; artifact was not published: "
            + json.dumps(acceptance, ensure_ascii=False)
        )

    first: nn.Linear = model.trunk[0]  # type: ignore[assignment]
    second: nn.Linear = model.trunk[2]  # type: ignore[assignment]
    head: nn.Linear = model.score  # type: ignore[assignment]
    output = {
        "schema_version": 1,
        "model_kind": "contextual_hero_firing_anchor_ranker",
        "source": "区域赛英雄真实42mm下一发位置为正样本；优先用同队同阶段真实发弹位置作负候选，不足时只用同队相邻阶段或同阶段其他队真实阵位补足；热图先验仅由训练系列赛生成；按系列赛隔离训练/验证/测试；无近战/远程伪标签",
        "supervision": {
            "positive": "hero_actual_next_42mm_firing_position",
            "alternatives": "actual_firing_positions_only",
            "negative_priority": [
                "same_team_same_phase",
                "same_team_adjacent_phase",
                "other_team_same_phase",
            ],
            "excluded_shortcuts": [
                "distance_from_current_position_to_candidate",
                "inferred_near_or_long_range_label",
                "hero_archetype_label",
            ],
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": np.round(mean, 7).tolist(),
        "feature_std": np.round(std, 7).tolist(),
        "teams": list(teams),
        "opponents": list(opponents),
        "candidate_count": CANDIDATES,
        "lookback_seconds": list(OFFSETS),
        "input_layout": {
            "numeric": len(FEATURE_NAMES),
            "team_one_hot": len(teams),
            "opponent_one_hot": len(opponents),
        },
        "base_score": {
            "kind": "log_training_heatmap_prior",
            "feature": "candidate_train_heatmap_prior",
            "epsilon": model.prior_epsilon,
        },
        "layers": [
            {"weight": rounded(first.weight), "bias": rounded(first.bias), "activation": "relu"},
            {"weight": rounded(second.weight), "bias": rounded(second.bias), "activation": "relu"},
            {"weight": rounded(head.weight), "bias": rounded(head.bias), "activation": "linear"},
        ],
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "acceptance": acceptance,
        "history": history,
        "metrics": metrics,
        "split_samples": {
            split: {
                "groups": len(values[0]),
                "team_groups": {
                    school: int(values[3][school]) for school in teams
                },
            }
            for split, values in built.items()
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "metrics": {
            split: {key: value for key, value in report.items() if key != "per_team"}
            for split, report in metrics.items()
        },
        "output": str(options.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
