#!/usr/bin/env python3
"""Train the ranged-hero deployment state from real 42 mm base hits.

The regional export has no explicit deployment-state column.  It does,
however, retain the observable consequence required for direct supervision:
an ordinary 42 mm base hit is 200 damage and a deployed hit is 300 damage.
Large-energy-mechanism activations can also create 300-damage hits, so every
sample in the maximum possible 60-second large-buff window is excluded.
Dart hits are a separate calibre and never enter this dataset.

The model consumes four real battle-state frames and predicts whether the
hero was deployed at the labelled hit.  A second supervised head is trained
only on confirmed deployment samples: it predicts whether the hero leaves the
deployment point two to ten seconds after receiving a hit.  Physical invariants
(no chassis movement while deployed and the two-second exit delay) remain
hard runtime rules rather than learned approximations.
"""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "hero_deployment_transformer.json"
OFFSETS = (5, 3, 1, 0)
GROUND_ROLES = ("英雄", "工程", "步兵3", "步兵4", "哨兵")
FIELD_DIAGONAL = math.hypot(28, 15)
ROW = {
    "type": 1,
    "side": 2,
    "hp": 3,
    "max_hp": 4,
    "x": 5,
    "y": 6,
    "shots17": 8,
    "shots42": 9,
    "coins": 10,
    "vulnerable": 11,
}
FEATURE_NAMES = (
    "elapsed_ratio",
    "remaining_ratio",
    "hero_x_canonical_ratio",
    "hero_y_canonical_ratio",
    "hero_hp_ratio",
    "hero_hp_loss_1s",
    "hero_speed_1s_ratio",
    "hero_speed_3s_ratio",
    "hero_42mm_shots_1s_ratio",
    "own_base_hp_ratio",
    "own_outpost_hp_ratio",
    "enemy_base_hp_ratio",
    "enemy_outpost_hp_ratio",
    "own_ground_hp_ratio",
    "enemy_ground_hp_ratio",
    "nearest_enemy_distance_ratio",
    "team_coins_ratio",
    "hero_vulnerable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--split-seed", type=int, default=7803)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=64)
    return parser.parse_args()


def opponent(side: str) -> str:
    return "蓝" if side == "红" else "红"


def canonical(point: tuple[float, float], side: str) -> tuple[float, float]:
    return point if side == "红" else (28 - point[0], 15 - point[1])


def frame_index(rows: list[list]) -> dict[tuple[str, str], list]:
    return {(row[ROW["side"]], row[ROW["type"]]): row for row in rows}


def hp_ratio(row: list | None) -> float:
    if not row:
        return 0.0
    return float(
        np.clip(
            float(row[ROW["hp"]] or 0) / max(1.0, float(row[ROW["max_hp"]] or 1)),
            0,
            1.5,
        )
    )


def ground_hp_ratio(frame: dict[tuple[str, str], list], side: str) -> float:
    rows = [frame.get((side, role)) for role in GROUND_ROLES]
    present = [row for row in rows if row]
    hp = sum(max(0.0, float(row[ROW["hp"]] or 0)) for row in present)
    maximum = sum(max(1.0, float(row[ROW["max_hp"]] or 1)) for row in present)
    return min(1.5, hp / max(1.0, maximum))


def position(row: list | None) -> tuple[float, float] | None:
    if not row or row[ROW["x"]] is None or row[ROW["y"]] is None:
        return None
    return float(row[ROW["x"]]), float(row[ROW["y"]])


def game_payload(path: Path) -> tuple[dict, dict[int, dict[tuple[str, str], list]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        game = json.load(handle)
    return game["info"], {
        int(second): frame_index(rows)
        for second, rows in game.get("frames", {}).items()
    }


def load_labels(
    db_path: Path,
    allowed_teams: set[str],
) -> dict[int, list[tuple[int, str, int]]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    large_buffs: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT game_id,学校名,时刻秒
        FROM events
        WHERE 事件类型='增益' AND 类别='大能量机关增益'
          AND 机器人类型='英雄'
        """
    ):
        large_buffs[(int(row["game_id"]), row["学校名"])].append(float(row["时刻秒"]))

    labels: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    seen: set[tuple[int, int, str]] = set()
    for row in connection.execute(
        """
        SELECT e.game_id,CAST(e.时刻秒 AS INT) second,ABS(e.数值) damage,
               CASE
                 WHEN e.学校名=m.红方学校 THEN m.蓝方学校
                 WHEN e.学校名=m.蓝方学校 THEN m.红方学校
               END attacker
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='受击' AND e.类别='42mm'
          AND e.机器人类型='基地' AND ABS(e.数值) IN (200,300)
        ORDER BY e.game_id,e.时刻秒
        """
    ):
        game_id = int(row["game_id"])
        second = int(row["second"])
        school = row["attacker"]
        if school not in allowed_teams:
            continue
        # The exported buff row does not retain activated-arm count, so use the
        # ruleset's longest possible large-buff duration.  Ambiguous samples
        # are discarded, never guessed.
        if any(
            0 <= second - start <= 60
            for start in large_buffs.get((game_id, school), ())
        ):
            continue
        key = (game_id, second, school)
        label = 1 if int(round(float(row["damage"]))) == 300 else 0
        if key in seen:
            continue
        seen.add(key)
        labels[game_id].append((second, school, label))
    connection.close()
    return dict(labels)


def load_hero_hit_seconds(
    db_path: Path,
    allowed_teams: set[str],
) -> dict[tuple[int, str], list[int]]:
    """Load incoming projectile-hit seconds, deduplicated within each second."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    hit_seconds: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in connection.execute(
        """
        SELECT game_id,学校名,CAST(时刻秒 AS INT) second
        FROM events
        WHERE 事件类型='受击' AND 机器人类型='英雄'
          AND 类别 IN ('17mm','42mm')
        """
    ):
        school = row[1]
        if school in allowed_teams:
            hit_seconds[(int(row[0]), school)].add(int(row[2]))
    connection.close()
    return {
        key: sorted(seconds)
        for key, seconds in hit_seconds.items()
    }


def frame_features(
    frames: dict[int, dict[tuple[str, str], list]],
    second: int,
    side: str,
) -> list[float] | None:
    frame = frames.get(second)
    if not frame:
        return None
    hero = frame.get((side, "英雄"))
    hero_position = position(hero)
    if (
        not hero
        or not hero_position
        or float(hero[ROW["hp"]] or 0) <= 0
    ):
        return None
    previous = frames.get(second - 1, {}).get((side, "英雄"))
    previous3 = frames.get(second - 3, {}).get((side, "英雄"))
    previous_position = position(previous)
    previous3_position = position(previous3)
    current_hp = hp_ratio(hero)
    previous_hp = hp_ratio(previous) if previous else current_hp
    speed1 = (
        math.dist(hero_position, previous_position)
        if previous_position
        else 0.0
    )
    speed3 = (
        math.dist(hero_position, previous3_position) / 3
        if previous3_position
        else speed1
    )
    shots_now = float(hero[ROW["shots42"]] or 0)
    shots_previous = float(previous[ROW["shots42"]] or 0) if previous else shots_now
    enemy = opponent(side)
    enemy_rows = [
        frame.get((enemy, role))
        for role in GROUND_ROLES
    ]
    enemy_positions = [
        value
        for value in (position(row) for row in enemy_rows)
        if value is not None
    ]
    nearest_enemy = min(
        (math.dist(hero_position, target) for target in enemy_positions),
        default=FIELD_DIAGONAL,
    )
    canonical_position = canonical(hero_position, side)
    return [
        min(1.0, second / 420),
        max(0.0, 1 - second / 420),
        canonical_position[0] / 28,
        canonical_position[1] / 15,
        current_hp,
        max(0.0, min(1.0, previous_hp - current_hp)),
        min(1.5, speed1 / 3),
        min(1.5, speed3 / 3),
        min(1.0, max(0.0, shots_now - shots_previous) / 2),
        hp_ratio(frame.get((side, "基地"))),
        hp_ratio(frame.get((side, "前哨站"))),
        hp_ratio(frame.get((enemy, "基地"))),
        hp_ratio(frame.get((enemy, "前哨站"))),
        ground_hp_ratio(frame, side),
        ground_hp_ratio(frame, enemy),
        min(1.0, nearest_enemy / FIELD_DIAGONAL),
        min(1.0, max(0.0, float(hero[ROW["coins"]] or 0)) / 2000),
        1.0 if hero[ROW["vulnerable"]] else 0.0,
    ]


def build_samples(
    paths: list[Path],
    labels: dict[int, list[tuple[int, str, int]]],
    hero_hit_seconds: dict[tuple[int, str], list[int]],
    teams: tuple[str, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Counter,
]:
    team_index = {school: index for index, school in enumerate(teams)}
    other_opponent = len(teams)
    sequences: list[list[list[float]]] = []
    outcomes: list[int] = []
    exit_outcomes: list[int] = []
    team_ids: list[int] = []
    opponent_ids: list[int] = []
    counts: Counter = Counter()

    def append_sample(
        sequence: list[list[float] | None],
        state_label: int,
        exit_label: int,
        school: str,
        opponent_school: str,
        source: str,
    ) -> None:
        if any(values is None for values in sequence):
            return
        sequences.append(sequence)  # type: ignore[arg-type]
        outcomes.append(state_label)
        exit_outcomes.append(exit_label)
        team_ids.append(team_index[school])
        opponent_ids.append(team_index.get(opponent_school, other_opponent))
        counts[(school, "deployed" if state_label else "mobile")] += 1
        if state_label == 1:
            counts[(school, "exit" if exit_label else "hold")] += 1
            counts[(school, source)] += 1

    for path in paths:
        game_id = int(path.name.removesuffix(".json.gz"))
        if game_id not in labels:
            continue
        info, frames = game_payload(path)
        side_by_school = {info["red"]: "红", info["blue"]: "蓝"}
        opponent_by_school = {info["red"]: info["blue"], info["blue"]: info["red"]}
        for second, school, label in labels[game_id]:
            if school not in team_index or school not in side_by_school:
                continue
            side = side_by_school[school]
            sequence = [
                frame_features(frames, second - offset, side)
                for offset in OFFSETS
            ]
            append_sample(
                sequence,
                label,
                0 if label == 1 else -1,
                school,
                opponent_by_school[school],
                "confirmed_hit_hold",
            )

        # A clean 300-damage base hit confirms that this hero was deployed.
        # It remains directly observable as deployed while its chassis stays
        # within localisation noise of that point.  This creates real,
        # per-second hold/exit-transition targets instead of a stationary
        # pseudo-label: every episode starts with an exact 300-damage event.
        for school, side in side_by_school.items():
            confirmations = sorted(
                second
                for second, label_school, label in labels[game_id]
                if label_school == school and label == 1
            )
            if not confirmations:
                continue
            incoming_seconds = set(
                hero_hit_seconds.get((game_id, school), ())
            )
            sampled_seconds: set[int] = set()
            for confirmation_second in confirmations:
                confirmation_position = position(
                    frames.get(confirmation_second, {}).get((side, "英雄"))
                )
                if confirmation_position is None:
                    continue
                for observed_second in range(
                    confirmation_second + 1,
                    max(frames) + 1,
                ):
                    if observed_second in sampled_seconds:
                        continue
                    observed_position = position(
                        frames.get(observed_second, {}).get((side, "英雄"))
                    )
                    observed_hero = frames.get(observed_second, {}).get(
                        (side, "英雄")
                    )
                    if (
                        observed_position is None
                        or observed_hero is None
                        or float(observed_hero[ROW["hp"]] or 0) <= 0
                        or math.dist(
                            confirmation_position,
                            observed_position,
                        ) > 0.45
                    ):
                        break
                    future_positions = [
                        position(
                            frames.get(
                                observed_second + offset,
                                {},
                            ).get((side, "英雄"))
                        )
                        for offset in range(2, 11)
                    ]
                    left_point = any(
                        future is not None
                        and math.dist(observed_position, future) >= 0.45
                        for future in future_positions
                    )
                    sequence = [
                        frame_features(
                            frames,
                            observed_second - offset,
                            side,
                        )
                        for offset in OFFSETS
                    ]
                    received_hit = observed_second in incoming_seconds
                    append_sample(
                        sequence,
                        1,
                        int(left_point),
                        school,
                        opponent_by_school[school],
                        (
                            "incoming_hit_exit"
                            if received_hit and left_point
                            else "incoming_hit_hold"
                            if received_hit
                            else "trajectory_exit"
                            if left_point
                            else "trajectory_hold"
                        ),
                    )
                    sampled_seconds.add(observed_second)
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(outcomes, dtype=np.int64),
        np.asarray(exit_outcomes, dtype=np.int64),
        np.asarray(team_ids, dtype=np.int64),
        np.asarray(opponent_ids, dtype=np.int64),
        counts,
    )


class HeroDeploymentTransformer(nn.Module):
    def __init__(
        self,
        feature_count: int,
        team_count: int,
        opponent_count: int,
        sequence_length: int,
        model_dim: int,
        heads: int,
        feedforward: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_count, model_dim)
        self.team_embedding = nn.Embedding(team_count, model_dim)
        self.opponent_embedding = nn.Embedding(opponent_count, model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, model_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.08,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.state_head = nn.Linear(model_dim, 2)
        self.exit_head = nn.Linear(model_dim, 2)

    def encode(
        self,
        values: torch.Tensor,
        team_ids: torch.Tensor,
        opponent_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.input_projection(values)
        encoded = (
            encoded
            + self.team_embedding(team_ids)[:, None, :]
            + self.opponent_embedding(opponent_ids)[:, None, :]
            + self.position_embedding
        )
        return self.encoder(encoded)[:, -1]

    def forward(
        self,
        values: torch.Tensor,
        team_ids: torch.Tensor,
        opponent_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_head(self.encode(values, team_ids, opponent_ids))

    def forward_both(
        self,
        values: torch.Tensor,
        team_ids: torch.Tensor,
        opponent_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(values, team_ids, opponent_ids)
        return self.state_head(encoded), self.exit_head(encoded)


def binary_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + ties * 0.5) / (len(positive) * len(negative)))


def choose_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    best = (0.5, -1.0)
    for threshold in np.linspace(0.02, 0.98, 193):
        predicted = probabilities >= threshold
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        tn = int(((predicted == 0) & (labels == 0)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        positive_recall = tp / max(1, tp + fn)
        negative_recall = tn / max(1, tn + fp)
        balanced_accuracy = (positive_recall + negative_recall) / 2
        if balanced_accuracy > best[1]:
            best = (float(threshold), balanced_accuracy)
    return best[0]


def choose_exit_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Prefer retaining a deployed hero while preserving learned exit signal."""
    best = (0.5, -1.0)
    for threshold in np.linspace(0.02, 0.98, 193):
        predicted = probabilities >= threshold
        positive = labels == 1
        negative = ~positive
        exit_recall = float(predicted[positive].mean()) if positive.any() else 0
        hold_recall = float((~predicted[negative]).mean()) if negative.any() else 0
        if hold_recall < 0.9:
            continue
        score = exit_recall + hold_recall * 0.01
        if score > best[1]:
            best = (float(threshold), score)
    return best[0]


@torch.no_grad()
def evaluate(
    model: HeroDeploymentTransformer,
    tensors: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    threshold: float,
) -> dict:
    values, labels, _exit_labels, teams, opponents = tensors
    logits = model(values, teams, opponents)
    probabilities = logits.softmax(dim=1)[:, 1].cpu().numpy()
    truth = labels.cpu().numpy()
    predicted = probabilities >= threshold
    accuracy = float((predicted == truth).mean())
    positive = truth == 1
    negative = ~positive
    return {
        "samples": int(len(truth)),
        "deployed_samples": int(positive.sum()),
        "mobile_samples": int(negative.sum()),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(binary_auc(probabilities, truth), 4),
        "deployed_recall": round(float(predicted[positive].mean()), 4) if positive.any() else 0,
        "mobile_recall": round(float((~predicted[negative]).mean()), 4) if negative.any() else 0,
        "mean_deployed_probability": round(float(probabilities[positive].mean()), 4)
        if positive.any() else 0,
        "mean_mobile_probability": round(float(probabilities[negative].mean()), 4)
        if negative.any() else 0,
        "majority_baseline": round(
            max(float(positive.mean()), float(negative.mean())),
            4,
        ),
    }


@torch.no_grad()
def evaluate_exit(
    model: HeroDeploymentTransformer,
    tensors: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    threshold: float,
) -> dict:
    values, _state_labels, exit_labels, teams, opponents = tensors
    valid = exit_labels >= 0
    if not valid.any():
        return {
            "samples": 0,
            "exit_samples": 0,
            "hold_samples": 0,
            "accuracy": 0,
            "roc_auc": 0.5,
        }
    _state_logits, exit_logits = model.forward_both(
        values[valid],
        teams[valid],
        opponents[valid],
    )
    probabilities = exit_logits.softmax(dim=1)[:, 1].cpu().numpy()
    truth = exit_labels[valid].cpu().numpy()
    predicted = probabilities >= threshold
    positive = truth == 1
    negative = ~positive
    return {
        "samples": int(len(truth)),
        "exit_samples": int(positive.sum()),
        "hold_samples": int(negative.sum()),
        "accuracy": round(float((predicted == truth).mean()), 4),
        "roc_auc": round(binary_auc(probabilities, truth), 4),
        "exit_recall": round(float(predicted[positive].mean()), 4)
        if positive.any()
        else 0,
        "hold_recall": round(float((~predicted[negative]).mean()), 4)
        if negative.any()
        else 0,
        "mean_exit_probability": round(float(probabilities[positive].mean()), 4)
        if positive.any()
        else 0,
        "mean_hold_probability": round(float(probabilities[negative].mean()), 4)
        if negative.any()
        else 0,
        "majority_baseline": round(
            max(float(positive.mean()), float(negative.mean())),
            4,
        ),
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
    labels = load_labels(options.db, set(teams))
    hero_hit_seconds = load_hero_hit_seconds(options.db, set(teams))
    splits = load_group_splits(options.data_dir, options.split_seed)
    built = {
        split: build_samples(paths, labels, hero_hit_seconds, teams)
        for split, paths in splits.items()
    }
    for split, values in built.items():
        print(
            f"{split}: {len(values[0])} samples "
            f"({int(values[1].sum())} deployed / "
            f"{len(values[1]) - int(values[1].sum())} mobile)"
        )
    if not len(built["train"][0]) or not len(built["val"][0]) or not len(built["test"][0]):
        raise RuntimeError("deployment dataset split is empty")

    train_values = built["train"][0]
    mean = train_values.reshape(-1, len(FEATURE_NAMES)).mean(axis=0)
    std = np.maximum(
        train_values.reshape(-1, len(FEATURE_NAMES)).std(axis=0),
        1e-4,
    )
    encoded = {
        split: (values[0] - mean) / std
        for split, values in built.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = {
        split: (
            torch.from_numpy(encoded[split]).to(device),
            torch.from_numpy(values[1]).to(device),
            torch.from_numpy(values[2]).to(device),
            torch.from_numpy(values[3]).to(device),
            torch.from_numpy(values[4]).to(device),
        )
        for split, values in built.items()
    }
    loaders = {}
    for split in ("train", "val"):
        (
            values,
            labels_tensor,
            exit_labels_tensor,
            teams_tensor,
            opponents_tensor,
        ) = tensors[split]
        loaders[split] = DataLoader(
            TensorDataset(
                values,
                labels_tensor,
                exit_labels_tensor,
                teams_tensor,
                opponents_tensor,
            ),
            batch_size=options.batch_size,
            shuffle=split == "train",
        )

    model = HeroDeploymentTransformer(
        len(FEATURE_NAMES),
        len(teams),
        len(opponents),
        len(OFFSETS),
        options.model_dim,
        options.heads,
        options.feedforward,
    ).to(device)
    train_labels = built["train"][1]
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        len(train_labels) / np.maximum(1, class_counts) / 2,
        dtype=torch.float32,
        device=device,
    )
    train_exit_labels = built["train"][2]
    train_exit_valid = train_exit_labels >= 0
    exit_counts = np.bincount(
        train_exit_labels[train_exit_valid],
        minlength=2,
    )
    exit_class_weights = torch.tensor(
        max(1, int(train_exit_valid.sum())) / np.maximum(1, exit_counts) / 2,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=2e-4,
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, options.epochs + 1):
        losses = {}
        for split in ("train", "val"):
            model.train(split == "train")
            total = 0.0
            samples = 0
            for (
                values,
                labels_tensor,
                exit_labels_tensor,
                teams_tensor,
                opponents_tensor,
            ) in loaders[split]:
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                state_logits, exit_logits = model.forward_both(
                    values,
                    teams_tensor,
                    opponents_tensor,
                )
                state_loss = nn.functional.cross_entropy(
                    state_logits,
                    labels_tensor,
                    weight=class_weights,
                )
                exit_valid = exit_labels_tensor >= 0
                exit_loss = (
                    nn.functional.cross_entropy(
                        exit_logits[exit_valid],
                        exit_labels_tensor[exit_valid],
                        weight=exit_class_weights,
                    )
                    if exit_valid.any()
                    else torch.zeros((), device=device)
                )
                loss = state_loss + 0.45 * exit_loss
                if split == "train":
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total += float(loss.detach()) * len(values)
                samples += len(values)
            losses[split] = total / max(1, samples)
        history.append({
            "epoch": epoch,
            "train": round(losses["train"], 6),
            "val": round(losses["val"], 6),
        })
        if losses["val"] < best_loss - 1e-5:
            best_loss = losses["val"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch}: train={losses['train']:.5f} "
                f"val={losses['val']:.5f}"
            )
        if stale >= options.patience:
            break
    if best_state is None:
        raise RuntimeError("deployment model did not train")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(
            tensors["val"][0],
            tensors["val"][3],
            tensors["val"][4],
        )
        val_probabilities = val_logits.softmax(dim=1)[:, 1].cpu().numpy()
    threshold = choose_threshold(val_probabilities, built["val"][1])
    with torch.no_grad():
        val_exit_valid = tensors["val"][2] >= 0
        _state_logits, val_exit_logits = model.forward_both(
            tensors["val"][0][val_exit_valid],
            tensors["val"][3][val_exit_valid],
            tensors["val"][4][val_exit_valid],
        )
        val_exit_probabilities = (
            val_exit_logits.softmax(dim=1)[:, 1].cpu().numpy()
        )
    exit_threshold = choose_exit_threshold(
        val_exit_probabilities,
        built["val"][2][built["val"][2] >= 0],
    )
    metrics = {
        split: evaluate(model, tensors[split], threshold)
        for split in ("train", "val", "test")
    }
    exit_metrics = {
        split: evaluate_exit(model, tensors[split], exit_threshold)
        for split in ("train", "val", "test")
    }
    acceptance = {
        "held_out_auc_above_random": metrics["test"]["roc_auc"] >= 0.75,
        "state_head_recalls_both_states": (
            metrics["test"]["deployed_recall"] >= 0.6
            and metrics["test"]["mobile_recall"] >= 0.6
        ),
        "both_states_present_in_test": (
            metrics["test"]["deployed_samples"] > 0
            and metrics["test"]["mobile_samples"] > 0
        ),
        "exit_head_has_both_outcomes_in_test": (
            exit_metrics["test"]["exit_samples"] > 0
            and exit_metrics["test"]["hold_samples"] > 0
        ),
        "exit_head_held_out_auc": exit_metrics["test"]["roc_auc"] >= 0.65,
    }
    if not all(acceptance.values()):
        raise RuntimeError(
            "deployment Transformer acceptance gate failed: "
            + json.dumps(
                {
                    "state_metrics": metrics,
                    "exit_metrics": exit_metrics,
                    "acceptance": acceptance,
                },
                ensure_ascii=False,
            )
        )

    layer: nn.TransformerEncoderLayer = model.encoder.layers[0]
    with torch.no_grad():
        probe_logits = model(
            tensors["test"][0][:1],
            tensors["test"][3][:1],
            tensors["test"][4][:1],
        )
        probe_probability = float(probe_logits.softmax(dim=1)[0, 1])
        _probe_state_logits, probe_exit_logits = model.forward_both(
            tensors["test"][0][:1],
            tensors["test"][3][:1],
            tensors["test"][4][:1],
        )
        probe_exit_probability = float(
            probe_exit_logits.softmax(dim=1)[0, 1]
        )
    all_counts = Counter()
    for values in built.values():
        all_counts.update(values[5])
    team_samples = {
        school: {
            "deployed": int(all_counts[(school, "deployed")]),
            "mobile": int(all_counts[(school, "mobile")]),
            "exit": int(all_counts[(school, "exit")]),
            "hold": int(all_counts[(school, "hold")]),
        }
        for school in teams
    }
    deployed_positions = np.concatenate([
        values[0][values[1] == 1, -1, 2:4]
        for values in built.values()
    ])
    deployment_zone = {
        "min": [
            round(max(0.0, float(deployed_positions[:, 0].min() * 28) - 0.3), 3),
            round(max(0.0, float(deployed_positions[:, 1].min() * 15) - 0.3), 3),
        ],
        "max": [
            round(min(28.0, float(deployed_positions[:, 0].max() * 28) + 0.3), 3),
            round(min(15.0, float(deployed_positions[:, 1].max() * 15) + 0.3), 3),
        ],
    }
    output = {
        "schema_version": 2,
        "model_kind": "hero_deployment_transformer",
        "source": (
            "基地42mm真实受击：排除大能量机关激活后最长60秒污染窗，"
            "300为部署正例、200为非部署负例；飞镖类别完全排除；"
            "按系列赛隔离训练/验证/测试"
        ),
        "supervision": {
            "positive": "direct_42mm_base_hit_damage_300_outside_large_buff_window",
            "negative": "direct_42mm_base_hit_damage_200_outside_large_buff_window",
            "excluded": [
                "dart_hits",
                "all_samples_within_60_seconds_after_large_energy_mechanism_activation",
                "inferred_hero_archetype_labels",
                "stationary_pseudo_labels",
            ],
            "exit_positive": (
                "exact_300_confirmed_deployment_episode_then_hero_moves_"
                "at_least_0.45m_within_seconds_2_to_10"
            ),
            "exit_negative": (
                "exact_300_confirmed_deployment_episode_and_hero_remains_"
                "within_0.45m_through_seconds_2_to_10"
            ),
        },
        "states": ["mobile", "deployed"],
        "sequence_offsets_seconds": list(OFFSETS),
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": np.round(mean, 7).tolist(),
        "feature_std": np.round(std, 7).tolist(),
        "teams": list(teams),
        "opponents": list(opponents),
        "input_layout": {
            "sequence_length": len(OFFSETS),
            "feature_count": len(FEATURE_NAMES),
            "model_dim": options.model_dim,
            "heads": options.heads,
            "feedforward": options.feedforward,
        },
        "thresholds": {
            "enter": round(min(0.9, threshold + 0.05), 4),
            "hold": round(max(0.1, threshold - 0.12), 4),
            "exit": round(exit_threshold, 4),
        },
        "runtime_rules": {
            "ranged_only": True,
            "movement_locked_while_deployed": True,
            "exit_delay_seconds": 2,
            "ordinary_42mm_damage": 200,
            "deployed_base_42mm_damage": 300,
            "deployed_base_range_m": 27.5,
            "deployed_defense_ratio": 0.25,
            "canonical_deployment_zone_m": deployment_zone,
        },
        "team_samples": team_samples,
        "layers": {
            "input_projection": {
                "weight": rounded(model.input_projection.weight),
                "bias": rounded(model.input_projection.bias),
            },
            "team_embedding": rounded(model.team_embedding.weight),
            "opponent_embedding": rounded(model.opponent_embedding.weight),
            "position_embedding": rounded(model.position_embedding[0]),
            "self_attention": {
                "in_proj_weight": rounded(layer.self_attn.in_proj_weight),
                "in_proj_bias": rounded(layer.self_attn.in_proj_bias),
                "out_proj_weight": rounded(layer.self_attn.out_proj.weight),
                "out_proj_bias": rounded(layer.self_attn.out_proj.bias),
            },
            "feedforward": {
                "linear1_weight": rounded(layer.linear1.weight),
                "linear1_bias": rounded(layer.linear1.bias),
                "linear2_weight": rounded(layer.linear2.weight),
                "linear2_bias": rounded(layer.linear2.bias),
            },
            "norm1": {
                "weight": rounded(layer.norm1.weight),
                "bias": rounded(layer.norm1.bias),
                "eps": layer.norm1.eps,
            },
            "norm2": {
                "weight": rounded(layer.norm2.weight),
                "bias": rounded(layer.norm2.bias),
                "eps": layer.norm2.eps,
            },
            "state_head": {
                "weight": rounded(model.state_head.weight),
                "bias": rounded(model.state_head.bias),
            },
            "exit_head": {
                "weight": rounded(model.exit_head.weight),
                "bias": rounded(model.exit_head.bias),
            },
        },
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "decision_threshold": round(threshold, 4),
        "history": history,
        "metrics": metrics,
        "exit_metrics": exit_metrics,
        "acceptance": acceptance,
        "inference_probe": {
            "standardized_sequence": np.round(encoded["test"][0], 7).tolist(),
            "team_index": int(built["test"][3][0]),
            "opponent_index": int(built["test"][4][0]),
            "deployed_probability": round(probe_probability, 7),
            "exit_probability": round(probe_exit_probability, 7),
        },
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"wrote {options.output} ({options.output.stat().st_size / 1024:.1f} KiB), "
        f"test={json.dumps(metrics['test'], ensure_ascii=False)}"
    )


if __name__ == "__main__":
    main()
