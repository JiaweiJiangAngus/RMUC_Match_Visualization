#!/usr/bin/env python3
"""Train the browser target-kind policy from attributed regional hit events.

The referee export identifies the victim of a hit and the role/calibre of each
shot, but not the attacker of each individual hit.  Samples therefore use the
same-game, same-second, same-calibre attribution already used by the simulator:
when several roles fire, the target label is weighted by their shot shares.
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
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "target_selection_model.json"
ROLES = ("英雄", "步兵3", "步兵4", "哨兵", "空中")
GROUND_ROLES = ("英雄", "工程", "步兵3", "步兵4", "哨兵")
TARGETS = ("robot", "outpost", "base")
STRUCTURE_POSITIONS = {
    "红": {"base": (2.66, 7.5), "outpost": (11.0, 3.25)},
    "蓝": {"base": (25.34, 7.5), "outpost": (17.0, 11.75)},
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
    "own_low_hp_ratio",
    "enemy_low_hp_ratio",
    "distance_enemy_base_ratio",
    "distance_enemy_outpost_ratio",
    "nearest_enemy_robot_distance_ratio",
    "team_coins_ratio",
    "self_vulnerable",
)
ROW = {
    "id": 0, "type": 1, "side": 2, "hp": 3, "max_hp": 4,
    "x": 5, "y": 6, "coins": 10, "vulnerable": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--split-seed", type=int, default=7803)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--hidden-size", type=int, default=48)
    return parser.parse_args()


def opponent(side: str) -> str:
    return "蓝" if side == "红" else "红"


def hp_ratio(row: list | None) -> float:
    if not row:
        return 0.0
    return max(0.0, min(1.25, float(row[ROW["hp"]] or 0) / max(1.0, float(row[ROW["max_hp"]] or 1))))


def frame_index(rows: list[list]) -> dict[tuple[str, str], list]:
    return {(row[ROW["side"]], row[ROW["type"]]): row for row in rows}


def ground_summary(
    frame: dict[tuple[str, str], list],
    side: str,
) -> tuple[list[list], float, float, float]:
    rows = [frame.get((side, role)) for role in GROUND_ROLES]
    present = [row for row in rows if row]
    alive = [row for row in present if float(row[ROW["hp"]] or 0) > 0]
    total_hp = sum(max(0.0, float(row[ROW["hp"]] or 0)) for row in present)
    total_max_hp = sum(max(1.0, float(row[ROW["max_hp"]] or 1)) for row in present)
    low_hp = sum(1 for row in alive if hp_ratio(row) <= 0.4)
    return (
        alive,
        len(alive) / len(GROUND_ROLES),
        min(1.25, total_hp / max(1.0, total_max_hp)),
        low_hp / len(GROUND_ROLES),
    )


def distance_ratio(start: tuple[float, float], end: tuple[float, float]) -> float:
    return min(1.0, math.hypot(end[0] - start[0], end[1] - start[1]) / math.hypot(28, 15))


def contextual_features(
    frames: dict[int, dict[tuple[str, str], list]],
    second: int,
    side: str,
    role: str,
) -> list[float] | None:
    frame = frames.get(second) or frames.get(second - 1)
    if not frame:
        return None
    self_row = frame.get((side, role))
    if not self_row or self_row[ROW["x"]] is None or self_row[ROW["y"]] is None or float(self_row[ROW["hp"]] or 0) <= 0:
        return None
    enemy = opponent(side)
    own_base = frame.get((side, "基地"))
    own_outpost = frame.get((side, "前哨站"))
    enemy_base = frame.get((enemy, "基地"))
    enemy_outpost = frame.get((enemy, "前哨站"))
    previous = frames.get(max(0, second - 5), {}).get((side, role))
    current_hp = hp_ratio(self_row)
    previous_hp = hp_ratio(previous) if previous else current_hp
    _, own_alive_ratio, own_ground_hp, own_low_hp = ground_summary(frame, side)
    enemy_alive, enemy_alive_ratio, enemy_ground_hp, enemy_low_hp = ground_summary(frame, enemy)
    alive_rows = [
        row for row in enemy_alive
        if row[ROW["x"]] is not None and row[ROW["y"]] is not None
    ]
    position = (float(self_row[ROW["x"]]), float(self_row[ROW["y"]]))
    nearest_enemy = min(
        (
            distance_ratio(position, (float(row[ROW["x"]]), float(row[ROW["y"]])))
            for row in alive_rows
        ),
        default=1.0,
    )
    enemy_base_position = STRUCTURE_POSITIONS[enemy]["base"]
    enemy_outpost_position = STRUCTURE_POSITIONS[enemy]["outpost"]
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
        own_ground_hp,
        enemy_ground_hp,
        max(-1.0, min(1.0, own_ground_hp - enemy_ground_hp)),
        own_low_hp,
        enemy_low_hp,
        distance_ratio(position, enemy_base_position),
        distance_ratio(position, enemy_outpost_position),
        nearest_enemy,
        min(1.0, max(0.0, float(self_row[ROW["coins"]] or 0)) / 2000),
        1.0 if self_row[ROW["vulnerable"]] else 0.0,
    ]


def load_attribution(
    db_path: Path,
) -> tuple[
    dict[tuple[int, int, str, str], Counter],
    dict[int, list[tuple[int, str, str, str, float]]],
]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    shot_roles: dict[tuple[int, int, str, str], Counter] = defaultdict(Counter)
    for row in connection.execute(
        """
        SELECT game_id,CAST(时刻秒 AS INT) second,学校名,类别,机器人类型,COUNT(*) shots
        FROM events
        WHERE 事件类型='发弹' AND 类别 IN ('17mm','42mm')
          AND 机器人类型 IN ('英雄','步兵3','步兵4','哨兵','空中')
        GROUP BY game_id,second,学校名,类别,机器人类型
        """
    ):
        shot_roles[
            (int(row["game_id"]), int(row["second"]), row["学校名"], row["类别"])
        ][row["机器人类型"]] = int(row["shots"])

    hits: dict[int, Counter] = defaultdict(Counter)
    for row in connection.execute(
        """
        SELECT e.game_id,CAST(e.时刻秒 AS INT) second,e.类别,e.机器人类型 victim_type,
               ABS(e.数值) damage,
               CASE WHEN e.学校名=m.红方学校 THEN m.蓝方学校 ELSE m.红方学校 END attacker
        FROM events e JOIN matches m USING(game_id)
        WHERE e.事件类型='受击' AND e.类别 IN ('17mm','42mm')
        """
    ):
        target = "outpost" if row["victim_type"] == "前哨站" else "base" if row["victim_type"] == "基地" else "robot"
        hits[int(row["game_id"])][
            (int(row["second"]), row["attacker"], row["类别"], target)
        ] += max(1.0, float(row["damage"] or 0))
    connection.close()
    return shot_roles, {
        game_id: [(second, school, calibre, target, damage) for (second, school, calibre, target), damage in values.items()]
        for game_id, values in hits.items()
    }


def build_rows(
    paths: list[Path],
    shot_roles: dict[tuple[int, int, str, str], Counter],
    hits: dict[int, list[tuple[int, str, str, str, float]]],
    teams: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Counter]:
    team_index = {school: index for index, school in enumerate(teams)}
    role_index = {role: index for index, role in enumerate(ROLES)}
    features: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    sample_counts: Counter = Counter()
    for path in paths:
        game_id = int(path.name.removesuffix(".json.gz"))
        if game_id not in hits:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            game = json.load(handle)
        info = game["info"]
        school_side = {info["red"]: "红", info["blue"]: "蓝"}
        frames = {
            int(second): frame_index(rows)
            for second, rows in game.get("frames", {}).items()
        }
        for second, school, calibre, target, _damage in hits[game_id]:
            if school not in team_index or school not in school_side:
                continue
            firing = shot_roles.get((game_id, second, school, calibre), Counter())
            total_shots = sum(firing.values())
            if not total_shots:
                continue
            side = school_side[school]
            for role, shots in firing.items():
                if role not in role_index:
                    continue
                numeric = contextual_features(frames, second, side, role)
                if numeric is None:
                    continue
                team_role = team_index[school] * len(ROLES) + role_index[role]
                categorical = [team_index[school], role_index[role], team_role]
                features.append(numeric + categorical)
                labels.append(TARGETS.index(target))
                weights.append(shots / total_shots)
                sample_counts[(school, role, target)] += shots / total_shots
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        sample_counts,
    )


def encoded_inputs(
    rows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    team_count: int,
) -> np.ndarray:
    numeric_count = len(FEATURE_NAMES)
    numeric = (rows[:, :numeric_count] - mean) / std
    team = rows[:, numeric_count].astype(np.int64)
    role = rows[:, numeric_count + 1].astype(np.int64)
    team_role = rows[:, numeric_count + 2].astype(np.int64)
    one_hot_team = np.eye(team_count, dtype=np.float32)[team]
    one_hot_role = np.eye(len(ROLES), dtype=np.float32)[role]
    one_hot_team_role = np.eye(team_count * len(ROLES), dtype=np.float32)[team_role]
    return np.concatenate((numeric, one_hot_team, one_hot_role, one_hot_team_role), axis=1)


class TargetSelectionMlp(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(TARGETS)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def metrics(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> dict:
    probabilities = logits.softmax(dim=1)
    predicted = probabilities.argmax(dim=1)
    accuracy = float(((predicted == labels).float() * weights).sum() / weights.sum())
    loss = float((nn.functional.cross_entropy(logits, labels, reduction="none") * weights).sum() / weights.sum())
    per_target = {}
    for index, target in enumerate(TARGETS):
        mask = labels == index
        denominator = weights[mask].sum()
        per_target[target] = {
            "samples": round(float(denominator), 1),
            "recall": round(float((((predicted == index) & mask).float() * weights).sum() / denominator), 4)
            if denominator > 0 else 0,
            "mean_probability": round(float((probabilities[mask, index] * weights[mask]).sum() / denominator), 4)
            if denominator > 0 else 0,
        }
    return {"weighted_log_loss": round(loss, 6), "weighted_accuracy": round(accuracy, 4), "per_target": per_target}


def rounded_values(tensor: torch.Tensor) -> list:
    return np.round(tensor.detach().cpu().numpy(), 7).tolist()


def main() -> None:
    options = parse_args()
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    splits = load_group_splits(options.data_dir, options.split_seed)
    # The browser simulator publishes the 44 qualified teams in TEAMS.  Keep
    # the categorical embedding in exactly that order/scope so the exported
    # model cannot silently index a school that is absent from the product.
    ordered_teams = tuple(sorted(entry.school for entry in TEAMS))
    shot_roles, hits = load_attribution(options.db)
    built = {
        split: build_rows(paths, shot_roles, hits, ordered_teams)
        for split, paths in splits.items()
    }
    train_rows = built["train"][0]
    mean = train_rows[:, :len(FEATURE_NAMES)].mean(axis=0)
    std = np.maximum(train_rows[:, :len(FEATURE_NAMES)].std(axis=0), 1e-4)
    encoded = {
        split: encoded_inputs(values[0], mean, std, len(ordered_teams))
        for split, values in built.items()
    }
    loaders = {}
    for split in ("train", "val"):
        dataset = TensorDataset(
            torch.from_numpy(encoded[split]),
            torch.from_numpy(built[split][1]),
            torch.from_numpy(built[split][2]),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=split == "train",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetSelectionMlp(encoded["train"].shape[1], options.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=1e-4)
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
            total_weight = 0.0
            for values, labels, weights in loaders[split]:
                values, labels, weights = values.to(device), labels.to(device), weights.to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(values)
                    sample_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
                    loss = (sample_loss * weights).sum() / weights.sum()
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                total_loss += float((sample_loss * weights).sum().detach())
                total_weight += float(weights.sum())
            epoch_losses[split] = total_loss / max(1e-6, total_weight)
        history.append({"epoch": epoch, **{key: round(value, 6) for key, value in epoch_losses.items()}})
        print(f"epoch {epoch:02d}: train={epoch_losses['train']:.5f} val={epoch_losses['val']:.5f}")
        if epoch_losses["val"] < best_loss - 1e-5:
            best_loss = epoch_losses["val"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= options.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    split_metrics = {}
    with torch.no_grad():
        for split in ("val", "test"):
            values = torch.from_numpy(encoded[split]).to(device)
            labels = torch.from_numpy(built[split][1]).to(device)
            weights = torch.from_numpy(built[split][2]).to(device)
            split_metrics[split] = metrics(model(values), labels, weights)

    first: nn.Linear = model.layers[0]  # type: ignore[assignment]
    second: nn.Linear = model.layers[2]  # type: ignore[assignment]
    all_counts = sum((values[3] for values in built.values()), Counter())
    team_role_samples = {
        school: {
            role: round(sum(all_counts[(school, role, target)] for target in TARGETS), 1)
            for role in ROLES
        }
        for school in ordered_teams
    }
    output = {
        "schema_version": 2,
        "model_kind": "contextual_target_mlp",
        "source": "区域赛同局同秒同口径发弹角色归因；含双方地面机器人存活、总血量与低血量态势；按系列赛隔离训练/验证/测试",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": np.round(mean, 7).tolist(),
        "feature_std": np.round(std, 7).tolist(),
        "teams": list(ordered_teams),
        "roles": list(ROLES),
        "targets": list(TARGETS),
        "input_layout": {
            "numeric": len(FEATURE_NAMES),
            "team_one_hot": len(ordered_teams),
            "role_one_hot": len(ROLES),
            "team_role_one_hot": len(ordered_teams) * len(ROLES),
        },
        "layers": [
            {"weight": rounded_values(first.weight), "bias": rounded_values(first.bias), "activation": "relu"},
            {"weight": rounded_values(second.weight), "bias": rounded_values(second.bias), "activation": "softmax"},
        ],
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_loss, 6),
        "history": history,
        "metrics": split_metrics,
        "split_samples": {
            split: {
                "rows": len(values[0]),
                "weighted": round(float(values[2].sum()), 1),
            }
            for split, values in built.items()
        },
        "team_role_samples": team_role_samples,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "metrics": split_metrics, "output": str(options.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
