#!/usr/bin/env python3
"""Train a team-agnostic, multi-horizon RMUC robot trajectory baseline.

The model consumes the recent multi-agent battlefield state and predicts the
selected robot's position 1/3/5/10/15 seconds into the future.  Games from the
same match are always kept in the same split to avoid round-to-round leakage.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "docs" / "data" / "games"
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "trajectory_mlp.pt"

FIELD_WIDTH_M = 28.0
FIELD_HEIGHT_M = 15.0
REGULATION_DURATION_S = 420
MOBILE_TYPES = ("英雄", "工程", "步兵3", "步兵4", "哨兵", "空中")
STRUCTURE_TYPES = ("基地", "前哨站")
SIDES = ("红", "蓝")
HISTORY_OFFSETS = (0, 1, 3, 5)
DEFAULT_HORIZONS = (1, 3, 5, 10, 15)

# Compact exported row layout:
# id,type,side,hp,max_hp,x,y,yaw,ammo17,ammo42,coins,vulnerable
ID, TYPE, SIDE, HP, MAX_HP, X, Y, YAW, AMMO17, AMMO42, COINS, VULNERABLE = range(12)


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    output: str
    horizons: tuple[int, ...]
    stride: int
    seed: int
    max_train_samples: int
    max_val_samples: int
    max_test_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    max_step_m: float


class TrajectoryMLP(nn.Module):
    """Small tabular baseline; the zero-initialized head starts as 'stay put'."""

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous, horizon_count * 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.horizon_count = horizon_count

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(inputs)).reshape(-1, self.horizon_count, 2)


def other_side(side: str) -> str:
    return "蓝" if side == "红" else "红"


def valid_position(row: list | None) -> bool:
    if row is None or row[X] is None or row[Y] is None:
        return False
    x, y = float(row[X]), float(row[Y])
    # Exact (0, 0) is used by a small number of invalid localization samples.
    return (
        0.0 <= x <= FIELD_WIDTH_M
        and 0.0 <= y <= FIELD_HEIGHT_M
        and not (x == 0.0 and y == 0.0)
    )


def canonical_xy(row: list, target_side: str) -> tuple[float, float]:
    x, y = float(row[X]), float(row[Y])
    if target_side == "蓝":
        x, y = FIELD_WIDTH_M - x, FIELD_HEIGHT_M - y
    return x / FIELD_WIDTH_M, y / FIELD_HEIGHT_M


def hp_ratio(row: list | None) -> float:
    if row is None or not row[MAX_HP]:
        return 0.0
    return float(np.clip(float(row[HP] or 0.0) / float(row[MAX_HP]), 0.0, 1.5))


def ammo_total(row: list | None) -> float:
    if row is None:
        return 0.0
    return float(row[AMMO17] or 0.0) + float(row[AMMO42] or 0.0)


def make_feature_names() -> list[str]:
    names: list[str] = []
    for offset in HISTORY_OFFSETS:
        time_name = "now" if offset == 0 else f"t_minus_{offset}"
        for relation in ("own", "enemy"):
            for robot_type in MOBILE_TYPES:
                prefix = f"{time_name}.{relation}.{robot_type}"
                names.extend(
                    [f"{prefix}.x", f"{prefix}.y", f"{prefix}.hp", f"{prefix}.present"]
                )
        for relation in ("own", "enemy"):
            for structure_type in STRUCTURE_TYPES:
                names.append(f"{time_name}.{relation}.{structure_type}.hp")
        names.extend([f"{time_name}.own.coins", f"{time_name}.enemy.coins"])

    names.extend(
        [
            "target.x",
            "target.y",
            "target.hp",
            "target.heading_sin",
            "target.heading_cos",
            "target.heading_present",
            "target.vulnerable",
        ]
    )
    for offset in HISTORY_OFFSETS[1:]:
        names.extend(
            [
                f"target.vx_{offset}_norm_per_s",
                f"target.vy_{offset}_norm_per_s",
                f"target.ammo_rate_{offset}",
            ]
        )
    names.extend(["time.elapsed", "time.remaining"])
    names.extend(f"target.type.{robot_type}" for robot_type in MOBILE_TYPES)
    return names


FEATURE_NAMES = make_feature_names()
TARGET_X_INDEX = FEATURE_NAMES.index("target.x")
TARGET_Y_INDEX = FEATURE_NAMES.index("target.y")
TARGET_VX3_INDEX = FEATURE_NAMES.index("target.vx_3_norm_per_s")
TARGET_VY3_INDEX = FEATURE_NAMES.index("target.vy_3_norm_per_s")


def index_frame(rows: list[list]) -> dict[tuple[str, str], list]:
    return {(row[SIDE], row[TYPE]): row for row in rows}


def normalized_coins(frame: dict[tuple[str, str], list], side: str) -> float:
    for robot_type in (*MOBILE_TYPES, *STRUCTURE_TYPES):
        row = frame.get((side, robot_type))
        if row is not None and row[COINS] is not None:
            # 2,000 is a soft scale, not a hard clip; the feature is standardized later.
            return float(row[COINS]) / 2000.0
    return 0.0


def sample_features(
    frames: dict[int, dict[tuple[str, str], list]],
    second: int,
    target_side: str,
    target_type: str,
    duration: int,
) -> list[float]:
    values: list[float] = []
    enemy_side = other_side(target_side)

    for offset in HISTORY_OFFSETS:
        frame = frames[second - offset]
        for side in (target_side, enemy_side):
            for robot_type in MOBILE_TYPES:
                row = frame.get((side, robot_type))
                present = valid_position(row)
                if present:
                    x, y = canonical_xy(row, target_side)  # type: ignore[arg-type]
                    values.extend((x, y, hp_ratio(row), 1.0))
                else:
                    values.extend((0.0, 0.0, hp_ratio(row), 0.0))
        for side in (target_side, enemy_side):
            for structure_type in STRUCTURE_TYPES:
                values.append(hp_ratio(frame.get((side, structure_type))))
        values.extend(
            (normalized_coins(frame, target_side), normalized_coins(frame, enemy_side))
        )

    current = frames[second][(target_side, target_type)]
    current_x, current_y = canonical_xy(current, target_side)
    yaw = current[YAW]
    if yaw is None:
        heading_sin, heading_cos, heading_present = 0.0, 0.0, 0.0
    else:
        canonical_yaw = float(yaw) + (180.0 if target_side == "蓝" else 0.0)
        radians = math.radians(canonical_yaw)
        heading_sin, heading_cos, heading_present = math.sin(radians), math.cos(radians), 1.0
    values.extend(
        (
            current_x,
            current_y,
            hp_ratio(current),
            heading_sin,
            heading_cos,
            heading_present,
            float(bool(current[VULNERABLE])),
        )
    )

    for offset in HISTORY_OFFSETS[1:]:
        previous = frames[second - offset][(target_side, target_type)]
        previous_x, previous_y = canonical_xy(previous, target_side)
        values.extend(
            (
                (current_x - previous_x) / offset,
                (current_y - previous_y) / offset,
                (ammo_total(current) - ammo_total(previous)) / offset / 50.0,
            )
        )

    safe_duration = max(1, duration)
    values.extend((second / safe_duration, max(0, duration - second) / safe_duration))
    values.extend(float(target_type == candidate) for candidate in MOBILE_TYPES)
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError(f"feature length {len(values)} != {len(FEATURE_NAMES)}")
    return values


def iter_game_samples(
    path: Path,
    horizons: Sequence[int],
    stride: int,
    max_step_m: float,
) -> Iterator[tuple[list[float], list[tuple[float, float]]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        game = json.load(handle)
    frames = {int(second): index_frame(rows) for second, rows in game["frames"].items()}
    if not frames:
        return
    min_second, max_second = min(frames), max(frames)
    max_history, max_horizon = max(HISTORY_OFFSETS), max(horizons)
    # Actual match duration is only known after the match and would leak whether
    # a game ends early.  Regulation time is available both live and offline.
    duration = REGULATION_DURATION_S

    start = min_second + max_history
    stop = max_second - max_horizon
    for second in range(start, stop + 1, stride):
        required_seconds = [second - offset for offset in HISTORY_OFFSETS]
        required_seconds.extend(second + horizon for horizon in horizons)
        if any(required not in frames for required in required_seconds):
            continue
        for target_side in SIDES:
            for target_type in MOBILE_TYPES:
                key = (target_side, target_type)
                target_rows = [frames[t].get(key) for t in range(second - max_history, second + max_horizon + 1)]
                if any(not valid_position(row) or float(row[HP] or 0.0) <= 0 for row in target_rows):
                    continue

                # Remove localization glitches and respawn teleports from the supervised target.
                positions = [canonical_xy(row, target_side) for row in target_rows]  # type: ignore[arg-type]
                implausible = False
                for (x0, y0), (x1, y1) in zip(positions, positions[1:]):
                    distance = math.hypot(
                        (x1 - x0) * FIELD_WIDTH_M,
                        (y1 - y0) * FIELD_HEIGHT_M,
                    )
                    if distance > max_step_m:
                        implausible = True
                        break
                if implausible:
                    continue

                try:
                    features = sample_features(
                        frames, second, target_side, target_type, duration
                    )
                except KeyError:
                    continue
                targets = [canonical_xy(frames[second + horizon][key], target_side) for horizon in horizons]
                yield features, targets


def load_group_splits(data_dir: Path, seed: int) -> dict[str, list[Path]]:
    catalog_path = data_dir.parent / "catalog.json"
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)

    game_to_group: dict[int, str] = {}
    for group, rounds in catalog["rounds"].items():
        for item in rounds:
            game_to_group[int(item["game_id"])] = group

    groups = sorted(set(game_to_group.values()))
    random.Random(seed).shuffle(groups)
    train_end = round(len(groups) * 0.70)
    val_end = train_end + round(len(groups) * 0.15)
    group_split = {
        group: "train" if index < train_end else "val" if index < val_end else "test"
        for index, group in enumerate(groups)
    }
    result: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for path in sorted(data_dir.glob("*.json.gz")):
        game_id = int(path.name.removesuffix(".json.gz"))
        group = game_to_group[game_id]
        result[group_split[group]].append(path)
    return result


def capped_rows(
    x: np.ndarray,
    y: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or len(x) <= limit:
        return x, y
    selected = np.sort(rng.choice(len(x), size=limit, replace=False))
    return x[selected], y[selected]


def build_split(
    paths: Iterable[Path],
    split: str,
    horizons: Sequence[int],
    stride: int,
    max_step_m: float,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    path_list = list(paths)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    started = time.monotonic()
    for index, path in enumerate(path_list, 1):
        samples = list(iter_game_samples(path, horizons, stride, max_step_m))
        if samples:
            x_part = np.asarray([sample[0] for sample in samples], dtype=np.float32)
            y_part = np.asarray([sample[1] for sample in samples], dtype=np.float32)
            x_parts.append(x_part)
            y_parts.append(y_part)
        if index == 1 or index % 50 == 0 or index == len(path_list):
            count = sum(len(part) for part in x_parts)
            print(
                f"[{split}] games {index}/{len(path_list)}, usable samples {count:,}, "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
    if not x_parts:
        raise RuntimeError(f"split {split!r} contains no usable samples")
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    return capped_rows(x, y, limit, rng)


def predict_batches(
    model: nn.Module,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            normalized = (x[start : start + batch_size] - mean) / std
            inputs = torch.from_numpy(normalized).to(device)
            outputs.append(model(inputs).cpu().numpy())
    return np.concatenate(outputs)


def metric_summary(errors_m: np.ndarray) -> dict[str, float]:
    return {
        "mean_error_m": round(float(np.mean(errors_m)), 4),
        "median_error_m": round(float(np.median(errors_m)), 4),
        "p90_error_m": round(float(np.quantile(errors_m, 0.90)), 4),
        "within_1m": round(float(np.mean(errors_m <= 1.0)), 4),
        "within_2m": round(float(np.mean(errors_m <= 2.0)), 4),
    }


def zone_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    # 4 m x 3 m tactical cells: 7 columns by 5 rows.
    pred_x = np.minimum((predicted[:, 0] * 7).astype(int), 6)
    pred_y = np.minimum((predicted[:, 1] * 5).astype(int), 4)
    true_x = np.minimum((actual[:, 0] * 7).astype(int), 6)
    true_y = np.minimum((actual[:, 1] * 5).astype(int), 4)
    return round(float(np.mean((pred_x == true_x) & (pred_y == true_y))), 4)


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    horizons: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> dict:
    residual = predict_batches(model, x, mean, std, device, batch_size)
    current = x[:, [TARGET_X_INDEX, TARGET_Y_INDEX]]
    learned = np.clip(current[:, None, :] + residual, 0.0, 1.0)
    stationary = np.repeat(current[:, None, :], len(horizons), axis=1)
    velocity = x[:, [TARGET_VX3_INDEX, TARGET_VY3_INDEX]]
    constant_velocity = np.stack(
        [np.clip(current + velocity * horizon, 0.0, 1.0) for horizon in horizons], axis=1
    )

    velocity_mps = np.linalg.norm(
        x[:, [TARGET_VX3_INDEX, TARGET_VY3_INDEX]]
        * np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32),
        axis=1,
    )
    subsets = {
        "all": np.ones(len(x), dtype=bool),
        "moving": velocity_mps >= 0.15,
    }
    report: dict[str, dict] = {
        "sample_counts": {name: int(mask.sum()) for name, mask in subsets.items()}
    }
    scale = np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32)
    for index, horizon in enumerate(horizons):
        horizon_report: dict[str, dict] = {}
        for subset_name, mask in subsets.items():
            subset_report: dict[str, dict] = {}
            for name, prediction in (
                ("stationary", stationary[:, index]),
                ("constant_velocity", constant_velocity[:, index]),
                ("learned_mlp", learned[:, index]),
            ):
                errors = np.linalg.norm(
                    (prediction[mask] - y[mask, index]) * scale, axis=1
                )
                values = metric_summary(errors)
                values["zone_accuracy"] = zone_accuracy(
                    prediction[mask], y[mask, index]
                )
                subset_report[name] = values
            horizon_report[subset_name] = subset_report
        report[f"{horizon}s"] = horizon_report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--stride", type=int, default=5, help="sample every N seconds")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=300_000)
    parser.add_argument("--max-val-samples", type=int, default=60_000)
    parser.add_argument("--max-test-samples", type=int, default=80_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-step-m", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(sorted(set(args.horizons)))
    if any(horizon <= 0 for horizon in horizons):
        raise SystemExit("all horizons must be positive")
    config = TrainConfig(
        data_dir=str(args.data_dir.resolve()),
        output=str(args.output.resolve()),
        horizons=horizons,
        stride=args.stride,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_step_m=args.max_step_m,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    splits = load_group_splits(args.data_dir, args.seed)
    print(
        "match-group split games: "
        + ", ".join(f"{name}={len(paths)}" for name, paths in splits.items()),
        flush=True,
    )

    limits = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    arrays = {
        split: build_split(
            paths,
            split,
            horizons,
            args.stride,
            args.max_step_m,
            limits[split],
            rng,
        )
        for split, paths in splits.items()
    }
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    print(
        f"capped samples: train={len(x_train):,}, val={len(x_val):,}, test={len(x_test):,}; "
        f"features={x_train.shape[1]}",
        flush=True,
    )

    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    current_train = x_train[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    current_val = x_val[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    train_targets = y_train - current_train
    val_targets = y_val - current_val
    normalized_train = ((x_train - mean) / std).astype(np.float32, copy=False)
    normalized_val = ((x_val - mean) / std).astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device: {device}", flush=True)
    model = TrajectoryMLP(len(FEATURE_NAMES), len(horizons)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = nn.SmoothL1Loss(beta=0.02)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized_train), torch.from_numpy(train_targets)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_inputs = torch.from_numpy(normalized_val)
    val_labels = torch.from_numpy(val_targets)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(inputs)
            seen += len(inputs)

        model.eval()
        val_loss_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(val_inputs), args.batch_size):
                inputs = val_inputs[start : start + args.batch_size].to(device)
                targets = val_labels[start : start + args.batch_size].to(device)
                val_loss_sum += float(loss_fn(model(inputs), targets)) * len(inputs)
        train_loss = total_loss / seen
        val_loss = val_loss_sum / len(val_inputs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch {epoch:02d}: train={train_loss:.6f} val={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    metrics = evaluate(
        model, x_test, y_test, mean, std, horizons, device, args.batch_size
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "model_kind": "multi_agent_trajectory_mlp",
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "model_kwargs": {
            "input_dim": len(FEATURE_NAMES),
            "horizon_count": len(horizons),
            "hidden_sizes": (256, 256, 128),
            "dropout": 0.08,
        },
        "feature_names": FEATURE_NAMES,
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "horizons": horizons,
        "history_offsets": HISTORY_OFFSETS,
        "field_size_m": (FIELD_WIDTH_M, FIELD_HEIGHT_M),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "sample_counts": {split: len(value[0]) for split, value in arrays.items()},
        "test_metrics": metrics,
    }
    torch.save(checkpoint, args.output)
    report_path = args.output.with_suffix(".metrics.json")
    report = {
        "checkpoint": str(args.output.resolve()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "sample_counts": checkpoint["sample_counts"],
        "metrics": metrics,
        "history": history,
        "config": asdict(config),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {args.output}", flush=True)
    print(f"saved metrics: {report_path}", flush=True)


if __name__ == "__main__":
    main()
