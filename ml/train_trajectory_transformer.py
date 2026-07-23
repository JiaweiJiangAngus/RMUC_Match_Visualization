#!/usr/bin/env python3
"""Train a real temporal Transformer on RMUC multi-agent trajectories."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .train_trajectory import (
        DEFAULT_DATA_DIR,
        DEFAULT_HORIZONS,
        FEATURE_NAMES,
        FIELD_HEIGHT_M,
        FIELD_WIDTH_M,
        HISTORY_OFFSETS,
        HP,
        MOBILE_TYPES,
        REGULATION_DURATION_S,
        SIDES,
        TARGET_VX3_INDEX,
        TARGET_VY3_INDEX,
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        capped_rows,
        canonical_xy,
        evaluate,
        hp_ratio,
        index_frame,
        load_group_splits,
        sample_features,
        valid_position,
    )
    from .trajectory_transformer import TemporalBattlefieldTransformer
except ImportError:
    from train_trajectory import (  # type: ignore[no-redef]
        DEFAULT_DATA_DIR,
        DEFAULT_HORIZONS,
        FEATURE_NAMES,
        FIELD_HEIGHT_M,
        FIELD_WIDTH_M,
        HISTORY_OFFSETS,
        HP,
        MOBILE_TYPES,
        REGULATION_DURATION_S,
        SIDES,
        TARGET_VX3_INDEX,
        TARGET_VY3_INDEX,
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        capped_rows,
        canonical_xy,
        evaluate,
        hp_ratio,
        index_frame,
        load_group_splits,
        sample_features,
        valid_position,
    )
    from trajectory_transformer import TemporalBattlefieldTransformer  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "trajectory_transformer.pt"
DAMAGE_FEATURE_NAMES = tuple(f"target.hp_loss_{offset}" for offset in HISTORY_OFFSETS[1:])
CANONICAL_SERVICE_ZONES = (
    {
        "shape": "rectangle",
        "center": (2.0, 1.65),
        "radius": (1.85, 1.45),
    },
    {
        "shape": "ellipse",
        "center": (2.66, 7.5),
        "radius": (1.8, 1.55),
    },
    {
        "shape": "half_ellipse",
        "center": (11.0, 3.25),
        "radius": (1.55, 1.35),
        "direction": (0.576, 0.817),
    },
)


@dataclass(frozen=True)
class TransformerTrainConfig:
    data_dir: str
    output: str
    horizons: tuple[int, ...]
    stride: int
    seed: int
    split_seed: int
    max_train_samples: int
    max_val_samples: int
    max_test_samples: int
    max_games_per_split: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    max_step_m: float
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--split-seed", type=int, default=7803,
        help="fixed match-group split selected for broader per-team validation/test coverage",
    )
    parser.add_argument("--max-train-samples", type=int, default=300_000)
    parser.add_argument("--max-val-samples", type=int, default=60_000)
    parser.add_argument("--max-test-samples", type=int, default=80_000)
    parser.add_argument(
        "--max-games-per-split", type=int, default=0,
        help="debug-only cap; 0 uses every game in each match-group split",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-step-m", type=float, default=8.0)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser.parse_args()


def collect_school_names(data_dir: Path) -> tuple[str, ...]:
    catalog_path = data_dir.parent / "catalog.json"
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    schools = {
        str(item[side])
        for rounds in catalog["rounds"].values()
        for item in rounds
        for side in ("red", "blue")
    }
    return tuple(sorted(schools))


def transformer_feature_names(school_names: Sequence[str]) -> list[str]:
    return [
        *FEATURE_NAMES,
        *DAMAGE_FEATURE_NAMES,
        *(f"target.school.{school}" for school in school_names),
        *(f"opponent.school.{school}" for school in school_names),
    ]


def transformer_sample_features(
    frames: dict[int, dict[tuple[str, str], list]],
    second: int,
    target_side: str,
    target_type: str,
    duration: int,
    target_school: str,
    opponent_school: str,
    school_names: Sequence[str],
) -> list[float]:
    values = sample_features(frames, second, target_side, target_type, duration)
    current = frames[second][(target_side, target_type)]
    current_hp = hp_ratio(current)
    for offset in HISTORY_OFFSETS[1:]:
        previous = frames[second - offset].get((target_side, target_type))
        values.append(max(0.0, hp_ratio(previous) - current_hp))
    values.extend(float(target_school == school) for school in school_names)
    values.extend(float(opponent_school == school) for school in school_names)
    return values


def iter_transformer_samples(
    path: Path,
    horizons: Sequence[int],
    stride: int,
    max_step_m: float,
    school_names: Sequence[str],
) -> Iterator[tuple[list[float], list[tuple[float, float]]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        game = json.load(handle)
    frames = {int(second): index_frame(rows) for second, rows in game["frames"].items()}
    if not frames:
        return
    min_second, max_second = min(frames), max(frames)
    max_history, max_horizon = max(HISTORY_OFFSETS), max(horizons)
    start = min_second + max_history
    stop = max_second - max_horizon
    for second in range(start, stop + 1, stride):
        required = [second - offset for offset in HISTORY_OFFSETS]
        required.extend(second + horizon for horizon in horizons)
        if any(value not in frames for value in required):
            continue
        for target_side in SIDES:
            target_school = str(game["info"]["red" if target_side == "红" else "blue"])
            opponent_school = str(game["info"]["blue" if target_side == "红" else "red"])
            for target_type in MOBILE_TYPES:
                key = (target_side, target_type)
                target_rows = [
                    frames[value].get(key)
                    for value in range(second - max_history, second + max_horizon + 1)
                ]
                if any(not valid_position(row) or float(row[HP] or 0) <= 0 for row in target_rows):
                    continue
                positions = [canonical_xy(row, target_side) for row in target_rows]
                if any(
                    math.hypot(
                        (right[0] - left[0]) * FIELD_WIDTH_M,
                        (right[1] - left[1]) * FIELD_HEIGHT_M,
                    ) > max_step_m
                    for left, right in zip(positions, positions[1:])
                ):
                    continue
                try:
                    features = transformer_sample_features(
                        frames, second, target_side, target_type,
                        REGULATION_DURATION_S, target_school, opponent_school,
                        school_names,
                    )
                except KeyError:
                    continue
                targets = [
                    canonical_xy(frames[second + horizon][key], target_side)
                    for horizon in horizons
                ]
                yield features, targets


def build_transformer_split(
    paths: Iterable[Path],
    split: str,
    horizons: Sequence[int],
    stride: int,
    max_step_m: float,
    limit: int,
    rng: np.random.Generator,
    school_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    path_list = list(paths)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    started = time.monotonic()
    for index, path in enumerate(path_list, 1):
        samples = list(iter_transformer_samples(
            path, horizons, stride, max_step_m, school_names,
        ))
        if samples:
            x_parts.append(np.asarray([item[0] for item in samples], dtype=np.float32))
            y_parts.append(np.asarray([item[1] for item in samples], dtype=np.float32))
        if index == 1 or index % 50 == 0 or index == len(path_list):
            count = sum(len(part) for part in x_parts)
            print(
                f"[{split}] games {index}/{len(path_list)}, usable samples {count:,}, "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )
    if not x_parts:
        raise RuntimeError(f"split {split!r} contains no usable samples")
    return capped_rows(np.concatenate(x_parts), np.concatenate(y_parts), limit, rng)


def sample_weights(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Reduce stationary/service-zone domination without deleting evidence."""
    velocity = np.linalg.norm(
        features[:, [TARGET_VX3_INDEX, TARGET_VY3_INDEX]]
        * np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32),
        axis=1,
    )
    current = features[:, [TARGET_X_INDEX, TARGET_Y_INDEX]]
    final_displacement = np.linalg.norm(
        (targets[:, -1] - current)
        * np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32),
        axis=1,
    )
    current_m = current * np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32)
    service = np.zeros(len(features), dtype=bool)
    for zone in CANONICAL_SERVICE_ZONES:
        center = zone["center"]
        radius = zone["radius"]
        dx = (current_m[:, 0] - center[0]) / radius[0]
        dy = (current_m[:, 1] - center[1]) / radius[1]
        if zone["shape"] == "rectangle":
            inside = (np.abs(dx) <= 1) & (np.abs(dy) <= 1)
        else:
            inside = dx * dx + dy * dy <= 1
        if zone["shape"] == "half_ellipse":
            direction = zone["direction"]
            inside &= (
                (current_m[:, 0] - center[0]) * direction[0]
                + (current_m[:, 1] - center[1]) * direction[1]
                >= 0
            )
        service |= inside

    # Keep valid firing-position holds learnable.  Only stationary service-zone
    # occupancy is strongly suppressed; this preserves behaviours such as a
    # long-range hero holding an anchor until incoming damage forces movement.
    weights = np.where(velocity >= 0.15, 1.0, 0.70).astype(np.float32)
    stationary_service = service & (final_displacement < 0.75)
    weights[stationary_service] *= 0.10
    damage_columns = [
        feature_names.index(name) for name in DAMAGE_FEATURE_NAMES
        if name in feature_names
    ]
    if damage_columns:
        recently_damaged = features[:, damage_columns].max(axis=1) > 0.005
        weights[recently_damaged] *= 1.50
    # Preserve a stable average learning rate after weighting.
    weights /= max(1e-6, float(weights.mean()))
    return weights


def weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_sample = F.smooth_l1_loss(
        prediction, target, beta=0.02, reduction="none"
    ).mean(dim=(1, 2))
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def rename_metric_model(report: dict) -> dict:
    for key, horizon in report.items():
        if not key.endswith("s") or not key[:-1].isdigit() or not isinstance(horizon, dict):
            continue
        for subset in horizon.values():
            if "learned_mlp" in subset:
                subset["learned_transformer"] = subset.pop("learned_mlp")
    return report


def main() -> None:
    args = parse_args()
    horizons = tuple(sorted(set(args.horizons)))
    if any(value <= 0 for value in horizons):
        raise SystemExit("all horizons must be positive")
    config = TransformerTrainConfig(
        data_dir=str(args.data_dir.resolve()), output=str(args.output.resolve()),
        horizons=horizons, stride=args.stride, seed=args.seed,
        split_seed=args.split_seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_games_per_split=args.max_games_per_split,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, max_step_m=args.max_step_m,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    splits = load_group_splits(args.data_dir, args.split_seed)
    school_names = collect_school_names(args.data_dir)
    feature_names = transformer_feature_names(school_names)
    if args.max_games_per_split > 0:
        splits = {
            name: paths[:args.max_games_per_split]
            for name, paths in splits.items()
        }
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
        split: build_transformer_split(
            paths, split, horizons, args.stride, args.max_step_m,
            limits[split], rng, school_names,
        )
        for split, paths in splits.items()
    }
    x_train, y_train = arrays["train"]
    x_val, y_val = arrays["val"]
    x_test, y_test = arrays["test"]
    print(
        f"capped samples: train={len(x_train):,}, val={len(x_val):,}, "
        f"test={len(x_test):,}; features={x_train.shape[1]}", flush=True,
    )

    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    train_target = y_train - x_train[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None]
    val_target = y_val - x_val[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None]
    normalized_train = ((x_train - mean) / std).astype(np.float32, copy=False)
    normalized_val = ((x_val - mean) / std).astype(np.float32, copy=False)
    train_weight = sample_weights(x_train, y_train, feature_names)
    val_weight = sample_weights(x_val, y_val, feature_names)
    print(
        f"sample weighting: mean={train_weight.mean():.3f}, "
        f"min={train_weight.min():.3f}, max={train_weight.max():.3f}",
        flush=True,
    )

    model_kwargs = {
        "input_dim": len(feature_names), "horizon_count": len(horizons),
        "d_model": args.d_model, "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward, "dropout": args.dropout,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalBattlefieldTransformer(**model_kwargs).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    print(f"training device: {device}; parameters={parameter_count:,}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_train), torch.from_numpy(train_target),
            torch.from_numpy(train_weight),
        ),
        batch_size=args.batch_size, shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_inputs = torch.from_numpy(normalized_val)
    val_targets = torch.from_numpy(val_target)
    val_weights = torch.from_numpy(val_weight)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for inputs, targets, weights in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_smooth_l1(model(inputs), targets, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
            seen += len(inputs)
        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for start in range(0, len(val_inputs), args.batch_size):
                inputs = val_inputs[start:start + args.batch_size].to(device)
                targets = val_targets[start:start + args.batch_size].to(device)
                weights = val_weights[start:start + args.batch_size].to(device)
                val_total += float(weighted_smooth_l1(model(inputs), targets, weights)) * len(inputs)
        train_loss = total / seen
        val_loss = val_total / len(val_inputs)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"epoch {epoch:02d}: train={train_loss:.6f} val={val_loss:.6f} "
            f"elapsed={time.monotonic() - started:.1f}s", flush=True,
        )
        if val_loss < best_val - 1e-6:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    metrics = rename_metric_model(evaluate(
        model, x_test, y_test, mean, std, horizons, device, args.batch_size
    ))
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    checkpoint = {
        "schema_version": 1,
        "model_kind": "temporal_battlefield_transformer",
        "model_state": {key: value.cpu() for key, value in model.state_dict().items()},
        "model_kwargs": model_kwargs,
        "feature_names": feature_names,
        "school_names": school_names,
        "feature_mean": torch.from_numpy(mean),
        "feature_std": torch.from_numpy(std),
        "horizons": horizons,
        "history_offsets": (0, 1, 3, 5),
        "field_size_m": (FIELD_WIDTH_M, FIELD_HEIGHT_M),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "parameter_count": parameter_count,
        "sample_counts": {split: len(value[0]) for split, value in arrays.items()},
        "sample_weighting": {
            "stationary_weight": 0.70,
            "stationary_service_multiplier": 0.10,
            "recent_damage_multiplier": 1.50,
            "moving_threshold_mps": 0.15,
        },
        "test_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "checkpoint": str(args.output.resolve()),
        "model_kind": checkpoint["model_kind"],
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "sample_counts": checkpoint["sample_counts"],
        "sample_weighting": checkpoint["sample_weighting"],
        "metrics": metrics,
        "history": history,
        "config": asdict(config),
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved checkpoint: {args.output}", flush=True)
    print(f"saved metrics: {report_path}", flush=True)


if __name__ == "__main__":
    main()
