#!/usr/bin/env python3
"""Train a multi-modal Transformer on exact RMUC future coordinates.

This is the non-regression counterpart to ``train_trajectory_transformer``.
It uses the same live battlefield features and match-group split, but learns a
mixture of continuous 2-D Gaussian destinations with maximum likelihood.
There are no near/far, team-style, anchor, or tactical-zone labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .train_trajectory import (
        DEFAULT_DATA_DIR,
        DEFAULT_HORIZONS,
        FIELD_HEIGHT_M,
        FIELD_WIDTH_M,
        MOBILE_TYPES,
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        load_group_splits,
        metric_summary,
        predict_batches,
        zone_accuracy,
    )
    from .train_trajectory_transformer import (
        DEFAULT_OUTPUT as REGRESSION_CHECKPOINT,
        build_transformer_split,
        collect_school_names,
        transformer_feature_names,
    )
    from .trajectory_transformer import (
        TemporalBattlefieldMixtureTransformer,
        TemporalBattlefieldTransformer,
    )
except ImportError:
    from train_trajectory import (  # type: ignore[no-redef]
        DEFAULT_DATA_DIR,
        DEFAULT_HORIZONS,
        FIELD_HEIGHT_M,
        FIELD_WIDTH_M,
        MOBILE_TYPES,
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        load_group_splits,
        metric_summary,
        predict_batches,
        zone_accuracy,
    )
    from train_trajectory_transformer import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT as REGRESSION_CHECKPOINT,
        build_transformer_split,
        collect_school_names,
        transformer_feature_names,
    )
    from trajectory_transformer import (  # type: ignore[no-redef]
        TemporalBattlefieldMixtureTransformer,
        TemporalBattlefieldTransformer,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "ml" / "artifacts" / "trajectory_mixture_transformer.pt"
LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class MixtureTrainConfig:
    data_dir: str
    output: str
    regression_checkpoint: str
    horizons: tuple[int, ...]
    stride: int
    seed: int
    split_seed: int
    max_train_samples: int
    max_val_samples: int
    max_test_samples: int
    max_games_per_split: int
    development_test_games: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    max_step_m: float
    mixture_count: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--regression-checkpoint", type=Path, default=REGRESSION_CHECKPOINT,
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--split-seed", type=int, default=7803)
    parser.add_argument("--max-train-samples", type=int, default=300_000)
    parser.add_argument("--max-val-samples", type=int, default=60_000)
    parser.add_argument("--max-test-samples", type=int, default=80_000)
    parser.add_argument("--max-games-per-split", type=int, default=0)
    parser.add_argument(
        "--development-test-games",
        type=int,
        default=8,
        help=(
            "exclude every match group touched by the first N old-test games; "
            "the default protects the final blind set from the initial smoke test"
        ),
    )
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-step-m", type=float, default=8.0)
    parser.add_argument("--mixture-count", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    return parser.parse_args()


def mixture_nll(
    logits: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    standardized = (target.unsqueeze(2) - means) / log_scales.exp()
    component_log_probability = (
        -0.5 * standardized.square() - log_scales - 0.5 * LOG_2PI
    ).sum(dim=-1)
    log_probability = torch.logsumexp(
        torch.log_softmax(logits, dim=-1) + component_log_probability,
        dim=-1,
    )
    return -log_probability.mean()


def reserve_development_test_groups(
    data_dir: Path,
    splits: dict[str, list[Path]],
    observed_game_count: int,
) -> tuple[dict[str, list[Path]], dict[str, object]]:
    """Remove entire match groups previously seen during development.

    The first smoke run inspected eight games from the historical test split.
    Removing only those files could leave another round of the same series in
    the final set, so this function removes every game sharing their catalog
    group.  Train and validation remain unchanged.
    """
    if observed_game_count <= 0:
        return splits, {
            "development_test_games_requested": 0,
            "excluded_groups": [],
            "excluded_games": [],
        }
    catalog_path = data_dir.parent / "catalog.json"
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    game_to_group = {
        int(item["game_id"]): group
        for group, rounds in catalog["rounds"].items()
        for item in rounds
    }
    observed_paths = splits["test"][:observed_game_count]
    observed_groups = {
        game_to_group[int(path.name.removesuffix(".json.gz"))]
        for path in observed_paths
    }
    excluded_paths = [
        path for path in splits["test"]
        if game_to_group[int(path.name.removesuffix(".json.gz"))]
        in observed_groups
    ]
    final_test = [
        path for path in splits["test"]
        if game_to_group[int(path.name.removesuffix(".json.gz"))]
        not in observed_groups
    ]
    if not final_test:
        raise RuntimeError("development reservation removed the whole test split")
    protected = {name: list(paths) for name, paths in splits.items()}
    protected["test"] = final_test
    return protected, {
        "development_test_games_requested": observed_game_count,
        "excluded_groups": sorted(observed_groups),
        "excluded_games": [path.name for path in excluded_paths],
        "final_blind_test_games": [path.name for path in final_test],
    }


def predict_mixture_batches(
    model: TemporalBattlefieldMixtureTransformer,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            normalized = (x[start:start + batch_size] - mean) / std
            inputs = torch.from_numpy(normalized).to(device)
            logits, means, log_scales = model(inputs)
            logits_parts.append(logits.cpu().numpy())
            mean_parts.append(means.cpu().numpy())
            scale_parts.append(log_scales.exp().cpu().numpy())
    return (
        np.concatenate(logits_parts),
        np.concatenate(mean_parts),
        np.concatenate(scale_parts),
    )


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def gaussian_mixture_nll_numpy(
    logits: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
) -> float:
    return float(gaussian_mixture_nll_values(
        logits, means, scales, targets,
    ).mean())


def gaussian_mixture_nll_values(
    logits: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    standardized = (targets[:, :, None, :] - means) / scales
    component = (
        -0.5 * standardized ** 2 - np.log(scales) - 0.5 * LOG_2PI
    ).sum(axis=-1)
    log_weights = np.log(np.maximum(1e-12, softmax(logits)))
    peak = np.max(log_weights + component, axis=-1, keepdims=True)
    values = peak[..., 0] + np.log(
        np.exp(log_weights + component - peak).sum(axis=-1)
    )
    return -values


def mixture_energy_scores_m(
    logits: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int = 20260728,
    draws: int = 48,
    batch_size: int = 2048,
) -> np.ndarray:
    """Monte-Carlo energy score in metres for every sample and horizon.

    Energy score is a proper scoring rule: it rewards both accurate locations
    and calibrated spread without granting an oracle the right to choose the
    closest mixture component after seeing the answer.
    """
    if draws < 2:
        raise ValueError("draws must be at least two")
    field_scale = np.asarray(
        [FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32,
    )
    result = np.empty(targets.shape[:2], dtype=np.float32)
    rng = np.random.default_rng(seed)

    def sample_distribution(
        probabilities: np.ndarray,
        batch_means: np.ndarray,
        batch_scales: np.ndarray,
    ) -> np.ndarray:
        cumulative = probabilities.cumsum(axis=-1)
        uniforms = rng.random(
            (*probabilities.shape[:2], draws), dtype=np.float32,
        )
        component_index = (
            uniforms[..., None] > cumulative[..., None, :]
        ).sum(axis=-1)
        component_index = np.minimum(
            component_index, probabilities.shape[-1] - 1,
        )
        selected_means = np.take_along_axis(
            batch_means[:, :, None, :, :],
            component_index[..., None, None],
            axis=3,
        )[:, :, :, 0, :]
        selected_scales = np.take_along_axis(
            batch_scales[:, :, None, :, :],
            component_index[..., None, None],
            axis=3,
        )[:, :, :, 0, :]
        noise = rng.standard_normal(
            selected_means.shape, dtype=np.float32,
        )
        return selected_means + selected_scales * noise

    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        batch_probabilities = softmax(logits[start:stop]).astype(
            np.float32, copy=False,
        )
        batch_means = means[start:stop].astype(np.float32, copy=False)
        batch_scales = scales[start:stop].astype(np.float32, copy=False)
        first = sample_distribution(
            batch_probabilities, batch_means, batch_scales,
        )
        second = sample_distribution(
            batch_probabilities, batch_means, batch_scales,
        )
        batch_targets = targets[start:stop, :, None, :]
        observation_distance = np.linalg.norm(
            (first - batch_targets) * field_scale, axis=-1,
        ).mean(axis=-1)
        pair_distance = np.linalg.norm(
            (first - second) * field_scale, axis=-1,
        ).mean(axis=-1)
        result[start:stop] = observation_distance - 0.5 * pair_distance
    return result


def calibrated_single_gaussian(
    regression: TemporalBattlefieldTransformer,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    val_residual = predict_batches(
        regression, x_val, mean, std, device, batch_size,
    )
    test_residual = predict_batches(
        regression, x_test, mean, std, device, batch_size,
    )
    current_val = x_val[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    current_test = x_test[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    val_error = y_val - (current_val + val_residual)
    scales = np.maximum(0.01, np.sqrt(np.mean(val_error ** 2, axis=0)))
    test_error = y_test - (current_test + test_residual)
    nll = (
        0.5 * (test_error / scales) ** 2 + np.log(scales) + 0.5 * LOG_2PI
    ).sum(axis=-1)
    return test_residual, nll, scales.astype(float).tolist()


def subset_metrics(
    prediction: np.ndarray,
    components: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray,
) -> dict:
    scale = np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M], dtype=np.float32)
    error = np.linalg.norm((prediction[mask] - actual[mask]) * scale, axis=1)
    component_error = np.linalg.norm(
        (components[mask] - actual[mask, None, :]) * scale,
        axis=-1,
    )
    values = metric_summary(error)
    values["zone_accuracy"] = zone_accuracy(prediction[mask], actual[mask])
    values["best_of_k_mean_error_m"] = round(float(component_error.min(axis=1).mean()), 4)
    values["best_of_k_within_1m"] = round(float(np.mean(component_error.min(axis=1) <= 1)), 4)
    return values


def distribution_subset_metrics(
    mixture_nll: np.ndarray,
    regression_nll: np.ndarray,
    mixture_energy: np.ndarray,
    regression_energy: np.ndarray,
    mask: np.ndarray,
    horizon_index: int,
) -> dict[str, float]:
    return {
        "mixture_nll": round(
            float(mixture_nll[mask, horizon_index].mean()), 6,
        ),
        "regression_nll": round(
            float(regression_nll[mask, horizon_index].mean()), 6,
        ),
        "mixture_energy_score_m": round(
            float(mixture_energy[mask, horizon_index].mean()), 6,
        ),
        "regression_energy_score_m": round(
            float(regression_energy[mask, horizon_index].mean()), 6,
        ),
    }


def evaluate_distribution(
    model: TemporalBattlefieldMixtureTransformer,
    regression: TemporalBattlefieldTransformer,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    regression_mean: np.ndarray,
    regression_std: np.ndarray,
    feature_names: Sequence[str],
    horizons: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> dict:
    logits, residual_means, scales = predict_mixture_batches(
        model, x_test, mean, std, device, batch_size,
    )
    current = x_test[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, None, :]
    components = np.clip(current + residual_means, 0, 1)
    probabilities = softmax(logits)
    weighted = np.clip(
        (components * probabilities[..., None]).sum(axis=2), 0, 1,
    )
    top_index = probabilities.argmax(axis=2)
    top = np.take_along_axis(
        components, top_index[..., None, None], axis=2,
    )[:, :, 0, :]
    regression_residual, regression_nll, regression_scales = calibrated_single_gaussian(
        regression, x_val, y_val, x_test, y_test,
        regression_mean, regression_std, device, batch_size,
    )
    regression_prediction = np.clip(
        x_test[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
        + regression_residual,
        0, 1,
    )
    residual_targets = (
        y_test
        - x_test[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    )
    mixture_nll = gaussian_mixture_nll_values(
        logits, residual_means, scales,
        residual_targets,
    )
    mixture_energy = mixture_energy_scores_m(
        logits, residual_means, scales, residual_targets,
    )
    regression_logits = np.zeros(
        (*regression_residual.shape[:2], 1), dtype=np.float32,
    )
    regression_means = regression_residual[:, :, None, :]
    regression_scale_array = np.asarray(
        regression_scales, dtype=np.float32,
    )[None, :, None, :]
    regression_scale_array = np.broadcast_to(
        regression_scale_array, regression_means.shape,
    )
    regression_energy = mixture_energy_scores_m(
        regression_logits, regression_means, regression_scale_array,
        residual_targets, seed=20260729,
    )
    role_masks = {
        role: x_test[:, feature_names.index(f"target.type.{role}")] > 0.5
        for role in MOBILE_TYPES
    }
    damage_columns = [
        feature_names.index(name)
        for name in ("target.hp_loss_1", "target.hp_loss_3", "target.hp_loss_5")
    ]
    damaged = x_test[:, damage_columns].max(axis=1) > 0.005
    report: dict[str, object] = {
        "distribution_nll": round(float(mixture_nll.mean()), 6),
        "calibrated_regression_nll": round(float(regression_nll.mean()), 6),
        "energy_score_m": round(float(mixture_energy.mean()), 6),
        "calibrated_regression_energy_score_m": round(
            float(regression_energy.mean()), 6,
        ),
        "regression_scale_by_horizon": regression_scales,
        "sample_counts": {
            "all": len(x_test),
            "hero": int(role_masks["英雄"].sum()),
            "hero_recently_damaged": int((role_masks["英雄"] & damaged).sum()),
        },
        "horizons": {},
    }
    for index, horizon in enumerate(horizons):
        subsets = {
            "all": np.ones(len(x_test), dtype=bool),
            "hero": role_masks["英雄"],
            "hero_recently_damaged": role_masks["英雄"] & damaged,
        }
        report["horizons"][f"{horizon}s"] = {
            subset: {
                "proper_scores": distribution_subset_metrics(
                    mixture_nll, regression_nll,
                    mixture_energy, regression_energy,
                    mask, index,
                ),
                "mixture_top": subset_metrics(
                    top[:, index], components[:, index], y_test[:, index], mask,
                ),
                "mixture_expectation": subset_metrics(
                    weighted[:, index], components[:, index], y_test[:, index], mask,
                ),
                "regression": {
                    **metric_summary(np.linalg.norm(
                        (regression_prediction[mask, index] - y_test[mask, index])
                        * np.asarray([FIELD_WIDTH_M, FIELD_HEIGHT_M]),
                        axis=1,
                    )),
                    "zone_accuracy": zone_accuracy(
                        regression_prediction[mask, index], y_test[mask, index],
                    ),
                },
            }
            for subset, mask in subsets.items()
            if mask.any()
        }
    report["acceptance"] = {
        "lower_test_nll_than_calibrated_regression": (
            float(mixture_nll.mean()) < float(regression_nll.mean())
        ),
        "lower_test_energy_score_than_calibrated_regression": (
            float(mixture_energy.mean()) < float(regression_energy.mean())
        ),
        "hero_10s_energy_score_better_than_regression": (
            report["horizons"]["10s"]["hero"]["proper_scores"]["mixture_energy_score_m"]
            < report["horizons"]["10s"]["hero"]["proper_scores"]["regression_energy_score_m"]
        ),
        "damaged_hero_has_at_least_100_blind_samples": (
            int((role_masks["英雄"] & damaged).sum()) >= 100
        ),
        "damaged_hero_10s_energy_score_better_than_regression": (
            report["horizons"]["10s"]["hero_recently_damaged"]["proper_scores"]["mixture_energy_score_m"]
            < report["horizons"]["10s"]["hero_recently_damaged"]["proper_scores"]["regression_energy_score_m"]
        ),
    }
    return report


def load_regression(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TemporalBattlefieldTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_kind") != "temporal_battlefield_transformer":
        raise ValueError("regression checkpoint is not a temporal battlefield Transformer")
    model = TemporalBattlefieldTransformer(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args()
    horizons = tuple(sorted(set(args.horizons)))
    if any(value <= 0 for value in horizons):
        raise SystemExit("all horizons must be positive")
    config = MixtureTrainConfig(
        data_dir=str(args.data_dir.resolve()),
        output=str(args.output.resolve()),
        regression_checkpoint=str(args.regression_checkpoint.resolve()),
        horizons=horizons,
        stride=args.stride,
        seed=args.seed,
        split_seed=args.split_seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        max_games_per_split=args.max_games_per_split,
        development_test_games=args.development_test_games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_step_m=args.max_step_m,
        mixture_count=args.mixture_count,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    splits = load_group_splits(args.data_dir, args.split_seed)
    splits, blind_split_audit = reserve_development_test_groups(
        args.data_dir, splits, args.development_test_games,
    )
    if args.max_games_per_split > 0:
        splits = {
            name: paths[:args.max_games_per_split]
            for name, paths in splits.items()
        }
    school_names = collect_school_names(args.data_dir)
    feature_names = transformer_feature_names(school_names)
    print(
        "match-group split games: "
        + ", ".join(f"{name}={len(paths)}" for name, paths in splits.items()),
        flush=True,
    )
    print(
        "blind-test reservation: "
        f"excluded_groups={len(blind_split_audit['excluded_groups'])}, "
        f"excluded_games={len(blind_split_audit['excluded_games'])}, "
        f"final_test_games={len(blind_split_audit['final_blind_test_games'])}",
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
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-5] = 1.0
    normalized_train = ((x_train - mean) / std).astype(np.float32, copy=False)
    normalized_val = ((x_val - mean) / std).astype(np.float32, copy=False)
    current_train = x_train[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    current_val = x_val[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    train_target = y_train - current_train
    val_target = y_val - current_val

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regression, regression_checkpoint = load_regression(
        args.regression_checkpoint, device,
    )
    if list(regression_checkpoint["feature_names"]) != feature_names:
        raise RuntimeError("regression baseline and mixture dataset feature schemas differ")
    if tuple(regression_checkpoint["horizons"]) != horizons:
        raise RuntimeError("regression baseline and mixture horizons differ")
    model_kwargs = {
        "input_dim": len(feature_names),
        "horizon_count": len(horizons),
        "mixture_count": args.mixture_count,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "min_log_scale": -4.6,
        "max_log_scale": -0.5,
    }
    model = TemporalBattlefieldMixtureTransformer(**model_kwargs).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    print(
        f"samples train={len(x_train):,} val={len(x_val):,} test={len(x_test):,}; "
        f"device={device}; parameters={parameter_count:,}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_train),
            torch.from_numpy(train_target.astype(np.float32)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_inputs = torch.from_numpy(normalized_val)
    val_targets = torch.from_numpy(val_target.astype(np.float32))
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = mixture_nll(*model(inputs), targets)
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
                val_total += float(mixture_nll(*model(inputs), targets)) * len(inputs)
        train_loss = total / max(1, seen)
        val_loss = val_total / len(val_inputs)
        history.append({
            "epoch": epoch,
            "train_nll": train_loss,
            "val_nll": val_loss,
        })
        print(
            f"epoch {epoch:02d}: train_nll={train_loss:.6f} "
            f"val_nll={val_loss:.6f} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    metrics = evaluate_distribution(
        model, regression,
        x_val, y_val, x_test, y_test,
        mean, std,
        regression_checkpoint["feature_mean"].numpy(),
        regression_checkpoint["feature_std"].numpy(),
        feature_names, horizons, device, args.batch_size,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    acceptance = metrics["acceptance"]
    if not all(acceptance.values()):
        raise RuntimeError(
            "blind-test acceptance gate failed; candidate checkpoint was not published: "
            + json.dumps(acceptance, ensure_ascii=False)
        )

    checkpoint = {
        "schema_version": 1,
        "model_kind": "temporal_battlefield_mixture_transformer",
        "model_state": {
            key: value.cpu() for key, value in model.state_dict().items()
        },
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
        "sample_counts": {
            split: len(value[0]) for split, value in arrays.items()
        },
        "training_target": "exact observed future canonical coordinates",
        "label_policy": "no handcrafted tactical labels",
        "test_metrics": metrics,
        "blind_split_audit": blind_split_audit,
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
        "training_target": checkpoint["training_target"],
        "label_policy": checkpoint["label_policy"],
        "metrics": metrics,
        "blind_split_audit": blind_split_audit,
        "history": history,
        "config": asdict(config),
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"accepted checkpoint: {args.output}", flush=True)
    print(f"saved metrics: {report_path}", flush=True)


if __name__ == "__main__":
    main()
