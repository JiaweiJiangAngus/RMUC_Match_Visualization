#!/usr/bin/env python3
"""Run the trained trajectory model on one exported RMUC game frame."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from train_trajectory import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT,
    FEATURE_NAMES,
    FIELD_HEIGHT_M,
    FIELD_WIDTH_M,
    MOBILE_TYPES,
    REGULATION_DURATION_S,
    SIDES,
    TARGET_X_INDEX,
    TARGET_Y_INDEX,
    TrajectoryMLP,
    canonical_xy,
    index_frame,
    sample_features,
    valid_position,
)


def world_xy(normalized_xy: np.ndarray, side: str) -> tuple[float, float]:
    x = float(normalized_xy[0]) * FIELD_WIDTH_M
    y = float(normalized_xy[1]) * FIELD_HEIGHT_M
    if side == "蓝":
        x, y = FIELD_WIDTH_M - x, FIELD_HEIGHT_M - y
    return round(x, 3), round(y, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--game-file", type=Path)
    parser.add_argument("--game-id", type=int)
    parser.add_argument("--second", type=int, default=200)
    parser.add_argument("--side", choices=SIDES)
    parser.add_argument("--robot-type", choices=MOBILE_TYPES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.game_file is not None and args.game_id is not None:
        raise SystemExit("choose either --game-file or --game-id")
    game_path = args.game_file
    if game_path is None:
        if args.game_id is None:
            raise SystemExit("one of --game-file or --game-id is required")
        game_path = DEFAULT_DATA_DIR / f"{args.game_id}.json.gz"
    if not game_path.is_file():
        raise SystemExit(f"game file does not exist: {game_path}")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise SystemExit("model feature schema does not match this code")
    model = TrajectoryMLP(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = checkpoint["feature_mean"].numpy()
    std = checkpoint["feature_std"].numpy()
    horizons = tuple(checkpoint["horizons"])

    with gzip.open(game_path, "rt", encoding="utf-8") as handle:
        game = json.load(handle)
    frames = {int(second): index_frame(rows) for second, rows in game["frames"].items()}
    if args.second not in frames:
        raise SystemExit(f"second {args.second} is not present in this game")
    duration = REGULATION_DURATION_S

    predictions: list[dict] = []
    for side in SIDES:
        if args.side is not None and side != args.side:
            continue
        for robot_type in MOBILE_TYPES:
            if args.robot_type is not None and robot_type != args.robot_type:
                continue
            current = frames[args.second].get((side, robot_type))
            if not valid_position(current):
                continue
            try:
                features = np.asarray(
                    sample_features(frames, args.second, side, robot_type, duration),
                    dtype=np.float32,
                )
            except KeyError:
                continue
            inputs = torch.from_numpy(((features - mean) / std)[None])
            with torch.inference_mode():
                residual = model(inputs)[0].numpy()
            current_canonical = features[[TARGET_X_INDEX, TARGET_Y_INDEX]]
            future_canonical = np.clip(current_canonical[None] + residual, 0.0, 1.0)

            future: list[dict] = []
            for index, horizon in enumerate(horizons):
                item: dict = {
                    "after_seconds": horizon,
                    "predicted_xy_m": world_xy(future_canonical[index], side),
                }
                actual_frame = frames.get(args.second + horizon)
                actual = actual_frame.get((side, robot_type)) if actual_frame else None
                if valid_position(actual):
                    actual_canonical = np.asarray(canonical_xy(actual, side))
                    actual_xy = world_xy(actual_canonical, side)
                    item["actual_xy_m"] = actual_xy
                    item["error_m"] = round(
                        float(
                            np.linalg.norm(
                                np.asarray(item["predicted_xy_m"])
                                - np.asarray(actual_xy)
                            )
                        ),
                        3,
                    )
                future.append(item)
            current_xy = world_xy(np.asarray(canonical_xy(current, side)), side)
            predictions.append(
                {
                    "side": side,
                    "robot_type": robot_type,
                    "robot_id": current[0],
                    "current_xy_m": current_xy,
                    "future": future,
                }
            )

    output = {
        "game_id": game["info"]["game_id"],
        "second": args.second,
        "model": str(args.model),
        "predictions": predictions,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
