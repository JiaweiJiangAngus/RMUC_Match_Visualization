#!/usr/bin/env python3
"""Export an accepted temporal trajectory Transformer for browser inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .train_trajectory_mixture_transformer import DEFAULT_OUTPUT
except ImportError:
    from train_trajectory_mixture_transformer import DEFAULT_OUTPUT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "data" / "models" / "trajectory_transformer.json"
DEFAULT_WEIGHTS = ROOT / "docs" / "data" / "models" / "trajectory_transformer.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    supported = {
        "temporal_battlefield_transformer",
        "temporal_battlefield_mixture_transformer",
    }
    if checkpoint["model_kind"] not in supported:
        raise SystemExit("checkpoint is not a supported temporal battlefield Transformer")
    arrays: list[np.ndarray] = []
    tensors: list[dict] = []
    offset = 0
    for name, value in checkpoint["model_state"].items():
        array = value.detach().cpu().numpy().astype("<f4", copy=False)
        flat = np.ascontiguousarray(array.reshape(-1))
        arrays.append(flat)
        tensors.append({
            "name": name, "shape": list(array.shape),
            "offset": offset, "length": int(flat.size),
        })
        offset += int(flat.size)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(arrays).tofile(args.weights)

    metrics = checkpoint.get("test_metrics", {})
    mixture = checkpoint["model_kind"] == "temporal_battlefield_mixture_transformer"
    reliability = {}
    for horizon in checkpoint["horizons"]:
        if mixture:
            item = metrics.get("horizons", {}).get(f"{horizon}s", {})
            reliability[str(horizon)] = {
                "all": item.get("all", {}).get("mixture_top", {}),
                "hero": item.get("hero", {}).get("mixture_top", {}),
                "hero_recently_damaged": item.get(
                    "hero_recently_damaged", {}
                ).get("mixture_top", {}),
            }
        else:
            item = metrics.get(f"{horizon}s", {})
            reliability[str(horizon)] = {
                "all": item.get("all", {}).get("learned_transformer", {}),
                "moving": item.get("moving", {}).get("learned_transformer", {}),
            }
    kwargs = checkpoint["model_kwargs"]
    manifest = {
        "schema_version": 2 if mixture else 1,
        "model_kind": checkpoint["model_kind"],
        "weights": f"./{args.weights.name}",
        "input_dim": int(kwargs["input_dim"]),
        "horizons": list(checkpoint["horizons"]),
        "history_offsets": list(checkpoint["history_offsets"]),
        "history_token_count": 4,
        "history_token_width": 54,
        "context_dim": int(kwargs["input_dim"]) - 4 * 54,
        "d_model": int(kwargs["d_model"]),
        "nhead": int(kwargs["nhead"]),
        "num_layers": int(kwargs["num_layers"]),
        "dim_feedforward": int(kwargs["dim_feedforward"]),
        "mixture_count": int(kwargs.get("mixture_count", 0)),
        "min_log_scale": float(kwargs.get("min_log_scale", 0)),
        "max_log_scale": float(kwargs.get("max_log_scale", 0)),
        "field_size_m": list(checkpoint["field_size_m"]),
        "feature_names": list(checkpoint["feature_names"]),
        "school_names": list(checkpoint.get("school_names", ())),
        "feature_mean": checkpoint["feature_mean"].numpy().astype(float).tolist(),
        "feature_std": checkpoint["feature_std"].numpy().astype(float).tolist(),
        "layer_norm_epsilon": 1e-5,
        "tensors": tensors,
        "reliability": reliability,
        "training": {
            "best_epoch": checkpoint.get("best_epoch"),
            "parameter_count": checkpoint.get("parameter_count"),
            "sample_counts": checkpoint.get("sample_counts", {}),
            "sample_weighting": checkpoint.get("sample_weighting", {}),
            "target": checkpoint.get("training_target"),
            "label_policy": checkpoint.get("label_policy"),
            "blind_split_audit": checkpoint.get("blind_split_audit", {}),
            "acceptance": metrics.get("acceptance", {}),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"exported {offset:,} float32 Transformer weights "
        f"({args.weights.stat().st_size / 1024:.1f} KiB) to {args.manifest.parent}"
    )


if __name__ == "__main__":
    main()
