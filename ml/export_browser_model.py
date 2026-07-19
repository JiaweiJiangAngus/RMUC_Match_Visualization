#!/usr/bin/env python3
"""Export the trained trajectory MLP as static browser assets.

The GitHub Pages client cannot import PyTorch.  This exporter writes a small
JSON manifest plus one little-endian float32 blob that the prediction Web
Worker evaluates directly in the browser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .train_trajectory import DEFAULT_OUTPUT
except ImportError:
    from train_trajectory import DEFAULT_OUTPUT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "data" / "models" / "trajectory_mlp.json"
DEFAULT_WEIGHTS = ROOT / "docs" / "data" / "models" / "trajectory_mlp.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    order = (
        "backbone.0.weight", "backbone.0.bias",
        "backbone.2.weight", "backbone.2.bias",
        "backbone.4.weight", "backbone.4.bias",
        "backbone.6.weight", "backbone.6.bias",
        "backbone.8.weight", "backbone.8.bias",
        "backbone.10.weight", "backbone.10.bias",
        "head.weight", "head.bias",
    )

    arrays: list[np.ndarray] = []
    tensors: list[dict] = []
    offset = 0
    for name in order:
        array = state[name].detach().cpu().numpy().astype("<f4", copy=False)
        flat = np.ascontiguousarray(array.reshape(-1))
        arrays.append(flat)
        tensors.append(
            {
                "name": name,
                "shape": list(array.shape),
                "offset": offset,
                "length": int(flat.size),
            }
        )
        offset += int(flat.size)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(arrays).tofile(args.weights)

    metrics = checkpoint.get("test_metrics", {})
    reliability = {}
    for horizon in checkpoint["horizons"]:
        item = metrics.get(f"{horizon}s", {})
        reliability[str(horizon)] = {
            "all": item.get("all", {}).get("learned_mlp", {}),
            "moving": item.get("moving", {}).get("learned_mlp", {}),
        }

    manifest = {
        "schema_version": 1,
        "model_kind": checkpoint["model_kind"],
        "weights": f"./{args.weights.name}",
        "input_dim": int(checkpoint["model_kwargs"]["input_dim"]),
        "hidden_sizes": list(checkpoint["model_kwargs"]["hidden_sizes"]),
        "horizons": list(checkpoint["horizons"]),
        "history_offsets": list(checkpoint["history_offsets"]),
        "field_size_m": list(checkpoint["field_size_m"]),
        "feature_names": list(checkpoint["feature_names"]),
        "feature_mean": checkpoint["feature_mean"].cpu().numpy().astype(float).tolist(),
        "feature_std": checkpoint["feature_std"].cpu().numpy().astype(float).tolist(),
        "layer_norm_epsilon": 1e-5,
        "tensors": tensors,
        "reliability": reliability,
        "training": {
            "best_epoch": checkpoint.get("best_epoch"),
            "sample_counts": checkpoint.get("sample_counts", {}),
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"exported {offset:,} float32 weights "
        f"({args.weights.stat().st_size / 1024:.1f} KiB) to {args.manifest.parent}"
    )


if __name__ == "__main__":
    main()
