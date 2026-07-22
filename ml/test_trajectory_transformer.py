import gzip
import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.train_trajectory import FEATURE_NAMES, REGULATION_DURATION_S, index_frame, sample_features
from ml.train_trajectory_transformer import sample_weights
from ml.trajectory_transformer import TemporalBattlefieldTransformer


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "ml" / "artifacts" / "trajectory_transformer.pt"
MANIFEST = ROOT / "docs" / "data" / "models" / "trajectory_transformer.json"
WEIGHTS = ROOT / "docs" / "data" / "models" / "trajectory_transformer.bin"


class TrajectoryTransformerTests(unittest.TestCase):
    def test_architecture_has_real_multihead_attention(self):
        model = TemporalBattlefieldTransformer(len(FEATURE_NAMES), 5)
        output = model(torch.zeros(2, len(FEATURE_NAMES)))
        self.assertEqual((2, 5, 2), tuple(output.shape))
        attention = model.encoder.layers[0].self_attn
        self.assertEqual(4, attention.num_heads)
        self.assertGreater(attention.in_proj_weight.numel(), 0)

    def test_stationary_service_samples_are_downweighted(self):
        x = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
        target_x = FEATURE_NAMES.index("target.x")
        target_y = FEATURE_NAMES.index("target.y")
        velocity_x = FEATURE_NAMES.index("target.vx_3_norm_per_s")
        x[0, target_x] = 1.8 / 28
        x[0, target_y] = 1.55 / 15
        x[1, target_x] = 14 / 28
        x[1, target_y] = 7.5 / 15
        x[1, velocity_x] = 1 / 28
        y = np.repeat(x[:, [target_x, target_y]][:, None], 5, axis=1)
        weights = sample_weights(x, y)
        self.assertLess(weights[0], weights[1])

    @unittest.skipUnless(
        shutil.which("node") and CHECKPOINT.exists() and MANIFEST.exists() and WEIGHTS.exists(),
        "trained/exported Transformer artifacts are required",
    )
    def test_browser_forward_matches_pytorch(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model = TemporalBattlefieldTransformer(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        game_path = next((ROOT / "docs" / "data" / "games").glob("*.json.gz"))
        with gzip.open(game_path, "rt", encoding="utf-8") as handle:
            game = json.load(handle)
        frames = {int(second): index_frame(rows) for second, rows in game["frames"].items()}
        second = next(value for value in sorted(frames) if value >= 20)
        side, role = next(
            key for key, row in frames[second].items()
            if key[1] in ("英雄", "步兵3", "步兵4", "哨兵") and row[5] is not None
        )
        features = np.asarray(
            sample_features(frames, second, side, role, REGULATION_DURATION_S),
            dtype=np.float32,
        )
        normalized = (features - checkpoint["feature_mean"].numpy()) / checkpoint["feature_std"].numpy()
        with torch.inference_mode():
            expected = model(torch.from_numpy(normalized[None]))[0].numpy().reshape(-1)
        script = r"""
const fs=require('fs'),core=require('./docs/prediction-worker.js');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const manifest=JSON.parse(fs.readFileSync(input.manifest,'utf8'));
const bytes=fs.readFileSync(input.weights);
const floats=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
const tensors=new Map(manifest.tensors.map(item=>[item.name,floats.subarray(item.offset,item.offset+item.length)]));
const model={manifest,tensors,mean:Float32Array.from(manifest.feature_mean),std:Float32Array.from(manifest.feature_std)};
process.stdout.write(JSON.stringify(Array.from(core.forward(model,Float32Array.from(input.features)))));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            input=json.dumps({
                "manifest": str(MANIFEST), "weights": str(WEIGHTS),
                "features": features.tolist(),
            }), capture_output=True, check=True,
        )
        actual = np.asarray(json.loads(result.stdout), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
