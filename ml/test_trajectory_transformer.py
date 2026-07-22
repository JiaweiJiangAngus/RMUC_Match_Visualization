import gzip
import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.train_trajectory import (
    DEFAULT_DATA_DIR, FEATURE_NAMES, REGULATION_DURATION_S,
    index_frame, load_group_splits, sample_features,
)
from ml.train_trajectory_transformer import (
    DAMAGE_FEATURE_NAMES, iter_transformer_samples,
    sample_weights, transformer_sample_features,
)
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

    @unittest.skipUnless(CHECKPOINT.exists(), "trained Transformer checkpoint is required")
    def test_checkpoint_is_team_and_damage_conditioned(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.assertIn("同济大学", checkpoint["school_names"])
        self.assertIn("target.hp_loss_1", checkpoint["feature_names"])
        self.assertIn("target.school.同济大学", checkpoint["feature_names"])
        self.assertEqual(252_394, checkpoint["parameter_count"])

    @unittest.skipUnless(CHECKPOINT.exists(), "trained Transformer checkpoint is required")
    def test_tongji_hero_leaves_anchor_more_after_damage_on_held_out_games(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model = TemporalBattlefieldTransformer(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        names = list(checkpoint["feature_names"])
        school_names = tuple(checkpoint["school_names"])
        school_index = names.index("target.school.同济大学")
        hero_index = names.index("target.type.英雄")
        damage_indices = [names.index(name) for name in DAMAGE_FEATURE_NAMES]
        test_paths = []
        for path in load_group_splits(
            DEFAULT_DATA_DIR, checkpoint["config"]["split_seed"]
        )["test"]:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                info = json.load(handle)["info"]
            if "同济大学" in (info["red"], info["blue"]):
                test_paths.append(path)
        rows = []
        for path in test_paths:
            for features, targets in iter_transformer_samples(
                path, tuple(checkpoint["horizons"]), 5, 8.0, school_names
            ):
                if features[school_index] and features[hero_index]:
                    rows.append((features, targets))
        x = np.asarray([row[0] for row in rows], dtype=np.float32)
        normalized = (
            (x - checkpoint["feature_mean"].numpy())
            / checkpoint["feature_std"].numpy()
        )
        with torch.inference_mode():
            residual = model(torch.from_numpy(normalized)).numpy()
        predicted_displacement = np.linalg.norm(
            residual[:, 3] * np.asarray([28, 15]), axis=1
        )
        damaged = x[:, damage_indices].max(axis=1) > 0.005
        self.assertGreater(int(damaged.sum()), 5)
        self.assertGreater(int((~damaged).sum()), 100)
        self.assertGreater(
            float(predicted_displacement[damaged].mean()),
            float(predicted_displacement[~damaged].mean()) * 1.5,
        )
        self.assertLess(
            float((predicted_displacement[damaged] < 0.75).mean()),
            float((predicted_displacement[~damaged] < 0.75).mean()),
        )

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
        school_names = tuple(checkpoint.get("school_names", ()))
        if school_names:
            target_school = str(game["info"]["red" if side == "红" else "blue"])
            opponent_school = str(game["info"]["blue" if side == "红" else "red"])
            values = transformer_sample_features(
                frames, second, side, role, REGULATION_DURATION_S,
                target_school, opponent_school, school_names,
            )
        else:
            values = sample_features(frames, second, side, role, REGULATION_DURATION_S)
        features = np.asarray(values, dtype=np.float32)
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
