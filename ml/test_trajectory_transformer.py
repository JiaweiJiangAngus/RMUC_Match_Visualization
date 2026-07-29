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
from ml.train_trajectory_mixture_transformer import (
    mixture_energy_scores_m, mixture_nll,
)
from ml.trajectory_transformer import (
    TemporalBattlefieldMixtureTransformer,
    TemporalBattlefieldTransformer,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "ml" / "artifacts" / "trajectory_transformer.pt"
MIXTURE_CHECKPOINT = (
    ROOT / "ml" / "artifacts" / "trajectory_mixture_transformer.pt"
)
CONDITIONING_AUDIT = MIXTURE_CHECKPOINT.with_suffix(
    ".conditioning_audit.json"
)
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

    def test_mixture_transformer_outputs_continuous_distribution(self):
        model = TemporalBattlefieldMixtureTransformer(
            len(FEATURE_NAMES), 5, mixture_count=8,
        )
        logits, means, log_scales = model(
            torch.zeros(2, len(FEATURE_NAMES)),
        )
        self.assertEqual((2, 5, 8), tuple(logits.shape))
        self.assertEqual((2, 5, 8, 2), tuple(means.shape))
        self.assertEqual((2, 5, 8, 2), tuple(log_scales.shape))
        self.assertTrue(torch.isfinite(
            mixture_nll(
                logits, means, log_scales, torch.zeros(2, 5, 2),
            )
        ))

    def test_energy_score_does_not_use_best_component_oracle(self):
        targets = np.zeros((512, 1, 2), dtype=np.float32)
        logits = np.zeros((512, 1, 1), dtype=np.float32)
        scales = np.full((512, 1, 1, 2), 0.01, dtype=np.float32)
        centered = np.zeros((512, 1, 1, 2), dtype=np.float32)
        displaced = np.full((512, 1, 1, 2), 0.2, dtype=np.float32)
        centered_score = mixture_energy_scores_m(
            logits, centered, scales, targets, seed=17,
        ).mean()
        displaced_score = mixture_energy_scores_m(
            logits, displaced, scales, targets, seed=17,
        ).mean()
        self.assertLess(float(centered_score), float(displaced_score))

    def test_stationary_service_samples_are_downweighted(self):
        x = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
        target_x = FEATURE_NAMES.index("target.x")
        target_y = FEATURE_NAMES.index("target.y")
        velocity_x = FEATURE_NAMES.index("target.vx_3_norm_per_s")
        # This corner is inside the rectangular supply zone but outside the
        # old ellipse, so it protects the trained mask from regressing.
        x[0, target_x] = 3.6 / 28
        x[0, target_y] = 2.9 / 15
        # This point is inside the full outpost ellipse but on the side facing
        # away from the central highland, so it must not count as service.
        x[1, target_x] = 10.2 / 28
        x[1, target_y] = 2.2 / 15
        x[2, target_x] = 14 / 28
        x[2, target_y] = 7.5 / 15
        x[2, velocity_x] = 1 / 28
        y = np.repeat(x[:, [target_x, target_y]][:, None], 5, axis=1)
        weights = sample_weights(x, y)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[0], weights[2])

    @unittest.skipUnless(CHECKPOINT.exists(), "trained Transformer checkpoint is required")
    def test_checkpoint_is_team_and_damage_conditioned(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.assertIn("同济大学", checkpoint["school_names"])
        self.assertIn("target.hp_loss_1", checkpoint["feature_names"])
        self.assertIn("target.school.同济大学", checkpoint["feature_names"])
        self.assertEqual(252_394, checkpoint["parameter_count"])

    @unittest.skipUnless(
        MIXTURE_CHECKPOINT.exists(), "trained mixture checkpoint is required"
    )
    def test_mixture_checkpoint_passed_blind_distribution_gates(self):
        checkpoint = torch.load(
            MIXTURE_CHECKPOINT, map_location="cpu", weights_only=True
        )
        self.assertEqual(
            "temporal_battlefield_mixture_transformer",
            checkpoint["model_kind"],
        )
        self.assertEqual(
            "exact observed future canonical coordinates",
            checkpoint["training_target"],
        )
        self.assertEqual("no handcrafted tactical labels", checkpoint["label_policy"])
        self.assertTrue(all(checkpoint["test_metrics"]["acceptance"].values()))
        self.assertGreaterEqual(
            checkpoint["test_metrics"]["sample_counts"]["hero_recently_damaged"],
            100,
        )
        self.assertGreaterEqual(
            len(checkpoint["blind_split_audit"]["final_blind_test_games"]),
            70,
        )

    @unittest.skipUnless(
        CONDITIONING_AUDIT.exists(), "frozen conditioning audit is required"
    )
    def test_frozen_model_uses_identity_damage_hp_and_time_context(self):
        report = json.loads(CONDITIONING_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(
            "frozen post-training audit; no retraining or threshold labels",
            report["policy"],
        )
        self.assertTrue(all(report["acceptance"].values()))
        self.assertGreater(report["hero_correct_identity_improvement"], 0)
        self.assertGreater(report["damaged_hero_damage_feature_improvement"], 0)
        self.assertGreater(report["hero_live_hp_improvement"], 0)
        self.assertGreater(report["hero_match_time_improvement"], 0)

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
        y = np.asarray([row[1] for row in rows], dtype=np.float32)
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
        velocity_indices = [
            names.index("target.vx_3_norm_per_s"),
            names.index("target.vy_3_norm_per_s"),
        ]
        stationary_anchor = np.linalg.norm(
            x[:, velocity_indices] * np.asarray([28, 15]), axis=1
        ) < 0.15
        observed_displacement = np.linalg.norm(
            (
                y[:, checkpoint["horizons"].index(10)]
                - x[:, [
                    names.index("target.x"), names.index("target.y")
                ]]
            ) * np.asarray([28, 15]),
            axis=1,
        )
        damaged_anchor = stationary_anchor & damaged
        undamaged_anchor = stationary_anchor & ~damaged
        self.assertGreaterEqual(int(damaged_anchor.sum()), 5)
        self.assertGreaterEqual(int(undamaged_anchor.sum()), 100)
        self.assertGreater(
            float((observed_displacement[damaged_anchor] > 0.75).mean()),
            float((observed_displacement[undamaged_anchor] > 0.75).mean()) * 3,
        )
        self.assertGreater(
            float((observed_displacement[undamaged_anchor] < 0.25).mean()),
            0.85,
        )
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

        mixture_checkpoint = torch.load(
            MIXTURE_CHECKPOINT, map_location="cpu", weights_only=True
        )
        self.assertEqual(
            list(mixture_checkpoint["feature_names"]), names
        )
        mixture_model = TemporalBattlefieldMixtureTransformer(
            **mixture_checkpoint["model_kwargs"]
        )
        mixture_model.load_state_dict(mixture_checkpoint["model_state"])
        mixture_model.eval()
        mixture_normalized = (
            (x - mixture_checkpoint["feature_mean"].numpy())
            / mixture_checkpoint["feature_std"].numpy()
        )
        with torch.inference_mode():
            logits, means, _log_scales = mixture_model(
                torch.from_numpy(mixture_normalized)
            )
        horizon_index = mixture_checkpoint["horizons"].index(10)
        probabilities = logits[:, horizon_index].softmax(dim=-1).numpy()
        component_displacement = np.linalg.norm(
            means[:, horizon_index].numpy() * np.asarray([28, 15]), axis=2
        )
        distribution_displacement = (
            probabilities * component_displacement
        ).sum(axis=1)
        self.assertGreater(
            float(distribution_displacement[damaged_anchor].mean()),
            float(distribution_displacement[undamaged_anchor].mean()) * 1.1,
        )

    @unittest.skipUnless(
        shutil.which("node") and MIXTURE_CHECKPOINT.exists()
        and MANIFEST.exists() and WEIGHTS.exists(),
        "trained/exported Transformer artifacts are required",
    )
    def test_browser_mixture_distribution_matches_pytorch(self):
        checkpoint = torch.load(
            MIXTURE_CHECKPOINT, map_location="cpu", weights_only=True
        )
        model = TemporalBattlefieldMixtureTransformer(**checkpoint["model_kwargs"])
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
            expected = [
                value[0].numpy().reshape(-1)
                for value in model(torch.from_numpy(normalized[None]))
            ]
        script = r"""
const fs=require('fs'),core=require('./docs/prediction-worker.js');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const manifest=JSON.parse(fs.readFileSync(input.manifest,'utf8'));
const bytes=fs.readFileSync(input.weights);
const floats=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
const tensors=new Map(manifest.tensors.map(item=>[item.name,floats.subarray(item.offset,item.offset+item.length)]));
const model={manifest,tensors,mean:Float32Array.from(manifest.feature_mean),std:Float32Array.from(manifest.feature_std)};
const result=core.forwardDistribution(model,Float32Array.from(input.features));
process.stdout.write(JSON.stringify([
  Array.from(result.logits),Array.from(result.means),Array.from(result.logScales),
]));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            input=json.dumps({
                "manifest": str(MANIFEST), "weights": str(WEIGHTS),
                "features": features.tolist(),
            }), capture_output=True, check=True,
        )
        actual = [
            np.asarray(value, dtype=np.float32)
            for value in json.loads(result.stdout)
        ]
        for browser, pytorch in zip(actual, expected):
            np.testing.assert_allclose(
                browser, pytorch, rtol=2e-4, atol=2e-5
            )


if __name__ == "__main__":
    unittest.main()
