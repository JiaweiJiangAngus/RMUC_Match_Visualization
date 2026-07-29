#!/usr/bin/env python3
"""Audit whether the frozen trajectory model uses team and damage context.

This script never trains or tunes the model.  On the untouched final match
groups it compares the accepted model with counterfactual inputs in which only
the target-school identity or recent-damage suffix is removed.  Exact future
coordinates remain the sole outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .train_trajectory import (
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        load_group_splits,
    )
    from .train_trajectory_mixture_transformer import (
        DEFAULT_OUTPUT,
        gaussian_mixture_nll_values,
        predict_mixture_batches,
        reserve_development_test_groups,
    )
    from .train_trajectory_transformer import (
        DAMAGE_FEATURE_NAMES,
        build_transformer_split,
    )
    from .trajectory_transformer import (
        TemporalBattlefieldMixtureTransformer,
    )
except ImportError:
    from train_trajectory import (  # type: ignore[no-redef]
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        load_group_splits,
    )
    from train_trajectory_mixture_transformer import (  # type: ignore[no-redef]
        DEFAULT_OUTPUT,
        gaussian_mixture_nll_values,
        predict_mixture_batches,
        reserve_development_test_groups,
    )
    from train_trajectory_transformer import (  # type: ignore[no-redef]
        DAMAGE_FEATURE_NAMES,
        build_transformer_split,
    )
    from trajectory_transformer import (  # type: ignore[no-redef]
        TemporalBattlefieldMixtureTransformer,
    )


DEFAULT_REPORT = DEFAULT_OUTPUT.with_suffix(".conditioning_audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--minimum-team-hero-samples", type=int, default=100)
    return parser.parse_args()


def nll_values(
    model: TemporalBattlefieldMixtureTransformer,
    x: np.ndarray,
    residual_targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    logits, means, scales = predict_mixture_batches(
        model, x, mean, std, device, batch_size,
    )
    return gaussian_mixture_nll_values(
        logits, means, scales, residual_targets,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    if (
        checkpoint.get("model_kind")
        != "temporal_battlefield_mixture_transformer"
    ):
        raise SystemExit("conditioning audit requires the mixture Transformer")
    config = checkpoint["config"]
    data_dir = Path(config["data_dir"])
    splits = load_group_splits(data_dir, int(config["split_seed"]))
    splits, audit = reserve_development_test_groups(
        data_dir, splits, int(config["development_test_games"]),
    )
    rng = np.random.default_rng(int(config["seed"]))
    x_test, y_test = build_transformer_split(
        splits["test"],
        "conditioning-audit",
        tuple(checkpoint["horizons"]),
        int(config["stride"]),
        float(config["max_step_m"]),
        int(config["max_test_samples"]),
        rng,
        tuple(checkpoint["school_names"]),
    )
    names = list(checkpoint["feature_names"])
    school_names = list(checkpoint["school_names"])
    hero = x_test[:, names.index("target.type.英雄")] > 0.5
    target_school_indices = [
        names.index(f"target.school.{school}") for school in school_names
    ]
    damage_indices = [
        names.index(name) for name in DAMAGE_FEATURE_NAMES
    ]
    live_hp_indices = [
        index for index, name in enumerate(names)
        if (
            name == "target.hp"
            or (name.endswith(".hp") and "hp_loss" not in name)
        )
    ]
    time_indices = [
        names.index(name) for name in ("time.elapsed", "time.remaining")
    ]
    damaged_hero = hero & (
        x_test[:, damage_indices].max(axis=1) > 0.005
    )
    residual_targets = (
        y_test
        - x_test[:, [TARGET_X_INDEX, TARGET_Y_INDEX]][:, None, :]
    )
    mean = checkpoint["feature_mean"].numpy()
    std = checkpoint["feature_std"].numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalBattlefieldMixtureTransformer(
        **checkpoint["model_kwargs"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    correct = nll_values(
        model, x_test, residual_targets,
        mean, std, device, args.batch_size,
    )
    wrong_school_x = x_test.copy()
    school_columns = wrong_school_x[:, target_school_indices].copy()
    wrong_school_x[:, target_school_indices] = np.roll(
        school_columns, shift=1, axis=1,
    )
    wrong_school = nll_values(
        model, wrong_school_x, residual_targets,
        mean, std, device, args.batch_size,
    )
    no_damage_x = x_test.copy()
    no_damage_x[:, damage_indices] = 0
    no_damage = nll_values(
        model, no_damage_x, residual_targets,
        mean, std, device, args.batch_size,
    )
    mean_hp_x = x_test.copy()
    mean_hp_x[:, live_hp_indices] = mean[live_hp_indices]
    mean_hp = nll_values(
        model, mean_hp_x, residual_targets,
        mean, std, device, args.batch_size,
    )
    mean_time_x = x_test.copy()
    mean_time_x[:, time_indices] = mean[time_indices]
    mean_time = nll_values(
        model, mean_time_x, residual_targets,
        mean, std, device, args.batch_size,
    )

    correct_per_sample = correct.mean(axis=1)
    wrong_per_sample = wrong_school.mean(axis=1)
    no_damage_per_sample = no_damage.mean(axis=1)
    mean_hp_per_sample = mean_hp.mean(axis=1)
    mean_time_per_sample = mean_time.mean(axis=1)
    team_report = {}
    eligible_improvements: list[float] = []
    for school, column in zip(school_names, target_school_indices):
        mask = hero & (x_test[:, column] > 0.5)
        if not mask.any():
            continue
        improvement = float(
            wrong_per_sample[mask].mean()
            - correct_per_sample[mask].mean()
        )
        team_report[school] = {
            "hero_samples": int(mask.sum()),
            "correct_team_nll": round(
                float(correct_per_sample[mask].mean()), 6,
            ),
            "wrong_team_nll": round(
                float(wrong_per_sample[mask].mean()), 6,
            ),
            "correct_identity_improvement": round(improvement, 6),
        }
        if int(mask.sum()) >= args.minimum_team_hero_samples:
            eligible_improvements.append(improvement)

    hero_improvement = float(
        wrong_per_sample[hero].mean() - correct_per_sample[hero].mean()
    )
    damage_improvement = float(
        no_damage_per_sample[damaged_hero].mean()
        - correct_per_sample[damaged_hero].mean()
    )
    live_hp_improvement = float(
        mean_hp_per_sample[hero].mean()
        - correct_per_sample[hero].mean()
    )
    time_improvement = float(
        mean_time_per_sample[hero].mean()
        - correct_per_sample[hero].mean()
    )
    positive_team_fraction = float(np.mean(
        np.asarray(eligible_improvements) > 0,
    )) if eligible_improvements else 0.0
    acceptance = {
        "correct_school_identity_improves_hero_nll": hero_improvement > 0,
        "correct_school_identity_improves_majority_of_test_teams": (
            len(eligible_improvements) >= 10
            and positive_team_fraction >= 0.60
        ),
        "recent_damage_features_improve_damaged_hero_nll": (
            int(damaged_hero.sum()) >= 100 and damage_improvement > 0
        ),
        "live_hp_state_improves_hero_nll": live_hp_improvement > 0,
        "match_time_improves_hero_nll": time_improvement > 0,
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "policy": (
            "frozen post-training audit; no retraining or threshold labels"
        ),
        "final_blind_test_games": len(audit["final_blind_test_games"]),
        "samples": {
            "all": len(x_test),
            "hero": int(hero.sum()),
            "hero_recently_damaged": int(damaged_hero.sum()),
            "eligible_teams": len(eligible_improvements),
        },
        "hero_correct_identity_nll": round(
            float(correct_per_sample[hero].mean()), 6,
        ),
        "hero_wrong_identity_nll": round(
            float(wrong_per_sample[hero].mean()), 6,
        ),
        "hero_correct_identity_improvement": round(
            hero_improvement, 6,
        ),
        "damaged_hero_correct_nll": round(
            float(correct_per_sample[damaged_hero].mean()), 6,
        ),
        "damaged_hero_no_damage_feature_nll": round(
            float(no_damage_per_sample[damaged_hero].mean()), 6,
        ),
        "damaged_hero_damage_feature_improvement": round(
            damage_improvement, 6,
        ),
        "hero_mean_imputed_hp_nll": round(
            float(mean_hp_per_sample[hero].mean()), 6,
        ),
        "hero_live_hp_improvement": round(live_hp_improvement, 6),
        "hero_mean_imputed_time_nll": round(
            float(mean_time_per_sample[hero].mean()), 6,
        ),
        "hero_match_time_improvement": round(time_improvement, 6),
        "positive_team_fraction": round(positive_team_fraction, 6),
        "teams": team_report,
        "acceptance": acceptance,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in report.items() if key != "teams"
    }, ensure_ascii=False, indent=2))
    if not all(acceptance.values()):
        raise SystemExit("conditioning audit failed; do not publish")


if __name__ == "__main__":
    main()
