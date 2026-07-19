#!/usr/bin/env python3
"""Operator-facing tactical inference built on the trajectory MLP.

The trajectory model stays responsible for state-conditioned motion.  This
adapter turns its coarse future coordinates into semantic destinations,
terrain passages, capability warnings, and empirically calibrated confidence
that a custom client can render directly.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from .train_trajectory import (
        DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT,
        FEATURE_NAMES,
        FIELD_HEIGHT_M,
        FIELD_WIDTH_M,
        HISTORY_OFFSETS,
        HP,
        REGULATION_DURATION_S,
        SIDES,
        TARGET_VX3_INDEX,
        TARGET_VY3_INDEX,
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        TrajectoryMLP,
        canonical_xy,
        index_frame,
        sample_features,
        valid_position,
    )
except ImportError:
    from train_trajectory import (  # type: ignore[no-redef]
        DEFAULT_DATA_DIR,
        DEFAULT_OUTPUT,
        FEATURE_NAMES,
        FIELD_HEIGHT_M,
        FIELD_WIDTH_M,
        HISTORY_OFFSETS,
        HP,
        REGULATION_DURATION_S,
        SIDES,
        TARGET_VX3_INDEX,
        TARGET_VY3_INDEX,
        TARGET_X_INDEX,
        TARGET_Y_INDEX,
        TrajectoryMLP,
        canonical_xy,
        index_frame,
        sample_features,
        valid_position,
    )

try:
    from analysis import terrain_crossing_points as terrain
except ModuleNotFoundError:
    import sys

    ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
    from analysis import terrain_crossing_points as terrain


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "ml" / "artifacts" / "trajectory_mlp.metrics.json"
DEFAULT_CAPABILITIES = ROOT / "analysis" / "outputs" / "team_ground_terrain_capabilities.json"
DEFAULT_TEAM_PRIORS = ROOT / "ml" / "artifacts" / "team_tactical_priors.json"
GROUND_TYPES = ("英雄", "工程", "步兵3", "步兵4", "哨兵")
DEFAULT_MC_SAMPLES = 32
PRIMARY_HORIZON = 10

ABILITY_ZH = {
    "fly_ramp": "飞坡",
    "road_tunnel": "公路隧道",
    "road_step": "公路台阶",
    "rough_road": "起伏路段",
    "central_highland_step": "中央高地台阶",
    "highland_tunnel": "高地隧道",
    "slope_43": "43°坡",
    "trapezoid_highland_step": "梯形高地台阶",
    "central_highland_400mm_jump": "400mm高地跳跃",
}
POSITIVE_CAPABILITY = {"人工确认", "已证实", "较强迹象"}

# Fixed structure centres read from the registered 28 m x 15 m map.  They are
# used only for operator-facing semantic labels, not as model inputs.
STRUCTURE_CENTRES = {
    "red_base": (2.2, 7.5, 2.0),
    "red_outpost": (6.1, 7.5, 1.45),
    "blue_outpost": (21.9, 7.5, 1.45),
    "blue_base": (25.8, 7.5, 2.0),
}


def world_samples(canonical: np.ndarray, side: str) -> np.ndarray:
    result = np.asarray(canonical, dtype=np.float32).copy()
    result[..., 0] *= FIELD_WIDTH_M
    result[..., 1] *= FIELD_HEIGHT_M
    if side == "蓝":
        result[..., 0] = FIELD_WIDTH_M - result[..., 0]
        result[..., 1] = FIELD_HEIGHT_M - result[..., 1]
    return result


def side_school(info: dict, side: str) -> str:
    return str(info.get("red" if side == "红" else "blue", ""))


def time_phase(second: float) -> str:
    if second < 60:
        return "opening"
    if second < 330:
        return "middle"
    return "endgame"


class TeamTacticalPrior:
    def __init__(self, path: Path) -> None:
        self.index: dict[tuple[str, str, str, str, int], tuple[int, dict[str, float]]] = {}
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            counts = record["destinations"]
            total = max(1, sum(counts.values()))
            key = (
                record["school"], record["role"], record["current_zone"],
                record["phase"], int(record["horizon"]),
            )
            self.index[key] = (
                int(record["samples"]),
                {zone: count / total for zone, count in counts.items()},
            )

    def get(
        self,
        school: str,
        role: str,
        current_zone: str,
        second: int,
        horizon: int,
    ) -> tuple[int, dict[str, float]]:
        return self.index.get(
            (school, role, current_zone, time_phase(second), horizon),
            (0, {}),
        )


class TacticalMap:
    def __init__(self) -> None:
        self.features = terrain.build_features()
        self.central = next(
            item for item in self.features if item.feature_id == "central_highland_region"
        )
        self.trapezoids = {
            item.side: item
            for item in self.features
            if item.category == "trapezoid_highland_top"
        }
        self.gates = tuple(item for item in self.features if item.kind == "crossing_gate")
        self.ledges = tuple(item for item in self.features if item.kind == "conditional_ledge")

    @staticmethod
    def _inside(x: float, y: float, feature: terrain.Feature) -> bool:
        px = terrain.field_to_map(x, y)
        return terrain.point_in_polygon(*px, feature.map_geometry_px)

    def zone(self, x: float, y: float, perspective_side: str) -> str:
        x = max(0.0, min(FIELD_WIDTH_M, float(x)))
        y = max(0.0, min(FIELD_HEIGHT_M, float(y)))
        own = "red" if perspective_side == "红" else "blue"
        enemy = "blue" if own == "red" else "red"

        for relation, key in (
            ("己方基地", f"{own}_base"),
            ("己方前哨站", f"{own}_outpost"),
            ("敌方前哨站", f"{enemy}_outpost"),
            ("敌方基地", f"{enemy}_base"),
        ):
            cx, cy, radius = STRUCTURE_CENTRES[key]
            if math.hypot(x - cx, y - cy) <= radius:
                return relation

        for map_side, feature in self.trapezoids.items():
            if self._inside(x, y, feature):
                relation = "己方" if map_side == own else "敌方"
                return f"{relation}梯形高地"
        if self._inside(x, y, self.central):
            return "中央高地"

        canonical_x = x if perspective_side == "红" else FIELD_WIDTH_M - x
        canonical_y = y if perspective_side == "红" else FIELD_HEIGHT_M - y
        if canonical_x < 7.0:
            depth = "己方后场"
        elif canonical_x < 12.0:
            depth = "己方前场"
        elif canonical_x <= 16.0:
            depth = "中央低地"
        elif canonical_x <= 21.0:
            depth = "敌方前场"
        else:
            depth = "敌方后场"
        if canonical_y >= 11.0:
            lane = "上路"
        elif canonical_y <= 4.0:
            lane = "下路"
        else:
            lane = "中路"
        return f"{depth}·{lane}"

    def route_passages(self, points: Iterable[tuple[float, float]]) -> list[dict]:
        route = list(points)
        passages: list[dict] = []
        seen: set[str] = set()
        for start, end in zip(route, route[1:]):
            start_px, end_px = terrain.field_to_map(*start), terrain.field_to_map(*end)
            segment_gates = [
                gate for gate in self.gates if self._crosses_gate(start, end, gate)
            ]
            for gate in segment_gates:
                if gate.feature_id in seen:
                    continue
                seen.add(gate.feature_id)
                prefix = "B" if gate.side == "blue" else "R"
                passages.append(
                    {
                        "feature_id": gate.feature_id,
                        "ability": gate.category,
                        "name": f"{prefix}{gate.gate_index} {ABILITY_ZH[gate.category]}",
                    }
                )
            if any(gate.category == "central_highland_step" for gate in segment_gates):
                continue
            ascends_highland = (
                not self._inside(*start, self.central)
                and self._inside(*end, self.central)
            )
            if ascends_highland and any(
                terrain.segment_hits_feature(start_px, end_px, ledge) for ledge in self.ledges
            ):
                key = "central_highland_400mm_jump"
                if key not in seen:
                    seen.add(key)
                    passages.append(
                        {
                            "feature_id": key,
                            "ability": key,
                            "name": ABILITY_ZH[key],
                        }
                    )
        return passages

    @staticmethod
    def _crosses_gate(
        start: tuple[float, float],
        end: tuple[float, float],
        gate: terrain.Feature,
    ) -> bool:
        start_px, end_px = terrain.field_to_map(*start), terrain.field_to_map(*end)
        if not terrain.segment_hits_feature(start_px, end_px, gate):
            return False
        field_points = [terrain.map_to_field(*point) for point in gate.map_geometry_px]
        horizontal = gate.category in {
            "central_highland_step", "fly_ramp", "rough_road", "highland_tunnel",
        }
        along_values = [point[0] if horizontal else point[1] for point in field_points]
        lateral_values = [point[1] if horizontal else point[0] for point in field_points]
        start_along = start[0] if horizontal else start[1]
        end_along = end[0] if horizontal else end[1]
        start_lateral = start[1] if horizontal else start[0]
        end_lateral = end[1] if horizontal else end[0]
        centre = (min(along_values) + max(along_values)) / 2
        threshold = (max(along_values) - min(along_values)) * 0.20
        crosses_centre = (
            (start_along - centre) * (end_along - centre) < 0
            and abs(start_along - centre) >= threshold
            and abs(end_along - centre) >= threshold
        )
        lateral_margin = 0.45
        lateral_ok = (
            min(lateral_values) - lateral_margin
            <= (start_lateral + end_lateral) / 2
            <= max(lateral_values) + lateral_margin
        )
        return crosses_centre and lateral_ok


class TacticalInferenceEngine:
    def __init__(
        self,
        model_path: Path = DEFAULT_OUTPUT,
        metrics_path: Path = DEFAULT_METRICS,
        capabilities_path: Path = DEFAULT_CAPABILITIES,
        team_priors_path: Path = DEFAULT_TEAM_PRIORS,
        mc_samples: int = DEFAULT_MC_SAMPLES,
    ) -> None:
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least 2")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint["feature_names"] != FEATURE_NAMES:
            raise ValueError("model feature schema does not match this code")
        self.model = TrajectoryMLP(**checkpoint["model_kwargs"])
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self._model_lock = threading.Lock()
        self.mean = checkpoint["feature_mean"].numpy()
        self.std = checkpoint["feature_std"].numpy()
        self.horizons = tuple(int(value) for value in checkpoint["horizons"])
        self.mc_samples = mc_samples
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        self.capabilities = self._load_capabilities(capabilities_path)
        self.team_prior = TeamTacticalPrior(team_priors_path)
        self.map = TacticalMap()

    @staticmethod
    def _load_capabilities(path: Path) -> dict[tuple[str, str, str], str]:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = {}
        for team in payload["teams"]:
            for robot in team["robots"]:
                for ability in robot["capabilities"]:
                    result[(team["school"], robot["role"], ability["ability"])] = ability["status"]
        return result

    def capability_status(self, school: str, role: str, ability: str) -> str:
        return self.capabilities.get((school, role, ability), "未知")

    def empirical_reliability(self, horizon: int, moving: bool) -> float:
        group = "moving" if moving else "all"
        return float(self.metrics[f"{horizon}s"][group]["learned_mlp"]["zone_accuracy"])

    def _mc_future(self, features: np.ndarray, side: str, seed: int) -> np.ndarray:
        normalized = ((features - self.mean) / self.std).astype(np.float32)
        batch = torch.from_numpy(np.repeat(normalized[None], self.mc_samples, axis=0))
        with self._model_lock:
            torch.manual_seed(seed)
            self.model.train()  # Enable dropout only; LayerNorm has no running statistics.
            with torch.inference_mode():
                residual = self.model(batch).numpy()
            self.model.eval()
        current = features[[TARGET_X_INDEX, TARGET_Y_INDEX]]
        canonical = np.clip(current[None, None, :] + residual, 0.0, 1.0)
        return world_samples(canonical, side)

    def predict(
        self,
        frames: dict[int, dict[tuple[str, str], list]],
        second: int,
        info: dict,
        side_filter: str | None = None,
        role_filter: str | None = None,
    ) -> dict:
        started = time.perf_counter()
        duration = REGULATION_DURATION_S
        if second not in frames:
            return {
                "schema_version": 1,
                "game_id": info.get("game_id"),
                "second": second,
                "purpose": "operator_tactical_hint_not_low_level_control",
                "predictions": [],
                "error": "current frame is missing",
                "total_inference_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        working_frames = dict(frames)
        available = sorted(frame_second for frame_second in frames if frame_second <= second)
        filled_history = []
        for offset in HISTORY_OFFSETS:
            wanted = second - offset
            if wanted in working_frames:
                continue
            candidates = [frame_second for frame_second in available if frame_second <= wanted]
            fallback = candidates[-1] if candidates else available[0]
            working_frames[wanted] = frames[fallback]
            filled_history.append({"wanted_second": wanted, "used_second": fallback})
        predictions = []
        for side in SIDES:
            if side_filter and side != side_filter:
                continue
            school = side_school(info, side)
            for role in GROUND_TYPES:
                if role_filter and role != role_filter:
                    continue
                current = working_frames.get(second, {}).get((side, role))
                if not valid_position(current) or float(current[HP] or 0.0) <= 0:
                    continue
                try:
                    features = np.asarray(
                        sample_features(working_frames, second, side, role, duration),
                        dtype=np.float32,
                    )
                except KeyError:
                    continue
                robot_started = time.perf_counter()
                samples = self._mc_future(features, side, seed=second * 1000 + int(current[0]))
                median_path = np.median(samples, axis=0)
                current_world = tuple(float(value) for value in world_samples(
                    np.asarray(canonical_xy(current, side)), side,
                ))
                speed = math.hypot(
                    float(features[TARGET_VX3_INDEX]) * FIELD_WIDTH_M,
                    float(features[TARGET_VY3_INDEX]) * FIELD_HEIGHT_M,
                )
                moving = speed >= 0.15
                current_zone = self.map.zone(*current_world, side)
                horizons = []
                for h_index, horizon in enumerate(self.horizons):
                    zone_counts = Counter(
                        self.map.zone(x, y, side) for x, y in samples[:, h_index, :]
                    )
                    model_probabilities = {
                        zone: count / self.mc_samples for zone, count in zone_counts.items()
                    }
                    prior_samples, prior_probabilities = self.team_prior.get(
                        school, role, current_zone, second, horizon,
                    )
                    # At most 30% historical style prior. Sparse records have
                    # less influence, while the live full-state model remains dominant.
                    prior_weight = 0.30 * prior_samples / (prior_samples + 60.0)
                    combined = {
                        zone: (1.0 - prior_weight) * model_probabilities.get(zone, 0.0)
                        + prior_weight * prior_probabilities.get(zone, 0.0)
                        for zone in set(model_probabilities) | set(prior_probabilities)
                    }
                    reliability = self.empirical_reliability(horizon, moving)
                    candidates = []
                    for zone, probability in sorted(
                        combined.items(), key=lambda item: item[1], reverse=True,
                    )[:3]:
                        candidates.append(
                            {
                                "zone": zone,
                                "probability": round(probability, 3),
                                "display_confidence": round(probability * reliability, 3),
                                "state_model_probability": round(
                                    model_probabilities.get(zone, 0.0), 3,
                                ),
                                "team_prior_probability": round(
                                    prior_probabilities.get(zone, 0.0), 3,
                                ),
                            }
                        )
                    centre = median_path[h_index]
                    spread = np.linalg.norm(samples[:, h_index, :] - centre[None, :], axis=1)
                    horizons.append(
                        {
                            "after_seconds": horizon,
                            "predicted_xy_m": [round(float(centre[0]), 3), round(float(centre[1]), 3)],
                            "candidate_zones": candidates,
                            "empirical_zone_accuracy": round(reliability, 3),
                            "team_prior_samples": prior_samples,
                            "team_prior_weight": round(prior_weight, 3),
                            "mc_p90_spread_m": round(float(np.percentile(spread, 90)), 3),
                        }
                    )

                primary_index = (
                    self.horizons.index(PRIMARY_HORIZON)
                    if PRIMARY_HORIZON in self.horizons
                    else len(self.horizons) - 1
                )
                primary = horizons[primary_index]
                primary_xy = tuple(map(float, median_path[primary_index]))
                primary_distance = math.dist(current_world, primary_xy)
                mean_route = [current_world] + [
                    tuple(map(float, point)) for point in median_path[:primary_index + 1]
                ]
                passages = (
                    [] if primary_distance < 0.8 else self.map.route_passages(mean_route)
                )
                warnings = []
                for passage in passages:
                    status = self.capability_status(school, role, passage["ability"])
                    passage["capability_status"] = status
                    passage["supported"] = status in POSITIVE_CAPABILITY
                    if status not in POSITIVE_CAPABILITY:
                        warnings.append(f"{passage['name']}能力证据为{status}")
                destination = primary["candidate_zones"][0]["zone"]
                route_labels = [current_zone, *(p["name"] for p in passages), destination]
                route_labels = [
                    value for index, value in enumerate(route_labels)
                    if index == 0 or value != route_labels[index - 1]
                ]
                if primary_distance < 0.8:
                    movement = "驻守/小范围调整"
                elif current_zone == destination:
                    movement = "区域内转移"
                elif passages:
                    movement = "经地形入口转移"
                else:
                    movement = "直接推进"
                predictions.append(
                    {
                        "school": school,
                        "side": side,
                        "robot_type": role,
                        "robot_id": int(current[0]),
                        "current_xy_m": [round(current_world[0], 3), round(current_world[1], 3)],
                        "current_zone": current_zone,
                        "speed_mps_3s": round(speed, 3),
                        "primary_destination": {
                            "after_seconds": primary["after_seconds"],
                            **primary["candidate_zones"][0],
                        },
                        "movement": movement,
                        "route_summary": " -> ".join(route_labels),
                        "passages": passages,
                        "warnings": warnings,
                        "horizons": horizons,
                        "inference_ms": round((time.perf_counter() - robot_started) * 1000, 3),
                    }
                )
        return {
            "schema_version": 1,
            "game_id": info.get("game_id"),
            "second": second,
            "purpose": "operator_tactical_hint_not_low_level_control",
            "history_filled": filled_history,
            "predictions": predictions,
            "total_inference_ms": round((time.perf_counter() - started) * 1000, 3),
        }


def load_game(path: Path) -> tuple[dict, dict[int, dict[tuple[str, str], list]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        game = json.load(handle)
    frames = {int(second): index_frame(rows) for second, rows in game["frames"].items()}
    return game["info"], frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-file", type=Path)
    parser.add_argument("--game-id", type=int)
    parser.add_argument("--second", type=int, default=200)
    parser.add_argument("--side", choices=SIDES)
    parser.add_argument("--robot-type", choices=GROUND_TYPES)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--capabilities", type=Path, default=DEFAULT_CAPABILITIES)
    parser.add_argument("--team-priors", type=Path, default=DEFAULT_TEAM_PRIORS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.game_file) == bool(args.game_id):
        raise SystemExit("provide exactly one of --game-file or --game-id")
    path = args.game_file or DEFAULT_DATA_DIR / f"{args.game_id}.json.gz"
    if not path.is_file():
        raise SystemExit(f"game file does not exist: {path}")
    info, frames = load_game(path)
    engine = TacticalInferenceEngine(
        model_path=args.model,
        metrics_path=args.metrics,
        capabilities_path=args.capabilities,
        team_priors_path=args.team_priors,
        mc_samples=args.mc_samples,
    )
    output = engine.predict(frames, args.second, info, args.side, args.robot_type)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
