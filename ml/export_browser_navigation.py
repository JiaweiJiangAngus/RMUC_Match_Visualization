#!/usr/bin/env python3
"""Export exact terrain topology and compact team capabilities for the web worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import terrain_crossing_points as terrain  # noqa: E402


DEFAULT_CAPABILITIES = ROOT / "analysis" / "outputs" / "team_ground_terrain_capabilities.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "models" / "terrain_navigation.json"
POSITIVE_STATUS = {"人工确认", "已证实", "较强迹象"}
HIGH_DIRECTIONS = {
    "central_highland_step": {"blue": "negative_x", "red": "positive_x"},
    "road_step": {"blue": "positive_y", "red": "negative_y"},
    "slope_43": {"blue": "negative_y", "red": "positive_y"},
    "trapezoid_highland_step": {"blue": "positive_y", "red": "negative_y"},
}


def field_geometry(feature: terrain.Feature) -> list[list[float]]:
    points = np.asarray(
        [terrain.map_to_field(*point) for point in feature.map_geometry_px],
        dtype=np.float32,
    )
    if len(points) > 100:
        points = cv2.approxPolyDP(
            points.reshape(-1, 1, 2), 0.02, feature.geometry_type == "polygon",
        ).reshape(-1, 2)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in points]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, default=DEFAULT_CAPABILITIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = terrain.build_features()
    by_id = {feature.feature_id: feature for feature in features}
    capability_payload = json.loads(args.capabilities.read_text(encoding="utf-8"))

    teams: dict[str, dict] = {}
    for team in capability_payload["teams"]:
        roles = {}
        for robot in team["robots"]:
            abilities = []
            reverse = {"crossings": 0, "games": 0, "allowed": False}
            for item in robot["capabilities"]:
                if item["status"] in POSITIVE_STATUS:
                    abilities.append(item["ability"])
                if item["ability"] == "fly_ramp":
                    evidence = item.get("trajectory_directions", {}).get("reverse", {})
                    reverse = {
                        "crossings": int(evidence.get("crossings", 0)),
                        "games": int(evidence.get("games", 0)),
                        "allowed": (
                            int(evidence.get("crossings", 0)) >= 2
                            and int(evidence.get("games", 0)) >= 2
                        ),
                    }
            roles[robot["role"]] = {"abilities": sorted(abilities), "reverse_fly_ramp": reverse}
        teams[team["school"]] = roles

    gates = []
    for feature in features:
        if feature.kind != "crossing_gate":
            continue
        gates.append(
            {
                "id": feature.feature_id,
                "side": feature.side,
                "category": feature.category,
                "gate_index": feature.gate_index,
                "center": [round(value, 4) for value in terrain.map_to_field(*feature.center_map_px)],
                "polygon": field_geometry(feature),
                "default_direction": (
                    "right_to_left" if feature.category == "fly_ramp" and feature.side == "blue"
                    else "left_to_right" if feature.category == "fly_ramp" and feature.side == "red"
                    else "asymmetric_up_down"
                ),
                "high_direction": HIGH_DIRECTIONS.get(feature.category, {}).get(feature.side),
            }
        )

    output = {
        "schema_version": 2,
        "field_size_m": [terrain.FIELD_WIDTH_M, terrain.FIELD_HEIGHT_M],
        "routing": {
            "grid_m": 0.35,
            "reverse_fly_ramp_min_crossings": 2,
            "reverse_fly_ramp_min_games": 2,
            "ascending_requires_positive_capability": True,
            "descending_uses_designed_entry_by_default": True,
        },
        "regions": {
            "central_highland": field_geometry(by_id["central_highland_region"]),
            "blue_trapezoid_highland": field_geometry(by_id["blue_trapezoid_highland_top"]),
            "red_trapezoid_highland": field_geometry(by_id["red_trapezoid_highland_top"]),
        },
        "gates": gates,
        "teams": teams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    reverse_roles = sum(
        role["reverse_fly_ramp"]["allowed"]
        for roles in teams.values()
        for role in roles.values()
    )
    print(
        f"exported {len(gates)} gates, {len(teams)} teams, "
        f"{reverse_roles} reverse-fly-ramp team/roles to {args.output}"
    )


if __name__ == "__main__":
    main()
