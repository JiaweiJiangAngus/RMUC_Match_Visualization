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
from analysis import road_enclosure_cv  # noqa: E402


DEFAULT_CAPABILITIES = ROOT / "analysis" / "outputs" / "team_ground_terrain_capabilities.json"
DEFAULT_MOTION_PROFILES = ROOT / "analysis" / "outputs" / "team_terrain_motion_profiles.json"
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "models" / "terrain_navigation.json"
POSITIVE_STATUS = {"人工确认", "已通过", "已证实", "较强迹象"}
TUNNEL_ABILITIES = {"road_tunnel", "highland_tunnel"}
HIGH_DIRECTIONS = {
    "central_highland_step": {"blue": "negative_x", "red": "positive_x"},
    "road_step": {"blue": "positive_y", "red": "negative_y"},
    "slope_43": {"blue": "negative_y", "red": "positive_y"},
    "trapezoid_highland_step": {"blue": "positive_y", "red": "negative_y"},
}

# Detection polygons stay tight so telemetry is labelled with the correct
# interface.  Routing blockers are deliberately larger: they close the entire
# physical opening and overlap its adjoining walls/terrain, preventing the
# 0.35 m navigation grid from treating the edge of a gate as a free crack.
ROUTING_BLOCKER_DIMENSIONS_PX = {
    "road_tunnel": (150, 96),
    "road_step": (150, 96),
    # 起伏路横跨整条道路，不能在两侧留下可绕过的“平地缝隙”。
    "rough_road": (380, 180),
    "central_highland_step": (220, 280),
    "slope_43": (185, 100),
    "trapezoid_highland_step": (145, 100),
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


def map_points_to_field(points: list | tuple, closed: bool = True) -> list[list[float]]:
    converted = np.asarray([terrain.map_to_field(*point) for point in points], dtype=np.float32)
    if len(converted) > 100:
        converted = cv2.approxPolyDP(converted.reshape(-1, 1, 2), 0.02, closed).reshape(-1, 2)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in converted]


def routing_blocker_geometry(feature: terrain.Feature) -> list[list[float]]:
    spec = next(item for item in terrain.BLUE_GATE_SPECS if item.category == feature.category)
    if feature.category == "fly_ramp":
        # B1/R1 occupies the outer lane between the field edge and the wall at
        # map y≈121.  It must meet, but not cover, the adjacent R6/B6 lane.
        blue = ((1064, 34), (1274, 34), (1274, 122), (1064, 122))
    elif feature.category == "rough_road":
        # Close the 0.64 m flat crack that remained between B3/R3 and B4/R4.
        # The road is continuous from the outer edge of the step to the field
        # boundary, so the blocker is intentionally asymmetric around label B4.
        blue = ((1680, 0), (2124, 0), (2124, 180), (1680, 180))
    elif feature.category == "highland_tunnel":
        # B6/R6 is bounded by the central highland on one side and the B1/R1
        # separator wall on the other.  Extending it to the field edge would
        # incorrectly block a robot that is legitimately using the fly ramp.
        blue = ((1073, 1028), (1283, 1028), (1283, 1163), (1073, 1163))
    else:
        length_px, width_px = ROUTING_BLOCKER_DIMENSIONS_PX[feature.category]
        blue = terrain.oriented_box(spec.center_px, spec.axis_angle_deg, length_px, width_px)
    geometry = blue if feature.side == "blue" else terrain.rotate_geometry_180(blue)
    return map_points_to_field(geometry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, default=DEFAULT_CAPABILITIES)
    parser.add_argument("--motion-profiles", type=Path, default=DEFAULT_MOTION_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = terrain.build_features()
    by_id = {feature.feature_id: feature for feature in features}
    enclosure = road_enclosure_cv.extract_blue_enclosures(terrain.MAP_PATH)
    capability_payload = json.loads(args.capabilities.read_text(encoding="utf-8"))
    motion_payload = json.loads(args.motion_profiles.read_text(encoding="utf-8"))

    teams: dict[str, dict] = {}
    for team in capability_payload["teams"]:
        roles = {}
        for robot in team["robots"]:
            abilities = []
            tunnel_observations = {}
            reverse = {"crossings": 0, "games": 0, "allowed": False}
            for item in robot["capabilities"]:
                if item["status"] in POSITIVE_STATUS:
                    abilities.append(item["ability"])
                if item["ability"] in TUNNEL_ABILITIES:
                    tunnel_observations[item["ability"]] = {
                        "crossings": int(item.get("trajectory_crossings", 0)),
                        "games": int(item.get("trajectory_games", 0)),
                        "allowed": item["status"] in POSITIVE_STATUS,
                        "training_label": item["training_label"],
                    }
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
            roles[robot["role"]] = {
                "abilities": sorted(abilities),
                "tunnel_observations": tunnel_observations,
                "reverse_fly_ramp": reverse,
                "terrain_motion_profiles": motion_payload["teams"][team["school"]][robot["role"]],
            }
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
                "routing_blocker_polygon": routing_blocker_geometry(feature),
                "default_direction": (
                    "right_to_left" if feature.category == "fly_ramp" and feature.side == "blue"
                    else "left_to_right" if feature.category == "fly_ramp" and feature.side == "red"
                    else "asymmetric_up_down"
                ),
                "high_direction": HIGH_DIRECTIONS.get(feature.category, {}).get(feature.side),
            }
        )

    static_obstacles = []
    for wall in enclosure.walls:
        blue_polygon = road_enclosure_cv.segment_polygon(wall)
        for side in ("blue", "red"):
            polygon = blue_polygon if side == "blue" else terrain.rotate_geometry_180(blue_polygon)
            static_obstacles.append(
                {
                    "id": wall.wall_id if side == "blue" else wall.wall_id.replace("blue_", "red_", 1),
                    "side": side,
                    "category": wall.category,
                    "polygon": map_points_to_field(polygon),
                    "blocks_movement": True,
                    "blocks_ground_fire": True,
                    "source": (
                        "opencv_canny_hough_map"
                        if side == "blue"
                        else "opencv_canny_hough_map_180_rotated"
                    ),
                }
            )

    blue_road_region = enclosure.region_px
    red_road_region = terrain.rotate_geometry_180(blue_road_region)

    output = {
        "schema_version": 9,
        "field_size_m": [terrain.FIELD_WIDTH_M, terrain.FIELD_HEIGHT_M],
        "routing": {
            "grid_m": 0.35,
            "reverse_fly_ramp_min_crossings": 2,
            "reverse_fly_ramp_min_games": 2,
            "ascending_requires_positive_capability": True,
            "descending_uses_designed_entry_by_default": True,
            "tunnel_capability_policy": "one_or_more_observed_complete_passages_else_blocked",
            "terrain_route_profiles": {
                "fly_ramp": {
                    "centerline_required": True,
                    "entry_clearance_m": 0.08,
                    "exit_clearance_m": 0.18,
                    "straight_runup_source": "team_role_terrain_motion_profile",
                },
                "central_highland_400mm_jump": {
                    "perpendicular_entry_required": True,
                    "straight_runup_m": 1.35,
                    "lip_clearance_m": 0.10,
                    "landing_m": 0.55,
                    "source": "field_geometry_and_motion_prior",
                },
                "service_return": {
                    "selection": "shortest_reachable_route",
                    "failed_route_retry_seconds": 3,
                    "own_half_staging_x_m": 6.5,
                },
            },
            # Coarse one-second traversal priors.  They intentionally model
            # setup/landing time as well as the obstacle itself; they can be
            # replaced by team-role empirical distributions when national
            # telemetry is available.
            "terrain_speed_multipliers": {
                "central_highland_step": {"up": 0.32, "down": 0.48},
                "road_step": {"up": 0.45, "down": 0.62},
                "trapezoid_highland_step": {"up": 0.34, "down": 0.5},
                "slope_43": {"up": 0.5, "down": 0.65},
                "central_highland_400mm_jump": {"up": 0.26, "down": 0.42},
                "fly_ramp": {"forward": 1.12, "reverse": 0.62},
                "rough_road": {"through": 0.58},
                "road_tunnel": {"through": 0.72},
                "highland_tunnel": {"through": 0.66},
            },
            "default_terrain_motion_profiles": motion_payload["global"],
        },
        "regions": {
            "central_highland": field_geometry(by_id["central_highland_region"]),
            "blue_trapezoid_highland": field_geometry(by_id["blue_trapezoid_highland_top"]),
            "red_trapezoid_highland": field_geometry(by_id["red_trapezoid_highland_top"]),
        },
        "enclosures": {
            "blue_road_region": map_points_to_field(blue_road_region),
            "red_road_region": map_points_to_field(red_road_region),
            "method": "opencv_canny_hough_with_user_confirmed_gate_openings",
        },
        "static_obstacles": static_obstacles,
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
        f"exported {len(gates)} gates, {len(static_obstacles)} static obstacles, {len(teams)} teams, "
        f"{reverse_roles} reverse-fly-ramp team/roles to {args.output}"
    )


if __name__ == "__main__":
    main()
