#!/usr/bin/env python3
"""Validate all five generated tactical heatmap modes for 96 teams."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEATMAP_DIR = ROOT / "docs" / "data" / "heatmaps"
HEATMAP_PAGE = ROOT / "docs" / "heatmap.html"
HEATMAP_SCRIPT = ROOT / "docs" / "heatmap.js"
REPLAY_PAGE = ROOT / "docs" / "index.html"
REPLAY_SCRIPT = ROOT / "docs" / "app.js"


def sparse_total(encoded: str, grid_length: int) -> int:
    total = 0
    for pair in encoded.split(","):
        if not pair:
            continue
        index_text, value_text = pair.split(":", 1)
        index = int(index_text)
        value = int(value_text)
        if not 0 <= index < grid_length:
            raise AssertionError(f"sparse index {index} outside grid")
        if value <= 0:
            raise AssertionError(f"sparse value {value} must be positive")
        total += value
    return total


class TeamHeatmapDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (HEATMAP_DIR / "config.json").read_text(encoding="utf-8")
        )

    def test_geometry_and_kernel_are_high_resolution(self) -> None:
        self.assertEqual(self.config["schema_version"], 5)
        self.assertEqual(
            self.config["modes"],
            ["position", "shots", "hits", "kills", "deaths"],
        )
        self.assertEqual(self.config["grid_width"], 280)
        self.assertEqual(self.config["grid_height"], 150)
        self.assertEqual(self.config["cell_size_metres"], 0.1)
        self.assertEqual(self.config["gaussian_sigma_metres"], 0.22)
        self.assertEqual(self.config["window_seconds"], 15)
        self.assertEqual(self.config["window_count"], 28)
        self.assertEqual(
            self.config["roles"],
            ["英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"],
        )
        self.assertEqual(self.config["firing_projectiles"], 1_200_632)
        self.assertEqual(self.config["firing_event_groups"], 336_226)
        self.assertEqual(
            self.config["firing_position_matching"],
            {
                "exact_groups": 336_224,
                "adjacent_1s_groups": 2,
                "missing_groups": 0,
            },
        )
        self.assertEqual(self.config["death_events"], 7_556)
        self.assertEqual(
            self.config["death_position_detection"],
            {
                "method": "同一robot_id相邻轨迹由HP>0转为HP=0时记录归零秒坐标",
                "missing_positions": 0,
            },
        )
        self.assertEqual(self.config["hit_events"], 162_304)
        self.assertEqual(self.config["hit_event_groups"], 97_092)
        self.assertEqual(
            self.config["hit_categories"],
            {
                "17mm": 134_881,
                "42mm": 836,
                "判罚": 1_700,
                "撞击": 24_722,
                "飞镖": 165,
            },
        )
        self.assertEqual(
            self.config["hit_position_matching"],
            {
                "exact_groups": 97_081,
                "adjacent_1s_groups": 11,
                "missing_groups": 0,
            },
        )
        self.assertEqual(self.config["kill_events"], 7_193)
        kill_detection = self.config["kill_position_detection"]
        self.assertEqual(kill_detection["weapon_hit_same_second"], 19)
        self.assertEqual(kill_detection["weapon_hit_previous_1s"], 7_174)
        self.assertEqual(kill_detection["excluded_noncombat_deaths"], 363)

    def test_all_96_teams_are_present_once(self) -> None:
        schools = self.config["schools"]
        names = [entry["school"] for entry in schools]
        self.assertEqual(len(schools), 96)
        self.assertEqual(len(set(names)), 96)
        by_region = {
            region: sum(entry["region"] == region for entry in schools)
            for region in self.config["regions"]
        }
        self.assertEqual(by_region, {
            "东部赛区": 32,
            "南部赛区": 32,
            "北部赛区": 32,
        })

    def test_all_school_aggregate_matches_team_totals(self) -> None:
        aggregate = self.config["aggregate"]
        self.assertEqual(aggregate["school"], "全部学校（96队）")
        self.assertEqual(aggregate["region"], "全部赛区")
        self.assertEqual(aggregate["games"], 613)
        self.assertEqual(aggregate["team_count"], 96)
        payload = json.loads(
            (HEATMAP_DIR / aggregate["file"]).read_text(encoding="utf-8")
        )
        grid_length = self.config["grid_width"] * self.config["grid_height"]
        fields = {
            "position": "samples",
            "shots": "shots",
            "hits": "hits",
            "kills": "kills",
            "deaths": "deaths",
        }
        for mode, field in fields.items():
            series = payload if mode == "position" else payload[mode]
            expected = sum(entry[field] for entry in self.config["schools"])
            self.assertEqual(series["samples"], expected)
            self.assertEqual(aggregate[field], expected)
            self.assertEqual(
                sparse_total(series["red"], grid_length)
                + sparse_total(series["blue"], grid_length),
                expected,
            )
            self.assertEqual(
                sum(window["samples"] for window in series["windows"]),
                expected,
            )
            self.assertEqual(
                sum(role["samples"] for role in series["roles"].values()),
                expected,
            )

    def test_sparse_files_match_catalog_totals(self) -> None:
        grid_length = self.config["grid_width"] * self.config["grid_height"]
        expected_files = set()
        for entry in self.config["schools"]:
            expected_files.add(entry["file"])
            payload = json.loads(
                (HEATMAP_DIR / entry["file"]).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["school"], entry["school"])
            self.assertEqual(payload["region"], entry["region"])
            self.assertEqual(payload["games"], entry["games"])
            self.assertEqual(payload["samples"], entry["samples"])
            self.assertEqual(payload["shots"]["samples"], entry["shots"])
            self.assertEqual(payload["deaths"]["samples"], entry["deaths"])
            self.assertEqual(payload["hits"]["samples"], entry["hits"])
            self.assertEqual(payload["kills"]["samples"], entry["kills"])
            self.assertGreater(payload["games"], 0)
            self.assertGreater(payload["samples"], 0)
            self.assertGreater(payload["shots"]["samples"], 0)
            self.assertGreater(payload["deaths"]["samples"], 0)
            self.assertGreater(payload["hits"]["samples"], 0)
            self.assertGreater(payload["kills"]["samples"], 0)
            self.assertEqual(
                sparse_total(payload["red"], grid_length)
                + sparse_total(payload["blue"], grid_length),
                payload["samples"],
            )
            self.assertEqual(len(payload["windows"]), self.config["window_count"])
            window_samples = 0
            for index, window in enumerate(payload["windows"]):
                self.assertEqual(window["start"], index * 15)
                self.assertEqual(window["end"], (index + 1) * 15)
                decoded_samples = (
                    sparse_total(window["red"], grid_length)
                    + sparse_total(window["blue"], grid_length)
                )
                self.assertEqual(decoded_samples, window["samples"])
                window_samples += decoded_samples
            self.assertEqual(window_samples, payload["samples"])
            self.assertEqual(set(payload["roles"]), set(self.config["roles"]))
            role_samples = 0
            for role in self.config["roles"]:
                role_payload = payload["roles"][role]
                decoded_role_samples = (
                    sparse_total(role_payload["red"], grid_length)
                    + sparse_total(role_payload["blue"], grid_length)
                )
                self.assertEqual(decoded_role_samples, role_payload["samples"])
                self.assertEqual(
                    len(role_payload["windows"]),
                    self.config["window_count"],
                )
                role_window_samples = sum(
                    sparse_total(window["red"], grid_length)
                    + sparse_total(window["blue"], grid_length)
                    for window in role_payload["windows"]
                )
                self.assertEqual(role_window_samples, role_payload["samples"])
                role_samples += decoded_role_samples
            self.assertEqual(role_samples, payload["samples"])

            shots = payload["shots"]
            self.assertEqual(
                sparse_total(shots["red"], grid_length)
                + sparse_total(shots["blue"], grid_length),
                shots["samples"],
            )
            self.assertEqual(len(shots["windows"]), self.config["window_count"])
            shot_window_samples = sum(
                sparse_total(window["red"], grid_length)
                + sparse_total(window["blue"], grid_length)
                for window in shots["windows"]
            )
            self.assertEqual(shot_window_samples, shots["samples"])
            self.assertEqual(set(shots["roles"]), set(self.config["roles"]))
            shot_role_samples = 0
            for role in self.config["roles"]:
                role_payload = shots["roles"][role]
                decoded_role_samples = (
                    sparse_total(role_payload["red"], grid_length)
                    + sparse_total(role_payload["blue"], grid_length)
                )
                self.assertEqual(decoded_role_samples, role_payload["samples"])
                self.assertEqual(
                    len(role_payload["windows"]),
                    self.config["window_count"],
                )
                role_window_samples = sum(
                    sparse_total(window["red"], grid_length)
                    + sparse_total(window["blue"], grid_length)
                    for window in role_payload["windows"]
                )
                self.assertEqual(role_window_samples, role_payload["samples"])
                shot_role_samples += decoded_role_samples
            self.assertEqual(shot_role_samples, shots["samples"])

            deaths = payload["deaths"]
            self.assertEqual(
                sparse_total(deaths["red"], grid_length)
                + sparse_total(deaths["blue"], grid_length),
                deaths["samples"],
            )
            self.assertEqual(len(deaths["windows"]), self.config["window_count"])
            death_window_samples = sum(
                sparse_total(window["red"], grid_length)
                + sparse_total(window["blue"], grid_length)
                for window in deaths["windows"]
            )
            self.assertEqual(death_window_samples, deaths["samples"])
            self.assertEqual(set(deaths["roles"]), set(self.config["roles"]))
            death_role_samples = 0
            for role in self.config["roles"]:
                role_payload = deaths["roles"][role]
                decoded_role_samples = (
                    sparse_total(role_payload["red"], grid_length)
                    + sparse_total(role_payload["blue"], grid_length)
                )
                self.assertEqual(decoded_role_samples, role_payload["samples"])
                self.assertEqual(
                    len(role_payload["windows"]),
                    self.config["window_count"],
                )
                role_window_samples = sum(
                    sparse_total(window["red"], grid_length)
                    + sparse_total(window["blue"], grid_length)
                    for window in role_payload["windows"]
                )
                self.assertEqual(role_window_samples, role_payload["samples"])
                death_role_samples += decoded_role_samples
            self.assertEqual(death_role_samples, deaths["samples"])

            for mode in ("hits", "kills"):
                series = payload[mode]
                self.assertEqual(
                    sparse_total(series["red"], grid_length)
                    + sparse_total(series["blue"], grid_length),
                    series["samples"],
                )
                self.assertEqual(
                    len(series["windows"]),
                    self.config["window_count"],
                )
                series_window_samples = sum(
                    sparse_total(window["red"], grid_length)
                    + sparse_total(window["blue"], grid_length)
                    for window in series["windows"]
                )
                self.assertEqual(series_window_samples, series["samples"])
                self.assertEqual(
                    set(series["roles"]),
                    set(self.config["roles"]),
                )
                series_role_samples = 0
                for role in self.config["roles"]:
                    role_payload = series["roles"][role]
                    decoded_role_samples = (
                        sparse_total(role_payload["red"], grid_length)
                        + sparse_total(role_payload["blue"], grid_length)
                    )
                    self.assertEqual(
                        decoded_role_samples,
                        role_payload["samples"],
                    )
                    self.assertEqual(
                        len(role_payload["windows"]),
                        self.config["window_count"],
                    )
                    role_window_samples = sum(
                        sparse_total(window["red"], grid_length)
                        + sparse_total(window["blue"], grid_length)
                        for window in role_payload["windows"]
                    )
                    self.assertEqual(
                        role_window_samples,
                        role_payload["samples"],
                    )
                    series_role_samples += decoded_role_samples
                self.assertEqual(series_role_samples, series["samples"])
        actual_files = {
            path.name for path in HEATMAP_DIR.glob("[0-9][0-9][0-9].json")
        }
        self.assertEqual(actual_files, expected_files)

    def test_render_uses_gaussian_density_not_a_60_square_grid(self) -> None:
        page = HEATMAP_PAGE.read_text(encoding="utf-8")
        script = HEATMAP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("gaussianKernel", script)
        self.assertIn("gaussianBlur", script)
        self.assertIn("gaussian_sigma_metres / state.config.cell_size_metres", script)
        self.assertNotIn("60 × 60", page)
        self.assertNotIn("60 * 60", script)
        self.assertIn('id="school-select"', page)
        self.assertIn('id="heatmap-type-select"', page)
        self.assertIn('<option value="shots">打弹热力图</option>', page)
        self.assertIn('<option value="hits">受击热力图</option>', page)
        self.assertIn('<option value="kills">击杀热力图</option>', page)
        self.assertIn('<option value="deaths">阵亡热力图</option>', page)
        self.assertIn('<option value="canonical-blue">统一蓝方</option>', page)
        self.assertIn('id="role-select"', page)
        self.assertIn('id="window-play"', page)
        self.assertIn("toggleWindowPlayback", script)
        self.assertIn("MODE_PRESENTATION", script)
        self.assertIn('roleLabel: "受击兵种"', script)
        self.assertIn('roleLabel: "被击杀兵种"', script)
        self.assertIn('kills: "kills"', script)
        self.assertIn('state.side === "canonical-blue"', script)
        self.assertIn("blue[index] + red[mirroredIndex]", script)
        self.assertIn('name === "all"', script)
        self.assertIn("state.config.aggregate", script)
        self.assertIn('option.value = "all"', script)
        self.assertIn('href="./app.css?v=24"', page)
        self.assertNotIn("RM_LADDER", page)

    def test_heatmap_is_a_separate_page_from_replay(self) -> None:
        replay_page = REPLAY_PAGE.read_text(encoding="utf-8")
        replay_script = REPLAY_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('href="./heatmap.html"', replay_page)
        self.assertNotIn('id="heatmap-toggle"', replay_page)
        self.assertNotIn("drawHeatmap", replay_script)


if __name__ == "__main__":
    unittest.main()
