#!/usr/bin/env python3
"""Validate the generated 96-team position heatmap dataset."""

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
        self.assertEqual(self.config["schema_version"], 1)
        self.assertEqual(self.config["grid_width"], 280)
        self.assertEqual(self.config["grid_height"], 150)
        self.assertEqual(self.config["cell_size_metres"], 0.1)
        self.assertEqual(self.config["gaussian_sigma_metres"], 0.22)

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
            self.assertGreater(payload["games"], 0)
            self.assertGreater(payload["samples"], 0)
            self.assertEqual(
                sparse_total(payload["red"], grid_length)
                + sparse_total(payload["blue"], grid_length),
                payload["samples"],
            )
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

    def test_heatmap_is_a_separate_page_from_replay(self) -> None:
        replay_page = REPLAY_PAGE.read_text(encoding="utf-8")
        replay_script = REPLAY_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('href="./heatmap.html"', replay_page)
        self.assertNotIn('id="heatmap-toggle"', replay_page)
        self.assertNotIn("drawHeatmap", replay_script)


if __name__ == "__main__":
    unittest.main()
