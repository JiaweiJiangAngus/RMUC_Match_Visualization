import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("terrain_crossing_points.py")
SPEC = importlib.util.spec_from_file_location("terrain_crossing_points", MODULE_PATH)
terrain = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = terrain
SPEC.loader.exec_module(terrain)


class TerrainSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = terrain.build_features()

    def test_eight_gates_per_side(self):
        gates = [feature for feature in self.features if feature.kind == "crossing_gate"]
        self.assertEqual(len(gates), 16)
        self.assertEqual({feature.gate_index for feature in gates}, set(range(1, 9)))

    def test_red_gate_centres_are_180_degree_rotations(self):
        by_id = {feature.feature_id: feature for feature in self.features}
        for spec in terrain.BLUE_GATE_SPECS:
            blue = by_id[f"blue_{spec.category}"]
            red = by_id[f"red_{spec.category}"]
            self.assertEqual(red.center_map_px, terrain.rotate_point_180(blue.center_map_px))

    def test_gate_centres_are_detectable(self):
        for feature in self.features:
            if feature.kind != "crossing_gate":
                continue
            x_m, y_m = terrain.map_to_field(*feature.center_map_px)
            matches = terrain.detect_features(x_m, y_m, self.features, padding_m=0)
            self.assertIn(feature.feature_id, {match["feature_id"] for match in matches})

    def test_fly_ramps_have_symmetric_default_directions(self):
        by_id = {feature.feature_id: terrain.feature_to_dict(feature) for feature in self.features}
        self.assertEqual("blue_right_to_left", by_id["blue_fly_ramp"]["physical_direction"])
        self.assertEqual("red_left_to_right", by_id["red_fly_ramp"]["physical_direction"])

    def test_400mm_ledge_requires_jump_capable(self):
        start = terrain.map_to_field(1450, 450)
        end = terrain.map_to_field(1560, 450)
        blocked = terrain.classify_transition(*start, *end, features=self.features)
        ledges = [hit for hit in blocked if hit["category"] == "central_highland_ledge_400mm"]
        self.assertTrue(ledges)
        self.assertTrue(all(hit["allowed"] is False for hit in ledges))
        allowed = terrain.classify_transition(
            *start, *end, capabilities={"jump_capable"}, features=self.features,
        )
        ledges = [hit for hit in allowed if hit["category"] == "central_highland_ledge_400mm"]
        self.assertTrue(all(hit["allowed"] is True for hit in ledges))

    def test_central_step_is_not_labelled_as_400mm_ledge(self):
        start = terrain.map_to_field(1440, 645)
        end = terrain.map_to_field(1610, 645)
        hits = terrain.classify_transition(*start, *end, features=self.features)
        categories = {hit["category"] for hit in hits}
        self.assertIn("central_highland_step", categories)
        self.assertNotIn("central_highland_ledge_400mm", categories)

    def test_b5_and_r5_are_expanded(self):
        by_id = {feature.feature_id: feature for feature in self.features}
        for feature_id in ("blue_central_highland_step", "red_central_highland_step"):
            geometry = by_id[feature_id].map_geometry_px
            width = max(x for x, _ in geometry) - min(x for x, _ in geometry)
            height = max(y for _, y in geometry) - min(y for _, y in geometry)
            self.assertAlmostEqual(width, 190)
            self.assertAlmostEqual(height, 240)

    def test_highland_ledge_uses_dense_opencv_edge(self):
        ledges = [feature for feature in self.features if feature.kind == "conditional_ledge"]
        self.assertEqual(len(ledges), 4)
        self.assertTrue(all(len(feature.map_geometry_px) >= 380 for feature in ledges))
        self.assertTrue(all("opencv_edge" in feature.source for feature in ledges))

    def test_trapezoid_boundary_is_dense_and_symmetric(self):
        by_id = {feature.feature_id: feature for feature in self.features}
        blue = by_id["blue_trapezoid_highland_boundary"]
        red = by_id["red_trapezoid_highland_boundary"]
        self.assertEqual(blue.kind, "terrain_boundary")
        self.assertGreaterEqual(len(blue.map_geometry_px), 600)
        xs = [x for x, _ in blue.map_geometry_px]
        ys = [y for _, y in blue.map_geometry_px]
        self.assertGreater(min(xs), 1800)
        self.assertGreater(min(ys), 830)
        self.assertLess(max(xs) - min(xs), 250)
        self.assertLess(max(ys) - min(ys), 200)
        self.assertEqual(
            red.map_geometry_px,
            terrain.rotate_geometry_180(blue.map_geometry_px),
        )


if __name__ == "__main__":
    unittest.main()
