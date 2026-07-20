import math
import unittest

from analysis import road_enclosure_cv
from analysis import terrain_crossing_points as terrain


class RoadEnclosureCvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = road_enclosure_cv.extract_blue_enclosures(terrain.MAP_PATH)

    def test_road_region_follows_detected_map_edges(self):
        polygon = self.data.region_px
        self.assertEqual(6, len(polygon))
        self.assertAlmostEqual(1138, polygon[0][0], delta=12)
        self.assertAlmostEqual(2026, polygon[1][0], delta=12)
        self.assertAlmostEqual(351, polygon[2][1], delta=12)

    def test_playable_walls_have_nonzero_collision_area(self):
        self.assertEqual(13, len(self.data.walls))
        for wall in self.data.walls:
            polygon = road_enclosure_cv.segment_polygon(wall)
            self.assertEqual(4, len(polygon))
            side = math.dist(polygon[0], polygon[3])
            self.assertGreater(side, 5, wall.wall_id)

    def test_registered_trace_coordinates_match_original_map(self):
        by_id = {wall.wall_id: wall for wall in self.data.walls}
        self.assertAlmostEqual(2026, by_id["blue_road_outer_right"].start_px[0], delta=12)
        self.assertAlmostEqual(2055, by_id["blue_base_left"].start_px[0], delta=12)
        self.assertAlmostEqual(855, by_id["blue_base_top"].start_px[1], delta=12)

    def test_corrected_wall_trace_keeps_openings(self):
        by_id = {wall.wall_id: wall for wall in self.data.walls}
        self.assertNotIn("blue_road_lower_right", by_id)
        self.assertIn("blue_fortress_outer", by_id)
        self.assertIn("blue_base_left", by_id)

    def test_manufactured_wall_axes_are_orthogonal(self):
        for wall in self.data.walls:
            dx = abs(wall.end_px[0] - wall.start_px[0])
            dy = abs(wall.end_px[1] - wall.start_px[1])
            if dx >= dy * 2:
                self.assertAlmostEqual(wall.start_px[1], wall.end_px[1], places=6)
            elif dy >= dx * 2:
                self.assertAlmostEqual(wall.start_px[0], wall.end_px[0], places=6)


if __name__ == "__main__":
    unittest.main()
