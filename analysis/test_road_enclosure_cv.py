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
        self.assertEqual(9, len(self.data.walls))
        for wall in self.data.walls:
            polygon = road_enclosure_cv.segment_polygon(wall)
            self.assertEqual(4, len(polygon))
            side = math.dist(polygon[0], polygon[3])
            self.assertGreater(side, 10, wall.wall_id)

    def test_supply_fences_are_detected_at_the_three_visible_rows(self):
        detected = self.data.detected_lines
        self.assertAlmostEqual(40, detected["supply_top"][0][1], delta=6)
        self.assertAlmostEqual(182, detected["supply_middle"][0][1], delta=6)
        self.assertAlmostEqual(271, detected["supply_lower"][0][1], delta=6)


if __name__ == "__main__":
    unittest.main()
