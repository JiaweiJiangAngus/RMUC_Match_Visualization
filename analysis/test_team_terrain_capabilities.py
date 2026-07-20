import unittest
from collections import defaultdict

from analysis import terrain_crossing_points as terrain
from analysis.team_terrain_capabilities import (
    DEFAULT_MANUAL_LABELS,
    Evidence,
    TrackPoint,
    build_gates,
    central_highland_jump_ascents,
    evidence_status,
    load_manual_confirmations,
)


def point(second, x, y, z, hp=200):
    return TrackPoint(float(second), float(x), float(y), float(z), float(hp))


class TerrainCapabilityDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features = terrain.build_features()

    def test_has_eight_symmetric_gate_pairs(self):
        gates = build_gates(self.features)
        self.assertEqual(16, len(gates))
        self.assertEqual(8, len({gate.ability for gate in gates}))
        for ability in {gate.ability for gate in gates}:
            self.assertEqual({"blue", "red"}, {gate.side for gate in gates if gate.ability == ability})

    def test_repeated_stable_non_entrance_ascent_is_retained(self):
        # A real database trace from the north-west 400 mm ledge, shortened to
        # the local window used by the detector.
        points = [
            point(23, 9.124, 12.238, 0.051),
            point(24, 9.726, 11.864, -0.157),
            point(25, 11.419, 10.623, 0.039),
            point(26, 11.638, 10.903, 0.615),
            point(27, 12.156, 11.086, 0.637),
        ]
        ascents = central_highland_jump_ascents(points, self.features)
        self.assertEqual(1, len(ascents))
        self.assertGreaterEqual(ascents[0]["height_gain_m"], 0.2)

    def test_highland_tunnel_exit_is_not_a_jump(self):
        # This trace enters beside B6; the drawn gate is narrower than the UWB
        # path, so the calibrated tunnel buffer must still reject it.
        points = [
            point(127, 18.105, 3.398, -0.318),
            point(128, 16.980, 2.810, -0.310),
            point(129, 15.815, 2.139, 0.120),
            point(130, 15.625, 1.781, -0.053),
            point(131, 15.221, 1.554, 0.141),
        ]
        self.assertEqual([], central_highland_jump_ascents(points, self.features))

    def test_stationary_edge_jitter_is_not_a_jump(self):
        points = [
            point(101, 11.920, 13.182, 0.322),
            point(102, 12.184, 13.322, 0.086),
            point(103, 12.357, 12.909, 0.484),
            point(104, 12.233, 12.833, 0.202),
            point(105, 11.826, 12.830, 0.283),
        ]
        self.assertEqual([], central_highland_jump_ascents(points, self.features))

    def test_manual_confirmations_load_and_override_trajectory_grade(self):
        evidence = defaultdict(Evidence)
        self.assertEqual(13, load_manual_confirmations(DEFAULT_MANUAL_LABELS, evidence))
        item = evidence[("上海交通大学", "英雄", "central_highland_400mm_jump")]
        item.trajectory_crossings = 0
        self.assertEqual(
            ("人工确认", 1.0, "positive_confirmed"),
            evidence_status(item, sample_games=10),
        )
        rejected = evidence[("东北大学", "步兵3", "road_tunnel")]
        self.assertEqual(
            ("人工排除", 1.0, "negative_confirmed"),
            evidence_status(rejected, sample_games=18),
        )

    def test_directional_evidence_is_counted_separately(self):
        item = Evidence()
        item.add_trajectory(1, 10, "blue_fly_ramp", {"traversal": "forward"})
        item.add_trajectory(2, 20, "blue_fly_ramp", {"traversal": "reverse"})
        item.add_trajectory(3, 30, "red_fly_ramp", {"traversal": "reverse"})
        self.assertEqual(1, item.trajectory_direction_counts["forward"])
        self.assertEqual(2, item.trajectory_direction_counts["reverse"])
        self.assertEqual({2, 3}, item.trajectory_direction_games["reverse"])

    def test_unobserved_tunnel_is_a_negative_training_label(self):
        self.assertEqual(
            ("样本未通过", 1.0, "negative_unobserved"),
            evidence_status(Evidence(), sample_games=12, ability="road_tunnel"),
        )

    def test_one_complete_tunnel_passage_is_enough_to_allow_it(self):
        item = Evidence()
        item.add_trajectory(7, 123, "blue_highland_tunnel")
        status, confidence, label = evidence_status(
            item, sample_games=12, ability="highland_tunnel",
        )
        self.assertEqual("已通过", status)
        self.assertGreaterEqual(confidence, 0.9)
        self.assertEqual("positive_observed", label)


if __name__ == "__main__":
    unittest.main()
