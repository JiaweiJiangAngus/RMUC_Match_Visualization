import unittest

from ml.tactical_api import create_app
from ml.tactical_inference import (
    DEFAULT_DATA_DIR,
    TacticalInferenceEngine,
    load_game,
)


class TacticalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = TacticalInferenceEngine(mc_samples=4)

    def test_operator_zones_use_team_perspective(self):
        self.assertEqual("己方基地", self.engine.map.zone(2.2, 7.5, "红"))
        self.assertEqual("敌方基地", self.engine.map.zone(2.2, 7.5, "蓝"))
        self.assertEqual("中央高地", self.engine.map.zone(14.0, 7.5, "红"))

    def test_manual_terrain_capability_is_visible_to_route_layer(self):
        self.assertEqual(
            "人工确认",
            self.engine.capability_status(
                "上海交通大学", "英雄", "central_highland_400mm_jump",
            ),
        )

    def test_prediction_includes_team_prior_and_operator_fields(self):
        info, frames = load_game(DEFAULT_DATA_DIR / "1778680444314.json.gz")
        output = self.engine.predict(frames, 200, info, "红", "步兵3")
        self.assertEqual(1, len(output["predictions"]))
        prediction = output["predictions"][0]
        self.assertEqual("上海交通大学", prediction["school"])
        self.assertIn("primary_destination", prediction)
        self.assertIn("route_summary", prediction)
        horizon_10 = next(
            item for item in prediction["horizons"] if item["after_seconds"] == 10
        )
        self.assertGreater(horizon_10["team_prior_samples"], 0)
        self.assertGreater(horizon_10["team_prior_weight"], 0)

    def test_opening_seconds_fill_unavailable_history(self):
        info, frames = load_game(DEFAULT_DATA_DIR / "1778680444314.json.gz")
        output = self.engine.predict(frames, 3, info, "红", "英雄")
        self.assertTrue(output["predictions"])
        self.assertTrue(output["history_filled"])

    def test_http_health_and_validation(self):
        client = create_app(self.engine).test_client()
        self.assertTrue(client.get("/health").get_json()["ok"])
        self.assertEqual(400, client.post("/infer", json={}).status_code)


if __name__ == "__main__":
    unittest.main()
