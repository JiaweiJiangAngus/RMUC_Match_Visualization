import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "docs" / "data" / "models" / "match_simulation.json"


class MatchSimulationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    def test_all_44_teams_have_28_turn_games(self):
        self.assertEqual(44, len(self.model["teams"]))
        self.assertEqual(28, self.model["bin_count"])
        self.assertEqual(15, self.model["bin_seconds"])
        for team in self.model["teams"].values():
            self.assertTrue(team["games"])
            self.assertTrue(all(len(game["bins"]) == 28 for game in team["games"]))

    def test_default_tdt_rps_matchup_is_available(self):
        self.assertEqual("TDT", self.model["teams"]["东北大学"]["team"])
        self.assertEqual("RPS", self.model["teams"]["中国石油大学（华东）"]["team"])

    def test_damage_aggregates_are_nonnegative(self):
        for team in self.model["teams"].values():
            aggregate = team["aggregate"]
            self.assertGreaterEqual(aggregate["damage_per_game"], 0)
            self.assertGreaterEqual(aggregate["base_damage_per_game"], 0)
            self.assertGreaterEqual(aggregate["outpost_damage_per_game"], 0)
            self.assertGreaterEqual(aggregate["fortress_enemy_seconds_per_game"], 0)


@unittest.skipUnless(shutil.which("node"), "Node.js is required for simulator tests")
class MatchSimulationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = r"""
const fs=require('fs');
const sim=require('./docs/match-simulator.js');
const model=JSON.parse(fs.readFileSync('./docs/data/models/match_simulation.json','utf8'));
function summary(state) {
  return {
    winner:state.outcome.winner,
    turn:state.turn,
    structures:state.structures,
    damage:state.damage,
    scores:[state.outcome.redScore,state.outcome.blueScore]
  };
}
const first=sim.runFullMatch(model,'东北大学','中国石油大学（华东）',20260719);
const repeat=sim.runFullMatch(model,'东北大学','中国石油大学（华东）',20260719);
const monte=sim.runMonteCarlo(model,'东北大学','中国石油大学（华东）',100,20260719);
console.log(JSON.stringify({first:summary(first),repeat:summary(repeat),monte}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        cls.result = json.loads(result.stdout)

    def test_fixed_seed_is_reproducible(self):
        self.assertEqual(self.result["first"], self.result["repeat"])

    def test_match_ends_and_health_stays_bounded(self):
        match = self.result["first"]
        self.assertLessEqual(match["turn"], 28)
        self.assertIn(match["winner"], {"red", "blue", "draw"})
        for side in ("red", "blue"):
            self.assertGreaterEqual(match["structures"][side]["base"], 0)
            self.assertLessEqual(match["structures"][side]["base"], 5000)
            self.assertGreaterEqual(match["structures"][side]["outpost"], 0)
            self.assertLessEqual(match["structures"][side]["outpost"], 1500)
            self.assertTrue(all(value >= 0 for value in match["damage"][side].values()))

    def test_monte_carlo_accounts_for_every_game(self):
        result = self.result["monte"]
        self.assertEqual(
            result["games"],
            result["redWins"] + result["blueWins"] + result["draws"],
        )
        self.assertGreaterEqual(result["redBaseHp"], 0)
        self.assertGreaterEqual(result["blueBaseHp"], 0)


if __name__ == "__main__":
    unittest.main()
