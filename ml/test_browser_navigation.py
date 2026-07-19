import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for browser-worker tests")
class BrowserTerrainNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = r"""
const fs=require('fs');
const core=require('./docs/prediction-worker.js');
const nav=JSON.parse(fs.readFileSync('./docs/data/models/terrain_navigation.json','utf8'));
function plan(start,end,school,role){
  const value=core.terrainRoute(nav,start,end,school,role);
  return {length:core.routeLength(value.route),points:value.route.length,corrected:value.corrected,passages:value.passages,target:value.target};
}
console.log(JSON.stringify({
  around:plan([5,7.5],[23,7.5],'未知学校','英雄'),
  blockedAscent:plan([6,7.5],[14,7.5],'未知学校','英雄'),
  reverseAllowed:plan([12,14.7],[16,14.7],'上海交通大学','步兵3'),
  reverseBlocked:plan([12,14.7],[16,14.7],'上海交通大学','英雄'),
  trapezoidBlocked:plan([24,1],[24,3],'未知学校','英雄'),
  trapezoidSlope:plan([24,1],[24,3],'五邑大学','英雄'),
  trapezoidDescent:plan([24,3],[24,1],'未知学校','英雄'),
  roadStepAscent:plan([19.7,12],[19.7,14.3],'未知学校','英雄'),
  roadStepDescent:plan([19.7,14.3],[19.7,12],'未知学校','英雄'),
  confirmedJump:plan([6,12],[14,10],'上海交通大学','英雄')
}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        cls.result = json.loads(result.stdout)

    def test_low_route_goes_around_central_highland(self):
        self.assertGreater(self.result["around"]["length"], 20.0)
        self.assertGreater(self.result["around"]["points"], 2)

    def test_unproven_ascent_stays_outside_highland(self):
        value = self.result["blockedAscent"]
        self.assertTrue(value["corrected"])
        self.assertIn("能力不足·停在地形外", value["passages"])
        self.assertLess(value["target"][0], 10.0)

    def test_reverse_fly_ramp_is_role_specific(self):
        self.assertIn("B1反飞坡", self.result["reverseAllowed"]["passages"])
        self.assertEqual(2, self.result["reverseAllowed"]["points"])
        self.assertNotIn("B1反飞坡", self.result["reverseBlocked"]["passages"])
        self.assertGreater(self.result["reverseBlocked"]["points"], 2)

    def test_trapezoid_ascent_and_descent_are_not_equivalent(self):
        self.assertTrue(self.result["trapezoidBlocked"]["corrected"])
        self.assertIn("B7上43°坡", self.result["trapezoidSlope"]["passages"])
        self.assertIn("B8下梯形高地台阶", self.result["trapezoidDescent"]["passages"])

    def test_road_step_ascent_requires_evidence_but_descent_does_not(self):
        self.assertGreater(self.result["roadStepAscent"]["points"], 2)
        self.assertIn("B3下公路台阶", self.result["roadStepDescent"]["passages"])

    def test_confirmed_jump_can_use_non_step_highland_edge(self):
        self.assertIn("400mm跳跃上高地", self.result["confirmedJump"]["passages"])


if __name__ == "__main__":
    unittest.main()
