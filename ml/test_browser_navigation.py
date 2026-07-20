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
const router=require('./docs/terrain-router.js');
const nav=JSON.parse(fs.readFileSync('./docs/data/models/terrain_navigation.json','utf8'));
function plan(start,end,school,role){
  const value=core.terrainRoute(nav,start,end,school,role);
  return {length:core.routeLength(value.route),points:value.route.length,corrected:value.corrected,passages:value.passages,target:value.target};
}
function validateAllTunnelGates(){
  const specs=[
    {id:'blue_road_tunnel',start:[18.55,14.5],end:[18.55,11.8],passage:'B2公路隧道'},
    {id:'red_road_tunnel',start:[9.43,.5],end:[9.43,3.1],passage:'R2公路隧道'},
    {id:'blue_highland_tunnel',start:[12.2,1.18],end:[16,1.18],passage:'B6高地隧道'},
    {id:'red_highland_tunnel',start:[15.8,13.83],end:[12.1,13.83],passage:'R6高地隧道'},
  ];
  const implementations=[['prediction-worker',core.terrainRoute],['full-match-router',router.terrainRoute]];
  const violations=[];
  let checks=0;
  let allowedChecks=0;
  let blockedChecks=0;
  for(const [implementation,routeFunction] of implementations){
    for(const [school,roles] of Object.entries(nav.teams)){
      for(const [role,profile] of Object.entries(roles)){
        for(const spec of specs){
          checks+=1;
          const gate=nav.gates.find(candidate=>candidate.id===spec.id);
          const allowed=profile.abilities.includes(gate.category);
          const value=routeFunction(nav,spec.start,spec.end,school,role);
          const crossed=value.route.slice(1).some((point,index)=>
            router.segmentHitsPolygon(value.route[index],point,gate.polygon));
          const labelled=value.passages.includes(spec.passage);
          const reached=Math.hypot(value.target[0]-spec.end[0],value.target[1]-spec.end[1])<.15;
          if(allowed) allowedChecks+=1; else blockedChecks+=1;
          if(crossed!==allowed||labelled!==allowed||!reached){
            violations.push({implementation,school,role,gate:spec.id,allowed,crossed,labelled,reached});
          }
        }
      }
    }
  }
  return {checks,allowedChecks,blockedChecks,teams:Object.keys(nav.teams).length,violations};
}
function validateStrictTunnelLabels(){
  const violations=[];
  let checks=0;
  for(const [school,roles] of Object.entries(nav.teams)){
    for(const [role,profile] of Object.entries(roles)){
      for(const category of ['road_tunnel','highland_tunnel']){
        checks+=1;
        const observation=profile.tunnel_observations[category];
        const allowed=profile.abilities.includes(category);
        const observed=observation.crossings>=1;
        const manualOverride=['positive_confirmed','negative_confirmed'].includes(observation.training_label);
        if(allowed!==observation.allowed
          || (!manualOverride&&allowed!==observed)
          || (observation.training_label==='negative_unobserved'&&(observed||allowed))){
          violations.push({school,role,category,allowed,...observation});
        }
      }
    }
  }
  return {checks,violations};
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
  ,neuInfantryTunnel:plan([18.55,14.5],[18.55,11.8],'东北大学','步兵3')
  ,neuSentinelTunnel:plan([18.55,14.5],[18.55,11.8],'东北大学','哨兵')
  ,speed:(()=>{
    const up=router.terrainRoute(nav,[19.7,12],[19.7,14.3],'东北大学','步兵3');
    const down=router.terrainRoute(nav,[19.7,14.3],[19.7,12],'东北大学','步兵3');
    const fly=router.terrainRoute(nav,[16,14.7],[12,14.7],'东北大学','步兵3');
    return {
      up:router.terrainMotion(nav,up.route[0],up.route,up.actions,2),
      down:router.terrainMotion(nav,down.route[0],down.route,down.actions,2),
      fly:router.terrainMotion(nav,fly.route[0],fly.route,fly.actions,2),
    };
  })()
  ,globalTunnelValidation:validateAllTunnelGates()
  ,strictTunnelLabels:validateStrictTunnelLabels()
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

    def test_neu_infantry_cannot_use_tunnel_but_confirmed_sentinel_can(self):
        infantry = self.result["neuInfantryTunnel"]
        sentinel = self.result["neuSentinelTunnel"]
        self.assertGreater(infantry["points"], 2)
        self.assertNotIn("B2公路隧道", infantry["passages"])
        self.assertEqual(2, sentinel["points"])
        self.assertIn("B2公路隧道", sentinel["passages"])

    def test_all_teams_and_ground_roles_obey_all_four_tunnel_gates(self):
        validation = self.result["globalTunnelValidation"]
        self.assertEqual(44, validation["teams"])
        # 44 teams × 5 ground roles × 4 red/blue tunnel gates × 2 routers.
        self.assertEqual(1760, validation["checks"])
        self.assertGreater(validation["allowedChecks"], 0)
        self.assertGreater(validation["blockedChecks"], 0)
        self.assertEqual([], validation["violations"])

    def test_unobserved_tunnel_passages_are_negative_for_every_team_role(self):
        validation = self.result["strictTunnelLabels"]
        self.assertEqual(440, validation["checks"])
        self.assertEqual([], validation["violations"])

    def test_terrain_motion_changes_speed_for_both_step_directions_and_fly_ramp(self):
        speed = self.result["speed"]
        self.assertEqual(0.45, speed["up"]["multiplier"])
        self.assertEqual(0.62, speed["down"]["multiplier"])
        self.assertEqual(1.12, speed["fly"]["multiplier"])
        self.assertIn("上公路台阶", speed["up"]["action"]["label"])
        self.assertIn("下公路台阶", speed["down"]["action"]["label"])


if __name__ == "__main__":
    unittest.main()
