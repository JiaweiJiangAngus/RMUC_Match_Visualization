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
  return {length:core.routeLength(value.route),points:value.route.length,route:value.route,corrected:value.corrected,passages:value.passages,target:value.target};
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
          const blockerCrossed=value.route.slice(1).some((point,index)=>
            router.segmentHitsPolygon(value.route[index],point,gate.routing_blocker_polygon));
          const labelled=value.passages.includes(spec.passage);
          const reached=Math.hypot(value.target[0]-spec.end[0],value.target[1]-spec.end[1])<.15;
          if(allowed) allowedChecks+=1; else blockedChecks+=1;
          if(crossed!==allowed||(!allowed&&blockerCrossed)||labelled!==allowed||(allowed&&!reached)){
            violations.push({implementation,school,role,gate:spec.id,allowed,crossed,blockerCrossed,labelled,reached});
          }
        }
      }
    }
  }
  return {checks,allowedChecks,blockedChecks,teams:Object.keys(nav.teams).length,violations};
}
function validateRoutingBlockers(){
  const violations=[];
  const bounds=polygon=>({
    minX:Math.min(...polygon.map(point=>point[0])),maxX:Math.max(...polygon.map(point=>point[0])),
    minY:Math.min(...polygon.map(point=>point[1])),maxY:Math.max(...polygon.map(point=>point[1])),
  });
  for(const gate of nav.gates){
    if(!gate.routing_blocker_polygon?.length){violations.push({gate:gate.id,error:'missing'});continue;}
    const detected=bounds(gate.polygon), blocker=bounds(gate.routing_blocker_polygon), epsilon=.001;
    if(blocker.minX>detected.minX+epsilon||blocker.maxX<detected.maxX-epsilon
      ||blocker.minY>detected.minY+epsilon||blocker.maxY<detected.maxY-epsilon){
      violations.push({gate:gate.id,error:'does_not_contain_detection',detected,blocker});
    }
  }
  const byId=Object.fromEntries(nav.gates.map(gate=>[gate.id,gate]));
  const blueRoadTunnel=bounds(byId.blue_road_tunnel.routing_blocker_polygon);
  const blueRoadStep=bounds(byId.blue_road_step.routing_blocker_polygon);
  const redRoadTunnel=bounds(byId.red_road_tunnel.routing_blocker_polygon);
  const redRoadStep=bounds(byId.red_road_step.routing_blocker_polygon);
  if(blueRoadTunnel.maxX+1e-3<blueRoadStep.minX)violations.push({error:'blue_B2_B3_gap'});
  if(redRoadStep.maxX+1e-3<redRoadTunnel.minX)violations.push({error:'red_R2_R3_gap'});
  const blueB6=bounds(byId.blue_highland_tunnel.routing_blocker_polygon);
  const redB6=bounds(byId.red_highland_tunnel.routing_blocker_polygon);
  const blueB1=bounds(byId.blue_fly_ramp.routing_blocker_polygon);
  const redB1=bounds(byId.red_fly_ramp.routing_blocker_polygon);
  if(blueB6.minY>redB1.maxY+1e-3)violations.push({error:'blue_B6_red_R1_gap'});
  if(redB6.maxY+1e-3<blueB1.minY)violations.push({error:'red_R6_blue_B1_gap'});
  if(blueB1.maxY<14.999)violations.push({error:'blue_B1_boundary_gap'});
  if(redB1.minY>.001)violations.push({error:'red_R1_boundary_gap'});
  const blueRough=bounds(byId.blue_rough_road.routing_blocker_polygon);
  const redRough=bounds(byId.red_rough_road.routing_blocker_polygon);
  if(blueRough.minX>blueRoadStep.maxX+1e-3)violations.push({error:'blue_B3_B4_gap'});
  if(redRough.maxX+1e-3<redRoadStep.minX)violations.push({error:'red_R3_R4_gap'});
  return {schema:nav.schema_version,gates:nav.gates.length,violations};
}
function validateDeniedBypasses(){
  const specs=[
    ['blue_fly_ramp',[16,14.7],[12,14.7]],['red_fly_ramp',[12,.3],[16,.3]],
    ['blue_road_tunnel',[18.55,14.5],[18.55,11.8]],['red_road_tunnel',[9.43,.5],[9.43,3.1]],
    ['blue_road_step',[19.7,11.8],[19.7,14.5]],['red_road_step',[8.3,3.1],[8.3,.5]],
    ['blue_rough_road',[20.6,14.3],[23.8,14.3]],['red_rough_road',[7.4,.7],[4.2,.7]],
    ['blue_highland_tunnel',[12.2,1.18],[16,1.18]],['red_highland_tunnel',[15.8,13.83],[12.1,13.83]],
  ];
  const implementations=[['prediction-worker',core.terrainRoute],['full-match-router',router.terrainRoute]];
  const violations=[];
  for(const [implementation,routeFunction] of implementations){
    for(const [gateId,start,end] of specs){
      const gate=nav.gates.find(item=>item.id===gateId);
      const value=routeFunction(nav,start,end,'未知学校','英雄');
      const crossed=value.route.slice(1).some((point,index)=>
        router.segmentHitsPolygon(value.route[index],point,gate.routing_blocker_polygon));
      if(crossed)violations.push({implementation,gateId,route:value.route});
    }
  }
  return {checks:implementations.length*specs.length,violations};
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
function validateEnclosureRoutes(){
  const cases=[
    {name:'blue_supply_lower_fence',start:[24.8,11.8],end:[24.8,13.0]},
    {name:'red_supply_lower_fence',start:[3.2,3.2],end:[3.2,2.0]},
    {name:'blue_road_lower_fence',start:[16.0,12.4],end:[16.0,13.8]},
    {name:'red_road_lower_fence',start:[12.0,2.6],end:[12.0,1.2]},
    {name:'blue_supply_zone',start:[24.0,10.0],end:[26.2,13.45]},
    {name:'red_supply_zone',start:[4.0,5.0],end:[1.8,1.55]},
  ];
  const walls=nav.static_obstacles.filter(item=>item.blocks_movement!==false);
  const implementations=[['prediction-worker',core.terrainRoute],['full-match-router',router.terrainRoute]];
  const violations=[];
  let checks=0;
  for(const [implementation,routeFunction] of implementations){
    for(const spec of cases){
      checks+=1;
      const value=routeFunction(nav,spec.start,spec.end,'东北大学','步兵3');
      const crossed=value.route.slice(1).some((point,index)=>walls.some(wall=>
        router.segmentHitsPolygon(value.route[index],point,wall.polygon)));
      const reached=Math.hypot(value.target[0]-spec.end[0],value.target[1]-spec.end[1])<.15;
      if(crossed||!reached||value.route.length<2) violations.push({implementation,...spec,crossed,reached,route:value.route});
    }
  }
  return {checks,walls:walls.length,violations};
}
console.log(JSON.stringify({
  around:plan([5,7.5],[23,7.5],'华南农业大学','步兵3'),
  blockedAscent:plan([6,7.5],[14,7.5],'未知学校','英雄'),
  reverseAllowed:plan([12,14.7],[16,14.7],'上海交通大学','步兵3'),
  reverseBlocked:plan([12,14.7],[16,14.7],'上海交通大学','英雄'),
  trapezoidBlocked:plan([24,1],[24,3],'未知学校','英雄'),
  trapezoidSlope:plan([24,1],[24,3],'五邑大学','英雄'),
  trapezoidDescent:plan([24,3],[24,1],'未知学校','英雄'),
  roadStepAscent:plan([19.7,12],[19.7,14.3],'未知学校','英雄'),
  roadStepDescent:plan([19.7,14.3],[19.7,12],'未知学校','英雄'),
  alignedRoadStepAscent:plan([19.35,12],[19.95,14.3],'东北大学','步兵3'),
  alignedRoadStepDescent:plan([19.95,14.3],[19.35,12],'东北大学','步兵3'),
  confirmedJump:plan([6,12],[14,10],'上海交通大学','英雄')
  ,neuInfantryTunnel:plan([18.55,14.5],[18.55,11.8],'东北大学','步兵3')
  ,neuSentinelTunnel:plan([18.55,14.5],[18.55,11.8],'东北大学','哨兵')
  ,speed:(()=>{
    const up=router.terrainRoute(nav,[19.7,12],[19.7,14.3],'东北大学','步兵3');
    const down=router.terrainRoute(nav,[19.7,14.3],[19.7,12],'东北大学','步兵3');
    const fly=router.terrainRoute(nav,[16,14.7],[12,14.7],'东北大学','步兵3');
    return {
      up:router.terrainMotion(nav,up.route[1],up.route.slice(1),up.actions,2),
      down:router.terrainMotion(nav,down.route[1],down.route.slice(1),down.actions,2),
      fly:router.terrainMotion(nav,fly.route[0],fly.route,fly.actions,2),
    };
  })()
  ,motionProfile:nav.teams['东北大学']['步兵3'].terrain_motion_profiles.fly_ramp
  ,roadStepMotionProfile:nav.teams['东北大学']['步兵3'].terrain_motion_profiles.road_step
  ,globalTunnelValidation:validateAllTunnelGates()
  ,strictTunnelLabels:validateStrictTunnelLabels()
  ,enclosureRoutes:validateEnclosureRoutes()
  ,routingBlockers:validateRoutingBlockers()
  ,deniedBypasses:validateDeniedBypasses()
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

    def test_road_step_model_aligns_ascent_and_descent_straight_through_gate(self):
        for key in ("alignedRoadStepAscent", "alignedRoadStepDescent"):
            value = self.result[key]
            self.assertFalse(value["corrected"])
            self.assertEqual(4, value["points"])
            self.assertAlmostEqual(value["route"][1][0], value["route"][2][0], places=6)
        profile = self.result["roadStepMotionProfile"]
        self.assertEqual("team_role", profile["source_scope"])
        self.assertGreaterEqual(profile["samples"], 5)
        self.assertTrue(profile["route_alignment_enabled"])
        for direction in ("up", "down"):
            self.assertEqual("team_role", profile["directions"][direction]["source_scope"])
            self.assertGreaterEqual(profile["directions"][direction]["samples"], 5)
            self.assertTrue(profile["directions"][direction]["route_alignment_enabled"])

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

    def test_every_terrain_gate_has_a_gap_free_routing_blocker(self):
        validation = self.result["routingBlockers"]
        self.assertEqual(8, validation["schema"])
        self.assertEqual(16, validation["gates"])
        self.assertEqual([], validation["violations"])

    def test_tunnel_bypass_is_reclassified_as_the_adjacent_step(self):
        value = self.result["neuInfantryTunnel"]
        self.assertIn("B3下公路台阶", value["passages"])

    def test_denied_routes_cannot_squeeze_around_any_external_gate(self):
        validation = self.result["deniedBypasses"]
        self.assertEqual(20, validation["checks"])
        self.assertEqual([], validation["violations"])

    def test_road_and_supply_enclosures_never_clip_in_either_router(self):
        validation = self.result["enclosureRoutes"]
        self.assertEqual(12, validation["checks"])
        self.assertGreaterEqual(validation["walls"], 18)
        self.assertEqual([], validation["violations"])

    def test_terrain_motion_changes_speed_for_both_step_directions_and_fly_ramp(self):
        speed = self.result["speed"]
        self.assertEqual(0.45, speed["up"]["multiplier"])
        self.assertEqual(0.62, speed["down"]["multiplier"])
        self.assertEqual(1.12, speed["fly"]["multiplier"])
        self.assertIn("上公路台阶", speed["up"]["action"]["label"])
        self.assertIn("下公路台阶", speed["down"]["action"]["label"])

    def test_fly_ramp_motion_profile_is_learned_per_team_role(self):
        profile = self.result["motionProfile"]
        self.assertEqual("team_role", profile["source_scope"])
        self.assertGreaterEqual(profile["samples"], 5)
        self.assertEqual(1, profile["alignment_probability"])
        self.assertEqual(3, len(profile["acceleration_multipliers"]))

    def test_all_44_teams_and_roles_have_terrain_behavior_profiles(self):
        navigation = json.loads(
            (ROOT / "docs" / "data" / "models" / "terrain_navigation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(44, len(navigation["teams"]))
        team_specific = {"fly_ramp": 0, "road_step_up": 0, "road_step_down": 0}
        for roles in navigation["teams"].values():
            self.assertEqual({"英雄", "工程", "步兵3", "步兵4", "哨兵"}, set(roles))
            for role in roles.values():
                profiles = role["terrain_motion_profiles"]
                self.assertEqual({"fly_ramp", "road_step"}, set(profiles))
                self.assertIn(profiles["fly_ramp"]["source_scope"], {"team_role", "global_fallback"})
                if profiles["fly_ramp"]["source_scope"] == "team_role":
                    team_specific["fly_ramp"] += 1
                for direction in ("up", "down"):
                    value = profiles["road_step"]["directions"][direction]
                    self.assertIn(value["source_scope"], {"team_role", "global_fallback"})
                    self.assertIn("straight_crossing_probability", value)
                    if value["source_scope"] == "team_role":
                        team_specific[f"road_step_{direction}"] += 1
        self.assertGreaterEqual(team_specific["fly_ramp"], 30)
        self.assertGreaterEqual(team_specific["road_step_up"], 60)
        self.assertGreaterEqual(team_specific["road_step_down"], 60)


if __name__ == "__main__":
    unittest.main()
