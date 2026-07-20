import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "docs" / "data" / "models" / "match_simulation.json"
FULL_MODEL_PATH = ROOT / "docs" / "data" / "models" / "full_simulation.json"


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


class FullSimulationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(FULL_MODEL_PATH.read_text(encoding="utf-8"))

    def test_every_team_has_six_robot_profiles(self):
        self.assertEqual(44, len(self.model["teams"]))
        expected = {"英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"}
        for team in self.model["teams"].values():
            self.assertEqual(expected, set(team["roles"]))
            for role in team["roles"].values():
                self.assertEqual(7, len(role["goals_by_minute"]))
                self.assertEqual(7, len(role["hp_by_minute"]))
                self.assertGreater(role["speed_mps"], 0)

    def test_rule_parameters_cover_agent_state_transitions(self):
        rules = self.model["rules"]
        self.assertEqual("V2.1.0", self.model["ruleset"]["version"])
        self.assertEqual(20, rules["damage"]["17mm"])
        self.assertEqual(200, rules["damage"]["42mm"])
        self.assertEqual(0.1, rules["heal_ratio_per_second"])
        self.assertEqual(0.25, rules["late_heal_ratio_per_second"])
        self.assertEqual(30, rules["respawn"]["timed_invulnerable_seconds"])
        self.assertEqual(5, rules["radar_uav_counter"]["max_uses"])
        self.assertEqual(45, rules["radar_uav_counter"]["lock_seconds"])
        self.assertEqual(4, rules["radar_uav_counter"]["buyout_from_use"])
        self.assertEqual(2, rules["radar_uav_counter"]["buyout_cost_multiplier"])
        self.assertEqual(30, rules["uav_support"]["initial_seconds"])
        self.assertEqual(20, rules["uav_support"]["periodic_seconds"])
        self.assertFalse(rules["uav_support"]["ordinary_damage"])
        self.assertFalse(rules["uav_support"]["healing_and_respawn"])
        self.assertEqual(180, rules["engineer_assembly_invulnerability_seconds"])

    def test_v210_hero_archetype_health_tables_are_exact(self):
        heroes = self.model["rules"]["hero_archetypes"]
        self.assertEqual(
            [260, 300, 330, 360, 400, 430, 460, 500, 530, 600],
            heroes["melee"]["hp_by_level"],
        )
        self.assertEqual(
            [200, 220, 240, 260, 280, 300, 320, 340, 360, 400],
            heroes["ranged"]["hp_by_level"],
        )
        for team in self.model["teams"].values():
            hero = team["roles"]["英雄"]
            self.assertEqual("ranged", hero["hero_archetype_default"])
            self.assertEqual(7, len(hero["level_by_minute"]))
            self.assertGreaterEqual(team["radar_counters_per_game"], 0)

    def test_service_zones_separate_ammo_and_healing(self):
        for side in ("red", "blue"):
            zones = self.model["service_zones"][side]
            self.assertEqual({"supply", "base", "outpost"}, set(zones))
            self.assertTrue(zones["supply"]["ammo"])
            self.assertTrue(zones["supply"]["heal"])
            for name in ("base", "outpost"):
                self.assertTrue(zones[name]["ammo"])
                self.assertFalse(zones[name]["heal"])

    def test_national_economy_and_technology_core_rules_are_exact(self):
        rules = self.model["rules"]
        self.assertEqual(400, rules["initial_coins"])
        self.assertEqual(
            [[61, 50], [121, 50], [181, 50], [241, 50], [301, 50], [361, 150]],
            rules["automatic_income"],
        )
        core = rules["technology_core"]
        self.assertEqual([0, 60, 120, 180], core["unlock_seconds"])
        self.assertEqual([50, 25, 25, 50], core["first_income_per_10"])
        self.assertEqual([5, 10, 15, 0], core["repeat_income_per_10"])
        self.assertEqual([5, 7, 10, 10], core["robot_level_cap_by_level"])
        self.assertEqual([0, 0, 0.25, 0.5], core["defense_ratio_by_level"])
        self.assertEqual(2000, core["level_four_base_hp_gain"])

    def test_all_teams_have_regional_core_timing_priors(self):
        for team in self.model["teams"].values():
            prior = team["economy_prior"]
            self.assertGreater(prior["games"], 0)
            self.assertEqual(4, len(prior["core_reach_rate"]))
            self.assertEqual(4, len(prior["core_completion_seconds"]))
            reach = prior["core_reach_rate"]
            self.assertTrue(all(reach[index] >= reach[index + 1] for index in range(3)))
            self.assertEqual(8, len(prior["regional_total_coins_by_minute"]))

    def test_uav_profiles_separate_helipad_and_airborne_transitions(self):
        for team in self.model["teams"].values():
            role = team["roles"]["空中"]
            navigation = role["uav_navigation"]
            self.assertEqual(navigation["home"], role["spawn"])
            self.assertLess(navigation["home"][0], 3.2)
            self.assertGreater(navigation["home"][1], 11)
            self.assertEqual(7, len(navigation["airborne_goals_by_minute"]))
            self.assertEqual(7, len(navigation["transitions_by_minute"]))
            self.assertGreater(navigation["samples"], 0)
            for goals in navigation["airborne_goals_by_minute"]:
                self.assertTrue(goals)
                self.assertTrue(all(not (point[0] < 3.2 and point[1] > 11) for point in goals))


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


@unittest.skipUnless(shutil.which("node"), "Node.js is required for full simulator tests")
class FullSimulationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(FULL_MODEL_PATH.read_text(encoding="utf-8"))
        script = r"""
const fs=require('fs');
const router=require('./docs/terrain-router.js');
const engine=require('./docs/full-match-engine.js');
const model=JSON.parse(fs.readFileSync('./docs/data/models/full_simulation.json','utf8'));
const nav=JSON.parse(fs.readFileSync('./docs/data/models/terrain_navigation.json','utf8'));
function run() {
  const schools={red:'东北大学',blue:'中国石油大学（华东）'};
  const result=engine.runMatch(model,nav,schools.red,schools.blue,20260719,router);
  const final=result.frames.at(-1);
  const tunnelGates=nav.gates.filter(gate=>gate.category.endsWith('tunnel'));
  return {
    frames:result.frames.length,
    robots:final.robots.length,
    final,
    eventTypes:[...new Set(result.events.map(item=>item.type))].sort(),
    positionsValid:result.frames.every(frame=>frame.robots.every(robot=>robot.x>=0&&robot.x<=28&&robot.y>=0&&robot.y<=15)),
    stateValid:result.frames.every(frame=>frame.robots.every(robot=>robot.hp>=0&&robot.hp<=robot.maxHp&&robot.ammo>=0&&robot.heat>=0)),
    uavValid:result.frames.every(frame=>frame.robots.filter(robot=>robot.role==='空中').every(robot=>robot.deaths===0&&!robot.serviceZone&&['parked','airborne','returning'].includes(robot.uavFlightState)&&!/补给|复活|战亡/.test(robot.status))),
    uavMaxStep:Math.max(...['red','blue'].flatMap(side=>result.frames.slice(1).map((frame,index)=>{
      const one=result.frames[index].robots.find(robot=>robot.key===side+':空中');
      const two=frame.robots.find(robot=>robot.key===side+':空中');
      return Math.hypot(two.x-one.x,two.y-one.y);
    }))),
    neuInfantryTunnelClear:result.frames.slice(1).every((frame,index)=>frame.robots
      .filter(robot=>robot.side==='red'&&['步兵3','步兵4'].includes(robot.role))
      .every(robot=>{
        const previous=result.frames[index].robots.find(item=>item.key===robot.key);
        return !tunnelGates.some(gate=>router.segmentHitsPolygon([previous.x,previous.y],[robot.x,robot.y],gate.polygon));
      })),
    groundForbiddenGateClear:result.frames.slice(1).every((frame,index)=>frame.robots
      .filter(robot=>robot.role!=='空中')
      .every(robot=>{
        const previous=result.frames[index].robots.find(item=>item.key===robot.key);
        const abilities=nav.teams[schools[robot.side]][robot.role].abilities;
        const blockers=nav.gates.filter(gate=>['road_tunnel','highland_tunnel','rough_road'].includes(gate.category)&&!abilities.includes(gate.category));
        return !blockers.some(gate=>router.segmentHitsPolygon([previous.x,previous.y],[robot.x,robot.y],gate.polygon));
      })),
    groundStaticWallClear:result.frames.slice(1).every((frame,index)=>frame.robots
      .filter(robot=>robot.role!=='空中')
      .every(robot=>{
        const previous=result.frames[index].robots.find(item=>item.key===robot.key);
        return !nav.static_obstacles.some(wall=>wall.blocks_movement!==false
          &&router.segmentHitsPolygon([previous.x,previous.y],[robot.x,robot.y],wall.polygon));
      })),
    terrainMotionFrames:result.frames.reduce((sum,frame)=>sum+frame.robots.filter(robot=>robot.role!=='空中'&&robot.terrainSpeedMultiplier!==1).length,0),
    terrainDirections:[...new Set(result.frames.flatMap(frame=>frame.robots.map(robot=>robot.terrainAction).filter(Boolean)))],
    signature:JSON.stringify({final,events:result.events})
  };
}
function probeZone(name) {
  const state=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',20260719,router);
  const robot=state.robots.find(item=>item.key==='red:英雄');
  robot.position=[...model.service_zones.red[name].center];
  robot.hp=100;
  robot.ammo=0;
  robot.shots=0;
  robot.weak=true;
  robot.weakKind='timed';
  robot.respawnedAt=0;
  state.teamState.red.coins=1000;
  engine.resupplyRobots(state);
  return {hp:robot.hp,ammo:robot.ammo,weak:robot.weak};
}
function probeV210() {
  const heroState=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',1,router,{heroArchetypes:{red:'melee',blue:'ranged'}});
  const melee=heroState.robots.find(item=>item.key==='red:英雄');
  const ranged=heroState.robots.find(item=>item.key==='blue:英雄');

  const timed=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',2,router,{buybackPolicy:'never'});
  timed.second=300;
  timed.teamState.red.coins=0;
  const timedRobot=timed.robots.find(item=>item.key==='red:步兵3');
  const timedAttacker=timed.robots.find(item=>item.key==='blue:英雄');
  timedRobot.position=[12.3,7.2];
  timedRobot.buybacks=2;
  engine.killRobot(timed,timedRobot,timedAttacker);
  const readRequired=timedRobot.respawnRequired;
  for(let index=0;index<readRequired;index+=1){timed.second+=1;engine.respawnRobots(timed);}
  const timedResult={position:timedRobot.position,hp:timedRobot.hp,maxHp:timedRobot.maxHp,weak:timedRobot.weak,mode:timedRobot.respawnMode};

  const instant=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',3,router,{buybackPolicy:'always'});
  instant.second=120;
  instant.teamState.red.coins=2000;
  const instantRobot=instant.robots.find(item=>item.key==='red:英雄');
  const instantAttacker=instant.robots.find(item=>item.key==='blue:英雄');
  instantRobot.position=[9.1,4.2];
  const instantCost=engine.immediateReviveCost(instant,instantRobot);
  engine.killRobot(instant,instantRobot,instantAttacker);
  const instantAtDeath={position:instantRobot.position,hp:instantRobot.hp,maxHp:instantRobot.maxHp,buybacks:instantRobot.buybacks,weak:instantRobot.weak,coins:instant.teamState.red.coins};
  for(let index=0;index<3;index+=1){instant.second+=1;engine.respawnRobots(instant);}

  const radar=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',4,router,{radarBuyoutPolicy:'never'});
  radar.second=240;
  radar.teamState.blue.coins=3000;
  const uav=radar.robots.find(item=>item.key==='blue:空中');
  uav.uavFlightState='airborne';
  uav.uavSupportActive=true;
  const baseCounterCost=engine.immediateReviveCost(radar,uav);
  engine.applyRadarCounter(radar,'red',false);
  engine.applyRadarCounter(radar,'red',false);
  engine.applyRadarCounter(radar,'red',false);
  const fourth=engine.applyRadarCounter(radar,'red',true);
  engine.applyRadarCounter(radar,'red',false);
  const sixth=engine.applyRadarCounter(radar,'red',false);
  return {
    heroHp:{melee:melee.maxHp,ranged:ranged.maxHp},
    readRequired,timedResult,
    instantCost,instantAtDeath,instantWeakAfter3:instantRobot.weak,
    radar:{count:uav.radarCounterCount,buyouts:uav.radarCounterBuyouts,fourthCost:fourth.buyoutCost,baseCounterCost,sixth},
  };
}
function probeUavRules() {
  const state=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',6,router);
  const uav=state.robots.find(item=>item.key==='red:空中');
  const initial={position:[...uav.position],home:[...uav.profile.uav_navigation.home],flight:uav.uavFlightState,support:uav.uavSupportSeconds,ammo:uav.ammo};
  uav.position=[...model.service_zones.red.supply.center];
  uav.hp=1;
  uav.ammo=123;
  engine.resupplyRobots(state);
  engine.killRobot(state,uav,state.robots.find(item=>item.key==='blue:英雄'));
  const excluded={hp:uav.hp,ammo:uav.ammo,deaths:uav.deaths,respawnMode:uav.respawnMode};
  uav.uavFlightState='airborne';
  uav.uavSupportActive=true;
  uav.uavSupportSeconds=1;
  state.teamState.red.coins=7;
  state.second=10;
  engine.updateUavSupport(state);
  const free={support:uav.uavSupportSeconds,coins:state.teamState.red.coins};
  state.second=11;
  engine.updateUavSupport(state);
  const paid={support:uav.uavSupportSeconds,coins:state.teamState.red.coins,paidSeconds:uav.uavPaidSupportSeconds};
  return {initial,excluded,free,paid};
}
function probeTechnologyCore() {
  const state=engine.createMatch(model,nav,'华南农业大学','中国石油大学（华东）',5,router);
  const initial={red:state.teamState.red.coins,blue:state.teamState.blue.coins};
  const initialAmmo=Object.fromEntries(state.robots.filter(item=>item.side==='red').map(item=>[item.role,item.ammo]));
  state.teamState.red.technologyCore.plan=[{level:1,plannedSecond:1,completedSecond:null}];
  const engineer=state.robots.find(item=>item.key==='red:工程');
  engineer.position=[...model.assembly_zones.red.center];
  state.second=1;
  engine.updateTechnologyCores(state);
  const completed=engine.snapshot(state);
  while(state.second<11) engine.stepMatch(state);
  const paid=engine.snapshot(state);
  return {initial,initialAmmo,completed:completed.teams.red,engineer:completed.robots.find(item=>item.key==='red:工程'),paid:paid.teams.red};
}
function probeHardRules() {
  const state=engine.createMatch(model,nav,'东北大学','中国石油大学（华东）',8,router);
  const infantry=state.robots.find(item=>item.key==='red:步兵3');
  infantry.position=[...model.service_zones.red.outpost.center];
  infantry.ammo=0; infantry.shots=0; infantry.shotBudget=100;
  infantry.weak=true; infantry.weakKind='timed';
  state.structures.red.outpost.hp=0;
  engine.resupplyRobots(state);
  const destroyedOutpost={ammo:infantry.ammo,weak:infantry.weak,zone:engine.serviceZoneAt(state,infantry,'ammo')};

  const engineer=state.robots.find(item=>item.key==='red:工程');
  engineer.position=[...model.assembly_zones.red.center];
  engineer.hp=1;
  engineer.assemblyInvulnerableSeconds=0;
  engine.chooseGoal(state,engineer);
  const protectedDecision={mode:engineer.mode,serviceTarget:engineer.serviceTarget};
  engineer.position=[...model.assembly_zones.red.center];
  engineer.assemblyInvulnerableSeconds=180;
  engine.chooseGoal(state,engineer);
  const exhaustedDecision={mode:engineer.mode,serviceTarget:engineer.serviceTarget};
  engineer.position=[...model.assembly_zones.red.center];
  engineer.assemblyInvulnerableSeconds=0;
  let protectedSeconds=0;
  for(let second=0;second<181;second+=1){engine.updateAssemblyProtection(state);if(engineer.assemblyProtected)protectedSeconds+=1;}

  const hero=state.robots.find(item=>item.key==='red:英雄');
  const uav=state.robots.find(item=>item.key==='red:空中');
  hero.position=[9,7.5]; uav.position=[9,7.5];
  return {
    destroyedOutpost,
    assembly:{protectedSeconds,used:engineer.assemblyInvulnerableSeconds,protectedAfterLimit:engineer.assemblyProtected,protectedDecision,exhaustedDecision},
    lineOfSight:{ground:engine.lineOfSight(state,hero,[19,7.5]),uav:engine.lineOfSight(state,uav,[19,7.5])},
  };
}
const first=run();
const repeat=run();
const zones={base:probeZone('base'),outpost:probeZone('outpost'),supply:probeZone('supply')};
console.log(JSON.stringify({first:{...first,signature:undefined},deterministic:first.signature===repeat.signature,zones,v210:probeV210(),uavRules:probeUavRules(),technologyCore:probeTechnologyCore(),hardRules:probeHardRules()}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        payload = json.loads(result.stdout)
        cls.result = payload["first"]
        cls.deterministic = payload["deterministic"]
        cls.zones = payload["zones"]
        cls.v210 = payload["v210"]
        cls.uav_rules = payload["uavRules"]
        cls.technology_core = payload["technologyCore"]
        cls.hard_rules = payload["hardRules"]

    def test_complete_match_has_421_frames_and_12_agents(self):
        self.assertEqual(421, self.result["frames"])
        self.assertEqual(12, self.result["robots"])
        self.assertTrue(self.result["final"]["finished"])
        self.assertIn(self.result["final"]["winner"], {"red", "blue", "draw"})

    def test_positions_health_ammo_and_heat_stay_valid(self):
        self.assertTrue(self.result["positionsValid"])
        self.assertTrue(self.result["stateValid"])
        for side in ("red", "blue"):
            structures = self.result["final"]["structures"][side]
            self.assertGreaterEqual(structures["base"], 0)
            self.assertLessEqual(structures["base"], 5000)
            self.assertGreaterEqual(structures["outpost"], 0)
            self.assertLessEqual(structures["outpost"], 1500)
        self.assertTrue(self.result["uavValid"])
        fastest_uav = max(
            self.model["teams"]["东北大学"]["roles"]["空中"]["speed_mps"],
            self.model["teams"]["中国石油大学（华东）"]["roles"]["空中"]["speed_mps"],
        )
        self.assertLessEqual(self.result["uavMaxStep"], fastest_uav + 0.01)

    def test_neu_infantry_never_interpolates_through_tunnel_and_terrain_changes_speed(self):
        self.assertTrue(self.result["neuInfantryTunnelClear"])
        self.assertTrue(self.result["groundForbiddenGateClear"])
        self.assertTrue(self.result["groundStaticWallClear"])
        self.assertGreater(self.result["terrainMotionFrames"], 0)
        actions = self.result["terrainDirections"]
        self.assertTrue(any("上公路台阶" in action for action in actions))
        self.assertTrue(any("下公路台阶" in action for action in actions))

    def test_simulation_contains_combat_supply_and_terrain_actions(self):
        event_types = set(self.result["eventTypes"])
        self.assertTrue({"hit", "supply", "terrain", "economy", "dart"}.issubset(event_types))

    def test_fixed_seed_is_deterministic(self):
        self.assertTrue(self.deterministic)

    def test_only_supply_zone_heals_but_all_service_points_clear_timed_weakness(self):
        for name in ("base", "outpost"):
            self.assertEqual(100, self.zones[name]["hp"])
            self.assertGreater(self.zones[name]["ammo"], 0)
            self.assertFalse(self.zones[name]["weak"])
        self.assertGreater(self.zones["supply"]["hp"], 100)
        self.assertGreater(self.zones["supply"]["ammo"], 0)
        self.assertFalse(self.zones["supply"]["weak"])

    def test_v210_hero_modes_change_level_one_health(self):
        self.assertEqual({"melee": 260, "ranged": 200}, self.v210["heroHp"])

    def test_v210_timed_and_immediate_respawn_choices(self):
        self.assertEqual(80, self.v210["readRequired"])
        timed = self.v210["timedResult"]
        self.assertEqual([12.3, 7.2], timed["position"])
        self.assertAlmostEqual(timed["maxHp"] * 0.1, timed["hp"])
        self.assertTrue(timed["weak"])
        self.assertEqual("timed", timed["mode"])
        instant = self.v210["instantAtDeath"]
        self.assertEqual([9.1, 4.2], instant["position"])
        self.assertEqual(instant["maxHp"], instant["hp"])
        self.assertEqual(1, instant["buybacks"])
        self.assertEqual(2000 - self.v210["instantCost"], instant["coins"])
        self.assertFalse(self.v210["instantWeakAfter3"])

    def test_v210_radar_counter_caps_at_five_and_fourth_can_buy_out(self):
        radar = self.v210["radar"]
        self.assertEqual(5, radar["count"])
        self.assertEqual(1, radar["buyouts"])
        self.assertEqual(radar["baseCounterCost"] * 2, radar["fourthCost"])
        self.assertIsNone(radar["sixth"])

    def test_v210_uav_uses_flight_support_instead_of_ground_lifecycle(self):
        rules = self.uav_rules
        self.assertEqual("parked", rules["initial"]["flight"])
        self.assertEqual(rules["initial"]["home"], rules["initial"]["position"])
        self.assertEqual(30, rules["initial"]["support"])
        self.assertEqual(750, rules["initial"]["ammo"])
        self.assertEqual({"hp": 1, "ammo": 123, "deaths": 0, "respawnMode": None}, rules["excluded"])
        self.assertEqual({"support": 0, "coins": 7}, rules["free"])
        self.assertEqual({"support": 0, "coins": 6, "paidSeconds": 1}, rules["paid"])

    def test_technology_core_starts_from_equal_400_and_pays_every_ten_seconds(self):
        core = self.technology_core
        self.assertEqual({"red": 400, "blue": 400}, core["initial"])
        self.assertEqual(300, core["initialAmmo"]["哨兵"])
        self.assertEqual(750, core["initialAmmo"]["空中"])
        self.assertEqual(1, core["completed"]["technologyCoreLevel"])
        self.assertEqual(50, core["completed"]["technologyCoreIncomePer10"])
        self.assertEqual(50, core["completed"]["technologyCoreEarnedCoins"])
        self.assertEqual(1, core["engineer"]["technologyCoreLevel"])
        self.assertEqual(100, core["paid"]["technologyCoreEarnedCoins"])

    def test_destroyed_outpost_disables_its_ammunition_and_interaction_zone(self):
        result = self.hard_rules["destroyedOutpost"]
        self.assertEqual(0, result["ammo"])
        self.assertTrue(result["weak"])
        self.assertIsNone(result["zone"])

    def test_engineer_gets_exactly_180_cumulative_assembly_protection_seconds(self):
        result = self.hard_rules["assembly"]
        self.assertEqual(180, result["protectedSeconds"])
        self.assertEqual(180, result["used"])
        self.assertFalse(result["protectedAfterLimit"])
        self.assertEqual(
            {"mode": "assembly_hold", "serviceTarget": None},
            result["protectedDecision"],
        )
        self.assertEqual(
            {"mode": "heal", "serviceTarget": "supply"},
            result["exhaustedDecision"],
        )

    def test_only_uav_can_ignore_central_highland_height_layer_for_fire(self):
        self.assertEqual({"ground": False, "uav": True}, self.hard_rules["lineOfSight"])


if __name__ == "__main__":
    unittest.main()
