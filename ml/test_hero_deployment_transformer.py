import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "ml" / "artifacts" / "hero_deployment_transformer.json"
FULL_MODEL = ROOT / "docs" / "data" / "models" / "full_simulation.json"


class HeroDeploymentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_real_supervision_and_held_out_gates_are_published(self):
        model = self.artifact
        self.assertEqual("hero_deployment_transformer", model["model_kind"])
        self.assertEqual(2, model["schema_version"])
        self.assertTrue(all(model["acceptance"].values()))
        self.assertIn("stationary_pseudo_labels", model["supervision"]["excluded"])
        self.assertIn(
            "all_samples_within_60_seconds_after_large_energy_mechanism_activation",
            model["supervision"]["excluded"],
        )
        self.assertGreater(model["metrics"]["test"]["roc_auc"], 0.9)
        self.assertGreater(model["exit_metrics"]["test"]["roc_auc"], 0.65)
        self.assertGreater(model["exit_metrics"]["test"]["exit_samples"], 0)
        self.assertGreater(model["exit_metrics"]["test"]["hold_samples"], 0)
        self.assertEqual(2, model["runtime_rules"]["exit_delay_seconds"])
        self.assertEqual(200, model["runtime_rules"]["ordinary_42mm_damage"])
        self.assertEqual(
            300,
            model["runtime_rules"]["deployed_base_42mm_damage"],
        )


@unittest.skipUnless(shutil.which("node"), "Node.js is required")
class HeroDeploymentBrowserTests(unittest.TestCase):
    def run_node(self, script):
        result = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_browser_transformer_matches_exported_pytorch_probe(self):
        output = self.run_node(
            r"""
const fs=require('fs');
const engine=require('./docs/full-match-engine.js');
const model=JSON.parse(fs.readFileSync(
  './ml/artifacts/hero_deployment_transformer.json','utf8',
));
const probe=model.inference_probe;
const result=engine.heroDeploymentTransformerForward(
  model,probe.standardized_sequence,probe.team_index,probe.opponent_index,
);
process.stdout.write(JSON.stringify({result,probe}));
"""
        )
        self.assertAlmostEqual(
            output["probe"]["deployed_probability"],
            output["result"]["deployedProbability"],
            places=6,
        )
        self.assertAlmostEqual(
            output["probe"]["exit_probability"],
            output["result"]["exitProbability"],
            places=6,
        )

    def test_deployment_locks_motion_and_exit_unlocks_after_two_seconds(self):
        output = self.run_node(
            r"""
const fs=require('fs');
const engine=require('./docs/full-match-engine.js');
const router=require('./docs/terrain-router.js');
const model=JSON.parse(fs.readFileSync(
  './docs/data/models/full_simulation.json','utf8',
));
const navigation=JSON.parse(fs.readFileSync(
  './docs/data/models/terrain_navigation.json','utf8',
));
const state=engine.createMatch(
  model,navigation,'同济大学','东北大学',20260729,router,
  {heroArchetypes:{red:'ranged',blue:'melee'}},
);
const hero=state.robots.find(robot=>robot.key==='red:英雄');
hero.hp=hero.maxHp;
hero.ammo=hero.profile.magazine;
hero.shotBudget=999;
const bounds=model.rules.hero_deployment.canonical_deployment_zone_m;
const point=[
  (bounds.min[0]+bounds.max[0])/2,
  (bounds.min[1]+bounds.max[1])/2,
];
model.hero_deployment_model.thresholds.enter=0;
model.hero_deployment_model.thresholds.exit=2;
for(let second=1;second<=8;second++){
  state.second=second;
  hero.position=[...point];
  engine.updateHeroDeploymentStates(state);
}
const entered=hero.deploymentState;
const damage={
  ordinary:engine.damagePerHit(
    state,{...hero,deploymentState:'mobile'},
    state.structures.blue.base,'42mm',
  ),
  deployed:engine.damagePerHit(
    state,{...hero,deploymentState:'deployed'},
    state.structures.blue.base,'42mm',
  ),
};
hero.goal=[20,7.5];
hero.route=[[...hero.position],[20,7.5]];
const before=[...hero.position];
engine.moveRobots(state);
const lockedDistance=router.distance(before,hero.position);
state.random=()=>0;
hero.lastDamageAt=8;
state.second=9;
engine.updateHeroDeploymentStates(state);
engine.moveRobots(state);
const atNine={state:hero.deploymentState,position:[...hero.position]};
state.second=10;
engine.updateHeroDeploymentStates(state);
engine.moveRobots(state);
const atTen={state:hero.deploymentState,position:[...hero.position]};
state.second=11;
engine.updateHeroDeploymentStates(state);
const atEleven={state:hero.deploymentState};
process.stdout.write(JSON.stringify({
  entered,damage,lockedDistance,atNine,atTen,atEleven,before,
}));
"""
        )
        self.assertEqual("deployed", output["entered"])
        self.assertEqual({"ordinary": 200, "deployed": 300}, output["damage"])
        self.assertAlmostEqual(0, output["lockedDistance"], places=9)
        self.assertEqual("undeploying", output["atNine"]["state"])
        self.assertEqual("undeploying", output["atTen"]["state"])
        self.assertEqual("mobile", output["atEleven"]["state"])
        self.assertEqual(output["before"], output["atNine"]["position"])
        self.assertEqual(output["before"], output["atTen"]["position"])


if __name__ == "__main__":
    unittest.main()
