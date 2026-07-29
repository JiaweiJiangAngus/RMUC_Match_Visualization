import json
import math
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required")
class HeroFiringRankerBrowserTests(unittest.TestCase):
    def test_browser_forward_matches_exported_network_and_ignores_style_labels(self):
        script = r"""
const fs=require('fs');
const engine=require('./docs/full-match-engine.js');
const router=require('./docs/terrain-router.js');
const simulation=JSON.parse(fs.readFileSync('./docs/data/models/full_simulation.json','utf8'));
const navigation=JSON.parse(fs.readFileSync('./docs/data/models/terrain_navigation.json','utf8'));
const state=engine.createMatch(
  simulation,navigation,'同济大学','东北大学',20260728,router,
);
state.second=173;
const hero=state.robots.find(robot=>robot.key==='red:英雄');
hero.hp=hero.maxHp*.73;
hero.recentDamage=[[171,320],[172,180]];
state.structures.red.outpost.hp=700;
state.structures.blue.outpost.hp=260;
state.teamState.red.coins=860;
const phase=Math.min(6,Math.floor(state.second/60));
const points=hero.profile.goals_by_minute[phase];
const maximum=Math.max(...points.map(point=>Number(point[2]||1)));
const candidate=[Number(points[0][0]),Number(points[0][1])];
const prior=Number(points[0][2]||1)/maximum;
const features=engine.heroFiringRankerFeatureMap(state,hero,candidate,prior);
const score=engine.contextualHeroFiringScore(state,hero,candidate,prior);
const tacticalBefore=engine.tacticalCandidateScore(state,hero,candidate,prior);
hero.profile.engagement_profile={
  style:'deliberately_invalid_style',
  preferred_range_m:0.01,
};
const styleScore=engine.contextualHeroFiringScore(state,hero,candidate,prior);
const tacticalAfter=engine.tacticalCandidateScore(state,hero,candidate,prior);
hero.hp=hero.maxHp*.22;
hero.recentDamage=[[171,900],[172,800]];
state.structures.red.base.hp=500;
state.structures.blue.outpost.hp=0;
const changedScore=engine.contextualHeroFiringScore(state,hero,candidate,prior);
process.stdout.write(JSON.stringify({
  features,score,styleScore,tacticalBefore,tacticalAfter,changedScore,
  school:hero.school,opponent:state.schools.blue,prior,
}));
"""
        result = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        output = json.loads(result.stdout)
        model = json.loads(
            (ROOT / "ml/artifacts/hero_firing_ranker.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertTrue(all(model["acceptance"].values()))
        self.assertGreater(
            model["metrics"]["test"]["top1_accuracy"],
            model["metrics"]["test"]["training_heatmap_top1_baseline"],
        )
        values = [
            (
                float(output["features"].get(name, 0))
                - float(model["feature_mean"][index])
            )
            / max(1e-4, float(model["feature_std"][index]))
            for index, name in enumerate(model["feature_names"])
        ]
        values.extend(
            1.0 if school == output["school"] else 0.0
            for school in model["teams"]
        )
        values.extend(
            1.0 if school == output["opponent"] else 0.0
            for school in model["opponents"]
        )
        for layer in model["layers"][:-1]:
            values = [
                max(
                    0.0,
                    float(bias) + sum(
                        float(weight) * value
                        for weight, value in zip(weights, values)
                    ),
                )
                for weights, bias in zip(layer["weight"], layer["bias"])
            ]
        head = model["layers"][-1]
        residual = float(head["bias"][0]) + sum(
            float(weight) * value
            for weight, value in zip(head["weight"][0], values)
        )
        expected = (
            math.log(
                float(output["prior"])
                + float(model["base_score"]["epsilon"])
            )
            + residual
        )
        self.assertAlmostEqual(expected, output["score"], places=6)
        self.assertAlmostEqual(output["score"], output["styleScore"], places=12)
        self.assertAlmostEqual(
            output["tacticalBefore"], output["tacticalAfter"], places=12,
        )
        self.assertGreater(
            abs(output["score"] - output["changedScore"]),
            1e-4,
            "ranker score must respond to live HP/damage/structure state",
        )


if __name__ == "__main__":
    unittest.main()
