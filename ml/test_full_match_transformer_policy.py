import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required")
class FullMatchTransformerPolicyTests(unittest.TestCase):
    def test_trained_transformer_drives_complete_sandbox_tactical_goals(self):
        script = r"""
const fs=require('fs');
const core=require('./docs/prediction-worker.js');
const bridge=require('./docs/full-match-transformer-policy.js');
const engine=require('./docs/full-match-engine.js');
const router=require('./docs/terrain-router.js');
const manifest=JSON.parse(fs.readFileSync('./docs/data/models/trajectory_transformer.json','utf8'));
const bytes=fs.readFileSync('./docs/data/models/trajectory_transformer.bin');
const floats=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
const tensors=new Map(manifest.tensors.map(item=>[
  item.name,floats.subarray(item.offset,item.offset+item.length),
]));
const model={
  manifest,tensors,
  mean:Float32Array.from(manifest.feature_mean),
  std:Float32Array.from(manifest.feature_std),
  targetX:manifest.feature_names.indexOf('target.x'),
  targetY:manifest.feature_names.indexOf('target.y'),
  targetVx3:manifest.feature_names.indexOf('target.vx_3_norm_per_s'),
  targetVy3:manifest.feature_names.indexOf('target.vy_3_norm_per_s'),
};
const simulation=JSON.parse(fs.readFileSync('./docs/data/models/full_simulation.json','utf8'));
const navigation=JSON.parse(fs.readFileSync('./docs/data/models/terrain_navigation.json','utf8'));
const trainedPolicy=bridge.createPolicy(model,core);
let componentCount=0;
const policy=(state,robot)=>{
  const result=trainedPolicy(state,robot);
  componentCount=Math.max(componentCount,Number(result?.components?.length||0));
  return result;
};
policy.record=trainedPolicy.record;
policy.metadata=trainedPolicy.metadata;
const state=engine.createMatch(
  simulation,navigation,'同济大学','东北大学',20260723,router,{transformerPolicy:policy},
);
let observed=0;
for(let second=0;second<45;second++){
  engine.stepMatch(state);
  observed+=state.robots.filter(robot=>robot.policySource==='state_heatmap_transformer').length;
}
process.stdout.write(JSON.stringify({policy:state.policy,observed,componentCount}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        output = json.loads(result.stdout)
        self.assertTrue(output["policy"]["active"])
        self.assertEqual(
            "temporal_battlefield_mixture_transformer",
            output["policy"]["modelKind"],
        )
        self.assertEqual(270_824, output["policy"]["parameterCount"])
        self.assertGreater(output["policy"]["decisions"], 20)
        self.assertGreater(output["observed"], 0)
        self.assertEqual(8, output["componentCount"])

    def test_stationary_prediction_is_rejected_but_neural_firing_anchors_are_preserved(self):
        script = r"""
const fs=require('fs');
const engine=require('./docs/full-match-engine.js');
const router=require('./docs/terrain-router.js');
const simulation=JSON.parse(fs.readFileSync('./docs/data/models/full_simulation.json','utf8'));
const navigation=JSON.parse(fs.readFileSync('./docs/data/models/terrain_navigation.json','utf8'));
const stationary=(state,robot)=>({target:[...robot.position],horizon:10});
stationary.metadata={modelKind:'test_transformer',parameterCount:1,horizon:10};
function bestLearnedAnchor(state,robot){
  const phase=Math.min(6,Math.floor(state.second/60));
  const points=robot.profile.goals_by_minute[phase];
  const maximum=Math.max(...points.map(point=>Number(point[2]||1)));
  return points.map(point=>({
    point:[Number(point[0]),Number(point[1])],
    score:engine.contextualHeroFiringScore(
      state,robot,[Number(point[0]),Number(point[1])],Number(point[2]||1)/maximum,
    ),
  })).sort((left,right)=>right.score-left.score)[0].point;
}
const state=engine.createMatch(simulation,navigation,'东北大学','同济大学',71,router,{transformerPolicy:stationary});
state.second=40;
const infantry=state.robots.find(robot=>robot.key==='red:步兵3');
infantry.position=[8,7.5];infantry.goal=[...infantry.position];infantry.route=[[...infantry.position]];
infantry.hp=infantry.maxHp;infantry.weak=false;infantry.ammo=infantry.profile.magazine;infantry.shotBudget=999;
infantry.lastMovedAt=0;infantry.lastFiredAt=-999;infantry.lastDamageAt=-999;
engine.chooseGoal(state,infantry);
const infantryResult={source:infantry.policySource,distance:router.distance(infantry.position,infantry.goal),status:infantry.status};
const hero=state.robots.find(robot=>robot.key==='blue:英雄');
hero.hp=hero.maxHp;hero.weak=false;hero.ammo=hero.profile.magazine;hero.shotBudget=999;
hero.lastMovedAt=0;hero.lastFiredAt=-999;hero.lastDamageAt=-999;
hero.position=engine.canonicalPoint(bestLearnedAnchor(state,hero),hero.side);
hero.goal=[...hero.position];hero.route=[[...hero.position]];
state.random=()=>.999;
engine.chooseGoal(state,hero);
const heroResult={source:hero.policySource,distance:router.distance(hero.position,hero.goal),status:hero.status};
const inferred=engine.createMatch(simulation,navigation,'五邑大学','东北大学',72,router,{transformerPolicy:stationary});
inferred.second=40;
const inferredHero=inferred.robots.find(robot=>robot.key==='red:英雄');
inferredHero.hp=inferredHero.maxHp;inferredHero.lastFiredAt=-999;inferredHero.lastDamageAt=-999;
inferredHero.position=engine.canonicalPoint(bestLearnedAnchor(inferred,inferredHero),inferredHero.side);
const inferredHold=engine.stateConditionedHeatmapGoal(
  inferred,inferredHero,[...inferredHero.position],[12,11.5],false,false,
);
const inferredMixtureHold=engine.stateConditionedHeatmapGoal(
  inferred,inferredHero,[12,11.5],[12,11.5],false,false,
  [{target:[12,11.5],weight:1,scale:[0.4,0.4]}],
);
inferredHero.lastDamageAt=inferred.second;
const inferredAfterHit=engine.stateConditionedHeatmapGoal(
  inferred,inferredHero,[12,11.5],[12,11.5],false,false,
  [{target:[12,11.5],weight:1,scale:[0.4,0.4]}],
);
process.stdout.write(JSON.stringify({
  infantryResult,heroResult,
  inferredResult:{
    manualLabel:inferredHero.profile.hero_archetype_evidence.engagement_style_label,
    modelKind:simulation.hero_firing_ranker.model_kind,
    source:inferredHold.sources[0],
    distance:router.distance(inferredHero.position,inferredHold.target),
    mixtureSource:inferredMixtureHold.sources[0],
    afterHitSources:inferredAfterHit.sources,
  },
}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual("rules", output["infantryResult"]["source"])
        self.assertGreater(output["infantryResult"]["distance"], 0.75)
        self.assertIn("脱离静止收敛", output["infantryResult"]["status"])
        self.assertEqual("state_heatmap_transformer", output["heroResult"]["source"])
        self.assertLess(output["heroResult"]["distance"], 0.01)
        self.assertIsNone(output["inferredResult"]["manualLabel"])
        self.assertEqual(
            "contextual_hero_firing_anchor_ranker",
            output["inferredResult"]["modelKind"],
        )
        self.assertEqual("learned_firing_hold", output["inferredResult"]["source"])
        self.assertLess(output["inferredResult"]["distance"], 0.01)
        self.assertEqual(
            "learned_firing_hold", output["inferredResult"]["mixtureSource"]
        )
        self.assertNotIn(
            "learned_firing_hold", output["inferredResult"]["afterHitSources"]
        )


if __name__ == "__main__":
    unittest.main()
