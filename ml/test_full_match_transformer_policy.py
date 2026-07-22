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
const policy=bridge.createPolicy(model,core);
const state=engine.createMatch(
  simulation,navigation,'同济大学','东北大学',20260723,router,{transformerPolicy:policy},
);
let observed=0;
for(let second=0;second<45;second++){
  engine.stepMatch(state);
  observed+=state.robots.filter(robot=>robot.policySource==='transformer').length;
}
process.stdout.write(JSON.stringify({policy:state.policy,observed}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        output = json.loads(result.stdout)
        self.assertTrue(output["policy"]["active"])
        self.assertEqual("temporal_battlefield_transformer", output["policy"]["modelKind"])
        self.assertEqual(252_394, output["policy"]["parameterCount"])
        self.assertGreater(output["policy"]["decisions"], 20)
        self.assertGreater(output["observed"], 0)


if __name__ == "__main__":
    unittest.main()
