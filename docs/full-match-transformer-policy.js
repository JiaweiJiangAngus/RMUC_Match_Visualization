"use strict";

(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.RMUCFullMatchTransformerPolicy = api;
})(typeof globalThis !== "undefined" ? globalThis : self, function buildPolicyApi() {
  const ROLE_ORDER = ["英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"];
  const OFFSETS = [0, 1, 3, 5];
  const FIELD_WIDTH = 28;
  const FIELD_HEIGHT = 15;

  function sideLabel(side) {
    return side === "red" ? "红" : "蓝";
  }

  function compactFrame(state) {
    const rows = [];
    state.robots.forEach((robot) => {
      const alive = robot.hp > 0;
      const weapon = robot.profile.weapon;
      rows.push([
        robot.id,
        robot.role,
        sideLabel(robot.side),
        Math.max(0, Number(robot.hp || 0)),
        Number(robot.maxHp || 0),
        alive ? Number(robot.position[0]) : null,
        alive ? Number(robot.position[1]) : null,
        Number(robot.yaw || 0),
        weapon === "17mm" ? Number(robot.ammo || 0) : 0,
        weapon === "42mm" ? Number(robot.ammo || 0) : 0,
        Number(state.teamState[robot.side].coins || 0),
        Boolean(robot.weak),
      ]);
    });
    ["red", "blue"].forEach((side, sideIndex) => {
      [["基地", "base"], ["前哨站", "outpost"]].forEach(([label, key], structureIndex) => {
        const structure = state.structures[side][key];
        rows.push([
          1000 + sideIndex * 10 + structureIndex,
          label,
          sideLabel(side),
          Math.max(0, Number(structure.hp || 0)),
          Number(structure.maxHp || 0),
          Number(structure.position[0]),
          Number(structure.position[1]),
          0, 0, 0,
          Number(state.teamState[side].coins || 0),
          false,
        ]);
      });
    });
    return rows;
  }

  function createPolicy(model, predictionCore) {
    if (!model?.manifest || !predictionCore?.buildFeatures || !predictionCore?.forward) {
      throw new Error("Transformer 战术策略缺少模型或推理核心");
    }
    const frames = new Map();
    const horizonIndex = Math.max(0, model.manifest.horizons.indexOf(10));
    const horizon = Number(model.manifest.horizons[horizonIndex]);

    function record(state) {
      frames.set(Number(state.second), compactFrame(state));
      const oldest = Number(state.second) - Math.max(...OFFSETS);
      for (const second of frames.keys()) if (second < oldest) frames.delete(second);
    }

    function policy(state, robot) {
      if (robot.role === "空中" || robot.hp <= 0 || state.second < 5) return null;
      const history = {};
      for (const offset of OFFSETS) {
        const frame = frames.get(Number(state.second) - offset);
        if (!frame) return null;
        history[String(offset)] = frame;
      }
      const targetSide = sideLabel(robot.side);
      const features = predictionCore.buildFeatures(
        history, state.second, targetSide, robot.role, state.duration,
        { "红": state.schools.red, "蓝": state.schools.blue }, model.manifest,
      );
      if (!features) return null;
      const residuals = predictionCore.forward(model, features);
      let canonicalX = Number(features[model.targetX]) + Number(residuals[horizonIndex * 2]);
      let canonicalY = Number(features[model.targetY]) + Number(residuals[horizonIndex * 2 + 1]);
      canonicalX = Math.max(0.003, Math.min(0.997, canonicalX));
      canonicalY = Math.max(0.006, Math.min(0.994, canonicalY));
      let x = canonicalX * FIELD_WIDTH;
      let y = canonicalY * FIELD_HEIGHT;
      if (robot.side === "blue") {
        x = FIELD_WIDTH - x;
        y = FIELD_HEIGHT - y;
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      return {
        target: [x, y],
        horizon,
        modelKind: model.manifest.model_kind,
      };
    }

    policy.record = record;
    policy.metadata = {
      active: true,
      modelKind: model.manifest.model_kind,
      parameterCount: Number(model.manifest.training?.parameter_count || 0),
      horizon,
    };
    return policy;
  }

  return { compactFrame, createPolicy };
});
