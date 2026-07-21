(function (root, factory) {
  "use strict";
  const api = factory(root.RMUCTerrainRouter);
  if (typeof module === "object" && module.exports) module.exports = api;
  root.RMUCFullMatchEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function (browserRouter) {
  "use strict";

  const SIDES = ["red", "blue"];
  const ROLE_ORDER = ["英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"];
  const ROLE_ID = { 英雄: 1, 工程: 2, 步兵3: 3, 步兵4: 4, 哨兵: 7, 空中: 6 };
  const ROLE_ACCURACY = { 英雄: 1, 工程: 0, 步兵3: 1.24, 步兵4: 1.18, 哨兵: 0.68, 空中: 0.9 };
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function hashSeed(value) {
    const text = String(value);
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function mulberry32(seed) {
    let state = seed >>> 0;
    return function random() {
      state += 0x6d2b79f5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function canonicalPoint(point, side) {
    return side === "red" ? [...point] : [28 - point[0], 15 - point[1]];
  }

  function otherSide(side) {
    return side === "red" ? "blue" : "red";
  }

  function weightedPoint(points, random) {
    const total = points.reduce((sum, point) => sum + Number(point[2] || 0), 0) || points.length;
    let draw = random() * total;
    for (const point of points) {
      draw -= Number(point[2] || 1);
      if (draw <= 0) return [Number(point[0]), Number(point[1])];
    }
    return [Number(points[0][0]), Number(points[0][1])];
  }

  function weightedItem(items, weightIndex, random) {
    if (!items?.length) return null;
    const total = items.reduce((sum, item) => sum + Number(item[weightIndex] || 0), 0) || items.length;
    let draw = random() * total;
    for (const item of items) {
      draw -= Number(item[weightIndex] || 1);
      if (draw <= 0) return item;
    }
    return items[0];
  }

  function sampleMatchAccuracy(teamProfile, weapon, random) {
    const learned = teamProfile.accuracy_models?.[weapon];
    const mean = Number(learned?.mean_probability ?? teamProfile.accuracy?.[weapon] ?? 0.1);
    const range = learned?.match_multiplier_range || [0.85, 1.15];
    const multiplier = Number(range[0]) + random() * (Number(range[1]) - Number(range[0]));
    return clamp(mean * multiplier, 0.018, 0.9);
  }

  function damagePerHit(state, robot, target, weapon) {
    const fallback = Number(state.model.rules.damage[weapon] || 0);
    if (robot.role !== "英雄" || weapon !== "42mm") return fallback;
    const targetType = target.kind || "robot";
    return Number(robot.profile.damage_per_hit_by_target?.[targetType]?.mode_damage || fallback);
  }

  function event(state, side, type, text, data) {
    state.events.push({ second: state.second, side, type, text, ...(data || {}) });
  }

  function robotLevel(profile, second) {
    const phase = Math.min(6, Math.floor(second / 60));
    if (profile.level_by_minute?.length) {
      return clamp(Number(profile.level_by_minute[phase] || profile.level_by_minute.at(-1) || 1), 1, 10);
    }
    return clamp(1 + phase, 1, 10);
  }

  function roleMaxHp(profile, second, heroArchetype, rules, levelOverride) {
    const level = Number(levelOverride || robotLevel(profile, second));
    const heroProfile = rules?.hero_archetypes?.[heroArchetype];
    if (heroProfile?.hp_by_level?.length) {
      return Number(heroProfile.hp_by_level[level - 1] || heroProfile.hp_by_level.at(-1));
    }
    const phase = Math.min(6, Math.floor(second / 60));
    return Number(profile.hp_by_minute[phase] || profile.hp_by_minute[profile.hp_by_minute.length - 1] || 200);
  }

  function reviveReadRequired(state, robot) {
    const rules = state.model.rules.respawn;
    return Math.round(
      Number(rules.read_base) + state.second / Number(rules.elapsed_seconds_divisor)
      + Number(rules.buyback_read_penalty) * robot.buybacks
    );
  }

  function immediateReviveCost(state, robot) {
    const rules = state.model.rules.respawn;
    const elapsedMinutes = Math.ceil(state.second / 60);
    const level = robotLevel(robot.profile, state.second);
    return elapsedMinutes * Number(rules.buyback_minute_cost) + level * Number(rules.buyback_level_cost);
  }

  function insideZone(position, zone) {
    if (!zone) return false;
    const radius = zone.radius || [1, 1];
    const dx = (position[0] - zone.center[0]) / Math.max(0.01, radius[0]);
    const dy = (position[1] - zone.center[1]) / Math.max(0.01, radius[1]);
    return dx * dx + dy * dy <= 1;
  }

  function serviceZoneAt(state, robot, capability) {
    const zones = state.model.service_zones?.[robot.side] || {};
    return Object.entries(zones).find(([name, zone]) => (
      zone[capability]
      && (name !== "outpost" || state.structures[robot.side].outpost.hp > 0)
      && insideZone(robot.position, zone)
    )) || null;
  }

  function serviceTarget(state, robot, kind) {
    const zones = state.model.service_zones?.[robot.side] || {};
    if (kind === "heal") return { name: "supply", zone: zones.supply };
    const candidates = Object.entries(zones)
      .filter(([name, zone]) => zone.ammo && (name !== "outpost" || state.structures[robot.side].outpost.hp > 0))
      .map(([name, zone]) => ({ name, zone, distance: state.router.distance(robot.position, zone.center) }))
      .sort((left, right) => left.distance - right.distance);
    return candidates[0] || { name: "supply", zone: zones.supply };
  }

  function insideAnyServiceZone(model, side, point) {
    return Object.values(model.service_zones?.[side] || {}).some((zone) => insideZone(point, zone));
  }

  function teamTargetPrior(state, robot) {
    const teamProfile = state.model.teams[robot.school];
    const priors = teamProfile.target_prior_by_30s || [];
    const phase = Math.min(priors.length - 1, Math.floor(state.second / 30));
    const entry = priors[Math.max(0, phase)] || {};
    const outpostAlive = state.structures[otherSide(robot.side)].outpost.hp > 0;
    const basePrior = (outpostAlive ? entry.outpost_alive : entry.outpost_down)
      || entry
      || (outpostAlive ? { robot: 0.45, outpost: 0.55, base: 0 } : { robot: 0.75, outpost: 0, base: 0.25 });
    if (!outpostAlive) return basePrior;
    const roleEvidence = teamProfile.outpost_attack_roles?.roles?.[robot.role];
    if (!roleEvidence) return basePrior;
    // Keep rare observed attacks possible, but do not turn a team-level
    // outpost style into an equal scripted assignment for every infantry.
    const preference = clamp(Number(roleEvidence.outpost_preference || 0.02), 0.02, 1);
    const adjustedOutpost = Number(basePrior.outpost || 0) * preference * preference;
    return {
      robot: Number(basePrior.robot || 0) + Number(basePrior.outpost || 0) - adjustedOutpost,
      outpost: adjustedOutpost,
      base: Number(basePrior.base || 0),
    };
  }

  function outpostAssaultActive(state, robot) {
    const campaign = state.teamState[robot.side];
    return Boolean(robot.outpostAssaultCommitted)
      && state.structures[otherSide(robot.side)].outpost.hp > 0
      // 还原“首次命中前的接敌路线”：快攻车需提前进入阵位并为结构目标保留弹药，
      // 不能到了实战首次命中时刻才开始转点。
      && state.second >= Math.max(1, Number(campaign.outpostAssaultStartSecond || 1) - 8)
      && state.second <= Number(campaign.outpostObjectiveSecond || 150) + 35;
  }

  function objectiveApproachPoint(state, robot, objective, points, forceServiceExit) {
    const objectivePoint = canonicalPoint(objective.position, robot.side);
    const range = Math.max(1.5, Number(robot.profile.range_m || 0));
    const preferredRange = Number(robot.profile.engagement_profile?.preferred_range_m || range * 0.7);
    const attackDistance = clamp(preferredRange, 1.5, range * 0.95);
    const legalObserved = points.filter((point) => {
      const candidate = [Number(point[0]), Number(point[1])];
      const worldCandidate = canonicalPoint(candidate, robot.side);
      return state.router.distance(candidate, objectivePoint) <= range * 0.82
        && (!forceServiceExit || !insideAnyServiceZone(state.model, "red", candidate))
        && lineOfSightFrom(state, worldCandidate, objective.position, robot.role);
    });
    const preferredObserved = robot.profile.engagement_profile?.style === "long_range"
      ? legalObserved.filter((point) => state.router.distance([Number(point[0]), Number(point[1])], objectivePoint) >= attackDistance * 0.72)
      : legalObserved;
    if (preferredObserved.length) return weightedPoint(preferredObserved, state.random);
    if (legalObserved.length) return weightedPoint(legalObserved, state.random);
    return [
      clamp(objectivePoint[0] - attackDistance, 0.1, 27.9), objectivePoint[1],
    ];
  }

  function tacticalCanonicalGoal(state, robot) {
    const phase = Math.min(6, Math.floor(state.second / 60));
    const current = canonicalPoint(robot.position, robot.side);
    const forceServiceExit = Boolean(robot.serviceExitPending);
    const points = robot.profile.goals_by_minute[phase] || robot.profile.goals_by_minute.at(-1) || [];
    const prior = teamTargetPrior(state, robot);
    const enemyStructures = state.structures[otherSide(robot.side)];
    const objectiveType = enemyStructures.outpost.hp > 0 ? "outpost" : "base";
    const objective = enemyStructures[objectiveType];
    robot.tacticalIntent = null;
    robot.objectiveKey = null;
    const committedAssault = objectiveType === "outpost" && outpostAssaultActive(state, robot);
    if (robot.profile.weapon && (
      committedAssault || state.random() < Math.min(0.92, Number(prior[objectiveType] || 0) * 0.95)
    )) {
      robot.serviceExitPending = false;
      robot.tacticalIntent = objectiveType;
      robot.objectiveKey = objective.key;
      return objectiveApproachPoint(state, robot, objective, points, forceServiceExit);
    }
    const transitions = robot.profile.transitions_by_minute?.[phase] || [];
    const nearbyTransitions = transitions.map((edge) => {
      const sourceDistance = state.router.distance(current, [Number(edge[0]), Number(edge[1])]);
      const adjustedWeight = Number(edge[4] || 1) / (0.35 + sourceDistance * sourceDistance);
      return [...edge, adjustedWeight, sourceDistance];
    }).filter((edge) => edge[6] <= 3.2 && (
      !forceServiceExit || !insideAnyServiceZone(state.model, "red", [Number(edge[2]), Number(edge[3])])
    ));
    const transition = weightedItem(nearbyTransitions, 5, state.random);
    let target;
    if (transition) {
      target = [Number(transition[2]), Number(transition[3])];
    } else {
      const eligible = forceServiceExit
        ? points.filter((point) => !insideAnyServiceZone(state.model, "red", [Number(point[0]), Number(point[1])]))
        : points;
      target = eligible.length ? weightedPoint(eligible, state.random) : [6.65, 7.5];
    }
    robot.serviceExitPending = false;
    target[0] = clamp(target[0] + (state.random() - 0.5) * 0.4, 0.1, 27.9);
    target[1] = clamp(target[1] + (state.random() - 0.5) * 0.4, 0.1, 14.9);
    return target;
  }

  function makeRobot(model, school, side, role, heroArchetype) {
    const profile = model.teams[school].roles[role];
    const archetype = role === "英雄" ? heroArchetype : null;
    const maxHp = roleMaxHp(profile, 0, archetype, model.rules);
    const canonicalSpawn = role === "空中" && profile.uav_navigation?.home
      ? profile.uav_navigation.home
      : profile.spawn;
    const spawn = canonicalPoint(canonicalSpawn, side);
    return {
      key: `${side}:${role}`,
      id: side === "red" ? ROLE_ID[role] : 100 + ROLE_ID[role],
      side,
      school,
      role,
      profile,
      level: robotLevel(profile, 0),
      heroArchetype: archetype,
      position: spawn,
      yaw: side === "red" ? 0 : 180,
      hp: maxHp,
      maxHp,
      ammo: 0,
      heat: 0,
      shots: 0,
      hits: 0,
      damage: 0,
      deaths: 0,
      kills: 0,
      respawnAt: null,
      respawnMode: null,
      respawnProgress: 0,
      respawnRequired: 0,
      respawnedAt: null,
      buybacks: 0,
      invulnerableUntil: 0,
      assemblyInvulnerableSeconds: 0,
      assemblyProtected: false,
      weak: false,
      weakKind: null,
      weakUntil: 0,
      boostUntil: 0,
      lastDamageAt: -999,
      radarCounterCount: 0,
      radarCounteredUntil: 0,
      radarCounterBuyouts: 0,
      uavFlightState: role === "空中" ? "parked" : null,
      uavSupportActive: false,
      uavSupportSeconds: role === "空中" ? Number(model.rules.uav_support?.initial_seconds || 30) : 0,
      uavPaidSupportSeconds: 0,
      uavNextStateAt: null,
      uavSortieEndsAt: null,
      lastKilledAt: -1,
      status: "部署",
      goal: [...spawn],
      route: [[...spawn]],
      passages: [],
      terrainActions: [],
      terrainSpeedMultiplier: 1,
      terrainAction: null,
      terrainMotionState: null,
      nextDecisionAt: 1,
      targetKey: null,
      objectiveKey: null,
      mode: "tactic",
      serviceTarget: null,
      serviceExitPending: false,
      serviceModeStartedAt: null,
      ammoServiceCooldownUntil: 0,
      outpostAssaultCommitted: false,
      shotBudget: 0,
    };
  }

  function technologyCorePlan(model, school, random) {
    const prior = model.teams[school].economy_prior || {};
    const reach = prior.core_reach_rate || [];
    const timing = prior.core_completion_seconds || [];
    const rules = model.rules.technology_core || {};
    const unlocks = rules.unlock_seconds || [0, 60, 120, 180];
    // A single draw preserves the empirical P(reach at least level N) curve.
    const reachDraw = random();
    let previous = -1;
    const plan = [];
    for (let index = 0; index < 4; index += 1) {
      if (reachDraw >= Number(reach[index] || 0)) break;
      const stats = timing[index] || {};
      const centre = Number(stats.median ?? (Number(unlocks[index] || 0) + 55));
      const low = Number(stats.p25 ?? (centre - 18));
      const high = Number(stats.p75 ?? (centre + 18));
      const spread = random() < 0.5 ? centre - low : high - centre;
      const sampled = Math.round(centre + (random() * 2 - 1) * Math.max(6, spread));
      const second = clamp(sampled, Math.max(Number(unlocks[index] || 0), previous + 12), 414);
      plan.push({ level: index + 1, plannedSecond: second, completedSecond: null });
      previous = second;
    }
    return plan;
  }

  function nextTechnologyCoreTask(state, side) {
    const core = state.teamState[side].technologyCore;
    return core.plan.find((task) => task.level === core.level + 1) || null;
  }

  function technologyCoreReadySecond(state, side, task) {
    const completions = state.teamState[side].technologyCore.completions;
    const previous = completions[completions.length - 1];
    return Math.max(Number(task?.plannedSecond || 0), previous ? previous.second + 12 : 0);
  }

  function allocateInitialAmmo(state, side) {
    const robots = state.robots.filter((robot) => robot.side === side && robot.profile.weapon);
    const initial = state.model.rules.initial_allowed_ammo || {};
    robots.forEach((robot) => {
      robot.ammo = Number(initial[robot.role] || 0);
    });
  }

  function technologyCoreRoute(state, robot, assembly) {
    const regular = state.router.terrainRoute(
      state.navigation, robot.position, assembly.center, robot.school, robot.role,
    );
    if (!regular.corrected || !assembly.entry_outside || !assembly.entry_inside) return regular;
    // A sampled core plan exists only when regional coin telemetry shows that
    // this team's engineer completed the assembly.  Route that engineer through
    // the actual central assembly entrance instead of treating the 400 mm edge
    // as a jump shortcut.
    const approach = state.router.terrainRoute(
      state.navigation, robot.position, assembly.entry_outside, robot.school, robot.role,
    );
    return {
      route: [...approach.route, [...assembly.entry_inside], [...assembly.center]],
      target: [...assembly.center],
      passages: [...new Set([...approach.passages, "工程装配入口"])],
      actions: [...(approach.actions || [])],
      corrected: false,
    };
  }

  function uavHome(robot) {
    return canonicalPoint(robot.profile.uav_navigation?.home || robot.profile.spawn, robot.side);
  }

  function sampledUavDuration(state, robot, field, fallback, low, high) {
    const centre = Number(robot.profile.uav_navigation?.[field] || fallback);
    return clamp(Math.round(centre * (0.8 + state.random() * 0.4)), low, high);
  }

  function uavStatus(state, robot, action) {
    const locked = Math.max(0, robot.radarCounteredUntil - state.second);
    const support = Math.max(0, Math.floor(robot.uavSupportSeconds));
    const paid = robot.uavSupportSeconds <= 0 && robot.uavSupportActive ? " · 付费支援" : "";
    const lock = locked ? ` · 发射机构锁定 ${locked}s` : "";
    robot.status = `${action} · 免费支援 ${support}s${paid}${lock}`;
  }

  function chooseUavAirGoal(state, robot) {
    const navigation = robot.profile.uav_navigation || {};
    const phase = Math.min(6, Math.floor(state.second / 60));
    const transitions = navigation.transitions_by_minute?.[phase] || [];
    const points = navigation.airborne_goals_by_minute?.[phase]
      || robot.profile.goals_by_minute?.[phase]
      || [[13.5, 12.5, 1]];
    const current = canonicalPoint(robot.position, robot.side);
    let selected = null;
    if (outpostAssaultActive(state, robot)) {
      const outpost = state.structures[otherSide(robot.side)].outpost;
      robot.tacticalIntent = "outpost";
      robot.objectiveKey = outpost.key;
      selected = objectiveApproachPoint(state, robot, outpost, points, false);
    } else {
      robot.tacticalIntent = null;
      robot.objectiveKey = null;
    }
    if (!selected && transitions.length) {
      let nearest = Infinity;
      transitions.forEach((edge) => {
        nearest = Math.min(nearest, Math.hypot(current[0] - Number(edge[0]), current[1] - Number(edge[1])));
      });
      // 从当前航点附近的真实 5 秒转移中抽样，不再独立抽全场热区。
      const local = transitions.filter((edge) => (
        Math.hypot(current[0] - Number(edge[0]), current[1] - Number(edge[1])) <= nearest + 0.75
      ));
      const edge = weightedItem(local, 4, state.random);
      if (edge) selected = [Number(edge[2]), Number(edge[3])];
    }
    if (!selected) {
      selected = weightedPoint(points, state.random);
    }
    selected[0] = clamp(selected[0] + (state.random() - 0.5) * 0.16, 0.1, 27.9);
    selected[1] = clamp(selected[1] + (state.random() - 0.5) * 0.16, 0.1, 14.9);
    const target = canonicalPoint(selected, robot.side);
    robot.goal = target;
    robot.route = [[...robot.position], [...target]];
    robot.passages = ["空中连续航迹"];
    robot.terrainActions = [];
    robot.nextDecisionAt = state.second + 5;
    uavStatus(state, robot, robot.tacticalIntent === "outpost" ? "前哨空中压制" : "空中巡航");
  }

  function startUavSortie(state, robot) {
    if (state.second >= state.duration || robot.role !== "空中" || robot.ammo <= 0 || robot.shots >= robot.shotBudget) return false;
    if (robot.uavSupportSeconds <= 0 && state.teamState[robot.side].coins < Number(state.model.rules.uav_support?.paid_cost_per_second || 1)) return false;
    robot.uavFlightState = "airborne";
    robot.uavSupportActive = true;
    robot.uavSortieEndsAt = state.second + sampledUavDuration(
      state, robot, "median_airborne_run_seconds", 90, 20, 210,
    );
    chooseUavAirGoal(state, robot);
    event(state, robot.side, "uav_takeoff", `空中机器人起飞，预计连续支援 ${robot.uavSortieEndsAt - state.second} 秒`);
    return true;
  }

  function parkUav(state, robot) {
    const home = uavHome(robot);
    robot.position = [...home];
    robot.goal = [...home];
    robot.route = [[...home]];
    robot.passages = [];
    robot.terrainActions = [];
    robot.uavFlightState = "parked";
    robot.uavSupportActive = false;
    robot.uavSortieEndsAt = null;
    robot.uavNextStateAt = state.second + sampledUavDuration(
      state, robot, "median_parked_run_seconds", 30, 8, 120,
    );
    uavStatus(state, robot, "停机坪待命");
    event(state, robot.side, "uav_land", `空中机器人返回停机坪，预计 ${robot.uavNextStateAt - state.second} 秒后再出动`);
  }

  function returnUav(state, robot, reason) {
    robot.uavFlightState = "returning";
    robot.goal = uavHome(robot);
    robot.route = [[...robot.position], [...robot.goal]];
    robot.passages = ["无人机返航"];
    robot.terrainActions = [];
    robot.nextDecisionAt = state.second + 2;
    uavStatus(state, robot, reason || "返航");
  }

  function updateUavSupport(state) {
    const rules = state.model.rules.uav_support || {};
    const interval = Number(rules.periodic_interval_seconds || 60);
    const grant = Number(rules.periodic_seconds || 20);
    const paidCost = Number(rules.paid_cost_per_second || 1);
    const periodic = state.second > 0 && state.second < state.duration && !(state.second % interval);
    state.robots.forEach((robot) => {
      if (robot.role !== "空中") return;
      // V2.1.0 的支援时间存在停机坪中，未起飞时只累积、不消耗。
      if (periodic) {
        robot.uavSupportSeconds += grant;
        event(state, robot.side, "uav_support", `停机坪免费支援时间 +${grant}s，现有 ${Math.floor(robot.uavSupportSeconds)}s`);
      }
      if (!robot.uavSupportActive || !["airborne", "returning"].includes(robot.uavFlightState)) return;
      if (robot.uavSupportSeconds > 0) {
        robot.uavSupportSeconds = Math.max(0, robot.uavSupportSeconds - 1);
        return;
      }
      const team = state.teamState[robot.side];
      if (team.coins >= paidCost) {
        team.coins -= paidCost;
        team.spent += paidCost;
        robot.uavPaidSupportSeconds += 1;
        state.stats[robot.side].uavPaidSupportSeconds += 1;
        return;
      }
      robot.uavSupportActive = false;
      returnUav(state, robot, "支援时间耗尽返航");
      event(state, robot.side, "uav_return", "空中支援时间耗尽且金币不足，返航");
    });
  }

  function moveUav(state, robot) {
    const home = uavHome(robot);
    if (robot.uavFlightState === "parked") {
      robot.position = [...home];
      robot.goal = [...home];
      robot.route = [[...home]];
      if (state.second >= robot.uavNextStateAt) startUavSortie(state, robot);
      else uavStatus(state, robot, "停机坪待命");
      return;
    }
    if (robot.uavFlightState === "airborne" && state.second >= robot.uavSortieEndsAt) {
      returnUav(state, robot, "本轮支援结束返航");
      event(state, robot.side, "uav_return", "空中机器人结束本轮支援，返航");
    }
    if (robot.uavFlightState === "airborne" && (state.second >= robot.nextDecisionAt || !robot.route?.length)) {
      chooseUavAirGoal(state, robot);
    } else if (robot.uavFlightState === "airborne") {
      uavStatus(state, robot, "空中巡航");
    } else if (robot.uavFlightState === "returning") {
      robot.goal = [...home];
      robot.route = [[...robot.position], [...home]];
      uavStatus(state, robot, robot.uavSupportActive ? "返航" : "无支援返航");
    }
    const moved = state.router.moveAlongRoute(robot.position, robot.route, Number(robot.profile.speed_mps));
    const previous = robot.position;
    robot.position = moved.position;
    robot.route = moved.route;
    if (state.router.distance(previous, robot.position) > 0.01) {
      robot.yaw = Math.atan2(robot.position[1] - previous[1], robot.position[0] - previous[0]) * 180 / Math.PI;
    }
    // 只在真正到达停机点后切换状态，不用阙值吸附造成最后一秒超速。
    if (robot.uavFlightState === "returning" && state.router.distance(robot.position, home) <= 0.005) parkUav(state, robot);
  }

  function createMatch(model, navigation, redSchool, blueSchool, seed, routerOverride, matchOptions) {
    if (!model?.teams?.[redSchool] || !model?.teams?.[blueSchool]) throw new Error("完整沙盘缺少战队参数");
    const router = routerOverride || browserRouter;
    if (!router?.terrainRoute) throw new Error("地形寻路器未加载");
    const seedValue = Number.isFinite(Number(seed)) ? Number(seed) >>> 0 : hashSeed(seed);
    const state = {
      model,
      navigation,
      router,
      random: mulberry32(seedValue),
      seed: seedValue,
      options: matchOptions || {},
      second: 0,
      duration: Number(model.duration_seconds) || 420,
      schools: { red: redSchool, blue: blueSchool },
      codes: { red: model.teams[redSchool].team, blue: model.teams[blueSchool].team },
      structures: {
        red: {
          base: { key: "red:base", side: "red", kind: "base", hp: model.rules.base_hp, maxHp: model.rules.base_hp, position: model.structures.red.base, armorOpen: false, armorOpenedBy: null, fortressCaptureSeconds: 0, fortressLastOccupiedAt: -999, fixedDartHits: 0 },
          outpost: { key: "red:outpost", side: "red", kind: "outpost", hp: model.rules.outpost_hp, maxHp: model.rules.outpost_hp, position: model.structures.red.outpost },
        },
        blue: {
          base: { key: "blue:base", side: "blue", kind: "base", hp: model.rules.base_hp, maxHp: model.rules.base_hp, position: model.structures.blue.base, armorOpen: false, armorOpenedBy: null, fortressCaptureSeconds: 0, fortressLastOccupiedAt: -999, fixedDartHits: 0 },
          outpost: { key: "blue:outpost", side: "blue", kind: "outpost", hp: model.rules.outpost_hp, maxHp: model.rules.outpost_hp, position: model.structures.blue.outpost },
        },
      },
      teamState: {},
      stats: {
        red: { damage: 0, robotDamage: 0, outpostDamage: 0, baseDamage: 0, shots17: 0, shots42: 0, hits: 0, kills: 0, supplies: 0, buybacks: 0, radarCounters: 0, uavCounterBuyouts: 0, uavPaidSupportSeconds: 0 },
        blue: { damage: 0, robotDamage: 0, outpostDamage: 0, baseDamage: 0, shots17: 0, shots42: 0, hits: 0, kills: 0, supplies: 0, buybacks: 0, radarCounters: 0, uavCounterBuyouts: 0, uavPaidSupportSeconds: 0 },
      },
      robots: [],
      events: [],
      finished: false,
      winner: null,
      reason: null,
    };
    SIDES.forEach((side) => {
      const school = state.schools[side];
      const initialCoins = Number(model.rules.initial_coins || 400);
      const teamProfile = model.teams[school];
      const attackWindows = teamProfile.outpost_attack_windows || [];
      const sampledWindow = attackWindows.length
        ? attackWindows[Math.floor(state.random() * attackWindows.length)]
        : null;
      const outpostDestroySamples = teamProfile.outpost_destroy_seconds || [];
      const sampledDestroy = sampledWindow?.destroy_second ?? (outpostDestroySamples.length
        ? outpostDestroySamples[Math.floor(state.random() * outpostDestroySamples.length)]
        : null);
      const outpostObjectiveSecond = Number(sampledDestroy || 150);
      const outpostFirstHitObjectiveSecond = Number(sampledWindow?.first_hit_second || Math.min(30, outpostObjectiveSecond * 0.3));
      state.teamState[side] = {
        coins: initialCoins,
        totalCoins: initialCoins,
        spent: 0,
        fortress: "neutral",
        dartWindows: 0,
        dartHits: 0,
        outpostObjectiveSecond,
        outpostFirstHitObjectiveSecond,
        outpostAssaultStartSecond: Math.max(1, Math.round(outpostFirstHitObjectiveSecond - 8)),
        outpostAssaultCount: 0,
        outpostAssaultAnnounced: false,
        weaponAccuracy: {
          "17mm": sampleMatchAccuracy(teamProfile, "17mm", state.random),
          "42mm": sampleMatchAccuracy(teamProfile, "42mm", state.random),
        },
        technologyCore: {
          level: 0,
          incomePer10: 0,
          defenseRatio: 0,
          levelCap: 5,
          earnedCoins: 0,
          plan: technologyCorePlan(model, school, state.random),
          completions: [],
        },
      };
      ROLE_ORDER.forEach((role) => {
        const requestedArchetype = state.options.heroArchetypes?.[side];
        const teamDefault = teamProfile.roles?.["英雄"]?.hero_archetype_default;
        const heroArchetype = ["melee", "ranged"].includes(requestedArchetype)
          ? requestedArchetype
          : (["melee", "ranged"].includes(teamDefault) ? teamDefault : "ranged");
        const robot = makeRobot(model, school, side, role, heroArchetype);
        robot.shotBudget = Math.max(0, Number(robot.profile.shots_per_game || 0) * (0.82 + state.random() * 0.36));
        if (role === "空中") {
          const observed = Number(robot.profile.uav_navigation?.first_takeoff_second ?? state.duration);
          robot.uavNextStateAt = observed >= state.duration
            ? state.duration + 1
            : clamp(Math.round(observed + (state.random() - 0.5) * Math.min(20, observed * 0.35)), 1, state.duration);
          robot.status = `停机坪待命 · 免费支援 ${Math.floor(robot.uavSupportSeconds)}s`;
        }
        state.robots.push(robot);
      });
      allocateInitialAmmo(state, side);
      const openingPrior = Number(teamProfile.target_prior_by_30s?.[0]?.outpost_alive?.outpost || 0);
      const armedRobots = state.robots.filter((robot) => robot.side === side && robot.profile.weapon);
      const roleEvidence = teamProfile.outpost_attack_roles?.roles || {};
      const eligible = armedRobots.map((robot) => ({
        robot,
        evidence: roleEvidence[robot.role] || {},
      })).filter((item) => item.evidence.primary_assault_role);
      eligible.forEach(({ robot, evidence }) => {
        const probability = Number(evidence.commitment_probability ?? clamp(0.2 + Number(evidence.game_rate || 0) * 0.75, 0.2, 0.92));
        if (state.random() < probability) robot.outpostAssaultCommitted = true;
      });
      if (openingPrior > 0.2 && eligible.length && !eligible.some(({ robot }) => robot.outpostAssaultCommitted)) {
        eligible.sort((left, right) => (
          Number(right.evidence.opening_share || right.evidence.share || 0)
          - Number(left.evidence.opening_share || left.evidence.share || 0)
        ));
        eligible[0].robot.outpostAssaultCommitted = true;
      }
      const committedCount = armedRobots.filter((robot) => robot.outpostAssaultCommitted).length;
      state.teamState[side].outpostAssaultCount = committedCount;
    });
    event(state, "system", "start", `${state.codes.red} vs ${state.codes.blue} 完整沙盘开局`);
    return state;
  }

  function updateEconomy(state) {
    const automatic = new Map(state.model.rules.automatic_income || [[61, 50], [121, 50], [181, 50], [241, 50], [301, 50], [361, 150]]);
    const income = Number(automatic.get(state.second) || 0);
    SIDES.forEach((side) => {
      const team = state.teamState[side];
      if (income) {
        team.coins += income;
        team.totalCoins += income;
        event(state, side, "economy", `国赛定时经济 +${income} 金币`);
      }
      const coreRules = state.model.rules.technology_core || {};
      const interval = Number(coreRules.income_interval_seconds || 10);
      team.technologyCore.completions.forEach((completion) => {
        if (state.second <= completion.second || (state.second - completion.second) % interval) return;
        team.coins += completion.income;
        team.totalCoins += completion.income;
        team.technologyCore.earnedCoins += completion.income;
      });
    });
  }

  function updateRobotLevel(state, robot) {
    const levelCap = Number(state.teamState[robot.side].technologyCore.levelCap || 5);
    const nextLevel = Math.min(levelCap, robotLevel(robot.profile, state.second));
    const nextMax = roleMaxHp(robot.profile, state.second, robot.heroArchetype, state.model.rules, nextLevel);
    robot.level = nextLevel;
    if (nextMax <= robot.maxHp) return;
    const gain = nextMax - robot.maxHp;
    robot.maxHp = nextMax;
    if (robot.hp > 0) robot.hp += gain;
  }

  function openBaseArmor(state, side, source) {
    const base = state.structures[side].base;
    if (base.armorOpen) return false;
    base.armorOpen = true;
    base.armorOpenedBy = source;
    event(state, otherSide(side), "base_armor", `${side === "red" ? "红方" : "蓝方"}基地护甲展开（${source}）`);
    return true;
  }

  function resolveFortresses(state) {
    SIDES.forEach((fortSide) => {
      const centre = state.model.structures[fortSide].fortress;
      const present = { red: 0, blue: 0 };
      const active = state.structures[fortSide].outpost.hp <= 0;
      if (active) {
        state.robots.forEach((robot) => {
          if (robot.hp <= 0 || robot.weak || !["英雄", "步兵3", "步兵4", "哨兵"].includes(robot.role)) return;
          if (state.router.distance(robot.position, centre) <= 1.3) present[robot.side] += 1;
        });
      }
      let owner = "neutral";
      if (present.red && present.blue) owner = "contested";
      else if (present.red) owner = "red";
      else if (present.blue) owner = "blue";
      if (state.teamState[fortSide].fortress !== owner) {
        state.teamState[fortSide].fortress = owner;
        if (owner !== "neutral") event(state, owner === "contested" ? "system" : owner, "fortress", `${fortSide === "red" ? "红方" : "蓝方"}堡垒：${owner === "contested" ? "双方争夺" : `${state.codes[owner]} 控制`}`);
      }
      const base = state.structures[fortSide].base;
      if (base.armorOpen) return;
      const rules = state.model.rules.base_armor || {};
      const attacker = otherSide(fortSide);
      const eligible = state.second >= Number(rules.enemy_fortress_unlock_second || 180)
        && (!rules.enemy_outpost_must_be_down || state.structures[fortSide].outpost.hp <= 0);
      if (!eligible) {
        base.fortressCaptureSeconds = 0;
        base.fortressLastOccupiedAt = -999;
        return;
      }
      if (owner === attacker) {
        base.fortressCaptureSeconds += 1;
        base.fortressLastOccupiedAt = state.second;
      } else if (state.second - base.fortressLastOccupiedAt > Number(rules.capture_grace_seconds || 3)) {
        base.fortressCaptureSeconds = 0;
      }
      if (base.fortressCaptureSeconds >= Number(rules.capture_seconds || 20)) {
        openBaseArmor(state, fortSide, "敌方堡垒连续占领20秒");
      }
    });
  }

  function needsHealing(robot) {
    if (robot.role === "空中") return false;
    return robot.weak || robot.hp / robot.maxHp < 0.43;
  }

  function needsAmmo(state, robot) {
    if (!robot.profile.weapon || robot.role === "空中" || robot.shots >= robot.shotBudget) return false;
    if (state.second < Number(robot.ammoServiceCooldownUntil || 0)) return false;
    return robot.ammo <= (robot.profile.weapon === "42mm" ? 1 : 8);
  }

  function needsService(state, robot) {
    if (robot.role === "空中") return false;
    return needsHealing(robot) || needsAmmo(state, robot);
  }

  function canShelterInAssembly(state, robot) {
    if (robot.role !== "工程" || robot.hp <= 0) return false;
    const limit = Number(state.model.rules.engineer_assembly_invulnerability_seconds || 180);
    return robot.assemblyInvulnerableSeconds < limit
      && insideZone(robot.position, state.model.assembly_zones?.[robot.side]);
  }

  function serviceRequiredForDecision(state, robot) {
    return needsService(state, robot) && !canShelterInAssembly(state, robot);
  }

  function chooseGoal(state, robot) {
    if (robot.role === "空中") {
      chooseUavAirGoal(state, robot);
      return;
    }
    robot.objectiveKey = null;
    let target;
    if (needsService(state, robot) && canShelterInAssembly(state, robot)) {
      const assembly = state.model.assembly_zones[robot.side];
      target = assembly.center;
      robot.mode = "assembly_hold";
      robot.serviceTarget = null;
      robot.status = "装配区无敌驻留 · 暂不回补给区";
    } else if (needsService(state, robot)) {
      const healing = needsHealing(robot);
      const service = serviceTarget(state, robot, healing ? "heal" : "ammo");
      target = service.zone.center;
      if (!['heal', 'ammo'].includes(robot.mode) || robot.mode !== (healing ? "heal" : "ammo")) {
        robot.serviceModeStartedAt = state.second;
      }
      robot.mode = healing ? "heal" : "ammo";
      robot.serviceTarget = service.name;
      robot.status = robot.weak ? "虚弱撤回补给区" : healing ? "残血前往补给区" : `前往${service.zone.label}补弹`;
    } else if (robot.role === "工程") {
      const task = nextTechnologyCoreTask(state, robot.side);
      const assembly = state.model.assembly_zones?.[robot.side];
      const approachSeconds = 36;
      const readySecond = task ? technologyCoreReadySecond(state, robot.side, task) : null;
      if (task && assembly && state.second >= readySecond - approachSeconds) {
        target = assembly.center;
        robot.mode = "technology_core";
        robot.serviceTarget = null;
        const remaining = Math.max(0, readySecond - state.second);
        robot.status = insideZone(robot.position, assembly)
          ? `装配科技核心 Lv.${task.level}${remaining ? ` · ${remaining}s` : " · 待确认"}`
          : `前往装配区 · 目标 Lv.${task.level}`;
      }
    }
    if (!target) {
      robot.serviceTarget = null;
      robot.mode = "tactic";
      const canonical = tacticalCanonicalGoal(state, robot);
      target = canonicalPoint(canonical, robot.side);
      robot.status = robot.role === "工程"
        ? `工程运营 · 科技核心 Lv.${state.teamState[robot.side].technologyCore.level}`
        : robot.tacticalIntent === "outpost" ? "前哨压制转点"
          : robot.tacticalIntent === "base" ? "基地压制转点" : "战术转点";
    }
    const assembly = robot.mode === "technology_core" ? state.model.assembly_zones?.[robot.side] : null;
    const planned = assembly
      ? technologyCoreRoute(state, robot, assembly)
      : state.router.terrainRoute(state.navigation, robot.position, target, robot.school, robot.role);
    robot.goal = planned.target;
    robot.route = planned.route;
    robot.passages = planned.passages;
    robot.terrainActions = planned.actions || [];
    robot.nextDecisionAt = robot.mode === "technology_core"
      ? state.second + 3
      : state.second + 6 + Math.floor(state.random() * 6);
    const meaningfulPassages = planned.passages.filter((passage) => passage !== "空中直达");
    if (meaningfulPassages.length) event(state, robot.side, "terrain", `${robot.role}：${meaningfulPassages.join("、")}`);
  }

  function completeTechnologyCore(state, side, task, engineer) {
    const team = state.teamState[side];
    const core = team.technologyCore;
    const rules = state.model.rules.technology_core;
    const levelIndex = task.level - 1;
    const income = Number(rules.first_income_per_10[levelIndex] || 0);
    task.completedSecond = state.second;
    core.level = task.level;
    core.incomePer10 += income;
    core.levelCap = Number(rules.robot_level_cap_by_level[levelIndex] || core.levelCap);
    core.defenseRatio = Number(rules.defense_ratio_by_level[levelIndex] || 0);
    core.completions.push({ level: task.level, second: state.second, income });
    // Regional total-coins telemetry starts the new ten-second stream at the
    // completion signal, so the first payment is settled immediately.
    team.coins += income;
    team.totalCoins += income;
    core.earnedCoins += income;
    if (task.level === 4) {
      const base = state.structures[side].base;
      const gain = Number(rules.level_four_base_hp_gain || 2000);
      base.maxHp += gain;
      base.hp += gain;
    }
    engineer.status = `已兑换科技核心 Lv.${task.level} · +${income}/10s`;
    engineer.nextDecisionAt = state.second + 4;
    event(state, side, "technology_core", `工程完成科技核心 Lv.${task.level}，经济提升至 +${core.incomePer10}/10秒`, {
      level: task.level,
      incomePer10: core.incomePer10,
    });
  }

  function updateTechnologyCores(state) {
    SIDES.forEach((side) => {
      const task = nextTechnologyCoreTask(state, side);
      if (!task || state.second < technologyCoreReadySecond(state, side, task)) return;
      const engineer = state.robots.find((robot) => robot.side === side && robot.role === "工程");
      const assembly = state.model.assembly_zones?.[side];
      if (!engineer || engineer.hp <= 0 || !insideZone(engineer.position, assembly)) {
        if (engineer?.hp > 0) {
          engineer.status = `科技核心 Lv.${task.level} 延迟 · 前往装配区`;
          engineer.nextDecisionAt = state.second;
        }
        return;
      }
      completeTechnologyCore(state, side, task, engineer);
    });
  }

  function crossesForbiddenTerrainGate(state, robot, start, end) {
    const profile = state.navigation.teams?.[robot.school]?.[robot.role] || {};
    const abilities = new Set(profile.abilities || []);
    const forbiddenGate = state.navigation.gates.some((gate) => {
      const blocker = gate.routing_blocker_polygon || gate.polygon;
      if (!state.router.segmentHitsPolygon(start, end, blocker)) return false;
      if (["rough_road", "road_tunnel", "highland_tunnel"].includes(gate.category)) {
        return !abilities.has(gate.category);
      }
      if (gate.category === "fly_ramp") {
        const forward = gate.side === "blue" ? end[0] < start[0] : end[0] > start[0];
        return !abilities.has("fly_ramp")
          || (!forward && !profile.reverse_fly_ramp?.allowed);
      }
      if (gate.category === "road_step") {
        const vector = {
          positive_y: [0, 1], negative_y: [0, -1],
          positive_x: [1, 0], negative_x: [-1, 0],
        }[gate.high_direction];
        const ascending = vector
          ? (end[0] - start[0]) * vector[0] + (end[1] - start[1]) * vector[1] > 0
          : true;
        return ascending && !abilities.has("road_step");
      }
      return false;
    });
    if (forbiddenGate) return true;
    return crossesStaticWall(state, start, end);
  }

  function crossesForbiddenSymmetricGate(state, robot, start, end) {
    return crossesForbiddenTerrainGate(state, robot, start, end);
  }

  function crossesStaticWall(state, start, end) {
    return (state.navigation.static_obstacles || []).some((obstacle) => (
      obstacle.blocks_movement !== false
      && state.router.segmentHitsPolygon(start, end, obstacle.polygon)
    ));
  }

  function enforceFrameWallClearance(state, frameStarts) {
    state.robots.forEach((robot) => {
      // UAVs fly above the wall layer and are intentionally unrestricted.
      if (robot.role === "空中" || robot.hp <= 0) return;
      const start = frameStarts.get(robot.key);
      if (!start || !crossesForbiddenTerrainGate(state, robot, start, robot.position)) return;
      robot.position = [...start];
      robot.route = [[...start]];
      robot.nextDecisionAt = state.second;
      robot.terrainAction = null;
      robot.terrainSpeedMultiplier = 1;
      robot.status = "墙体/地形封口阻止 · 重新规划";
    });
  }

  function learnedTerrainMotion(state, robot, terrain) {
    const action = terrain.action;
    if (!action || action.category !== "fly_ramp") {
      robot.terrainMotionState = null;
      return terrain;
    }
    const key = `${action.id}:${action.direction}`;
    const profile = state.navigation.teams?.[robot.school]?.[robot.role]
      ?.terrain_motion_profiles?.fly_ramp
      || state.navigation.routing?.default_terrain_motion_profiles?.fly_ramp;
    if (!profile) return terrain;
    if (robot.terrainMotionState?.key !== key) {
      const alignmentProbability = Number(profile.alignment_probability || 0);
      const aligns = state.random() < alignmentProbability;
      const conditionalStop = clamp(Number(profile.full_stop_probability || 0) / Math.max(0.001, alignmentProbability), 0, 1);
      const stops = aligns && state.random() < conditionalStop;
      robot.terrainMotionState = {
        key,
        alignmentRemaining: aligns ? Number(profile.alignment_seconds || 1) : 0,
        alignmentMultiplier: stops ? 0 : Number(profile.alignment_multiplier || 0.2),
        accelerationIndex: aligns ? 0 : Number(profile.acceleration_multipliers?.length || 0),
      };
    }
    const motion = robot.terrainMotionState;
    if (motion.alignmentRemaining > 0) {
      motion.alignmentRemaining -= 1;
      return {
        multiplier: motion.alignmentMultiplier,
        action: { ...action, label: `${action.label}·起点对位${motion.alignmentMultiplier === 0 ? "停顿" : "减速"}` },
      };
    }
    const acceleration = profile.acceleration_multipliers || [];
    if (motion.accelerationIndex < acceleration.length) {
      const multiplier = Number(acceleration[motion.accelerationIndex]);
      motion.accelerationIndex += 1;
      return { multiplier, action: { ...action, label: `${action.label}·加速 ${motion.accelerationIndex}/${acceleration.length}` } };
    }
    return { multiplier: Number(profile.cruise_multiplier || terrain.multiplier || 1), action };
  }

  function moveRobots(state) {
    state.robots.forEach((robot) => {
      robot.heat = Math.max(0, robot.heat - Number(robot.profile.cooling_per_second || 0));
      if (robot.role === "空中") {
        moveUav(state, robot);
        return;
      }
      updateRobotLevel(state, robot);
      if (robot.hp <= 0) return;
      const requiresService = serviceRequiredForDecision(state, robot);
      const serviceChanged = requiresService && !["heal", "ammo"].includes(robot.mode);
      const recovered = !requiresService && ["heal", "ammo"].includes(robot.mode);
      const serviceInvalid = ["heal", "ammo"].includes(robot.mode)
        && robot.serviceTarget === "outpost"
        && state.structures[robot.side].outpost.hp <= 0;
      if (state.second >= robot.nextDecisionAt || !robot.route?.length || serviceChanged || recovered || serviceInvalid) chooseGoal(state, robot);
      const nominalSpeed = Number(robot.profile.speed_mps) * (robot.weak ? 0.88 : 1) * (robot.hp / robot.maxHp < 0.25 ? 0.9 : 1);
      const previousTerrainAction = robot.terrainAction;
      const rawTerrain = state.router.terrainMotion
        ? state.router.terrainMotion(
          state.navigation, robot.position, robot.route, robot.terrainActions, nominalSpeed,
        )
        : { multiplier: 1, action: null };
      const terrain = learnedTerrainMotion(state, robot, rawTerrain);
      const speed = nominalSpeed * Number(terrain.multiplier ?? 1);
      robot.terrainSpeedMultiplier = Number(terrain.multiplier ?? 1);
      robot.terrainAction = terrain.action?.label || null;
      if (robot.terrainAction) {
        robot.status = `${robot.terrainAction} · ${Math.round(robot.terrainSpeedMultiplier * 100)}% 速度`;
      } else if (previousTerrainAction && /% 速度$/.test(robot.status)) {
        robot.status = robot.mode === "technology_core" ? "前往科技核心装配区"
          : robot.mode === "assembly_hold" ? "装配区无敌驻留" : "战术转点";
      }
      const moved = state.router.moveAlongRoute(robot.position, robot.route, speed, true);
      const previous = robot.position;
      if (crossesForbiddenSymmetricGate(state, robot, previous, moved.position)) {
        robot.route = [[...previous]];
        robot.nextDecisionAt = state.second;
        robot.status = "围挡/地形门禁阻止 · 重新规划";
      } else {
        robot.position = moved.position;
        robot.route = moved.route;
      }
      if (state.router.distance(previous, robot.position) > 0.01) {
        robot.yaw = Math.atan2(robot.position[1] - previous[1], robot.position[0] - previous[0]) * 180 / Math.PI;
      }
    });
  }

  function resupplyRobots(state) {
    state.robots.forEach((robot) => {
      if (robot.hp <= 0 || robot.role === "空中") return;
      const healingZone = serviceZoneAt(state, robot, "heal");
      const ammoZone = serviceZoneAt(state, robot, "ammo");
      if (!healingZone && !ammoZone) return;
      const beforeHp = robot.hp;
      let purchasedAmmo = false;
      if (healingZone) {
        const outOfCombat = state.second - robot.lastDamageAt >= Number(state.model.rules.out_of_combat_seconds || 6);
        const lateFastHeal = state.second >= Number(state.model.rules.late_heal_start_second || 240) && outOfCombat;
        const healRatio = lateFastHeal
          ? Number(state.model.rules.late_heal_ratio_per_second || 0.25)
          : Number(state.model.rules.heal_ratio_per_second || 0.1);
        robot.hp = Math.min(robot.maxHp, robot.hp + robot.maxHp * healRatio);
      }
      if (robot.weak && robot.weakKind === "timed" && ammoZone) {
        robot.weak = false;
        robot.weakKind = null;
        const minimumInvulnerable = Number(state.model.rules.respawn.minimum_invulnerable_after_zone_seconds || 10);
        robot.invulnerableUntil = Math.max(state.second, Number(robot.respawnedAt || state.second) + minimumInvulnerable);
        robot.status = `${ammoZone[1].label}解除虚弱`;
        event(state, robot.side, "supply", `${robot.role} 在${ammoZone[1].label}解除虚弱`, { zone: ammoZone[0] });
      }
      const weapon = robot.profile.weapon;
      if (weapon && robot.role !== "空中" && ammoZone && robot.ammo < robot.profile.magazine * 0.35 && robot.shots < robot.shotBudget) {
        const team = state.teamState[robot.side];
        const maximumBundles = Math.min(5, Math.floor(team.coins / 10));
        const missing = robot.profile.magazine - robot.ammo;
        const roundsPerBundle = weapon === "42mm" ? 1 : 10;
        const bundles = Math.min(maximumBundles, Math.ceil(missing / roundsPerBundle));
        if (bundles > 0) {
          const rounds = bundles * roundsPerBundle;
          robot.ammo += rounds;
          team.coins -= bundles * 10;
          team.spent += bundles * 10;
          state.stats[robot.side].supplies += 1;
          purchasedAmmo = true;
          event(state, robot.side, "supply", `${robot.role} 在${ammoZone[1].label}补给 ${rounds} 发 ${weapon}`, { zone: ammoZone[0] });
        }
      }
      if (robot.hp > beforeHp + 0.5) robot.status = "补给区回血";
      else if (purchasedAmmo) robot.status = `${ammoZone[1].label}补弹完成`;
      else if (robot.mode === "ammo" && needsAmmo(state, robot)) robot.status = `${ammoZone[1].label}等待补弹`;
      const waitedWithoutCoins = robot.mode === "ammo"
        && !purchasedAmmo
        && state.teamState[robot.side].coins < 10
        && state.second - Number(robot.serviceModeStartedAt ?? state.second) >= 6;
      if (waitedWithoutCoins) {
        robot.ammoServiceCooldownUntil = state.second + 20;
        robot.serviceExitPending = true;
        robot.nextDecisionAt = state.second;
        robot.status = "金币不足 · 先离开补弹区";
      }
      if (!needsService(state, robot) && ["heal", "ammo"].includes(robot.mode)) {
        robot.serviceExitPending = true;
        robot.nextDecisionAt = state.second;
        robot.serviceModeStartedAt = null;
      }
    });
  }

  function updateAssemblyProtection(state) {
    const limit = Number(state.model.rules.engineer_assembly_invulnerability_seconds || 180);
    state.robots.forEach((robot) => {
      robot.assemblyProtected = false;
      if (robot.role !== "工程" || robot.hp <= 0 || robot.assemblyInvulnerableSeconds >= limit) return;
      const assembly = state.model.assembly_zones?.[robot.side];
      if (!insideZone(robot.position, assembly)) return;
      robot.assemblyProtected = true;
      robot.assemblyInvulnerableSeconds = Math.min(limit, robot.assemblyInvulnerableSeconds + 1);
      robot.status = `装配区无敌 · 累计 ${robot.assemblyInvulnerableSeconds}/${limit}s`;
    });
  }

  function lineOfSightFrom(state, start, end, role) {
    if (role === "空中") return true;
    const wallBlocked = (state.navigation.static_obstacles || []).some((obstacle) => (
      obstacle.blocks_ground_fire !== false
      && state.router.segmentHitsPolygon(start, end, obstacle.polygon)
    ));
    if (wallBlocked) return false;
    const startRegion = state.router.regionAt(state.navigation, start);
    const endRegion = state.router.regionAt(state.navigation, end);
    if (startRegion && endRegion) return startRegion.id === endRegion.id;
    // 站在高地上向下射击，或从地面攻击高地边缘的前哨站，并不等于
    // 子弹穿过整块高地。只有发射点和目标都在地面、弹道横穿高地时才阻断。
    if (startRegion || endRegion) return true;
    return !Object.values(state.navigation.regions).some((polygon) => state.router.segmentHitsPolygon(start, end, polygon));
  }

  function lineOfSight(state, attacker, end) {
    return lineOfSightFrom(state, attacker.position, end, attacker.role);
  }

  function targetCandidates(state, robot) {
    const enemy = otherSide(robot.side);
    const range = Number(robot.profile.range_m || 0);
    const candidates = state.robots
      .filter((target) => target.side === enemy && target.role !== "空中" && target.hp > 0
        && state.second >= target.invulnerableUntil && !target.assemblyProtected)
      .map((target) => ({ entity: target, distance: state.router.distance(robot.position, target.position), type: "robot" }))
      .filter((candidate) => candidate.distance <= range && lineOfSight(state, robot, candidate.entity.position));
    const structures = state.structures[enemy];
    const structureOrder = structures.outpost.hp > 0 ? [structures.outpost, structures.base] : [structures.base];
    structureOrder.forEach((structure) => {
      if (structure.kind === "outpost"
        && state.second < Number(state.teamState[robot.side].outpostFirstHitObjectiveSecond || 0)) return;
      const distance = state.router.distance(robot.position, structure.position);
      if (structure.hp > 0 && distance <= range && lineOfSight(state, robot, structure.position)) {
        candidates.push({ entity: structure, distance, type: structure.kind });
      }
    });
    if (candidates.length <= 1) return candidates;

    const groups = { robot: [], outpost: [], base: [] };
    candidates.forEach((candidate) => groups[candidate.type].push(candidate));
    groups.robot.sort((left, right) => (
      left.entity.hp / left.entity.maxHp + left.distance * 0.035
    ) - (
      right.entity.hp / right.entity.maxHp + right.distance * 0.035
    ));
    groups.outpost.sort((left, right) => left.distance - right.distance);
    groups.base.sort((left, right) => left.distance - right.distance);

    // A hero already beside the base must actually take the available top/front
    // armour shot instead of randomly ignoring the structure for a distant robot.
    if (robot.role === "英雄" && groups.base[0]?.distance <= 3.5) {
      return [...groups.base, ...groups.robot, ...groups.outpost];
    }

    if (robot.objectiveKey === structures.outpost.key && groups.outpost.length) {
      return [...groups.outpost, ...groups.robot, ...groups.base];
    }

    const prior = teamTargetPrior(state, robot);
    const available = Object.entries(groups)
      .filter(([, values]) => values.length)
      .map(([type]) => [type, Math.max(0.02, Number(prior[type] || 0))]);
    const selectedType = weightedItem(available, 1, state.random)?.[0] || available[0][0];
    return [...groups[selectedType], ...Object.entries(groups).filter(([type]) => type !== selectedType).flatMap(([, values]) => values)];
  }

  function fireWeapons(state) {
    const pending = [];
    state.robots.forEach((robot) => { robot.targetKey = null; });
    state.robots.forEach((robot) => {
      const weapon = robot.profile.weapon;
      if (robot.hp <= 0 || robot.weak || !weapon || robot.ammo <= 0) return;
      if (robot.role === "空中" && (
        robot.uavFlightState !== "airborne" || !robot.uavSupportActive || state.second < robot.radarCounteredUntil
      )) return;
      const heatPerShot = Number(state.model.rules.heat_per_shot[weapon]);
      const heatRoom = Math.max(0, Number(robot.profile.heat_limit) - robot.heat);
      const heatShots = Math.floor(heatRoom / heatPerShot);
      if (heatShots <= 0) return;
      const budgetShots = Math.max(0, Math.floor(robot.shotBudget - robot.shots));
      if (budgetShots <= 0) return;
      const candidates = targetCandidates(state, robot);
      if (!candidates.length) return;
      const target = candidates[0];
      const burst = Math.min(weapon === "42mm" ? 1 : 10, robot.ammo, heatShots, budgetShots, Number(robot.profile.burst_per_active_second || 1));
      let shots = Math.floor(burst);
      if (state.random() < burst - shots) shots += 1;
      if (!shots) return;
      const teamProfile = state.model.teams[robot.school];
      const baseAccuracy = Number(state.teamState[robot.side].weaponAccuracy?.[weapon] ?? teamProfile.accuracy[weapon] ?? 0.1);
      const distanceFactor = 1.12 - 0.47 * target.distance / Math.max(1, Number(robot.profile.range_m));
      const outpostDeadline = Math.max(20, Number(state.teamState[robot.side].outpostObjectiveSecond || 150));
      const structureAccuracy = target.type === "robot" ? 1
        : target.type === "outpost" ? 1.35 + 0.75 * clamp((state.second + 20) / outpostDeadline, 0, 1)
          : target.entity.armorOpen ? 1.35 : weapon === "42mm" ? 1.15 : 0.42;
      const accuracy = clamp(baseAccuracy * ROLE_ACCURACY[robot.role] * distanceFactor * structureAccuracy, 0.018, target.type === "robot" ? 0.78 : 0.9);
      let hits = 0;
      for (let index = 0; index < shots; index += 1) if (state.random() < accuracy) hits += 1;
      robot.ammo -= shots;
      robot.heat += shots * heatPerShot;
      robot.shots += shots;
      robot.hits += hits;
      robot.targetKey = target.entity.key;
      robot.status = target.type === "robot" ? `对枪 ${target.entity.role}` : `攻击${target.entity.kind === "base" ? "基地" : "前哨"}`;
      state.stats[robot.side][weapon === "42mm" ? "shots42" : "shots17"] += shots;
      state.stats[robot.side].hits += hits;
      if (hits) pending.push({
        attacker: robot,
        target: target.entity,
        hits,
        weapon,
        damage: hits * damagePerHit(state, robot, target.entity, weapon),
      });
    });
    applyDamage(state, pending);
  }

  function applyDamage(state, pending) {
    pending.forEach((hit) => {
      // V2.1.0: 空中机器人不适用攻击伤害和撞击伤害。
      if (hit.target.role === "空中") return;
      if (hit.target.invulnerableUntil > state.second || hit.target.assemblyProtected) return;
      let damage = hit.damage;
      if (hit.target.kind === "base" && !hit.target.armorOpen && hit.weapon === "17mm") {
        const ordinary = Number(state.model.rules.damage["17mm"] || 20);
        const topArmor = Number(state.model.rules.damage.base_top_17mm || 5);
        damage *= topArmor / Math.max(1, ordinary);
      }
      const groundOrStructure = hit.target.kind || (hit.target.role && hit.target.role !== "空中");
      if (groundOrStructure) {
        damage *= 1 - Number(state.teamState[hit.target.side]?.technologyCore?.defenseRatio || 0);
      }
      const actual = Math.min(hit.target.hp, damage);
      hit.target.hp = Math.max(0, hit.target.hp - actual);
      if (hit.target.role) hit.target.lastDamageAt = state.second;
      hit.attacker.damage += actual;
      const stats = state.stats[hit.attacker.side];
      stats.damage += actual;
      if (hit.target.kind === "base") stats.baseDamage += actual;
      else if (hit.target.kind === "outpost") stats.outpostDamage += actual;
      else stats.robotDamage += actual;
      if (actual >= 80 || hit.target.hp <= 0) {
        const targetName = hit.target.role || (hit.target.kind === "base" ? "基地" : "前哨站");
        event(state, hit.attacker.side, "hit", `${hit.attacker.role} ${hit.hits} 发${hit.weapon}命中${targetName}，伤害 ${Math.round(actual)}`);
      }
      if (hit.target.hp <= 0) {
        if (hit.target.role) killRobot(state, hit.target, hit.attacker);
        else event(state, hit.attacker.side, "destroy", `${hit.attacker.role} 击毁${hit.target.side === "red" ? "红方" : "蓝方"}${hit.target.kind === "base" ? "基地" : "前哨站"}`);
      }
    });
  }

  function shouldBuyback(state, robot, cost) {
    const policySetting = state.options.buybackPolicy;
    const policy = (typeof policySetting === "object" ? policySetting?.[robot.side] : policySetting) || "auto";
    const team = state.teamState[robot.side];
    if (team.coins < cost) return false;
    if (policy === "always") return true;
    if (policy === "never") return false;
    const remaining = state.duration - state.second;
    if (remaining <= 8 || team.coins - cost < 20) return false;
    const roleValue = { 英雄: 0.72, 工程: 0.28, 步兵3: 0.52, 步兵4: 0.52, 哨兵: 0.68, 空中: 0.42 }[robot.role] || 0.4;
    const baseUrgency = state.structures[robot.side].base.hp / state.structures[robot.side].base.maxHp < 0.45 ? 0.18 : 0;
    const waitUrgency = clamp(robot.respawnRequired / Math.max(10, remaining), 0, 0.45);
    return state.random() < clamp(roleValue + baseUrgency + waitUrgency - cost / Math.max(500, team.coins) * 0.38, 0.08, 0.94);
  }

  function buybackRobot(state, robot, cost) {
    if (robot.role === "空中") return;
    const rules = state.model.rules.respawn;
    const team = state.teamState[robot.side];
    team.coins -= cost;
    team.spent += cost;
    robot.buybacks += 1;
    robot.maxHp = roleMaxHp(robot.profile, state.second, robot.heroArchetype, state.model.rules, robot.level);
    robot.hp = robot.maxHp * Number(rules.buyback_hp_ratio || 1);
    robot.respawnAt = null;
    robot.respawnMode = "buyback";
    robot.respawnProgress = 0;
    robot.respawnedAt = state.second;
    robot.weak = true;
    robot.weakKind = "buyback";
    robot.weakUntil = state.second + Number(rules.buyback_weak_seconds || 3);
    robot.invulnerableUntil = state.second + Number(rules.buyback_invulnerable_seconds || 3);
    robot.boostUntil = state.second + Number(rules.buyback_chassis_boost_seconds || 4);
    robot.status = `当场买活 · ${cost} 金币`;
    robot.nextDecisionAt = state.second + 1;
    state.stats[robot.side].buybacks += 1;
    event(state, robot.side, "buyback", `${robot.role} 花费 ${cost} 金币原地满血买活`);
  }

  function killRobot(state, robot, attacker) {
    // 空中机器人没有普通战亡、补血或复活流程；雷达只锁定发射机构。
    if (robot.role === "空中") return;
    if (robot.respawnAt != null || robot.lastKilledAt === state.second) return;
    robot.lastKilledAt = state.second;
    robot.deaths += 1;
    robot.hp = 0;
    robot.heat = 0;
    robot.respawnMode = "reading";
    robot.respawnProgress = 0;
    robot.respawnRequired = reviveReadRequired(state, robot);
    robot.respawnAt = state.second + robot.respawnRequired;
    robot.status = `战亡读条 0/${robot.respawnRequired}`;
    robot.targetKey = null;
    if (attacker) {
      attacker.kills += 1;
      state.stats[attacker.side].kills += 1;
      event(state, attacker.side, "kill", `${attacker.role} 击毁 ${robot.role}`);
    }
    const cost = immediateReviveCost(state, robot);
    if (shouldBuyback(state, robot, cost)) buybackRobot(state, robot, cost);
    else event(state, robot.side, "respawn_decision", `${robot.role} 选择原地读条，需 ${robot.respawnRequired} 点进度`);
  }

  function timedRespawn(state, robot) {
    if (robot.role === "空中") return;
    const rules = state.model.rules.respawn;
    robot.maxHp = roleMaxHp(robot.profile, state.second, robot.heroArchetype, state.model.rules, robot.level);
    robot.hp = robot.maxHp * Number(rules.timed_hp_ratio || 0.1);
    robot.respawnAt = null;
    robot.respawnMode = "timed";
    robot.respawnedAt = state.second;
    robot.invulnerableUntil = state.second + Number(rules.timed_invulnerable_seconds || 30);
    robot.weak = true;
    robot.weakKind = "timed";
    robot.status = "原地复活 · 10%血量 · 虚弱";
    robot.nextDecisionAt = state.second;
    event(state, robot.side, "respawn", `${robot.role} 完成读条并原地复活，恢复 10% 血量`);
  }

  function respawnRobots(state) {
    state.robots.forEach((robot) => {
      if (robot.role === "空中") return;
      if (robot.hp > 0) {
        if (robot.weakKind === "buyback" && state.second >= robot.weakUntil) {
          robot.weak = false;
          robot.weakKind = null;
          robot.respawnMode = null;
          robot.status = "买活保护结束";
          robot.nextDecisionAt = state.second;
        }
        return;
      }
      if (robot.respawnMode !== "reading") return;
      const rules = state.model.rules.respawn;
      const supply = state.model.service_zones?.[robot.side]?.supply;
      const fast = insideZone(robot.position, supply)
        || state.structures[robot.side].base.hp < Number(rules.fast_base_hp_below || 2000);
      const progress = fast ? Number(rules.fast_progress_per_second || 4) : Number(rules.normal_progress_per_second || 1);
      robot.respawnProgress = Math.min(robot.respawnRequired, robot.respawnProgress + progress);
      const remaining = Math.max(0, robot.respawnRequired - robot.respawnProgress);
      robot.respawnAt = state.second + Math.ceil(remaining / progress);
      robot.status = `战亡读条 ${robot.respawnProgress}/${robot.respawnRequired}${fast ? " · 4×" : ""}`;
      if (robot.respawnProgress >= robot.respawnRequired) timedRespawn(state, robot);
    });
  }

  function shouldBuyRadarCounterOut(state, target, cost, forced) {
    const team = state.teamState[target.side];
    if (team.coins < cost) return false;
    if (forced === true) return true;
    if (forced === false) return false;
    const policySetting = state.options.radarBuyoutPolicy;
    const policy = (typeof policySetting === "object" ? policySetting?.[target.side] : policySetting) || "auto";
    if (policy === "always") return true;
    if (policy === "never") return false;
    const remaining = state.duration - state.second;
    const ammunitionValue = clamp((target.shotBudget - target.shots) / Math.max(1, target.shotBudget), 0, 1);
    return remaining > 45 && team.coins - cost >= 20
      && state.random() < clamp(0.25 + ammunitionValue * 0.55 - cost / Math.max(800, team.coins) * 0.22, 0.08, 0.8);
  }

  function applyRadarCounter(state, attackingSide, forcedBuyout) {
    const targetSide = otherSide(attackingSide);
    const target = state.robots.find((robot) => robot.side === targetSide && robot.role === "空中");
    const rules = state.model.rules.radar_uav_counter;
    if (!target || target.uavFlightState !== "airborne" || !target.uavSupportActive
      || target.radarCounterCount >= Number(rules.max_uses || 5)) return null;
    target.radarCounterCount += 1;
    target.radarCounteredUntil = state.second + Number(rules.lock_seconds || 45);
    target.status = `雷达反制 #${target.radarCounterCount} · ${rules.lock_seconds}s`;
    state.stats[attackingSide].radarCounters += 1;
    event(state, attackingSide, "radar", `第 ${target.radarCounterCount} 次反制${state.codes[targetSide]}空中，锁定 ${rules.lock_seconds} 秒`);

    let buyoutCost = 0;
    let boughtOut = false;
    if (target.radarCounterCount >= Number(rules.buyout_from_use || 4)) {
      buyoutCost = immediateReviveCost(state, target) * Number(rules.buyout_cost_multiplier || 2);
      if (shouldBuyRadarCounterOut(state, target, buyoutCost, forcedBuyout)) {
        const team = state.teamState[target.side];
        team.coins -= buyoutCost;
        team.spent += buyoutCost;
        target.radarCounteredUntil = state.second;
        target.radarCounterBuyouts += 1;
        target.uavSupportActive = true;
        uavStatus(state, target, `解除第 ${target.radarCounterCount} 次反制`);
        state.stats[target.side].uavCounterBuyouts += 1;
        event(state, target.side, "radar_buyout", `空中花费 ${buyoutCost} 金币解除反制并立即恢复支援`);
        boughtOut = true;
      }
    }
    return { target, count: target.radarCounterCount, buyoutCost, boughtOut };
  }

  function radarCounterUavs(state) {
    SIDES.forEach((side) => {
      const target = state.robots.find((robot) => robot.side === otherSide(side) && robot.role === "空中");
      if (!target || target.uavFlightState !== "airborne" || !target.uavSupportActive
        || target.ammo <= 0 || target.shots >= target.shotBudget) return;
      if (state.second < target.radarCounteredUntil) return;
      const expected = Number(state.model.teams[state.schools[side]].radar_counters_per_game || 0);
      const perSecond = clamp(expected + 0.05, 0.05, Number(state.model.rules.radar_uav_counter.max_uses || 5)) / state.duration;
      if (state.random() < perSecond) applyRadarCounter(state, side);
    });
  }

  function dartStrike(state) {
    if (!state.second || state.second % 90) return;
    SIDES.forEach((side) => {
      const teamProfile = state.model.teams[state.schools[side]];
      const enemy = otherSide(side);
      state.teamState[side].dartWindows += 1;
      const perWindow = clamp(Number(teamProfile.dart_hits_per_game || 0) / 4, 0, 0.95);
      event(state, side, "dart", "飞镖闸门开启");
      if (state.random() >= perWindow) return;
      const target = state.structures[enemy].outpost.hp > 0 ? state.structures[enemy].outpost : state.structures[enemy].base;
      let nominal;
      let dartMode = null;
      if (target.kind === "outpost") {
        nominal = Number(state.model.rules.damage.outpost_dart || 750);
      } else {
        const modes = teamProfile.dart_base_modes || Object.entries(state.model.rules.dart_base_damage_modes || {})
          .map(([mode, damage]) => ({ mode, damage, weight: 1 }));
        dartMode = weightedItem(modes.map((mode) => [mode, Number(mode.weight || 1)]), 1, state.random)?.[0]
          || { mode: "fixed", damage: 200 };
        nominal = Number(dartMode.damage);
      }
      const actual = Math.min(target.hp, nominal);
      target.hp -= actual;
      state.teamState[side].dartHits += 1;
      state.stats[side].damage += actual;
      state.stats[side][target.kind === "base" ? "baseDamage" : "outpostDamage"] += actual;
      if (target.kind === "base") {
        if (["random_moving", "terminal_moving"].includes(dartMode.mode)) {
          openBaseArmor(state, enemy, `飞镖${dartMode.mode === "terminal_moving" ? "末端移动目标" : "随机移动目标"}命中`);
        } else {
          target.fixedDartHits += 1;
          if (target.fixedDartHits >= 4) openBaseArmor(state, enemy, "固定类目标4发全部命中");
        }
      }
      const residual = actual === nominal ? "" : `（实际扣除 ${actual}）`;
      event(state, side, "dart", `飞镖命中${target.kind === "base" ? "基地" : "前哨站"}，规则伤害 ${nominal}${residual}`);
      if (target.hp <= 0) event(state, side, "destroy", `飞镖击毁${target.kind === "base" ? "基地" : "前哨站"}`);
    });
  }

  function separateRobots(state) {
    for (let left = 0; left < state.robots.length; left += 1) {
      const one = state.robots[left];
      if (one.hp <= 0 || one.role === "空中") continue;
      for (let right = left + 1; right < state.robots.length; right += 1) {
        const two = state.robots[right];
        if (two.hp <= 0 || two.role === "空中") continue;
        const distance = state.router.distance(one.position, two.position);
        if (distance >= 0.42 || distance < 1e-5) continue;
        const ux = (one.position[0] - two.position[0]) / distance;
        const uy = (one.position[1] - two.position[1]) / distance;
        const push = (0.42 - distance) / 2;
        const oneCandidate = [clamp(one.position[0] + ux * push, 0, 28), clamp(one.position[1] + uy * push, 0, 15)];
        const twoCandidate = [clamp(two.position[0] - ux * push, 0, 28), clamp(two.position[1] - uy * push, 0, 15)];
        const oneLayer = state.router.regionAt(state.navigation, one.position)?.id || "ground";
        const twoLayer = state.router.regionAt(state.navigation, two.position)?.id || "ground";
        const oneCandidateLayer = state.router.regionAt(state.navigation, oneCandidate)?.id || "ground";
        const twoCandidateLayer = state.router.regionAt(state.navigation, twoCandidate)?.id || "ground";
        if (oneLayer === oneCandidateLayer && !crossesForbiddenSymmetricGate(state, one, one.position, oneCandidate)) one.position = oneCandidate;
        if (twoLayer === twoCandidateLayer && !crossesForbiddenSymmetricGate(state, two, two.position, twoCandidate)) two.position = twoCandidate;
      }
    }
  }

  function finishMatch(state) {
    if (state.finished) return;
    const redBase = state.structures.red.base.hp;
    const blueBase = state.structures.blue.base.hp;
    if (redBase <= 0 && blueBase > 0) { state.winner = "blue"; state.reason = "红方基地被击毁"; }
    else if (blueBase <= 0 && redBase > 0) { state.winner = "red"; state.reason = "蓝方基地被击毁"; }
    else if (redBase !== blueBase) { state.winner = redBase > blueBase ? "red" : "blue"; state.reason = "比赛结束时基地血量领先"; }
    else { state.winner = "draw"; state.reason = "基地剩余血量相同"; }
    state.finished = true;
    event(state, "system", "finish", `${state.winner === "draw" ? "平局" : `${state.codes[state.winner]} 胜`}：${state.reason}`);
  }

  function stepMatch(state) {
    if (state.finished) return state;
    state.second += 1;
    SIDES.forEach((side) => {
      const campaign = state.teamState[side];
      if (!campaign.outpostAssaultAnnounced
        && campaign.outpostAssaultCount > 0
        && state.second >= campaign.outpostAssaultStartSecond) {
        campaign.outpostAssaultAnnounced = true;
        event(state, side, "objective", `${campaign.outpostAssaultCount} 台战斗机器人执行前哨快攻任务`);
      }
    });
    updateEconomy(state);
    respawnRobots(state);
    const groundFrameStarts = new Map(
      state.robots
        .filter((robot) => robot.role !== "空中" && robot.hp > 0)
        .map((robot) => [robot.key, [...robot.position]]),
    );
    updateUavSupport(state);
    moveRobots(state);
    separateRobots(state);
    // The UI interpolates between per-second snapshots.  Although each route
    // leg and separation push is legal on its own, their combined start/end
    // chord can cut through a wall corner.  Validate the final visible chord.
    enforceFrameWallClearance(state, groundFrameStarts);
    resupplyRobots(state);
    updateTechnologyCores(state);
    updateAssemblyProtection(state);
    resolveFortresses(state);
    radarCounterUavs(state);
    fireWeapons(state);
    dartStrike(state);
    if (state.structures.red.base.hp <= 0 || state.structures.blue.base.hp <= 0 || state.second >= state.duration) finishMatch(state);
    return state;
  }

  function snapshot(state) {
    return {
      second: state.second,
      structures: {
        red: {
          base: state.structures.red.base.hp,
          baseMax: state.structures.red.base.maxHp,
          baseArmorOpen: state.structures.red.base.armorOpen,
          baseArmorOpenedBy: state.structures.red.base.armorOpenedBy,
          fortressCaptureSeconds: state.structures.red.base.fortressCaptureSeconds,
          outpost: state.structures.red.outpost.hp,
          outpostMax: state.structures.red.outpost.maxHp,
        },
        blue: {
          base: state.structures.blue.base.hp,
          baseMax: state.structures.blue.base.maxHp,
          baseArmorOpen: state.structures.blue.base.armorOpen,
          baseArmorOpenedBy: state.structures.blue.base.armorOpenedBy,
          fortressCaptureSeconds: state.structures.blue.base.fortressCaptureSeconds,
          outpost: state.structures.blue.outpost.hp,
          outpostMax: state.structures.blue.outpost.maxHp,
        },
      },
      teams: {
        red: {
          coins: state.teamState.red.coins,
          totalCoins: state.teamState.red.totalCoins,
          fortress: state.teamState.red.fortress,
          technologyCoreLevel: state.teamState.red.technologyCore.level,
          technologyCoreIncomePer10: state.teamState.red.technologyCore.incomePer10,
          technologyCoreEarnedCoins: state.teamState.red.technologyCore.earnedCoins,
        },
        blue: {
          coins: state.teamState.blue.coins,
          totalCoins: state.teamState.blue.totalCoins,
          fortress: state.teamState.blue.fortress,
          technologyCoreLevel: state.teamState.blue.technologyCore.level,
          technologyCoreIncomePer10: state.teamState.blue.technologyCore.incomePer10,
          technologyCoreEarnedCoins: state.teamState.blue.technologyCore.earnedCoins,
        },
      },
      robots: state.robots.map((robot) => {
        const core = state.teamState[robot.side].technologyCore;
        const nextCore = nextTechnologyCoreTask(state, robot.side);
        return {
          key: robot.key, id: robot.id, side: robot.side, role: robot.role,
          x: robot.position[0], y: robot.position[1], yaw: robot.yaw,
          hp: robot.hp, maxHp: robot.maxHp, ammo: robot.ammo, heat: robot.heat,
          level: robot.level, heroArchetype: robot.heroArchetype,
          sampledWeaponAccuracy: Number(state.teamState[robot.side].weaponAccuracy?.[robot.profile.weapon] || 0),
          damagePerHitByTarget: robot.role === "英雄" ? robot.profile.damage_per_hit_by_target : null,
          shots: robot.shots, hits: robot.hits, kills: robot.kills, deaths: robot.deaths,
          weak: robot.weak, invulnerable: state.second < robot.invulnerableUntil || robot.assemblyProtected,
          assemblyProtected: robot.assemblyProtected,
          assemblyInvulnerableSeconds: robot.assemblyInvulnerableSeconds,
          assemblyInvulnerableRemaining: Math.max(0, Number(state.model.rules.engineer_assembly_invulnerability_seconds || 180) - robot.assemblyInvulnerableSeconds),
          respawnIn: robot.respawnAt == null ? 0 : Math.max(0, robot.respawnAt - state.second),
          respawnMode: robot.respawnMode, respawnProgress: robot.respawnProgress,
          respawnRequired: robot.respawnRequired, buybacks: robot.buybacks,
          radarCounterCount: robot.radarCounterCount,
          radarCounteredIn: Math.max(0, robot.radarCounteredUntil - state.second),
          radarCounterBuyouts: robot.radarCounterBuyouts,
          uavFlightState: robot.uavFlightState,
          uavSupportActive: robot.uavSupportActive,
          uavSupportSeconds: robot.uavSupportSeconds,
          uavPaidSupportSeconds: robot.uavPaidSupportSeconds,
          uavRadarWeaponLocked: robot.role === "空中" && state.second < robot.radarCounteredUntil,
          technologyCoreLevel: core.level,
          technologyCoreIncomePer10: core.incomePer10,
          technologyCoreEarnedCoins: core.earnedCoins,
          technologyCoreNextLevel: nextCore?.level || null,
          technologyCorePlannedIn: nextCore ? Math.max(0, technologyCoreReadySecond(state, robot.side, nextCore) - state.second) : null,
          status: robot.status, targetKey: robot.targetKey, objectiveKey: robot.objectiveKey,
          terrainAction: robot.terrainAction,
          terrainSpeedMultiplier: robot.terrainSpeedMultiplier,
          serviceZone: robot.role === "空中" ? "" : (serviceZoneAt(state, robot, "heal") || serviceZoneAt(state, robot, "ammo"))?.[1]?.label || "",
          goal: [...robot.goal], route: robot.route.map((point) => [...point]), passages: [...robot.passages],
        };
      }),
      stats: JSON.parse(JSON.stringify(state.stats)),
      finished: state.finished,
      winner: state.winner,
      reason: state.reason,
    };
  }

  function runMatch(model, navigation, redSchool, blueSchool, seed, routerOverride, matchOptions) {
    const state = createMatch(model, navigation, redSchool, blueSchool, seed, routerOverride, matchOptions);
    const frames = [snapshot(state)];
    while (!state.finished) {
      stepMatch(state);
      frames.push(snapshot(state));
    }
    return { state, frames, events: state.events };
  }

  return {
    ROLE_ORDER, hashSeed, mulberry32, canonicalPoint, robotLevel, roleMaxHp,
    insideZone, serviceZoneAt, reviveReadRequired, immediateReviveCost,
    createMatch, stepMatch, chooseGoal, tacticalCanonicalGoal, targetCandidates,
    moveRobots, resupplyRobots, killRobot, respawnRobots,
    canShelterInAssembly, serviceRequiredForDecision,
    applyRadarCounter, radarCounterUavs, updateUavSupport, updateTechnologyCores,
    updateAssemblyProtection, lineOfSight, snapshot, runMatch,
    openBaseArmor, resolveFortresses, dartStrike, applyDamage,
    crossesStaticWall, crossesForbiddenTerrainGate, enforceFrameWallClearance,
  };
});
