(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.RMUCMatchSimulator = api;
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", api.initPage, { once: true });
    } else {
      api.initPage();
    }
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const DEFAULT_RED = "东北大学";
  const DEFAULT_BLUE = "中国石油大学（华东）";
  const BASE_HP = 5000;
  const OUTPOST_HP = 1500;
  const AUTO_INTERVAL_MS = 360;
  const DAMAGE_KEYS = ["base", "outpost", "mobile", "mm17", "mm42", "dart"];

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function hashSeed(value) {
    const text = String(value);
    let hash = 2166136261;
    for (let i = 0; i < text.length; i += 1) {
      hash ^= text.charCodeAt(i);
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

  function indexesFor(model) {
    const indexes = {};
    model.bin_schema.forEach((name, index) => { indexes[name] = index; });
    return indexes;
  }

  function emptyDamage() {
    return { base: 0, outpost: 0, mobile: 0, mm17: 0, mm42: 0, dart: 0 };
  }

  function phaseSignal(team, turn, random, indexes) {
    const games = team.games || [];
    const width = Object.keys(indexes).length;
    if (!games.length) return new Array(width).fill(0);
    const sampled = games[Math.floor(random() * games.length)];
    const sampledBin = sampled.bins[turn] || new Array(width).fill(0);
    const mean = new Array(width).fill(0);
    let count = 0;
    games.forEach((game) => {
      if (!game.bins[turn]) return;
      count += 1;
      game.bins[turn].forEach((value, index) => { mean[index] += Number(value) || 0; });
    });
    if (count) mean.forEach((value, index) => { mean[index] = value / count; });

    const pace = 0.72 + random() * 0.58;
    const result = sampledBin.map((value, index) => {
      const blended = (Number(value) || 0) * 0.7 + mean[index] * 0.3;
      return blended * pace * (0.88 + random() * 0.24);
    });
    return result;
  }

  function fortressOwner(homeSignal, visitorSignal, homeSide, visitorSide, indexes) {
    const home = homeSignal[indexes.own_fortress_seconds] || 0;
    const visitor = visitorSignal[indexes.enemy_fortress_seconds] || 0;
    if (Math.max(home, visitor) < 1.2) return "neutral";
    if (Math.abs(home - visitor) < 2.2) return "contested";
    return home > visitor ? homeSide : visitorSide;
  }

  function resolveFortresses(redSignal, blueSignal, indexes) {
    return {
      red: fortressOwner(redSignal, blueSignal, "red", "blue", indexes),
      blue: fortressOwner(blueSignal, redSignal, "blue", "red", indexes),
    };
  }

  function defenseModifier(model, defender) {
    const globalMean = Number(model.global.mean_received_damage_per_game) || 1;
    const received = Number(defender.aggregate.received_per_game) || globalMean;
    return clamp(Math.pow(received / globalMean, 0.3), 0.8, 1.2);
  }

  function categoryShares(signal, aggregate, indexes) {
    const values = [
      signal[indexes.damage_17mm] || 0,
      signal[indexes.damage_42mm] || 0,
      signal[indexes.damage_dart] || 0,
    ];
    let total = values.reduce((sum, value) => sum + value, 0);
    if (total <= 0) {
      const history = aggregate.damage_by_category || {};
      values[0] = Number(history["17mm"]) || 1;
      values[1] = Number(history["42mm"]) || 0;
      values[2] = Number(history["飞镖"]) || 0;
      total = values.reduce((sum, value) => sum + value, 0);
    }
    return values.map((value) => value / Math.max(total, 1));
  }

  function applyAttack(state, side, signal, fortresses, model, indexes) {
    const opponent = side === "red" ? "blue" : "red";
    const attacker = model.teams[state.teams[side]];
    const defender = model.teams[state.teams[opponent]];
    const opponentStructures = state.structures[opponent];
    const defense = defenseModifier(model, defender);
    const holdsEnemyFortress = fortresses[opponent] === side;
    const buffWindows = signal[indexes.buff_windows] || 0;
    const boost = defense
      * (1 + (holdsEnemyFortress ? 0.06 : 0))
      * (1 + Math.min(buffWindows * 0.02, 0.08))
      * (opponentStructures.outpost <= 0 ? 1.08 : 1);

    const proposed = {
      base: Math.max(0, (signal[indexes.base_damage] || 0) * boost),
      outpost: Math.max(0, (signal[indexes.outpost_damage] || 0) * boost),
      mobile: Math.max(0, (signal[indexes.mobile_damage] || 0) * boost),
    };
    const actual = {
      base: Math.min(opponentStructures.base, proposed.base),
      outpost: Math.min(opponentStructures.outpost, proposed.outpost),
      mobile: proposed.mobile,
    };
    opponentStructures.base = Math.max(0, opponentStructures.base - actual.base);
    opponentStructures.outpost = Math.max(0, opponentStructures.outpost - actual.outpost);

    const total = actual.base + actual.outpost + actual.mobile;
    const shares = categoryShares(signal, attacker.aggregate, indexes);
    const damage = state.damage[side];
    damage.base += actual.base;
    damage.outpost += actual.outpost;
    damage.mobile += actual.mobile;
    damage.mm17 += total * shares[0];
    damage.mm42 += total * shares[1];
    damage.dart += total * shares[2];

    const utility = state.utility[side];
    utility.dartHits += signal[indexes.dart_hits] || 0;
    utility.dartGates += signal[indexes.dart_gate_opens] || 0;
    utility.buffs += buffWindows;
    utility.terrain += signal[indexes.terrain_actions] || 0;
    utility.ownFortressSeconds += signal[indexes.own_fortress_seconds] || 0;
    utility.enemyFortressSeconds += signal[indexes.enemy_fortress_seconds] || 0;
    if (fortresses[opponent] === side) utility.enemyFortressTurns += 1;
    if (fortresses[side] === side) utility.ownFortressTurns += 1;
    return actual;
  }

  function chooseAction(signal, actual, indexes) {
    const dart = signal[indexes.damage_dart] || 0;
    const enemyFortress = signal[indexes.enemy_fortress_seconds] || 0;
    const ownFortress = signal[indexes.own_fortress_seconds] || 0;
    const terrain = signal[indexes.terrain_actions] || 0;
    const buff = signal[indexes.buff_windows] || 0;
    if (dart >= 80 && actual.base >= 45) return "飞镖窗口 · 磨基地";
    if (actual.base >= 70 && actual.base >= actual.outpost) return "持续磨基地";
    if (actual.outpost >= 70) return "集火拆前哨";
    if (enemyFortress >= 4) return "压进敌方堡垒";
    if (actual.mobile >= 100) return "正面换血";
    if (ownFortress >= 4) return "驻守己方堡垒";
    if (buff >= 0.4) return "争夺能量机关";
    if (terrain >= 0.4) return "高地 / 飞坡转点";
    return "运营转点";
  }

  function tacticalScore(state, side) {
    const damage = state.damage[side];
    const utility = state.utility[side];
    return 55 * clamp(damage.base / BASE_HP, 0, 1)
      + 18 * clamp(damage.outpost / OUTPOST_HP, 0, 1)
      + 12 * clamp(damage.mobile / 5000, 0, 1)
      + 8 * clamp(utility.enemyFortressTurns / Math.max(state.maxTurns * 0.35, 1), 0, 1)
      + 4 * clamp(utility.dartHits / 4, 0, 1)
      + 3 * clamp(utility.buffs / 3, 0, 1);
  }

  function finalizeMatch(state) {
    if (state.finished) return state;
    const redBase = state.structures.red.base;
    const blueBase = state.structures.blue.base;
    const redScore = tacticalScore(state, "red");
    const blueScore = tacticalScore(state, "blue");
    let winner = "draw";
    let reason = "战术评分接近";
    if (redBase <= 0 && blueBase > 0) {
      winner = "blue";
      reason = "红方基地被击毁";
    } else if (blueBase <= 0 && redBase > 0) {
      winner = "red";
      reason = "蓝方基地被击毁";
    } else if (Math.abs(redScore - blueScore) >= 0.25) {
      winner = redScore > blueScore ? "red" : "blue";
      reason = state.turn >= state.maxTurns ? "420 秒战术评分领先" : "双方基地同时失守，战术评分领先";
    }
    state.finished = true;
    state.outcome = { winner, reason, redScore, blueScore };
    state.logs.unshift({
      time: Math.max(0, state.maxTurns * state.binSeconds - state.turn * state.binSeconds),
      side: "system",
      code: "终局",
      text: winner === "draw" ? `平局：${reason}` : `${state.codes[winner]} 胜：${reason}`,
    });
    return state;
  }

  function createMatch(model, redTeam, blueTeam, seed) {
    if (!model || !model.teams || !model.teams[redTeam] || !model.teams[blueTeam]) {
      throw new Error("未知战队，无法创建模拟对局");
    }
    const seedValue = Number.isFinite(Number(seed)) ? Number(seed) >>> 0 : hashSeed(seed);
    return {
      teams: { red: redTeam, blue: blueTeam },
      codes: { red: model.teams[redTeam].team || redTeam, blue: model.teams[blueTeam].team || blueTeam },
      seed: seedValue,
      random: mulberry32(seedValue),
      turn: 0,
      maxTurns: Number(model.bin_count) || 28,
      binSeconds: Number(model.bin_seconds) || 15,
      structures: {
        red: { base: BASE_HP, outpost: OUTPOST_HP },
        blue: { base: BASE_HP, outpost: OUTPOST_HP },
      },
      damage: { red: emptyDamage(), blue: emptyDamage() },
      utility: {
        red: { dartHits: 0, dartGates: 0, buffs: 0, terrain: 0, ownFortressSeconds: 0, enemyFortressSeconds: 0, ownFortressTurns: 0, enemyFortressTurns: 0 },
        blue: { dartHits: 0, dartGates: 0, buffs: 0, terrain: 0, ownFortressSeconds: 0, enemyFortressSeconds: 0, ownFortressTurns: 0, enemyFortressTurns: 0 },
      },
      fortresses: { red: "neutral", blue: "neutral" },
      actions: { red: "部署", blue: "部署" },
      logs: [],
      finished: false,
      outcome: null,
    };
  }

  function stepMatch(model, state) {
    if (state.finished) return state;
    const indexes = indexesFor(model);
    const redSignal = phaseSignal(model.teams[state.teams.red], state.turn, state.random, indexes);
    const blueSignal = phaseSignal(model.teams[state.teams.blue], state.turn, state.random, indexes);
    const fortresses = resolveFortresses(redSignal, blueSignal, indexes);
    const redOutpostBefore = state.structures.blue.outpost;
    const blueOutpostBefore = state.structures.red.outpost;
    const redBaseBefore = state.structures.blue.base;
    const blueBaseBefore = state.structures.red.base;
    const redActual = applyAttack(state, "red", redSignal, fortresses, model, indexes);
    const blueActual = applyAttack(state, "blue", blueSignal, fortresses, model, indexes);
    state.fortresses = fortresses;
    state.actions.red = chooseAction(redSignal, redActual, indexes);
    state.actions.blue = chooseAction(blueSignal, blueActual, indexes);
    state.turn += 1;
    const remaining = Math.max(0, state.maxTurns * state.binSeconds - state.turn * state.binSeconds);

    function addTurnLog(side, actual, signal) {
      const pieces = [];
      if (actual.base >= 1) pieces.push(`基地 ${Math.round(actual.base)}`);
      if (actual.outpost >= 1) pieces.push(`前哨 ${Math.round(actual.outpost)}`);
      if (actual.mobile >= 1) pieces.push(`机器人 ${Math.round(actual.mobile)}`);
      if ((signal[indexes.dart_hits] || 0) >= 0.35) pieces.push("飞镖命中窗口");
      state.logs.unshift({
        time: remaining,
        side,
        code: state.codes[side],
        text: `${state.actions[side]}${pieces.length ? `｜${pieces.join(" · ")}` : ""}`,
      });
    }
    addTurnLog("blue", blueActual, blueSignal);
    addTurnLog("red", redActual, redSignal);

    if (redOutpostBefore > 0 && state.structures.blue.outpost <= 0) {
      state.logs.unshift({ time: remaining, side: "red", code: state.codes.red, text: "击毁蓝方前哨站，基地压力上升" });
    }
    if (blueOutpostBefore > 0 && state.structures.red.outpost <= 0) {
      state.logs.unshift({ time: remaining, side: "blue", code: state.codes.blue, text: "击毁红方前哨站，基地压力上升" });
    }
    if (redBaseBefore > 0 && state.structures.blue.base <= 0) {
      state.logs.unshift({ time: remaining, side: "red", code: state.codes.red, text: "击毁蓝方基地" });
    }
    if (blueBaseBefore > 0 && state.structures.red.base <= 0) {
      state.logs.unshift({ time: remaining, side: "blue", code: state.codes.blue, text: "击毁红方基地" });
    }
    if (state.structures.red.base <= 0 || state.structures.blue.base <= 0 || state.turn >= state.maxTurns) {
      finalizeMatch(state);
    }
    return state;
  }

  function runFullMatch(model, redTeam, blueTeam, seed) {
    const state = createMatch(model, redTeam, blueTeam, seed);
    while (!state.finished) stepMatch(model, state);
    return state;
  }

  function runMonteCarlo(model, redTeam, blueTeam, count, seed) {
    const games = Math.max(1, Math.floor(Number(count) || 500));
    const baseSeed = Number.isFinite(Number(seed)) ? Number(seed) >>> 0 : hashSeed(seed);
    const result = {
      games,
      redWins: 0,
      blueWins: 0,
      draws: 0,
      redBaseHp: 0,
      blueBaseHp: 0,
      redScore: 0,
      blueScore: 0,
    };
    for (let index = 0; index < games; index += 1) {
      const state = runFullMatch(model, redTeam, blueTeam, (baseSeed + Math.imul(index + 1, 2654435761)) >>> 0);
      if (state.outcome.winner === "red") result.redWins += 1;
      else if (state.outcome.winner === "blue") result.blueWins += 1;
      else result.draws += 1;
      result.redBaseHp += state.structures.red.base;
      result.blueBaseHp += state.structures.blue.base;
      result.redScore += state.outcome.redScore;
      result.blueScore += state.outcome.blueScore;
    }
    ["redBaseHp", "blueBaseHp", "redScore", "blueScore"].forEach((key) => {
      result[key] /= games;
    });
    return result;
  }

  function formatNumber(value) {
    return Math.round(Number(value) || 0).toLocaleString("zh-CN");
  }

  function formatClock(seconds) {
    const safe = Math.max(0, Math.round(seconds));
    return `${String(Math.floor(safe / 60)).padStart(2, "0")}:${String(safe % 60).padStart(2, "0")}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function initPage() {
    const panel = document.getElementById("simulator");
    if (!panel || panel.dataset.initialized === "true") return;
    panel.dataset.initialized = "true";
    const elements = {
      status: document.getElementById("sim-model-status"),
      redSelect: document.getElementById("sim-red-team"),
      blueSelect: document.getElementById("sim-blue-team"),
      reset: document.getElementById("sim-reset"),
      step: document.getElementById("sim-step"),
      auto: document.getElementById("sim-auto"),
      monteButton: document.getElementById("sim-monte-carlo"),
      turn: document.getElementById("sim-turn"),
      clock: document.getElementById("sim-clock"),
      state: document.getElementById("sim-state"),
      scoreboard: document.getElementById("sim-scoreboard"),
      fortresses: document.getElementById("sim-fortresses"),
      redAction: document.getElementById("sim-red-action"),
      blueAction: document.getElementById("sim-blue-action"),
      redProfile: document.getElementById("sim-red-profile"),
      blueProfile: document.getElementById("sim-blue-profile"),
      damage: document.getElementById("sim-damage-breakdown"),
      monte: document.getElementById("sim-monte-result"),
      log: document.getElementById("sim-log"),
      logCount: document.getElementById("sim-log-count"),
    };
    let model = null;
    let match = null;
    let autoTimer = null;
    let nextSeed = Date.now() >>> 0;
    let modelLoading = false;

    function stopAuto() {
      if (autoTimer) window.clearInterval(autoTimer);
      autoTimer = null;
      elements.auto.classList.remove("running");
      elements.auto.textContent = "▶ 自动演算";
    }

    function renderProfile(side) {
      const team = model.teams[match.teams[side]];
      const aggregate = team.aggregate;
      const category = aggregate.damage_by_category || {};
      const categoryTotal = (category["17mm"] || 0) + (category["42mm"] || 0) + (category["飞镖"] || 0) || 1;
      const target = side === "red" ? elements.redProfile : elements.blueProfile;
      target.innerHTML = `
        <div class="sim-profile-head"><span>${escapeHtml(team.team)} · ${escapeHtml(team.region)}</span><h3>${escapeHtml(team.school)}</h3></div>
        <div class="sim-profile-metrics">
          <div><span>历史局数 / 胜率</span><b>${aggregate.games} / ${(aggregate.win_rate * 100).toFixed(1)}%</b></div>
          <div><span>场均总伤害</span><b>${formatNumber(aggregate.damage_per_game)}</b></div>
          <div><span>场均磨基地</span><b>${formatNumber(aggregate.base_damage_per_game)}</b></div>
          <div><span>场均拆前哨</span><b>${formatNumber(aggregate.outpost_damage_per_game)}</b></div>
          <div><span>敌堡驻留 / 局</span><b>${Number(aggregate.fortress_enemy_seconds_per_game || 0).toFixed(1)}s</b></div>
          <div><span>飞镖命中 / 局</span><b>${Number(aggregate.dart_hits_per_game || 0).toFixed(2)}</b></div>
        </div>
        <p class="sim-style">${escapeHtml(aggregate.style || "常规阵地运营")}</p>
        <div class="sim-profile-foot">火力构成：17mm ${(category["17mm"] / categoryTotal * 100 || 0).toFixed(0)}%　42mm ${(category["42mm"] / categoryTotal * 100 || 0).toFixed(0)}%　飞镖 ${(category["飞镖"] / categoryTotal * 100 || 0).toFixed(0)}%</div>`;
    }

    function structureHtml(side, label, value, max) {
      const percentage = clamp(value / max * 100, 0, 100);
      return `<div class="sim-structure"><div class="sim-structure-line"><span>${label}</span><b>${formatNumber(value)} / ${formatNumber(max)}</b></div><div class="sim-health"><i style="width:${percentage.toFixed(2)}%"></i></div></div>`;
    }

    function fortressLabel(owner) {
      if (owner === "red") return "红方控制";
      if (owner === "blue") return "蓝方控制";
      if (owner === "contested") return "争夺中";
      return "无人占领";
    }

    function renderDamage() {
      const labels = [
        ["base", "基地"], ["outpost", "前哨"], ["mobile", "机器人"],
        ["mm17", "17mm"], ["mm42", "42mm"], ["dart", "飞镖"],
      ];
      const rows = labels.map(([key, label]) => {
        const red = match.damage.red[key];
        const blue = match.damage.blue[key];
        const max = Math.max(red, blue, 1);
        return `<div class="sim-damage-row"><span>${label}</span><div class="sim-mini-track"><i style="width:${red / max * 100}%"></i></div><b>${formatNumber(red)}</b><div class="sim-mini-track blue"><i style="width:${blue / max * 100}%"></i></div><b>${formatNumber(blue)}</b></div>`;
      }).join("");
      elements.damage.innerHTML = `<div class="sim-damage-header"><span>对象</span><span>红方</span><span>${escapeHtml(match.codes.red)}</span><span>蓝方</span><span>${escapeHtml(match.codes.blue)}</span></div>${rows}`;
    }

    function renderLog() {
      elements.logCount.textContent = `${match.logs.length} 条`;
      if (!match.logs.length) {
        elements.log.innerHTML = '<div class="empty">开局后显示双方战术动作</div>';
        return;
      }
      elements.log.innerHTML = match.logs.slice(0, 80).map((entry) => `
        <div class="sim-log-entry ${entry.side}"><time>${formatClock(entry.time)}</time><b>${escapeHtml(entry.code)}</b><span>${escapeHtml(entry.text)}</span></div>`).join("");
    }

    function renderMatch() {
      renderProfile("red");
      renderProfile("blue");
      elements.turn.textContent = `第 ${match.turn} / ${match.maxTurns} 手`;
      elements.clock.textContent = formatClock(match.maxTurns * match.binSeconds - match.turn * match.binSeconds);
      if (match.finished) {
        const winner = match.outcome.winner;
        elements.state.textContent = winner === "draw" ? "终局 · 平局" : `终局 · ${match.codes[winner]} 胜`;
        elements.state.className = "finished";
      } else {
        elements.state.textContent = match.turn ? "演算中" : "等待落子";
        elements.state.className = "";
      }
      elements.scoreboard.innerHTML = `
        <div class="sim-side-structures red"><div class="sim-side-title"><span>红方</span><b>${escapeHtml(match.codes.red)}</b></div>${structureHtml("red", "基地", match.structures.red.base, BASE_HP)}${structureHtml("red", "前哨", match.structures.red.outpost, OUTPOST_HP)}</div>
        <div class="sim-side-structures blue"><div class="sim-side-title"><span>蓝方</span><b>${escapeHtml(match.codes.blue)}</b></div>${structureHtml("blue", "基地", match.structures.blue.base, BASE_HP)}${structureHtml("blue", "前哨", match.structures.blue.outpost, OUTPOST_HP)}</div>`;
      elements.fortresses.innerHTML = `
        <div class="sim-fortress ${match.fortresses.red === "red" ? "red-held" : match.fortresses.red === "blue" ? "blue-held" : ""}"><span>红方堡垒</span><b>${fortressLabel(match.fortresses.red)}</b></div>
        <div class="sim-fortress ${match.fortresses.blue === "red" ? "red-held" : match.fortresses.blue === "blue" ? "blue-held" : ""}"><span>蓝方堡垒</span><b>${fortressLabel(match.fortresses.blue)}</b></div>`;
      elements.redAction.textContent = match.actions.red;
      elements.blueAction.textContent = match.actions.blue;
      elements.step.disabled = match.finished;
      renderDamage();
      renderLog();
      if (match.finished) stopAuto();
    }

    function resetMatch(seed) {
      stopAuto();
      nextSeed = seed === undefined ? (nextSeed + 1) >>> 0 : seed >>> 0;
      match = createMatch(model, elements.redSelect.value, elements.blueSelect.value, nextSeed);
      elements.monte.innerHTML = '<div class="sim-monte-empty">点“模拟 500 局”查看对阵分布</div>';
      renderMatch();
    }

    function stepOnce() {
      if (!match || match.finished) return;
      stepMatch(model, match);
      renderMatch();
    }

    function toggleAuto() {
      if (autoTimer) {
        stopAuto();
        return;
      }
      if (match.finished) resetMatch();
      elements.auto.classList.add("running");
      elements.auto.textContent = "Ⅱ 暂停演算";
      stepOnce();
      if (!match.finished) autoTimer = window.setInterval(stepOnce, AUTO_INTERVAL_MS);
    }

    function renderMonteCarlo(result) {
      const redRate = result.redWins / result.games * 100;
      const blueRate = result.blueWins / result.games * 100;
      const drawRate = result.draws / result.games * 100;
      elements.monte.innerHTML = `
        <div class="sim-monte-matchup">${escapeHtml(match.codes.red)} vs ${escapeHtml(match.codes.blue)} · ${result.games} 局</div>
        <div class="sim-monte-bars">
          <div class="sim-monte-row"><span>红方胜</span><div class="sim-monte-track"><i style="width:${redRate}%"></i></div><b>${redRate.toFixed(1)}%</b></div>
          <div class="sim-monte-row blue"><span>蓝方胜</span><div class="sim-monte-track"><i style="width:${blueRate}%"></i></div><b>${blueRate.toFixed(1)}%</b></div>
          <div class="sim-monte-row"><span>战术平局</span><div class="sim-monte-track"><i style="width:${drawRate}%;background:var(--gold)"></i></div><b>${drawRate.toFixed(1)}%</b></div>
        </div>
        <p class="sim-monte-summary">平均剩余基地血量：红 ${formatNumber(result.redBaseHp)} / 蓝 ${formatNumber(result.blueBaseHp)}<br>平均战术评分：红 ${result.redScore.toFixed(1)} / 蓝 ${result.blueScore.toFixed(1)}</p>`;
    }

    function simulateMany() {
      elements.monteButton.disabled = true;
      elements.monteButton.textContent = "推演中…";
      window.setTimeout(() => {
        const result = runMonteCarlo(model, elements.redSelect.value, elements.blueSelect.value, 500, nextSeed ^ 0x9e3779b9);
        renderMonteCarlo(result);
        elements.monteButton.disabled = false;
        elements.monteButton.textContent = "模拟 500 局";
      }, 20);
    }

    function populate(modelData) {
      model = modelData;
      const teams = Object.values(model.teams).sort((left, right) => {
        const stage = String(left.stage).localeCompare(String(right.stage), "zh-CN");
        return stage || String(left.team).localeCompare(String(right.team), "zh-CN");
      });
      const options = teams.map((team) => `<option value="${escapeHtml(team.school)}">${escapeHtml(team.team)} · ${escapeHtml(team.school)}</option>`).join("");
      elements.redSelect.innerHTML = options;
      elements.blueSelect.innerHTML = options;
      elements.redSelect.value = model.teams[DEFAULT_RED] ? DEFAULT_RED : teams[0].school;
      elements.blueSelect.value = model.teams[DEFAULT_BLUE] ? DEFAULT_BLUE : teams[Math.min(1, teams.length - 1)].school;
      elements.status.textContent = `${teams.length} 队 · ${model.bin_seconds}s / 手 · 数据就绪`;
      elements.status.classList.add("ready");
      elements.reset.disabled = false;
      elements.step.disabled = false;
      elements.auto.disabled = false;
      elements.monteButton.disabled = false;
      resetMatch(hashSeed(`${elements.redSelect.value}|${elements.blueSelect.value}|first`));
    }

    elements.reset.addEventListener("click", () => resetMatch());
    elements.step.addEventListener("click", stepOnce);
    elements.auto.addEventListener("click", toggleAuto);
    elements.monteButton.addEventListener("click", simulateMany);
    elements.redSelect.addEventListener("change", () => resetMatch());
    elements.blueSelect.addEventListener("change", () => resetMatch());

    function loadModel() {
      if (modelLoading || model) return;
      modelLoading = true;
      elements.status.textContent = "正在后台载入快速棋谱…";
      fetch("./data/models/match_simulation.json?v=1")
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then(populate)
        .catch((error) => {
          elements.status.textContent = `棋谱载入失败：${error.message}`;
          elements.status.classList.add("error");
        });
    }

    elements.reset.disabled = true;
    elements.step.disabled = true;
    elements.auto.disabled = true;
    elements.monteButton.disabled = true;
    elements.status.textContent = "快速棋谱已延迟加载 · 进入模拟区后启动";
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        loadModel();
      }, { rootMargin: "240px" });
      observer.observe(panel);
    } else {
      window.setTimeout(loadModel, 100);
    }
  }

  return {
    BASE_HP,
    OUTPOST_HP,
    DAMAGE_KEYS,
    hashSeed,
    mulberry32,
    createMatch,
    stepMatch,
    runFullMatch,
    runMonteCarlo,
    tacticalScore,
    initPage,
  };
});
