(function () {
  "use strict";

  if (typeof document === "undefined") return;
  const DEFAULT_RED = "东北大学";
  const DEFAULT_BLUE = "中国石油大学（华东）";
  const ROLE_LABEL = { 英雄: "1", 工程: "2", 步兵3: "3", 步兵4: "4", 哨兵: "AI", 空中: "6" };
  const HERO_ARCHETYPE_LABEL = { melee: "近战优先", ranged: "远程优先" };
  const ENGAGEMENT_LABEL = { long_range: "远程吊射", close_pressure: "近身压制", flexible: "灵活站位" };
  const UAV_FLIGHT_LABEL = { parked: "停机坪", airborne: "空中巡航", returning: "正在返航" };
  const COLORS = { red: "#ff526c", blue: "#48a0ff", gold: "#f3bd4d", green: "#38d39f" };
  const MAP_RATIO = 1125 / 2048;
  const FIELD_Y_SPAN = 15 / 17;
  const FIELD_Y_OFFSET = 1 / 17;
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

  function init() {
    const canvas = document.getElementById("full-sim-canvas");
    if (!canvas || canvas.dataset.initialized === "true") return;
    canvas.dataset.initialized = "true";
    const elements = {
      canvas,
      status: document.getElementById("full-sim-status"),
      redSelect: document.getElementById("full-red-team"),
      blueSelect: document.getElementById("full-blue-team"),
      redHeroMode: document.getElementById("full-red-hero-mode"),
      blueHeroMode: document.getElementById("full-blue-hero-mode"),
      create: document.getElementById("full-sim-new"),
      play: document.getElementById("full-sim-play"),
      back: document.getElementById("full-sim-back"),
      forward: document.getElementById("full-sim-forward"),
      speed: document.getElementById("full-sim-speed"),
      slider: document.getElementById("full-sim-slider"),
      clock: document.getElementById("full-sim-clock"),
      result: document.getElementById("full-sim-result"),
      redRoster: document.getElementById("full-red-roster"),
      blueRoster: document.getElementById("full-blue-roster"),
      stats: document.getElementById("full-sim-stats"),
      seed: document.getElementById("full-sim-seed"),
      selectedLabel: document.getElementById("full-selected-label"),
      detail: document.getElementById("full-robot-detail"),
      eventCount: document.getElementById("full-event-count"),
      events: document.getElementById("full-sim-events"),
    };
    const context = canvas.getContext("2d");
    const mapImage = new Image();
    mapImage.src = "./assets/map.webp";
    let model = null;
    let navigation = null;
    let simulation = null;
    let playhead = 0;
    let playing = false;
    let playRequested = false;
    let playbackSpeed = Number(elements.speed.value) || 1;
    let selectedKey = "red:英雄";
    let seed = Date.now() >>> 0;
    let lastAnimation = performance.now();
    let lastUiSecond = -1;
    let simulationWorker = null;
    let simulationRequestId = 0;
    let activeSimulationRequest = 0;
    let simulationDataLoading = false;
    let simulationComplete = false;
    let expectedSimulationFrames = 421;

    function updateHeroAutoLabel(side) {
      if (!model) return;
      const teamSelect = side === "red" ? elements.redSelect : elements.blueSelect;
      const heroSelect = side === "red" ? elements.redHeroMode : elements.blueHeroMode;
      const archetype = model.teams[teamSelect.value]?.roles?.["英雄"]?.hero_archetype_default || "ranged";
      const auto = heroSelect.querySelector('option[value="auto"]');
      if (auto) auto.textContent = `英雄：自动（${HERO_ARCHETYPE_LABEL[archetype]}）`;
    }

    function publishMatchup() {
      const detail = { red: elements.redSelect.value, blue: elements.blueSelect.value, source: "full" };
      window.RMUC_SIMULATOR_MATCHUP = { red: detail.red, blue: detail.blue };
      window.dispatchEvent(new CustomEvent("rmuc:simulator-matchup", { detail }));
    }

    function teamSelectionChanged() {
      updateHeroAutoLabel("red");
      updateHeroAutoLabel("blue");
      publishMatchup();
      generate();
    }

    function formatClock(second) {
      const remaining = Math.max(0, 420 - Math.round(second));
      return `${String(Math.floor(remaining / 60)).padStart(2, "0")}:${String(remaining % 60).padStart(2, "0")}`;
    }

    function formatNumber(value) {
      return Math.round(Number(value) || 0).toLocaleString("zh-CN");
    }

    function sideCode(side) {
      return simulation?.state.codes[side] || (side === "red" ? "红方" : "蓝方");
    }

    function mapPoint(x, y, width, height) {
      const u = clamp(x / 28, 0, 1);
      const fieldV = 1 - clamp(y / 15, 0, 1);
      return [u * width, (FIELD_Y_OFFSET + fieldV * FIELD_Y_SPAN) * height];
    }

    function worldPoint(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const u = clamp((clientX - rect.left) / rect.width, 0, 1);
      const textureV = clamp((clientY - rect.top) / rect.height, FIELD_Y_OFFSET, FIELD_Y_OFFSET + FIELD_Y_SPAN);
      const fieldV = (textureV - FIELD_Y_OFFSET) / FIELD_Y_SPAN;
      return [u * 28, (1 - fieldV) * 15];
    }

    function interpolatedFrame() {
      if (!simulation) return null;
      const index = clamp(Math.floor(playhead), 0, simulation.frames.length - 1);
      const afterIndex = Math.min(simulation.frames.length - 1, index + 1);
      const frame = simulation.frames[index];
      const after = simulation.frames[afterIndex];
      const alpha = clamp(playhead - index, 0, 1);
      const afterMap = new Map(after.robots.map((robot) => [robot.key, robot]));
      return {
        base: frame,
        robots: frame.robots.map((robot) => {
          const next = afterMap.get(robot.key) || robot;
          return { ...robot, x: robot.x + (next.x - robot.x) * alpha, y: robot.y + (next.y - robot.y) * alpha, yaw: robot.yaw + (next.yaw - robot.yaw) * alpha };
        }),
      };
    }

    function canvasSize() {
      const cssWidth = Math.max(320, canvas.clientWidth || 800);
      const cssHeight = cssWidth * MAP_RATIO;
      const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
      canvas.style.height = `${cssHeight}px`;
      if (canvas.width !== Math.round(cssWidth * pixelRatio) || canvas.height !== Math.round(cssHeight * pixelRatio)) {
        canvas.width = Math.round(cssWidth * pixelRatio);
        canvas.height = Math.round(cssHeight * pixelRatio);
      }
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      return { width: cssWidth, height: cssHeight, scale: clamp(cssWidth / 1200, 0.45, 1.4) };
    }

    function drawStructure(frame, side, kind, position, width, height, scale) {
      const hp = frame.structures[side][kind];
      const armorOpen = kind === "base" && frame.structures[side].baseArmorOpen;
      const max = kind === "base"
        ? Number(frame.structures[side].baseMax || model.rules.base_hp)
        : Number(frame.structures[side].outpostMax || model.rules.outpost_hp);
      const [x, y] = mapPoint(position[0], position[1], width, height);
      const radius = (kind === "base" ? 15 : 12) * scale;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fillStyle = hp > 0 ? "rgba(5,12,18,.92)" : "rgba(35,36,38,.88)";
      context.fill();
      context.strokeStyle = hp > 0 ? (armorOpen ? COLORS.gold : COLORS[side]) : "#687581";
      context.lineWidth = 2.4 * scale;
      context.stroke();
      context.fillStyle = "#fff";
      context.font = `900 ${Math.max(9, 10 * scale)}px sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(kind === "base" ? (armorOpen ? "基开" : "基闭") : "前", x, y);
      const barWidth = 42 * scale;
      const barY = y + radius + 4 * scale;
      context.fillStyle = "rgba(2,6,10,.9)";
      context.fillRect(x - barWidth / 2, barY, barWidth, 4 * scale);
      context.fillStyle = hp / max > 0.4 ? COLORS.green : COLORS.gold;
      context.fillRect(x - barWidth / 2, barY, barWidth * clamp(hp / max, 0, 1), 4 * scale);
    }

    function drawAssemblyZone(frame, side, width, height, scale) {
      const zone = model.assembly_zones?.[side];
      if (!zone) return;
      const [x, y] = mapPoint(zone.center[0], zone.center[1], width, height);
      const radiusX = zone.radius[0] * width / 28;
      const radiusY = zone.radius[1] * FIELD_Y_SPAN * height / 15;
      context.beginPath();
      context.ellipse(x, y, radiusX, radiusY, 0, 0, Math.PI * 2);
      context.fillStyle = side === "red" ? "rgba(255,82,108,.09)" : "rgba(72,160,255,.09)";
      context.fill();
      context.setLineDash([2 * scale, 3 * scale]);
      context.strokeStyle = COLORS[side];
      context.lineWidth = 1.5 * scale;
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = "#f4d98e";
      context.font = `800 ${Math.max(7, 8 * scale)}px sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "bottom";
      context.fillText(`科技核心 Lv.${frame.teams[side].technologyCoreLevel}`, x, y - radiusY - 2 * scale);
    }

    function drawServiceZone(side, name, zone, width, height, scale) {
      if (!zone) return;
      const [x, y] = mapPoint(zone.center[0], zone.center[1], width, height);
      const radiusX = zone.radius[0] * width / 28;
      const radiusY = zone.radius[1] * FIELD_Y_SPAN * height / 15;
      const isSupply = Boolean(zone.heal);
      context.beginPath();
      context.ellipse(x, y, radiusX, radiusY, 0, 0, Math.PI * 2);
      context.fillStyle = isSupply ? "rgba(56,211,159,.16)" : side === "red" ? "rgba(255,82,108,.11)" : "rgba(72,160,255,.11)";
      context.fill();
      context.setLineDash(isSupply ? [] : [4 * scale, 3 * scale]);
      context.strokeStyle = isSupply ? COLORS.green : COLORS[side];
      context.lineWidth = 1.5 * scale;
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = isSupply ? "#b8ffe9" : "#eef6fa";
      context.font = `800 ${Math.max(7, 8 * scale)}px sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "bottom";
      const label = isSupply ? "补血·补弹" : name === "base" ? "基地补弹" : "前哨下补弹";
      context.fillText(label, x, y - radiusY - 2 * scale);
    }

    function drawFortress(frame, side, width, height, scale) {
      const centre = model.structures[side].fortress;
      const [x, y] = mapPoint(centre[0], centre[1], width, height);
      const owner = frame.teams[side].fortress;
      context.beginPath();
      context.arc(x, y, 18 * scale, 0, Math.PI * 2);
      context.setLineDash([4 * scale, 3 * scale]);
      context.lineWidth = 2 * scale;
      context.strokeStyle = owner === "red" ? COLORS.red : owner === "blue" ? COLORS.blue : COLORS.gold;
      context.globalAlpha = owner === "neutral" ? 0.45 : 0.9;
      context.stroke();
      context.setLineDash([]);
      context.globalAlpha = 1;
    }

    function drawMap() {
      const size = canvasSize();
      const frameData = interpolatedFrame();
      context.clearRect(0, 0, size.width, size.height);
      if (mapImage.complete && mapImage.naturalWidth) context.drawImage(mapImage, 0, 0, size.width, size.height);
      else { context.fillStyle = "#080d12"; context.fillRect(0, 0, size.width, size.height); }
      if (!frameData) return;
      const frame = frameData.base;
      const robots = frameData.robots;
      ["red", "blue"].forEach((side) => {
        Object.entries(model.service_zones?.[side] || {}).forEach(([name, zone]) => {
          drawServiceZone(side, name, zone, size.width, size.height, size.scale);
        });
        drawAssemblyZone(frame, side, size.width, size.height, size.scale);
      });
      drawFortress(frame, "red", size.width, size.height, size.scale);
      drawFortress(frame, "blue", size.width, size.height, size.scale);
      ["red", "blue"].forEach((side) => {
        drawStructure(frame, side, "base", model.structures[side].base, size.width, size.height, size.scale);
        drawStructure(frame, side, "outpost", model.structures[side].outpost, size.width, size.height, size.scale);
      });

      const selected = robots.find((robot) => robot.key === selectedKey);
      if (selected?.route?.length > 1) {
        context.beginPath();
        selected.route.forEach((point, index) => {
          const [x, y] = mapPoint(point[0], point[1], size.width, size.height);
          if (index) context.lineTo(x, y); else context.moveTo(x, y);
        });
        context.setLineDash([6 * size.scale, 4 * size.scale]);
        context.strokeStyle = COLORS[selected.side];
        context.lineWidth = 2.2 * size.scale;
        context.stroke();
        context.setLineDash([]);
        const [gx, gy] = mapPoint(selected.goal[0], selected.goal[1], size.width, size.height);
        context.beginPath(); context.arc(gx, gy, 5 * size.scale, 0, Math.PI * 2); context.stroke();
      }

      const entityPositions = new Map(robots.map((robot) => [robot.key, [robot.x, robot.y]]));
      ["red", "blue"].forEach((side) => {
        entityPositions.set(`${side}:base`, model.structures[side].base);
        entityPositions.set(`${side}:outpost`, model.structures[side].outpost);
      });
      robots.forEach((robot) => {
        if (!robot.objectiveKey || robot.hp <= 0 || robot.targetKey === robot.objectiveKey) return;
        const objective = entityPositions.get(robot.objectiveKey);
        if (!objective) return;
        const [x1, y1] = mapPoint(robot.x, robot.y, size.width, size.height);
        const [x2, y2] = mapPoint(objective[0], objective[1], size.width, size.height);
        context.beginPath(); context.moveTo(x1, y1); context.lineTo(x2, y2);
        context.setLineDash([5 * size.scale, 5 * size.scale]);
        context.strokeStyle = "rgba(243,189,77,.46)"; context.lineWidth = 1.4 * size.scale; context.stroke();
        context.setLineDash([]);
      });
      robots.forEach((robot) => {
        if (!robot.targetKey || robot.hp <= 0) return;
        const target = entityPositions.get(robot.targetKey);
        if (!target) return;
        const [x1, y1] = mapPoint(robot.x, robot.y, size.width, size.height);
        const [x2, y2] = mapPoint(target[0], target[1], size.width, size.height);
        context.beginPath(); context.moveTo(x1, y1); context.lineTo(x2, y2);
        context.strokeStyle = "rgba(243,189,77,.72)"; context.lineWidth = 1.3 * size.scale; context.stroke();
      });

      robots.forEach((robot) => {
        const [x, y] = mapPoint(robot.x, robot.y, size.width, size.height);
        const radius = (robot.role === "空中" ? 11 : 10) * size.scale;
        if (robot.invulnerable || robot.weak) {
          context.beginPath(); context.arc(x, y, radius + 5 * size.scale, 0, Math.PI * 2);
          context.strokeStyle = robot.invulnerable ? COLORS.gold : COLORS.green;
          context.lineWidth = 2 * size.scale; context.stroke();
        }
        if (robot.radarCounteredIn > 0) {
          context.beginPath(); context.arc(x, y, radius + 10 * size.scale, 0, Math.PI * 2);
          context.setLineDash([3 * size.scale, 3 * size.scale]);
          context.strokeStyle = COLORS.gold; context.lineWidth = 2.2 * size.scale; context.stroke();
          context.setLineDash([]);
        }
        if (robot.key === selectedKey) {
          context.beginPath(); context.arc(x, y, radius + 8 * size.scale, 0, Math.PI * 2);
          context.strokeStyle = "#fff"; context.lineWidth = 2 * size.scale; context.stroke();
        }
        context.beginPath(); context.arc(x + 1.5, y + 2, radius + 1, 0, Math.PI * 2);
        context.fillStyle = "rgba(0,0,0,.55)"; context.fill();
        context.beginPath(); context.arc(x, y, radius, 0, Math.PI * 2);
        context.fillStyle = robot.hp > 0 ? COLORS[robot.side] : robot.side === "red" ? "#6d2632" : "#224e70";
        context.globalAlpha = robot.hp > 0 ? 1 : 0.58;
        context.fill(); context.globalAlpha = 1;
        context.strokeStyle = "#f7fbfd"; context.lineWidth = 1.1 * size.scale; context.stroke();
        context.fillStyle = "#fff"; context.font = `900 ${Math.max(8, 9 * size.scale)}px sans-serif`;
        context.textAlign = "center"; context.textBaseline = "middle";
        context.fillText(ROLE_LABEL[robot.role], x, y);
        const hpRatio = robot.role === "空中"
          ? (robot.uavSupportActive ? 1 : robot.uavFlightState === "returning" ? 0.55 : 0.2)
          : robot.maxHp ? robot.hp / robot.maxHp : 0;
        const barWidth = radius * 2.5;
        context.fillStyle = "rgba(1,5,8,.9)"; context.fillRect(x - barWidth / 2, y - radius - 8 * size.scale, barWidth, 3.5 * size.scale);
        context.fillStyle = robot.role === "空中" ? COLORS.gold : hpRatio > 0.4 ? COLORS.green : COLORS.gold;
        context.fillRect(x - barWidth / 2, y - radius - 8 * size.scale, barWidth * clamp(hpRatio, 0, 1), 3.5 * size.scale);
        if (robot.hp <= 0 && robot.respawnIn) {
          context.fillStyle = "#fff"; context.font = `800 ${Math.max(7, 8 * size.scale)}px sans-serif`;
          context.fillText(`${robot.respawnIn}s`, x, y + radius + 7 * size.scale);
        }
      });
    }

    function renderRoster(side, frame) {
      const target = side === "red" ? elements.redRoster : elements.blueRoster;
      const robots = frame.robots.filter((robot) => robot.side === side);
      const team = frame.teams[side];
      target.innerHTML = `<div class="full-roster-head"><div><b>${escapeHtml(sideCode(side))}</b><small>核心 Lv.${team.technologyCoreLevel} · +${formatNumber(team.technologyCoreIncomePer10)}/10s</small></div><span>金币 ${formatNumber(team.coins)}</span></div><div class="full-roster-list">${robots.map((robot) => `
        <div class="full-robot-row ${robot.hp <= 0 ? "dead" : ""} ${robot.key === selectedKey ? "selected" : ""}" data-full-robot="${robot.key}">
          <span class="full-role-token">${ROLE_LABEL[robot.role]}</span>
          <div class="full-robot-info"><b>${robot.role}${robot.heroArchetype ? ` · ${HERO_ARCHETYPE_LABEL[robot.heroArchetype]}` : robot.role === "工程" ? ` · 核心 Lv.${robot.technologyCoreLevel}` : ""}</b><span>${escapeHtml(robot.status)}</span></div>
          <span class="full-robot-hp">${robot.role === "空中" ? escapeHtml(UAV_FLIGHT_LABEL[robot.uavFlightState] || "空中") : `${formatNumber(robot.hp)}/${formatNumber(robot.maxHp)}`}<br>${robot.role === "空中" ? `支援 ${formatNumber(robot.uavSupportSeconds)}s · 弹 ${formatNumber(robot.ammo)}` : robot.respawnIn ? `复活 ${robot.respawnIn}s` : robot.role === "工程" ? `+${formatNumber(robot.technologyCoreIncomePer10)}/10s` : `弹 ${formatNumber(robot.ammo)}`}</span>
        </div>`).join("")}</div>`;
      target.querySelectorAll("[data-full-robot]").forEach((row) => row.addEventListener("click", () => {
        selectedKey = row.dataset.fullRobot;
        renderUi(true);
        drawMap();
      }));
    }

    function renderStats(frame) {
      const red = frame.stats.red;
      const blue = frame.stats.blue;
      const behaviorSummary = (side) => {
        const school = side === "red" ? elements.redSelect.value : elements.blueSelect.value;
        const profile = model.teams[school]?.behavior_profile;
        if (!profile) return "暂无队伍画像";
        const hero = profile.hero;
        const roles = profile.outpost.primary_roles.length ? profile.outpost.primary_roles.join("/") : "无固定兵种";
        return `${HERO_ARCHETYPE_LABEL[hero.archetype] || hero.archetype}·${ENGAGEMENT_LABEL[hero.engagement_style] || hero.engagement_style} ${Number(hero.preferred_range_m).toFixed(1)}m·命中 ${(hero.accuracy_42mm * 100).toFixed(1)}%·前哨 ${roles}·空中份额 ${(profile.outpost.uav_attributed_share * 100).toFixed(1)}%`;
      };
      elements.stats.innerHTML = `
        <div class="full-stat-score"><div><strong>${escapeHtml(sideCode("red"))}</strong><span>红方</span></div><b>${formatNumber(frame.structures.red.base)} : ${formatNumber(frame.structures.blue.base)}</b><div class="blue"><strong>${escapeHtml(sideCode("blue"))}</strong><span>蓝方</span></div></div>
        <div class="full-behavior-compare">
          <div class="red"><span>红方·44 队独立画像</span><b>${escapeHtml(behaviorSummary("red"))}</b></div>
          <div class="blue"><span>蓝方·44 队独立画像</span><b>${escapeHtml(behaviorSummary("blue"))}</b></div>
        </div>
        <div class="full-stat-table">
          <div class="full-stat-row"><span>累计伤害</span><b>${formatNumber(red.damage)}</b><b>${formatNumber(blue.damage)}</b></div>
          <div class="full-stat-row"><span>机器人伤害</span><b>${formatNumber(red.robotDamage)}</b><b>${formatNumber(blue.robotDamage)}</b></div>
          <div class="full-stat-row"><span>前哨 / 基地</span><b>${formatNumber(red.outpostDamage)} / ${formatNumber(red.baseDamage)}</b><b>${formatNumber(blue.outpostDamage)} / ${formatNumber(blue.baseDamage)}</b></div>
          <div class="full-stat-row"><span>基地护甲 / 堡垒开甲进度</span><b>${frame.structures.red.baseArmorOpen ? `展开 · ${escapeHtml(frame.structures.red.baseArmorOpenedBy || "已开甲")}` : `闭合 · ${formatNumber(frame.structures.red.fortressCaptureSeconds)}s`}</b><b>${frame.structures.blue.baseArmorOpen ? `展开 · ${escapeHtml(frame.structures.blue.baseArmorOpenedBy || "已开甲")}` : `闭合 · ${formatNumber(frame.structures.blue.fortressCaptureSeconds)}s`}</b></div>
          <div class="full-stat-row"><span>17 / 42mm 发弹</span><b>${red.shots17} / ${red.shots42}</b><b>${blue.shots17} / ${blue.shots42}</b></div>
          <div class="full-stat-row"><span>击毁 / 补给</span><b>${red.kills} / ${red.supplies}</b><b>${blue.kills} / ${blue.supplies}</b></div>
          <div class="full-stat-row"><span>买活 / 雷达反制</span><b>${red.buybacks} / ${red.radarCounters}</b><b>${blue.buybacks} / ${blue.radarCounters}</b></div>
          <div class="full-stat-row"><span>空中付费解锁</span><b>${red.uavCounterBuyouts}</b><b>${blue.uavCounterBuyouts}</b></div>
          <div class="full-stat-row"><span>空中付费支援秒数</span><b>${red.uavPaidSupportSeconds}</b><b>${blue.uavPaidSupportSeconds}</b></div>
          <div class="full-stat-row"><span>科技核心 / 核心经济</span><b>Lv.${frame.teams.red.technologyCoreLevel} / +${formatNumber(frame.teams.red.technologyCoreIncomePer10)}</b><b>Lv.${frame.teams.blue.technologyCoreLevel} / +${formatNumber(frame.teams.blue.technologyCoreIncomePer10)}</b></div>
          <div class="full-stat-row"><span>核心累计产金</span><b>${formatNumber(frame.teams.red.technologyCoreEarnedCoins)}</b><b>${formatNumber(frame.teams.blue.technologyCoreEarnedCoins)}</b></div>
        </div>`;
    }

    function renderDetail(frame) {
      const robot = frame.robots.find((item) => item.key === selectedKey);
      if (!robot) return;
      const coreDetails = robot.role === "工程" ? `
          <div><span>已兑科技核心</span><b>Lv.${robot.technologyCoreLevel} / 4</b></div>
          <div><span>科技核心经济</span><b>+${formatNumber(robot.technologyCoreIncomePer10)} / 10s</b></div>
          <div><span>下一等级计划</span><b>${robot.technologyCoreNextLevel ? `Lv.${robot.technologyCoreNextLevel} · ${formatNumber(robot.technologyCorePlannedIn)}s` : "本局暂无"}</b></div>
          <div><span>核心累计产金</span><b>${formatNumber(robot.technologyCoreEarnedCoins)}</b></div>
          <div><span>装配区累计无敌</span><b>${formatNumber(robot.assemblyInvulnerableSeconds)} / 180s</b></div>
          <div><span>装配区保护</span><b>${robot.assemblyProtected ? `生效中 · 剩余 ${formatNumber(robot.assemblyInvulnerableRemaining)}s` : "未生效"}</b></div>` : "";
      const uavDetails = robot.role === "空中" ? `
          <div><span>空中状态</span><b>${escapeHtml(UAV_FLIGHT_LABEL[robot.uavFlightState] || "未知")}${robot.uavSupportActive ? " · 支援已开启" : ""}</b></div>
          <div><span>免费 / 付费支援</span><b>${formatNumber(robot.uavSupportSeconds)}s / ${formatNumber(robot.uavPaidSupportSeconds)}s</b></div>
          <div><span>发射机构</span><b>${robot.uavRadarWeaponLocked ? `锁定中 · ${robot.radarCounteredIn}s` : "可用"}</b></div>
          <div><span>承受攻击 / 补给复活</span><b>规则不适用</b></div>` : "";
      const terrainProfile = navigation?.teams?.[robot.school]?.[robot.role];
      const terrainAbilityLabels = {
        fly_ramp: "飞坡", rough_road: "起伏路", road_tunnel: "公路隧道",
        highland_tunnel: "高地隧道", road_step: "公路台阶",
        central_highland_step: "中央高地台阶", central_highland_400mm_jump: "400mm 跳高地",
        slope_43: "43°坡", trapezoid_highland_step: "梯形高地台阶",
      };
      const terrainAbilities = (terrainProfile?.abilities || []).map((ability) => terrainAbilityLabels[ability] || ability);
      const terrainDetails = robot.role !== "空中" ? `
          <div><span>已学习地形能力</span><b>${escapeHtml(terrainAbilities.join(" / ") || "无特殊跨越")}</b></div>
          <div><span>当前地形动作</span><b>${escapeHtml(robot.terrainAction || "常规地面")}</b></div>
          <div><span>地形速度倍率</span><b>${Math.round((robot.terrainSpeedMultiplier ?? 1) * 100)}%</b></div>` : "";
      const weaponModel = robot.sampledWeaponAccuracy ? `
          <div><span>本局基础命中率</span><b>${(robot.sampledWeaponAccuracy * 100).toFixed(1)}% · 每发随机</b></div>
          ${robot.role === "英雄" ? `<div><span>42mm 单发模型</span><b>机器人 ${formatNumber(robot.damagePerHitByTarget?.robot?.mode_damage || 200)} / 前哨 ${formatNumber(robot.damagePerHitByTarget?.outpost?.mode_damage || 200)} / 基地 ${formatNumber(robot.damagePerHitByTarget?.base?.mode_damage || 200)}</b></div>` : ""}` : "";
      elements.selectedLabel.textContent = `${sideCode(robot.side)} · ${robot.role}`;
      elements.detail.className = "";
      elements.detail.innerHTML = `
        <div class="full-detail-grid">
          <div><span>坐标</span><b>${robot.x.toFixed(1)}, ${robot.y.toFixed(1)}</b></div>
          <div><span>等级 / 整机</span><b>Lv.${robot.level}${robot.heroArchetype ? ` · ${HERO_ARCHETYPE_LABEL[robot.heroArchetype]}` : ""}</b></div>
          <div><span>血量</span><b>${robot.role === "空中" ? "规则不适用" : `${formatNumber(robot.hp)} / ${formatNumber(robot.maxHp)}`}</b></div>
          <div><span>弹量 / 热量</span><b>${formatNumber(robot.ammo)} / ${formatNumber(robot.heat)}</b></div>
          <div><span>发弹 / 命中</span><b>${robot.shots} / ${robot.hits}</b></div>
          <div><span>击毁 / 战亡</span><b>${robot.kills} / ${robot.deaths}</b></div>
          <div><span>复活状态</span><b>${robot.role === "空中" ? "不适用" : robot.respawnIn ? `${robot.respawnIn}s` : robot.weak ? "虚弱" : robot.invulnerable ? "无敌" : "正常"}</b></div>
          <div><span>所在补给区域</span><b>${robot.role === "空中" ? "不适用" : escapeHtml(robot.serviceZone || "无")}</b></div>
          <div><span>复活决策</span><b>${robot.role === "空中" ? "不适用" : robot.respawnMode === "reading" ? `读条 ${robot.respawnProgress}/${robot.respawnRequired}` : `买活 ${robot.buybacks} 次`}</b></div>
          <div><span>当前选点控制器</span><b>${robot.role === "空中" ? "无人机状态机" : robot.policySource === "transformer" ? "Temporal Transformer" : "规则/任务约束"}</b></div>
          <div><span>无人机反制</span><b>${robot.role === "空中" ? `${robot.radarCounterCount}/5 · 剩 ${robot.radarCounteredIn}s` : "—"}</b></div>
          ${weaponModel}
          ${coreDetails}
          ${uavDetails}
          ${terrainDetails}
        </div>
        <p class="full-detail-status">当前：${escapeHtml(robot.status)}<br>任务：${robot.objectiveKey ? escapeHtml(robot.objectiveKey.endsWith(":outpost") ? "进攻前哨站" : "进攻基地") : "无指定结构目标"}<br>路线：${robot.passages.length ? escapeHtml(robot.passages.join("、")) : "常规可通行路线"}</p>`;
    }

    function renderEvents(second) {
      const visible = simulation.events.filter((item) => item.second <= second && !["terrain"].includes(item.type)).slice(-60).reverse();
      elements.eventCount.textContent = `${visible.length} 条`;
      if (!visible.length) { elements.events.innerHTML = '<div class="full-empty">等待交火</div>'; return; }
      elements.events.innerHTML = visible.map((item) => `
        <div class="full-event-row ${item.side}"><time>${formatClock(item.second)}</time><b>${item.side === "red" || item.side === "blue" ? escapeHtml(sideCode(item.side)) : "系统"}</b><span>${escapeHtml(item.text)}</span></div>`).join("");
    }

    function renderUi(force) {
      if (!simulation) return;
      const second = clamp(Math.floor(playhead), 0, simulation.frames.length - 1);
      if (!force && second === lastUiSecond) return;
      lastUiSecond = second;
      const frame = simulation.frames[second];
      elements.slider.value = String(second);
      elements.clock.textContent = formatClock(second);
      if (frame.finished) elements.result.textContent = `${frame.winner === "draw" ? "平局" : `${sideCode(frame.winner)} 胜`} · ${frame.reason}`;
      else elements.result.textContent = `第 ${second} 秒 · 逐车状态同步`;
      renderRoster("red", frame);
      renderRoster("blue", frame);
      renderStats(frame);
      renderDetail(frame);
      renderEvents(second);
    }

    function stop() {
      playing = false;
      elements.play.textContent = "▶ 播放";
    }

    function startPlayback() {
      if (!simulation) return false;
      if (simulationComplete && playhead >= simulation.frames.length - 1) playhead = 0;
      playRequested = false;
      playing = true;
      elements.play.textContent = "Ⅱ 暂停";
      lastAnimation = performance.now();
      return true;
    }

    function acceptSimulation(result, latencyMs) {
      simulation = result;
      simulationComplete = true;
      playhead = 0;
      lastUiSecond = -1;
      selectedKey = "red:英雄";
      elements.slider.max = String(simulation.frames.length - 1);
      elements.seed.textContent = `seed ${seed}`;
      const latency = Number.isFinite(Number(latencyMs)) ? ` · 后台 ${Math.round(latencyMs)}ms` : "";
      elements.status.textContent = `${simulation.frames.length} 帧 · ${simulation.events.length} 事件${latency} · 已生成`;
      elements.status.className = "ready";
      elements.create.disabled = false;
      renderUi(true);
      drawMap();
      if (playRequested) startPlayback();
    }

    function startStreamingSimulation(message) {
      simulation = { state: message.state, frames: [message.frame], events: [] };
      simulationComplete = false;
      expectedSimulationFrames = Number(message.expectedFrames || 421);
      playhead = 0;
      lastUiSecond = -1;
      selectedKey = "red:英雄";
      elements.slider.max = "0";
      elements.seed.textContent = `seed ${seed}`;
      const policyText = message.state.policy?.active
        ? `Transformer ${formatNumber(message.state.policy.parameterCount)} 参数已接管战术选点`
        : "Transformer 不可用 · 统计策略回退";
      elements.status.textContent = `${policyText} · 1/${expectedSimulationFrames} 帧`;
      elements.status.className = "ready";
      elements.create.disabled = false;
      renderUi(true);
      drawMap();
      if (playRequested) startPlayback();
    }

    function appendSimulationChunk(message) {
      if (!simulation) return;
      if (message.frames?.length) simulation.frames.push(...message.frames);
      if (message.events?.length) simulation.events.push(...message.events);
      if (message.policy) simulation.state.policy = message.policy;
      simulationComplete = Boolean(message.complete);
      elements.slider.max = String(simulation.frames.length - 1);
      if (simulationComplete) {
        const policy = simulation.state.policy || {};
        elements.status.textContent = `${simulation.frames.length} 帧 · Transformer 选点 ${formatNumber(policy.decisions || 0)} 次 · 规则约束 ${formatNumber(policy.constrained || 0)} 次 · 后台 ${Math.round(Number(message.latencyMs) || 0)}ms · 已生成`;
      } else {
        elements.status.textContent = `Transformer 实时推演 · ${simulation.frames.length}/${expectedSimulationFrames} 帧 · 已选点 ${formatNumber(simulation.state.policy?.decisions || 0)} 次`;
      }
      elements.status.className = "ready";
    }

    function simulationError(message) {
      playRequested = false;
      elements.status.textContent = `推演失败：${message}`;
      elements.status.className = "error";
      elements.create.disabled = false;
    }

    function ensureSimulationWorker() {
      if (simulationWorker) return simulationWorker;
      if (!("Worker" in window)) return null;
      const worker = new Worker("./full-match-worker.js?v=13");
      worker.onmessage = (event) => {
        const message = event.data || {};
        if (message.type === "ready") return;
        if (message.requestId !== activeSimulationRequest) return;
        if (message.type === "started") startStreamingSimulation(message);
        else if (message.type === "chunk") appendSimulationChunk(message);
        else if (message.type === "result") acceptSimulation(message.result, message.latencyMs);
        else if (message.type === "error") simulationError(message.message || "未知错误");
      };
      worker.onerror = (event) => {
        simulationWorker = null;
        simulationError(`后台线程错误：${event.message || "未知错误"}`);
      };
      worker.postMessage({ type: "initialize", model, navigation });
      simulationWorker = worker;
      return worker;
    }

    function generate() {
      if (!model || !navigation) return;
      stop();
      seed = (seed + 1) >>> 0;
      const requestId = ++simulationRequestId;
      activeSimulationRequest = requestId;
      elements.create.disabled = true;
      elements.status.textContent = "后台推演 420 秒…页面可继续操作";
      elements.status.className = "";
      elements.result.textContent = "子线程生成逐车轨迹、交火与补给事件";
      const worker = ensureSimulationWorker();
      if (worker) {
        worker.postMessage({
          type: "run",
          requestId,
          redSchool: elements.redSelect.value,
          blueSchool: elements.blueSelect.value,
          seed,
          matchOptions: { heroArchetypes: { red: elements.redHeroMode.value, blue: elements.blueHeroMode.value } },
        });
        return;
      }
      // Very old browsers without Worker support retain a functional fallback.
      window.setTimeout(() => {
        if (requestId !== activeSimulationRequest) return;
        try {
          const started = performance.now();
          const result = window.RMUCFullMatchEngine.runMatch(
            model, navigation, elements.redSelect.value, elements.blueSelect.value, seed, null,
            { heroArchetypes: { red: elements.redHeroMode.value, blue: elements.blueHeroMode.value } },
          );
          acceptSimulation(result, performance.now() - started);
        } catch (error) {
          simulationError(error.message);
        }
      }, 20);
    }

    function populate() {
      const teams = Object.entries(model.teams).sort((left, right) => String(left[1].team).localeCompare(String(right[1].team), "zh-CN"));
      const options = teams.map(([school, team]) => `<option value="${escapeHtml(school)}">${escapeHtml(team.team)} · ${escapeHtml(school)}</option>`).join("");
      elements.redSelect.innerHTML = options;
      elements.blueSelect.innerHTML = options;
      const shared = window.RMUC_SIMULATOR_MATCHUP || {};
      elements.redSelect.value = model.teams[shared.red] ? shared.red : model.teams[DEFAULT_RED] ? DEFAULT_RED : teams[0][0];
      elements.blueSelect.value = model.teams[shared.blue] ? shared.blue : model.teams[DEFAULT_BLUE] ? DEFAULT_BLUE : teams[1][0];
      updateHeroAutoLabel("red");
      updateHeroAutoLabel("blue");
      publishMatchup();
      elements.status.textContent = `${teams.length} 队 · ${model.ruleset?.version || "规则参数"} 就绪 · 进入沙盘后后台生成`;
      elements.status.className = "ready";
      // The model is loaded only after the simulator panel enters the loader
      // observer, so a second visibility observer would add latency here.
      generate();
    }

    function animation(now) {
      if (playing && simulation) {
        // A queued animation frame can carry a timestamp from just before the
        // play-button handler reset lastAnimation. Never let that race move the
        // playhead below frame zero and terminate the playback loop.
        const delta = clamp((now - lastAnimation) / 1000, 0, 0.2);
        lastAnimation = now;
        playhead = Math.max(0, playhead + delta * playbackSpeed);
        if (playhead >= simulation.frames.length - 1) {
          playhead = simulation.frames.length - 1;
          if (simulationComplete) stop();
        }
        renderUi(false);
        drawMap();
      }
      window.requestAnimationFrame(animation);
    }

    elements.create.addEventListener("click", () => { playRequested = false; generate(); });
    elements.play.addEventListener("click", () => {
      if (playing) { playRequested = false; stop(); return; }
      if (startPlayback()) return;
      playRequested = true;
      elements.play.textContent = "… 准备中";
      elements.status.textContent = "正在载入参数并生成对局，完成后自动播放";
      if (model && navigation) generate();
      else loadSimulationData();
    });
    elements.back.addEventListener("click", () => { playhead = Math.max(0, Math.floor(playhead) - 1); renderUi(true); drawMap(); });
    elements.forward.addEventListener("click", () => { if (!simulation) return; playhead = Math.min(simulation.frames.length - 1, Math.floor(playhead) + 1); renderUi(true); drawMap(); });
    elements.speed.addEventListener("change", () => { playbackSpeed = Number(elements.speed.value) || 1; });
    elements.slider.addEventListener("input", () => { if (!simulation) return; playhead = Number(elements.slider.value); renderUi(true); drawMap(); });
    elements.redSelect.addEventListener("change", teamSelectionChanged);
    elements.blueSelect.addEventListener("change", teamSelectionChanged);
    elements.redHeroMode.addEventListener("change", generate);
    elements.blueHeroMode.addEventListener("change", generate);
    window.addEventListener("rmuc:simulator-matchup", (event) => {
      const detail = event.detail || {};
      if (detail.source === "full" || !model || !model.teams[detail.red] || !model.teams[detail.blue]) return;
      const changed = elements.redSelect.value !== detail.red || elements.blueSelect.value !== detail.blue;
      elements.redSelect.value = detail.red;
      elements.blueSelect.value = detail.blue;
      updateHeroAutoLabel("red");
      updateHeroAutoLabel("blue");
      if (changed) generate();
    });
    canvas.addEventListener("click", (event) => {
      if (!simulation) return;
      const point = worldPoint(event.clientX, event.clientY);
      const frame = simulation.frames[Math.floor(playhead)];
      const nearest = frame.robots.map((robot) => ({ robot, distance: Math.hypot(robot.x - point[0], robot.y - point[1]) })).sort((left, right) => left.distance - right.distance)[0];
      if (nearest && nearest.distance <= 1.1) { selectedKey = nearest.robot.key; renderUi(true); drawMap(); }
    });
    mapImage.addEventListener("load", drawMap);
    window.addEventListener("resize", drawMap);

    function loadSimulationData() {
      if (simulationDataLoading || model) return;
      simulationDataLoading = true;
      elements.status.textContent = "正在后台载入沙盘参数…";
      Promise.all([
        fetch("./data/models/full_simulation.json?v=12").then((response) => { if (!response.ok) throw new Error(`逐车参数 HTTP ${response.status}`); return response.json(); }),
        fetch("./data/models/terrain_navigation.json?v=24").then((response) => { if (!response.ok) throw new Error(`地形图 HTTP ${response.status}`); return response.json(); }),
      ]).then(([modelData, navigationData]) => {
        model = modelData;
        navigation = navigationData;
        populate();
      }).catch((error) => {
        elements.status.textContent = `完整沙盘载入失败：${error.message}`;
        elements.status.className = "error";
        elements.create.disabled = true;
      });
    }

    elements.create.disabled = true;
    elements.status.textContent = "沙盘已延迟加载 · 进入此区域后启动";
    const simulatorPanel = canvas.closest("#simulator") || canvas;
    if ("IntersectionObserver" in window) {
      const loaderObserver = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        loaderObserver.disconnect();
        loadSimulationData();
      }, { rootMargin: "240px" });
      loaderObserver.observe(simulatorPanel);
    } else {
      window.setTimeout(loadSimulationData, 100);
    }
    window.requestAnimationFrame(animation);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, { once: true });
  else init();
})();
