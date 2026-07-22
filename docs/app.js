"use strict";

const $ = (selector) => document.querySelector(selector);
const COLORS = { red: "#ff526c", blue: "#48a0ff", deadRed: "#761d2d", deadBlue: "#174a78", gold: "#f3bd4d", green: "#38d39f" };
const ROLE_ORDER = ["基地", "前哨站", "英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"];
const ROLE_LABEL = { "英雄":"1", "工程":"2", "步兵3":"3", "步兵4":"4", "哨兵":"AI", "空中":"6" };
const STRUCTURES = [
  ["红", "基地", .095, .500, "基"], ["红", "前哨站", .393, .750, "前"],
  ["蓝", "基地", .905, .500, "基"], ["蓝", "前哨站", .607, .250, "前"],
];
const R = { id:0, type:1, side:2, hp:3, max:4, x:5, y:6, yaw:7, a17:8, a42:9, coins:10, vulnerable:11 };
const E = { sec:0, type:1, robot:2, side:3, category:4, value:5, note:6, target:7 };
const FIELD_WIDTH_METERS = 28;
const FIELD_HEIGHT_METERS = 15;
const MAP_HEIGHT_METERS = 17;
const FIELD_Y_SPAN = FIELD_HEIGHT_METERS / MAP_HEIGHT_METERS;
const FIELD_Y_OFFSET = (1 - FIELD_Y_SPAN) / 2;
const STATIC_DATA = Boolean(window.RMUC_STATIC_DATA);
const MEMORY_KEY = "rmuc-dashboard-memory-v1";
const MEMORY_ENABLED_KEY = "rmuc-dashboard-memory-enabled";

function emptyMemory() {
  return {region:"",selection:{region:"",matchNo:"",gameId:""},matches:{},rounds:{},positions:{},speed:1};
}
function readMemory() {
  try {
    const saved = JSON.parse(localStorage.getItem(MEMORY_KEY) || "null") || {};
    return {...emptyMemory(),...saved,selection:{...emptyMemory().selection,...(saved.selection||{})},matches:{...(saved.matches||{})},rounds:{...(saved.rounds||{})},positions:{...(saved.positions||{})}};
  } catch (_) { return emptyMemory(); }
}
function readMemoryEnabled() {
  try { return localStorage.getItem(MEMORY_ENABLED_KEY) !== "false"; }
  catch (_) { return true; }
}

let memory = readMemory();
let memoryEnabled = readMemoryEnabled();
let memorySaveTimer = 0;

function writeMemory() {
  if (!memoryEnabled) return;
  const recent = Object.entries(memory.positions).sort((a,b)=>Number(b[1]?.updated||0)-Number(a[1]?.updated||0)).slice(0,50);
  memory.positions = Object.fromEntries(recent);
  try { localStorage.setItem(MEMORY_KEY,JSON.stringify(memory)); } catch (_) {}
}
function persistMemory(immediate=false) {
  if (!memoryEnabled) return;
  clearTimeout(memorySaveTimer);
  if (immediate) writeMemory();
  else memorySaveTimer = setTimeout(writeMemory,180);
}
function restoreSelect(select,value) {
  if (value == null) return false;
  const option = [...select.options].find(item=>String(item.value)===String(value));
  if (!option) return false;
  select.value = option.value;
  return true;
}
function rememberSelection(immediate=false) {
  if (!memoryEnabled) return;
  const region=$("#region-select").value;
  const matchNo=$("#match-select").value;
  const gameId=$("#round-select").value;
  if (!region||!matchNo||!gameId) return;
  memory.region=region;
  if (region&&matchNo) memory.matches[region]=matchNo;
  if (region&&matchNo&&gameId) memory.rounds[`${region}::${matchNo}`]=gameId;
  memory.selection={region,matchNo,gameId};
  persistMemory(immediate);
}
function rememberPlayhead(immediate=false) {
  if (!memoryEnabled || !state.game) return;
  const gameId = String(state.game.info.game_id);
  memory.positions[gameId] = {second:Math.floor(state.playhead),updated:Date.now()};
  memory.speed = state.speed;
  persistMemory(immediate);
}

const state = {
  game: null, playhead: 1, speed: 1, playing: false, lastSecond: -1,
  lastAnimation: performance.now(), lastDraw: 0, dirty: true, tracks: new Map(),
  uavCounterWindows: new Map(),
  predictionEnabled: false, predictionWorker: null, predictionReady: false,
  predictionPending: false, predictionRequestId: 0, predictionActiveRequest: 0,
  predictionGeneration: 0, predictionWantedSecond: null, predictionSecond: -1,
  predictionLatency: 0, predictions: [],
};

const mapCanvas = $("#map-canvas");
const mapCtx = mapCanvas.getContext("2d");
const timelineCanvas = $("#timeline-canvas");
const timelineCtx = timelineCanvas.getContext("2d");
const mapImage = new Image();
mapImage.src = STATIC_DATA ? "./assets/map.webp" : "/assets/map.webp";
mapImage.onload = () => { state.dirty = true; drawMap(); };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch]));
}
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function fmtTime(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;
}
function frameAt(second) {
  if (!state.game) return [];
  const frames = state.game.frames;
  return frames[second] || frames[second - 1] || [];
}
function robotKey(robot) { return `${robot[R.side]}:${robot[R.id]}`; }
function colorFor(side) { return side === "红" ? COLORS.red : COLORS.blue; }
function showToast(message,kind="error") {
  const toast = $("#toast"); toast.textContent = message; toast.classList.toggle("success",kind==="success"); toast.classList.add("show");
  setTimeout(() => { toast.classList.remove("show"); toast.classList.remove("success"); }, 3200);
}
function setLoading(active) { $("#loading").classList.toggle("hidden", !active); }

let staticCatalogPromise = null;

async function loadCompressedJson(url) {
  const response = await fetch(url, {cache:"force-cache"});
  if (!response.ok) throw new Error(`静态数据读取失败：HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    if (!("DecompressionStream" in window)) throw new Error("当前浏览器不支持压缩数据，请升级 Chrome、Edge 或 Safari");
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function getStaticJson(url) {
  staticCatalogPromise ||= fetch("./data/catalog.json", {cache:"force-cache"}).then(response => {
    if (!response.ok) throw new Error(`静态目录读取失败：HTTP ${response.status}`);
    return response.json();
  });
  const request = new URL(url, "https://rmuc-static.invalid");
  if (request.pathname === "/api/info") {
    return {name:"RMUC 2026 区域赛数据中心",phone_url:"GitHub Pages 静态版",database:"分局压缩数据"};
  }
  const catalog = await staticCatalogPromise;
  if (request.pathname === "/api/regions") return {regions:catalog.regions};
  if (request.pathname === "/api/matches") {
    return {matches:catalog.matches[request.searchParams.get("region")] || []};
  }
  if (request.pathname === "/api/rounds") {
    const key = `${request.searchParams.get("region")}::${request.searchParams.get("match_no")}`;
    return {rounds:catalog.rounds[key] || []};
  }
  if (request.pathname === "/api/game") {
    const gameId = request.searchParams.get("game_id");
    if (!/^\d+$/.test(gameId || "")) throw new Error("无效的比赛编号");
    return loadCompressedJson(`./data/games/${gameId}.json.gz`);
  }
  throw new Error(`未知静态接口：${request.pathname}`);
}

async function getJson(url) {
  if (STATIC_DATA) return getStaticJson(url);
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function init() {
  try {
    const regions = await getJson("/api/regions");
    fillSelect($("#region-select"), regions.regions.map(region => [region, region]));
    if (memoryEnabled) restoreSelect($("#region-select"),memory.selection.region||memory.region);
    await loadMatches();
  } catch (error) { setLoading(false); showToast(error.message); }
}
function fillSelect(select, options) {
  select.innerHTML = options.map(([value,label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
}
async function loadMatches() {
  setLoading(true);
  try {
    const region = $("#region-select").value;
    const data = await getJson(`/api/matches?region=${encodeURIComponent(region)}`);
    fillSelect($("#match-select"), data.matches.map(match => [
      match.match_no, `第${match.match_no}场 · ${match.red} vs ${match.blue}`
    ]));
    const selected=memory.selection;
    const rememberedMatch=selected.region===region&&selected.matchNo ? selected.matchNo : memory.matches[region];
    if (memoryEnabled) restoreSelect($("#match-select"),rememberedMatch);
    if (memoryEnabled) {
      memory.region = region;
      memory.matches[region] = $("#match-select").value;
      persistMemory();
    }
    await loadRounds();
  } catch (error) { setLoading(false); showToast(error.message); }
}
async function loadRounds() {
  setLoading(true);
  try {
    const region = $("#region-select").value;
    const matchNo = $("#match-select").value;
    const data = await getJson(`/api/rounds?region=${encodeURIComponent(region)}&match_no=${encodeURIComponent(matchNo)}`);
    fillSelect($("#round-select"), data.rounds.map(round => [round.game_id, `第${round.round_no}局`]));
    const memoryKey = `${region}::${matchNo}`;
    const selected=memory.selection;
    const rememberedRound=selected.region===region&&String(selected.matchNo)===String(matchNo)&&selected.gameId
      ? selected.gameId : memory.rounds[memoryKey];
    if (memoryEnabled) restoreSelect($("#round-select"),rememberedRound);
    rememberSelection(true);
    await loadGame();
  } catch (error) { setLoading(false); showToast(error.message); }
}
async function loadGame() {
  setLoading(true); stopPlayback();
  try {
    const gameId = $("#round-select").value;
    state.game = await getJson(`/api/game?game_id=${encodeURIComponent(gameId)}`);
    state.predictionGeneration += 1;
    state.predictionPending = false;
    state.predictionWantedSecond = null;
    state.predictionSecond = -1;
    state.predictions = [];
    buildTracks();
    const seconds = Object.keys(state.game.frames).map(Number);
    const firstSecond = seconds.length ? Math.min(...seconds) : 0;
    const rememberedSecond = Number(memory.positions[String(gameId)]?.second);
    state.playhead = memoryEnabled && Number.isFinite(rememberedSecond)
      ? clamp(rememberedSecond,firstSecond,state.game.info.duration) : firstSecond;
    state.lastSecond = -1;
    $("#time-slider").max = state.game.info.duration;
    $("#time-slider").value = state.playhead;
    renderState(Math.floor(state.playhead));
    rememberPlayhead();
    state.dirty = true;
    drawMap(); drawTimeline();
  } catch (error) { showToast(error.message); }
  finally { setLoading(false); }
}
function buildTracks() {
  state.tracks = new Map();
  state.uavCounterWindows = new Map([["red",[]],["blue",[]],["红",[]],["蓝",[]]]);
  if (!state.game) return;
  for (const [second, robots] of Object.entries(state.game.frames)) {
    for (const robot of robots) {
      if (["基地","前哨站"].includes(robot[R.type]) || robot[R.x] == null || robot[R.y] == null) continue;
      const key = robotKey(robot);
      if (!state.tracks.has(key)) state.tracks.set(key, []);
      state.tracks.get(key).push([Number(second), Number(robot[R.x]), Number(robot[R.y]), robot[R.side]]);
    }
  }
  for (const event of state.game.events) {
    if (event[E.type] !== "雷达反制UAV" || !state.uavCounterWindows.has(event[E.side])) continue;
    const second = Number(event[E.sec]);
    if (!Number.isFinite(second)) continue;
    const windows = state.uavCounterWindows.get(event[E.side]);
    const current = windows.at(-1);
    if (current && second < current.end) {
      current.end += 45;
      current.count += 1;
    } else {
      windows.push({start:second,end:second+45,count:1});
    }
  }
}

function uavCounterStatus(side, second) {
  const windows = state.uavCounterWindows.get(side) || [];
  const active = windows.find(window => second >= window.start && second < window.end);
  return active ? {active:true,remaining:Math.ceil(active.end-second),count:active.count} : {active:false,remaining:0,count:0};
}

function structureHtml(title, robot, side, reverse=false) {
  const hp = Number(robot?.[R.hp] || 0), max = Number(robot?.[R.max] || 0);
  const ratio = max ? clamp(hp / max * 100, 0, 100) : 0;
  const label = `<span>${title}</span>`, bar = `<div class="health-track"><div class="health-fill" style="width:${ratio}%"></div></div>`;
  const value = `<b>${hp.toLocaleString()}/${max.toLocaleString()}</b>`;
  return `<div class="structure">${reverse ? value + bar + label : label + bar + value}</div>`;
}
function updateTopHud(robots, second) {
  const info = state.game.info;
  const find = (side,type) => robots.find(robot => robot[R.side] === side && robot[R.type] === type);
  for (const side of ["红","蓝"]) {
    const isBlue = side === "蓝", other = isBlue ? "红" : "蓝";
    const school = isBlue ? info.blue : info.red;
    const winner = info.winner === side ? " · WIN" : "";
    const allSideEvents = state.game.events.filter(event => event[E.side] === side);
    const pastSideEvents = allSideEvents.filter(event => event[E.sec] <= second);
    const gates = pastSideEvents.filter(event => event[E.type] === "飞镖闸门开").length;
    const gateTotal = allSideEvents.filter(event => event[E.type] === "飞镖闸门开").length;
    const hits = pastSideEvents.filter(event => event[E.type] === "飞镖命中");
    const hitTotal = allSideEvents.filter(event => event[E.type] === "飞镖命中").length;
    const damage = hits.reduce((sum,event) => sum + Math.abs(Number(event[E.value] || 0)), 0);
    const counters = state.game.events.filter(event => event[E.type] === "雷达反制UAV" && event[E.side] === other);
    const counterNow = counters.filter(event => event[E.sec] <= second).length;
    const marked = robots.filter(robot => robot[R.side] === other && robot[R.vulnerable] && !["基地","前哨站"].includes(robot[R.type])).length;
    const base = find(side,"基地"), outpost = find(side,"前哨站");
    const structures = isBlue
      ? structureHtml("前哨",outpost,side,true) + structureHtml("基地",base,side,true)
      : structureHtml("基地",base,side) + structureHtml("前哨",outpost,side);
    $(`#top-${isBlue ? "blue" : "red"}`).innerHTML = `
      <div class="team-title">${side}方　${escapeHtml(school)}${winner}</div>
      <div class="structure-row">${structures}</div>
      <div class="special-line">飞镖　门 ${gates}/${gateTotal} · 命中 ${hits.length}/${hitTotal} · 伤害 ${damage.toLocaleString()}　　雷达　标记 ${marked} · 反制 ${counterNow}/${counters.length}</div>`;
  }
  $("#match-center").textContent = `第${info.match_no}场 · 第${info.round_no}局　${info.winner}方胜　${fmtTime(info.duration)}`;
}

function updateTeamPanel(side, robots) {
  const isBlue = side === "蓝", info = state.game.info;
  const school = isBlue ? info.blue : info.red;
  const own = robots.filter(robot => robot[R.side] === side);
  const byType = new Map(own.map(robot => [robot[R.type],robot]));
  const coins = Math.max(0,...own.map(robot => Number(robot[R.coins] || 0)));
  const mobile = own.filter(robot => !["基地","前哨站"].includes(robot[R.type]));
  const alive = mobile.filter(robot => Number(robot[R.hp]) > 0).length;
  const ammo17 = mobile.reduce((sum,robot) => sum + Number(robot[R.a17] || 0),0);
  const ammo42 = mobile.reduce((sum,robot) => sum + Number(robot[R.a42] || 0),0);
  const rows = ROLE_ORDER.map(role => {
    const robot = byType.get(role), hp = Number(robot?.[R.hp] || 0), max = Number(robot?.[R.max] || 0);
    const ratio = max ? clamp(hp / max * 100,0,100) : 0;
    return `<div class="robot-row"><span class="role">${role}</span><div class="health-track"><div class="health-fill" style="width:${ratio}%"></div></div><span class="hp">${robot ? `${hp.toFixed(0)}/${max.toFixed(0)}` : "—"}</span></div>`;
  }).join("");
  $(`#${isBlue ? "blue" : "red"}-panel`).innerHTML = `
    <div class="team-header"><h3>${side}方 · ${escapeHtml(school)}</h3><span class="coins">剩余金币 ${coins.toLocaleString()}</span></div>
    <div class="robot-list">${rows}</div>
    <div class="team-summary">在线 ${alive}/${mobile.length}　累计发弹<br>17mm　${ammo17.toLocaleString()}　　42mm　${ammo42.toLocaleString()}</div>`;
}

function eventDetails(event) {
  const values = [];
  if (event[E.category]) values.push(event[E.category]);
  if (event[E.value] != null) values.push(Number(event[E.value]).toLocaleString());
  if (event[E.target]) values.push(`→ ${event[E.target]}`);
  if (event[E.note]) values.push(event[E.note]);
  return values.join(" · ") || "—";
}
function updateEvents(second) {
  const past = state.game.events.filter(event => event[E.sec] <= second);
  const rows = past.slice(-9).reverse();
  $("#event-count").textContent = `${past.length} 条`;
  $("#event-list").innerHTML = rows.length ? rows.map(event => `
    <div class="event-row"><span>${Math.floor(event[E.sec])}s</span>
      <span class="${event[E.side] === "红" ? "side-red" : "side-blue"}">${event[E.side] || "—"}</span>
      <span class="${event[E.type] === "受击" ? "hit" : ""}">${escapeHtml(event[E.type])}</span>
      <span class="actor">${escapeHtml(event[E.robot] || "—")}</span>
      <span>${escapeHtml(eventDetails(event))}</span></div>`).join("") : `<div class="empty">当前时刻暂无事件</div>`;
}
function renderState(second) {
  if (!state.game) return;
  const robots = frameAt(second);
  updateTopHud(robots,second); updateTeamPanel("红",robots); updateTeamPanel("蓝",robots); updateEvents(second);
  const info = state.game.info;
  const mobile = robots.filter(robot => !["基地","前哨站"].includes(robot[R.type]));
  const alive = mobile.filter(robot => Number(robot[R.hp]) > 0).length;
  const eventTotal = Object.values(state.game.event_counts || {}).reduce((sum,value) => sum + Number(value || 0),0);
  const overviewMatch = $("#overview-match");
  if (overviewMatch) overviewMatch.textContent = `${info.region} · ${info.match_no}-${info.round_no}`;
  const overviewDuration = $("#overview-duration");
  if (overviewDuration) overviewDuration.textContent = fmtTime(info.duration);
  const overviewRobots = $("#overview-robots");
  if (overviewRobots) overviewRobots.textContent = `${alive}/${mobile.length}`;
  const overviewEvents = $("#overview-events");
  if (overviewEvents) overviewEvents.textContent = eventTotal.toLocaleString();
  $("#map-match-label").textContent = `${info.region} 第${info.match_no}场 · 第${info.round_no}局 ${info.winner}方胜`;
  $("#red-legend").textContent = info.red; $("#blue-legend").textContent = info.blue;
  $("#map-time").textContent = `T + ${String(second).padStart(3,"0")}s`;
  $("#time-output").textContent = `${fmtTime(second)} / ${fmtTime(info.duration)}`;
  $("#time-slider").value = second;
  drawTimeline(); state.lastSecond = second; state.dirty = true;
  schedulePrediction(second);
}

function setPredictionStatus(text,kind="") {
  const label=$("#prediction-status");
  if (!label) return;
  label.textContent=text;
  label.classList.toggle("ready",kind==="ready");
  label.classList.toggle("error",kind==="error");
}
function ensurePredictionWorker() {
  if (state.predictionWorker) return true;
  if (!("Worker" in window)) {
    setPredictionStatus("浏览器不支持后台预测","error");
    return false;
  }
  const worker=new Worker("./prediction-worker.js?v=26");
  worker.onmessage=event=>{
    const message=event.data||{};
    if (message.type==="status") {
      state.predictionReady=message.status==="ready";
      if (state.predictionEnabled) setPredictionStatus(message.text,message.status==="ready"?"ready":"");
      return;
    }
    if (message.type==="error") {
      if (message.requestId===state.predictionActiveRequest) state.predictionPending=false;
      if (message.generation!==state.predictionGeneration) return;
      state.predictions=[]; state.predictionSecond=-1;
      setPredictionStatus(`预测失败：${message.message}`,"error"); state.dirty=true;
      return;
    }
    if (message.type!=="result") return;
    if (message.requestId===state.predictionActiveRequest) state.predictionPending=false;
    if (message.generation!==state.predictionGeneration||!state.predictionEnabled) return;
    const currentSecond=Math.floor(state.playhead);
    if (message.second===currentSecond) {
      state.predictions=message.predictions||[];
      state.predictionSecond=message.second;
      state.predictionLatency=Number(message.latencyMs||0);
      const adjusted=state.predictions.filter(item=>item.terrainAdjusted).length;
      setPredictionStatus(`${state.predictions.length} 台 · ${state.predictionLatency.toFixed(0)}ms · 地形修正 ${adjusted}`,"ready");
      state.dirty=true;
    }
    const wanted=state.predictionWantedSecond;
    state.predictionWantedSecond=null;
    if (wanted!=null&&wanted!==message.second) schedulePrediction(wanted);
    else if (currentSecond!==message.second) schedulePrediction(currentSecond);
  };
  worker.onerror=event=>{
    state.predictionPending=false; state.predictionReady=false; state.predictions=[];
    setPredictionStatus(`预测线程错误：${event.message||"未知错误"}`,"error");
  };
  state.predictionWorker=worker;
  return true;
}
function schedulePrediction(second) {
  if (!state.predictionEnabled||!state.game) return;
  second=Math.floor(Number(second));
  if (second<5) {
    state.predictions=[]; state.predictionSecond=-1;
    setPredictionStatus("需要前 5 秒轨迹","ready");
    return;
  }
  if (state.predictionPending) {
    state.predictionWantedSecond=second;
    return;
  }
  if (!ensurePredictionWorker()) return;
  const history={};
  for (const offset of [0,1,3,5]) history[String(offset)]=frameAt(Math.max(0,second-offset));
  const requestId=++state.predictionRequestId;
  state.predictionActiveRequest=requestId;
  state.predictionPending=true;
  state.predictionWantedSecond=null;
  state.predictionWorker.postMessage({
    type:"predict", requestId, generation:state.predictionGeneration, second,
    duration:state.game.info.duration, history,
    schools:{"红":state.game.info.red,"蓝":state.game.info.blue},
  });
}
function togglePrediction() {
  state.predictionEnabled=!state.predictionEnabled;
  const button=$("#prediction-button");
  button.classList.toggle("active",state.predictionEnabled);
  button.setAttribute("aria-pressed",String(state.predictionEnabled));
  button.textContent=state.predictionEnabled?"预测 开":"预测 关";
  if (!state.predictionEnabled) {
    state.predictions=[]; state.predictionSecond=-1; state.predictionWantedSecond=null;
    setPredictionStatus("预测已关闭"); state.dirty=true;
    return;
  }
  setPredictionStatus(state.predictionReady?"Transformer 预测已开启":"模型加载中",state.predictionReady?"ready":"");
  schedulePrediction(Math.floor(state.playhead));
}

function canvasSize(canvas, ratio, fixedHeight=null) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, canvas.parentElement.clientWidth - (canvas === timelineCanvas ? 0 : 0));
  const height = fixedHeight || width / ratio;
  canvas.style.height = `${height}px`;
  const pixelW = Math.round(width*dpr), pixelH = Math.round(height*dpr);
  if (canvas.width !== pixelW || canvas.height !== pixelH) { canvas.width=pixelW; canvas.height=pixelH; }
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  return { width, height, dpr };
}
function mapPoint(x,y,width,height) {
  const u=clamp(x/FIELD_WIDTH_METERS,0,1);
  const fieldV=1-clamp(y/FIELD_HEIGHT_METERS,0,1);
  return [u*width,(FIELD_Y_OFFSET+fieldV*FIELD_Y_SPAN)*height];
}
function uvPoint(u,v,width,height) { return [u*width,v*height]; }

function drawMap() {
  if (!state.game || !mapImage.complete) return;
  const ratio = mapImage.naturalWidth / mapImage.naturalHeight;
  const {width,height} = canvasSize(mapCanvas,ratio);
  mapCtx.clearRect(0,0,width,height); mapCtx.drawImage(mapImage,0,0,width,height);
  mapCtx.fillStyle="rgba(2,7,12,.12)"; mapCtx.fillRect(0,0,width,height);
  const second = Math.floor(state.playhead), alpha = state.playhead-second;
  const robots = frameAt(second), next = new Map(frameAt(second+1).map(robot => [robotKey(robot),robot]));
  const scale = clamp(width/850,.78,1.7);

  for (const points of state.tracks.values()) {
    const trailSecond=Math.floor(state.playhead);
    const recent = points.filter(point => point[0] >= trailSecond-5 && point[0] <= trailSecond);
    if (recent.length<2) continue;
    mapCtx.beginPath();
    recent.forEach((point,index) => { const [x,y]=mapPoint(point[1],point[2],width,height); index ? mapCtx.lineTo(x,y) : mapCtx.moveTo(x,y); });
    mapCtx.strokeStyle = recent[0][3] === "红" ? "rgba(255,82,108,.52)" : "rgba(72,160,255,.52)";
    mapCtx.lineWidth=2*scale; mapCtx.stroke();
  }

  drawPredictions(width,height,scale,second);

  for (const [side,type,u,v,label] of STRUCTURES) {
    const robot=robots.find(item=>item[R.side]===side&&item[R.type]===type);
    drawStructure(...uvPoint(u,v,width,height),side,label,robot,scale);
  }
  for (const robot of robots) {
    if (["基地","前哨站"].includes(robot[R.type]) || robot[R.x]==null || robot[R.y]==null) continue;
    let x=Number(robot[R.x]),y=Number(robot[R.y]); const after=next.get(robotKey(robot));
    if (after && after[R.x]!=null && after[R.y]!=null) { x+=(Number(after[R.x])-x)*alpha; y+=(Number(after[R.y])-y)*alpha; }
    drawRobot(...mapPoint(x,y,width,height),robot,scale,state.playhead);
  }
  state.dirty=false;
}
function drawPredictions(width,height,scale,second) {
  if (!state.predictionEnabled||state.predictionSecond!==second||!state.predictions.length) return;
  mapCtx.save();
  for (const prediction of state.predictions) {
    const route=prediction.route?.length?prediction.route:prediction.points.filter(point=>point.horizon<=10);
    if (route.length<2) continue;
    const color=colorFor(prediction.side), [startX,startY]=mapPoint(prediction.current[0],prediction.current[1],width,height);
    mapCtx.beginPath(); mapCtx.moveTo(startX,startY);
    for (const point of route) mapCtx.lineTo(...mapPoint(point.x,point.y,width,height));
    mapCtx.setLineDash([6*scale,4*scale]);
    mapCtx.strokeStyle=color; mapCtx.globalAlpha=.84; mapCtx.lineWidth=2.2*scale; mapCtx.stroke();
    mapCtx.setLineDash([]); mapCtx.globalAlpha=1;
    const previous=route.at(-2), endpoint=route.at(-1);
    const [previousX,previousY]=mapPoint(previous.x,previous.y,width,height);
    const [endX,endY]=mapPoint(endpoint.x,endpoint.y,width,height);
    const angle=Math.atan2(endY-previousY,endX-previousX),arrowLength=8*scale,arrowHalf=4.5*scale;
    mapCtx.beginPath();
    mapCtx.moveTo(endX,endY);
    mapCtx.lineTo(endX-Math.cos(angle)*arrowLength+Math.sin(angle)*arrowHalf,endY-Math.sin(angle)*arrowLength-Math.cos(angle)*arrowHalf);
    mapCtx.lineTo(endX-Math.cos(angle)*arrowLength-Math.sin(angle)*arrowHalf,endY-Math.sin(angle)*arrowLength+Math.cos(angle)*arrowHalf);
    mapCtx.closePath(); mapCtx.fillStyle=color; mapCtx.fill();
  }
  mapCtx.restore();
}
function drawStructure(x,y,side,label,robot,scale) {
  const radius=14*scale, hp=Number(robot?.[R.hp]||0),max=Number(robot?.[R.max]||0),alive=!robot||hp>0;
  mapCtx.beginPath(); mapCtx.arc(x,y,radius,0,Math.PI*2); mapCtx.fillStyle=alive?"rgba(7,13,20,.92)":"rgba(28,30,34,.92)"; mapCtx.fill();
  mapCtx.lineWidth=2.6*scale; mapCtx.strokeStyle=alive?colorFor(side):"#687581"; mapCtx.stroke();
  mapCtx.fillStyle=alive?"#f5f9fc":"#8c98a2"; mapCtx.font=`900 ${Math.max(11,11*scale)}px sans-serif`; mapCtx.textAlign="center"; mapCtx.textBaseline="middle"; mapCtx.fillText(label,x,y);
  if (!robot) return;
  const barW=44*scale,barH=5*scale,top=y+radius+5*scale,ratio=max?clamp(hp/max,0,1):0;
  mapCtx.fillStyle="rgba(3,7,11,.9)"; mapCtx.fillRect(x-barW/2,top,barW,barH);
  mapCtx.fillStyle=ratio>.45?COLORS.green:ratio>.2?COLORS.gold:COLORS.red; mapCtx.fillRect(x-barW/2,top,barW*ratio,barH);
  mapCtx.fillStyle="#f2f7fb"; mapCtx.font=`800 ${Math.max(9,8*scale)}px sans-serif`; mapCtx.fillText(hp.toLocaleString(),x,top+barH+8*scale);
}
function drawRobot(x,y,robot,scale,second) {
  const side=robot[R.side], label=ROLE_LABEL[robot[R.type]]||"?", radius=(robot[R.type]==="空中"?14:12.5)*scale;
  const hp=Number(robot[R.hp]||0),max=Number(robot[R.max]||0),dead=hp<=0;
  const avatarColor=dead?(side==="红"?COLORS.deadRed:COLORS.deadBlue):colorFor(side);
  const countered=robot[R.type]==="空中"&&uavCounterStatus(side,second).active;
  if (robot[R.vulnerable]||countered) { mapCtx.beginPath(); mapCtx.arc(x,y,radius+6*scale,0,Math.PI*2); mapCtx.strokeStyle=COLORS.gold; mapCtx.lineWidth=3*scale; mapCtx.stroke(); }
  mapCtx.beginPath(); mapCtx.arc(x+2*scale,y+3*scale,radius+1,0,Math.PI*2); mapCtx.fillStyle="rgba(0,0,0,.5)"; mapCtx.fill();
  mapCtx.beginPath(); mapCtx.arc(x,y,radius,0,Math.PI*2); mapCtx.fillStyle=avatarColor; mapCtx.fill(); mapCtx.strokeStyle="#f4f9fc"; mapCtx.lineWidth=1.4*scale; mapCtx.stroke();
  mapCtx.fillStyle="#fff"; mapCtx.font=`900 ${Math.max(10,(label==="AI"?9:11)*scale)}px sans-serif`; mapCtx.textAlign="center"; mapCtx.textBaseline="middle"; mapCtx.fillText(label,x,y);
  if (robot[R.yaw]!=null) {
    const angle=(Number(robot[R.yaw])-90)*Math.PI/180,ux=Math.cos(angle),uy=Math.sin(angle),px=-uy,py=ux;
    const base=radius+1.5*scale,tip=radius+9*scale,half=3.2*scale;
    mapCtx.beginPath();
    mapCtx.moveTo(x+ux*tip,y+uy*tip);
    mapCtx.lineTo(x+ux*base+px*half,y+uy*base+py*half);
    mapCtx.lineTo(x+ux*base-px*half,y+uy*base-py*half);
    mapCtx.closePath();
    mapCtx.fillStyle="#fff";mapCtx.fill();
    mapCtx.strokeStyle="rgba(3,7,11,.88)";mapCtx.lineWidth=.9*scale;mapCtx.stroke();
  }
  const ratio=max?clamp(hp/max,0,1):0,barW=radius*2.5,barH=4*scale,top=y-radius-12*scale;
  mapCtx.fillStyle="rgba(3,7,11,.9)";mapCtx.fillRect(x-barW/2,top,barW,barH);mapCtx.fillStyle=ratio>.45?COLORS.green:ratio>.2?COLORS.gold:COLORS.red;mapCtx.fillRect(x-barW/2,top,barW*ratio,barH);
  if (robot[R.type]!=="工程") {
    const caliber=robot[R.a42]!=null?"42":robot[R.a17]!=null?"17":"";
    const fired=caliber==="42"?robot[R.a42]:robot[R.a17];
    if (caliber&&fired!=null) {
      const ammoText=`已发 ${Math.max(0,Math.round(Number(fired)||0)).toLocaleString()}`;
      mapCtx.font=`800 ${Math.max(10,9.5*scale)}px sans-serif`;
      mapCtx.textAlign="center";mapCtx.textBaseline="middle";
      const pillH=Math.max(14,14*scale),pillW=mapCtx.measureText(ammoText).width+9*scale,pillY=y+radius+6*scale;
      mapCtx.fillStyle="rgba(3,7,11,.86)";mapCtx.beginPath();mapCtx.roundRect(x-pillW/2,pillY,pillW,pillH,4*scale);mapCtx.fill();
      mapCtx.fillStyle="#f4f8fb";mapCtx.fillText(ammoText,x,pillY+pillH/2);
    }
  }
}

function drawTimeline() {
  if (!state.game) return;
  const height = parseFloat(getComputedStyle(timelineCanvas).height) || 180;
  const {width} = canvasSize(timelineCanvas,5.5,height);
  timelineCtx.clearRect(0,0,width,height); timelineCtx.fillStyle="#0b1620";timelineCtx.fillRect(34,4,width-44,height-28);
  const chart={x:34,y:4,w:width-44,h:height-28},duration=state.game.info.duration;
  timelineCtx.font="11px sans-serif";timelineCtx.textAlign="center";timelineCtx.textBaseline="top";
  for(let i=0;i<8;i++){const x=chart.x+chart.w*i/7;timelineCtx.strokeStyle="rgba(82,106,121,.35)";timelineCtx.beginPath();timelineCtx.moveTo(x,chart.y);timelineCtx.lineTo(x,chart.y+chart.h);timelineCtx.stroke();timelineCtx.fillStyle="#8ba0b1";timelineCtx.fillText(`${Math.round(duration*i/7)}s`,x,chart.y+chart.h+5);}
  const bucketCount=Math.max(50,Math.min(Math.floor(chart.w/3),duration)),buckets=Array.from({length:bucketCount},()=>[0,0,0]);
  for(const [sec,shot,hit,other] of state.game.timeline){const index=Math.min(bucketCount-1,Math.floor(sec/duration*bucketCount));buckets[index][0]+=shot;buckets[index][1]+=hit;buckets[index][2]+=other;}
  const max=Math.max(1,...buckets.map(values=>values.reduce((a,b)=>a+b,0))),barW=chart.w/bucketCount;
  buckets.forEach((values,index)=>{let bottom=chart.y+chart.h;values.forEach((value,i)=>{if(!value)return;const h=chart.h*value/max;timelineCtx.fillStyle=[COLORS.blue,COLORS.red,COLORS.gold][i];timelineCtx.fillRect(chart.x+index*barW,bottom-h,Math.max(1,barW-.4),h);bottom-=h;});});
  const cursor=chart.x+chart.w*state.playhead/duration;timelineCtx.strokeStyle="#fff";timelineCtx.lineWidth=1.5;timelineCtx.beginPath();timelineCtx.moveTo(cursor,chart.y-2);timelineCtx.lineTo(cursor,chart.y+chart.h+2);timelineCtx.stroke();
}

function seek(second) {
  if (!state.game) return;
  state.playhead=clamp(Number(second),0,state.game.info.duration);state.lastSecond=-1;renderState(Math.floor(state.playhead));state.dirty=true;
  rememberPlayhead();
}
function stopPlayback(){state.playing=false;$("#play-button").textContent="▶ 播放";}
function togglePlayback(){
  if(!state.game){showToast("比赛数据尚未载入，请稍候再播放");return;}
  if(state.playhead>=state.game.info.duration)seek(0);
  state.playing=!state.playing;
  $("#play-button").textContent=state.playing?"Ⅱ 暂停":"▶ 播放";
  state.lastAnimation=performance.now();
}
function animation(now){
  const dt=Math.min(.2,(now-state.lastAnimation)/1000);state.lastAnimation=now;
  if(state.playing&&state.game){state.playhead+=dt*state.speed;if(state.playhead>=state.game.info.duration){state.playhead=state.game.info.duration;stopPlayback();}const second=Math.floor(state.playhead);if(second!==state.lastSecond){renderState(second);rememberPlayhead();}state.dirty=true;}
  if(state.dirty&&now-state.lastDraw>30){drawMap();drawTimeline();state.lastDraw=now;}
  requestAnimationFrame(animation);
}

$("#region-select").addEventListener("change",()=>{
  if (memoryEnabled) {
    const region=$("#region-select").value;
    memory.region=region;
    memory.selection={region,matchNo:memory.matches[region]||"",gameId:""};
    persistMemory(true);
  }
  loadMatches();
});
$("#match-select").addEventListener("change",()=>{
  if (memoryEnabled) {
    const region=$("#region-select").value,matchNo=$("#match-select").value;
    memory.region=region;memory.matches[region]=matchNo;
    memory.selection={region,matchNo,gameId:memory.rounds[`${region}::${matchNo}`]||""};
    persistMemory(true);
  }
  loadRounds();
});
$("#round-select").addEventListener("change",()=>{rememberSelection(true);loadGame();});
$("#play-button").addEventListener("click",togglePlayback);
$("#back-button").addEventListener("click",()=>seek(state.playhead-5));
$("#forward-button").addEventListener("click",()=>seek(state.playhead+5));
$("#time-slider").addEventListener("input",event=>seek(event.target.value));
$("#speed-select").addEventListener("change",event=>{state.speed=Number(event.target.value);memory.speed=state.speed;persistMemory();});
$("#prediction-button")?.addEventListener("click",togglePrediction);
timelineCanvas.addEventListener("pointerdown",event=>{if(!state.game)return;const rect=timelineCanvas.getBoundingClientRect(),x=clamp(event.clientX-rect.left-34,0,rect.width-44);seek(x/(rect.width-44)*state.game.info.duration);});
$("#map-fullscreen")?.addEventListener("click",async()=>{
  try {
    if(document.fullscreenElement) await document.exitFullscreen();
    else await $(".map-stage").requestFullscreen();
  } catch(error) { showToast("当前浏览器不支持地图全屏"); }
});
new ResizeObserver(()=>{state.dirty=true;drawMap();drawTimeline();}).observe($(".map-stage"));
new ResizeObserver(()=>drawTimeline()).observe($(".timeline-panel"));

function initDisplayControls() {
  const root = document.documentElement;
  const themeButton = $("#theme-toggle");
  const backgroundButton = $("#background-toggle");
  const memoryButton = $("#memory-toggle");
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  const syncLabels = () => {
    const day = root.dataset.theme === "day";
    const simple = root.dataset.background === "simple";
    if (themeButton) themeButton.textContent = day ? "☀ 白昼" : "☾ 黑夜";
    if (backgroundButton) backgroundButton.textContent = simple ? "▤ 简洁背景" : "▧ 动态背景";
    if (memoryButton) memoryButton.textContent = memoryEnabled ? "记忆 开" : "记忆 关";
    memoryButton?.classList.toggle("active",memoryEnabled);
    if (themeMeta) themeMeta.content = day ? "#edf2f6" : "#081019";
  };
  themeButton?.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "day" ? "night" : "day";
    localStorage.setItem("rmuc-dashboard-theme", root.dataset.theme);
    syncLabels(); state.dirty = true; drawTimeline();
  });
  backgroundButton?.addEventListener("click", () => {
    root.dataset.background = root.dataset.background === "simple" ? "fancy" : "simple";
    localStorage.setItem("rmuc-dashboard-background", root.dataset.background);
    syncLabels();
  });
  memoryButton?.addEventListener("click", () => {
    memoryEnabled = !memoryEnabled;
    try { localStorage.setItem(MEMORY_ENABLED_KEY,String(memoryEnabled)); } catch (_) {}
    if (!memoryEnabled) {
      clearTimeout(memorySaveTimer);
      memory = emptyMemory();
      try { localStorage.removeItem(MEMORY_KEY); } catch (_) {}
      showToast("浏览记忆已关闭并清除","success");
    } else {
      rememberSelection(true);
      rememberPlayhead(true);
      showToast("浏览记忆已开启","success");
    }
    syncLabels();
  });
  const rememberedSpeed = Number(memory.speed);
  if (memoryEnabled && [...$("#speed-select").options].some(option=>Number(option.value)===rememberedSpeed)) {
    state.speed = rememberedSpeed;
    $("#speed-select").value = String(rememberedSpeed);
  }
  document.querySelectorAll("[data-scroll-target]").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-scroll-target]").forEach(item => item.classList.toggle("active",item.dataset.scrollTarget===button.dataset.scrollTarget));
      document.getElementById(button.dataset.scrollTarget)?.scrollIntoView({behavior:"smooth",block:"start"});
    });
  });
  syncLabels();
}

window.addEventListener("pagehide",()=>{rememberSelection(true);rememberPlayhead(true);});
initDisplayControls(); init(); requestAnimationFrame(animation);
