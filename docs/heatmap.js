"use strict";

const CONFIG_URL = "./data/heatmaps/config.json?v=9";
const MAP_WIDTH_METRES = 29;
const MAP_HEIGHT_METRES = 16;
const FIELD_WIDTH_METRES = 28;
const FIELD_HEIGHT_METRES = 15;
const FIELD_X_MARGIN_METRES = (MAP_WIDTH_METRES - FIELD_WIDTH_METRES) / 2;
const FIELD_Y_MARGIN_METRES = (MAP_HEIGHT_METRES - FIELD_HEIGHT_METRES) / 2;
const MODE_PRESENTATION = Object.freeze({
  position: {
    label: "位置热图",
    roleLabel: "兵种",
    fullLabel: "整局位置汇总",
    unit: "个位置样本",
    description: "亮度表示该校机器人在所选比赛时段出现在对应位置的累计密度",
    note: "每个位置样本以 σ=0.22m 的二维正态分布展开",
    aria: "战队位置密度热图",
  },
  shots: {
    label: "打弹热力图",
    roleLabel: "发弹兵种",
    fullLabel: "整局发弹汇总",
    unit: "发",
    description: "亮度表示该校机器人在所选比赛时段从对应位置发射弹丸的累计密度",
    note: "每发弹丸按射手当秒坐标记录，并以 σ=0.22m 的二维正态分布展开",
    aria: "战队打弹位置热力图",
  },
  hits: {
    label: "受击热力图",
    roleLabel: "受击兵种",
    fullLabel: "整局受击汇总",
    unit: "次受击",
    description: "亮度表示该校机器人在所选比赛时段于对应位置被裁判系统记录受击的累计密度",
    note: "包含17mm、42mm、撞击与判罚等受击事件，并以 σ=0.22m 展开",
    aria: "战队受击位置热力图",
  },
  kills: {
    label: "击杀热力图",
    roleLabel: "被击杀兵种",
    fullLabel: "整局击杀汇总",
    unit: "次击杀",
    description: "亮度表示该校造成击杀的敌方机器人于对应位置血量归零的累计密度",
    note: "仅统计有17mm/42mm受击证据的敌方归零；兵种表示被击杀兵种",
    aria: "战队击杀位置热力图",
  },
  deaths: {
    label: "阵亡热力图",
    roleLabel: "阵亡兵种",
    fullLabel: "整局阵亡汇总",
    unit: "次阵亡",
    description: "亮度表示该校机器人在所选比赛时段于对应位置阵亡的累计密度",
    note: "每次阵亡按血量归零当秒坐标记录，并以 σ=0.22m 的二维正态分布展开",
    aria: "战队阵亡位置热力图",
  },
});
const state = {
  config: null,
  region: "all",
  school: "东北大学",
  side: "canonical",
  role: "all",
  mode: "position",
  range: "full",
  windowIndex: 0,
  schoolData: null,
  cache: new Map(),
  densityCache: new Map(),
  renderToken: 0,
};
let playbackTimer = null;

const canvas = document.querySelector("#heatmap-canvas");
const context = canvas.getContext("2d");
const fieldImage = document.querySelector("#field-image");

function decodeSparse(encoded, length) {
  const values = new Float32Array(length);
  if (!encoded) return values;
  for (const pair of encoded.split(",")) {
    if (!pair) continue;
    const separator = pair.indexOf(":");
    const index = Number(pair.slice(0, separator));
    const value = Number(pair.slice(separator + 1));
    if (index >= 0 && index < length && Number.isFinite(value)) values[index] = value;
  }
  return values;
}

function gaussianKernel(sigmaCells) {
  const radius = Math.max(1, Math.ceil(sigmaCells * 3));
  const values = new Float32Array(radius * 2 + 1);
  let total = 0;
  for (let offset = -radius; offset <= radius; offset += 1) {
    const value = Math.exp(-(offset * offset) / (2 * sigmaCells * sigmaCells));
    values[offset + radius] = value;
    total += value;
  }
  for (let index = 0; index < values.length; index += 1) values[index] /= total;
  return { values, radius };
}

function gaussianBlur(source, width, height, sigmaCells) {
  const { values: kernel, radius } = gaussianKernel(sigmaCells);
  const horizontal = new Float32Array(source.length);
  const result = new Float32Array(source.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let density = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleX = Math.max(0, Math.min(width - 1, x + offset));
        density += source[y * width + sampleX] * kernel[offset + radius];
      }
      horizontal[y * width + x] = density;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let density = 0;
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sampleY = Math.max(0, Math.min(height - 1, y + offset));
        density += horizontal[sampleY * width + x] * kernel[offset + radius];
      }
      result[y * width + x] = density;
    }
  }
  return result;
}

function heatColor(value) {
  if (value < 0.2) return [0, Math.round(value * 5 * 229), 255];
  if (value < 0.4) return [0, 229 + Math.round((value - 0.2) * 130), Math.round((0.4 - value) * 5 * 255)];
  if (value < 0.65) return [Math.round((value - 0.4) * 4 * 255), 255, 0];
  if (value < 0.82) return [255, 255 - Math.round((value - 0.65) / 0.17 * 78), 0];
  return [255, Math.round((1 - value) / 0.18 * 177), 0];
}

function combinedDensity(data) {
  const cacheKey = `${state.school}:${state.mode}:${state.side}:${state.role}:${state.range}:${state.windowIndex}`;
  if (state.densityCache.has(cacheKey)) return state.densityCache.get(cacheKey);
  const { grid_width: width, grid_height: height } = state.config;
  const length = width * height;
  const red = decodeSparse(data.red, length);
  const blue = decodeSparse(data.blue, length);
  const combined = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const mirroredIndex = length - 1 - index;
    if (state.side === "canonical") {
      combined[index] = red[index] + blue[mirroredIndex];
    } else if (state.side === "canonical-blue") {
      combined[index] = blue[index] + red[mirroredIndex];
    } else {
      combined[index] = red[index] + blue[index];
    }
  }
  const sigmaCells = state.config.gaussian_sigma_metres / state.config.cell_size_metres;
  const density = gaussianBlur(combined, width, height, sigmaCells);
  state.densityCache.set(cacheKey, density);
  return density;
}

function activeSeries() {
  if (!state.schoolData) return null;
  const seriesName = {
    shots: "shots",
    hits: "hits",
    kills: "kills",
    deaths: "deaths",
  }[state.mode];
  const root = seriesName ? state.schoolData[seriesName] : state.schoolData;
  if (!root) return null;
  return state.role === "all"
    ? root
    : root.roles?.[state.role] || null;
}

function activePayload() {
  const series = activeSeries();
  if (!series) return null;
  if (state.range === "full") return series;
  return series.windows?.[state.windowIndex] || null;
}

function resizeCanvas() {
  const rectangle = document.querySelector("#heatmap-stage").getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rectangle.width * dpr));
  canvas.height = Math.max(1, Math.round(rectangle.height * dpr));
  canvas.style.width = `${rectangle.width}px`;
  canvas.style.height = `${rectangle.height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: rectangle.width, height: rectangle.height };
}

function drawHeatmap() {
  if (!state.config || !state.schoolData || !fieldImage.complete) return;
  const token = ++state.renderToken;
  const payload = activePayload();
  if (!payload) return;
  const density = combinedDensity(payload);
  if (token !== state.renderToken) return;
  const { width: canvasWidth, height: canvasHeight } = resizeCanvas();
  context.clearRect(0, 0, canvasWidth, canvasHeight);
  const gridWidth = state.config.grid_width;
  const gridHeight = state.config.grid_height;
  const peak = density.reduce((maximum, value) => Math.max(maximum, value), 0);
  if (peak <= 0) return;

  const layer = document.createElement("canvas");
  layer.width = gridWidth;
  layer.height = gridHeight;
  const layerContext = layer.getContext("2d");
  const pixels = layerContext.createImageData(gridWidth, gridHeight);
  const logPeak = Math.log1p(peak);
  for (let sourceY = 0; sourceY < gridHeight; sourceY += 1) {
    const targetY = gridHeight - 1 - sourceY;
    for (let x = 0; x < gridWidth; x += 1) {
      const densityValue = density[sourceY * gridWidth + x];
      const value = Math.log1p(densityValue) / logPeak;
      if (value < 0.018) continue;
      const [red, green, blue] = heatColor(value);
      const pixel = (targetY * gridWidth + x) * 4;
      pixels.data[pixel] = red;
      pixels.data[pixel + 1] = green;
      pixels.data[pixel + 2] = blue;
      pixels.data[pixel + 3] = Math.round((0.18 + value * 0.72) * 255);
    }
  }
  layerContext.putImageData(pixels, 0, 0);
  const fieldLeft = canvasWidth * FIELD_X_MARGIN_METRES / MAP_WIDTH_METRES;
  const fieldTop = canvasHeight * FIELD_Y_MARGIN_METRES / MAP_HEIGHT_METRES;
  const fieldWidth = canvasWidth * FIELD_WIDTH_METRES / MAP_WIDTH_METRES;
  const fieldHeight = canvasHeight * FIELD_HEIGHT_METRES / MAP_HEIGHT_METRES;
  context.save();
  context.globalCompositeOperation = "screen";
  context.imageSmoothingEnabled = true;
  context.drawImage(layer, fieldLeft, fieldTop, fieldWidth, fieldHeight);
  context.restore();
}

function schoolEntry(name) {
  if (name === "all") return state.config.aggregate;
  if (name.startsWith("region:")) {
    const region = name.slice("region:".length);
    return state.config.region_aggregates.find((entry) => entry.region === region);
  }
  return state.config.schools.find((entry) => entry.school === name);
}

function regionAggregate(region) {
  return region === "all"
    ? state.config.aggregate
    : state.config.region_aggregates.find((entry) => entry.region === region);
}

function aggregateValue(region) {
  return region === "all" ? "all" : `region:${region}`;
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function updatePlaybackUI() {
  if (!state.config) return;
  const presentation = MODE_PRESENTATION[state.mode] || MODE_PRESENTATION.position;
  const isWindow = state.range === "window";
  const windowSeconds = state.config.window_seconds;
  const start = state.windowIndex * windowSeconds;
  const end = start + windowSeconds;
  const slider = document.querySelector("#window-range");
  slider.max = String(state.config.window_count - 1);
  slider.value = String(state.windowIndex);
  document.querySelector("#window-label").textContent = isWindow
    ? `${formatTime(start)}–${formatTime(end)}`
    : presentation.fullLabel;
  document.querySelector("#map-time").textContent = isWindow
    ? `T + ${formatTime(start)}–${formatTime(end)}`
    : "整局汇总";
  document.querySelector("#range-select").value = state.range;
  document.querySelector("#window-prev").disabled = isWindow && state.windowIndex === 0;
  document.querySelector("#window-next").disabled =
    isWindow && state.windowIndex === state.config.window_count - 1;
  const entry = schoolEntry(state.school);
  if (!entry) return;
  const series = activeSeries();
  const windowSamples = series?.windows?.[state.windowIndex]?.samples || 0;
  const roleLabel = state.role === "all" ? "全部机器人" : state.role;
  document.querySelector("#role-label").textContent = presentation.roleLabel;
  document.querySelector("#selected-school").textContent = `${entry.school} · ${presentation.label}`;
  document.querySelector("#heatmap-description").textContent = presentation.description;
  document.querySelector("#heatmap-kernel-note").textContent = presentation.note;
  canvas.setAttribute("aria-label", presentation.aria);
  document.querySelector("#selected-stats").textContent = isWindow
    ? `${roleLabel} · ${formatTime(start)}–${formatTime(end)} · ${windowSamples.toLocaleString()} ${presentation.unit}`
    : `${entry.games} 局 · ${roleLabel} · ${(series?.samples || 0).toLocaleString()} ${presentation.unit}`;
}

function stopWindowPlayback() {
  if (playbackTimer !== null) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  const button = document.querySelector("#window-play");
  if (button) {
    button.textContent = "▶ 播放";
    button.classList.remove("active");
  }
}

function setWindowIndex(index, stopPlayback = true) {
  if (!state.config) return;
  if (stopPlayback) stopWindowPlayback();
  state.range = "window";
  state.windowIndex = Math.max(0, Math.min(state.config.window_count - 1, Number(index)));
  updatePlaybackUI();
  drawHeatmap();
}

function setRange(range) {
  stopWindowPlayback();
  state.range = range === "window" ? "window" : "full";
  updatePlaybackUI();
  drawHeatmap();
}

function toggleWindowPlayback() {
  if (!state.config || !state.schoolData) return;
  if (playbackTimer !== null) {
    stopWindowPlayback();
    return;
  }
  if (state.range !== "window" || state.windowIndex >= state.config.window_count - 1) {
    state.range = "window";
    state.windowIndex = 0;
  }
  const button = document.querySelector("#window-play");
  button.textContent = "Ⅱ 暂停";
  button.classList.add("active");
  updatePlaybackUI();
  drawHeatmap();
  playbackTimer = setInterval(() => {
    if (state.windowIndex >= state.config.window_count - 1) {
      stopWindowPlayback();
      return;
    }
    setWindowIndex(state.windowIndex + 1, false);
  }, 1100);
}

async function loadSchool(name) {
  const entry = schoolEntry(name);
  if (!entry) return;
  stopWindowPlayback();
  state.school = name;
  state.densityCache.clear();
  document.querySelector("#heatmap-loading").classList.remove("hidden");
  document.querySelector("#selected-school").textContent = entry.school;
  document.querySelector("#selected-region").textContent = entry.region;
  document.querySelector("#selected-stats").textContent = "读取热图数据…";
  const select = document.querySelector("#school-select");
  if ([...select.options].some((option) => option.value === name)) select.value = name;
  try {
    if (!state.cache.has(entry.file)) {
      const response = await fetch(`./data/heatmaps/${entry.file}?v=9`, { cache: "force-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.cache.set(entry.file, await response.json());
    }
    state.schoolData = state.cache.get(entry.file);
    updatePlaybackUI();
    drawHeatmap();
    const url = new URL(location.href);
    url.searchParams.set("school", name);
    url.searchParams.set("mode", state.mode);
    history.replaceState(null, "", url);
  } catch (error) {
    document.querySelector("#selected-stats").textContent = `热图读取失败：${error.message}`;
  } finally {
    document.querySelector("#heatmap-loading").classList.add("hidden");
  }
}

function renderRegions() {
  const select = document.querySelector("#region-select");
  for (const region of ["all", ...state.config.regions]) {
    const option = document.createElement("option");
    option.value = region;
    option.textContent = region === "all" ? "全部赛区" : region;
    select.appendChild(option);
  }
  select.value = state.region;
  select.addEventListener("change", () => {
    state.region = select.value;
    const current = state.config.schools.find((entry) => (
      entry.school === state.school
      && (state.region === "all" || entry.region === state.region)
    ));
    const first = current || regionAggregate(state.region);
    renderSchoolOptions();
    if (first) loadSchool(current ? current.school : aggregateValue(state.region));
  });
}

function renderSchoolOptions() {
  const select = document.querySelector("#school-select");
  const schools = state.config.schools.filter((entry) => (
    state.region === "all" || entry.region === state.region
  ));
  select.replaceChildren();
  const aggregate = regionAggregate(state.region);
  if (aggregate) {
    const option = document.createElement("option");
    option.value = aggregateValue(state.region);
    option.textContent = `${aggregate.school} · ${aggregate.games} 局`;
    option.selected = state.school === option.value;
    select.appendChild(option);
  }
  for (const entry of schools) {
    const option = document.createElement("option");
    option.value = entry.school;
    option.textContent = `${entry.school} · ${entry.games} 局`;
    option.selected = entry.school === state.school;
    select.appendChild(option);
  }
}

function renderRoleOptions() {
  const select = document.querySelector("#role-select");
  for (const role of state.config.roles) {
    const option = document.createElement("option");
    option.value = role;
    option.textContent = role;
    select.appendChild(option);
  }
}

function initDisplayControls() {
  const root = document.documentElement;
  const themeButton = document.querySelector("#theme-toggle");
  const backgroundButton = document.querySelector("#background-toggle");
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  const syncLabels = () => {
    const day = root.dataset.theme === "day";
    const simple = root.dataset.background === "simple";
    themeButton.textContent = day ? "☀ 白昼" : "☾ 黑夜";
    backgroundButton.textContent = simple ? "▤ 简洁背景" : "▧ 动态背景";
    themeMeta.content = day ? "#edf2f6" : "#081019";
  };
  themeButton.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "day" ? "night" : "day";
    localStorage.setItem("rmuc-dashboard-theme", root.dataset.theme);
    syncLabels();
    drawHeatmap();
  });
  backgroundButton.addEventListener("click", () => {
    root.dataset.background = root.dataset.background === "simple" ? "fancy" : "simple";
    localStorage.setItem("rmuc-dashboard-background", root.dataset.background);
    syncLabels();
  });
  syncLabels();
}

async function init() {
  try {
    const response = await fetch(CONFIG_URL, { cache: "force-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.config = await response.json();
    document.querySelector("#metric-teams").textContent = `${state.config.schools.length} 支`;
    document.querySelector("#metric-windows").textContent =
      `${state.config.window_count} × ${state.config.window_seconds}s`;
    document.querySelector("#metric-resolution").textContent =
      `${state.config.grid_width} × ${state.config.grid_height}`;
    document.querySelector("#metric-sigma").textContent =
      `σ ${state.config.gaussian_sigma_metres.toFixed(2)}m`;
    const requested = new URL(location.href).searchParams.get("school");
    const requestedMode = new URL(location.href).searchParams.get("mode");
    const requestedSide = new URL(location.href).searchParams.get("side");
    const requestedRole = new URL(location.href).searchParams.get("role");
    if (requested && schoolEntry(requested)) {
      state.school = requested;
      const requestedEntry = schoolEntry(requested);
      if (requested.startsWith("region:")) state.region = requestedEntry.region;
    }
    if (state.config.modes?.includes(requestedMode)) state.mode = requestedMode;
    if (["all", ...state.config.roles].includes(requestedRole)) {
      state.role = requestedRole;
    }
    if (["canonical", "canonical-blue", "bilateral"].includes(requestedSide)) {
      state.side = requestedSide;
    }
    renderRegions();
    renderSchoolOptions();
    renderRoleOptions();
    document.querySelector("#heatmap-type-select").value = state.mode;
    document.querySelector("#side-select").value = state.side;
    document.querySelector("#role-select").value = state.role;
    document.querySelector("#heatmap-type-select").addEventListener("change", (event) => {
      stopWindowPlayback();
      state.mode = state.config.modes.includes(event.target.value)
        ? event.target.value
        : "position";
      state.densityCache.clear();
      const url = new URL(location.href);
      url.searchParams.set("mode", state.mode);
      history.replaceState(null, "", url);
      updatePlaybackUI();
      drawHeatmap();
    });
    document.querySelector("#side-select").addEventListener("change", (event) => {
      state.side = event.target.value;
      state.densityCache.clear();
      const url = new URL(location.href);
      url.searchParams.set("side", state.side);
      history.replaceState(null, "", url);
      drawHeatmap();
    });
    document.querySelector("#school-select").addEventListener("change", (event) => loadSchool(event.target.value));
    document.querySelector("#role-select").addEventListener("change", (event) => {
      state.role = event.target.value;
      state.densityCache.clear();
      const url = new URL(location.href);
      url.searchParams.set("role", state.role);
      history.replaceState(null, "", url);
      updatePlaybackUI();
      drawHeatmap();
    });
    document.querySelector("#range-select").addEventListener("change", (event) => setRange(event.target.value));
    document.querySelector("#window-prev").addEventListener("click", () => setWindowIndex(state.windowIndex - 1));
    document.querySelector("#window-next").addEventListener("click", () => setWindowIndex(state.windowIndex + 1));
    document.querySelector("#window-play").addEventListener("click", toggleWindowPlayback);
    document.querySelector("#window-range").addEventListener("input", (event) => setWindowIndex(event.target.value));
    updatePlaybackUI();
    await loadSchool(state.school);
  } catch (error) {
    document.querySelector("#heatmap-loading").textContent = `热图配置读取失败：${error.message}`;
    document.querySelector("#selected-stats").textContent = "数据不可用";
  }
}

fieldImage.addEventListener("load", drawHeatmap);
new ResizeObserver(drawHeatmap).observe(document.querySelector("#heatmap-stage"));
initDisplayControls();
init();
