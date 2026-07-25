"use strict";

const CONFIG_URL = "./data/heatmaps/config.json?v=1";
const FIELD_HEIGHT_WITH_MARGIN = 17;
const state = {
  config: null,
  region: "all",
  school: "东北大学",
  side: "canonical",
  search: "",
  schoolData: null,
  cache: new Map(),
  renderToken: 0,
};

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
  const { grid_width: width, grid_height: height } = state.config;
  const length = width * height;
  const red = decodeSparse(data.red, length);
  const blue = decodeSparse(data.blue, length);
  const combined = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const blueIndex = state.side === "canonical" ? length - 1 - index : index;
    combined[index] = red[index] + blue[blueIndex];
  }
  const sigmaCells = state.config.gaussian_sigma_metres / state.config.cell_size_metres;
  return gaussianBlur(combined, width, height, sigmaCells);
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
  const density = combinedDensity(state.schoolData);
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
  const fieldTop = canvasHeight / FIELD_HEIGHT_WITH_MARGIN;
  const fieldHeight = canvasHeight * 15 / FIELD_HEIGHT_WITH_MARGIN;
  context.save();
  context.globalCompositeOperation = "screen";
  context.imageSmoothingEnabled = true;
  context.drawImage(layer, 0, fieldTop, canvasWidth, fieldHeight);
  context.restore();
}

function schoolEntry(name) {
  return state.config.schools.find((entry) => entry.school === name);
}

async function loadSchool(name) {
  const entry = schoolEntry(name);
  if (!entry) return;
  state.school = name;
  document.querySelector("#heatmap-loading").classList.remove("hidden");
  document.querySelector("#selected-school").textContent = entry.school;
  document.querySelector("#selected-region").textContent = entry.region;
  document.querySelector("#selected-stats").textContent = `${entry.games} 局 · ${entry.samples.toLocaleString()} 个存活位置样本`;
  document.querySelectorAll("#school-filters button").forEach((button) => {
    button.classList.toggle("active", button.dataset.school === name);
  });
  document.querySelectorAll("#region-filters button").forEach((button) => {
    button.classList.toggle("active", button.dataset.region === state.region);
  });
  try {
    if (!state.cache.has(entry.file)) {
      const response = await fetch(`./data/heatmaps/${entry.file}?v=1`, { cache: "force-cache" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.cache.set(entry.file, await response.json());
    }
    state.schoolData = state.cache.get(entry.file);
    drawHeatmap();
    const url = new URL(location.href);
    url.searchParams.set("school", name);
    history.replaceState(null, "", url);
  } catch (error) {
    document.querySelector("#selected-stats").textContent = `热图读取失败：${error.message}`;
  } finally {
    document.querySelector("#heatmap-loading").classList.add("hidden");
  }
}

function renderRegions() {
  const container = document.querySelector("#region-filters");
  for (const region of ["all", ...state.config.regions]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pill";
    button.dataset.region = region;
    button.textContent = region === "all" ? "全部赛区" : region.replace("赛区", "");
    button.addEventListener("click", () => {
      state.region = region;
      state.search = "";
      document.querySelector("#school-search").value = "";
      const first = region === "all"
        ? schoolEntry(state.school) || state.config.schools[0]
        : state.config.schools.find((entry) => entry.region === region);
      renderSchools();
      if (first) loadSchool(first.school);
    });
    container.appendChild(button);
  }
}

function renderSchools() {
  const container = document.querySelector("#school-filters");
  const keyword = state.search.trim().toLowerCase();
  const schools = state.config.schools.filter((entry) => (
    (state.region === "all" || entry.region === state.region)
    && (!keyword || entry.school.toLowerCase().includes(keyword))
  ));
  container.replaceChildren();
  for (const entry of schools) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.school = entry.school;
    button.textContent = entry.school;
    button.title = `${entry.school} · ${entry.games} 局`;
    button.classList.toggle("active", entry.school === state.school);
    button.addEventListener("click", () => loadSchool(entry.school));
    container.appendChild(button);
  }
}

async function init() {
  try {
    const response = await fetch(CONFIG_URL, { cache: "force-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.config = await response.json();
    document.querySelector("#metric-teams").textContent = state.config.schools.length;
    document.querySelector("#metric-resolution").textContent =
      `${state.config.grid_width} × ${state.config.grid_height}`;
    document.querySelector("#metric-sigma").textContent =
      `σ ${state.config.gaussian_sigma_metres.toFixed(2)}m`;
    const requested = new URL(location.href).searchParams.get("school");
    if (requested && schoolEntry(requested)) state.school = requested;
    renderRegions();
    renderSchools();
    document.querySelectorAll("[data-side]").forEach((button) => {
      button.addEventListener("click", () => {
        state.side = button.dataset.side;
        document.querySelectorAll("[data-side]").forEach((item) => item.classList.toggle("active", item === button));
        drawHeatmap();
      });
    });
    document.querySelector("#school-search").addEventListener("input", (event) => {
      state.search = event.target.value;
      renderSchools();
    });
    await loadSchool(state.school);
  } catch (error) {
    document.querySelector("#heatmap-loading").textContent = `热图配置读取失败：${error.message}`;
    document.querySelector("#selected-stats").textContent = "数据不可用";
  }
}

fieldImage.addEventListener("load", drawHeatmap);
new ResizeObserver(drawHeatmap).observe(document.querySelector("#heatmap-stage"));
init();
