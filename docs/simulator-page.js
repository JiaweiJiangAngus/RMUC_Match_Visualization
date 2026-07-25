"use strict";

function initSimulatorPageControls() {
  const root = document.documentElement;
  const themeButton = document.querySelector("#theme-toggle");
  const backgroundButton = document.querySelector("#background-toggle");
  const themeMeta = document.querySelector('meta[name="theme-color"]');

  function syncLabels() {
    const day = root.dataset.theme === "day";
    const simple = root.dataset.background === "simple";
    themeButton.textContent = day ? "☀ 白昼" : "☾ 黑夜";
    backgroundButton.textContent = simple ? "▤ 简洁背景" : "▧ 动态背景";
    themeMeta.content = day ? "#edf2f6" : "#081019";
  }

  themeButton.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "day" ? "night" : "day";
    localStorage.setItem("rmuc-dashboard-theme", root.dataset.theme);
    syncLabels();
    window.dispatchEvent(new Event("resize"));
  });
  backgroundButton.addEventListener("click", () => {
    root.dataset.background = root.dataset.background === "simple" ? "fancy" : "simple";
    localStorage.setItem("rmuc-dashboard-background", root.dataset.background);
    syncLabels();
  });
  syncLabels();
}

initSimulatorPageControls();
