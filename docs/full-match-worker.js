"use strict";

// The 421-second simulation performs terrain-aware routing for twelve robots.
// Keep that CPU work off the UI thread so loading and scrolling stay responsive.
self.RMUC_EMBEDDED_PREDICTION = true;
importScripts(
  "./terrain-router.js?v=7",
  "./prediction-worker.js?v=26",
  "./full-match-transformer-policy.js?v=1",
  "./full-match-engine.js?v=19",
);
delete self.RMUC_EMBEDDED_PREDICTION;

let model = null;
let navigation = null;
let activeRequestId = 0;
let transformerModelPromise = null;

function ensureTransformerModel() {
  if (!transformerModelPromise) {
    transformerModelPromise = self.RMUCPredictionCore.loadModel().catch(() => null);
  }
  return transformerModelPromise;
}

async function streamMatch(message) {
  const requestId = message.requestId;
  activeRequestId = requestId;
  const started = performance.now();
  const transformerModel = await ensureTransformerModel();
  if (requestId !== activeRequestId) return;
  let state;
  try {
    const transformerPolicy = transformerModel
      ? self.RMUCFullMatchTransformerPolicy.createPolicy(
        transformerModel, self.RMUCPredictionCore,
      )
      : null;
    state = self.RMUCFullMatchEngine.createMatch(
      model,
      navigation,
      message.redSchool,
      message.blueSchool,
      message.seed,
      self.RMUCTerrainRouter,
      { ...message.matchOptions, transformerPolicy },
    );
  } catch (error) {
    self.postMessage({ type: "error", requestId, message: error?.message || String(error) });
    return;
  }
  let sentEvents = 0;
  self.postMessage({
    type: "started",
    requestId,
    expectedFrames: state.duration + 1,
    state: {
      codes: state.codes, duration: state.duration, seed: state.seed,
      policy: { ...state.policy },
    },
    frame: self.RMUCFullMatchEngine.snapshot(state),
  });

  function pump() {
    if (requestId !== activeRequestId) return;
    const frames = [];
    const sliceStarted = performance.now();
    // Short slices keep the worker responsive to a new matchup request while
    // still filling the playback buffer much faster than real time.
    while (!state.finished && frames.length < 12 && performance.now() - sliceStarted < 8) {
      self.RMUCFullMatchEngine.stepMatch(state);
      frames.push(self.RMUCFullMatchEngine.snapshot(state));
    }
    const events = state.events.slice(sentEvents);
    sentEvents = state.events.length;
    self.postMessage({
      type: "chunk",
      requestId,
      frames,
      events,
      complete: state.finished,
      latencyMs: performance.now() - started,
      policy: { ...state.policy },
    });
    if (!state.finished) self.setTimeout(pump, 0);
  }
  self.setTimeout(pump, 0);
}

self.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "initialize") {
    model = message.model;
    navigation = message.navigation;
    ensureTransformerModel();
    self.postMessage({ type: "ready" });
    return;
  }
  if (message.type !== "run") return;
  if (!model || !navigation) {
    self.postMessage({ type: "error", requestId: message.requestId, message: "沙盘参数尚未加载" });
    return;
  }
  streamMatch(message).catch((error) => {
    self.postMessage({ type: "error", requestId: message.requestId, message: error?.message || String(error) });
  });
};
