#!/usr/bin/env python3
"""Local HTTP adapter for the operator-facing tactical inference engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request

try:
    from .tactical_inference import (
        DEFAULT_CAPABILITIES,
        DEFAULT_MC_SAMPLES,
        DEFAULT_METRICS,
        DEFAULT_OUTPUT,
        DEFAULT_TEAM_PRIORS,
        GROUND_TYPES,
        SIDES,
        TacticalInferenceEngine,
        index_frame,
    )
except ImportError:
    from tactical_inference import (  # type: ignore[no-redef]
        DEFAULT_CAPABILITIES,
        DEFAULT_MC_SAMPLES,
        DEFAULT_METRICS,
        DEFAULT_OUTPUT,
        DEFAULT_TEAM_PRIORS,
        GROUND_TYPES,
        SIDES,
        TacticalInferenceEngine,
        index_frame,
    )


def create_app(engine: TacticalInferenceEngine) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "service": "rmuc-tactical-inference",
                "horizons": engine.horizons,
                "mc_samples": engine.mc_samples,
            }
        )

    @app.post("/infer")
    def infer():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        try:
            second = int(payload["second"])
            info = dict(payload["info"])
            raw_frames = payload["frames"]
            frames = {
                int(frame_second): index_frame(rows)
                for frame_second, rows in raw_frames.items()
            }
            side = payload.get("side")
            role = payload.get("robot_type")
            if side is not None and side not in SIDES:
                raise ValueError(f"side must be one of {SIDES}")
            if role is not None and role not in GROUND_TYPES:
                raise ValueError(f"robot_type must be one of {GROUND_TYPES}")
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(engine.predict(frames, second, info, side, role))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--capabilities", type=Path, default=DEFAULT_CAPABILITIES)
    parser.add_argument("--team-priors", type=Path, default=DEFAULT_TEAM_PRIORS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = TacticalInferenceEngine(
        model_path=args.model,
        metrics_path=args.metrics,
        capabilities_path=args.capabilities,
        team_priors_path=args.team_priors,
        mc_samples=args.mc_samples,
    )
    app = create_app(engine)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
