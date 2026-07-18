#!/usr/bin/env python3
"""将 RMUC SQLite 数据导出为可由 GitHub Pages 直接托管的静态站点。"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from web_app import APP_DIR, DEFAULT_DB, MAP_PATH, ApiDatabase


DOCS_DIR = APP_DIR / "docs"


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_if_changed(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == payload:
        return False
    path.write_bytes(payload)
    return True


def build_catalog(db: ApiDatabase):
    regions = db.regions()
    matches = {}
    rounds = {}
    game_ids = []
    for region in regions:
        region_matches = db.matches(region)
        matches[region] = region_matches
        for match in region_matches:
            key = f"{region}::{match['match_no']}"
            match_rounds = db.rounds(region, match["match_no"])
            rounds[key] = match_rounds
            game_ids.extend(item["game_id"] for item in match_rounds)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game_count": len(game_ids),
        "regions": regions,
        "matches": matches,
        "rounds": rounds,
    }, game_ids


def export_frontend():
    source = (APP_DIR / "index.html").read_text(encoding="utf-8")
    source = source.replace('href="/app.css?v=5"', 'href="./app.css?v=5"')
    source = source.replace(
        '<script src="/app.js?v=5"></script>',
        '<script>window.RMUC_STATIC_DATA = true;</script>\n  <script src="./app.js?v=5"></script>',
    )
    write_if_changed(DOCS_DIR / "index.html", source.encode("utf-8"))
    write_if_changed(DOCS_DIR / "app.css", (APP_DIR / "web" / "app.css").read_bytes())
    write_if_changed(DOCS_DIR / "app.js", (APP_DIR / "web" / "app.js").read_bytes())
    write_if_changed(DOCS_DIR / ".nojekyll", b"")
    destination = DOCS_DIR / "assets" / "map.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.stat().st_size != MAP_PATH.stat().st_size:
        shutil.copyfile(MAP_PATH, destination)


def parse_args():
    parser = argparse.ArgumentParser(description="导出 RMUC 2026 GitHub Pages 静态站点")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 数据集路径")
    parser.add_argument("--limit", type=int, default=0, help="仅导出前 N 局，用于快速检查")
    return parser.parse_args()


def main():
    args = parse_args()
    db = ApiDatabase(args.db.resolve())
    export_frontend()
    catalog, game_ids = build_catalog(db)
    if args.limit > 0:
        game_ids = game_ids[:args.limit]
        catalog["game_count"] = len(game_ids)
        catalog["preview_only"] = True
    write_if_changed(DOCS_DIR / "data" / "catalog.json", json_bytes(catalog))

    games_dir = DOCS_DIR / "data" / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    expected = {f"{game_id}.json.gz" for game_id in game_ids}
    for index, game_id in enumerate(game_ids, 1):
        payload = gzip.compress(json_bytes(db.game(game_id)), compresslevel=9, mtime=0)
        write_if_changed(games_dir / f"{game_id}.json.gz", payload)
        if index == 1 or index % 25 == 0 or index == len(game_ids):
            size_mb = sum(path.stat().st_size for path in games_dir.glob("*.json.gz")) / 1024 / 1024
            print(f"已导出 {index}/{len(game_ids)} 局，压缩数据 {size_mb:.1f} MB", flush=True)

    if not args.limit:
        for stale in games_dir.glob("*.json.gz"):
            if stale.name not in expected:
                stale.unlink()

    total_mb = sum(path.stat().st_size for path in DOCS_DIR.rglob("*") if path.is_file()) / 1024 / 1024
    print(f"GitHub Pages 已生成：{DOCS_DIR}", flush=True)
    print(f"共 {len(game_ids)} 局，docs 总大小 {total_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
