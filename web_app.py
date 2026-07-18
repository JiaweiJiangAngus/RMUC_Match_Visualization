#!/usr/bin/env python3
"""RMUC 2026 响应式 Web 回放服务，仅使用 Python 标准库。"""

from __future__ import annotations

import argparse
import gzip
import ipaddress
import json
import mimetypes
import socket
import sqlite3
import struct
import threading
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"
DEFAULT_DB = APP_DIR.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
MAP_PATH = APP_DIR.parent / "TDT" / "Client" / "resource" / "Map.png"


class ApiDatabase:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path

    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        return conn

    def regions(self):
        order = {"东部赛区": 0, "南部赛区": 1, "北部赛区": 2}
        with self.connect() as conn:
            values = [row[0] for row in conn.execute("SELECT DISTINCT 赛区 FROM matches")]
        return sorted(values, key=lambda value: order.get(value, 99))

    def matches(self, region: str):
        sql = """
            SELECT 场次号,MIN(红方学校),MIN(蓝方学校),COUNT(*)
            FROM matches WHERE 赛区=? GROUP BY 场次号 ORDER BY 场次号
        """
        with self.connect() as conn:
            return [
                {"match_no": row[0], "red": row[1], "blue": row[2], "rounds": row[3]}
                for row in conn.execute(sql, (region,))
            ]

    def rounds(self, region: str, match_no: int):
        sql = """
            SELECT 局号,game_id,红方学校,蓝方学校,胜方,开始时间,时长秒
            FROM matches WHERE 赛区=? AND 场次号=? ORDER BY 局号
        """
        with self.connect() as conn:
            return [
                {
                    "round_no": row[0], "game_id": row[1], "red": row[2], "blue": row[3],
                    "winner": row[4], "started_at": row[5], "duration": row[6],
                }
                for row in conn.execute(sql, (region, match_no))
            ]

    def game(self, game_id: int):
        with self.connect() as conn:
            match = conn.execute(
                """SELECT 赛区,场次号,赛程,局号,game_id,红方学校,蓝方学校,
                          胜方,开始时间,时长秒 FROM matches WHERE game_id=?""",
                (game_id,),
            ).fetchone()
            if match is None:
                raise LookupError(f"game_id {game_id} 不存在")
            info = {
                "region": match[0], "match_no": match[1], "schedule": match[2],
                "round_no": match[3], "game_id": match[4], "red": match[5],
                "blue": match[6], "winner": match[7], "started_at": match[8],
                "duration": match[9],
            }

            frames = defaultdict(list)
            ts_sql = """
                SELECT CAST(时刻秒 AS INT),robot_id,机器人类型,阵营,当前血量,最大血量,
                       x,y,枪口朝向,累计17mm发弹,累计42mm发弹,队伍剩余金币,是否易伤
                FROM timeseries WHERE game_id=? ORDER BY 时刻秒,阵营,robot_id
            """
            for row in conn.execute(ts_sql, (game_id,)):
                # 紧凑数组减少手机端传输量：id,type,side,hp,max,x,y,yaw,a17,a42,coins,vulnerable
                frames[int(row[0])].append(list(row[1:]))

            counts = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT 事件类型,COUNT(*) FROM events WHERE game_id=? GROUP BY 事件类型",
                    (game_id,),
                )
            }
            timeline = defaultdict(lambda: [0, 0, 0])
            for row in conn.execute(
                """SELECT CAST(时刻秒 AS INT),事件类型,COUNT(*) FROM events
                   WHERE game_id=? GROUP BY CAST(时刻秒 AS INT),事件类型""",
                (game_id,),
            ):
                index = 0 if row[1] == "发弹" else 1 if row[1] == "受击" else 2
                timeline[int(row[0])][index] += int(row[2])

            events = []
            ev_sql = """
                SELECT 时刻秒,事件类型,机器人类型,阵营,类别,数值,备注,目标类型
                FROM events WHERE game_id=? AND 事件类型<>'发弹' ORDER BY 时刻秒
            """
            for row in conn.execute(ev_sql, (game_id,)):
                events.append(list(row))

        max_second = max(frames, default=0)
        info["duration"] = max(int(info["duration"] or 0), max_second)
        return {
            "info": info,
            "frames": frames,
            "events": events,
            "event_counts": counts,
            "timeline": [[second, *values] for second, values in sorted(timeline.items())],
        }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RMUCVisualizer/1.0"

    @property
    def app(self):
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/regions":
                return self.send_json({"regions": self.app.db.regions()})
            if parsed.path == "/api/matches":
                query = parse_qs(parsed.query)
                region = query.get("region", [""])[0]
                return self.send_json({"matches": self.app.db.matches(region)})
            if parsed.path == "/api/rounds":
                query = parse_qs(parsed.query)
                region = query.get("region", [""])[0]
                match_no = int(query.get("match_no", ["0"])[0])
                return self.send_json({"rounds": self.app.db.rounds(region, match_no)})
            if parsed.path == "/api/game":
                query = parse_qs(parsed.query)
                game_id = int(query.get("game_id", ["0"])[0])
                return self.send_json(self.app.db.game(game_id))
            if parsed.path == "/api/info":
                return self.send_json({
                    "name": "RMUC 2026 区域赛数据中心",
                    "phone_url": self.app.phone_url,
                    "database": self.app.db.path.name,
                })
            if parsed.path == "/assets/map.png":
                return self.send_file(MAP_PATH, "image/png", cache=True)
            return self.send_static(parsed.path)
        except (ValueError, LookupError) as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.send_json({"error": f"服务器读取失败：{exc}"}, status=500)

    def send_static(self, url_path: str):
        if url_path in ("", "/"):
            return self.send_file(APP_DIR / "index.html", "text/html; charset=utf-8", cache=False)
        relative = unquote(url_path.lstrip("/"))
        candidate = (WEB_DIR / relative).resolve()
        if WEB_DIR.resolve() not in candidate.parents and candidate != WEB_DIR.resolve():
            return self.send_error(403)
        if not candidate.is_file():
            return self.send_error(404)
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_file(candidate, mime, cache=candidate.name != "index.html")

    def send_file(self, path: Path, mime: str, cache: bool = False):
        if not path.is_file():
            return self.send_error(404)
        payload = path.read_bytes()
        self.send_payload(payload, mime, cache=cache)

    def send_json(self, value, status: int = 200):
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_payload(payload, "application/json; charset=utf-8", status=status, cache=False)

    def send_payload(self, payload: bytes, mime: str, status: int = 200, cache: bool = False):
        use_gzip = len(payload) > 1024 and "gzip" in self.headers.get("Accept-Encoding", "")
        if use_gzip:
            payload = gzip.compress(payload, compresslevel=5)
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")


def local_ip() -> str:
    candidates = set()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        candidates.add(sock.getsockname()[0])
    except OSError:
        pass
    finally:
        sock.close()

    try:
        candidates.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass

    # Linux 下补充枚举真实网卡；系统代理/TUN 可能接管上面的 UDP 路由。
    try:
        import fcntl
        for _, name in socket.if_nameindex():
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                packed = struct.pack("256s", name[:15].encode("utf-8"))
                address = fcntl.ioctl(probe.fileno(), 0x8915, packed)[20:24]
                candidates.add(socket.inet_ntoa(address))
            except OSError:
                pass
            finally:
                probe.close()
    except (ImportError, OSError):
        pass

    def score(value: str) -> int:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return -1
        if address.is_loopback or address.is_link_local or address.is_multicast:
            return -1
        # 198.18.0.0/15 常被代理软件用作虚拟地址，其他设备通常无法访问。
        if address in ipaddress.ip_network("198.18.0.0/15"):
            return 0
        if address in ipaddress.ip_network("192.168.0.0/16"):
            return 100
        if address in ipaddress.ip_network("10.0.0.0/8"):
            return 90
        if address in ipaddress.ip_network("172.16.0.0/12"):
            return 80
        return 20

    usable = sorted(candidates, key=lambda value: (score(value), value), reverse=True)
    return usable[0] if usable and score(usable[0]) >= 0 else "127.0.0.1"


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description="RMUC 2026 手机/Windows Web 可视化")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 数据集路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8876, help="监听端口")
    parser.add_argument("--open", action="store_true", help="启动后打开本机浏览器")
    return parser.parse_args()


def main():
    args = parse_args()
    db = ApiDatabase(args.db.resolve())
    server = DashboardServer((args.host, args.port), DashboardHandler)
    server.app = type("DashboardApp", (), {})()
    server.app.db = db
    server.app.phone_url = f"http://{local_ip()}:{args.port}"
    local_url = f"http://127.0.0.1:{args.port}"
    print("RMUC 2026 Web 可视化已启动")
    print(f"本机浏览器：{local_url}")
    print(f"手机/其他 Windows 电脑（同一局域网）：{server.app.phone_url}")
    print("按 Ctrl+C 停止服务")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(local_url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
