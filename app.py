#!/usr/bin/env python3
"""RMUC 2026 区域赛 SQLite 数据可视化窗口。"""

from __future__ import annotations

import argparse
import bisect
import math
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB = APP_DIR.parent / "RMUC2026区域赛数据" / "rmuc_2026_region_dataset.sqlite"
TDT_MAP_PATH = APP_DIR.parent / "TDT" / "Client" / "resource" / "Map.png"

BG = "#081019"
PANEL = "#101c28"
PANEL_2 = "#152433"
TEXT = "#e8f2fb"
MUTED = "#8193a5"
RED = "#ff526c"
BLUE = "#48a0ff"
GOLD = "#f3bd4d"
GREEN = "#38d39f"


@dataclass
class MatchInfo:
    region: str
    match_no: int
    schedule: str
    round_no: int
    game_id: int
    red_school: str
    blue_school: str
    winner: str
    start_time: str
    duration: int


@dataclass
class RobotState:
    second: int
    robot_id: int
    robot_type: str
    side: str
    school: str
    hp: float
    max_hp: float
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    heading: Optional[float]
    ammo17: Optional[float]
    ammo42: Optional[float]
    coins: Optional[float]
    vulnerable: bool


@dataclass
class EventRow:
    second: float
    event_type: str
    robot_type: str
    side: str
    category: str
    value: Optional[float]
    note: str
    target_type: str


class DataStore:
    """只读、按局查询，避免把 1.2 GB 数据库整体载入内存。"""

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

    def regions(self) -> List[str]:
        preferred = {"东部赛区": 0, "南部赛区": 1, "北部赛区": 2}
        rows = [r[0] for r in self.conn.execute("SELECT DISTINCT 赛区 FROM matches")]
        return sorted(rows, key=lambda value: preferred.get(value, 99))

    def matches(self, region: str) -> List[Tuple[int, str, str]]:
        sql = """
            SELECT 场次号, MIN(红方学校), MIN(蓝方学校)
            FROM matches WHERE 赛区=? GROUP BY 场次号 ORDER BY 场次号
        """
        return [(int(r[0]), r[1], r[2]) for r in self.conn.execute(sql, (region,))]

    def rounds(self, region: str, match_no: int) -> List[MatchInfo]:
        sql = """
            SELECT 赛区,场次号,赛程,局号,game_id,红方学校,蓝方学校,
                   胜方,开始时间,时长秒
            FROM matches WHERE 赛区=? AND 场次号=? ORDER BY 局号
        """
        return [MatchInfo(*tuple(r)) for r in self.conn.execute(sql, (region, match_no))]

    def load_game(self, info: MatchInfo):
        frames: Dict[int, List[RobotState]] = defaultdict(list)
        tracks: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = defaultdict(list)
        sql = """
            SELECT CAST(时刻秒 AS INT),robot_id,机器人类型,阵营,学校名,
                   当前血量,最大血量,x,y,z,枪口朝向,累计17mm发弹,
                   累计42mm发弹,队伍剩余金币,是否易伤
            FROM timeseries WHERE game_id=? ORDER BY 时刻秒,阵营,robot_id
        """
        for r in self.conn.execute(sql, (info.game_id,)):
            state = RobotState(
                int(r[0]), int(r[1]), r[2] or "未知", r[3] or "", r[4] or "",
                float(r[5] or 0), float(r[6] or 0), r[7], r[8], r[9], r[10],
                r[11], r[12], r[13], bool(r[14]),
            )
            frames[state.second].append(state)
            if (
                state.robot_type not in ("基地", "前哨站")
                and state.x is not None and state.y is not None
                and -2 <= state.x <= 30 and -2 <= state.y <= 17
            ):
                tracks[(state.side, state.robot_id)].append(
                    (state.second, float(state.x), float(state.y))
                )

        event_counts = {
            r[0]: int(r[1])
            for r in self.conn.execute(
                "SELECT 事件类型,COUNT(*) FROM events WHERE game_id=? GROUP BY 事件类型",
                (info.game_id,),
            )
        }
        timeline = defaultdict(lambda: {"发弹": 0, "受击": 0, "其他": 0})
        for r in self.conn.execute(
            """
            SELECT CAST(时刻秒 AS INT),事件类型,COUNT(*) FROM events
            WHERE game_id=? GROUP BY CAST(时刻秒 AS INT),事件类型
            """,
            (info.game_id,),
        ):
            group = r[1] if r[1] in ("发弹", "受击") else "其他"
            timeline[int(r[0])][group] += int(r[2])

        visible_events: List[EventRow] = []
        for r in self.conn.execute(
            """
            SELECT 时刻秒,事件类型,机器人类型,阵营,类别,数值,备注,目标类型
            FROM events WHERE game_id=? AND 事件类型<>'发弹'
            ORDER BY 时刻秒
            """,
            (info.game_id,),
        ):
            visible_events.append(
                EventRow(float(r[0]), r[1] or "", r[2] or "", r[3] or "",
                         r[4] or "", r[5], r[6] or "", r[7] or "")
            )
        return frames, tracks, dict(timeline), visible_events, event_counts


class SummaryCard(QFrame):
    def __init__(self, title: str, accent: str, parent=None):
        super().__init__(parent)
        self.setObjectName("summaryCard")
        self.setProperty("accent", accent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(17, 12, 17, 12)
        layout.setSpacing(3)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sub = QLabel("等待载入")
        self.sub.setObjectName("cardSub")
        self.sub.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.sub)

    def set_data(self, value: str, sub: str):
        self.value.setText(value)
        self.sub.setText(sub)


class MiniHealthBar(QWidget):
    def __init__(self, title: str, color: str, reverse: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("miniHealth")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self.title = QLabel(title)
        self.title.setObjectName("miniHealthTitle")
        self.title.setFixedWidth(36)
        self.value = QLabel("—")
        self.value.setObjectName("miniHealthValue")
        self.value.setFixedWidth(98)
        self.value.setAlignment(Qt.AlignRight if not reverse else Qt.AlignLeft)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setMinimumWidth(82)
        self.bar.setFixedHeight(14)
        self.bar.setStyleSheet(
            "QProgressBar{background:#263746;border:0;border-radius:6px;}"
            f"QProgressBar::chunk{{background:{color};border-radius:6px;}}"
        )
        widgets = (self.value, self.bar, self.title) if reverse else (self.title, self.bar, self.value)
        for widget in widgets:
            layout.addWidget(widget, 1 if widget is self.bar else 0)

    def set_state(self, state: Optional[RobotState]):
        if state is None or state.max_hp <= 0:
            self.bar.setValue(0)
            self.value.setText("—")
            return
        ratio = max(0.0, min(1.0, state.hp / state.max_hp))
        self.bar.setValue(round(ratio * 1000))
        self.value.setText(f"{int(state.hp):,}/{int(state.max_hp):,}")


class TeamTopHud(QWidget):
    def __init__(self, side: str, color: str, reverse: bool = False, parent=None):
        super().__init__(parent)
        self.side = side
        self.reverse = reverse
        self.setObjectName("teamTopHud")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.name = QLabel(f"{side}方 · 等待载入")
        self.name.setObjectName("topTeamName")
        self.name.setStyleSheet(f"color:{color};")
        self.name.setAlignment(Qt.AlignRight | Qt.AlignVCenter if reverse else Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.name)

        health_row = QHBoxLayout()
        health_row.setSpacing(12)
        self.base = MiniHealthBar("基地", color, reverse)
        self.outpost = MiniHealthBar("前哨", color, reverse)
        health_widgets = (self.outpost, self.base) if reverse else (self.base, self.outpost)
        for widget in health_widgets:
            health_row.addWidget(widget, 1)
        layout.addLayout(health_row)

        self.special = QLabel("飞镖 —   ·   雷达 —")
        self.special.setObjectName("topSpecial")
        self.special.setAlignment(Qt.AlignRight | Qt.AlignVCenter if reverse else Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.special)

    def set_identity(self, school: str, winner: bool):
        self.name.setText(f"{self.side}方  {school}" + ("  · WIN" if winner else ""))

    def set_structures(self, base: Optional[RobotState], outpost: Optional[RobotState]):
        self.base.set_state(base)
        self.outpost.set_state(outpost)

    def set_special(self, text: str):
        self.special.setText(text)


class BattlefieldWidget(QWidget):
    """轻量 QPainter 战场俯视图，适合随时间滑块实时刷新。"""

    FIELD_W = 28.0
    FIELD_H = 15.0
    # 与 TDT/Client/src/qt/window/map/QtClientWindowMap.cpp 的 StaticMarker 一致。
    STRUCTURES = (
        ("红", "基地", 0.095, 0.500, "基"),
        ("红", "前哨站", 0.393, 0.750, "前"),
        ("蓝", "基地", 0.905, 0.500, "基"),
        ("蓝", "前哨站", 0.607, 0.250, "前"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setObjectName("battlefield")
        self.frames: Dict[int, List[RobotState]] = {}
        self.tracks = {}
        self.second = 0
        self.time_value = 0.0
        self.red_name = "红方"
        self.blue_name = "蓝方"
        self.match_badge = "等待载入比赛"
        self.map_image = QImage(str(TDT_MAP_PATH))

    def set_game(self, frames, tracks, red_name: str, blue_name: str):
        self.frames = frames
        self.tracks = tracks
        self.red_name = red_name
        self.blue_name = blue_name
        self.update()

    def set_second(self, second: float):
        self.time_value = max(0.0, float(second))
        self.second = int(self.time_value)
        self.update()

    def set_match_info(self, info: MatchInfo):
        self.match_badge = (
            f"{info.region}  第{info.match_no}场 · 第{info.round_no}局  "
            f"{info.winner}方胜  {info.duration // 60:02d}:{info.duration % 60:02d}"
        )
        self.update()

    @staticmethod
    def _color(side: str) -> QColor:
        return QColor(RED if side == "红" else BLUE)

    def _point(self, x: float, y: float, rect: QRectF) -> QPointF:
        x = max(0.0, min(self.FIELD_W, x))
        y = max(0.0, min(self.FIELD_H, y))
        return QPointF(
            rect.left() + x / self.FIELD_W * rect.width(),
            rect.bottom() - y / self.FIELD_H * rect.height(),
        )

    @staticmethod
    def _point_uv(u: float, v: float, rect: QRectF) -> QPointF:
        return QPointF(rect.left() + u * rect.width(), rect.top() + v * rect.height())

    def _map_rect(self, bounds: QRectF) -> QRectF:
        if self.map_image.isNull():
            return bounds
        ratio = self.map_image.width() / self.map_image.height()
        if bounds.width() / bounds.height() > ratio:
            height = bounds.height()
            width = height * ratio
        else:
            width = bounds.width()
            height = width / ratio
        return QRectF(
            bounds.center().x() - width / 2,
            bounds.center().y() - height / 2,
            width,
            height,
        )

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))

        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Noto Sans CJK SC", 17, QFont.Bold))
        painter.drawText(QRectF(18, 12, self.width() - 36, 26), Qt.AlignLeft, "实时战场 · 俯视图")
        painter.setFont(QFont("Noto Sans CJK SC", 11))
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(18, 37, self.width() - 36, 20),
            Qt.AlignLeft,
            "TDT Client 2026 场地地图 · 最近 20 秒轨迹 · 坐标单位：米",
        )
        painter.drawText(
            QRectF(self.width() * .48, 16, self.width() * .49 - 18, 22),
            Qt.AlignRight | Qt.AlignVCenter,
            self.match_badge,
        )

        map_bounds = QRectF(25, 67, self.width() - 50, self.height() - 98)
        field = self._map_rect(map_bounds)
        marker_scale = max(1.15, min(1.75, min(field.width() / 760.0, field.height() / 400.0)))
        radius = 12.0
        path = QPainterPath()
        path.addRoundedRect(field, radius, radius)
        painter.fillPath(path, QColor("#0a1118"))
        painter.save()
        painter.setClipPath(path)

        # 直接复用 TDT Client QtUI 的战术地图资源。
        if not self.map_image.isNull():
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(field, self.map_image)
            painter.fillRect(field, QColor(2, 7, 12, 42))
        else:
            painter.fillRect(field, QColor("#102d35"))
            painter.setPen(QPen(QColor(123, 164, 170, 32), 1))
            for i in range(1, 14):
                x = field.left() + field.width() * i / 14
                painter.drawLine(QPointF(x, field.top()), QPointF(x, field.bottom()))
            for i in range(1, 8):
                y = field.top() + field.height() * i / 8
                painter.drawLine(QPointF(field.left(), y), QPointF(field.right(), y))

        # 轨迹
        for key, points in self.tracks.items():
            side, _robot_id = key
            recent = [p for p in points if self.time_value - 20 <= p[0] <= self.time_value]
            if len(recent) < 2:
                continue
            color = self._color(side)
            color.setAlpha(85)
            painter.setPen(QPen(color, 1.6))
            trail = QPainterPath(self._point(recent[0][1], recent[0][2], field))
            for _, x, y in recent[1:]:
                trail.lineTo(self._point(x, y, field))
            painter.drawPath(trail)

        # QtUI 的静态结构标记位置，血量来自当前回放帧。
        states = self.frames.get(self.second, [])
        if not states and self.second > 0:
            states = self.frames.get(self.second - 1, [])
        next_states = {
            (state.side, state.robot_id): state
            for state in self.frames.get(self.second + 1, [])
        }
        structure_states = {
            (state.side, state.robot_type): state
            for state in states
            if state.robot_type in ("基地", "前哨站")
        }
        for side, robot_type, u, v, label in self.STRUCTURES:
            self._draw_structure(
                painter,
                structure_states.get((side, robot_type)),
                self._point_uv(u, v, field),
                side,
                label,
                marker_scale,
            )

        # 机器人实体
        for state in states:
            if state.robot_type in ("基地", "前哨站") or state.x is None or state.y is None:
                continue
            if not (-3 <= state.x <= 31 and -3 <= state.y <= 18):
                continue
            x, y = float(state.x), float(state.y)
            next_state = next_states.get((state.side, state.robot_id))
            alpha = self.time_value - self.second
            if (
                next_state is not None
                and next_state.x is not None and next_state.y is not None
                and -3 <= next_state.x <= 31 and -3 <= next_state.y <= 18
            ):
                x += (float(next_state.x) - x) * alpha
                y += (float(next_state.y) - y) * alpha
            self._draw_robot(painter, state, self._point(x, y, field), marker_scale)

        painter.restore()
        painter.setPen(QPen(QColor("#37505d"), 1.3))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(field, radius, radius)

        # 图例
        painter.setFont(QFont("Noto Sans CJK SC", 11))
        self._legend(painter, field.left(), self.height() - 23, QColor(RED), self.red_name)
        self._legend(painter, field.left() + min(250, field.width() * .45), self.height() - 23, QColor(BLUE), self.blue_name)
        painter.setPen(QColor(MUTED))
        painter.drawText(QRectF(field.right() - 100, self.height() - 31, 100, 22), Qt.AlignRight | Qt.AlignVCenter, f"T + {self.second:03d}s")

    def _draw_structure(
        self,
        painter: QPainter,
        state: Optional[RobotState],
        pos: QPointF,
        side: str,
        label: str,
        scale: float,
    ):
        color = self._color(side)
        alive = state is None or state.hp > 0
        radius = 13.0 * scale
        painter.setBrush(QColor(7, 13, 20, 225) if alive else QColor(26, 28, 32, 225))
        stroke = color if alive else QColor("#687581")
        painter.setPen(QPen(stroke, 2.2))
        painter.drawEllipse(pos, radius, radius)
        painter.setPen(QColor("#f5f9fc") if alive else QColor("#8c98a2"))
        painter.setFont(QFont("Noto Sans CJK SC", max(9, round(9 * scale)), QFont.Black))
        painter.drawText(
            QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2),
            Qt.AlignCenter,
            label,
        )
        if state is None:
            return
        ratio = state.hp / state.max_hp if state.max_hp else 0.0
        ratio = max(0.0, min(1.0, ratio))
        bar_width = 40 * scale
        bar = QRectF(pos.x() - bar_width / 2, pos.y() + radius + 4, bar_width, max(4, 4 * scale))
        painter.fillRect(bar, QColor(3, 7, 11, 220))
        painter.fillRect(
            QRectF(bar.left(), bar.top(), bar.width() * ratio, bar.height()),
            QColor(GREEN if ratio > .45 else GOLD if ratio > .2 else RED),
        )
        painter.setPen(QColor("#eef6fb"))
        painter.setFont(QFont("Noto Sans CJK SC", max(8, round(8 * scale)), QFont.Bold))
        painter.drawText(
            QRectF(pos.x() - 30 * scale, bar.bottom() + 1, 60 * scale, 12 * scale),
            Qt.AlignCenter,
            f"{int(state.hp):,}",
        )

    def _draw_robot(self, painter, state: RobotState, pos: QPointF, scale: float):
        color = self._color(state.side)
        radius = (13.0 if state.robot_type == "空中" else 11.5) * scale
        if state.vulnerable:
            glow = QColor(GOLD)
            glow.setAlpha(190)
            painter.setPen(QPen(glow, 2.3))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(pos, radius + 5 * scale, radius + 5 * scale)

        shadow = QColor(0, 0, 0, 100)
        painter.setBrush(shadow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(pos.x() + 2, pos.y() + 3), radius + 1, radius + 1)
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#f3f8fc"), 1.2))
        painter.drawEllipse(pos, radius, radius)

        short = {"英雄": "1", "工程": "2", "步兵3": "3", "步兵4": "4", "空中": "6", "哨兵": "AI"}.get(state.robot_type, "?")
        painter.setPen(QColor("#ffffff"))
        label_size = 7.5 if short == "AI" else 9.0
        painter.setFont(QFont("Noto Sans CJK SC", max(8, round(label_size * scale)), QFont.Black))
        painter.drawText(QRectF(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2), Qt.AlignCenter, short)

        # 枪口方向（数据角度为罗盘角，缺失时不绘制）
        if state.heading is not None:
            angle = math.radians(float(state.heading) - 90)
            heading_length = radius + 7 * scale
            endpoint = QPointF(pos.x() + math.cos(angle) * heading_length, pos.y() + math.sin(angle) * heading_length)
            painter.setPen(QPen(QColor("#ffffff"), 1.5 * scale))
            painter.drawLine(pos, endpoint)

        # 实体上方微型血条
        hp_ratio = state.hp / state.max_hp if state.max_hp else 0
        hp_ratio = max(0.0, min(1.0, hp_ratio))
        bar_width = radius * 2.5
        bar = QRectF(pos.x() - bar_width / 2, pos.y() - radius - 7 * scale, bar_width, max(3, 3 * scale))
        painter.fillRect(bar, QColor(4, 9, 14, 190))
        hp_color = QColor(GREEN if hp_ratio > .45 else GOLD if hp_ratio > .2 else RED)
        painter.fillRect(QRectF(bar.left(), bar.top(), bar.width() * hp_ratio, bar.height()), hp_color)

    @staticmethod
    def _legend(painter, x, y, color, text):
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(x + 5, y + 5), 5, 5)
        painter.setPen(QColor(MUTED))
        clipped = text if len(text) <= 16 else text[:15] + "…"
        painter.drawText(QRectF(x + 16, y - 4, 220, 20), Qt.AlignVCenter, clipped)


class TeamPanel(QFrame):
    ROBOT_ORDER = ["基地", "前哨站", "英雄", "工程", "步兵3", "步兵4", "哨兵", "空中"]

    def __init__(self, side: str, color: str, parent=None):
        super().__init__(parent)
        self.side = side
        self.color = color
        self.setObjectName("teamPanel")
        self.setMinimumWidth(275)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 13)
        layout.setSpacing(9)
        top = QHBoxLayout()
        self.name = QLabel(f"{side}方")
        self.name.setObjectName("teamName")
        self.name.setStyleSheet(f"color:{color};")
        self.name.setWordWrap(True)
        self.coins = QLabel("金币 —")
        self.coins.setObjectName("coins")
        top.addWidget(self.name, 1)
        top.addWidget(self.coins)
        layout.addLayout(top)
        self.rows = {}
        for robot_type in self.ROBOT_ORDER:
            row = QHBoxLayout()
            row.setSpacing(7)
            label = QLabel(robot_type)
            label.setObjectName("robotLabel")
            label.setFixedWidth(54)
            progress = QProgressBar()
            progress.setRange(0, 1000)
            progress.setValue(0)
            progress.setTextVisible(False)
            progress.setFixedHeight(11)
            progress.setStyleSheet(
                "QProgressBar{background:#263746;border:0;border-radius:5px;}"
                f"QProgressBar::chunk{{background:{color};border-radius:5px;}}"
            )
            value = QLabel("—")
            value.setObjectName("hpValue")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setFixedWidth(86)
            row.addWidget(label)
            row.addWidget(progress, 1)
            row.addWidget(value)
            layout.addLayout(row)
            self.rows[robot_type] = (progress, value)
        layout.addStretch(1)
        self.summary = QLabel("存活 — · 17mm — · 42mm —")
        self.summary.setObjectName("teamSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    def set_team(self, name: str):
        self.name.setText(f"{self.side}方 · {name}")

    def update_states(self, states: List[RobotState]):
        by_type = {s.robot_type: s for s in states if s.side == self.side}
        coin_values = [s.coins for s in by_type.values() if s.coins is not None]
        coins = int(max(coin_values)) if coin_values else 0
        self.coins.setText(f"剩余金币 {coins:,}")
        mobile = [s for s in by_type.values() if s.robot_type not in ("基地", "前哨站")]
        alive = sum(1 for s in mobile if s.hp > 0)
        ammo17 = int(sum(s.ammo17 or 0 for s in mobile))
        ammo42 = int(sum(s.ammo42 or 0 for s in mobile))
        self.summary.setText(
            f"在线 {alive}/{len(mobile)}    累计发弹\n"
            f"17mm  {ammo17:,}    42mm  {ammo42:,}"
        )
        for robot_type, (progress, value) in self.rows.items():
            state = by_type.get(robot_type)
            if not state:
                progress.setValue(0)
                value.setText("—")
                continue
            ratio = state.hp / state.max_hp if state.max_hp else 0
            progress.setValue(round(max(0.0, min(1.0, ratio)) * 1000))
            value.setText(f"{int(state.hp)}/{int(state.max_hp)}")


class TimelineWidget(QWidget):
    seekRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(195)
        self.setMaximumHeight(220)
        self.duration = 420
        self.second = 0
        self.data = {}
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, data, duration):
        self.data = data
        self.duration = max(1, duration)
        self.update()

    def set_second(self, second):
        self.second = second
        self.update()

    def _chart_rect(self):
        return QRectF(45, 28, self.width() - 63, self.height() - 54)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL))
        painter.setFont(QFont("Noto Sans CJK SC", 14, QFont.Bold))
        painter.setPen(QColor(TEXT))
        painter.drawText(QRectF(15, 6, 180, 20), Qt.AlignVCenter, "全局事件时间线")
        painter.setFont(QFont("Noto Sans CJK SC", 11))
        painter.setPen(QColor(MUTED))
        painter.drawText(QRectF(self.width() - 250, 6, 235, 20), Qt.AlignRight | Qt.AlignVCenter, "蓝：发弹  红：受击  黄：其他")
        rect = self._chart_rect()
        painter.fillRect(rect, QColor("#0b1620"))
        for i in range(8):
            x = rect.left() + rect.width() * i / 7
            painter.setPen(QPen(QColor(72, 94, 108, 65), 1))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            sec = int(self.duration * i / 7)
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(x - 20, rect.bottom() + 4, 40, 17), Qt.AlignCenter, f"{sec}s")

        bucket_count = max(50, min(int(rect.width() / 3), self.duration))
        buckets = [[0, 0, 0] for _ in range(bucket_count)]
        for second, counts in self.data.items():
            idx = min(bucket_count - 1, int(second / self.duration * bucket_count))
            buckets[idx][0] += counts.get("发弹", 0)
            buckets[idx][1] += counts.get("受击", 0)
            buckets[idx][2] += counts.get("其他", 0)
        max_total = max([sum(v) for v in buckets] + [1])
        width = rect.width() / bucket_count
        for i, values in enumerate(buckets):
            x = rect.left() + i * width
            bottom = rect.bottom()
            for amount, color in zip(values, (BLUE, RED, GOLD)):
                if amount <= 0:
                    continue
                height = rect.height() * amount / max_total
                painter.fillRect(QRectF(x, bottom - height, max(1.0, width - .4), height), QColor(color))
                bottom -= height

        cursor_x = rect.left() + rect.width() * self.second / self.duration
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawLine(QPointF(cursor_x, rect.top() - 3), QPointF(cursor_x, rect.bottom() + 3))
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cursor_x, rect.top() - 3), 3.5, 3.5)

    def mousePressEvent(self, event):
        rect = self._chart_rect()
        if rect.contains(event.pos()):
            ratio = (event.x() - rect.left()) / rect.width()
            self.seekRequested.emit(round(max(0.0, min(1.0, ratio)) * self.duration))


class MainWindow(QMainWindow):
    def __init__(self, db_path: Path):
        super().__init__()
        self.store = DataStore(db_path)
        self.current_info: Optional[MatchInfo] = None
        self.round_infos: List[MatchInfo] = []
        self.frames = {}
        self.tracks = {}
        self.timeline_data = {}
        self.events: List[EventRow] = []
        self.event_seconds: List[float] = []
        self.event_counts = {}
        self.playhead = 0.0
        self.speed = 1.0
        self.duration = 420

        self.setWindowTitle("RMUC 2026 区域赛 · 数据可视化")
        self.resize(1700, 1040)
        self.setMinimumSize(1360, 860)
        self._build_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick)

        self._load_regions()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 18)
        outer.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("RMUC 2026 区域赛数据中心")
        title.setObjectName("appTitle")
        subtitle = QLabel("3 个赛区 · 613 局比赛 · 逐秒状态与事件回放")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)

        self.region_combo = self._combo(142)
        self.match_combo = self._combo(310)
        self.round_combo = self._combo(100)
        header.addWidget(self._field("赛区", self.region_combo))
        header.addWidget(self._field("场次", self.match_combo))
        header.addWidget(self._field("局次", self.round_combo))
        outer.addLayout(header)

        match_hud = QFrame()
        match_hud.setObjectName("matchHud")
        match_hud.setFixedHeight(104)
        match_hud_layout = QHBoxLayout(match_hud)
        match_hud_layout.setContentsMargins(14, 6, 14, 6)
        match_hud_layout.setSpacing(14)
        self.top_red = TeamTopHud("红", RED)
        self.hud_center = QLabel("比赛回放")
        self.hud_center.setObjectName("hudCenter")
        self.hud_center.setAlignment(Qt.AlignCenter)
        self.top_blue = TeamTopHud("蓝", BLUE, reverse=True)
        match_hud_layout.addWidget(self.top_red, 5)
        match_hud_layout.addWidget(self.hud_center, 2)
        match_hud_layout.addWidget(self.top_blue, 5)
        outer.addWidget(match_hud)

        middle = QSplitter(Qt.Horizontal)
        middle.setChildrenCollapsible(False)
        self.red_team = TeamPanel("红", RED)
        self.blue_team = TeamPanel("蓝", BLUE)
        self.field = BattlefieldWidget()
        middle.addWidget(self.red_team)
        middle.addWidget(self.field)
        middle.addWidget(self.blue_team)
        middle.setStretchFactor(0, 0)
        middle.setStretchFactor(1, 1)
        middle.setStretchFactor(2, 0)
        middle.setSizes([295, 1010, 295])
        outer.addWidget(middle, 1)

        control = QFrame()
        control.setObjectName("controlBar")
        control_layout = QHBoxLayout(control)
        control_layout.setContentsMargins(14, 8, 14, 8)
        self.play_btn = QPushButton("▶  播放")
        self.play_btn.setObjectName("primaryButton")
        self.play_btn.setFixedWidth(112)
        self.play_btn.clicked.connect(self._toggle_play)
        back = QPushButton("−5s")
        forward = QPushButton("+5s")
        back.clicked.connect(lambda: self._seek(self.slider.value() - 5))
        forward.clicked.connect(lambda: self._seek(self.slider.value() + 5))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.duration)
        self.slider.valueChanged.connect(self._on_slider)
        self.time_label = QLabel("00:00 / 07:00")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setFixedWidth(138)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5×", "1×", "2×", "4×"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setFixedWidth(82)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(back)
        control_layout.addWidget(forward)
        control_layout.addWidget(self.slider, 1)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(self.speed_combo)
        outer.addWidget(control)

        bottom = QSplitter(Qt.Horizontal)
        bottom.setChildrenCollapsible(False)
        self.timeline = TimelineWidget()
        self.timeline.seekRequested.connect(self._seek)
        bottom.addWidget(self.timeline)
        self.event_table = QTableWidget(0, 5)
        self.event_table.setHorizontalHeaderLabels(["时刻", "阵营", "事件", "主体", "详情"])
        self.event_table.setObjectName("eventTable")
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.event_table.setSelectionMode(QTableWidget.NoSelection)
        self.event_table.setFocusPolicy(Qt.NoFocus)
        self.event_table.horizontalHeader().setStretchLastSection(True)
        self.event_table.verticalHeader().setDefaultSectionSize(28)
        self.event_table.setColumnWidth(0, 72)
        self.event_table.setColumnWidth(1, 60)
        self.event_table.setColumnWidth(2, 98)
        self.event_table.setColumnWidth(3, 88)
        self.event_table.setMinimumWidth(620)
        self.event_table.setMinimumHeight(195)
        self.event_table.setMaximumHeight(220)
        bottom.addWidget(self.event_table)
        bottom.setStretchFactor(0, 5)
        bottom.setStretchFactor(1, 5)
        outer.addWidget(bottom)

        self.region_combo.currentTextChanged.connect(self._region_changed)
        self.match_combo.currentIndexChanged.connect(self._match_changed)
        self.round_combo.currentIndexChanged.connect(self._round_changed)

    @staticmethod
    def _combo(width):
        combo = QComboBox()
        combo.setFixedWidth(width)
        return combo

    @staticmethod
    def _field(label_text, widget):
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return box

    def _apply_style(self):
        self.setStyleSheet(f"""
            * {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; font-size: 12px; }}
            QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
            QLabel {{ background: transparent; }}
            QLabel#appTitle {{ font-size: 28px; font-weight: 700; color: #f0f7fd; }}
            QLabel#appSubtitle {{ font-size: 13px; color: {MUTED}; }}
            QLabel#fieldLabel {{ font-size: 12px; color: {MUTED}; }}
            QComboBox {{ background: {PANEL_2}; border: 1px solid #294052; border-radius: 7px;
                         padding: 8px 12px; min-height: 24px; color: {TEXT}; font-size: 13px; }}
            QComboBox:hover {{ border-color: #49718c; }}
            QComboBox::drop-down {{ border: 0; width: 22px; }}
            QComboBox QAbstractItemView {{ background: {PANEL_2}; color: {TEXT};
                                          selection-background-color: #275071; border: 1px solid #3a5264; }}
            QFrame#summaryCard, QFrame#teamPanel, QFrame#controlBar, QFrame#matchHud {{ background: {PANEL};
                     border: 1px solid #203444; border-radius: 10px; }}
            QLabel#hudRed {{ color: {RED}; font-size: 12px; font-weight: 700; }}
            QLabel#hudCenter {{ color: {GOLD}; font-size: 16px; font-weight: 800; }}
            QLabel#hudBlue {{ color: {BLUE}; font-size: 12px; font-weight: 700; }}
            QLabel#hudEvents {{ color: {GREEN}; font-size: 10px; }}
            QWidget#teamTopHud, QWidget#miniHealth {{ background: transparent; }}
            QLabel#topTeamName {{ font-size: 14px; font-weight: 800; }}
            QLabel#miniHealthTitle {{ color: #a8bac8; font-size: 11px; font-weight: 700; }}
            QLabel#miniHealthValue {{ color: #e3edf4; font-size: 11px; font-weight: 700;
                                      font-family: "JetBrains Mono", monospace; }}
            QLabel#topSpecial {{ color: #a9bfce; font-size: 11px; font-weight: 600; }}
            QLabel#cardTitle {{ color: {MUTED}; font-size: 10px; }}
            QLabel#cardValue {{ color: {TEXT}; font-size: 15px; font-weight: 700; }}
            QLabel#cardSub {{ color: {MUTED}; font-size: 10px; }}
            QLabel#teamName {{ font-size: 15px; font-weight: 800; }}
            QLabel#coins {{ color: {GOLD}; font-size: 12px; font-weight: 700; }}
            QLabel#robotLabel {{ color: #bdccd7; font-size: 11px; font-weight: 600; }}
            QLabel#hpValue {{ color: #c3d1db; font-size: 11px; font-weight: 600; }}
            QLabel#teamSummary {{ color: #a7bdcc; background: #0b1721; border: 1px solid #2c4558;
                                  border-radius: 7px; padding: 12px; font-size: 12px; font-weight: 600; }}
            QLabel#timeLabel {{ font-family: "JetBrains Mono", monospace; font-size: 13px;
                                font-weight: 700; color: #d6e4ee; }}
            QPushButton {{ background: #172838; color: #b7c7d4; border: 1px solid #2a4254;
                           border-radius: 7px; padding: 8px 12px; font-size: 12px; font-weight: 600; }}
            QPushButton:hover {{ background: #20384c; color: white; }}
            QPushButton#primaryButton {{ background: #16755e; color: white; border-color: #26977c; font-weight: 700; }}
            QPushButton#primaryButton:hover {{ background: #1a8b70; }}
            QSlider::groove:horizontal {{ height: 4px; background: #2a3c4b; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: {GREEN}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: #f3fbff; width: 14px; margin: -5px 0; border-radius: 7px; }}
            QSplitter::handle {{ background: {BG}; width: 9px; height: 9px; }}
            QTableWidget#eventTable {{ background: {PANEL}; alternate-background-color: #122130;
                color: #c4d3df; border: 1px solid #203444; border-radius: 8px; gridline-color: #20303e;
                font-size: 12px; }}
            QTableWidget#eventTable::item {{ padding: 4px; }}
            QHeaderView::section {{ background: #162737; color: {MUTED}; border: 0;
                border-right: 1px solid #263b4b; padding: 6px; font-size: 12px; font-weight: 700; }}
            QScrollBar:vertical {{ background: #101b25; width: 8px; }}
            QScrollBar::handle:vertical {{ background: #344b5d; border-radius: 4px; min-height: 20px; }}
        """)

    def _load_regions(self):
        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        self.region_combo.addItems(self.store.regions())
        self.region_combo.blockSignals(False)
        if self.region_combo.count():
            self._region_changed(self.region_combo.currentText())

    def _region_changed(self, region: str):
        if not region:
            return
        rows = self.store.matches(region)
        self.match_combo.blockSignals(True)
        self.match_combo.clear()
        for match_no, red, blue in rows:
            label = f"第{match_no}场 · {self._short(red)} vs {self._short(blue)}"
            self.match_combo.addItem(label, match_no)
        self.match_combo.blockSignals(False)
        if self.match_combo.count():
            self.match_combo.setCurrentIndex(0)
            self._match_changed(0)

    def _match_changed(self, index: int):
        if index < 0:
            return
        match_no = self.match_combo.itemData(index)
        if match_no is None:
            return
        self.round_infos = self.store.rounds(self.region_combo.currentText(), int(match_no))
        self.round_combo.blockSignals(True)
        self.round_combo.clear()
        for info in self.round_infos:
            self.round_combo.addItem(f"第{info.round_no}局", info.game_id)
        self.round_combo.blockSignals(False)
        if self.round_combo.count():
            self.round_combo.setCurrentIndex(0)
            self._round_changed(0)

    def _round_changed(self, index: int):
        if index < 0 or index >= len(self.round_infos):
            return
        self._stop()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            info = self.round_infos[index]
            frames, tracks, timeline, events, counts = self.store.load_game(info)
            self.current_info = info
            self.frames = frames
            self.tracks = tracks
            self.timeline_data = timeline
            self.events = events
            self.event_seconds = [e.second for e in events]
            self.event_counts = counts
            self.duration = max(info.duration, max(frames.keys(), default=0))
            self.slider.blockSignals(True)
            self.slider.setRange(0, self.duration)
            first_second = min(frames.keys(), default=0)
            self.slider.setValue(first_second)
            self.slider.blockSignals(False)
            self.playhead = float(first_second)
            self.field.set_game(frames, tracks, info.red_school, info.blue_school)
            self.timeline.set_data(timeline, self.duration)
            self.red_team.set_team(info.red_school)
            self.blue_team.set_team(info.blue_school)
            self._update_hud()
            self.field.set_match_info(info)
            self._set_second(first_second)
        except sqlite3.Error as exc:
            QMessageBox.critical(self, "读取失败", f"无法读取该局数据：\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _update_hud(self):
        info = self.current_info
        if not info:
            return
        self.top_red.set_identity(info.red_school, info.winner == "红")
        self.top_blue.set_identity(info.blue_school, info.winner == "蓝")
        self.hud_center.setText(
            f"第{info.match_no}场 · 第{info.round_no}局   {info.winner}方胜   "
            f"{self._format_duration(info.duration)}"
        )

    def _toggle_play(self):
        if self.timer.isActive():
            self._stop()
        else:
            if self.slider.value() >= self.duration:
                self._seek(0)
            self.timer.start()
            self.play_btn.setText("Ⅱ  暂停")

    def _stop(self):
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "play_btn"):
            self.play_btn.setText("▶  播放")

    def _tick(self):
        self.playhead += self.speed * self.timer.interval() / 1000.0
        if self.playhead >= self.duration:
            self._seek(self.duration)
            self._stop()
            return
        second = int(self.playhead)
        self.slider.blockSignals(True)
        self.slider.setValue(second)
        self.slider.blockSignals(False)
        self._set_second(second, self.playhead)

    def _speed_changed(self, index):
        self.speed = [0.5, 1.0, 2.0, 4.0][max(0, index)]

    def _seek(self, second: int):
        second = max(0, min(self.duration, int(second)))
        self.playhead = float(second)
        self.slider.setValue(second)

    def _on_slider(self, second: int):
        self.playhead = float(second)
        self._set_second(second)

    def _set_second(self, second: int, visual_time: Optional[float] = None):
        self.field.set_second(second if visual_time is None else visual_time)
        self.timeline.set_second(second)
        states = self.frames.get(second, [])
        if not states and second > 0:
            states = self.frames.get(second - 1, [])
        self.red_team.update_states(states)
        self.blue_team.update_states(states)
        self._update_live_hud(states, second)
        self.time_label.setText(f"{self._format_duration(second)} / {self._format_duration(self.duration)}")
        self._update_event_table(second)

    def _update_live_hud(self, states: List[RobotState], second: int):
        info = self.current_info
        if not info:
            return
        structures = {
            (state.side, state.robot_type): state
            for state in states
            if state.robot_type in ("基地", "前哨站")
        }

        self.top_red.set_structures(
            structures.get(("红", "基地")),
            structures.get(("红", "前哨站")),
        )
        self.top_blue.set_structures(
            structures.get(("蓝", "基地")),
            structures.get(("蓝", "前哨站")),
        )

        end = bisect.bisect_right(self.event_seconds, second)
        past_events = self.events[:end]

        def special_text(side: str) -> str:
            other_side = "蓝" if side == "红" else "红"
            all_dart = [e for e in self.events if e.side == side]
            past_dart = [e for e in past_events if e.side == side]
            gate_total = sum(e.event_type == "飞镖闸门开" for e in all_dart)
            gate_count = sum(e.event_type == "飞镖闸门开" for e in past_dart)
            all_hits = [e for e in all_dart if e.event_type == "飞镖命中"]
            dart_hits = [e for e in past_dart if e.event_type == "飞镖命中"]
            dart_damage = int(sum(abs(e.value or 0) for e in dart_hits))
            all_counter = [
                e for e in self.events
                if e.event_type == "雷达反制UAV" and e.side == other_side
            ]
            past_counter = [
                e for e in past_events
                if e.event_type == "雷达反制UAV" and e.side == other_side
            ]
            marked_now = sum(
                state.vulnerable
                for state in states
                if state.side == other_side and state.robot_type not in ("基地", "前哨站")
            )
            return (
                f"飞镖  门 {gate_count}/{gate_total} · 命中 {len(dart_hits)}/{len(all_hits)}"
                f" · 伤害 {dart_damage:,}    雷达  标记 {marked_now} · 反制 {len(past_counter)}/{len(all_counter)}"
            )

        self.top_red.set_special(special_text("红"))
        self.top_blue.set_special(special_text("蓝"))

    def _update_event_table(self, second: int):
        # 当前时刻之前最近 7 条事件，时间向前推进时自然形成赛事解说流。
        end = bisect.bisect_right(self.event_seconds, second)
        rows = self.events[max(0, end - 9):end]
        rows.reverse()
        self.event_table.setRowCount(len(rows))
        for row_idx, event in enumerate(rows):
            details = event.category
            if event.value is not None:
                value = int(event.value) if float(event.value).is_integer() else event.value
                details += (" · " if details else "") + str(value)
            if event.target_type:
                details += (" → " if details else "→ ") + event.target_type
            if event.note:
                details += (" · " if details else "") + event.note
            values = [f"{event.second:.0f}s", event.side, event.event_type, event.robot_type, details or "—"]
            for col, text_value in enumerate(values):
                item = QTableWidgetItem(text_value)
                if col == 1:
                    item.setForeground(QColor(RED if text_value == "红" else BLUE))
                elif col == 2 and text_value == "受击":
                    item.setForeground(QColor(GOLD))
                self.event_table.setItem(row_idx, col, item)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"

    @staticmethod
    def _short(text: str) -> str:
        return text if len(text) <= 8 else text[:7] + "…"

    def closeEvent(self, event):
        self._stop()
        self.store.conn.close()
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description="RMUC 2026 区域赛数据可视化")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 数据库路径")
    parser.add_argument("--offscreen-check", action="store_true", help="校验并离屏渲染后退出")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.offscreen_check:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("RMUC 2026 数据可视化")
    try:
        window = MainWindow(args.db.resolve())
    except (FileNotFoundError, sqlite3.Error) as exc:
        QMessageBox.critical(None, "数据集不可用", f"找不到或无法打开数据集：\n{exc}")
        return 2
    window.show()
    if args.offscreen_check:
        app.processEvents()
        window.grab().save(str(APP_DIR / "界面预览.png"))
        print(
            f"OK region={window.region_combo.currentText()} "
            f"match={window.match_combo.currentData()} round={window.round_combo.currentText()} "
            f"frames={len(window.frames)} events={sum(window.event_counts.values())}"
        )
        window.close()
        return 0
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
