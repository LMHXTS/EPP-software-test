# -*- coding: utf-8 -*-
"""records.py — 不良姿势事件记录（持续时间格式）"""

import json
import time
import os


class PostureRecords:
    """姿势事件记录器。仅记录触发告警的不良姿势，保存起止时间和持续时间。"""

    MAX_RECORDS = 200

    def __init__(self, filepath="/root/Desktop/elec_project/records.json"):
        self.filepath = filepath
        self._records = []
        # 当前进行中的告警
        self._active_event = None       # {start_time, status, angles[]}
        self._load()

    # ---- 事件生命周期 ----
    def start_event(self, status, neck_angle, spine_angle):
        """告警触发时调用（不良姿势已持续超过阈值时间）"""
        if self._active_event is not None:
            return  # 已有进行中的事件
        self._active_event = {
            "date": time.strftime("%Y-%m-%d"),
            "start": time.strftime("%H:%M:%S"),
            "start_ts": time.time(),
            "status": status,
            "neck_sum": float(neck_angle),
            "spine_sum": float(spine_angle),
            "samples": 1,
        }

    def update_event(self, neck_angle, spine_angle):
        """告警持续中调用，累积角度样本"""
        if self._active_event is None:
            return
        self._active_event["neck_sum"] += float(neck_angle)
        self._active_event["spine_sum"] += float(spine_angle)
        self._active_event["samples"] += 1

    def end_event(self):
        """告警结束时调用，保存事件到记录列表"""
        if self._active_event is None:
            return
        e = self._active_event
        dur = int(time.time() - e["start_ts"])
        # 只记录持续超过 2 秒的事件（过滤闪报）
        if dur < 2:
            self._active_event = None
            return

        record = {
            "date": e["date"],
            "start": e["start"],
            "end": time.strftime("%H:%M:%S"),
            "duration_sec": dur,
            "status": e["status"],
            "neck_angle": round(e["neck_sum"] / e["samples"], 1),
            "spine_angle": round(e["spine_sum"] / e["samples"], 1),
        }
        self._records.append(record)
        if len(self._records) > self.MAX_RECORDS:
            self._records = self._records[-self.MAX_RECORDS:]
        self._active_event = None
        self._save()

    # ---- 查询 ----
    def get_all(self):
        """返回全部记录（最近在前）"""
        return list(reversed(self._records))

    def clear(self):
        self._records = []
        self._active_event = None
        self._save()

    # ---- 持久化 ----
    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._records, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._records = []
