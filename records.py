# -*- coding: utf-8 -*-
"""records.py — 姿态检测记录存储与查询"""

import json
import time
import os


class PostureRecords:
    """姿态事件记录器，存 JSON 文件"""

    MAX_RECORDS = 500

    def __init__(self, filepath="/tmp/posture_records.json"):
        self.filepath = filepath
        self._records = []
        self._last_status = ""       # 防抖：状态不变时不重复记录
        self._last_record_time = 0   # 同状态最小间隔（秒）
        self._load()

    # ---- 写入 ----
    def add(self, status, neck_angle, spine_angle):
        """添加一条检测记录（自动去重 + 防抖）"""
        # 只记录有意义的姿势变化
        if status == self._last_status:
            return  # 同状态不重复
        # 同状态至少间隔 5 秒
        now = time.time()
        if now - self._last_record_time < 5:
            # 但如果状态变化了，允许更新
            return

        record = {
            "time": time.strftime("%H:%M:%S"),
            "date": time.strftime("%Y-%m-%d"),
            "status": status,
            "neck_angle": round(float(neck_angle), 1),
            "spine_angle": round(float(spine_angle), 1),
        }
        self._records.append(record)
        self._last_status = status
        self._last_record_time = now

        # 超过上限时裁剪
        if len(self._records) > self.MAX_RECORDS:
            self._records = self._records[-self.MAX_RECORDS:]

        self._save()

    # ---- 查询 ----
    def get_all(self):
        """返回全部记录（最近在前）"""
        return list(reversed(self._records))

    def get_warnings(self):
        """仅返回告警记录"""
        return [r for r in reversed(self._records) if "Warning" in r["status"]]

    def get_today(self):
        """今日记录"""
        today = time.strftime("%Y-%m-%d")
        return [r for r in reversed(self._records) if r["date"] == today]

    def clear(self):
        """清空记录"""
        self._records = []
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
