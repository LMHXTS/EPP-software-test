# -*- coding: utf-8 -*-
"""widgets.py — 自定义 Tkinter 组件：弧形仪表盘 + 状态胶囊"""

import tkinter as tk
from ui.theme import T


class ArcGauge(tk.Canvas):
    """半圆弧仪表盘，显示 0°–60° 的角度值。
    颜色从鼠尾草绿（正常）→ 珊瑚色（警告）→ 玫瑰色（严重）渐变。"""

    def __init__(self, parent, width=180, height=130, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=T.IVORY, highlightthickness=0, **kw)
        self.w, self.h = width, height
        self.value = 0.0
        self.threshold = 25.0
        self._draw()

    def _draw(self):
        self.delete("all")
        cx, cy = self.w / 2, self.h - 10
        r = min(self.w, self.h) - 18

        # 背景圆弧（0° 到 180°）
        self.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=180, extent=180, style="arc",
                         outline=T.SAND, width=10)

        # 数值弧（根据当前角度填满）
        angle = min(self.value / 60.0 * 180, 180)
        if self.value > self.threshold:
            c = T.ROSE if self.value > self.threshold * 1.6 else T.CORAL
        else:
            c = T.SAGE
        self.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=180, extent=-angle, style="arc",
                         outline=c, width=10)

        # 中心数值文字
        self.create_text(cx, cy - 16, text=f"{self.value:.1f}",
                          font=(T.FONT, 26, "bold"), fill=T.CHARCOAL)
        self.create_text(cx, cy + 14, text="deg",
                          font=(T.FONT, 10), fill=T.WARMGRY)

    def set(self, value, threshold):
        old_v, old_t = self.value, self.threshold
        self.value, self.threshold = value, threshold
        if abs(old_v - value) > 0.05 or abs(old_t - threshold) > 0.05:
            self._draw()


class StatusPill(tk.Canvas):
    """圆角胶囊形状态徽章，根据姿势状态切换颜色。"""

    def __init__(self, parent, width=280, height=52, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=T.IVORY, highlightthickness=0, **kw)
        self.w, self.h = width, height
        self.text = ""
        self.color = T.SAND
        self._draw()

    def _draw(self):
        self.delete("all")
        r = self.h / 2
        # 胶囊背景
        self._rounded_rect(2, 2, self.w - 2, self.h - 2, r, fill=self.color)
        # 文字
        self.create_text(self.w / 2, self.h / 2, text=self.text,
                          font=(T.FONT, 16, "bold"), fill=T.WHITE)

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        """用四角圆形 + 中间矩形模拟圆角长方形"""
        f = kw.get("fill", "")
        self.create_oval(x1, y1, x1 + 2 * r, y1 + 2 * r, fill=f, outline="")
        self.create_oval(x2 - 2 * r, y1, x2, y1 + 2 * r, fill=f, outline="")
        self.create_oval(x1, y2 - 2 * r, x1 + 2 * r, y2, fill=f, outline="")
        self.create_oval(x2 - 2 * r, y2 - 2 * r, x2, y2, fill=f, outline="")
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=f, outline="")
        self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=f, outline="")

    def set(self, text, color):
        if self.text != text or self.color != color:
            self.text, self.color = text, color
            self._draw()
