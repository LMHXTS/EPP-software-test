# 在板子上运行此脚本，自动替换 ui/app.py 中的控制面板代码
# 用法: python3 scripts/panel_update.py

import re

with open("ui/app.py", "r", encoding="utf-8") as f:
    content = f.read()

old_start = "# -- 右侧：控制面板 --"
old_end = "tk.Button(bottom, text=\"退出\""

# 找到旧面板的起止行号
lines = content.split("\n")
start_line = None
end_line = None
for i, line in enumerate(lines):
    if old_start in line and start_line is None:
        start_line = i
    if start_line is not None and old_end in line:
        end_line = i + 15  # 往后多取几行
        break

print(f"Old panel: lines {start_line+1} - {end_line+1}")

new_panel = '''        # -- 右侧：控制面板 --
        p = tk.Frame(self._detect_frame, bg=T.IVORY, width=self.pw, height=self.sh)
        p.place(x=self.vw, y=0)
        p.pack_propagate(False)
        pad = 28

        # ---- 标题区 ----
        head = tk.Frame(p, bg=T.IVORY, height=135)
        head.pack(fill='x', padx=pad, pady=(50, 0))
        head.pack_propagate(False)

        tk.Label(head, text="Ascend NPU", font=(T.FONT, 16, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY, anchor='w').pack(fill='x')
        tk.Label(head, text="姿态检测", font=(T.FONT, 44, "bold"),
                 fg=T.CHARCOAL, bg=T.IVORY, anchor='w').pack(fill='x')
        tk.Label(head, text="实时脊柱监测",
                 font=(T.FONT, 18), fg=T.WARMGRY, bg=T.IVORY,
                 anchor='w').pack(fill='x')

        # ---- 状态胶囊 ----
        tk.Frame(p, bg=T.IVORY, height=40).pack()
        pill_frame = tk.Frame(p, bg=T.IVORY)
        pill_frame.pack(fill='x', padx=pad)
        self.pill = StatusPill(pill_frame, width=self.pw - pad * 2, height=72)
        self.pill.pack()
        self.pill.set("Initializing...", T.WARMGRY)

        # ---- 弧形仪表盘行 ----
        tk.Frame(p, bg=T.IVORY, height=36).pack()
        gauges = tk.Frame(p, bg=T.IVORY)
        gauges.pack(fill='x', padx=pad - 4)

        neck_col = tk.Frame(gauges, bg=T.IVORY)
        neck_col.pack(side='left', expand=True, fill='both')
        tk.Label(neck_col, text="颈部", font=(T.FONT, 16, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY).pack()
        self.neck_gauge = ArcGauge(neck_col, width=190, height=170)
        self.neck_gauge.pack()
        tk.Label(neck_col, text="头部前倾角度",
                 font=(T.FONT, 15), fg=T.WARMGRY, bg=T.IVORY).pack()

        spine_col = tk.Frame(gauges, bg=T.IVORY)
        spine_col.pack(side='left', expand=True, fill='both')
        tk.Label(spine_col, text="脊柱", font=(T.FONT, 16, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY).pack()
        self.spine_gauge = ArcGauge(spine_col, width=190, height=170)
        self.spine_gauge.pack()
        tk.Label(spine_col, text="驼背 / 圆肩",
                 font=(T.FONT, 15), fg=T.WARMGRY, bg=T.IVORY).pack()

        # ---- 分割线 ----
        tk.Frame(p, bg=T.IVORY, height=40).pack()
        self._div(p, pad)

        # ---- 阈值滑块 ----
        tk.Frame(p, bg=T.IVORY, height=25).pack()
        tk.Label(p, text="告警阈值", font=(T.FONT, 16, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY, anchor='w').pack(fill='x', padx=pad)
        tk.Frame(p, bg=T.IVORY, height=16).pack()

        self._slider_block(p, pad, "颈部前倾告警",
                           PostureConfig.TH_NECK, self._on_neck,
                           self, '_neck_th_label')

        tk.Frame(p, bg=T.IVORY, height=32).pack()

        self._slider_block(p, pad, "脊柱弯曲告警",
                           PostureConfig.TH_SPINE, self._on_spine,
                           self, '_spine_th_label')

        # ---- 分割线 ----
        tk.Frame(p, bg=T.IVORY, height=40).pack()
        self._div(p, pad)

        # ---- 启停按钮 ----
        tk.Frame(p, bg=T.IVORY, height=28).pack()
        self.btn = tk.Button(
            p, text="停止检测", font=(T.FONT, 22, "bold"),
            fg=T.WHITE, bg=T.ROSE, relief="flat", bd=0,
            activeforeground=T.WHITE, activebackground="#B85A50",
            padx=24, pady=24, cursor="hand2",
            command=self._toggle
        )
        self.btn.pack(fill='x', padx=pad)

        # ---- 页面切换按钮 ----
        tk.Frame(p, bg=T.IVORY, height=22).pack()
        self.page_btn = tk.Button(
            p, text="检测记录",
            font=(T.FONT, 16), fg=T.CHARCOAL, bg=T.MIST,
            relief="flat", padx=14, pady=14, cursor="hand2",
            command=self._toggle_page
        )
        self.page_btn.pack(fill='x', padx=pad)

        # ---- 蜂鸣器 + 马达开关 ----
        tk.Frame(p, bg=T.IVORY, height=22).pack()
        sw_frame = tk.Frame(p, bg=T.IVORY)
        sw_frame.pack(fill='x', padx=pad)

        self.buzzer_btn = tk.Button(
            sw_frame, text="🔔 蜂鸣器: 开", font=(T.FONT, 15, "bold"),
            fg=T.WHITE, bg=T.SAGE, relief="flat", bd=0,
            padx=16, pady=14, cursor="hand2",
            command=self._toggle_buzzer
        )
        self.buzzer_btn.pack(side='left', fill='x', expand=True, padx=(0, 4))

        self.motor_btn = tk.Button(
            sw_frame, text="📳 马达: 关", font=(T.FONT, 15, "bold"),
            fg=T.CHARCOAL, bg=T.MIST, relief="flat", bd=0,
            padx=16, pady=14, cursor="hand2",
            command=self._toggle_motor
        )
        self.motor_btn.pack(side='left', fill='x', expand=True, padx=(4, 0))

        # ---- FPS + 退出（底部） ----
        bottom = tk.Frame(p, bg=T.IVORY)
        bottom.pack(side='bottom', fill='x', padx=pad, pady=(0, 30))

        self.fps_label = tk.Label(bottom, text="— 帧/秒", font=(T.FONT, 15),
                                   fg=T.WARMGRY, bg=T.IVORY)
        self.fps_label.pack(side='left')

        tk.Button(bottom, text="退出", font=(T.FONT, 14),
                  fg=T.WARMGRY, bg=T.IVORY,
                  activeforeground=T.CHARCOAL, activebackground=T.MIST,
                  relief="flat", padx=10, pady=6,
                  cursor="hand2", command=self._exit
                  ).pack(side='right')'''

# 替换
new_lines = lines[:start_line] + new_panel.split("\n") + lines[end_line:]
with open("ui/app.py", "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))

print("Panel updated! Check ui/app.py")
