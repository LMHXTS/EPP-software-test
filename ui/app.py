# -*- coding: utf-8 -*-
"""app.py — Tkinter 主窗口：UI 布局 + 后台 NPU 线程调度"""

import sys, signal, time, threading, warnings
import tkinter as tk
from tkinter import ttk
import cv2, numpy as np

warnings.filterwarnings('ignore', message='.*numpy_to_ptr.*')

_ENC = 'utf-8'
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding=_ENC, errors='replace')
    except Exception:
        pass

from config import PostureConfig
from engine import (init_resources, cap, npu_lock, context,
                    input_dev_ptr, img_size, model_id,
                    input_dataset, output_dataset,
                    output_dev_ptr, output_size, out_host_ptr,
                    parse_npu_output, acl)
from posture import analyze_spine_posture
from renderer import render_ui
from ui.theme import T
from ui.widgets import ArcGauge, StatusPill


class PostureApp:
    """全屏原生姿态检测 UI"""

    def __init__(self):
        # 线程安全共享状态
        self._lock = threading.Lock()
        self._running = True
        self._detection_on = True
        self._ppm_bytes = None
        self._status = "Initializing..."
        self._neck_angle = 0.0
        self._spine_angle = 0.0
        self._fps = 0.0
        self._paused = False

        # 构建主窗口
        self.root = tk.Tk()
        self.root.title("Posture")
        self.root.configure(bg=T.BARK)
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self._toggle_fs())
        self.root.bind('<F11>', lambda e: self._toggle_fs())

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()
        self.pw = int(self.sw * 0.32)
        self.vw = self.sw - self.pw

        self._build_ui()
        self._init_npu()
        self._start_npu_thread()

    # ================================================================
    #  UI 构建
    # ================================================================
    def _build_ui(self):
        # -- 左侧：视频区域（深色包围） --
        video_frame = tk.Frame(self.root, bg=T.BLACK, width=self.vw, height=self.sh)
        video_frame.place(x=0, y=0)
        video_frame.pack_propagate(False)

        # 内边框（2px 树皮色边线）
        inner = tk.Frame(video_frame, bg=T.BARK, width=self.vw - 4, height=self.sh - 4)
        inner.place(x=2, y=2)

        self.video_label = tk.Label(inner, bg=T.BLACK, borderwidth=0)
        self.video_label.place(x=0, y=0, width=self.vw - 4, height=self.sh - 4)

        # 暂停遮罩
        self.pause_label = tk.Label(
            video_frame, text="PAUSED",
            font=(T.FONT, 42, "bold"), fg=T.WHITE, bg=T.BLACK, justify="center"
        )

        # -- 右侧：控制面板 --
        p = tk.Frame(self.root, bg=T.IVORY, width=self.pw, height=self.sh)
        p.place(x=self.vw, y=0)
        p.pack_propagate(False)
        pad = 28

        # ---- 标题区 ----
        head = tk.Frame(p, bg=T.IVORY, height=120)
        head.pack(fill='x', padx=pad, pady=(pad + 16, 0))
        head.pack_propagate(False)

        tk.Label(head, text="Ascend NPU", font=(T.FONT, 12, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY, anchor='w').pack(fill='x')
        tk.Label(head, text="Posture", font=(T.FONT, 36, "bold"),
                 fg=T.CHARCOAL, bg=T.IVORY, anchor='w').pack(fill='x')
        tk.Label(head, text="Real-time spinal monitoring",
                 font=(T.FONT, 12), fg=T.WARMGRY, bg=T.IVORY,
                 anchor='w').pack(fill='x')

        # ---- 状态胶囊 ----
        tk.Frame(p, bg=T.IVORY, height=28).pack()
        pill_frame = tk.Frame(p, bg=T.IVORY)
        pill_frame.pack(fill='x', padx=pad)
        self.pill = StatusPill(pill_frame, width=self.pw - pad * 2, height=58)
        self.pill.pack()
        self.pill.set("Initializing...", T.WARMGRY)

        # ---- 弧形仪表盘行 ----
        tk.Frame(p, bg=T.IVORY, height=22).pack()
        gauges = tk.Frame(p, bg=T.IVORY)
        gauges.pack(fill='x', padx=pad - 4)

        # 颈部仪表盘
        neck_col = tk.Frame(gauges, bg=T.IVORY)
        neck_col.pack(side='left', expand=True, fill='both')
        tk.Label(neck_col, text="NECK", font=(T.FONT, 10, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY).pack()
        self.neck_gauge = ArcGauge(neck_col, width=170, height=140)
        self.neck_gauge.pack()
        tk.Label(neck_col, text="Forward head tilt",
                 font=(T.FONT, 10), fg=T.WARMGRY, bg=T.IVORY).pack()

        # 脊柱仪表盘
        spine_col = tk.Frame(gauges, bg=T.IVORY)
        spine_col.pack(side='left', expand=True, fill='both')
        tk.Label(spine_col, text="SPINE", font=(T.FONT, 10, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY).pack()
        self.spine_gauge = ArcGauge(spine_col, width=170, height=140)
        self.spine_gauge.pack()
        tk.Label(spine_col, text="Slouch / hunch",
                 font=(T.FONT, 10), fg=T.WARMGRY, bg=T.IVORY).pack()

        # ---- 分割线 ----
        tk.Frame(p, bg=T.IVORY, height=26).pack()
        self._div(p, pad)

        # ---- 阈值滑块 ----
        tk.Frame(p, bg=T.IVORY, height=20).pack()
        tk.Label(p, text="THRESHOLDS", font=(T.FONT, 10, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY, anchor='w').pack(fill='x', padx=pad)
        tk.Frame(p, bg=T.IVORY, height=10).pack()

        self._slider_block(p, pad, "Neck forward alert",
                           PostureConfig.TH_NECK, self._on_neck,
                           self, '_neck_th_label')

        tk.Frame(p, bg=T.IVORY, height=16).pack()

        self._slider_block(p, pad, "Spine slouch alert",
                           PostureConfig.TH_SPINE, self._on_spine,
                           self, '_spine_th_label')

        # ---- 分割线 ----
        tk.Frame(p, bg=T.IVORY, height=26).pack()
        self._div(p, pad)

        # ---- 启停按钮 ----
        tk.Frame(p, bg=T.IVORY, height=22).pack()
        self.btn = tk.Button(
            p, text="STOP DETECTION", font=(T.FONT, 15, "bold"),
            fg=T.WHITE, bg=T.ROSE, relief="flat", bd=0,
            activeforeground=T.WHITE, activebackground="#B85A50",
            padx=20, pady=16, cursor="hand2",
            command=self._toggle
        )
        self.btn.pack(fill='x', padx=pad)

        # ---- FPS + 退出（底部） ----
        bottom = tk.Frame(p, bg=T.IVORY)
        bottom.pack(side='bottom', fill='x', padx=pad, pady=(0, 24))

        self.fps_label = tk.Label(bottom, text="— fps", font=(T.FONT, 10),
                                   fg=T.WARMGRY, bg=T.IVORY)
        self.fps_label.pack(side='left')

        tk.Button(bottom, text="Exit", font=(T.FONT, 10),
                  fg=T.WARMGRY, bg=T.IVORY,
                  activeforeground=T.CHARCOAL, activebackground=T.MIST,
                  relief="flat", padx=8, pady=4,
                  cursor="hand2", command=self._exit
                  ).pack(side='right')

    def _div(self, parent, pad):
        """细线分割器"""
        d = tk.Frame(parent, bg=T.SAND, height=1)
        d.pack(fill='x', padx=pad)

    def _slider_block(self, parent, pad, label, value, cmd, store_obj, attr):
        """构建标签 + 滑块的组合行"""
        row = tk.Frame(parent, bg=T.IVORY)
        row.pack(fill='x', padx=pad)

        lbl = tk.Label(row, text=label, font=(T.FONT, 13),
                        fg=T.CHARCOAL, bg=T.IVORY, anchor='w')
        lbl.pack(side='left')

        val_lbl = tk.Label(row, text=f"{value:.1f}°", font=(T.FONT, 14, "bold"),
                            fg=T.SAGE, bg=T.IVORY, anchor='e')
        val_lbl.pack(side='right')
        setattr(store_obj, attr, val_lbl)

        s = ttk.Scale(parent, from_=5, to=60, value=value,
                       length=self.pw - pad * 2, command=cmd)
        s.pack(pady=(6, 0))

        base = attr.replace('_label', '')
        setattr(store_obj, base + '_slider', s)

    # ================================================================
    #  NPU 推理线程
    # ================================================================
    def _init_npu(self):
        try:
            init_resources()
        except Exception as e:
            self.pill.set(f"NPU Error: {e}", T.ROSE)

    def _start_npu_thread(self):
        threading.Thread(target=self._npu_loop, daemon=True).start()

    def _npu_loop(self):
        """后台线程：摄像头读取 → NPU 推理 → 姿势分析 → PPM 编码"""
        acl.rt.set_context(context)
        fc = 0
        print("[NPU] loop started")
        while self._running:
            t0 = time.perf_counter()
            ret, orig = cap.read()
            if not ret:
                time.sleep(0.1); continue

            oh, ow = orig.shape[:2]
            do_infer = self._detection_on

            if do_infer:
                # 预处理：缩放 + 转 RGB + CHW + 归一化
                img = cv2.resize(orig, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)
                inp = np.expand_dims(img, 0).astype(np.float32) / 255.0
                inp = np.ascontiguousarray(inp)

                with npu_lock:
                    p = acl.util.numpy_to_ptr(inp)
                    acl.rt.memcpy(input_dev_ptr, img_size, p, img_size, 1)
                    acl.mdl.execute(model_id, input_dataset, output_dataset)
                    acl.rt.memcpy(out_host_ptr, output_size,
                                  output_dev_ptr, output_size, 2)
                    raw = acl.util.ptr_to_bytes(out_host_ptr, output_size)
                    arr = np.frombuffer(raw, dtype=np.float32) if raw else np.array([])

                try:
                    _, kp = parse_npu_output(arr, conf_threshold=0.15)
                except Exception:
                    kp = None
            else:
                kp = None

            # 计算 FPS
            ms = (time.perf_counter() - t0) * 1000
            fps = 1000.0 / ms if ms > 0 else 0.0

            # 姿势分析 + 视频渲染
            if kp is not None:
                kp = kp.copy()
                sx, sy = ow / 640.0, oh / 640.0
                for pt in kp:
                    pt[0] *= sx; pt[1] *= sy
                a = analyze_spine_posture(kp)
                if a.get("error"):
                    status, na, sa = "No Person", 0.0, 0.0
                else:
                    status, na, sa = a["status"], a["neck_angle"], a["spine_angle"]
                disp = render_ui(orig, a, fps=fps)
            else:
                status, na, sa = "No Person", 0.0, 0.0
                disp = orig
                cv2.putText(disp, f"FPS: {fps:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 缩小到显示分辨率 → PPM 编码
            small = cv2.resize(disp, (T.DISP_W, T.DISP_H))
            ok, ppm = cv2.imencode('.ppm', small)
            if not ok: continue

            # 每 100 帧或前 10 帧无人时打印调试
            fc += 1
            if fc % 100 == 0 or (fc < 10 and kp is None):
                print(f"[NPU] frame={fc} status={status} fps={fps:.1f}")

            # 更新线程安全共享状态
            with self._lock:
                self._ppm_bytes = ppm.tobytes()
                self._status = status
                self._neck_angle = na
                self._spine_angle = sa
                self._fps = fps
                self._paused = not do_infer

    # ================================================================
    #  显示刷新（主线程，33ms / ~30fps）
    # ================================================================
    def _refresh_display(self):
        with self._lock:
            ppm = self._ppm_bytes
            status = self._status
            na = self._neck_angle
            sa = self._spine_angle
            fps = self._fps
            paused = self._paused

        if ppm:
            try:
                img = tk.PhotoImage(data=ppm)
                self.video_label.config(image=img)
                self.video_label.image = img
            except Exception:
                pass

        if paused:
            self.pause_label.place(x=4, y=4, width=self.vw - 4, height=self.sh - 4)
            self.pause_label.lift()
        else:
            self.pause_label.place_forget()

        # 更新状态胶囊
        if "Warning" in status:
            if "Tilt" in status or "Slouching" in status:
                pill_c, pill_t = T.ROSE, status.upper()
            else:
                pill_c, pill_t = T.CORAL, status.upper()
        elif status == "Standard Posture":
            pill_c, pill_t = T.SAGE, "GOOD POSTURE"
        else:
            pill_c, pill_t = T.WARMGRY, status.upper()
        self.pill.set(pill_t, pill_c)

        # 更新弧形仪表盘
        self.neck_gauge.set(na, PostureConfig.TH_NECK)
        self.spine_gauge.set(sa, PostureConfig.TH_SPINE)

        self.fps_label.config(text=f"{fps:.1f} fps")

        self.root.after(33, self._refresh_display)

    # ================================================================
    #  交互回调
    # ================================================================
    def _toggle(self):
        self._detection_on = not self._detection_on
        if self._detection_on:
            self.btn.config(text="STOP DETECTION", bg=T.ROSE,
                            activebackground="#B85A50")
        else:
            self.btn.config(text="START DETECTION", bg=T.SAGE,
                            activebackground="#6F9A7D")

    def _on_neck(self, val):
        v = float(val)
        PostureConfig.TH_NECK = v
        self._neck_th_label.config(text=f"{v:.1f}°")

    def _on_spine(self, val):
        v = float(val)
        PostureConfig.TH_SPINE = v
        self._spine_th_label.config(text=f"{v:.1f}°")

    def _toggle_fs(self):
        self.root.attributes('-fullscreen',
                             not self.root.attributes('-fullscreen'))

    def _exit(self):
        self._running = False
        self.root.destroy()

    def run(self):
        print(">>> Posture — Soft Clinical Edition <<<")
        signal.signal(signal.SIGINT, lambda *a: self._exit())
        signal.signal(signal.SIGTERM, lambda *a: self._exit())
        self.root.protocol("WM_DELETE_WINDOW", self._exit)
        self.root.after(500, self._refresh_display)
        self.root.mainloop()
