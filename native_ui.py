# -*- coding: utf-8 -*-
"""
native_ui.py — 基于 Tkinter 的原生 UI，用于 HDMI 触摸屏本地显示
架构: 后台 NPU 线程(full speed) + Tkinter 主线程(30fps 刷新显示)
无需 Flask / 浏览器 / 网络 / PIL
"""

import sys
import signal
import time
import threading
import warnings
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

# 忽略 acl 的废弃警告
warnings.filterwarnings('ignore', message='.*numpy_to_ptr.*')

# ---- 编码兜底 ----
_ENC = 'utf-8'
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding=_ENC, errors='replace')
    except Exception:
        pass

import detect_main as dm


class PostureApp:
    """全屏原生姿态检测 UI — 暗色主题 + 触摸友好"""

    # 配色
    BG = "#1a1a2e"
    PANEL = "#16213e"
    ACCENT = "#0f3460"
    GREEN = "#4ecca3"
    RED = "#e23e57"
    ORANGE = "#f0a500"
    TEXT = "#eeeeee"
    SUBTEXT = "#a0a0b0"

    # 显示分辨率（低分辨率→ PPM 编码快→ 帧率高）
    DISP_W = 800
    DISP_H = 450

    def __init__(self):
        # ---- 线程安全共享状态 ----
        self._lock = threading.Lock()
        self._running = True
        self._detection_on = True

        # NPU 线程写入，UI 线程读取
        self._ppm_bytes = None       # 预编码的 PPM 帧数据
        self._status = "Initializing..."
        self._neck_angle = 0.0
        self._spine_angle = 0.0
        self._fps = 0.0
        self._paused = False         # 是否显示暂停遮罩

        # ---- 构建 UI ----
        self.root = tk.Tk()
        self.root.title("Posture Detection")
        self.root.configure(bg=self.BG)
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self._toggle_fs())
        self.root.bind('<F11>', lambda e: self._toggle_fs())

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()
        self.pw = int(self.sw * 0.28)  # 右侧面板宽度
        self.vw = self.sw - self.pw    # 视频区域宽度
        self.vh = self.sh

        self._build_ui()

        # ---- 初始化 NPU 并启动后台线程 ----
        self._init_npu()
        self._start_npu_thread()

    # ================================================================
    #  UI 构建
    # ================================================================
    def _build_ui(self):
        # 左侧 — 视频显示
        self.video_label = tk.Label(self.root, bg="#000", borderwidth=0)
        self.video_label.place(x=0, y=0, width=self.vw, height=self.vh)

        # 暂停遮罩
        self.pause_label = tk.Label(
            self.root, text="⏸  PAUSED",
            font=("Segoe UI", 40, "bold"),
            fg="white", bg="#111111", justify="center"
        )

        # 右侧 — 控制面板
        p = tk.Frame(self.root, bg=self.PANEL, width=self.pw, height=self.sh)
        p.place(x=self.vw, y=0)
        p.pack_propagate(False)

        s = {"fg": self.TEXT, "bg": self.PANEL}
        sg = {"fg": self.SUBTEXT, "bg": self.PANEL}

        tk.Label(p, text="Ascend NPU", font=("Segoe UI", 20, "bold"),
                 fg=self.GREEN, bg=self.PANEL).pack(pady=(30, 3))
        tk.Label(p, text="Real-Time Posture Detection",
                 font=("Segoe UI", 10), **sg).pack(pady=(0, 20))

        ttk.Separator(p, orient='horizontal').pack(fill='x', padx=16)

        # 状态
        tk.Label(p, text="STATUS", font=("Segoe UI", 9, "bold"),
                 **sg).pack(pady=(18, 3))
        self.st_label = tk.Label(p, text="Initializing...",
                                 font=("Segoe UI", 18, "bold"),
                                 fg=self.TEXT, bg=self.PANEL,
                                 wraplength=self.pw - 20, justify="center")
        self.st_label.pack(pady=(0, 12))

        # 角度
        af = tk.Frame(p, bg=self.PANEL)
        af.pack(fill='x', padx=16)
        tk.Label(af, text="Neck Angle", **sg).pack(anchor='w')
        self.neck_v = tk.Label(af, text="--°", font=("Segoe UI", 28, "bold"),
                               fg=self.GREEN, bg=self.PANEL)
        self.neck_v.pack(anchor='w', pady=(0, 8))

        tk.Label(af, text="Spine Angle", **sg).pack(anchor='w')
        self.spine_v = tk.Label(af, text="--°", font=("Segoe UI", 28, "bold"),
                                fg=self.GREEN, bg=self.PANEL)
        self.spine_v.pack(anchor='w', pady=(0, 16))

        ttk.Separator(p, orient='horizontal').pack(fill='x', padx=16)

        # 阈值滑块
        tk.Label(p, text="THRESHOLDS", font=("Segoe UI", 9, "bold"),
                 **sg).pack(pady=(14, 6))

        tk.Label(p, text="Neck Forward", **s).pack(anchor='w', padx=20)
        self.nth_v = tk.Label(p, text=f"{dm.PostureConfig.TH_NECK:.1f}°",
                              font=("Segoe UI", 12, "bold"),
                              fg=self.GREEN, bg=self.PANEL)
        self.nth_v.pack(anchor='e', padx=20)
        self.neck_slider = ttk.Scale(p, from_=5, to=60,
                                     value=dm.PostureConfig.TH_NECK,
                                     length=self.pw - 44,
                                     command=self._on_neck)
        self.neck_slider.pack(pady=(0, 10))

        tk.Label(p, text="Spine Slouch", **s).pack(anchor='w', padx=20)
        self.sth_v = tk.Label(p, text=f"{dm.PostureConfig.TH_SPINE:.1f}°",
                              font=("Segoe UI", 12, "bold"),
                              fg=self.GREEN, bg=self.PANEL)
        self.sth_v.pack(anchor='e', padx=20)
        self.spine_slider = ttk.Scale(p, from_=5, to=60,
                                      value=dm.PostureConfig.TH_SPINE,
                                      length=self.pw - 44,
                                      command=self._on_spine)
        self.spine_slider.pack()

        ttk.Separator(p, orient='horizontal').pack(fill='x', padx=16, pady=16)

        # 启停按钮
        self.btn = tk.Button(
            p, text="⏸  STOP", font=("Segoe UI", 16, "bold"),
            fg="white", bg=self.RED, relief="flat", bd=0,
            padx=16, pady=12, cursor="hand2",
            command=self._toggle
        )
        self.btn.pack(fill='x', padx=16, pady=4)

        # FPS
        self.fps_label = tk.Label(p, text="FPS: --", **sg)
        self.fps_label.pack(pady=(16, 4))

        # 退出
        tk.Button(p, text="Exit", font=("Segoe UI", 9),
                  fg=self.SUBTEXT, bg=self.ACCENT,
                  relief="flat", padx=8, pady=4,
                  cursor="hand2", command=self._exit
                  ).pack(side='bottom', pady=16)

    # ================================================================
    #  NPU 初始化 + 后台推理线程
    # ================================================================
    def _init_npu(self):
        try:
            dm.init_resources()
        except Exception as e:
            self.st_label.config(text=f"NPU Error: {e}", fg=self.RED)

    def _start_npu_thread(self):
        t = threading.Thread(target=self._npu_loop, daemon=True)
        t.start()

    def _npu_loop(self):
        """后台线程: 摄像头读取 → NPU 推理 → 姿势分析 → PPM 编码"""
        dm.acl.rt.set_context(dm.context)
        frame_count = 0

        while self._running:
            t0 = time.perf_counter()
            ret, orig = dm.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            oh, ow = orig.shape[:2]
            do_infer = self._detection_on

            if do_infer:
                # 预处理 640×640
                img = cv2.resize(orig, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)
                inp = np.expand_dims(img, 0).astype(np.float32) / 255.0
                inp = np.ascontiguousarray(inp)

                with dm.npu_lock:
                    p = dm.acl.util.numpy_to_ptr(inp)
                    dm.acl.rt.memcpy(dm.input_dev_ptr, dm.img_size, p, dm.img_size, 1)
                    dm.acl.mdl.execute(dm.model_id, dm.input_dataset, dm.output_dataset)
                    dm.acl.rt.memcpy(dm.out_host_ptr, dm.output_size,
                                     dm.output_dev_ptr, dm.output_size, 2)
                    raw = dm.acl.util.ptr_to_bytes(dm.out_host_ptr, dm.output_size)
                    arr = np.frombuffer(raw, dtype=np.float32) if raw else np.array([])

                try:
                    _, kp = dm.parse_npu_output(arr, conf_threshold=0.15)
                except Exception:
                    kp = None
            else:
                kp = None

            # FPS
            ms = (time.perf_counter() - t0) * 1000
            fps = 1000.0 / ms if ms > 0 else 0.0

            # 分析 + 渲染到原始尺寸帧
            if kp is not None:
                kp = kp.copy()
                sx, sy = ow / 640.0, oh / 640.0
                for pt in kp:
                    pt[0] *= sx
                    pt[1] *= sy
                a = dm.analyze_spine_posture(kp)
                if a.get("error"):
                    status, na, sa = "No Person", 0.0, 0.0
                else:
                    status, na, sa = a["status"], a["neck_angle"], a["spine_angle"]
                disp = dm.render_ui(orig, a, fps=fps)
            else:
                status, na, sa = "No Person", 0.0, 0.0
                disp = orig
                cv2.putText(disp, f"NPU FPS: {fps:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 缩放到显示分辨率 → PPM 编码（imencode 内部处理 BGR→RGB）
            small = cv2.resize(disp, (self.DISP_W, self.DISP_H))
            ok, ppm = cv2.imencode('.ppm', small)
            if not ok:
                continue

            # 调试: 每100帧打印一次
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[NPU] frame {frame_count}, status={status}, fps={fps:.1f}")

            # 更新共享状态
            with self._lock:
                self._ppm_bytes = ppm.tobytes()
                self._status = status
                self._neck_angle = na
                self._spine_angle = sa
                self._fps = fps
                self._paused = not do_infer

    # ================================================================
    #  Tkinter 显示刷新 (30fps)
    # ================================================================
    def _refresh_display(self):
        """主线程: 31ms 间隔刷新画面 + 状态面板"""
        with self._lock:
            ppm = self._ppm_bytes
            status = self._status
            na = self._neck_angle
            sa = self._spine_angle
            fps = self._fps
            paused = self._paused

        # 更新视频帧
        if ppm:
            try:
                img = tk.PhotoImage(data=ppm)
                self.video_label.config(image=img)
                self.video_label.image = img
            except Exception:
                pass

        # 暂停遮罩
        if paused:
            self.pause_label.place(x=0, y=0, width=self.vw, height=self.vh)
            self.pause_label.lift()
        else:
            self.pause_label.place_forget()

        # 状态面板
        if "Warning" in status:
            c = self.RED if ("Tilt" in status or "Slouching" in status) else self.ORANGE
        elif status == "Standard Posture":
            c = self.GREEN
        else:
            c = self.SUBTEXT
        self.st_label.config(text=status, fg=c)

        nc = self.RED if na > dm.PostureConfig.TH_NECK else self.GREEN
        self.neck_v.config(text=f"{na:.1f}°", fg=nc)
        sc = self.RED if sa > dm.PostureConfig.TH_SPINE else self.GREEN
        self.spine_v.config(text=f"{sa:.1f}°", fg=sc)
        self.fps_label.config(text=f"FPS: {fps:.1f}")

        # 继续调度
        self.root.after(31, self._refresh_display)

    # ================================================================
    #  交互回调
    # ================================================================
    def _toggle(self):
        self._detection_on = not self._detection_on
        if self._detection_on:
            self.btn.config(text="⏸  STOP", bg=self.RED)
        else:
            self.btn.config(text="▶  START", bg=self.GREEN)

    def _on_neck(self, val):
        v = float(val)
        dm.PostureConfig.TH_NECK = v
        self.nth_v.config(text=f"{v:.1f}°")

    def _on_spine(self, val):
        v = float(val)
        dm.PostureConfig.TH_SPINE = v
        self.sth_v.config(text=f"{v:.1f}°")

    def _toggle_fs(self):
        self.root.attributes('-fullscreen',
                             not self.root.attributes('-fullscreen'))

    def _exit(self):
        """安全退出 — 只关窗口，不主动释放 NPU（避免 ACL 内核调用卡死）"""
        self._running = False
        self.root.destroy()

    # ================================================================
    #  启动
    # ================================================================
    def run(self):
        print(">>> Native UI Starting (threaded mode) <<<")

        # Ctrl+C / kill 信号处理
        signal.signal(signal.SIGINT, lambda *a: self._exit())
        signal.signal(signal.SIGTERM, lambda *a: self._exit())

        self.root.protocol("WM_DELETE_WINDOW", self._exit)
        self.root.after(500, self._refresh_display)
        self.root.mainloop()


if __name__ == '__main__':
    PostureApp().run()
