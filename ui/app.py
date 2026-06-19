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
import engine
from posture import analyze_spine_posture
from renderer import render_ui
from ui.theme import T
from ui.widgets import ArcGauge, StatusPill
from m0 import M0Controller
from records import PostureRecords


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

        # M0 核心板 + 检测记录
        self.m0 = M0Controller()
        self.m0.connect()
        self.records = PostureRecords()
        # 告警延迟：不良姿势持续 X 秒后才触发
        self._bad_posture_start = 0.0
        self._bad_posture_delay = 3  # 秒

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

        self._page = "detect"  # 当前页面: detect / records

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

        # ---- 页面切换按钮 ----
        tk.Frame(p, bg=T.IVORY, height=10).pack()
        self.page_btn = tk.Button(
            p, text="DETECTION RECORDS",
            font=(T.FONT, 11), fg=T.CHARCOAL, bg=T.MIST,
            relief="flat", padx=14, pady=8, cursor="hand2",
            command=self._toggle_page
        )
        self.page_btn.pack(fill='x', padx=pad)

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

    def _build_records_page(self):
        """构建检测记录页面（卡片式布局 + 逐条删除）"""
        p = tk.Frame(self.root, bg=T.IVORY, width=self.sw, height=self.sh)
        p.place(x=0, y=0)
        p.pack_propagate(False)
        pad = 50

        # ---- 顶部 ----
        top = tk.Frame(p, bg=T.IVORY, height=100)
        top.pack(fill='x', padx=pad, pady=(pad, 0))
        top.pack_propagate(False)

        tk.Label(top, text="Detection Records", font=(T.FONT, 30, "bold"),
                 fg=T.CHARCOAL, bg=T.IVORY, anchor='w').pack(fill='x')
        sub = tk.Frame(top, bg=T.IVORY)
        sub.pack(fill='x')
        tk.Label(sub, text="不良姿势事件历史",
                 font=(T.FONT, 12), fg=T.WARMGRY, bg=T.IVORY,
                 anchor='w').pack(side='left')
        tk.Button(sub, text="CLEAR ALL", font=(T.FONT, 10),
                  fg=T.ROSE, bg=T.IVORY,
                  activeforeground=T.WHITE, activebackground=T.ROSE,
                  relief="flat", padx=8, pady=2, cursor="hand2",
                  command=self._clear_all_records
                  ).pack(side='right')

        self._records_empty = tk.Label(top, text="暂无记录",
                                        font=(T.FONT, 14), fg=T.WARMGRY, bg=T.IVORY)
        self._records_empty.pack_forget()

        # ---- 可滚动卡片区域 ----
        canvas_h = self.sh - 220  # 预留顶部标题(100) + 底部按钮栏(60) + margin
        canvas = tk.Canvas(p, bg=T.IVORY, highlightthickness=0,
                            width=self.sw - pad * 2, height=canvas_h)
        scroll = ttk.Scrollbar(p, orient="vertical", command=canvas.yview)
        self._rec_scroll_frame = tk.Frame(canvas, bg=T.IVORY)

        self._rec_scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._rec_scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        # 鼠标滚轮支持
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        canvas.pack(side='left', fill='both', expand=True, padx=(pad, 0))
        scroll.pack(side='right', fill='y', padx=(0, pad))

        self._rec_canvas = canvas

        # ---- 底部按钮栏 ----
        bar = tk.Frame(p, bg=T.IVORY, height=60)
        bar.pack(side='bottom', fill='x', padx=pad, pady=(10, 30))
        bar.pack_propagate(False)

        tk.Button(bar, text="← BACK TO DETECTION",
                  font=(T.FONT, 14, "bold"), fg=T.WHITE, bg=T.SAGE,
                  relief="flat", padx=24, pady=14, cursor="hand2",
                  command=self._toggle_page
                  ).pack(side='left')

        tk.Button(bar, text="CLEAR ALL",
                  font=(T.FONT, 12), fg=T.ROSE, bg=T.IVORY,
                  activeforeground=T.WHITE, activebackground=T.ROSE,
                  relief="flat", padx=16, pady=10, cursor="hand2",
                  command=self._clear_all_records
                  ).pack(side='right')

        # 总数
        total = len(self.records.get_all())
        self._rec_total = tk.Label(bar, text=f"共 {total} 条记录",
                                    font=(T.FONT, 11), fg=T.WARMGRY, bg=T.IVORY)
        self._rec_total.pack(side='right', padx=20)

        self._records_page = p
        self._refresh_records()

    def _refresh_records(self):
        """刷新记录卡片列表"""
        for w in self._rec_scroll_frame.winfo_children():
            w.destroy()

        records = self.records.get_all()
        if not records:
            self._records_empty.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self._records_empty.place_forget()

        pad = 8
        card_w = self.sw - 130  # 左 margin 50 + 右 scrollbar ~30 + padding

        for i, r in enumerate(records):
            card = tk.Frame(self._rec_scroll_frame, bg=T.CREAM,
                             width=card_w, height=52)
            card.pack(fill='x', padx=0, pady=4)
            card.pack_propagate(False)

            # 左侧竖线（颜色标识）
            status = r["status"]
            if "Slouching" in status or "Tilt" in status:
                bar_c = T.ROSE
            elif "Hunchback" in status:
                bar_c = T.CORAL
            else:
                bar_c = T.CORAL
            tk.Frame(card, bg=bar_c, width=4).place(x=0, y=0, height=52)

            # 时间
            tk.Label(card, text=r["time"], font=(T.FONT, 10),
                     fg=T.WARMGRY, bg=T.CREAM, anchor='w'
                     ).place(x=14, y=4, width=80)

            # 状态标签
            short_status = status.replace("Warning: ", "").replace("!", "")
            tk.Label(card, text=short_status, font=(T.FONT, 12, "bold"),
                     fg=T.CHARCOAL, bg=T.CREAM, anchor='w'
                     ).place(x=14, y=24, width=260)

            # 角度数值
            angles = f"N:{r['neck_angle']:.1f}°  S:{r['spine_angle']:.1f}°"
            tk.Label(card, text=angles, font=(T.FONT, 11),
                     fg=T.WARMGRY, bg=T.CREAM, anchor='e'
                     ).place(x=card_w - 200, y=16, width=110)

            # 删除按钮（右对齐，始终可见）
            btn = tk.Button(card, text="×", font=(T.FONT, 15, "bold"),
                            fg=T.WARMGRY, bg=T.CREAM,
                            activeforeground=T.ROSE, activebackground=T.MIST,
                            relief="flat", bd=0, padx=8, pady=4,
                            cursor="hand2",
                            command=lambda idx=i: self._delete_record(idx))
            btn.place(x=card_w - 50, y=10, width=36, height=32)

    def _delete_record(self, idx):
        """删除第 idx 条记录"""
        records = self.records.get_all()
        if 0 <= idx < len(records):
            del self.records._records[len(self.records._records) - 1 - idx]
            self.records._save()
            self._refresh_records()

    def _clear_all_records(self):
        """清空全部记录"""
        self.records.clear()
        self._refresh_records()

    def _toggle_page(self):
        """切换检测页 / 记录页"""
        if self._page == "detect":
            self._page = "records"
            for w in self.root.place_slaves():
                w.place_forget()
            try:
                self.root.unbind_all("<MouseWheel>")
            except Exception:
                pass
            self._build_records_page()
            self.root.update()
        else:
            self._page = "detect"
            self.root.unbind_all("<MouseWheel>")
            self._records_page.destroy()
            delattr(self, '_records_page')
            self._build_ui()
            self.root.update()

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
            engine.init_resources()
        except Exception as e:
            self.pill.set(f"NPU Error: {e}", T.ROSE)

    def _start_npu_thread(self):
        threading.Thread(target=self._npu_loop, daemon=True).start()

    def _npu_loop(self):
        """后台线程：摄像头读取 → NPU 推理 → 姿势分析 → PPM 编码"""
        engine.acl.rt.set_context(engine.context)
        fc = 0
        print("[NPU] loop started")
        while self._running:
            t0 = time.perf_counter()
            ret, orig = engine.cap.read()
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

                with engine.npu_lock:
                    p = engine.acl.util.numpy_to_ptr(inp)
                    engine.acl.rt.memcpy(engine.input_dev_ptr, engine.img_size, p, engine.img_size, 1)
                    engine.acl.mdl.execute(engine.model_id, engine.input_dataset, engine.output_dataset)
                    engine.acl.rt.memcpy(engine.out_host_ptr, engine.output_size,
                                  engine.output_dev_ptr, engine.output_size, 2)
                    raw = engine.acl.util.ptr_to_bytes(engine.out_host_ptr, engine.output_size)
                    arr = np.frombuffer(raw, dtype=np.float32) if raw else np.array([])

                try:
                    _, kp = engine.parse_npu_output(arr, conf_threshold=0.15)
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

            # M0 告警（延迟触发：不良姿势需持续 > self._bad_posture_delay 秒）
            now = time.perf_counter()
            if "Warning" in status:
                if self._bad_posture_start == 0:
                    self._bad_posture_start = now
                elif now - self._bad_posture_start >= self._bad_posture_delay:
                    if "Slouching" in status or "Tilt" in status:
                        self.m0.alert('severe')
                    else:
                        self.m0.alert('warning')
                    # 记录（仅警告）
                    self.records.add(status, na, sa)
            else:
                if self._bad_posture_start > 0:
                    # 不良姿势结束，重置
                    self._bad_posture_start = 0
                self.m0.alert('good' if status == "Standard Posture" else 'none')

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

        # 如果在记录页，每 2 秒静默刷新
        if self._page == "records" and hasattr(self, '_records_page'):
            if not hasattr(self, '_last_rec_refresh'):
                self._last_rec_refresh = 0
            now = time.perf_counter()
            if now - self._last_rec_refresh > 2:
                self._refresh_records()
                self._last_rec_refresh = now

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
        self.m0.alert('none')
        self.m0.close()
        self.root.destroy()

    def run(self):
        print(">>> Posture — Soft Clinical Edition <<<")
        signal.signal(signal.SIGINT, lambda *a: self._exit())
        signal.signal(signal.SIGTERM, lambda *a: self._exit())
        self.root.protocol("WM_DELETE_WINDOW", self._exit)
        self.root.after(500, self._refresh_display)
        self.root.mainloop()
