# -*- coding: utf-8 -*-
"""
native_ui.py —  Ascend NPU Real-Time Posture Detection
"Soft Clinical" design — warm minimalism meets health-tech precision.
Tkinter + Canvas arc gauges + touch-friendly controls.
"""

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

import detect_main as dm


# ============================================================
#  Design Tokens  —  "Japanese Wellness Clinic"
# ============================================================
class T:
    CREAM    = "#F7F3EE"   # warm paper white
    IVORY    = "#EDE8E2"   # slightly deeper, for panel bg
    SAND     = "#D9D2C5"   # borders / dividers
    CHARCOAL = "#2C2824"   # primary text
    WARMGRY  = "#7D7872"   # secondary text
    SAGE     = "#8BB59A"   # good posture — soft green
    CORAL    = "#DF9889"   # mild warning — warm terracotta
    ROSE     = "#CF6B5E"   # bad posture — muted red
    BARK     = "#4A423A"   # dark accent, for video surround
    MIST     = "#F0EDE8"   # hover / active states
    WHITE    = "#FFFFFF"
    BLACK    = "#1A1816"

    DISP_W   = 800         # video render width
    DISP_H   = 450

    FONT     = "Lato"              # warm humanist sans — smooth & modern
    MONO     = "DejaVu Sans Mono"   # clean monospace

    RADIUS   = 16          # corner radius (simulated)


# ============================================================
#  Circular Gauge  —  Canvas-drawn arc indicator
# ============================================================
class ArcGauge(tk.Canvas):
    """A semicircular arc gauge showing a value from 0 to 60 degrees.
    Color transitions from sage (low) → coral (mid) → rose (high)."""

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

        # background arc (0° to 180°)
        self.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=180, extent=180, style="arc",
                         outline=T.SAND, width=10)

        # value arc
        angle = min(self.value / 60.0 * 180, 180)
        if self.value > self.threshold:
            c = T.ROSE if self.value > self.threshold * 1.6 else T.CORAL
        else:
            c = T.SAGE
        self.create_arc(cx - r, cy - r, cx + r, cy + r,
                         start=180, extent=-angle, style="arc",
                         outline=c, width=10)

        # center value text
        self.create_text(cx, cy - 16, text=f"{self.value:.1f}",
                          font=(T.FONT, 26, "bold"), fill=T.CHARCOAL)
        self.create_text(cx, cy + 14, text="deg",
                          font=(T.FONT, 10), fill=T.WARMGRY)

    def set(self, value, threshold):
        old_v, old_t = self.value, self.threshold
        self.value, self.threshold = value, threshold
        if abs(old_v - value) > 0.05 or abs(old_t - threshold) > 0.05:
            self._draw()


# ============================================================
#  Status Pill  —  Canvas-drawn rounded status badge
# ============================================================
class StatusPill(tk.Canvas):
    """A rounded pill badge that changes color with posture status."""

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
        # pill background
        self.create_rounded_rect(2, 2, self.w - 2, self.h - 2,
                                  r, fill=self.color, outline="")
        # text
        self.create_text(self.w / 2, self.h / 2, text=self.text,
                          font=(T.FONT, 16, "bold"), fill=T.WHITE)

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kw):
        """Draw a pill / rounded rectangle on canvas with corner circles + fill rects."""
        self.create_oval(x1, y1, x1 + 2 * r, y1 + 2 * r, fill=kw.get("fill"), outline="")
        self.create_oval(x2 - 2 * r, y1, x2, y1 + 2 * r, fill=kw.get("fill"), outline="")
        self.create_oval(x1, y2 - 2 * r, x1 + 2 * r, y2, fill=kw.get("fill"), outline="")
        self.create_oval(x2 - 2 * r, y2 - 2 * r, x2, y2, fill=kw.get("fill"), outline="")
        self.create_rectangle(x1 + r, y1, x2 - r, y2, fill=kw.get("fill"), outline="")
        self.create_rectangle(x1, y1 + r, x2, y2 - r, fill=kw.get("fill"), outline="")

    def set(self, text, color):
        if self.text != text or self.color != color:
            self.text, self.color = text, color
            self._draw()


# ============================================================
#  Main Application
# ============================================================
class PostureApp:

    def __init__(self):
        self._lock = threading.Lock()
        self._running = True
        self._detection_on = True
        self._ppm_bytes = None
        self._status = "Initializing..."
        self._neck_angle = 0.0
        self._spine_angle = 0.0
        self._fps = 0.0
        self._paused = False

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
    #  UI Construction
    # ================================================================
    def _build_ui(self):
        # -- Left: video zone with dark surround --
        video_frame = tk.Frame(self.root, bg=T.BLACK, width=self.vw, height=self.sh)
        video_frame.place(x=0, y=0)
        video_frame.pack_propagate(False)

        # subtle inner border (2px charcoal line)
        inner = tk.Frame(video_frame, bg=T.BARK, width=self.vw - 4, height=self.sh - 4)
        inner.place(x=2, y=2)

        self.video_label = tk.Label(inner, bg=T.BLACK, borderwidth=0)
        self.video_label.place(x=0, y=0, width=self.vw - 4, height=self.sh - 4)

        # pause overlay
        self.pause_label = tk.Label(
            video_frame, text="PAUSED",
            font=(T.FONT, 42, "bold"), fg=T.WHITE, bg=T.BLACK, justify="center"
        )

        # -- Right: control panel --
        p = tk.Frame(self.root, bg=T.IVORY, width=self.pw, height=self.sh)
        p.place(x=self.vw, y=0)
        p.pack_propagate(False)
        pad = 28

        # ---- Header ----
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

        # ---- Status Pill ----
        tk.Frame(p, bg=T.IVORY, height=28).pack()  # spacer
        pill_frame = tk.Frame(p, bg=T.IVORY)
        pill_frame.pack(fill='x', padx=pad)
        self.pill = StatusPill(pill_frame, width=self.pw - pad * 2, height=58)
        self.pill.pack()
        self.pill.set("Initializing...", T.WARMGRY)

        # ---- Arc Gauges Row ----
        tk.Frame(p, bg=T.IVORY, height=22).pack()
        gauges = tk.Frame(p, bg=T.IVORY)
        gauges.pack(fill='x', padx=pad - 4)

        # Neck gauge
        neck_col = tk.Frame(gauges, bg=T.IVORY)
        neck_col.pack(side='left', expand=True, fill='both')
        tk.Label(neck_col, text="NECK", font=(T.FONT, 10, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY).pack()
        self.neck_gauge = ArcGauge(neck_col, width=170, height=140)
        self.neck_gauge.pack()
        tk.Label(neck_col, text="Forward head tilt",
                 font=(T.FONT, 10), fg=T.WARMGRY, bg=T.IVORY).pack()

        # Spine gauge
        spine_col = tk.Frame(gauges, bg=T.IVORY)
        spine_col.pack(side='left', expand=True, fill='both')
        tk.Label(spine_col, text="SPINE", font=(T.FONT, 10, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY).pack()
        self.spine_gauge = ArcGauge(spine_col, width=170, height=140)
        self.spine_gauge.pack()
        tk.Label(spine_col, text="Slouch / hunch",
                 font=(T.FONT, 10), fg=T.WARMGRY, bg=T.IVORY).pack()

        # ---- Divider ----
        tk.Frame(p, bg=T.IVORY, height=26).pack()
        self._div(p, pad)

        # ---- Threshold Sliders ----
        tk.Frame(p, bg=T.IVORY, height=20).pack()
        tk.Label(p, text="THRESHOLDS", font=(T.FONT, 10, "bold"),
                 fg=T.WARMGRY, bg=T.IVORY, anchor='w').pack(fill='x', padx=pad)
        tk.Frame(p, bg=T.IVORY, height=10).pack()

        self._slider_block(p, pad, "Neck forward alert",
                           dm.PostureConfig.TH_NECK, self._on_neck,
                           self, '_neck_th_label')

        tk.Frame(p, bg=T.IVORY, height=16).pack()

        self._slider_block(p, pad, "Spine slouch alert",
                           dm.PostureConfig.TH_SPINE, self._on_spine,
                           self, '_spine_th_label')

        # ---- Divider ----
        tk.Frame(p, bg=T.IVORY, height=26).pack()
        self._div(p, pad)

        # ---- Toggle Button ----
        tk.Frame(p, bg=T.IVORY, height=22).pack()
        self.btn = tk.Button(
            p, text="STOP DETECTION", font=(T.FONT, 15, "bold"),
            fg=T.WHITE, bg=T.ROSE, relief="flat", bd=0,
            activeforeground=T.WHITE, activebackground="#B85A50",
            padx=20, pady=16, cursor="hand2",
            command=self._toggle
        )
        self.btn.pack(fill='x', padx=pad)

        # ---- FPS + Exit (bottom) ----
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
        """A subtle hairline divider."""
        d = tk.Frame(parent, bg=T.SAND, height=1)
        d.pack(fill='x', padx=pad)

    def _slider_block(self, parent, pad, label, value, cmd, store_obj, attr):
        """Build a label + slider row."""
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

        # store slider ref
        base = attr.replace('_label', '')
        setattr(store_obj, base + '_slider', s)

    # ================================================================
    #  NPU + Threading  (logic unchanged from working version)
    # ================================================================
    def _init_npu(self):
        try:
            dm.init_resources()
        except Exception as e:
            self.pill.set(f"NPU Error: {e}", T.ROSE)

    def _start_npu_thread(self):
        threading.Thread(target=self._npu_loop, daemon=True).start()

    def _npu_loop(self):
        dm.acl.rt.set_context(dm.context)
        fc = 0
        while self._running:
            t0 = time.perf_counter()
            ret, orig = dm.cap.read()
            if not ret:
                time.sleep(0.1); continue

            oh, ow = orig.shape[:2]
            do_infer = self._detection_on

            if do_infer:
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

            ms = (time.perf_counter() - t0) * 1000
            fps = 1000.0 / ms if ms > 0 else 0.0

            if kp is not None:
                kp = kp.copy()
                sx, sy = ow / 640.0, oh / 640.0
                for pt in kp:
                    pt[0] *= sx; pt[1] *= sy
                a = dm.analyze_spine_posture(kp)
                if a.get("error"):
                    status, na, sa = "No Person", 0.0, 0.0
                else:
                    status, na, sa = a["status"], a["neck_angle"], a["spine_angle"]
                disp = dm.render_ui(orig, a, fps=fps)
            else:
                status, na, sa = "No Person", 0.0, 0.0
                disp = orig
                cv2.putText(disp, f"FPS: {fps:.1f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            small = cv2.resize(disp, (T.DISP_W, T.DISP_H))
            ok, ppm = cv2.imencode('.ppm', small)
            if not ok: continue

            with self._lock:
                self._ppm_bytes = ppm.tobytes()
                self._status = status
                self._neck_angle = na
                self._spine_angle = sa
                self._fps = fps
                self._paused = not do_infer

    # ================================================================
    #  Display Refresh
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

        # Status pill
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

        # Arc gauges
        self.neck_gauge.set(na, dm.PostureConfig.TH_NECK)
        self.spine_gauge.set(sa, dm.PostureConfig.TH_SPINE)

        self.fps_label.config(text=f"{fps:.1f} fps")

        self.root.after(33, self._refresh_display)

    # ================================================================
    #  Interaction
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
        dm.PostureConfig.TH_NECK = v
        self._neck_th_label.config(text=f"{v:.1f}°")

    def _on_spine(self, val):
        v = float(val)
        dm.PostureConfig.TH_SPINE = v
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


if __name__ == '__main__':
    PostureApp().run()
