# -*- coding: utf-8 -*-
"""
native_ui.py — 基于 Tkinter 的原生 UI，用于 HDMI 触摸屏本地显示
直接调用 detect_main 的 NPU 推理逻辑，无需 Flask / 浏览器 / 网络

用法:
    python3 native_ui.py

依赖: Python stdlib (tkinter) + numpy + cv2 + acl + detect_main.py
"""

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

# ---- 编码兜底 (同 detect_main) ----
_ENC = 'utf-8'
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding=_ENC, errors='replace')
    except Exception:
        pass

# ---- 导入 detect_main 的核心逻辑 ----
import detect_main as dm


# ==========================================
# 原生 UI 类
# ==========================================
class PostureApp:
    """全屏原生姿态检测 UI"""

    # 配色方案（暗色主题，适合长时间监控）
    BG_COLOR = "#1e1e24"
    PANEL_BG = "#2a2a35"
    GREEN = "#00ff66"
    RED = "#f44336"
    ORANGE = "#ff9800"
    TEXT_COLOR = "#f5f5f7"
    TEXT_SECONDARY = "#aaa"

    def __init__(self):
        # 检测状态
        self.detection_running = True
        self.latest_status = "Initializing..."
        self.latest_neck_angle = 0.0
        self.latest_spine_angle = 0.0
        self.current_fps = 0.0

        # 构建 UI
        self.root = tk.Tk()
        self.root.title("Ascend NPU - Posture Detection")
        self.root.configure(bg=self.BG_COLOR)

        # 全屏设置
        self.root.attributes('-fullscreen', True)
        self.root.bind('<Escape>', lambda e: self._toggle_fullscreen())
        self.root.bind('<F11>', lambda e: self._toggle_fullscreen())

        # 获取屏幕尺寸
        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()

        # 布局比例: 左侧 72% 视频，右侧 28% 控制面板
        self.video_w = int(self.sw * 0.72)
        self.video_h = self.sh
        self.panel_w = self.sw - self.video_w

        self._build_ui()
        self._init_npu()

        # 帧更新定时器 ID
        self._frame_timer = None

    # --------------------------------------------------
    # UI 构建
    # --------------------------------------------------
    def _build_ui(self):
        """构建全屏触摸 UI 布局"""
        # --- 左侧: 视频区域 ---
        self.video_label = tk.Label(
            self.root, bg="#000000", borderwidth=0,
            width=self.video_w, height=self.video_h
        )
        self.video_label.place(x=0, y=0, width=self.video_w, height=self.video_h)

        # 视频区域上的检测暂停遮罩（初始隐藏）
        self.pause_overlay = tk.Label(
            self.root, text="DETECTION\nPAUSED",
            font=("Segoe UI", 36, "bold"),
            fg="white", bg="#000000",
            justify="center"
        )

        # --- 右侧: 控制面板 ---
        px = self.video_w  # panel x
        panel = tk.Frame(self.root, bg=self.PANEL_BG,
                         width=self.panel_w, height=self.sh)
        panel.place(x=px, y=0, width=self.panel_w, height=self.sh)
        panel.pack_propagate(False)

        # 标题
        tk.Label(panel, text="Ascend NPU",
                 font=("Segoe UI", 22, "bold"),
                 fg=self.GREEN, bg=self.PANEL_BG
                 ).pack(pady=(40, 5))
        tk.Label(panel, text="Posture Detection",
                 font=("Segoe UI", 14),
                 fg=self.TEXT_SECONDARY, bg=self.PANEL_BG
                 ).pack(pady=(0, 30))

        # 分隔线
        ttk.Separator(panel, orient='horizontal').pack(fill='x', padx=20, pady=5)

        # --- 姿态状态指示器 ---
        tk.Label(panel, text="STATUS",
                 font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT_SECONDARY, bg=self.PANEL_BG
                 ).pack(pady=(25, 5))

        self.status_label = tk.Label(panel, text="Initializing...",
                                     font=("Segoe UI", 20, "bold"),
                                     fg=self.TEXT_COLOR, bg=self.PANEL_BG,
                                     wraplength=self.panel_w - 30,
                                     justify="center")
        self.status_label.pack(pady=(0, 20))

        # --- 角度数值 ---
        angle_frame = tk.Frame(panel, bg=self.PANEL_BG)
        angle_frame.pack(fill='x', padx=20, pady=5)

        tk.Label(angle_frame, text="Neck Angle",
                 font=("Segoe UI", 11), fg=self.TEXT_SECONDARY,
                 bg=self.PANEL_BG).pack(anchor='w')
        self.neck_val_label = tk.Label(angle_frame, text="0.0°",
                                       font=("Segoe UI", 32, "bold"),
                                       fg=self.GREEN, bg=self.PANEL_BG)
        self.neck_val_label.pack(anchor='w', pady=(0, 10))

        tk.Label(angle_frame, text="Spine Angle",
                 font=("Segoe UI", 11), fg=self.TEXT_SECONDARY,
                 bg=self.PANEL_BG).pack(anchor='w')
        self.spine_val_label = tk.Label(angle_frame, text="0.0°",
                                        font=("Segoe UI", 32, "bold"),
                                        fg=self.GREEN, bg=self.PANEL_BG)
        self.spine_val_label.pack(anchor='w', pady=(0, 20))

        # 分隔线
        ttk.Separator(panel, orient='horizontal').pack(fill='x', padx=20, pady=5)

        # --- 阈值滑块 ---
        tk.Label(panel, text="THRESHOLDS",
                 font=("Segoe UI", 10, "bold"),
                 fg=self.TEXT_SECONDARY, bg=self.PANEL_BG
                 ).pack(pady=(20, 10))

        # 颈部阈值
        neck_slider_frame = tk.Frame(panel, bg=self.PANEL_BG)
        neck_slider_frame.pack(fill='x', padx=20, pady=5)
        tk.Label(neck_slider_frame, text="Neck Forward Alert",
                 font=("Segoe UI", 11), fg=self.TEXT_COLOR,
                 bg=self.PANEL_BG).pack(anchor='w')
        self.neck_th_val = tk.Label(neck_slider_frame,
                                    text=f"{dm.PostureConfig.TH_NECK:.1f}°",
                                    font=("Segoe UI", 14, "bold"),
                                    fg=self.GREEN, bg=self.PANEL_BG)
        self.neck_th_val.pack(anchor='e', pady=(0, 3))

        self.neck_slider = ttk.Scale(
            neck_slider_frame, from_=5, to=60,
            value=dm.PostureConfig.TH_NECK,
            orient='horizontal', length=self.panel_w - 50,
            command=self._on_neck_slider
        )
        self.neck_slider.pack()

        # 脊柱阈值
        spine_slider_frame = tk.Frame(panel, bg=self.PANEL_BG)
        spine_slider_frame.pack(fill='x', padx=20, pady=(15, 5))
        tk.Label(spine_slider_frame, text="Spine Slouch Alert",
                 font=("Segoe UI", 11), fg=self.TEXT_COLOR,
                 bg=self.PANEL_BG).pack(anchor='w')
        self.spine_th_val = tk.Label(spine_slider_frame,
                                     text=f"{dm.PostureConfig.TH_SPINE:.1f}°",
                                     font=("Segoe UI", 14, "bold"),
                                     fg=self.GREEN, bg=self.PANEL_BG)
        self.spine_th_val.pack(anchor='e', pady=(0, 3))

        self.spine_slider = ttk.Scale(
            spine_slider_frame, from_=5, to=60,
            value=dm.PostureConfig.TH_SPINE,
            orient='horizontal', length=self.panel_w - 50,
            command=self._on_spine_slider
        )
        self.spine_slider.pack()

        # 分隔线
        ttk.Separator(panel, orient='horizontal').pack(fill='x', padx=20, pady=20)

        # --- 启停按钮 ---
        self.toggle_btn = tk.Button(
            panel, text="⏸  STOP",
            font=("Segoe UI", 18, "bold"),
            fg="white", bg=self.RED,
            activeforeground="white", activebackground="#c62828",
            relief="flat", bd=0,
            padx=20, pady=15,
            cursor="hand2",
            command=self._toggle_detection
        )
        self.toggle_btn.pack(fill='x', padx=20, pady=5)

        # --- FPS 显示 ---
        self.fps_label = tk.Label(panel, text="FPS: --",
                                  font=("Segoe UI", 11),
                                  fg=self.TEXT_SECONDARY, bg=self.PANEL_BG)
        self.fps_label.pack(pady=(20, 5))

        # --- 退出按钮 ---
        tk.Button(panel, text="Exit App",
                  font=("Segoe UI", 9),
                  fg=self.TEXT_SECONDARY, bg="#3a3a4a",
                  activeforeground="white", activebackground="#555",
                  relief="flat", bd=0,
                  padx=10, pady=5,
                  cursor="hand2",
                  command=self._on_exit
                  ).pack(side='bottom', pady=20)

    # --------------------------------------------------
    # NPU 初始化
    # --------------------------------------------------
    def _init_npu(self):
        """初始化 NPU 资源和摄像头"""
        try:
            dm.init_resources()
            print("[NativeUI] NPU resources initialized.")
        except Exception as e:
            print(f"[NativeUI] NPU init failed: {e}")
            self.status_label.config(text=f"Error: {e}", fg=self.RED)

    # --------------------------------------------------
    # 帧处理循环 (通过 root.after 调度)
    # --------------------------------------------------
    def _process_frame(self):
        """单帧处理: 摄像头 → NPU → 分析 → 渲染 → 显示"""
        loop_start = time.perf_counter()

        try:
            ret, orig_frame = dm.cap.read()
            if not ret:
                self.status_label.config(text="Camera Error", fg=self.RED)
                self._frame_timer = self.root.after(100, self._process_frame)
                return

            orig_h, orig_w = orig_frame.shape[:2]

            if self.detection_running:
                # 预处理
                img = cv2.resize(orig_frame, (640, 640))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.transpose(2, 0, 1)
                img_final = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
                img_final = np.ascontiguousarray(img_final)

                # NPU 推理（线程安全）
                with dm.npu_lock:
                    img_ptr = dm.acl.util.numpy_to_ptr(img_final)
                    dm.acl.rt.memcpy(dm.input_dev_ptr, dm.img_size,
                                     img_ptr, dm.img_size, 1)
                    dm.acl.mdl.execute(dm.model_id, dm.input_dataset, dm.output_dataset)
                    dm.acl.rt.memcpy(dm.out_host_ptr, dm.output_size,
                                     dm.output_dev_ptr, dm.output_size, 2)

                    bytes_data = dm.acl.util.ptr_to_bytes(dm.out_host_ptr, dm.output_size)
                    result_array = np.frombuffer(bytes_data, dtype=np.float32)

                # 解析结果
                box, keypoints = dm.parse_npu_output(result_array, conf_threshold=0.15)
            else:
                box, keypoints = None, None

            # 计算 FPS
            total_ms = (time.perf_counter() - loop_start) * 1000
            self.current_fps = 1000.0 / total_ms if total_ms > 0 else 0.0

            # 姿势分析 + 渲染
            if keypoints is not None:
                keypoints = keypoints.copy()
                scale_x = orig_w / 640.0
                scale_y = orig_h / 640.0
                for pt in keypoints:
                    pt[0] *= scale_x
                    pt[1] *= scale_y

                analysis = dm.analyze_spine_posture(keypoints)
                if not analysis.get("error"):
                    self.latest_status = analysis["status"]
                    self.latest_neck_angle = analysis["neck_angle"]
                    self.latest_spine_angle = analysis["spine_angle"]
                else:
                    self.latest_status = "No Person"
                    self.latest_neck_angle = 0.0
                    self.latest_spine_angle = 0.0
                frame_to_show = dm.render_ui(orig_frame, analysis, fps=self.current_fps)
            else:
                self.latest_status = "No Person"
                self.latest_neck_angle = 0.0
                self.latest_spine_angle = 0.0
                frame_to_show = orig_frame
                fps_text = f"NPU E2E FPS: {self.current_fps:.1f}"
                cv2.putText(frame_to_show, fps_text, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 缩放帧到视频区域大小
            frame_resized = cv2.resize(frame_to_show, (self.video_w, self.video_h))

            # 转换为 Tkinter PhotoImage (via PPM, 无需 PIL)
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            _, ppm = cv2.imencode('.ppm', rgb)
            photo = tk.PhotoImage(data=ppm.tobytes())

            self.video_label.config(image=photo)
            self.video_label.image = photo  # 保持引用，防止 GC

            # 隐藏/显示暂停遮罩
            if not self.detection_running:
                self.pause_overlay.place(x=0, y=0,
                                         width=self.video_w, height=self.video_h)
            else:
                self.pause_overlay.place_forget()

            # 更新状态面板
            self._update_status_panel()

        except Exception as e:
            print(f"[NativeUI] Frame error: {e}")

        # 调度下一帧 (~15ms → 最大 ~66 fps)
        self._frame_timer = self.root.after(15, self._process_frame)

    # --------------------------------------------------
    # UI 更新
    # --------------------------------------------------
    def _update_status_panel(self):
        """更新右侧控制面板的状态显示"""
        status = self.latest_status

        # 状态文字和颜色
        if "Warning" in status:
            if "Tilt" in status or "Slouching" in status:
                color = self.RED
            else:
                color = self.ORANGE
        elif status == "Standard Posture":
            color = self.GREEN
        else:
            color = self.TEXT_SECONDARY

        self.status_label.config(text=status, fg=color)

        # 角度
        neck_color = self.RED if self.latest_neck_angle > dm.PostureConfig.TH_NECK else self.GREEN
        spine_color = self.RED if self.latest_spine_angle > dm.PostureConfig.TH_SPINE else self.GREEN

        self.neck_val_label.config(
            text=f"{self.latest_neck_angle:.1f}°", fg=neck_color)
        self.spine_val_label.config(
            text=f"{self.latest_spine_angle:.1f}°", fg=spine_color)

        # FPS
        self.fps_label.config(text=f"NPU FPS: {self.current_fps:.1f}")

    # --------------------------------------------------
    # 交互回调
    # --------------------------------------------------
    def _toggle_detection(self):
        """启停检测"""
        self.detection_running = not self.detection_running
        if self.detection_running:
            self.toggle_btn.config(text="⏸  STOP", bg=self.RED,
                                   activebackground="#c62828")
        else:
            self.toggle_btn.config(text="▶  START", bg=self.GREEN,
                                   activebackground="#00a844")

    def _on_neck_slider(self, val):
        v = float(val)
        dm.PostureConfig.TH_NECK = v
        self.neck_th_val.config(text=f"{v:.1f}°")

    def _on_spine_slider(self, val):
        v = float(val)
        dm.PostureConfig.TH_SPINE = v
        self.spine_th_val.config(text=f"{v:.1f}°")

    def _toggle_fullscreen(self):
        """切换全屏"""
        is_fs = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fs)

    def _on_exit(self):
        """安全退出"""
        print("[NativeUI] Exiting...")
        if self._frame_timer:
            self.root.after_cancel(self._frame_timer)
        try:
            dm.cleanup_resources()
        except Exception:
            pass
        self.root.destroy()

    # --------------------------------------------------
    # 启动
    # --------------------------------------------------
    def run(self):
        """启动应用主循环"""
        print("\n>>> Native UI Starting... <<<")
        print(">>> HDMI Display + Touch Screen <<<")
        print(">>> Press ESC or F11 to toggle fullscreen <<<")

        # 启动第一帧（延迟 500ms 让 UI 先渲染）
        self._frame_timer = self.root.after(500, self._process_frame)

        # 设置窗口关闭时的清理
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

        self.root.mainloop()


# ==========================================
# 入口
# ==========================================
if __name__ == '__main__':
    app = PostureApp()
    app.run()
