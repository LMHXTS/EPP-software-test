# -*- coding: utf-8 -*-
"""
simple_ui.py — 基于 OpenCV HighGUI 的姿态检测 UI
零额外依赖（只用 cv2 + numpy + acl），直接全屏显示到 HDMI
支持触摸屏（鼠标回调）和键盘快捷键
"""
import sys
import time
import cv2
import numpy as np

_ENC = 'utf-8'
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding=_ENC, errors='replace')
    except Exception:
        pass

import detect_main as dm

WINDOW = "Posture Detection"
win_created = False


def make_fullscreen():
    """全屏窗口"""
    global win_created
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    win_created = True


def on_neck(val):
    dm.PostureConfig.TH_NECK = float(val)


def on_spine(val):
    dm.PostureConfig.TH_SPINE = float(val)


def on_mouse(event, x, y, flags, param):
    """触摸/鼠标事件：点击右侧面板区域可启停检测"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # 右侧面板：屏幕 72% 之后
        sw = cv2.getWindowImageRect(WINDOW)[2] if win_created else 1920
        if x > sw * 0.72:
            param["running"] = not param["running"]


def draw_panel(frame, status, neck_angle, spine_angle, fps, running):
    """在帧上绘制控制面板（右侧 28%）"""
    h, w = frame.shape[:2]
    px = int(w * 0.72)  # panel x start
    pw = w - px

    # 面板背景
    cv2.rectangle(frame, (px, 0), (w, h), (22, 30, 46), -1)

    def put(text, y, size=0.6, color=(255, 255, 255), bold=False):
        cv2.putText(frame, text, (px + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color,
                    2 if bold else 1, cv2.LINE_AA)

    cy = 30

    # 标题
    put("Ascend NPU", cy, 0.7, (78, 204, 163), True)
    cy += 35
    put("Posture Detection", cy, 0.5, (160, 160, 176))
    cy += 50

    # 状态
    put("STATUS", cy, 0.45, (160, 160, 176))
    cy += 25
    if "Warning" in status:
        c = (87, 62, 226) if "Tilt" in status or "Slouching" in status else (0, 165, 240)
    elif status == "Standard Posture":
        c = (78, 204, 163)
    else:
        c = (160, 160, 176)
    put(status, cy, 0.7, c, True)
    cy += 50

    # 角度
    put("Neck Angle", cy, 0.45, (160, 160, 176))
    cy += 25
    nc = (87, 62, 226) if neck_angle > dm.PostureConfig.TH_NECK else (78, 204, 163)
    put(f"{neck_angle:.1f} deg", cy, 0.65, nc, True)
    cy += 40

    put("Spine Angle", cy, 0.45, (160, 160, 176))
    cy += 25
    sc = (87, 62, 226) if spine_angle > dm.PostureConfig.TH_SPINE else (78, 204, 163)
    put(f"{spine_angle:.1f} deg", cy, 0.65, sc, True)
    cy += 60

    # 阈值
    put(f"Neck Thresh: {dm.PostureConfig.TH_NECK:.0f} deg", cy, 0.45, (200, 200, 200))
    cy += 30
    put(f"Spine Thresh: {dm.PostureConfig.TH_SPINE:.0f} deg", cy, 0.45, (200, 200, 200))
    cy += 50

    # 启停按钮
    if running:
        cv2.rectangle(frame, (px + 14, cy), (px + pw - 14, cy + 42), (87, 62, 226), -1)
        put("STOP (SPACE)", cy + 30, 0.5, (255, 255, 255), True)
    else:
        cv2.rectangle(frame, (px + 14, cy), (px + pw - 14, cy + 42), (78, 204, 163), -1)
        put("START (SPACE)", cy + 30, 0.5, (255, 255, 255), True)
    cy += 60

    # FPS
    put(f"FPS: {fps:.1f}", cy, 0.45, (160, 160, 176))

    return frame


def main():
    print(">>> Starting Simple UI (OpenCV HighGUI) <<<")
    dm.init_resources()
    dm.acl.rt.set_context(dm.context)

    make_fullscreen()

    # 创建滑块（阈值用键盘调整，HighGUI 滑块不友好）
    cv2.createTrackbar("Neck Th", WINDOW, int(dm.PostureConfig.TH_NECK), 60, on_neck)
    cv2.createTrackbar("Spine Th", WINDOW, int(dm.PostureConfig.TH_SPINE), 60, on_spine)

    state = {"running": True}
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    running = True
    while running:
        t0 = time.perf_counter()
        ret, orig = dm.cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        oh, ow = orig.shape[:2]

        if state["running"]:
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
                arr = np.frombuffer(raw, dtype=np.float32)

            _, kp = dm.parse_npu_output(arr, conf_threshold=0.15)
        else:
            kp = None

        ms = (time.perf_counter() - t0) * 1000
        fps = 1000.0 / ms if ms > 0 else 0.0

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

        # 右侧面板
        disp = draw_panel(disp, status, na, sa, fps, state["running"])

        cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            running = False
        elif key == 32:  # SPACE
            state["running"] = not state["running"]
        elif key == ord('f') or key == ord('F'):  # F = fullscreen toggle
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_NORMAL if cv2.getWindowProperty(
                                      WINDOW, cv2.WND_PROP_FULLSCREEN) > 0
                                  else cv2.WINDOW_FULLSCREEN)

    cv2.destroyAllWindows()
    print(">>> Exited <<<")


if __name__ == '__main__':
    main()
