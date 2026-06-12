# -*- coding: utf-8 -*-
"""renderer.py — OpenCV 帧渲染：骨骼叠加 + 姿态信息面板"""

import cv2


def render_ui(frame, analysis, fps=0.0):
    """在视频帧上绘制骨骼连线、关键点及姿态信息面板"""
    if analysis.get("error"):
        cv2.putText(frame, analysis["error"], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    pts = analysis["points"]
    color = analysis["color"]

    # 骨骼主干连线
    cv2.line(frame, (int(pts[0][0]), int(pts[0][1])),
             (int(pts[1][0]), int(pts[1][1])), color, 4)
    cv2.line(frame, (int(pts[1][0]), int(pts[1][1])),
             (int(pts[2][0]), int(pts[2][1])), color, 4)
    cv2.line(frame, (int(pts[2][0]), int(pts[2][1])),
             (int(pts[3][0]), int(pts[3][1])), color, 4)

    # 关键点（白点）
    for pt in pts:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), -1)

    # 右下角半透明信息面板
    h, w = frame.shape[:2]
    panel_w, panel_h = 230, 50
    x1, y1 = w - panel_w - 10, h - panel_h - 10
    x2, y2 = w - 10, h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 面板文字
    cv2.putText(frame, analysis["status"], (x1 + 7, y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, f"Neck Angle: {analysis['neck_angle']:.1f}",
                (x1 + 7, y1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
    cv2.putText(frame, f"Spine Angle: {analysis['spine_angle']:.1f}",
                (x1 + 7, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)

    # FPS（描黑边增强对比）
    fps_text = f"NPU E2E FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (21, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, fps_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame
