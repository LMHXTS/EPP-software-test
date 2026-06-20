# -*- coding: utf-8 -*-
"""renderer.py — OpenCV 帧渲染：骨骼叠加 + 姿态信息面板"""

import cv2


def render_ui(frame, analysis, fps=0.0):
    """在视频帧上绘制骨骼连线及关键点（不画信息面板，省 frame.copy）"""
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

    # FPS（精简版）
    cv2.putText(frame, f"FPS:{fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return frame
