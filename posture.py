# -*- coding: utf-8 -*-
"""posture.py — 人体脊柱姿态分析（纯算法，不依赖 NPU）"""

import numpy as np
from config import PostureConfig


def calculate_angle_with_y_axis(p1, p2):
    """计算两点连线与 Y 轴（垂直方向）的夹角"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    if dy == 0:
        return 90.0
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def analyze_spine_posture(keypoints):
    """分析人体脊柱形态，判断是否存在不良姿势"""
    try:
        p_nt = keypoints[PostureConfig.IDX_NECK_TOP]
        p_nb = keypoints[PostureConfig.IDX_NECK_BOTTOM]
        p_sm = keypoints[PostureConfig.IDX_SPINE_MID]
        p_sb = keypoints[PostureConfig.IDX_SPINE_BASE]
    except IndexError:
        return {"error": f"关键点缺失，仅 {len(keypoints)} 个"}

    # 如果任一关键点坐标为 0，说明被遮挡或未检测到
    if any(p[0] == 0 for p in [p_nt, p_nb, p_sm, p_sb]):
        return {"error": "脊柱关键点被遮挡或缺失"}

    # 计算颈部和脊柱相对垂直线的倾斜角度
    neck_angle = calculate_angle_with_y_axis(p_nt, p_nb)
    spine_angle = calculate_angle_with_y_axis(p_nb, p_sb)
    # 颈部与脊柱的角度差，用于评估整体弧度
    curve_diff = abs(neck_angle - spine_angle)

    # 默认绿色 = 健康姿势
    color = (0, 255, 0)

    # 根据阈值判定姿态
    # 按优先级判定：脊柱 → 颈部 → 曲线（三种独立指标，互不重合）
    if spine_angle > PostureConfig.TH_SPINE:
        status = u"⚠ 驼背"                    # 脊柱弯曲
        color = (0, 0, 255)                   # 红色
    elif neck_angle > PostureConfig.TH_NECK:
        status = u"⚠ 头部前倾"                 # 脖子前倾
        color = (0, 165, 255)                  # 橙色
    elif curve_diff > PostureConfig.TH_CURVE:
        status = u"⚠ 圆肩驼背"                 # 肩颈异常
        color = (0, 165, 255)
    else:
        status = u"✓ 标准姿势"                  # 标准姿势

    return {
        "points": [p_nt, p_nb, p_sm, p_sb],
        "neck_angle": neck_angle,
        "spine_angle": spine_angle,
        "curve_diff": curve_diff,
        "status": status,
        "color": color,
    }
