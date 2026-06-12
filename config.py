# -*- coding: utf-8 -*-
"""config.py — 全局阈值配置"""


class PostureConfig:
    """姿态检测关键点索引与判定阈值"""

    # 关键点索引（对应 YOLOv8-pose 输出的关节点序列）
    IDX_NECK_TOP = 0       # 颈部顶部
    IDX_NECK_BOTTOM = 1    # 颈部底部
    IDX_SPINE_MID = 2      # 脊柱中点
    IDX_SPINE_BASE = 3     # 脊柱底部

    # 判定阈值（角度）
    TH_NECK = 25.0          # 脖子前倾的阈值角度
    TH_SPINE = 15.0         # 脊柱弯曲（驼背）的阈值角度
    TH_CURVE = 20.0         # 颈椎与脊柱的角度差阈值
