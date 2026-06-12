# -*- coding: utf-8 -*-
"""engine.py — 昇腾 NPU 硬件管理、模型推理与输出解析"""

import sys
import threading
import cv2
import numpy as np
import acl

# ---- 编码兜底 ----
_ENC = 'utf-8'
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding=_ENC, errors='replace')
    except Exception:
        pass

# ==========================================
# 线程安全锁（保护 NPU 推理资源，防止并发访问冲突）
# ==========================================
npu_lock = threading.Lock()

# ==========================================
# NPU 全局资源句柄
# ==========================================
cap = None
model_id = None
model_desc = None
input_dev_ptr = None
output_dev_ptr = None
out_host_ptr = None
input_dataset = None
output_dataset = None
input_data_buffer = None
output_data_buffer = None
context = None
device_id = 0
img_size = 0
output_size = 0


def init_resources():
    """初始化昇腾 NPU 硬件资源、加载模型、打开摄像头"""
    global cap, model_id, model_desc, input_dev_ptr, output_dev_ptr, out_host_ptr
    global input_dataset, output_dataset, input_data_buffer, output_data_buffer
    global context, device_id, img_size, output_size

    # 1. 基础环境初始化
    acl.init()
    acl.rt.set_device(device_id)                      # 指定 NPU 设备号
    context, _ = acl.rt.create_context(device_id)      # 创建执行上下文

    # 2. 加载 OM 模型
    model_id, _ = acl.mdl.load_from_file("yolov8n-pose_b1.om")
    model_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(model_desc, model_id)

    # 3. 分配输入内存与数据集
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    img_size = dummy_input.nbytes

    input_dev_ptr, _ = acl.rt.malloc(img_size, 2)       # Device 侧
    input_dataset = acl.mdl.create_dataset()
    input_data_buffer = acl.create_data_buffer(input_dev_ptr, img_size)
    acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer)

    # 4. 分配输出内存与数据集
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
    output_dev_ptr, _ = acl.rt.malloc(output_size, 2)
    output_dataset = acl.mdl.create_dataset()
    output_data_buffer = acl.create_data_buffer(output_dev_ptr, output_size)
    acl.mdl.add_dataset_buffer(output_dataset, output_data_buffer)

    # 5. 分配 Host 侧接收内存
    out_host_ptr, _ = acl.rt.malloc_host(output_size)

    # 6. 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("摄像头无法打开！")
    print("--- NPU 资源预分配完成 ---")


def parse_npu_output(result_array, conf_threshold=0.10):
    """解析 NPU 推理结果（针对 YOLOv8-pose 17×8400 输出结构）
    多人时选取检测框面积最大者（最近的人），忽略背景干扰。"""
    output = result_array.reshape(17, 8400)
    confidences = output[4, :]

    candidate_indices = np.where(confidences > conf_threshold)[0]
    if len(candidate_indices) == 0:
        return None, None

    best_idx = candidate_indices[0]
    best_area = 0.0

    # 遍历候选，选面积最大的目标
    for idx in candidate_indices:
        cx, cy, w, h = output[0:4, idx]
        area = w * h
        if area > best_area:
            best_area = area
            best_idx = idx

    # 提取边界框
    cx, cy, w, h = output[0:4, best_idx]
    box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    # 提取关键点 (4 个 × 3 坐标) → 仅保留 (x, y)
    keypoints_raw = output[5:17, best_idx]
    keypoints = keypoints_raw.reshape(4, 3)[:, :2]

    return box, keypoints
