# -*- coding: utf-8 -*-
"""detect_main.py — 基于昇腾(Ascend) NPU 的实时姿态检测核心库
包含：NPU 硬件管理、模型推理、姿态分析、帧渲染
Web 服务部分已拆分至 web_ui.py，原生 UI 在 native_ui.py"""
import sys
import locale
import time
import threading
import cv2
import numpy as np
import acl

# ---- 编码兜底逻辑 ----
# 在昇腾板子上系统 locale 可能不是 UTF-8，导致 print() 中文时抛出
# UnicodeEncodeError。此处强制将 stdout/stderr 重配置为 UTF-8。
# 注意：Python 3 已移除 sys.setdefaultencoding，改用 reconfigure()。
_ENC = 'utf-8'
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, 'reconfigure'):
            _s.reconfigure(encoding=_ENC, errors='replace')
    except Exception:
        pass

# ==========================================
# ==========================================
# 全局配置与姿态判断逻辑
# ==========================================
# ==========================================
class PostureConfig:
    # 定义关键点索引（对应YOLOv8-pose输出的关节点序列）
    IDX_NECK_TOP = 0        # 颈部顶部
    IDX_NECK_BOTTOM = 1     # 颈部底部
    IDX_SPINE_MID = 2       # 脊柱中点
    IDX_SPINE_BASE = 3      # 脊柱底部
    
    # 定义判定阈值（角度）
    TH_NECK = 25.0          # 脖子前倾的阈值角度
    TH_SPINE = 15.0         # 脊柱弯曲（驼背）的阈值角度
    TH_CURVE = 20.0         # 颈椎与脊柱的角度差阈值

def calculate_angle_with_y_axis(p1, p2):
    """计算两点连线与Y轴（垂直方向）的夹角"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    if dy == 0:
        return 90.0
    # 使用反正切函数计算弧度并转换为角度
    return np.degrees(np.arctan2(abs(dx), abs(dy)))

def analyze_spine_posture(keypoints):
    """分析人体脊柱形态，判断是否存在不良姿势"""
    try:
        # 提取需要的关键点坐标
        p_nt = keypoints[PostureConfig.IDX_NECK_TOP]
        p_nb = keypoints[PostureConfig.IDX_NECK_BOTTOM]
        p_sm = keypoints[PostureConfig.IDX_SPINE_MID]
        p_sb = keypoints[PostureConfig.IDX_SPINE_BASE]
    except IndexError:
        return {"error": f"Points lost, detected {len(keypoints)} points"}

    # 如果任一关键点坐标为0，则说明该关键点未被检测到或被遮挡
    if any(p[0] == 0 for p in [p_nt, p_nb, p_sm, p_sb]):
        return {"error": "Spine keypoints occluded or missing"}

    # 计算颈部和脊柱分别相对于垂直线的倾斜角度
    neck_angle = calculate_angle_with_y_axis(p_nt, p_nb)
    spine_angle = calculate_angle_with_y_axis(p_nb, p_sb)
    # 计算颈部与脊柱之间的角度差，用于评估整体弧度
    curve_diff = abs(neck_angle - spine_angle)

    # 默认使用绿色表示健康姿势
    color = (0, 255, 0) 
    # 根据阈值进行姿态判定，设置对应提示文字和颜色
    if spine_angle > PostureConfig.TH_SPINE:
        status = "Warning: Slouching!"  # 驼背/坐姿坍塌
        color = (0, 0, 255) # 红色警告
    elif neck_angle > PostureConfig.TH_NECK:
        status = "Warning: Forward Head!" # 脖子前倾
        color = (0, 165, 255) # 橙色警告
    elif curve_diff > PostureConfig.TH_CURVE:
        status = "Warning: Hunchback!" # 圆肩驼背
        color = (0, 165, 255) 
    else:
        status = "Standard Posture" # 标准姿势
    
    # 整体身体倾斜程度评估：计算脖顶与脊柱基底在X轴与Y轴的偏移比例
    dx_total = abs(p_nt[0] - p_sb[0])
    dy_total = abs(p_nt[1] - p_sb[1])
    if dy_total > 0 and (dx_total / dy_total) > 0.6:
        status = "Warning: Severe Body Tilt!" # 严重身体倾斜
        color = (0, 0, 255) 

    return {
        "points": [p_nt, p_nb, p_sm, p_sb],
        "neck_angle": neck_angle,
        "spine_angle": spine_angle,
        "curve_diff": curve_diff,
        "status": status,
        "color": color,
    }

# ==========================================
# ==========================================
# NPU 输出解析与帧渲染
# ==========================================
# ==========================================
def parse_npu_output(result_array, conf_threshold=0.10):
    """解析NPU返回的推理结果数组（针对YOLOv8-pose结构）
    当画面中存在多个目标时，选择检测框面积最大的那个（通常是最近的人），
    避免背景中其他人对主目标产生干扰。
    置信度阈值降低至 0.10 以提高弱光/远距离场景下的检出率。"""
    # 模型输出形如 (17, 8400)：前4行是bbox，第5行是置信度，后12行是4个关键点(x,y,conf)
    output = result_array.reshape(17, 8400)
    confidences = output[4, :]          # 提取所有的目标检测框置信度
    
    # 找出所有超过阈值的候选索引
    candidate_indices = np.where(confidences > conf_threshold)[0]
    if len(candidate_indices) == 0:
        return None, None
    
    best_idx = candidate_indices[0]
    best_area = 0.0
    
    # 遍历所有候选，选择面积（w * h）最大的目标
    for idx in candidate_indices:
        cx, cy, w, h = output[0:4, idx]
        area = w * h
        if area > best_area:
            best_area = area
            best_idx = idx
    
    # 提取最佳目标的边界框
    cx, cy, w, h = output[0:4, best_idx]
    box = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    # 提取选中目标的12项关键点数据，重新缩放为(4, 3)并丢弃置信度仅保留(x, y)
    keypoints_raw = output[5:17, best_idx] 
    keypoints = keypoints_raw.reshape(4, 3)[:, :2]
    
    # YOLOv8-pose模型输出的关键点坐标此处是相对于特征图或锚点框，代码基于此做偏移修正
    # 将相对于框内的坐标加上左上角的绝对位置得到绝对坐标
    
    return box, keypoints

def render_ui(frame, analysis, fps=0.0):
    """在视频流帧上渲染UI元素，包括检测点连线及姿态信息面板"""
    # 如果分析过程中出现错误（如关键点缺失），显示报警红字后直接返回
    if analysis.get("error"):
        cv2.putText(frame, analysis["error"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    
    pts = analysis["points"]
    color = analysis["color"]
    
    # 绘制颈部、上背部、下背部的身体主干连接线
    cv2.line(frame, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), color, 4)
    cv2.line(frame, (int(pts[1][0]), int(pts[1][1])), (int(pts[2][0]), int(pts[2][1])), color, 4)
    cv2.line(frame, (int(pts[2][0]), int(pts[2][1])), (int(pts[3][0]), int(pts[3][1])), color, 4)
    
    # 绘制捕捉到的每一个身体特征点(白点)
    for pt in pts:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), -1)
    
    # 计算信息面板的绘制坐标 (放置在画面右下角)
    h, w = frame.shape[:2]
    panel_width, panel_height = 230, 50
    x1, y1 = w - panel_width - 10, h - panel_height - 10
    x2, y2 = w - 10, h - 10
    
    # 创建黑色半透明的信息面板背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1) 
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) # 权值混合实现Alpha透明度
    
    # 渲染具体的姿态提示及角度数值
    cv2.putText(frame, analysis["status"], (x1 + 7, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(frame, f"Neck Angle: {analysis['neck_angle']:.1f}", (x1 + 7, y1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
    cv2.putText(frame, f"Spine Angle: {analysis['spine_angle']:.1f}", (x1 + 7, y1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)

    # 渲染端到端处理整体FPS(描黑边使白色/绿色文字更加清晰)
    fps_text = f"NPU E2E FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (21, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# ==========================================
# ==========================================
# NPU 硬件资源管理
# ==========================================
# ==========================================
# 线程锁：保护NPU推理资源，防止在多客户端连接时（Flask开启多线程）引发并发访问冲突
npu_lock = threading.Lock()

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
    """初始化基于昇腾(Ascend)NPU模型推理的所有硬件资源与数据结构"""
    global cap, model_id, model_desc, input_dev_ptr, output_dev_ptr, out_host_ptr
    global input_dataset, output_dataset, input_data_buffer, output_data_buffer
    global context, device_id, img_size, output_size
    
    # 1. 基础环境初始化
    acl.init()
    acl.rt.set_device(device_id) # 指定当前使用的NPU设备号
    context, _ = acl.rt.create_context(device_id) # 创建针对执行的上下文
    
    # 2. 模型加载与描述创建
    model_id, _ = acl.mdl.load_from_file("yolov8n-pose_b1.om") # 加载OM格式模型文件
    model_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(model_desc, model_id)

    # 3. 输入数据内存与数据集分配
    dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    img_size = dummy_input.nbytes # 根据输入格式预先计算出需要的字节大小
    
    input_dev_ptr, _ = acl.rt.malloc(img_size, 2) # 分配Device(NPU)侧内存
    input_dataset = acl.mdl.create_dataset()
    input_data_buffer = acl.create_data_buffer(input_dev_ptr, img_size)
    acl.mdl.add_dataset_buffer(input_dataset, input_data_buffer) # 将显存Buffer放入输入Dataset

    # 4. 输出数据内存与数据集分配
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0) # 读取模型的输出层大小
    output_dev_ptr, _ = acl.rt.malloc(output_size, 2)
    output_dataset = acl.mdl.create_dataset()
    output_data_buffer = acl.create_data_buffer(output_dev_ptr, output_size)
    acl.mdl.add_dataset_buffer(output_dataset, output_data_buffer)
    
    # 5. 分配Host主机侧内存用于接收并读取推理结果的返回数据
    out_host_ptr, _ = acl.rt.malloc_host(output_size)

    # 6. 初始化本地摄像头设备
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Error: Camera hardware cannot be opened!")
    print("--- All NPU Resources Pre-allocated Successfully ---")

def cleanup_resources():
    """释放所有NPU相关资源(清理内存、关闭摄像头、上下文销毁)，避免显存泄漏"""
    global cap, model_id, model_desc, input_dev_ptr, output_dev_ptr, out_host_ptr
    global input_dataset, output_dataset, input_data_buffer, output_data_buffer
    global context, device_id
    
    if cap: cap.release()                                   # 释放摄像头占用的流柄
    if out_host_ptr: acl.rt.free_host(out_host_ptr)         # 释放主机侧接收内存
    if input_dev_ptr: acl.rt.free(input_dev_ptr)            # 释放设备侧输入内存
    if output_dev_ptr: acl.rt.free(output_dev_ptr)          # 释放设备侧输出内存
    if input_data_buffer: acl.destroy_data_buffer(input_data_buffer) # 销毁数据缓冲区
    if output_data_buffer: acl.destroy_data_buffer(output_data_buffer)
    if input_dataset: acl.mdl.destroy_dataset(input_dataset)# 销毁数据集对象
    if output_dataset: acl.mdl.destroy_dataset(output_dataset)
    if model_desc: acl.mdl.destroy_desc(model_desc)         # 销毁关联的模型描述对象
    if model_id: acl.mdl.unload(model_id)                   # 安全卸载在跑模型
    if context: acl.rt.destroy_context(context)             # 销毁NPU上下文环境
    
    acl.rt.reset_device(device_id)                          # 重置并释放当前NPU设备
    print("--- All NPU Resources Released ---")

# ==========================================
# ==========================================
# 实时视频流生成器（Web MJPEG 流）
# ==========================================
# ==========================================
def generate_frames():
    """从摄像头读取图像，送入NPU推理并向Flask前端不断推送渲染效果后的JPEG图像"""
    global cap, model_id, input_dev_ptr, output_dev_ptr, out_host_ptr
    global input_dataset, output_dataset, img_size, output_size, context
    
    # 确保当前线程正确挂载了NPU上下文环境
    acl.rt.set_context(context)
    
    # 获取共享状态字典（由 register_routes 设置）
    detection_state = register_routes.detection_state
    latest_posture = register_routes.latest_posture
    
    while True:
        loop_start_time = time.perf_counter()
        ret, orig_frame = cap.read() # 获取最新摄像头帧
        if not ret:
            break
            
        orig_h, orig_w = orig_frame.shape[:2]

        # 图片预处理：适应模型输入要求(比如缩放至640x640，转存为RGB格式)
        img = cv2.resize(orig_frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) # 转换维度为 CHW
        # 添加Batch维度，归一化并显式转成连续(ascontiguousarray)的float32数组内存
        img_final = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        img_final = np.ascontiguousarray(img_final)

        # 检查检测开关状态：若暂停则跳过NPU推理，直接推送原始帧
        if detection_state["running"]:
            # 启用线程锁进行模型推理保护(在多线程同时访问接口读取帧时防止NPU调用争抢)
            with npu_lock:
                # 建立NumPy数据至NPU所需C指针的转换
                img_ptr = acl.util.numpy_to_ptr(img_final)
                # 拷贝图像数据至NPU显存缓冲区 (Host -> Device)
                acl.rt.memcpy(input_dev_ptr, img_size, img_ptr, img_size, 1)
                # 在NPU核心触发模型推理执行逻辑 (同步模式)
                acl.mdl.execute(model_id, input_dataset, output_dataset)
                # 将推导结果从NPU侧取回Host侧的对应内存区域内 (Device -> Host)
                acl.rt.memcpy(out_host_ptr, output_size, output_dev_ptr, output_size, 2)
                
                # 转换Host内存内容为字节流并转成NumPy数组，用于进一步分析
                bytes_data = acl.util.ptr_to_bytes(out_host_ptr, output_size)
                result_array = np.frombuffer(bytes_data, dtype=np.float32)

                # 解析YOLOv8-pose结果数组获取目标框及关节点
                box, keypoints = parse_npu_output(result_array, conf_threshold=0.15)
        else:
            box, keypoints = None, None
        
        # 计算包含NPU前向传播和解析逻辑在内的总延迟FPS (End-To-End)
        total_time_ms = (time.perf_counter() - loop_start_time) * 1000
        end_to_end_fps = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0

        if keypoints is not None:
            keypoints = keypoints.copy() 
            # 考虑到分析是在640x640进行的，通过映射系数转换回原始1280x720摄像头画布尺寸
            scale_x = orig_w / 640.0
            scale_y = orig_h / 640.0
            for pt in keypoints:
                pt[0] *= scale_x
                pt[1] *= scale_y
                
            # 执行姿势分析判断处理
            analysis = analyze_spine_posture(keypoints)
            # 更新共享状态供前端 /api/status 轮询
            if not analysis.get("error"):
                latest_posture["status"] = analysis["status"]
                latest_posture["neck_angle"] = analysis["neck_angle"]
                latest_posture["spine_angle"] = analysis["spine_angle"]
            else:
                latest_posture["status"] = "No Person"
                latest_posture["neck_angle"] = 0.0
                latest_posture["spine_angle"] = 0.0
            # 根据姿势问题叠加框线和提示
            frame_to_show = render_ui(orig_frame, analysis, fps=end_to_end_fps) 
        else:
            # 未检测到人像则保持原图并直接绘制FPS
            latest_posture["status"] = "No Person"
            latest_posture["neck_angle"] = 0.0
            latest_posture["spine_angle"] = 0.0
            frame_to_show = orig_frame
            fps_text = f"NPU E2E FPS: {end_to_end_fps:.1f}"
            cv2.putText(frame_to_show, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 当前结果帧重新编码回JPEG图片格式
        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        frame_bytes = buffer.tobytes()
        
        # 以Multipart混合边界(boundary=frame)的特殊结构流式响应，给客户端构建动态画面感知
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ==========================================
# 主入口
# ==========================================
if __name__ == '__main__':
    from web_ui import app, register_routes

    try:
        init_resources()

        # 将路由注册到 Flask app
        register_routes(PostureConfig, generate_frames)

        print("\n>>> AI Web Interactive Server Live! <<<")
        print(">>> Open your browser and visit: http://BOARD_IP:5000 <<<")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    finally:
        cleanup_resources()
