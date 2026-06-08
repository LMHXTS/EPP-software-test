# -*- coding: utf-8 -*-
"""
web_ui_template.py — 网页前端显示模块
包含 Flask 应用初始化、HTML 模板、路由、API 接口
从 npu_detect_web_ui.py 中分离出来以减少主文件的冗长
网页效果与 npu_detect_web.py 完全一致
"""
from flask import Flask, Response, request, jsonify, render_template_string

# ==========================================
# Flask 应用初始化
# ==========================================
app = Flask(__name__)

# ==========================================
# HTML 模板 — 与 npu_detect_web.py 保持一致
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Ascend Pose Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1e1e24;
            color: #f5f5f7;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 { color: #00ff66; margin-bottom: 10px; font-weight: 400; }
        .status-bar {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 18px;
            background: #2a2a35;
            padding: 12px 24px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .toggle-btn {
            padding: 10px 28px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            outline: none;
            letter-spacing: 0.5px;
        }
        .toggle-btn.running {
            background: #e53935;
            color: #fff;
            box-shadow: 0 0 18px rgba(229,57,53,0.45);
        }
        .toggle-btn.running:hover { background: #c62828; }
        .toggle-btn.stopped {
            background: #00c853;
            color: #fff;
            box-shadow: 0 0 18px rgba(0,200,83,0.45);
        }
        .toggle-btn.stopped:hover { background: #00a844; }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #ccc;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        .status-dot.on { background: #00ff66; box-shadow: 0 0 10px #00ff66; }
        .status-dot.off { background: #666; }
        .posture-status-box {
            margin-left: 20px;
            padding: 6px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            background: #1a1a22;
            border: 1px solid #3a3a4a;
        }
        .posture-status-box.good { color: #00ff66; border-color: #00ff66; }
        .posture-status-box.warn { color: #ff9800; border-color: #ff9800; }
        .posture-status-box.bad { color: #f44336; border-color: #f44336; }
        .posture-status-box.idle { color: #888; border-color: #555; }
        .container {
            display: flex;
            gap: 30px;
            background: #2a2a35;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .video-box {
            position: relative;
            border: 2px solid #3a3a4a;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
        }
        .video-box.paused::after {
            content: "DETECTION PAUSED";
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%,-50%);
            color: rgba(255,255,255,0.7);
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 4px;
            pointer-events: none;
        }
        .control-panel {
            width: 300px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 25px;
        }
        .slider-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        label { font-size: 14px; color: #aaa; }
        .value-display { font-weight: bold; color: #00ff66; float: right; }
        input[type=range] {
            width: 100%;
            accent-color: #00ff66;
            cursor: pointer;
        }
        .card-title {
            font-size: 18px;
            border-bottom: 1px solid #3a3a4a;
            padding-bottom: 10px;
            margin-top: 0;
            color: #00ff66;
        }
    </style>
</head>
<body>
    <h1>Ascend NPU - Real-Time Posture Detection</h1>
    <div class="status-bar">
        <button id="toggleBtn" class="toggle-btn running" onclick="toggleDetection()">STOP</button>
        <div class="status-indicator">
            <span>System:</span>
            <span id="sysDot" class="status-dot on"></span>
            <span id="sysLabel">Running</span>
        </div>
        <div id="postureBox" class="posture-status-box idle">No Person</div>
    </div>
    <div class="container">
        <div class="video-box" id="videoBox">
            <img id="videoImg" src="/video_feed">
        </div>
        <div class="control-panel">
            <h3 class="card-title">Detection Thresholds</h3>
            <div class="slider-group">
                <label>Neck Forward Angle <span id="neckVal" class="value-display">{{ neck_th }}&deg;</span></label>
                <input type="range" id="neckSlider" min="5" max="60" value="{{ neck_th }}" step="0.5"
                       oninput="updateConfig()">
            </div>
            <div class="slider-group">
                <label>Spine Slouch Angle <span id="spineVal" class="value-display">{{ spine_th }}&deg;</span></label>
                <input type="range" id="spineSlider" min="5" max="60" value="{{ spine_th }}" step="0.5"
                       oninput="updateConfig()">
            </div>
        </div>
    </div>
    <script>
        var detectionRunning = true;
        var statusPollTimer = null;

        function toggleDetection() {
            detectionRunning = !detectionRunning;
            var btn = document.getElementById('toggleBtn');
            var dot = document.getElementById('sysDot');
            var label = document.getElementById('sysLabel');
            var videoBox = document.getElementById('videoBox');
            var img = document.getElementById('videoImg');
            var postureBox = document.getElementById('postureBox');

            if (detectionRunning) {
                btn.textContent = 'STOP';
                btn.className = 'toggle-btn running';
                dot.className = 'status-dot on';
                label.textContent = 'Running';
                videoBox.classList.remove('paused');
                postureBox.className = 'posture-status-box idle';
                postureBox.textContent = 'Detecting...';
                img.src = '/video_feed';
                startStatusPolling();
            } else {
                btn.textContent = 'START';
                btn.className = 'toggle-btn stopped';
                dot.className = 'status-dot off';
                label.textContent = 'Stopped';
                videoBox.classList.add('paused');
                postureBox.className = 'posture-status-box idle';
                postureBox.textContent = 'Paused';
                img.src = '';
                stopStatusPolling();
            }

            fetch('/api/toggle_detection', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ running: detectionRunning })
            });
        }

        function startStatusPolling() {
            stopStatusPolling();
            statusPollTimer = setInterval(fetchStatus, 500);
        }

        function stopStatusPolling() {
            if (statusPollTimer) {
                clearInterval(statusPollTimer);
                statusPollTimer = null;
            }
        }

        function fetchStatus() {
            fetch('/api/status')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (!detectionRunning) return;
                    var box = document.getElementById('postureBox');
                    if (data.error) {
                        box.textContent = data.error;
                        box.className = 'posture-status-box idle';
                    } else if (data.status === 'Standard Posture') {
                        box.textContent = 'Good: ' + data.status;
                        box.className = 'posture-status-box good';
                    } else if (data.status && data.status.indexOf('Warning') !== -1) {
                        box.textContent = data.status;
                        box.className = (data.status.indexOf('Tilt') !== -1 || data.status.indexOf('Slouching') !== -1)
                            ? 'posture-status-box bad' : 'posture-status-box warn';
                    } else {
                        box.textContent = data.status || 'No Person';
                        box.className = 'posture-status-box idle';
                    }
                })
                .catch(function() {});
        }

        function updateConfig() {
            let neckTh = document.getElementById('neckSlider').value;
            let spineTh = document.getElementById('spineSlider').value;
            document.getElementById('neckVal').innerText = neckTh + "\u00B0";
            document.getElementById('spineVal').innerText = spineTh + "\u00B0";
            fetch('/api/update_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ neck_th: parseFloat(neckTh), spine_th: parseFloat(spineTh) })
            });
        }

        // 页面加载后自动开始轮询状态
        startStatusPolling();
    </script>
</body>
</html>"""


# ==========================================
# 注册 Flask 路由 (Routes)
# ==========================================

def register_routes(posture_config, generate_frames_fn):
    """
    将路由注册到 Flask app 上。
    参数:
        posture_config: PostureConfig 类，用于读取默认阈值
        generate_frames_fn: 视频流生成器函数
    """

    # 全局检测状态和最新姿态结果（线程安全共享）
    detection_state = {"running": True}
    latest_posture = {"status": "No Person", "neck_angle": 0.0, "spine_angle": 0.0}

    @app.route('/')
    def index():
        """渲染Flask主页，展示HTML监控前端界面并注入配置好的容忍阈值"""
        return render_template_string(HTML_TEMPLATE, neck_th=posture_config.TH_NECK, spine_th=posture_config.TH_SPINE)

    @app.route('/video_feed')
    def video_feed():
        """视频数据流接口：以多分支混合形式(multipart/x-mixed-replace)连续推送图片帧至前端<img>标签"""
        return Response(generate_frames_fn(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/update_config', methods=['POST'])
    def update_config():
        """应用层API接口：接收网页控制面板传递的数值，动态更新检测角度等相关配置参数"""
        data = request.json
        if 'neck_th' in data:
            posture_config.TH_NECK = data['neck_th']   # 更新颈椎角度阈值
        if 'spine_th' in data:
            posture_config.TH_SPINE = data['spine_th'] # 更新躯干脊柱前移阈值
        return jsonify({"status": "success"})

    @app.route('/api/toggle_detection', methods=['POST'])
    def toggle_detection():
        """接收前端开关按钮信号，控制检测启停"""
        data = request.json
        if 'running' in data:
            detection_state["running"] = data['running']
        return jsonify({"status": "success", "running": detection_state["running"]})

    @app.route('/api/status', methods=['GET'])
    def get_status():
        """返回当前最新的人体姿态状态供前端轮询显示"""
        return jsonify(latest_posture.copy())

    # 将状态字典暴露出去，供 generate_frames 中更新
    register_routes.detection_state = detection_state
    register_routes.latest_posture = latest_posture
