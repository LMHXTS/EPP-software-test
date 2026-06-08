#!/bin/bash
# ============================================================
# start_kiosk.sh — 桌面环境下的 Kiosk 启动脚本
# 负责后台启动 Flask 并全屏打开 Chromium
# ============================================================

PROJECT_DIR="$HOME/detect"
LOCK_FILE="/tmp/pose_kiosk.lock"

# --- 防止重复启动 ---
if [ -f "$LOCK_FILE" ]; then
    echo "Kiosk already running (lock file exists). Exiting."
    exit 0
fi
touch "$LOCK_FILE"

# --- 启动 Flask 服务 ---
cd "$PROJECT_DIR"
python3 detect_main.py >> /tmp/flask_app.log 2>&1 &
FLASK_PID=$!

# --- 等待 Flask 就绪 ---
for i in $(seq 1 30); do
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# --- 全屏启动 Chromium ---
chromium-browser \
    --kiosk \
    --no-first-run \
    --no-default-browser-check \
    --disable-sync \
    --disable-translate \
    --disable-infobars \
    --disable-session-crashed-bubble \
    http://localhost:5000 >> /tmp/kiosk.log 2>&1

# --- Chromium 关闭后清理 ---
kill $FLASK_PID 2>/dev/null
rm -f "$LOCK_FILE"
