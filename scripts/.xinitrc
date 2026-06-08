#!/bin/bash
# ============================================================
# .xinitrc — X11 会话启动脚本
# 1. 后台启动 Flask 姿态检测服务
# 2. 等待服务就绪后启动 Chromium 全屏显示
# 3. Chromium 关闭时自动清理
# ============================================================

# 项目路径（根据实际部署位置修改）
PROJECT_DIR="$HOME/detect"

# 日志文件
LOG_FILE="/tmp/kiosk_startup.log"
echo "$(date): Starting kiosk session..." > "$LOG_FILE"

# --- 清理可能残留的 Chromium 锁文件 ---
rm -f ~/.config/chromium/SingletonLock 2>/dev/null || true

# --- 后台启动姿态检测 Flask 服务 ---
cd "$PROJECT_DIR"
python3 detect_main.py >> /tmp/flask_app.log 2>&1 &
FLASK_PID=$!
echo "Flask PID: $FLASK_PID" >> "$LOG_FILE"

# --- 等待 Flask 服务就绪（最多等待 30 秒） ---
echo "Waiting for Flask server..." >> "$LOG_FILE"
for i in $(seq 1 30); do
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        echo "Flask ready after ${i}s" >> "$LOG_FILE"
        break
    fi
    sleep 1
done

# --- 隐藏鼠标指针（触摸屏友好） ---
unclutter -idle 1 &

# --- 启动 Chromium Kiosk 全屏模式 ---
# --kiosk: 全屏无边框
# --no-first-run: 跳过首次运行向导
# --disable-sync: 关闭同步提示
# --disable-translate: 关闭翻译提示
chromium-browser \
    --kiosk \
    --no-first-run \
    --no-default-browser-check \
    --disable-sync \
    --disable-translate \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-restore-session-state \
    --check-for-update-interval=31536000 \
    --window-size=1920,1080 \
    http://localhost:5000 >> "$LOG_FILE" 2>&1

# --- Chromium 退出后的清理 ---
echo "Chromium exited, cleaning up..." >> "$LOG_FILE"
kill $FLASK_PID 2>/dev/null || true
wait $FLASK_PID 2>/dev/null || true

echo "$(date): Kiosk session ended." >> "$LOG_FILE"
