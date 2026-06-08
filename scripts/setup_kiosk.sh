#!/bin/bash
# ============================================================
# setup_kiosk.sh — HDMI Kiosk 显示方案一键安装脚本
# 适用: Ubuntu 22.04 + Ascend NPU 板卡
# 效果: 开机自动全屏显示姿态检测 Dashboard 到 HDMI 屏幕
# ============================================================
set -e

echo "=== Ascend Pose Detection - HDMI Kiosk Setup ==="
echo "Target: Auto-start fullscreen dashboard on HDMI display"
echo ""

# ---- 0. 确认以 root 或 sudo 权限运行 ----
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERR] Please run with sudo:  sudo bash setup_kiosk.sh"
    exit 1
fi

# 获取实际用户名（非 root）
REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo $USER)}"
REAL_HOME=$(eval echo ~$REAL_USER)
echo "[INFO] Target user: $REAL_USER  Home: $REAL_HOME"

# ---- 1. 安装依赖包 ----
echo ""
echo ">>> Step 1: Installing dependencies..."

apt-get update -qq

# X11 最小安装
apt-get install -y -qq xinit xserver-xorg x11-xserver-utils

# Chromium 浏览器（kiosk 模式）
apt-get install -y -qq chromium-browser --no-install-recommends 2>/dev/null || \
    apt-get install -y -qq chromium --no-install-recommends 2>/dev/null || true

# 辅助工具
apt-get install -y -qq curl unclutter

echo "[OK] Dependencies installed."

# ---- 2. 部署 .xinitrc (X 会话启动脚本) ----
echo ""
echo ">>> Step 2: Deploying .xinitrc..."

XINITRC_FILE="$REAL_HOME/.xinitrc"

cat > "$XINITRC_FILE" << 'XINITRC_EOF'
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

# 如果 Flask 没能启动，仍然尝试打开 Chromium（让用户看到错误）

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
XINITRC_EOF

chown "$REAL_USER:$REAL_USER" "$XINITRC_FILE"
chmod +x "$XINITRC_FILE"
echo "[OK] .xinitrc deployed to $XINITRC_FILE"

# ---- 3. 部署 .bash_profile (自动启动 X) ----
echo ""
echo ">>> Step 3: Deploying .bash_profile for auto X startup..."

BASH_PROFILE="$REAL_HOME/.bash_profile"

cat > "$BASH_PROFILE" << 'BASH_PROFILE_EOF'
# Auto-start X11 on tty1 after login
if [ -z "$DISPLAY" ] && [ "$(tty)" = "/dev/tty1" ]; then
    echo "Starting X11 kiosk session..."
    exec startx
fi
BASH_PROFILE_EOF

chown "$REAL_USER:$REAL_USER" "$BASH_PROFILE"
echo "[OK] .bash_profile deployed to $BASH_PROFILE"

# ---- 4. 配置自动登录 tty1 ----
echo ""
echo ">>> Step 4: Configuring auto-login on tty1..."

mkdir -p /etc/systemd/system/getty@tty1.service.d

cat > /etc/systemd/system/getty@tty1.service.d/override.conf << OVERRIDE_EOF
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $REAL_USER --noclear %I \$TERM
OVERRIDE_EOF

systemctl daemon-reload
echo "[OK] Auto-login configured for user '$REAL_USER' on tty1"

# ---- 5. 完成 ----
echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Make sure the project is at: $REAL_HOME/detect/"
echo "     (current location: $(pwd))"
echo ""
echo "  2. Reboot to test:"
echo "     sudo reboot"
echo ""
echo "  3. HDMI display will show the full Dashboard"
echo "     The Web UI is still accessible at:"
echo "     http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "  Debug logs:"
echo "    /tmp/kiosk_startup.log"
echo "    /tmp/flask_app.log"
echo ""
echo "============================================"
