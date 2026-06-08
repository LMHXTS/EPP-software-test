#!/bin/bash
# ============================================================
# setup_kiosk_desktop.sh — 桌面环境 Kiosk 一键配置
# 适用: Ubuntu 22.04 已安装 GNOME 桌面 + Ascend NPU
# 效果: 开机登录后自动全屏显示姿态检测 Dashboard
# ============================================================
set -e

echo "=== Kiosk Setup (Desktop Edition) ==="

REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(eval echo ~$REAL_USER)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---- 1. 修复项目路径 ----
echo ">>> Fixing project path in scripts..."

# 更新 start_kiosk.sh 中的路径
sed -i "s|PROJECT_DIR=.*|PROJECT_DIR=\"$PROJECT_DIR\"|" "$SCRIPT_DIR/start_kiosk.sh"
chmod +x "$SCRIPT_DIR/start_kiosk.sh"

# ---- 2. 更新 .desktop 文件中的用户名和路径 ----
echo ">>> Creating autostart entry..."
mkdir -p "$REAL_HOME/.config/autostart"

sed "s|\$(whoami)|$REAL_USER|g" "$SCRIPT_DIR/pose-kiosk.desktop" \
    | sed "s|/home/.*/detect|$PROJECT_DIR|g" \
    > "$REAL_HOME/.config/autostart/pose-kiosk.desktop"

chown "$REAL_USER:$REAL_USER" "$REAL_HOME/.config/autostart/pose-kiosk.desktop"

# ---- 3. 可选: 关闭屏幕休眠 ----
echo ">>> Disabling screen blanking for kiosk mode..."
sudo -u "$REAL_USER" gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true

# ---- 完成 ----
echo ""
echo "=== Done ==="
echo "Autostart entry created at: $REAL_HOME/.config/autostart/pose-kiosk.desktop"
echo ""
echo "Reboot to test:  sudo reboot"
echo "Or test now:     bash $SCRIPT_DIR/start_kiosk.sh"
