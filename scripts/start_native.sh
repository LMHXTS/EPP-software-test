#!/bin/bash
# ============================================================
# start_native.sh — HDMI 检测 + 平滑字体 + 强制本地显示
# 由 XFCE autostart 触发，天然运行在 DISPLAY=:0 上
# ============================================================
cd /root/Desktop/elec_project

# 等待 HDMI 就绪（最多等 10 秒）
STATUS="disconnected"
for i in $(seq 1 10); do
    STATUS=$(cat /sys/class/drm/card0-VGA-1/status 2>/dev/null)
    if [ "$STATUS" = "connected" ]; then
        break
    fi
    sleep 1
done

if [ "$STATUS" != "connected" ]; then
    echo "$(date): No HDMI display, skipping." >> /tmp/native_ui.log
    exit 0
fi

echo "$(date): HDMI detected, launching on :0" >> /tmp/native_ui.log

# 加载 Ascend 工具链环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export DISPLAY=:0
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtk8.6.so
/usr/local/miniconda3/bin/python3 native_ui.py >> /tmp/native_ui.log 2>&1
