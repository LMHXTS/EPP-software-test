#!/bin/bash
# ============================================================
# start_native.sh — 启动脚本 (HDMI 检测 + 平滑字体)
# ============================================================
cd /root/Desktop/elec_project

# 等待 HDMI 就绪（最多等 10 秒）
for i in $(seq 1 10); do
    STATUS=$(cat /sys/class/drm/card0-VGA-1/status 2>/dev/null)
    if [ "$STATUS" = "connected" ]; then
        break
    fi
    sleep 1
done

if [ "$STATUS" != "connected" ]; then
    echo "$(date): No HDMI display, skipping UI" >> /tmp/native_ui.log
    exit 0
fi

echo "$(date): HDMI detected, starting UI" >> /tmp/native_ui.log
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtk8.6.so
python3 native_ui.py >> /tmp/native_ui.log 2>&1
