#!/bin/bash
cd /root/Desktop/elec_project

# 防止重复启动
LOCKFILE=/tmp/native_ui.lock
if [ -f "$LOCKFILE" ]; then
    echo "$(date): Already running, skip." >> /tmp/native_ui.log
    exit 0
fi
touch "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# 清锁屏
killall xfce4-screensaver 2>/dev/null

# 等 HDMI
STATUS="disconnected"
for i in $(seq 1 10); do
    STATUS=$(cat /sys/class/drm/card0-VGA-1/status 2>/dev/null)
    [ "$STATUS" = "connected" ] && break
    sleep 1
done
[ "$STATUS" != "connected" ] && { echo "$(date): No HDMI." >> /tmp/native_ui.log; exit 0; }

# 等摄像头驱动完全初始化
for i in $(seq 1 30); do
    if [ -e /dev/video0 ]; then
        sleep 5
        break
    fi
    sleep 1
done

echo "$(date): Launching" >> /tmp/native_ui.log

ASCEND=/usr/local/Ascend/ascend-toolkit/latest
exec env DISPLAY=:0 \
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libtk8.6.so \
    LD_LIBRARY_PATH=${ASCEND}/lib64:${ASCEND}/compiler/lib64 \
    PYTHONPATH=${ASCEND}/python/site-packages:${ASCEND}/opp/built-in/op_impl/ai_core/tbe \
    ASCEND_TOOLKIT_HOME=${ASCEND} \
    /usr/local/miniconda3/bin/python3 main.py >> /tmp/native_ui.log 2>&1
