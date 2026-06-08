#!/bin/bash
# ============================================================
# start_native.sh — 自动启动原生姿态检测 UI
# 无需 Flask / Chromium / 网络
# ============================================================

PROJECT_DIR="$HOME/detect"
LOG_FILE="/tmp/native_ui.log"

echo "$(date): Starting native posture UI..." >> "$LOG_FILE"

cd "$PROJECT_DIR"
python3 native_ui.py >> "$LOG_FILE" 2>&1
