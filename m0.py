# -*- coding: utf-8 -*-
"""m0.py — M0 核心板串口通信模块（UART / CH340 / 115200bps）
协议: ASCII 文本 + \r\n 分隔
命令: BUZZER ON/OFF, MOTOR ON/OFF
响应: OK / ERROR

告警模式（蜂鸣器间隔区分等级）:
  严重 (Slouching/Tilt) → 200ms 急促蜂鸣 + 马达振动
  警告 (Forward Head/Hunchback) → 500ms 间隔蜂鸣
  良好 (Standard Posture) → 全部关闭
"""

import threading
import time

try:
    import serial
except ImportError:
    serial = None


class M0Controller:
    """M0 核心板控制器：蜂鸣器 + 振动马达 + 告警模式"""

    # 告警模式：(蜂鸣器开ms, 蜂鸣器关ms, 马达模式)
    # 马达模式: "off"=关闭, "continuous"=持续振动, "pulse"=跟随蜂鸣器节奏
    PATTERNS = {
        "severe": (200, 200, "continuous"),
        "warning": (500, 500, "pulse"),
        "good": (0, 0, "off"),
        "none": (0, 0, "off"),
    }

    def __init__(self, port="/dev/ttyUSB0", baud=115200, timeout=0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self._lock = threading.Lock()
        self._connected = False

        # 告警线程
        self._alert_thread = None
        self._alert_level = "none"  # 当前告警等级
        self._alert_running = False
        self._motor_enabled = False  # 马达暂禁用

    # ---- 连接 ----
    def connect(self):
        """打开串口连接。未连接时静默失败，不影响主程序。"""
        if serial is None:
            print("[M0] pyserial 未安装，跳过")
            return False
        try:
            self._ser = serial.Serial(
                self.port, self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            self._connected = True
            self._read_ready()
            print(f"[M0] 已连接 {self.port}")
            return True
        except (serial.SerialException, OSError) as e:
            print(f"[M0] 连接失败 {self.port}: {e}")
            self._connected = False
            return False

    def close(self):
        """关闭串口"""
        self._stop_alert()
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False

    @property
    def is_connected(self):
        return self._connected and self._ser and self._ser.is_open

    # ---- 告警模式（推荐使用） ----
    def alert(self, level):
        """设置告警等级: 'severe' / 'warning' / 'good' / 'none'
        后台线程自动控制蜂鸣器间隔和马达。"""
        if level not in self.PATTERNS:
            return
        self._alert_level = level
        if level in ("good", "none"):
            self._stop_alert()
            self.all_off()
        else:
            self._start_alert()

    def _start_alert(self):
        """启动告警后台线程（如果未运行）"""
        if self._alert_thread and self._alert_thread.is_alive():
            return
        self._alert_running = True
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._alert_thread.start()

    def _stop_alert(self):
        """停止告警后台线程，关闭所有输出"""
        self._alert_running = False
        if self._alert_thread:
            self._alert_thread.join(timeout=1)
            self._alert_thread = None
        self.all_off()

    def _alert_loop(self):
        """后台线程：按当前等级循环蜂鸣器，马达按模式独立控制"""
        while self._alert_running:
            on_ms, off_ms, motor_mode = self.PATTERNS[self._alert_level]
            if on_ms == 0:
                break
            # 蜂鸣器开
            self.buzzer(True)
            # 马达：continuous 持续振 / pulse 跟随蜂鸣器节奏
            if motor_mode == "continuous" and self._motor_enabled:
                self.motor(True)
            elif motor_mode == "pulse" and self._motor_enabled:
                self.motor(True)
            time.sleep(on_ms / 1000.0)
            # 蜂鸣器关
            self.buzzer(False)
            # 马达：continuous 不关 / pulse 跟随关
            if motor_mode == "pulse" and self._motor_enabled:
                self.motor(False)
            time.sleep(off_ms / 1000.0)

    # ---- 基础命令 ----
    def buzzer(self, on=True):
        """控制蜂鸣器 PA12 PWM（32kHz 50%占空比）"""
        self._send("BUZZER ON" if on else "BUZZER OFF")

    def motor(self, on=True):
        """控制振动马达 PB20 高/低电平（禁用时不发送）"""
        if self._motor_enabled:
            self._send("MOTOR ON" if on else "MOTOR OFF")

    def check(self):
        """查询当前状态，返回字符串如 'BUZZER:ON MOTOR:OFF'，失败返回 None"""
        if not self.is_connected:
            return None
        with self._lock:
            try:
                self._ser.write(b"CHECK\r\n")
                time.sleep(0.05)
                resp = self._ser.readline().decode("ascii").strip()
                return resp if resp else None
            except (serial.SerialException, OSError):
                self._connected = False
                return None

    def all_off(self):
        """关闭所有输出。马达仅在其开关开启时才发送 OFF"""
        self.buzzer(False)
        self.motor(False)  # motor() 内部已判断 _motor_enabled

    # ---- 内部 ----
    def _send(self, cmd):
        """发送命令 + 读取响应"""
        if not self.is_connected:
            return
        with self._lock:
            try:
                self._ser.write(f"{cmd}\r\n".encode("ascii"))
                time.sleep(0.05)
                resp = self._ser.readline().decode("ascii").strip()
                if resp and resp != "OK":
                    print(f"[M0] 响应异常: {cmd} -> {resp}")
            except (serial.SerialException, OSError) as e:
                print(f"[M0] 发送失败: {e}")
                self._connected = False

    def _read_ready(self):
        """读取上电消息 'EPP_G3507 ready'"""
        try:
            line = self._ser.readline().decode("ascii").strip()
            if line:
                print(f"[M0] {line}")
        except Exception:
            pass

    def __repr__(self):
        return f"M0Controller(port={self.port}, connected={self.is_connected})"
