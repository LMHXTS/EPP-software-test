# -*- coding: utf-8 -*-
"""m0.py — M0 核心板串口通信模块（UART / CH340 / 115200bps）
协议: ASCII 文本 + \r\n 分隔
命令: BUZZER ON/OFF, MOTOR ON/OFF
响应: OK / ERROR
"""

import threading
import time

try:
    import serial
except ImportError:
    serial = None


class M0Controller:
    """M0 核心板控制器：蜂鸣器 + 振动马达"""

    def __init__(self, port="/dev/ttyUSB0", baud=115200, timeout=0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self._lock = threading.Lock()
        self._connected = False

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
            # 等待上电消息
            self._read_ready()
            print(f"[M0] 已连接 {self.port}")
            return True
        except (serial.SerialException, OSError) as e:
            print(f"[M0] 连接失败 {self.port}: {e}")
            self._connected = False
            return False

    def close(self):
        """关闭串口"""
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._connected = False

    @property
    def is_connected(self):
        return self._connected and self._ser and self._ser.is_open

    # ---- 命令 ----
    def buzzer(self, on=True):
        """控制蜂鸣器 PA12 PWM"""
        self._send("BUZZER ON" if on else "BUZZER OFF")

    def motor(self, on=True):
        """控制振动马达 PB20 高/低电平"""
        self._send("MOTOR ON" if on else "MOTOR OFF")

    def all_off(self):
        """关闭所有输出"""
        self.buzzer(False)
        self.motor(False)

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
