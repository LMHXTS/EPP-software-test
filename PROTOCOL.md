# EPP_G3507_group5 通信协议 v1.1

## 物理层

| 项目 | 参数 |
|------|------|
| 接口 | UART1 |
| 芯片引脚 | TX=PA17, RX=PA18 |
| 波特率 | 115200 bps |
| 数据位 | 8 |
| 校验位 | None |
| 停止位 | 1 |
| 流控 | None |

## 帧格式

- **编码**: ASCII 文本
- **行分隔符**: `\r\n` (CR+LF)
- **大小写敏感**: 是（命令全大写）
- **最大帧长**: 32 字节

## 上位机 → MCU 命令

| 命令 | 动作 |
|------|------|
| `BUZZER ON` | 蜂鸣器启动 (PA12, TIMG0 PWM, 32kHz, 50%占空比) |
| `BUZZER OFF` | 蜂鸣器停止 |
| `MOTOR ON` | 振动马达启动 (PB20 高电平) |
| `MOTOR OFF` | 振动马达停止 (PB20 低电平) |
| `CHECK` | 查询当前状态 |

## MCU → 上位机 响应

| 命令 | 响应 |
|------|------|
| `BUZZER ON/OFF` | `OK` |
| `MOTOR ON/OFF` | `OK` |
| `CHECK` | `BUZZER:ON MOTOR:OFF` 或 `BUZZER:OFF MOTOR:ON` 等 |
| 未知命令 | `ERROR` |

## 上电消息

MCU 启动后发送: `EPP_G3507 ready`

## 示例交互

```
MCU:  EPP_G3507 ready
HOST: CHECK
MCU:  BUZZER:OFF MOTOR:OFF
HOST: BUZZER ON
MCU:  OK
HOST: CHECK
MCU:  BUZZER:ON MOTOR:OFF
HOST: MOTOR ON
MCU:  OK
HOST: CHECK
MCU:  BUZZER:ON MOTOR:ON
HOST: BUZZER OFF
MCU:  OK
HOST: HELLO
MCU:  ERROR
```

## 注意事项

- 5 个按键功能已禁用（后期可按需启用，协议预留 `KEY 1 PRESSED`~`KEY 5 PRESSED`）
- 串口通信通过 CH340 连接（核心板自带），PC 端识别为 COM 端口
