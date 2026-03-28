# 通信协议

> 本文档定义TCP通信的封包结构、消息格式和粘包处理机制。

---

## 封包结构

### 整体结构

```
┌────────────────────────────────────────────────────────────────┐
│                        TCP Byte Stream                          │
├────────────┬───────────────────────────────────────────────────┤
│  Length    │                   Message Body                     │
│  (8 bytes) │                   (N bytes)                        │
└────────────┴───────────────────────────────────────────────────┘

Message Body 结构:
┌────────────┬────────────┬────────────┬────────────┬───────────┐
│  MsgType   │  ClientID  │ Timestamp  │  RoundID   │  Payload  │
│  (4 bytes) │  (4 bytes) │  (8 bytes) │  (4 bytes) │  (N bytes)│
└────────────┴────────────┴────────────┴────────────┴───────────┘
```

### 字段说明

| 字段 | 大小 | 类型 | 说明 |
|------|------|------|------|
| Length | 8 bytes | uint64 (big-endian) | Message Body 的总长度 |
| MsgType | 4 bytes | uint32 (big-endian) | 消息类型 |
| ClientID | 4 bytes | uint32 (big-endian) | 客户端ID，0表示无 |
| Timestamp | 8 bytes | float64 (big-endian) | 发送时间戳 |
| RoundID | 4 bytes | uint32 (big-endian) | 轮次ID，0表示无 |
| Payload | N bytes | pickle序列化 | 消息负载 |

**头部总长度：24 bytes**

---

## 消息类型定义

```python
# protocol/constants.py

class MsgType(IntEnum):
    """消息类型枚举"""

    # ========== 中心化模式 ==========
    MODEL_BROADCAST = 1      # Server -> Client: 下发全局模型
    MODEL_UPDATE = 2         # Client -> Server: 上传训练后模型
    CLIENT_REGISTER = 3      # Client -> Server: 注册请求
    CLIENT_ACK = 4           # Server -> Client: 注册确认

    # ========== P2P模式 ==========
    NODE_JOIN = 10           # 节点加入网络
    NODE_LEAVE = 11          # 节点离开网络
    RING_PASS = 12           # 环形传递模型
    HEARTBEAT = 13           # 心跳检测
    HEARTBEAT_ACK = 14       # 心跳响应
    TOPOLOGY_UPDATE = 15     # 拓扑更新通知

    # ========== 控制消息 ==========
    TRAIN_START = 20         # 开始训练信号
    TRAIN_COMPLETE = 21      # 训练完成信号
    ROUND_END = 22           # 轮次结束信号
    ERROR = 99               # 错误消息
```

---

## 消息格式定义

### 中心化模式消息

#### CLIENT_REGISTER（客户端注册）

```
方向: Client -> Server

Payload:
{
    "client_info": {
        "data_size": int,      # 本地数据量
        "model_version": str   # 支持的模型版本
    }
}
```

#### CLIENT_ACK（注册确认）

```
方向: Server -> Client

Payload:
{
    "client_id": int,          # 分配的客户端ID
    "config": {                # 训练配置
        "epochs": int,
        "lr": float,
        "batch_size": int
    }
}
```

#### MODEL_BROADCAST（广播模型）

```
方向: Server -> Client

Payload:
{
    "weights": Dict[str, Tensor],  # 模型权重
    "round_id": int,               # 当前轮次
    "model_version": str           # 模型版本
}
```

#### MODEL_UPDATE（上传更新）

```
方向: Client -> Server

Payload:
{
    "weights": Dict[str, Tensor],  # 训练后权重
    "round_id": int,               # 轮次ID
    "metrics": {                   # 训练指标
        "loss": float,
        "accuracy": float,
        "num_samples": int,
        "training_time": float
    }
}
```

### P2P模式消息

#### NODE_JOIN（节点加入）

```
方向: 新节点 -> 环中节点

Payload:
{
    "node_id": int,           # 节点ID
    "address": (str, int),    # 节点地址 (host, port)
    "timestamp": float        # 加入时间
}
```

#### NODE_LEAVE（节点离开）

```
方向: 离开节点 -> 相邻节点

Payload:
{
    "node_id": int,           # 离开节点ID
    "reason": str             # 离开原因
}
```

#### RING_PASS（环形传递）

```
方向: 当前节点 -> 下一节点

Payload:
{
    "weights": Dict[str, Tensor],  # 模型权重
    "round_id": int,               # 轮次ID
    "hop_count": int,              # 已跳数
    "origin_id": int,              # 发起节点ID
    "metrics": {                   # 本地训练指标
        "loss": float,
        "accuracy": float
    }
}
```

#### HEARTBEAT（心跳检测）

```
方向: 节点间双向

Payload:
{
    "node_id": int,           # 发送节点ID
    "timestamp": float        # 发送时间
}
```

#### HEARTBEAT_ACK（心跳响应）

```
方向: 节点间双向

Payload:
{
    "node_id": int,           # 响应节点ID
    "timestamp": float,       # 原始时间戳
    "status": str             # 节点状态 ("ok" | "busy")
}
```

#### TOPOLOGY_UPDATE（拓扑更新）

```
方向: 任意节点广播

Payload:
{
    "event": str,             # 事件类型 ("join" | "leave" | "failure")
    "node_id": int,           # 相关节点ID
    "new_ring_order": [int],  # 新的环形顺序
    "timestamp": float
}
```

### 控制消息

#### ERROR（错误消息）

```
方向: 任意

Payload:
{
    "error_code": int,        # 错误码
    "error_msg": str,         # 错误信息
    "recoverable": bool       # 是否可恢复
}
```

---

## 粘包处理

### 问题描述

TCP是面向字节流的协议，不保证消息边界。当发送多个消息时，可能出现：

1. **粘包**：多个消息被合并成一个TCP段接收
2. **拆包**：一个消息被拆分成多个TCP段接收

### 解决方案：长度前缀法

#### 发送方流程

```
1. 构造 Message Body (24字节头部 + Payload)
2. 计算 Body 长度 L
3. 发送: [8字节长度 L] + [Body]
```

```python
def send_message(conn: socket, msg: Message) -> None:
    # 1. 序列化消息
    body = encode_message(msg)

    # 2. 构造长度前缀
    length = struct.pack('!Q', len(body))

    # 3. 发送
    conn.sendall(length + body)
```

#### 接收方流程

```
1. 先接收 8 字节，解析得到长度 L
2. 循环接收直到累积 L 字节
3. 解析完整消息
```

```python
def recv_message(conn: socket, timeout: float = None) -> Message:
    conn.settimeout(timeout)

    # 1. 接收长度前缀
    length_data = recv_exact(conn, 8)
    length = struct.unpack('!Q', length_data)[0]

    # 2. 接收完整消息体
    body = recv_exact(conn, length)

    # 3. 解析消息
    return decode_message(body)


def recv_exact(conn: socket, size: int) -> bytes:
    """精确接收指定字节数"""
    data = b''
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data
```

---

## 序列化规范

### 模型权重序列化

```python
import pickle
import io
import torch

def serialize_weights(weights: Dict[str, torch.Tensor]) -> bytes:
    """
    序列化模型权重

    使用 BytesIO + pickle 保证跨平台兼容
    """
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    return buffer.getvalue()


def deserialize_weights(data: bytes) -> Dict[str, torch.Tensor]:
    """
    反序列化模型权重
    """
    buffer = io.BytesIO(data)
    return torch.load(buffer)
```

### Payload序列化

```python
import pickle

def serialize_payload(payload: Any) -> bytes:
    """序列化消息负载"""
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_payload(data: bytes) -> Any:
    """反序列化消息负载"""
    return pickle.loads(data)
```

---

## 超时处理

### 超时配置

```yaml
# config.yaml
network:
  connect_timeout: 10.0      # 连接超时
  send_timeout: 30.0         # 发送超时
  recv_timeout: 30.0         # 接收超时
  heartbeat_timeout: 5.0     # 心跳超时
  round_timeout: 60.0        # 单轮总超时
```

### 超时异常处理

```python
class NetworkTimeoutError(Exception):
    """网络超时异常"""
    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"{operation} timed out after {timeout}s")


def send_with_timeout(
    conn: socket,
    data: bytes,
    timeout: float
) -> None:
    """带超时的发送"""
    conn.settimeout(timeout)
    try:
        conn.sendall(data)
    except socket.timeout:
        raise NetworkTimeoutError("send", timeout)


def recv_with_timeout(
    conn: socket,
    size: int,
    timeout: float
) -> bytes:
    """带超时的接收"""
    conn.settimeout(timeout)
    try:
        return recv_exact(conn, size)
    except socket.timeout:
        raise NetworkTimeoutError("recv", timeout)
```

---

## 错误码定义

```python
class ErrorCode(IntEnum):
    """错误码枚举"""

    # 连接错误 (1-99)
    CONNECTION_REFUSED = 1
    CONNECTION_TIMEOUT = 2
    CONNECTION_CLOSED = 3

    # 协议错误 (100-199)
    INVALID_MESSAGE = 100
    DECODE_ERROR = 101
    INVALID_CHECKSUM = 102

    # 业务错误 (200-299)
    UNKNOWN_CLIENT = 200
    ROUND_MISMATCH = 201
    MODEL_VERSION_MISMATCH = 202

    # 系统错误 (300-399)
    INTERNAL_ERROR = 300
    OUT_OF_MEMORY = 301
```

---

## 示例代码

### 完整编解码实现

```python
# protocol/codec.py

import struct
import time
import pickle
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class Message:
    """消息结构"""
    msg_type: int
    client_id: Optional[int]
    payload: Any
    timestamp: float
    round_id: Optional[int] = None


class Codec:
    """
    编解码器

    封包结构:
    [8字节: 总长度] [4字节: 消息类型] [4字节: 客户端ID]
    [8字节: 时间戳] [4字节: round_id] [N字节: payload]
    """

    HEADER_SIZE = 24
    LENGTH_SIZE = 8

    def encode(self, msg: Message) -> bytes:
        """编码消息"""
        # 序列化payload
        payload_bytes = pickle.dumps(msg.payload, protocol=pickle.HIGHEST_PROTOCOL)

        # 构建头部 (big-endian)
        header = struct.pack(
            '!II d I',
            msg.msg_type,
            msg.client_id or 0,
            msg.timestamp,
            msg.round_id or 0
        )

        # 构建完整消息体
        body = header + payload_bytes

        # 添加长度前缀
        length = struct.pack('!Q', len(body))

        return length + body

    def decode_header(self, data: bytes) -> tuple:
        """解码头部"""
        if len(data) < self.HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {self.HEADER_SIZE}")

        msg_type, client_id, timestamp, round_id = struct.unpack('!II d I', data[:self.HEADER_SIZE])

        return (
            msg_type,
            client_id if client_id != 0 else None,
            timestamp,
            round_id if round_id != 0 else None
        )

    def decode(self, data: bytes) -> Message:
        """解码完整消息"""
        msg_type, client_id, timestamp, round_id = self.decode_header(data)

        # 解析payload
        payload_bytes = data[self.HEADER_SIZE:]
        payload = pickle.loads(payload_bytes)

        return Message(
            msg_type=msg_type,
            client_id=client_id,
            payload=payload,
            timestamp=timestamp,
            round_id=round_id
        )

    def get_message_length(self, length_data: bytes) -> int:
        """解析消息长度"""
        if len(length_data) < self.LENGTH_SIZE:
            raise ValueError(f"Length data too short: {len(length_data)}")
        return struct.unpack('!Q', length_data)[0]
```

### 连接封装

```python
# transport/connection.py

import socket
from typing import Optional, Tuple
from protocol.codec import Codec, Message

class Connection:
    """TCP连接封装"""

    def __init__(self, sock: socket.socket, codec: Codec = None):
        self._sock = sock
        self._codec = codec or Codec()
        self._connected = True

    def send(self, msg: Message) -> None:
        """发送消息"""
        data = self._codec.encode(msg)
        self._sock.sendall(data)

    def recv(self, timeout: Optional[float] = None) -> Message:
        """接收消息（处理粘包）"""
        if timeout is not None:
            self._sock.settimeout(timeout)

        # 1. 接收长度前缀
        length_data = self._recv_exact(8)
        length = self._codec.get_message_length(length_data)

        # 2. 接收完整消息体
        body = self._recv_exact(length)

        # 3. 解码消息
        return self._codec.decode(body)

    def _recv_exact(self, size: int) -> bytes:
        """精确接收指定字节数"""
        data = b''
        while len(data) < size:
            chunk = self._sock.recv(size - len(data))
            if not chunk:
                self._connected = False
                raise ConnectionError("Connection closed by peer")
            data += chunk
        return data

    def close(self) -> None:
        """关闭连接"""
        if self._connected:
            self._sock.close()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def remote_address(self) -> Tuple[str, int]:
        return self._sock.getpeername()
```
