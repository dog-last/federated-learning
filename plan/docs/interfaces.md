# 接口定义

> 本文档定义所有模块间的接口契约。各成员应严格遵循接口定义，确保模块可独立开发、无缝集成。

---

## 核心类型

```python
# src/core/types.py

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum

class MsgType(IntEnum):
    """消息类型枚举"""
    # 中心化模式
    MODEL_BROADCAST = 1      # Server -> Client: 下发全局模型
    MODEL_UPDATE = 2         # Client -> Server: 上传训练后模型
    CLIENT_REGISTER = 3      # Client -> Server: 注册请求
    CLIENT_ACK = 4           # Server -> Client: 注册确认

    # P2P模式
    NODE_JOIN = 10           # 节点加入网络
    NODE_LEAVE = 11          # 节点离开网络
    RING_PASS = 12           # 环形传递模型
    HEARTBEAT = 13           # 心跳检测
    HEARTBEAT_ACK = 14       # 心跳响应
    TOPOLOGY_UPDATE = 15     # 拓扑更新通知

    # 控制消息
    TRAIN_START = 20         # 开始训练信号
    TRAIN_COMPLETE = 21      # 训练完成信号
    ROUND_END = 22           # 轮次结束信号
    ERROR = 99               # 错误消息


@dataclass
class TrainingResult:
    """训练结果"""
    loss: float
    accuracy: float
    num_samples: int
    training_time: float


@dataclass
class RoundStats:
    """轮次统计"""
    round_id: int
    broadcast_time: float
    training_times: Dict[int, float]
    collect_times: Dict[int, float]
    aggregate_time: float
    total_time: float
    participating_clients: List[int]
    timeout_clients: List[int]
    global_accuracy: float


Weights = Dict[str, Any]  # 模型权重类型别名
```

---

## 模型层接口（成员A实现）

### IModel

```python
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from core.types import Weights, TrainingResult

class IModel(ABC):
    """模型接口"""

    @abstractmethod
    def get_weights(self) -> Weights:
        """
        获取模型权重

        Returns:
            Weights: 模型权重字典 {layer_name: tensor}
        """
        pass

    @abstractmethod
    def set_weights(self, weights: Weights) -> None:
        """
        设置模型权重

        Args:
            weights: 模型权重字典
        """
        pass

    @abstractmethod
    def train_epoch(
        self,
        dataloader: DataLoader,
        lr: float,
        epoch: int = 1
    ) -> TrainingResult:
        """
        训练一个epoch

        Args:
            dataloader: 训练数据加载器
            lr: 学习率
            epoch: 当前epoch编号

        Returns:
            TrainingResult: 训练结果
        """
        pass

    @abstractmethod
    def evaluate(self, dataloader: DataLoader) -> TrainingResult:
        """
        评估模型

        Args:
            dataloader: 测试数据加载器

        Returns:
            TrainingResult: 评估结果
        """
        pass

    @property
    @abstractmethod
    def model_size(self) -> int:
        """模型参数数量"""
        pass
```

### IPartitioner

```python
from typing import List, Any

class IPartitioner(ABC):
    """数据划分接口"""

    @abstractmethod
    def partition(
        self,
        dataset: Any,
        num_clients: int,
        strategy: str = "iid"
    ) -> List[Any]:
        """
        将数据集划分为num_clients份

        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            strategy: 划分策略 ("iid" | "non_iid")

        Returns:
            List[Any]: 划分后的数据集列表
        """
        pass

    @abstractmethod
    def get_client_data(self, client_id: int) -> Any:
        """
        获取指定客户端的数据集

        Args:
            client_id: 客户端ID

        Returns:
            客户端数据集
        """
        pass
```

---

## 客户端接口（成员A实现）

### IClient

```python
from core.types import Weights

class IClient(ABC):
    """客户端接口"""

    @abstractmethod
    def connect(self, host: str, port: int) -> None:
        """
        连接服务端

        Args:
            host: 服务端地址
            port: 服务端端口
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def register(self) -> int:
        """
        向服务端注册

        Returns:
            int: 分配的客户端ID
        """
        pass

    @abstractmethod
    def receive_model(self, timeout: float = 30.0) -> Weights:
        """
        接收全局模型

        Args:
            timeout: 接收超时时间

        Returns:
            Weights: 模型权重
        """
        pass

    @abstractmethod
    def send_update(self, weights: Weights) -> None:
        """
        发送模型更新

        Args:
            weights: 训练后的模型权重
        """
        pass

    @abstractmethod
    def run(self, num_rounds: int) -> None:
        """
        运行客户端主循环

        Args:
            num_rounds: 参与的轮次数
        """
        pass

    @property
    @abstractmethod
    def client_id(self) -> int:
        """客户端ID"""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接"""
        pass
```

### ITrainer

```python
class ITrainer(ABC):
    """本地训练器接口"""

    @abstractmethod
    def train(
        self,
        model: IModel,
        dataloader: DataLoader,
        epochs: int,
        lr: float
    ) -> TrainingResult:
        """
        执行本地训练

        Args:
            model: 模型实例
            dataloader: 训练数据
            epochs: 训练轮数
            lr: 学习率

        Returns:
            TrainingResult: 训练结果
        """
        pass
```

---

## 服务端接口（成员B实现）

### IServer

```python
from typing import List, Dict, Optional
from core.types import Weights, RoundStats

class IServer(ABC):
    """服务端接口"""

    @abstractmethod
    def start(self, port: int) -> None:
        """
        启动服务端

        Args:
            port: 监听端口
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止服务端"""
        pass

    @abstractmethod
    def wait_for_clients(
        self,
        num_clients: int,
        timeout: float = 60.0
    ) -> List[int]:
        """
        等待客户端连接

        Args:
            num_clients: 期望的客户端数量
            timeout: 等待超时时间

        Returns:
            List[int]: 已连接的客户端ID列表
        """
        pass

    @abstractmethod
    def run_round(self, round_id: int) -> RoundStats:
        """
        执行一轮联邦学习

        Args:
            round_id: 轮次ID

        Returns:
            RoundStats: 轮次统计信息
        """
        pass

    @abstractmethod
    def run(self, num_rounds: int) -> None:
        """
        运行服务端主循环

        Args:
            num_rounds: 总轮次数
        """
        pass

    @property
    @abstractmethod
    def global_weights(self) -> Weights:
        """当前全局模型权重"""
        pass
```

### IAggregator

```python
class IAggregator(ABC):
    """聚合器接口"""

    @abstractmethod
    def aggregate(
        self,
        weights_list: List[Weights],
        client_sizes: Optional[List[int]] = None
    ) -> Weights:
        """
        聚合多个模型权重

        Args:
            weights_list: 模型权重列表
            client_sizes: 各客户端数据量（用于加权平均）

        Returns:
            Weights: 聚合后的权重
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """聚合器名称"""
        pass
```

### IClientManager

```python
from protocol.interfaces import IConnection

class IClientManager(ABC):
    """客户端连接管理器接口"""

    @abstractmethod
    def add_client(self, conn: IConnection) -> int:
        """
        添加客户端连接

        Args:
            conn: 连接对象

        Returns:
            int: 分配的客户端ID
        """
        pass

    @abstractmethod
    def remove_client(self, client_id: int) -> None:
        """
        移除客户端

        Args:
            client_id: 客户端ID
        """
        pass

    @abstractmethod
    def get_connection(self, client_id: int) -> Optional[IConnection]:
        """
        获取客户端连接

        Args:
            client_id: 客户端ID

        Returns:
            IConnection: 连接对象，不存在返回None
        """
        pass

    @abstractmethod
    def broadcast(
        self,
        data: bytes,
        exclude: Optional[List[int]] = None
    ) -> Dict[int, bool]:
        """
        广播数据到所有客户端

        Args:
            data: 要广播的数据
            exclude: 排除的客户端ID列表

        Returns:
            Dict[int, bool]: 各客户端发送状态
        """
        pass

    @abstractmethod
    def collect(
        self,
        timeout: float
    ) -> Dict[int, Optional[bytes]]:
        """
        收集所有客户端数据

        Args:
            timeout: 超时时间

        Returns:
            Dict[int, Optional[bytes]]: 各客户端数据，超时为None
        """
        pass

    @property
    @abstractmethod
    def client_ids(self) -> List[int]:
        """所有客户端ID列表"""
        pass

    @property
    @abstractmethod
    def num_clients(self) -> int:
        """当前客户端数量"""
        pass
```

### IRoundCoordinator

```python
class IRoundCoordinator(ABC):
    """轮次协调器接口"""

    @abstractmethod
    def start_round(self, round_id: int) -> None:
        """
        开始新一轮

        Args:
            round_id: 轮次ID
        """
        pass

    @abstractmethod
    def broadcast_model(self, weights: Weights) -> float:
        """
        广播全局模型

        Args:
            weights: 模型权重

        Returns:
            float: 广播耗时（秒）
        """
        pass

    @abstractmethod
    def collect_updates(
        self,
        timeout: float
    ) -> Dict[int, Optional[Weights]]:
        """
        收集客户端更新

        Args:
            timeout: 超时时间

        Returns:
            Dict[int, Optional[Weights]]: 各客户端更新，超时为None
        """
        pass

    @abstractmethod
    def end_round(
        self,
        aggregated_weights: Weights
    ) -> RoundStats:
        """
        结束轮次

        Args:
            aggregated_weights: 聚合后的权重

        Returns:
            RoundStats: 轮次统计
        """
        pass
```

---

## P2P接口（成员B实现）

### IP2PNode

```python
class IP2PNode(ABC):
    """P2P节点接口"""

    @abstractmethod
    def start(self, port: int) -> None:
        """
        启动节点

        Args:
            port: 监听端口
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止节点"""
        pass

    @abstractmethod
    def join_ring(self, bootstrap_node: Tuple[str, int]) -> None:
        """
        加入环形网络

        Args:
            bootstrap_node: 引导节点地址 (host, port)
        """
        pass

    @abstractmethod
    def leave_ring(self) -> None:
        """离开环形网络"""
        pass

    @abstractmethod
    def pass_model(self, weights: Weights) -> bool:
        """
        传递模型到下一节点

        Args:
            weights: 模型权重

        Returns:
            bool: 是否成功传递
        """
        pass

    @abstractmethod
    def run(self, num_rounds: int) -> None:
        """
        运行节点主循环

        Args:
            num_rounds: 参与的轮次数
        """
        pass

    @property
    @abstractmethod
    def node_id(self) -> int:
        """节点ID"""
        pass

    @property
    @abstractmethod
    def next_node(self) -> Optional[Tuple[str, int]]:
        """下一节点地址"""
        pass

    @property
    @abstractmethod
    def prev_node(self) -> Optional[Tuple[str, int]]:
        """上一节点地址"""
        pass
```

### ITopologyManager

```python
class ITopologyManager(ABC):
    """拓扑管理器接口"""

    @abstractmethod
    def get_next_node(self, current_id: int) -> Optional[int]:
        """
        获取下一节点ID

        Args:
            current_id: 当前节点ID

        Returns:
            Optional[int]: 下一节点ID，环断裂返回None
        """
        pass

    @abstractmethod
    def handle_failure(self, failed_id: int) -> None:
        """
        处理节点故障，重构拓扑

        Args:
            failed_id: 故障节点ID
        """
        pass

    @abstractmethod
    def add_node(self, node_id: int, address: Tuple[str, int]) -> None:
        """
        添加节点

        Args:
            node_id: 节点ID
            address: 节点地址
        """
        pass

    @abstractmethod
    def remove_node(self, node_id: int) -> None:
        """
        移除节点

        Args:
            node_id: 节点ID
        """
        pass

    @property
    @abstractmethod
    def ring_order(self) -> List[int]:
        """环形顺序（节点ID列表）"""
        pass
```

### IFailureDetector

```python
class IFailureDetector(ABC):
    """故障检测器接口"""

    @abstractmethod
    def start(self) -> None:
        """启动故障检测"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止故障检测"""
        pass

    @abstractmethod
    def heartbeat(self, target_id: int) -> bool:
        """
        发送心跳到目标节点

        Args:
            target_id: 目标节点ID

        Returns:
            bool: 是否收到响应
        """
        pass

    @abstractmethod
    def check(self, node_id: int, timeout: float) -> bool:
        """
        检查节点是否存活

        Args:
            node_id: 节点ID
            timeout: 超时时间

        Returns:
            bool: 节点是否存活
        """
        pass

    @abstractmethod
    def get_failed_nodes(self) -> List[int]:
        """
        获取故障节点列表

        Returns:
            List[int]: 故障节点ID列表
        """
        pass

    @abstractmethod
    def on_failure(self, callback: Callable[[int], None]) -> None:
        """
        注册故障回调

        Args:
            callback: 回调函数，参数为故障节点ID
        """
        pass
```

---

## 协议层接口（成员C实现）

### IMessage

```python
from core.types import MsgType

class IMessage(ABC):
    """消息接口"""

    @property
    @abstractmethod
    def msg_type(self) -> MsgType:
        """消息类型"""
        pass

    @property
    @abstractmethod
    def client_id(self) -> Optional[int]:
        """客户端ID"""
        pass

    @property
    @abstractmethod
    def payload(self) -> Any:
        """消息负载"""
        pass

    @property
    @abstractmethod
    def timestamp(self) -> float:
        """发送时间戳"""
        pass

    @property
    @abstractmethod
    def round_id(self) -> Optional[int]:
        """轮次ID"""
        pass
```

### ICodec

```python
class ICodec(ABC):
    """编解码接口"""

    @abstractmethod
    def encode(self, message: IMessage) -> bytes:
        """
        编码消息为字节流

        Args:
            message: 消息对象

        Returns:
            bytes: 编码后的字节流
        """
        pass

    @abstractmethod
    def decode(self, data: bytes) -> IMessage:
        """
        解码字节流为消息

        Args:
            data: 字节流

        Returns:
            IMessage: 消息对象
        """
        pass
```

### ISerializer

```python
class ISerializer(ABC):
    """序列化接口"""

    @abstractmethod
    def serialize_weights(self, weights: Weights) -> bytes:
        """
        序列化模型权重

        Args:
            weights: 模型权重

        Returns:
            bytes: 序列化后的字节流
        """
        pass

    @abstractmethod
    def deserialize_weights(self, data: bytes) -> Weights:
        """
        反序列化模型权重

        Args:
            data: 字节流

        Returns:
            Weights: 模型权重
        """
        pass

    @abstractmethod
    def get_size(self, weights: Weights) -> int:
        """
        获取序列化后的大小

        Args:
            weights: 模型权重

        Returns:
            int: 字节数
        """
        pass
```

---

## 传输层接口（成员C实现）

### IConnection

```python
class IConnection(ABC):
    """连接接口"""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """
        发送数据

        Args:
            data: 要发送的字节流
        """
        pass

    @abstractmethod
    def recv(
        self,
        size: int,
        timeout: Optional[float] = None
    ) -> bytes:
        """
        接收指定大小的数据

        Args:
            size: 期望接收的字节数
            timeout: 超时时间（秒）

        Returns:
            bytes: 接收到的数据

        Raises:
            TimeoutError: 接收超时
        """
        pass

    @abstractmethod
    def recv_all(
        self,
        timeout: Optional[float] = None
    ) -> bytes:
        """
        接收完整消息（处理粘包）

        先接收8字节长度，再接收完整数据

        Args:
            timeout: 超时时间

        Returns:
            bytes: 完整消息数据
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭连接"""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """是否已连接"""
        pass

    @property
    @abstractmethod
    def remote_address(self) -> Tuple[str, int]:
        """远程地址"""
        pass
```

### IListener

```python
class IListener(ABC):
    """监听器接口"""

    @abstractmethod
    def bind(self, host: str, port: int) -> None:
        """
        绑定地址

        Args:
            host: 主机地址
            port: 端口号
        """
        pass

    @abstractmethod
    def accept(
        self,
        timeout: Optional[float] = None
    ) -> IConnection:
        """
        接受连接

        Args:
            timeout: 超时时间

        Returns:
            IConnection: 连接对象

        Raises:
            TimeoutError: 接受超时
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭监听"""
        pass

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """是否正在监听"""
        pass

    @property
    @abstractmethod
    def address(self) -> Tuple[str, int]:
        """监听地址"""
        pass
```

---

## 工具层接口（成员D实现）

### ILogger

```python
class ILogger(ABC):
    """日志接口"""

    @abstractmethod
    def info(self, msg: str, **kwargs) -> None:
        """记录INFO级别日志"""
        pass

    @abstractmethod
    def warning(self, msg: str, **kwargs) -> None:
        """记录WARNING级别日志"""
        pass

    @abstractmethod
    def error(self, msg: str, **kwargs) -> None:
        """记录ERROR级别日志"""
        pass

    @abstractmethod
    def log_round(self, stats: RoundStats) -> None:
        """记录轮次日志"""
        pass

    @abstractmethod
    def log_network(
        self,
        event: str,
        client_id: Optional[int] = None,
        size: Optional[int] = None,
        time: Optional[float] = None
    ) -> None:
        """记录网络事件日志"""
        pass
```

### IMetricsCollector

```python
class IMetricsCollector(ABC):
    """指标收集器接口"""

    @abstractmethod
    def record_round(self, stats: RoundStats) -> None:
        """记录轮次统计"""
        pass

    @abstractmethod
    def get_accuracy_history(self) -> List[float]:
        """获取准确率历史"""
        pass

    @abstractmethod
    def get_loss_history(self) -> List[float]:
        """获取损失历史"""
        pass

    @abstractmethod
    def get_round_times(self) -> List[float]:
        """获取各轮耗时"""
        pass

    @abstractmethod
    def export(self, path: str) -> None:
        """导出指标到文件"""
        pass

    @abstractmethod
    def plot_accuracy(self, path: str) -> None:
        """绘制准确率曲线并保存"""
        pass
```

---

## 接口依赖关系

```
IModel ─────────────────────────────────────────┐
                                                 │
IPartitioner ───────────────────────────────────┤
                                                 │
IClient ────────────────────────────────────────┤
        └──> IConnection                         │
        └──> ICodec                              │
        └──> IModel                              │
                                                 │
IServer ─────────────────────────────────────────┤
        └──> IListener                           │
        └──> IClientManager                      │
        └──> IAggregator                         │
        └──> IRoundCoordinator                   │
                                                 │
IClientManager ─────────────────────────────────┤
        └──> IConnection                         │
                                                 │
IRoundCoordinator ──────────────────────────────┤
        └──> IClientManager                      │
        └──> ISerializer                         │
                                                 │
IP2PNode ───────────────────────────────────────┤
        └──> IConnection                         │
        └──> ITopologyManager                    │
        └──> IFailureDetector                    │
                                                 │
ITopologyManager ───────────────────────────────┤
        └──> IFailureDetector                    │
                                                 │
ICodec ─────────────────────────────────────────┤
        └──> IMessage                            │
        └──> ISerializer                         │
                                                 │
IConnection ────────────────────────────────────┤
        └──> ICodec (for recv_all)               │
                                                 │
ILogger ────────────────────────────────────────┤
        └──> RoundStats                          │
                                                 │
IMetricsCollector ──────────────────────────────┘
        └──> RoundStats
```
