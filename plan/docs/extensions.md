# 扩展接口

> 本文档为选做功能预留接口，确保架构可扩展性。

---

## 选做功能列表

| 功能 | 分值 | 说明 |
|------|------|------|
| Split Federated Learning | 10-15分 | 模型分割，Client-Server协同计算 |
| 跨设备真实部署 | 10分 | 局域网内多PC真实部署 |

---

## Split Federated Learning 接口预留

### 概述

Split Federated Learning (SplitFed) 将模型分割为两部分：
- **Client端模型**：前几层，处理原始数据
- **Server端模型**：后几层，完成推理和反向传播

```
┌─────────────┐                    ┌─────────────┐
│   Client    │                    │   Server    │
│             │                    │             │
│  ┌───────┐  │    Smashed Data    │  ┌───────┐  │
│  │ Layer │  │ ─────────────────> │  │ Layer │  │
│  │   1   │  │                    │  │   N/2 │  │
│  ├───────┤  │                    │  ├───────┤  │
│  │ Layer │  │    Gradients       │  │ Layer │  │
│  │   2   │  │ <───────────────── │  │N/2+1 │  │
│  └───────┘  │                    │  └───────┘  │
│             │                    │             │
│  [Client    │                    │  [Server   │
│   Model]    │                    │   Model]   │
└─────────────┘                    └─────────────┘
```

### 接口定义

```python
# src/core/interfaces.py (扩展)

class ISplitModel(ABC):
    """分割模型接口"""

    @abstractmethod
    def get_client_model(self) -> IModel:
        """获取客户端模型"""
        pass

    @abstractmethod
    def get_server_model(self) -> IModel:
        """获取服务端模型"""
        pass

    @abstractmethod
    def get_split_point(self) -> int:
        """获取分割点（层索引）"""
        pass


class ISplitClient(ABC):
    """SplitFed客户端接口"""

    @abstractmethod
    def forward_pass(self, data: Tensor) -> Tensor:
        """
        前向传播到分割点

        Args:
            data: 原始输入数据

        Returns:
            Tensor: Smashed data（分割点输出）
        """
        pass

    @abstractmethod
    def backward_pass(self, gradients: Tensor) -> None:
        """
        接收服务端梯度，更新客户端模型

        Args:
            gradients: 来自服务端的梯度
        """
        pass


class ISplitServer(ABC):
    """SplitFed服务端接口"""

    @abstractmethod
    def forward_pass(self, smashed_data: Tensor) -> Tensor:
        """
        从分割点继续前向传播

        Args:
            smashed_data: 客户端发送的中间数据

        Returns:
            Tensor: 最终输出
        """
        pass

    @abstractmethod
    def backward_pass(self, loss: Tensor) -> Tensor:
        """
        反向传播，返回梯度给客户端

        Args:
            loss: 损失值

        Returns:
            Tensor: 传递给客户端的梯度
        """
        pass
```

### 消息类型扩展

```python
# protocol/constants.py (扩展)

class MsgType(IntEnum):
    # ... 原有消息类型 ...

    # SplitFed消息
    SMASHED_DATA = 30        # Client -> Server: 发送中间数据
    GRADIENTS = 31           # Server -> Client: 返回梯度
    SPLIT_TRAIN_START = 32   # 开始Split训练
    SPLIT_TRAIN_END = 33     # 结束Split训练
```

### 配置扩展

```yaml
# config/splitfed.yaml

model:
  name: "SplitCNN"
  split_point: 3  # 在第3层分割

training:
  mode: "split"  # "standard" | "split"
  split:
    client_lr: 0.01
    server_lr: 0.01
```

---

## 跨设备真实部署 接口预留

### 概述

支持在局域网内多台PC上部署，通过真实IP通信。

### 配置扩展

```yaml
# config/deployment.yaml

network:
  mode: "lan"  # "localhost" | "lan"

  # localhost模式
  localhost:
    host: "127.0.0.1"
    port: 9000

  # lan模式
  lan:
    server:
      host: "192.168.1.100"  # 服务端真实IP
      port: 9000
    clients:
      - id: 1
        host: "192.168.1.101"  # 客户端1 IP
        port: 9001
      - id: 2
        host: "192.168.1.102"  # 客户端2 IP
        port: 9002
      - id: 3
        host: "192.168.1.103"  # 客户端3 IP
        port: 9003

    # 网络配置
    timeout_multiplier: 2.0  # LAN环境下超时时间倍数
    retry_attempts: 3        # 连接重试次数
```

### 接口扩展

```python
# src/core/interfaces.py (扩展)

class IDeploymentConfig(ABC):
    """部署配置接口"""

    @abstractmethod
    def get_server_address(self) -> Tuple[str, int]:
        """获取服务端地址"""
        pass

    @abstractmethod
    def get_client_address(self, client_id: int) -> Tuple[str, int]:
        """获取客户端地址"""
        pass

    @abstractmethod
    def is_localhost(self) -> bool:
        """是否为本地模式"""
        pass

    @abstractmethod
    def get_timeout(self, operation: str) -> float:
        """
        获取操作超时时间

        Args:
            operation: 操作类型

        Returns:
            float: 超时时间（秒）
        """
        pass
```

### 启动脚本扩展

```bash
# 服务端启动
python scripts/run_server.py --config config/deployment.yaml --mode lan

# 客户端启动（在不同PC上）
python scripts/run_client.py --config config/deployment.yaml --client-id 1 --mode lan
python scripts/run_client.py --config config/deployment.yaml --client-id 2 --mode lan
python scripts/run_client.py --config config/deployment.yaml --client-id 3 --mode lan
```

### 网络发现接口（可选）

```python
# src/utils/discovery.py

class INetworkDiscovery(ABC):
    """网络发现接口"""

    @abstractmethod
    def discover_server(self, broadcast_port: int = 9000) -> Optional[Tuple[str, int]]:
        """
        通过广播发现服务端

        Args:
            broadcast_port: 广播端口

        Returns:
            Optional[Tuple[str, int]]: 服务端地址
        """
        pass

    @abstractmethod
    def announce_presence(self, port: int) -> None:
        """
        广播自身存在

        Args:
            port: 本地端口
        """
        pass
```

---

## 架构扩展点

### 模块注册机制

```python
# src/core/registry.py

class ModuleRegistry:
    """模块注册器"""

    _models = {}
    _aggregators = {}
    _partitioners = {}

    @classmethod
    def register_model(cls, name: str):
        """注册模型"""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def register_aggregator(cls, name: str):
        """注册聚合器"""
        def decorator(agg_class):
            cls._aggregators[name] = agg_class
            return agg_class
        return decorator

    @classmethod
    def get_model(cls, name: str) -> type:
        """获取模型类"""
        return cls._models.get(name)

    @classmethod
    def get_aggregator(cls, name: str) -> type:
        """获取聚合器类"""
        return cls._aggregators.get(name)


# 使用示例
@ModuleRegistry.register_model("simple_cnn")
class SimpleCNN(IModel):
    pass

@ModuleRegistry.register_aggregator("fedavg")
class FedAvg(IAggregator):
    pass
```

### 配置驱动加载

```python
# src/utils/config_loader.py

def load_model(config: dict) -> IModel:
    """根据配置加载模型"""
    model_name = config["model"]["name"]
    model_class = ModuleRegistry.get_model(model_name)
    return model_class(**config["model"].get("params", {}))


def load_aggregator(config: dict) -> IAggregator:
    """根据配置加载聚合器"""
    agg_name = config["aggregator"]["name"]
    agg_class = ModuleRegistry.get_aggregator(agg_name)
    return agg_class(**config["aggregator"].get("params", {}))
```

---

## 扩展功能实现指南

### 实现SplitFed

1. **实现分割模型**
   - 继承 `ISplitModel`
   - 定义分割点
   - 分别实现客户端和服务端模型

2. **扩展消息处理**
   - 添加 `SMASHED_DATA` 和 `GRADIENTS` 消息处理

3. **修改训练流程**
   - 客户端：前向传播 -> 发送Smashed Data -> 接收梯度 -> 更新
   - 服务端：接收Smashed Data -> 完成前向 -> 计算损失 -> 反向传播 -> 发送梯度

### 实现跨设备部署

1. **配置真实IP**
   - 修改 `config/deployment.yaml`
   - 填入各设备的真实IP地址

2. **调整超时参数**
   - LAN环境下适当增加超时时间

3. **测试网络连通性**
   - 确保防火墙允许指定端口
   - 使用 `ping` 和 `telnet` 测试连通性

---

## 扩展功能测试

### SplitFed测试用例

```python
# tests/unit/test_splitfed.py

def test_split_model_forward():
    """测试分割模型前向传播"""
    model = SplitCNN(split_point=3)
    client_model = model.get_client_model()
    server_model = model.get_server_model()

    x = torch.randn(1, 3, 32, 32)
    smashed = client_model.forward(x)
    output = server_model.forward(smashed)

    assert output.shape == (1, 10)


def test_split_model_backward():
    """测试分割模型反向传播"""
    # ... 测试梯度传递
```

### 跨设备部署测试用例

```python
# tests/integration/test_lan_deployment.py

def test_lan_connection():
    """测试LAN连接"""
    config = load_config("config/deployment.yaml")
    server_addr = config.get_server_address()

    conn = Connection()
    conn.connect(*server_addr, timeout=10.0)

    assert conn.is_connected
```
