# 配置参数详解

本文档详细说明 `config/default.yaml` 中每个配置项的含义、取值范围和使用方法。

---

## 1. 模型配置 (model)

```yaml
model:
  name: "simple_cnn"
  input_channels: 3
  num_classes: 10
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"simple_cnn"` | 模型名称，目前支持 `simple_cnn`，可扩展为 `resnet18`、`mlp` 等 |
| `input_channels` | int | `3` | 输入通道数，CIFAR-10 为 RGB 图像，固定为 3 |
| `num_classes` | int | `10` | 分类数，CIFAR-10 固定为 10 类 |

**扩展说明**：
- 模型通过工厂模式创建，新增模型只需在 `src/model/factory.py` 注册
- 模型需实现 `IModel` 接口（见 `plan/docs/interfaces.md`）

---

## 2. 数据集配置 (dataset)

```yaml
dataset:
  name: "cifar10"
  data_dir: "./data"
  num_clients: 3
  partition_strategy: "iid"
  alpha: 0.5
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"cifar10"` | 数据集名称，目前支持 CIFAR-10 |
| `data_dir` | string | `"./data"` | 数据存储目录，首次运行会自动下载 |
| `num_clients` | int | `3` | 客户端数量，建议 3-10 个 |
| `partition_strategy` | string | `"iid"` | 数据划分策略：`iid` 或 `non_iid` |
| `alpha` | float | `0.5` | **Non-IID 参数**，Dirichlet 分布浓度参数 |

### 2.1 数据划分策略详解

#### IID 模式 (`partition_strategy: "iid"`)

- **含义**：数据独立同分布，每个客户端的数据类别分布与全局分布一致
- **实现**：将数据随机打乱后均匀分配给各客户端
- **适用场景**：验证基础 FL 算法，作为 baseline

#### Non-IID 模式 (`partition_strategy: "non_iid"`)

- **含义**：数据非独立同分布，模拟真实场景中客户端数据异构
- **实现**：使用 **Dirichlet 分布**（狄利克雷分布）控制数据异构程度
- **alpha 参数详解**：

| alpha 值 | 数据异构程度 | 说明 |
|----------|--------------|------|
| `alpha = ∞` | 完全 IID | 每个客户端的数据分布与全局一致 |
| `alpha = 1.0` | 轻度异构 | 每个客户端倾向于某些类别，但仍有其他类别数据 |
| `alpha = 0.5` | 中度异构 | 客户端数据类别分布差异明显（**默认值**） |
| `alpha = 0.1` | 高度异构 | 每个客户端几乎只有 1-2 个类别的数据 |
| `alpha → 0` | 极端异构 | 每个客户端只有单一类别数据 |

**数学原理**：
```
对于每个客户端 k，其类别分布 π_k ~ Dir(alpha * p)
其中 p 是全局类别分布（CIFAR-10 中 p = [0.1, 0.1, ..., 0.1]）
alpha 越小，π_k 越稀疏，即客户端数据越集中在少数类别
```

**代码示例**（数据划分逻辑）：
```python
import numpy as np

def partition_non_iid(labels, num_clients, alpha):
    """
    使用 Dirichlet 分布划分 Non-IID 数据

    Args:
        labels: 所有样本的标签数组
        num_clients: 客户端数量
        alpha: Dirichlet 浓度参数

    Returns:
        client_indices: 每个客户端的样本索引列表
    """
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # 获取类别 c 的所有样本索引
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        # 从 Dirichlet 分布采样每个客户端的类别 c 比例
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)

        # 分配给各客户端
        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(
                class_indices[start:start + count]
            )
            start += count

    return client_indices
```

---

## 3. 训练配置 (training)

```yaml
training:
  rounds: 10
  epochs_per_round: 2
  learning_rate: 0.01
  batch_size: 32
  momentum: 0.9
  weight_decay: 0.0001
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rounds` | int | `10` | 联邦学习总轮次 |
| `epochs_per_round` | int | `2` | 每轮本地训练的 epoch 数 |
| `learning_rate` | float | `0.01` | 学习率 |
| `batch_size` | int | `32` | 批次大小 |
| `momentum` | float | `0.9` | SGD 动量参数 |
| `weight_decay` | float | `0.0001` | L2 正则化系数 |

**调参建议**：
- `rounds`：根据收敛情况调整，通常 10-50 轮可达到 85%+ 准确率
- `epochs_per_round`：Non-IID 场景建议增大到 5，减少通信轮数
- `learning_rate`：Non-IID 场景可能需要降低，如 0.001

---

## 4. 网络配置 (network)

```yaml
network:
  mode: "centralized"
  host: "127.0.0.1"
  port: 9000
  connect_timeout: 10.0
  send_timeout: 30.0
  recv_timeout: 30.0
  round_timeout: 60.0
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | `"centralized"` | 通信模式：`centralized`（中心化）或 `decentralized`（去中心化） |
| `host` | string | `"127.0.0.1"` | Server 监听地址，真实部署改为实际 IP |
| `port` | int | `9000` | Server 监听端口 |
| `connect_timeout` | float | `10.0` | TCP 连接超时（秒） |
| `send_timeout` | float | `30.0` | 发送数据超时（秒） |
| `recv_timeout` | float | `30.0` | 接收数据超时（秒） |
| `round_timeout` | float | `60.0` | 单轮聚合超时（秒），用于掉队者处理 |

### 4.1 通信模式详解

#### 中心化模式 (`mode: "centralized"`)

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Client1 │     │ Client2 │     │ Client3 │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              ┌──────▼──────┐
              │    Server   │
              │  (聚合节点)  │
              └─────────────┘
```

- **流程**：Server 广播模型 → Clients 本地训练 → Clients 上传梯度 → Server 聚合
- **适用场景**：基础功能验证、FedAvg/FedProx 聚合

#### 去中心化模式 (`mode: "decentralized"`)

```
环形拓扑（P2P）：

    Client1 ──► Client2
       ▲            │
       │            ▼
    Client4 ◄── Client3
```

- **流程**：模型参数沿环形拓扑传递，每个节点接收前驱节点的模型，本地训练后传递给后继节点
- **适用场景**：扩展功能1（去中心化通信）
- **注意**：此模式下 `p2p` 配置块生效

---

## 5. P2P 配置 (p2p)

```yaml
p2p:
  topology: "ring"
  heartbeat_interval: 5.0
  heartbeat_timeout: 10.0
  retry_count: 3
  retry_delay: 1.0
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `topology` | string | `"ring"` | 拓扑类型，目前仅支持 `ring`（环形） |
| `heartbeat_interval` | float | `5.0` | 心跳发送间隔（秒） |
| `heartbeat_timeout` | float | `10.0` | 心跳超时阈值（秒），超过则判定节点故障 |
| `retry_count` | int | `3` | 连接失败重试次数 |
| `retry_delay` | float | `1.0` | 重试间隔（秒） |

### 5.1 环形拓扑故障跳过机制

当节点故障时，环形拓扑自动跳过故障节点：

```
正常情况：Client1 → Client2 → Client3 → Client4 → Client1

Client3 故障时：
1. Client2 检测到 Client3 无响应（心跳超时）
2. Client2 尝试连接 Client4（跳过 Client3）
3. 新拓扑：Client1 → Client2 → Client4 → Client1
```

**实现逻辑**：
```python
def get_next_alive_node(current_id, nodes):
    """
    获取下一个存活节点，跳过故障节点

    Args:
        current_id: 当前节点 ID
        nodes: 所有节点列表 [(id, host, port, alive), ...]

    Returns:
        next_node: 下一个存活的节点信息
    """
    n = len(nodes)
    for offset in range(1, n):
        next_id = (current_id + offset) % n
        if nodes[next_id].alive:
            return nodes[next_id]
    return None  # 所有节点都故障
```

### 5.2 P2P 配置是否强制保留？

**回答：是的，P2P 配置必须保留**

原因：
1. **扩展功能1（10分）** 要求实现去中心化通信
2. 去中心化模式 (`network.mode: "decentralized"`) 依赖 `p2p` 配置块
3. 即使默认使用中心化模式，配置文件也需包含所有模式的配置，便于切换

**配置切换示例**：
```yaml
# 中心化模式（默认）
network:
  mode: "centralized"
  # p2p 配置块存在但不生效

# 切换到去中心化模式
network:
  mode: "decentralized"
  # p2p 配置块生效
p2p:
  topology: "ring"
  # ...
```

---

## 6. 聚合配置 (aggregator)

```yaml
aggregator:
  name: "fedavg"
  fedprox_mu: 0.01
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"fedavg"` | 聚合算法名称：`fedavg` 或 `fedprox` |
| `fedprox_mu` | float | `0.01` | FedProx 近端项系数，仅当 `name: "fedprox"` 时生效 |

### 6.1 FedAvg 算法

**标准联邦平均算法**，无需额外参数。

**聚合公式**：
```
w_{t+1} = Σ (n_k / n) * w_k^t
```
其中：
- `w_{t+1}`：第 t+1 轮全局模型参数
- `w_k^t`：第 k 个客户端第 t 轮的本地模型参数
- `n_k`：第 k 个客户端的样本数
- `n`：总样本数

### 6.2 FedProx 算法详解

**FedProx 是什么？**

FedProx（Federated Proximal）是 FedAvg 的改进版本，专门解决 **Non-IID 数据异构** 问题。

**核心思想**：
- 在本地训练目标函数中添加**近端项（proximal term）**
- 限制本地模型不要偏离全局模型太远
- 提高在异构数据上的收敛稳定性

**本地训练目标函数**：
```
min_w  F_k(w) + (μ/2) * ||w - w^t||²
        ↑              ↑
    本地损失      近端项（约束偏离程度）
```

其中：
- `F_k(w)`：客户端 k 的本地损失函数
- `w^t`：第 t 轮全局模型参数
- `μ`：近端项系数（即配置中的 `fedprox_mu`）

**fedprox_mu 参数详解**：

| mu 值 | 效果 | 适用场景 |
|-------|------|----------|
| `mu = 0` | 退化为 FedAvg | IID 数据 |
| `mu = 0.001` | 轻度约束 | 轻度 Non-IID |
| `mu = 0.01` | 中度约束（**默认**） | 中度 Non-IID |
| `mu = 0.1` | 强约束 | 高度 Non-IID |
| `mu = 1.0` | 极强约束 | 极端 Non-IID |

**代码示例**：
```python
import torch

def fedprox_local_train(model, global_model, dataloader, mu, lr, epochs):
    """
    FedProx 本地训练

    Args:
        model: 本地模型
        global_model: 全局模型（用于计算近端项）
        mu: 近端项系数
        lr: 学习率
        epochs: 本地训练轮数
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    global_params = list(global_model.parameters())

    for _ in range(epochs):
        for data, target in dataloader:
            optimizer.zero_grad()

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 添加近端项
            proximal_term = 0.0
            for local_param, global_param in zip(model.parameters(), global_params):
                proximal_term += torch.sum((local_param - global_param) ** 2)

            loss += (mu / 2) * proximal_term

            # 反向传播
            loss.backward()
            optimizer.step()
```

### 6.3 如何选择聚合算法？

| 场景 | 推荐算法 | 配置 |
|------|----------|------|
| IID 数据 | FedAvg | `name: "fedavg"` |
| 轻度 Non-IID (`alpha ≥ 1.0`) | FedAvg 或 FedProx | `name: "fedprox"`, `fedprox_mu: 0.001` |
| 中度 Non-IID (`alpha ≈ 0.5`) | FedProx | `name: "fedprox"`, `fedprox_mu: 0.01` |
| 高度 Non-IID (`alpha ≤ 0.1`) | FedProx | `name: "fedprox"`, `fedprox_mu: 0.1` |

---

## 7. 日志配置 (logging)

```yaml
logging:
  level: "INFO"
  log_dir: "./logs"
  console_output: true
  file_output: true
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | 日志级别：`DEBUG` < `INFO` < `WARNING` < `ERROR` |
| `log_dir` | string | `"./logs"` | 日志文件存储目录 |
| `console_output` | bool | `true` | 是否输出到控制台 |
| `file_output` | bool | `true` | 是否输出到文件 |

**日志格式**（见 `plan/docs/monitoring.md`）：
```
[2024-01-15 10:30:45] [INFO] [Server] Round 1/10 started
[2024-01-15 10:30:47] [INFO] [Client-1] Received model, size=1.2MB
```

---

## 8. 输出配置 (output)

```yaml
output:
  checkpoint_dir: "./outputs/checkpoints"
  figure_dir: "./outputs/figures"
  save_checkpoint_every: 5
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_dir` | string | `"./outputs/checkpoints"` | 模型检查点存储目录 |
| `figure_dir` | string | `"./outputs/figures"` | 训练曲线图存储目录 |
| `save_checkpoint_every` | int | `5` | 每 N 轮保存一次检查点 |

---

## 9. 完整配置示例

### 9.1 基础配置（IID 数据 + FedAvg）

```yaml
model:
  name: "simple_cnn"
  input_channels: 3
  num_classes: 10

dataset:
  name: "cifar10"
  data_dir: "./data"
  num_clients: 3
  partition_strategy: "iid"

training:
  rounds: 10
  epochs_per_round: 2
  learning_rate: 0.01
  batch_size: 32

network:
  mode: "centralized"
  host: "127.0.0.1"
  port: 9000
  round_timeout: 60.0

aggregator:
  name: "fedavg"

logging:
  level: "INFO"
```

### 9.2 Non-IID 配置（中度异构 + FedProx）

```yaml
dataset:
  partition_strategy: "non_iid"
  alpha: 0.5

aggregator:
  name: "fedprox"
  fedprox_mu: 0.01
```

### 9.3 去中心化配置（环形拓扑）

```yaml
network:
  mode: "decentralized"

p2p:
  topology: "ring"
  heartbeat_interval: 5.0
  heartbeat_timeout: 10.0
  retry_count: 3
```

### 9.4 高度 Non-IID 配置

```yaml
dataset:
  partition_strategy: "non_iid"
  alpha: 0.1

training:
  epochs_per_round: 5  # 增加本地训练
  learning_rate: 0.001  # 降低学习率

aggregator:
  name: "fedprox"
  fedprox_mu: 0.1  # 增强约束
```

---

## 10. 配置加载接口

配置通过 `IConfig` 接口加载（见 `plan/docs/interfaces.md`）：

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    name: str
    input_channels: int
    num_classes: int

@dataclass
class DatasetConfig:
    name: str
    data_dir: str
    num_clients: int
    partition_strategy: Literal["iid", "non_iid"]
    alpha: float

@dataclass
class TrainingConfig:
    rounds: int
    epochs_per_round: int
    learning_rate: float
    batch_size: int
    momentum: float
    weight_decay: float

@dataclass
class NetworkConfig:
    mode: Literal["centralized", "decentralized"]
    host: str
    port: int
    connect_timeout: float
    send_timeout: float
    recv_timeout: float
    round_timeout: float

@dataclass
class P2PConfig:
    topology: Literal["ring"]
    heartbeat_interval: float
    heartbeat_timeout: float
    retry_count: int
    retry_delay: float

@dataclass
class AggregatorConfig:
    name: Literal["fedavg", "fedprox"]
    fedprox_mu: float

@dataclass
class LoggingConfig:
    level: str
    log_dir: str
    console_output: bool
    file_output: bool

@dataclass
class OutputConfig:
    checkpoint_dir: str
    figure_dir: str
    save_checkpoint_every: int

@dataclass
class Config:
    """顶层配置类"""
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    network: NetworkConfig
    p2p: P2PConfig
    aggregator: AggregatorConfig
    logging: LoggingConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data["model"]),
            dataset=DatasetConfig(**data["dataset"]),
            training=TrainingConfig(**data["training"]),
            network=NetworkConfig(**data["network"]),
            p2p=P2PConfig(**data["p2p"]),
            aggregator=AggregatorConfig(**data["aggregator"]),
            logging=LoggingConfig(**data["logging"]),
            output=OutputConfig(**data["output"]),
        )
```

---

## 11. 常见问题

### Q1: alpha 和 fedprox_mu 有什么关系？

- **alpha**：控制数据异构程度（数据层面）
- **fedprox_mu**：控制算法对异构数据的适应能力（算法层面）

**经验法则**：`fedprox_mu ≈ alpha` 是一个合理的起点，然后根据实验结果微调。

### Q2: 什么时候需要调整 round_timeout？

- 网络环境差时，增大 `round_timeout`
- 客户端计算能力差异大时，增大 `round_timeout`
- 模型较大时，增大 `round_timeout`

### Q3: P2P 模式下如何处理节点加入/离开？

当前设计为固定节点列表，动态加入/离开作为可选扩展功能预留接口。

### Q4: 如何验证配置是否正确？

```python
config = Config.from_yaml("config/default.yaml")
print(config.dataset.partition_strategy)  # "iid" or "non_iid"
print(config.aggregator.name)  # "fedavg" or "fedprox"
```
