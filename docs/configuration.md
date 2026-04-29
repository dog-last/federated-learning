# 配置说明

本文档详细说明联邦学习系统的配置选项。

## 配置文件结构

```yaml
mode: centralized  # 运行模式

model:
  # 模型配置

dataset:
  # 数据集配置

training:
  # 训练配置

# 模式特定配置
server:       # 中心化模式
clients:      # 中心化模式
p2p:          # 去中心化模式
peers:        # 去中心化模式

aggregator:
  # 聚合算法配置

logging:
  # 日志配置

output:
  # 输出配置
```

## 通用配置

### mode

- **类型**: `string`
- **可选值**: `"centralized"`, `"decentralized"`
- **说明**: 选择运行模式

### model

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"simple_cnn"` | 模型名称 |
| `input_channels` | int | 1 | 输入通道数（MNIST=1, CIFAR-10=3） |
| `num_classes` | int | 10 | 分类数 |

### dataset

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"mnist"` | 数据集名称（`"mnist"` 或 `"cifar10"`） |
| `data_dir` | string | `"./data"` | 数据存储目录 |
| `num_clients` | int | 3 | 客户端/节点数量 |
| `partition_strategy` | string | `"iid"` | 数据划分策略（`"iid"` 或 `"non_iid"`） |
| `alpha` | float | 0.5 | Dirichlet 分布参数（Non-IID 时使用） |

**数据划分策略说明：**

- **IID**: 数据随机均匀分配给各客户端，每个客户端的数据分布与整体相同
- **Non-IID**: 使用 Dirichlet 分布模拟真实场景，alpha 越小数据分布越不均衡
  - `alpha = 0.1`: 高度非均衡，每个客户端只有少量类别
  - `alpha = 1.0`: 中等非均衡
  - `alpha = 10.0`: 接近 IID

### training

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rounds` | int | 10 | 联邦学习轮数 |
| `epochs_per_round` | int | 2 | 每轮本地训练的 epoch 数 |
| `learning_rate` | float | 0.01 | 学习率 |
| `batch_size` | int | 32 | 批量大小 |
| `momentum` | float | 0.9 | SGD 动量 |
| `weight_decay` | float | 0.0001 | L2 正则化系数 |

#### early_stopping（早停配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `false` | 是否启用早停 |
| `patience` | int | 5 | 容忍多少轮没有改善 |
| `min_delta` | float | 0.001 | 改善的最小阈值 |
| `monitor` | string | `"accuracy"` | 监控指标（`"accuracy"` 或 `"loss"`） |
| `mode` | string | `"max"` | 优化方向（`"max"` 或 `"min"`） |

**早停配置示例：**

```yaml
training:
  early_stopping:
    enabled: true
    patience: 3
    min_delta: 0.005
    monitor: "accuracy"
    mode: "max"
```

**说明：**
- 当 `monitor: "accuracy"` 时，`mode` 应为 `"max"`（越高越好）
- 当 `monitor: "loss"` 时，`mode` 应为 `"min"`（越低越好）
- 如果连续 `patience` 轮没有改善超过 `min_delta`，训练将提前停止

### aggregator

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"fedavg"` | 聚合算法（`"fedavg"` 或 `"fedprox"`） |
| `fedprox_mu` | float | 0.01 | FedProx 的 mu 参数（仅 FedProx 使用） |

**聚合算法说明：**

- **FedAvg**: 标准的联邦平均算法，对各客户端更新进行加权平均
- **FedProx**: 在 FedAvg 基础上添加近端项，限制本地模型与全局模型的偏差，更适合 Non-IID 场景

### logging

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | 日志级别（`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`） |
| `log_dir` | string | `"./logs"` | 日志文件目录 |
| `console_output` | bool | `true` | 是否输出到控制台 |
| `file_output` | bool | `true` | 是否输出到文件 |

### output

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_dir` | string | `"./outputs/checkpoints"` | 模型检查点保存目录 |
| `figure_dir` | string | `"./outputs/figures"` | 图表保存目录 |
| `save_checkpoint_every` | int | 5 | 每隔多少轮保存检查点 |

## 中心化模式配置

### server

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | string | `"0.0.0.0"` | 服务器监听地址 |
| `port` | int | 9000 | 服务器监听端口 |
| `address` | string | `"127.0.0.1"` | 服务器对外地址 |

#### timeouts

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `connect` | float | 10.0 | 连接超时（秒） |
| `send` | float | 30.0 | 发送超时（秒） |
| `recv` | float | 30.0 | 接收超时（秒） |
| `round` | float | 60.0 | 每轮训练超时（秒） |

### clients

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `server_address` | string | `null` | 服务器地址（可选） |
| `nodes` | list | `[]` | 客户端节点列表 |

#### nodes 列表项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | int | - | 客户端 ID |
| `host` | string | `"127.0.0.1"` | 客户端主机地址 |

## 去中心化模式配置

### p2p

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `topology` | string | `"ring"` | P2P 拓扑结构 |
| `heartbeat_interval` | float | 5.0 | 心跳间隔（秒） |
| `heartbeat_timeout` | float | 10.0 | 心跳超时（秒） |
| `retry_count` | int | 3 | 重试次数 |
| `retry_delay` | float | 1.0 | 重试延迟（秒） |

### peers

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `local` | bool | `true` | 是否本地运行所有节点 |
| `nodes` | list | `[]` | 对等节点列表 |

#### nodes 列表项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | int | - | 节点 ID |
| `host` | string | `"127.0.0.1"` | 节点主机地址 |
| `port` | int | - | 节点监听端口 |

## 完整配置示例

### 中心化模式 - IID 数据

```yaml
mode: centralized

model:
  name: "simple_cnn"
  input_channels: 1
  num_classes: 10

dataset:
  name: "mnist"
  data_dir: "./data"
  num_clients: 5
  partition_strategy: "iid"
  alpha: 0.5

training:
  rounds: 20
  epochs_per_round: 2
  learning_rate: 0.01
  batch_size: 32
  momentum: 0.9
  weight_decay: 0.0001
  early_stopping:
    enabled: true
    patience: 5
    min_delta: 0.001
    monitor: "accuracy"
    mode: "max"

server:
  host: "0.0.0.0"
  port: 9000
  address: "127.0.0.1"
  timeouts:
    connect: 10.0
    send: 30.0
    recv: 120.0
    round: 180.0

clients:
  nodes:
    - { id: 1, host: "127.0.0.1" }
    - { id: 2, host: "127.0.0.1" }
    - { id: 3, host: "127.0.0.1" }
    - { id: 4, host: "127.0.0.1" }
    - { id: 5, host: "127.0.0.1" }

aggregator:
  name: "fedavg"

logging:
  level: "INFO"
  log_dir: "./logs"
  console_output: true
  file_output: true

output:
  checkpoint_dir: "./outputs/checkpoints"
  figure_dir: "./outputs/figures"
  save_checkpoint_every: 5
```

### 去中心化模式 - Non-IID 数据

```yaml
mode: decentralized

model:
  name: "simple_cnn"
  input_channels: 1
  num_classes: 10

dataset:
  name: "mnist"
  data_dir: "./data"
  num_clients: 3
  partition_strategy: "non_iid"
  alpha: 0.3

training:
  rounds: 15
  epochs_per_round: 3
  learning_rate: 0.01
  batch_size: 32
  momentum: 0.9
  weight_decay: 0.0001
  early_stopping:
    enabled: true
    patience: 4
    min_delta: 0.002
    monitor: "loss"
    mode: "min"

p2p:
  topology: "ring"
  heartbeat_interval: 5.0
  heartbeat_timeout: 10.0
  retry_count: 3
  retry_delay: 1.0

peers:
  local: true
  nodes:
    - { id: 1, host: "127.0.0.1", port: 9001 }
    - { id: 2, host: "127.0.0.1", port: 9002 }
    - { id: 3, host: "127.0.0.1", port: 9003 }

aggregator:
  name: "fedprox"
  fedprox_mu: 0.01

logging:
  level: "INFO"
  log_dir: "./logs"
  console_output: true
  file_output: true

output:
  checkpoint_dir: "./outputs/checkpoints"
  figure_dir: "./outputs/figures"
  save_checkpoint_every: 5
```

### CIFAR-10 配置

```yaml
mode: centralized

model:
  name: "simple_cnn"
  input_channels: 3  # CIFAR-10 是 RGB 图像
  num_classes: 10

dataset:
  name: "cifar10"
  data_dir: "./data"
  num_clients: 10
  partition_strategy: "non_iid"
  alpha: 0.5

training:
  rounds: 50
  epochs_per_round: 5
  learning_rate: 0.01
  batch_size: 64
  momentum: 0.9
  weight_decay: 0.0005
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.005
    monitor: "accuracy"
    mode: "max"

# ... 其他配置
```

## 配置验证

系统会在启动时验证配置：

1. **模式一致性**: 检查 `mode` 与相关配置是否匹配
2. **客户端数量**: 检查 `dataset.num_clients` 与节点列表数量是否一致
3. **必需字段**: 检查模式特定的必需配置是否存在

配置错误时会抛出 `ValueError` 并提示具体问题。
