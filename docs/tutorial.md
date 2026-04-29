# 使用教程

本文档提供联邦学习系统的详细使用教程。

## 目录

1. [快速开始](#快速开始)
2. [中心化模式教程](#中心化模式教程)
3. [去中心化模式教程](#去中心化模式教程)
4. [高级功能](#高级功能)
5. [故障排除](#故障排除)

## 快速开始

### 1. 环境准备

确保已安装 Python >= 3.13 和 uv：

```bash
# 检查 Python 版本
python --version  # 应显示 3.13 或更高

# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装项目

```bash
# 克隆仓库
git clone <repository-url>
cd fed

# 使用 uv 安装依赖
uv sync

# 验证安装
uv run python -c "import src; print('安装成功')"
```

### 3. 第一次运行

```bash
# 运行中心化模式（默认配置）
uv run ./scripts/run_fed.py --config ./config/centralized.yaml

# 或运行去中心化模式
uv run ./scripts/run_fed.py --config ./config/decentralized.yaml
```

## 中心化模式教程

### 基本流程

中心化模式采用传统的服务器-客户端架构：

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Client 1│     │ Client 2│     │ Client 3│
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
                     ▼
              ┌─────────────┐
              │   Server    │
              │ (Aggregator)│
              └─────────────┘
```

### 单节点运行（测试）

```bash
# 使用默认配置运行
uv run ./scripts/run_fed.py --config ./config/centralized.yaml
```

这将启动：
- 1 个服务器（端口 9000）
- 3 个客户端

### 自定义配置

创建自定义配置文件 `my_config.yaml`：

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

server:
  host: "0.0.0.0"
  port: 9000
  timeouts:
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
  console_output: true
```

运行：

```bash
uv run ./scripts/run_fed.py --config ./my_config.yaml
```

### 分布式运行（多机器）

1. **服务器机器**：

```yaml
# server_config.yaml
mode: centralized

server:
  host: "0.0.0.0"
  port: 9000
  address: "192.168.1.100"  # 服务器实际 IP

# ... 其他配置
```

```bash
uv run ./scripts/run_server.py --config ./server_config.yaml
```

2. **客户端机器**：

```yaml
# client_config.yaml
mode: centralized

server:
  address: "192.168.1.100"  # 服务器 IP
  port: 9000

# ... 其他配置
```

```bash
uv run ./scripts/run_client.py --config ./client_config.yaml --client-id 1
```

## 去中心化模式教程

### 基本流程

去中心化模式采用 P2P 环形拓扑：

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Node 1  │────►│ Node 2  │────►│ Node 3  │
│ (Bootstrap)   │         │     │         │
└─────────┘◄────└─────────┘◄────└─────────┘
```

### 单节点运行

```bash
uv run ./scripts/run_fed.py --config ./config/decentralized.yaml
```

这将启动：
- 3 个 P2P 节点（端口 9001, 9002, 9003）
- Node 1 作为引导节点

### 手动启动节点

如果需要分别启动每个节点（例如分布式部署）：

1. **启动引导节点**（Node 1）：

```bash
uv run ./scripts/run_p2p_node.py \
  --config ./config/decentralized.yaml \
  --node-id 1 \
  --port 9001
```

2. **启动其他节点**：

```bash
# Node 2
uv run ./scripts/run_p2p_node.py \
  --config ./config/decentralized.yaml \
  --node-id 2 \
  --port 9002 \
  --bootstrap 127.0.0.1:9001

# Node 3
uv run ./scripts/run_p2p_node.py \
  --config ./config/decentralized.yaml \
  --node-id 3 \
  --port 9003 \
  --bootstrap 127.0.0.1:9001
```

### 分布式部署

在多机器上部署：

**机器 1**（Node 1，引导节点）：

```bash
uv run ./scripts/run_p2p_node.py \
  --config ./config/decentralized.yaml \
  --node-id 1 \
  --port 9001
```

**机器 2**（Node 2）：

```bash
uv run ./scripts/run_p2p_node.py \
  --config ./config/decentralized.yaml \
  --node-id 2 \
  --port 9002 \
  --bootstrap 192.168.1.101:9001  # 机器 1 的 IP
```

**机器 3**（Node 3）：

```bash
uv run ./scripts/run_p2p_node.py \
  --config ./config/decentralized.yaml \
  --node-id 3 \
  --port 9003 \
  --bootstrap 192.168.1.101:9001  # 机器 1 的 IP
```

## 高级功能

### 1. 早停机制

早停可以在模型性能不再提升时自动停止训练，节省时间和计算资源。

**配置示例**：

```yaml
training:
  rounds: 100  # 设置较大的最大轮数
  early_stopping:
    enabled: true
    patience: 5        # 容忍 5 轮没有改善
    min_delta: 0.005   # 改善阈值 0.5%
    monitor: "accuracy" # 监控准确率
    mode: "max"        # 准确率越高越好
```

**使用场景**：

- **监控准确率**：适合分类任务，当准确率不再提升时停止
- **监控损失**：适合回归任务，当损失不再下降时停止

### 2. Non-IID 数据划分

模拟真实场景中数据分布不均衡的情况：

```yaml
dataset:
  name: "mnist"
  num_clients: 10
  partition_strategy: "non_iid"
  alpha: 0.3  # 越小越不均衡
```

**alpha 参数选择**：

| alpha 值 | 数据分布 | 适用场景 |
|---------|---------|---------|
| 0.1 | 高度非均衡 | 极端异构场景 |
| 0.5 | 中等非均衡 | 一般异构场景 |
| 1.0 | 轻度非均衡 | 轻微异构场景 |
| 10.0 | 接近 IID | 对比实验 |

**配合 FedProx 使用**：

```yaml
aggregator:
  name: "fedprox"
  fedprox_mu: 0.01  # 控制本地模型与全局模型的偏差
```

### 3. 测试集评估

系统会自动在测试集上评估模型性能：

- **中心化模式**：服务器在聚合后评估全局模型
- **去中心化模式**：每个节点在本地训练后评估自己的模型

评估结果会显示在日志中：

```
[Server] Round 5/10 - Accuracy: 92.34%, Loss: 0.2456
```

### 4. 模型检查点

自动保存模型检查点：

```yaml
output:
  checkpoint_dir: "./outputs/checkpoints"
  save_checkpoint_every: 5  # 每 5 轮保存一次
```

检查点文件：`checkpoint_round_{round_id}.pt`

## 故障排除

### 1. 连接超时

**问题**：客户端无法连接到服务器

**解决方案**：

```bash
# 检查服务器是否启动
netstat -tlnp | grep 9000

# 检查防火墙设置
sudo ufw allow 9000/tcp

# 增加超时时间
server:
  timeouts:
    connect: 30.0
    recv: 300.0
```

### 2. 内存不足

**问题**：训练过程中内存溢出

**解决方案**：

```yaml
training:
  batch_size: 16  # 减小批量大小
  epochs_per_round: 1  # 减少每轮 epoch 数
```

### 3. 数据下载失败

**问题**：无法下载 MNIST/CIFAR-10 数据集

**解决方案**：

```bash
# 手动下载数据集
uv run ./scripts/download_datasets.py --dataset mnist --data-dir ./data

# 或使用镜像
export HF_ENDPOINT=https://hf-mirror.com
uv run ./scripts/run_fed.py --config ./config/centralized.yaml
```

### 4. 端口被占用

**问题**：启动时提示端口已被占用

**解决方案**：

```bash
# 查找占用端口的进程
lsof -i :9000

# 终止进程
kill -9 <PID>

# 或修改配置文件使用其他端口
server:
  port: 9001  # 使用其他端口
```

### 5. 早停不生效

**问题**：配置了早停但训练没有提前停止

**检查**：

```yaml
training:
  early_stopping:
    enabled: true  # 确保启用
    patience: 5    # 确保 patience > 0
    min_delta: 0.001  # 确保 min_delta 不是太大
```

### 6. 去中心化模式节点无法加入

**问题**：节点加入环形网络时超时

**解决方案**：

```bash
# 确保引导节点已启动并监听
netstat -tlnp | grep 9001

# 检查 bootstrap 地址是否正确
uv run ./scripts/run_p2p_node.py \
  --bootstrap 127.0.0.1:9001  # 确保 IP 和端口正确

# 增加超时时间（修改源码中的 timeout 参数）
```

## 性能优化建议

### 1. 网络优化

```yaml
server:
  timeouts:
    recv: 120.0  # 根据网络延迟调整
    send: 60.0
```

### 2. 计算优化

```yaml
training:
  batch_size: 64  # 根据 GPU 内存调整
  epochs_per_round: 1  # 减少通信开销
```

### 3. 数据预加载

首次运行后会自动缓存划分好的数据：

```
data/
└── partitioned/
    ├── client_1.pt
    ├── client_2.pt
    ├── client_3.pt
    └── test.pt
```

后续运行会直接加载缓存，无需重新划分。

## 示例场景

### 场景 1：快速测试

```yaml
# 快速测试配置
training:
  rounds: 3
  epochs_per_round: 1
  batch_size: 64
```

### 场景 2：高精度训练

```yaml
# 高精度训练配置
training:
  rounds: 100
  epochs_per_round: 5
  learning_rate: 0.01
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
```

### 场景 3：异构数据场景

```yaml
# 异构数据场景
dataset:
  partition_strategy: "non_iid"
  alpha: 0.3

aggregator:
  name: "fedprox"
  fedprox_mu: 0.01
```
