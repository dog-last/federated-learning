# Federated Learning System

一个支持中心化和去中心化模式的联邦学习系统，使用 TCP 通信进行节点间数据传输。

## 功能特性

- **双模式支持**
  - **中心化模式 (Centralized)**: 传统的服务器-客户端架构，使用 FedAvg 等聚合算法
  - **去中心化模式 (Decentralized)**: P2P 环形拓扑结构，节点间直接通信

- **数据集支持**
  - MNIST
  - CIFAR-10

- **高级功能**
  - 早停机制 (Early Stopping)
  - 测试集评估
  - 非独立同分布 (Non-IID) 数据划分
  - 模型检查点保存

## 快速开始

### 环境要求

- Python >= 3.13
- uv (推荐) 或 pip

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd fed

# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e ".[dev]"
```

### 运行联邦学习训练

#### 中心化模式

```bash
uv run ./scripts/run_fed.py --config ./config/centralized.yaml
```

#### 去中心化模式

```bash
uv run ./scripts/run_fed.py --config ./config/decentralized.yaml
```

## 项目结构

```
fed/
├── config/                 # 配置文件
│   ├── centralized.yaml   # 中心化模式配置
│   └── decentralized.yaml # 去中心化模式配置
├── docs/                  # 文档
│   ├── architecture.md    # 架构设计
│   ├── configuration.md   # 配置说明
│   ├── tutorial.md        # 使用教程
│   └── testing.md         # 测试说明
├── scripts/               # 运行脚本
│   ├── run_fed.py        # 主启动脚本
│   ├── run_server.py     # 启动服务器（中心化）
│   ├── run_client.py     # 启动客户端（中心化）
│   ├── run_p2p_node.py   # 启动 P2P 节点（去中心化）
│   ├── download_datasets.py
│   └── split_dataset.py
├── src/                   # 源代码
│   ├── core/             # 核心类型和接口
│   ├── server/           # 服务器实现（中心化）
│   ├── client/           # 客户端实现（中心化）
│   ├── p2p/              # P2P 实现（去中心化）
│   ├── model/            # 模型定义
│   ├── data/             # 数据加载和划分
│   ├── transport/        # 网络传输层
│   ├── protocol/         # 通信协议
│   └── utils/            # 工具函数
└── tests/                # 测试
    ├── unit/             # 单元测试
    └── integration/      # 集成测试
```

## 配置说明

### 基础配置

```yaml
mode: centralized  # 或 decentralized

model:
  name: "simple_cnn"
  input_channels: 1
  num_classes: 10

dataset:
  name: "mnist"           # 或 "cifar10"
  data_dir: "./data"
  num_clients: 3
  partition_strategy: "iid"  # 或 "non_iid"
  alpha: 0.5              # Dirichlet 分布参数（Non-IID 时使用）

training:
  rounds: 10              # 训练轮数
  epochs_per_round: 2     # 每轮本地训练 epoch 数
  learning_rate: 0.01
  batch_size: 32
  momentum: 0.9
  weight_decay: 0.0001
  early_stopping:         # 早停配置
    enabled: false
    patience: 5
    min_delta: 0.001
    monitor: "accuracy"   # 或 "loss"
    mode: "max"           # "max" 或 "min"
```

更多配置选项请参考 [docs/configuration.md](docs/configuration.md)。

## 运行测试

### 运行所有测试

```bash
uv run pytest
```

### 运行单元测试

```bash
uv run pytest tests/unit/
```

### 运行集成测试

```bash
uv run pytest tests/integration/
```

### 生成测试覆盖率报告

```bash
uv run pytest --cov=src --cov-report=html
```

## 文档

- [架构设计](docs/architecture.md) - 系统架构和模块设计
- [配置说明](docs/configuration.md) - 详细的配置选项说明
- [使用教程](docs/tutorial.md) - 详细的使用教程和示例
- [测试说明](docs/testing.md) - 测试运行和开发指南

## 开发

### 代码格式化

```bash
uv run ruff format .
```

### 代码检查

```bash
uv run ruff check .
```

### 类型检查

```bash
uv run mypy src/
```

## 许可证

MIT License
