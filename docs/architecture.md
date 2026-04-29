# 架构设计

本文档描述联邦学习系统的整体架构和模块设计。

## 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Federated Learning System                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────┐           │
│  │   Centralized Mode  │      │  Decentralized Mode │           │
│  │                     │      │                     │           │
│  │  ┌───────────────┐  │      │  ┌───────────────┐  │           │
│  │  │    Server     │  │      │  │   Node 1      │──┼──┐        │
│  │  │  (Aggregator) │  │      │  │  (Ring Node)  │  │  │        │
│  │  └───────┬───────┘  │      │  └───────┬───────┘  │  │        │
│  │          │          │      │          │          │  │        │
│  │    ┌─────┴─────┐    │      │    ┌─────┴─────┐    │  │        │
│  │    │           │    │      │    │           │    │  │        │
│  │ ┌──┴──┐     ┌──┴──┐ │      │ ┌──┴──┐     ┌──┴──┐ │  │        │
│  │ │Client│     │Client│ │      │ │Node2│     │Node3│ │  │        │
│  │ │  1   │     │  2   │ │      │ │     │◄────┘     │ │  │        │
│  │ └──┬──┘     └──┬──┘ │      │ └──┬──┘           │ │  │        │
│  │    │           │    │      │    │              │ │  │        │
│  │    └─────┬─────┘    │      │    └──────────────┘ │  │        │
│  │          │          │      │                     │  │        │
│  └──────────┼──────────┘      └─────────────────────┘  │        │
│             │                                          │        │
│             └──────────────────────────────────────────┘        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Shared Components                           │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │  Model  │  │  Data   │  │Transport│  │ Protocol│    │    │
│  │  │  Layer  │  │  Layer  │  │  Layer  │  │  Layer  │    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 模块设计

### 1. Core 模块 (`src/core/`)

核心类型定义和接口。

```
core/
├── types.py        # 数据类：Config, TrainingResult, RoundStats 等
├── interfaces.py   # 抽象接口：IModel, IClient, IServer, IP2PNode 等
└── exceptions.py   # 自定义异常
```

### 2. Server 模块 (`src/server/`) - 中心化模式

```
server/
├── federated_server.py   # 联邦学习服务器主类
├── base.py              # 服务器基类
├── aggregator.py        # 聚合算法（FedAvg, FedProx）
├── client_manager.py    # 客户端连接管理
└── round_coordinator.py # 训练轮次协调
```

**工作流程：**

1. 服务器启动并等待客户端连接
2. 每轮训练：
   - 广播全局模型给所有客户端
   - 收集客户端更新
   - 聚合更新生成新的全局模型
   - 在测试集上评估模型

### 3. Client 模块 (`src/client/`) - 中心化模式

```
client/
├── federated_client.py  # 联邦学习客户端主类
├── base.py             # 客户端基类
├── trainer.py          # 本地训练器
└── evaluator.py        # 本地评估器
```

**工作流程：**

1. 连接到服务器
2. 接收全局模型
3. 本地训练
4. 发送模型更新

### 4. P2P 模块 (`src/p2p/`) - 去中心化模式

```
p2p/
├── ring_node.py         # 环形拓扑节点
├── topology.py          # 拓扑管理
├── failure_detector.py  # 故障检测
└── recovery.py          # 故障恢复
```

**工作流程：**

1. 节点启动并加入环形网络
2. 每轮训练：
   - 本地训练
   - 将模型传递给环中的下一个节点
   - 在测试集上评估模型

### 5. Model 模块 (`src/model/`)

```
model/
├── simple_cnn.py   # SimpleCNN 模型
├── base.py        # 模型基类
└── registry.py    # 模型注册表
```

支持的模型：
- `simple_cnn`: 简单的卷积神经网络，适用于 MNIST/CIFAR-10

### 6. Data 模块 (`src/data/`)

```
data/
├── dataset.py     # 数据集加载
├── loader.py      # DataLoader 创建
└── partitioner.py # 数据划分策略
```

数据划分策略：
- **IID**: 独立同分布，随机均匀分配
- **Non-IID**: 使用 Dirichlet 分布模拟真实场景

### 7. Transport 模块 (`src/transport/`)

```
transport/
├── connection.py  # TCP 连接封装
├── listener.py    # TCP 监听器
└── timeout.py     # 超时控制
```

特性：
- 长度前缀消息分帧
- TCP 优化（NODELAY, 缓冲区大小）
- 零拷贝接收

### 8. Protocol 模块 (`src/protocol/`)

```
protocol/
├── message.py     # 消息结构
├── codec.py       # 编码/解码
├── serializer.py  # 模型序列化
└── constants.py   # 协议常量
```

消息类型：
- 中心化：`MODEL_BROADCAST`, `MODEL_UPDATE`, `CLIENT_REGISTER`
- 去中心化：`NODE_JOIN`, `NODE_LEAVE`, `RING_PASS`

### 9. Utils 模块 (`src/utils/`)

```
utils/
├── early_stopping.py  # 早停机制
├── logger.py         # 日志记录
├── metrics.py        # 指标收集
├── timer.py          # 计时器
└── visualizer.py     # 可视化
```

## 通信协议

### 消息格式

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Length (8B)    │   Type (1B)     │   Payload (N)   │
│   (uint64)      │   (uint8)       │   (protobuf)    │
└─────────────────┴─────────────────┴─────────────────┘
```

### 中心化模式通信流程

```
Client                      Server
  │                           │
  │──── CLIENT_REGISTER ─────►│
  │◄──────── CLIENT_ACK ─────│
  │                           │
  │◄──── MODEL_BROADCAST ────│
  │                           │
  │────── LOCAL_TRAINING ────┤
  │                           │
  │───── MODEL_UPDATE ──────►│
  │         (repeat)          │
```

### 去中心化模式通信流程

```
Node 1          Node 2          Node 3
  │               │               │
  │◄── NODE_JOIN ─┤               │
  │─── NODE_ACK ─►│               │
  │               │               │
  │◄── RING_PASS ─┤               │
  │─── RING_PASS ─┼──────────────►│
  │               │◄── RING_PASS ─┤
  │               │               │
  │    (circular) │    (circular) │
```

## 数据流

### 中心化模式数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │     │   Network   │     │   Server    │
│   Data      │────►│   Training  │────►│  Aggregate  │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Test Data  │
                                        │  Evaluate   │
                                        └─────────────┘
```

### 去中心化模式数据流

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Node 1  │────►│ Node 2  │────►│ Node 3  │
│ Train   │     │ Train   │     │ Train   │
│ Pass    │     │ Pass    │     │ Pass    │
└────┬────┘     └────┬────┘     └────┬────┘
     │               │               │
     │               │               │
     └───────────────┴───────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  Test Data  │
              │  Evaluate   │
              └─────────────┘
```

## 扩展性设计

### 添加新的聚合算法

1. 实现 `IAggregator` 接口
2. 在 `create_aggregator()` 中注册

### 添加新的数据集

1. 在 `dataset.py` 中添加加载逻辑
2. 实现对应的 transforms
3. 更新 `get_input_channels()` 和 `get_num_classes()`

### 添加新的模型

1. 实现 `IModel` 接口
2. 在 `registry.py` 中注册

### 添加新的 P2P 拓扑

1. 实现 `ITopologyManager` 接口
2. 实现对应的节点类

## 性能优化

1. **TCP 优化**
   - TCP_NODELAY 禁用 Nagle 算法
   - 增大缓冲区大小

2. **序列化优化**
   - 使用 PyTorch 原生序列化
   - 支持 GPU 张量

3. **零拷贝接收**
   - 预分配缓冲区
   - 使用 memoryview

4. **并发处理**
   - 每个连接独立线程
   - 异步消息处理
