# 文件结构

> 本文档定义项目的完整目录结构和各文件职责。

---

## 完整目录结构

```
fed/
├── config/                           # 配置文件目录
│   ├── default.yaml                  # 默认配置
│   ├── centralized.yaml              # 中心化模式配置
│   └── decentralized.yaml            # 去中心化模式配置
│
├── src/                              # 源代码目录
│   ├── __init__.py
│   │
│   ├── core/                         # 核心抽象层
│   │   ├── __init__.py
│   │   ├── interfaces.py             # 接口定义（共同维护）
│   │   ├── types.py                  # 类型定义
│   │   └── exceptions.py             # 自定义异常
│   │
│   ├── model/                        # 模型模块【成员A】
│   │   ├── __init__.py
│   │   ├── base.py                   # 模型基类
│   │   ├── simple_cnn.py             # SimpleCNN实现
│   │   └── registry.py               # 模型注册器
│   │
│   ├── data/                         # 数据模块【成员A】
│   │   ├── __init__.py
│   │   ├── dataset.py                # 数据集加载
│   │   ├── partitioner.py            # 数据划分策略
│   │   └── loader.py                 # DataLoader封装
│   │
│   ├── client/                       # 客户端模块【成员A】
│   │   ├── __init__.py
│   │   ├── base.py                   # 客户端基类
│   │   ├── federated_client.py       # 联邦客户端
│   │   ├── trainer.py                # 本地训练器
│   │   └── evaluator.py              # 模型评估器
│   │
│   ├── server/                       # 服务端模块【成员B】
│   │   ├── __init__.py
│   │   ├── base.py                   # 服务端基类
│   │   ├── federated_server.py       # 联邦服务端
│   │   ├── aggregator.py             # 聚合策略（FedAvg等）
│   │   ├── client_manager.py         # 客户端连接管理
│   │   ├── round_coordinator.py      # 轮次协调器
│   │   └── checkpoint.py             # 模型检查点
│   │
│   ├── p2p/                          # 去中心化模块【成员B】
│   │   ├── __init__.py
│   │   ├── node.py                   # P2P节点基类
│   │   ├── ring_node.py              # 环形节点实现
│   │   ├── topology.py               # 拓扑管理
│   │   ├── failure_detector.py       # 故障检测器
│   │   └── recovery.py               # 故障恢复
│   │
│   ├── protocol/                     # 协议模块【成员C】
│   │   ├── __init__.py
│   │   ├── message.py                # 消息定义
│   │   ├── codec.py                  # 编解码器
│   │   ├── serializer.py             # 序列化
│   │   └── constants.py              # 协议常量
│   │
│   ├── transport/                    # 传输模块【成员C】
│   │   ├── __init__.py
│   │   ├── connection.py             # 连接管理
│   │   ├── listener.py               # 监听器
│   │   └── timeout.py                # 超时控制
│   │
│   └── utils/                        # 工具模块【成员D】
│       ├── __init__.py
│       ├── logger.py                 # 日志管理
│       ├── metrics.py                # 指标统计
│       ├── timer.py                  # 计时器
│       ├── config_loader.py          # 配置加载
│       └── visualizer.py             # 可视化工具
│
├── scripts/                          # 脚本目录
│   ├── download_cifar10.py           # 下载CIFAR-10
│   ├── split_dataset.py              # 划分数据集
│   ├── run_server.py                 # 启动服务端
│   ├── run_client.py                 # 启动客户端
│   └── run_p2p_node.py               # 启动P2P节点
│
├── tests/                            # 测试目录【成员D】
│   ├── __init__.py
│   ├── conftest.py                   # pytest配置
│   ├── unit/                         # 单元测试
│   │   ├── __init__.py
│   │   ├── test_model.py
│   │   ├── test_protocol.py
│   │   ├── test_transport.py
│   │   └── test_aggregator.py
│   └── integration/                  # 集成测试
│       ├── __init__.py
│       ├── test_centralized.py
│       └── test_decentralized.py
│
├── docs/                             # 文档目录【成员D】
│   ├── ARCHITECTURE.md               # 架构文档
│   ├── API.md                        # API文档
│   ├── PROTOCOL.md                   # 协议文档
│   ├── DEPLOYMENT.md                 # 部署文档
│   ├── TEST_REPORT.md                # 测试报告
│   └── USER_MANUAL.md                # 用户手册
│
├── data/                             # 数据目录（.gitignore）
│   ├── raw/                          # 原始数据
│   └── partitioned/                  # 划分后数据
│       ├── client_1.pt
│       ├── client_2.pt
│       └── client_3.pt
│
├── logs/                             # 日志目录（.gitignore）
│   └── *.log
│
├── outputs/                          # 输出目录（.gitignore）
│   ├── checkpoints/                  # 模型检查点
│   └── figures/                      # 图表
│
├── pyproject.toml                    # 项目配置
├── .python-version                   # Python版本
├── .gitignore                        # Git忽略配置
├── .pre-commit-config.yaml           # Pre-commit配置
├── CONTRIBUTING.md                   # 协作规范
└── README.md                         # 项目说明
```

---

## 文件职责详解

### 成员A：模型、数据、客户端

| 文件 | 职责 | 依赖 |
|------|------|------|
| `model/base.py` | 定义模型抽象基类 | core/interfaces.py |
| `model/simple_cnn.py` | SimpleCNN模型实现 | model/base.py |
| `model/registry.py` | 模型注册器，支持通过配置加载模型 | - |
| `data/dataset.py` | CIFAR-10数据集加载 | torchvision |
| `data/partitioner.py` | IID/Non-IID数据划分策略 | torch |
| `data/loader.py` | DataLoader封装 | torch.utils.data |
| `client/base.py` | 客户端抽象基类 | core/interfaces.py |
| `client/federated_client.py` | 联邦客户端实现 | client/base.py, protocol/*, transport/* |
| `client/trainer.py` | 本地训练逻辑 | model/*, data/* |
| `client/evaluator.py` | 模型评估 | model/* |

### 成员B：服务端开发

| 文件 | 职责 | 依赖 |
|------|------|------|
| `server/base.py` | 服务端抽象基类 | core/interfaces.py |
| `server/federated_server.py` | 联邦服务端实现 | server/base.py, protocol/*, transport/* |
| `server/aggregator.py` | FedAvg等聚合算法 | torch |
| `server/client_manager.py` | 客户端连接管理、多线程处理 | transport/* |
| `server/round_coordinator.py` | 轮次协调、超时控制 | server/client_manager.py, server/aggregator.py |
| `server/checkpoint.py` | 模型保存/加载 | torch |
| `p2p/node.py` | P2P节点抽象基类 | core/interfaces.py |
| `p2p/ring_node.py` | 环形节点实现 | p2p/node.py, p2p/topology.py |
| `p2p/topology.py` | 环形拓扑管理 | - |
| `p2p/failure_detector.py` | 心跳检测、故障发现 | protocol/* |
| `p2p/recovery.py` | 故障恢复、拓扑重构 | p2p/topology.py |

### 成员C：网络协议设计

| 文件 | 职责 | 依赖 |
|------|------|------|
| `protocol/constants.py` | 消息类型、错误码常量 | - |
| `protocol/message.py` | 消息数据结构定义 | core/types.py |
| `protocol/codec.py` | 编解码、粘包处理 | protocol/message.py, protocol/serializer.py |
| `protocol/serializer.py` | 模型权重序列化/反序列化 | torch, pickle |
| `transport/connection.py` | TCP连接封装、发送/接收 | socket, protocol/codec.py |
| `transport/listener.py` | TCP监听器、接受连接 | socket |
| `transport/timeout.py` | 超时控制工具 | - |

### 成员D：测试与日志

| 文件 | 职责 | 依赖 |
|------|------|------|
| `utils/logger.py` | 结构化日志输出 | logging |
| `utils/metrics.py` | 指标收集、统计 | - |
| `utils/timer.py` | 耗时统计工具 | time |
| `utils/config_loader.py` | YAML配置加载 | pyyaml |
| `utils/visualizer.py` | 准确率曲线绘制 | matplotlib |
| `tests/unit/*.py` | 单元测试 | pytest |
| `tests/integration/*.py` | 集成测试 | pytest |
| `docs/*.md` | 各类文档 | - |

---

## 共享文件

以下文件需要早期定稿，各成员只读引用：

| 文件 | 维护者 | 说明 |
|------|--------|------|
| `core/interfaces.py` | 共同维护 | 所有接口定义，开发前定稿 |
| `core/types.py` | 共同维护 | 核心类型定义 |
| `core/exceptions.py` | 共同维护 | 自定义异常 |
| `config/*.yaml` | 成员A | 配置文件，其他人只读取 |

---

## 模块依赖图

```
┌─────────────────────────────────────────────────────────────┐
│                        Application                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Client    │  │   Server    │  │     P2P Ring Node   │ │
│  │  (成员A)    │  │  (成员B)    │  │       (成员B)       │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────────┼────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                         Core Layer                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  interfaces.py | types.py | exceptions.py              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│   Model/Data    │ │    Protocol     │ │       Utils         │
│    (成员A)      │ │    (成员C)      │ │      (成员D)        │
└─────────────────┘ └─────────────────┘ └─────────────────────┘
                           │
                           ▼
                    ┌─────────────────┐
                    │   Transport     │
                    │    (成员C)      │
                    └─────────────────┘
```

---

## Git协作策略

### 分支命名

```
main                    # 主分支
├── feature/model       # 成员A: 模型开发
├── feature/data        # 成员A: 数据处理
├── feature/client      # 成员A: 客户端开发
├── feature/server      # 成员B: 服务端开发
├── feature/p2p         # 成员B: P2P开发
├── feature/protocol    # 成员C: 协议开发
├── feature/transport   # 成员C: 传输层开发
└── feature/test-logs   # 成员D: 测试与日志
```

### 合并顺序

1. **Phase 1**: `core/interfaces.py` 定稿，合并到 main
2. **Phase 2**: 成员C完成 `protocol/` 和 `transport/`，合并到 main
3. **Phase 3**: 成员A完成 `model/`、`data/`，合并到 main
4. **Phase 4**: 成员A完成 `client/`，成员B完成 `server/`，并行合并
5. **Phase 5**: 成员B完成 `p2p/`，合并到 main
6. **Phase 6**: 成员D完成测试，合并到 main
