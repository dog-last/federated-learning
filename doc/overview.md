# 项目总览

## 1. 项目目标

本项目实现一个用于 MNIST 分类的联邦学习实验框架，支持两种训练模式：

- **Centralized（集中式）**：传统联邦学习，客户端本地训练后上传模型参数，服务端聚合
- **SplitFed（分割联邦）**：将模型分割在客户端和服务端，中间传输激活值和梯度

项目同时保留集中式训练路径，便于比较不同训练方式的通信成本、收敛行为与监控表现。

## 2. 项目架构

```
federated-learning/
├── manager.py              # 实验主入口，编排整个训练流程
├── config.json             # 主配置文件
├── pyproject.toml          # uv 项目配置
├── uv.lock                 # 锁定依赖版本
│
├── core/                   # 核心训练逻辑
│   ├── server.py           # 联邦学习服务端
│   ├── client.py           # 联邦学习客户端
│   └── communicator.py     # TCP 通信协议
│
├── utils/                  # 工具模块
│   ├── monitor_api.py      # FastAPI 监控服务
│   ├── monitoring.py       # 监控事件上报
│   └── training_controller.py  # 训练进程控制
│
├── scripts/                # 数据准备脚本
│   ├── prepare_mnist.py    # MNIST 数据下载与拆分
│   └── split.py            # 数据拆分工具
│
├── tests/                  # 测试套件
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
│
├── model.py                # 模型定义
├── data/splits/            # 数据文件（自动生成）
└── logs/                   # 训练日志（自动生成）
```

## 3. 运行主链路

一次完整实验的主链路如下：

```
┌─────────────────────────────────────────────────────────────────┐
│                        manager.py                               │
│  1. 读取 config.json                                            │
│  2. 检查数据文件，必要时调用 prepare_mnist.py                    │
│  3. 启动 monitor_api.py（监控服务）                              │
│  4. 启动 core.server（服务端进程）                               │
│  5. 启动多个 core.client（客户端进程）                           │
│  6. 等待训练完成，回收进程                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  monitor_api  │    │    server     │    │    clients    │
│  (port 9000)  │    │  (port 8000)  │    │ (port 8001+)  │
└───────────────┘    └───────────────┘    └───────────────┘
        ▲                     ▲                     │
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  communicator   │
                    │  (TCP 协议)     │
                    └─────────────────┘
```

### 详细流程

1. **配置加载**：`manager.py` 读取 `config.json`，获取训练参数、网络拓扑、监控配置
2. **数据准备**：检查 `data/splits/` 下的数据文件，缺失时自动运行 `scripts/prepare_mnist.py`
3. **监控启动**：启动 `utils/monitor_api.py` 提供 HTTP API（默认 http://127.0.0.1:9000）
4. **服务端启动**：启动 `core.server` 进程，监听配置指定的端口
5. **客户端启动**：为每个配置的客户端启动 `core.client` 进程
6. **训练执行**：服务端与客户端通过 `core.communicator.TCPCommunicator` 进行 TCP 通信
7. **事件上报**：`utils.monitoring.MonitorReporter` 将训练事件上报到监控服务
8. **进程回收**：训练完成后，`manager.py` 优雅地停止所有进程

## 4. 核心模块职责

### core/server.py

服务端负责：
- 读取配置并选择计算设备（CPU/GPU）
- 加载服务端测试集
- 接收客户端连接，协调训练轮次
- 聚合客户端模型更新（FedAvg）或协调 SplitFed 流程
- 记录网络统计并上报监控事件

### core/client.py

客户端负责：
- 加载对应客户端的本地拆分数据
- 构建训练、验证和测试 DataLoader
- 运行本地训练（Centralized 模式）
- 或运行 SplitFed 的前向与反向流程
- 上报训练指标和网络事件

### core/communicator.py

TCP 通信协议实现：
- 固定前缀协议：8字节长度 + 4字节魔数（`SF26`）+ pickle 序列化负载
- 支持 gzip 压缩（可配置）
- 处理连接管理、超时、重试

### utils/monitor_api.py

监控服务：
- FastAPI 提供 HTTP API
- 实时终端渲染（Rich）
- 训练事件收集与查询
- 训练控制（启动/停止）

### utils/training_controller.py

训练控制器（API 模式使用）：
- 进程编排与管理
- 训练状态机（stopped/starting/running/stopping）
- 优雅停止控制

## 5. 数据流

### Centralized 模式

```
Client 1:  Local Data → Local Train → Model Update ──┐
                                                    ├──► Server: Aggregate
Client 2:  Local Data → Local Train → Model Update ──┤
                                                    │
Client 3:  Local Data → Local Train → Model Update ──┘
                              ▲
                              │
                         Global Model
```

### SplitFed 模式

```
Client 1:  Local Data → Forward (Client Side) → Activations ──┐
                                                              ├──► Server: Forward/Backward (Server Side)
Client 2:  Local Data → Forward (Client Side) → Activations ──┤         │
                                                              │         ▼
Client 3:  Local Data → Forward (Client Side) → Activations ──┘    Gradients
                                                                         │
                                                                         ▼
                                                              ◄── Backward (Client Side)
```

## 6. 代码分层

| 层级 | 文件/目录 | 职责 |
|------|----------|------|
| 编排层 | `manager.py` | 实验编排，进程启动与回收 |
| 控制层 | `utils/training_controller.py` | 训练状态管理，API 控制 |
| 服务层 | `utils/monitor_api.py` | 监控服务，事件收集 |
| 协议层 | `core/communicator.py` | TCP 通信协议 |
| 节点层 | `core/server.py`, `core/client.py` | 训练节点实现 |
| 模型层 | `model.py` | 模型定义 |
| 数据层 | `scripts/prepare_mnist.py` | 数据准备与拆分 |
| 工具层 | `utils/monitoring.py` | 监控上报封装 |

## 7. 推荐阅读顺序

如果是第一次接手这个项目，建议按下面顺序看代码和文档：

1. **README.md** - 项目概览和快速开始
2. **doc/configuration.md** - 了解配置项
3. **doc/data-preparation.md** - 了解数据格式
4. **doc/core-workflow.md** - 了解通信协议和训练流程
5. **doc/monitoring.md** - 了解监控机制
6. **doc/testing-and-development.md** - 了解测试方法

### 代码阅读路径

```
1. manager.py          # 入口，了解整体流程
   │
   ├──► config.json     # 了解配置结构
   │
   ├──► scripts/prepare_mnist.py  # 了解数据准备
   │
   ├──► core/server.py  # 了解服务端逻辑
   │
   ├──► core/client.py  # 了解客户端逻辑
   │
   ├──► core/communicator.py  # 了解通信协议
   │
   └──► utils/monitor_api.py  # 了解监控服务
```

## 8. 关键技术点

### 8.1 进程模型

- `manager.py` 作为主进程，创建子进程运行服务端和客户端
- 每个客户端独立进程，模拟真实联邦学习环境
- 进程间通过 TCP 通信，而非共享内存

### 8.2 通信协议

- 自定义 TCP 协议，基于 pickle 序列化
- 支持压缩减少传输量
- 支持延迟和丢包模拟（用于测试鲁棒性）

### 8.3 监控机制

- 基于 HTTP API 的事件上报
- 实时终端渲染（Rich Live）
- 支持通过 API 控制训练（启动/停止）

### 8.4 数据拆分

- 非 IID 分布拆分，适合联邦学习实验
- 每个客户端有独立的训练/验证/测试集
- 服务端保留全局测试集用于评估

## 9. 扩展建议

### 新增训练模式

1. 在 `model.py` 添加新模型
2. 在 `core/server.py` 和 `core/client.py` 实现对应逻辑
3. 更新 `config.json` 的 `mode` 选项
4. 添加对应测试

### 新增数据集

1. 在 `scripts/` 添加数据准备脚本
2. 更新数据格式检查逻辑
3. 修改 `core/client.py` 和 `core/server.py` 的数据加载
4. 更新配置文档

### 新增监控指标

1. 在 `utils/monitoring.py` 添加新的事件类型
2. 在训练代码中添加上报点
3. 在 `utils/monitor_api.py` 添加渲染逻辑

## 10. 调试技巧

### 查看日志

```bash
# 实时查看服务端日志
tail -f logs/server-*.log

# 查看客户端日志
tail -f logs/client-*-*.log
```

### 手动运行组件

```bash
# 单独运行数据准备
uv run python scripts/prepare_mnist.py

# 单独运行监控服务
uv run python -m uvicorn utils.monitor_api:app --host 127.0.0.1 --port 9000

# 单独运行服务端
uv run python -m core.server

# 单独运行客户端
uv run python -m core.client client_1
```

### 使用调试模式

```bash
# 禁用实时渲染，查看原始日志
# 修改 config.json: monitoring.render_mode = "plain"

# 使用 pdb 调试
uv run python -m pdb manager.py
```
