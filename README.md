# 联邦学习实验框架

这是一个面向 MNIST 分类的联邦学习/分布式训练项目，支持**集中式训练（Centralized）**与**SplitFed**两种模式。项目使用 Python 3.13+ 开发，采用 [uv](https://docs.astral.sh/uv/) 进行包管理。

## 项目架构

```
federated-learning/
├── manager.py              # 实验主入口，负责编排整个训练流程
├── config.json             # 主配置文件（训练参数、网络拓扑、监控配置）
├── pyproject.toml          # uv 项目配置与依赖管理
├── uv.lock                 # 锁定依赖版本
│
├── core/                   # 核心训练逻辑
│   ├── server.py           # 联邦学习服务端（聚合、协调）
│   ├── client.py           # 联邦学习客户端（本地训练）
│   └── communicator.py     # TCP 通信协议实现
│
├── utils/                  # 工具模块
│   ├── monitor_api.py      # FastAPI 监控服务与实时渲染
│   ├── monitoring.py       # 监控事件上报封装
│   └── training_controller.py  # 训练进程控制（API 方式启动时使用）
│
├── scripts/                # 数据准备脚本
│   ├── prepare_mnist.py    # MNIST 数据下载与拆分
│   └── split.py            # 数据拆分工具
│
├── tests/                  # 测试套件
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
│
├── model.py                # 模型定义（CNN、SplitFed 模型）
├── data/splits/            # 数据文件存放目录（自动生成）
└── logs/                   # 训练日志存放目录（自动生成）
```

## 快速开始

### 1. 环境准备

确保已安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖

```bash
uv sync
```

这会安装所有生产依赖和开发依赖（pytest 等）。

### 3. 运行联邦学习训练

#### 方式一：使用 manager.py（推荐）

```bash
uv run python manager.py
```

`manager.py` 会自动：
1. 检查并准备 MNIST 数据（首次运行）
2. 启动监控 API 服务（默认 http://127.0.0.1:9000）
3. 启动服务端进程
4. 启动多个客户端进程
5. 协调训练流程并收集日志

#### 方式二：手动分步启动（用于调试）

```bash
# 1. 准备数据
uv run python scripts/prepare_mnist.py

# 2. 启动监控服务（终端1）
uv run python -m uvicorn utils.monitor_api:app --host 127.0.0.1 --port 9000

# 3. 启动服务端（终端2）
uv run python -m core.server

# 4. 启动客户端（终端3，多个）
uv run python -m core.client client_1
uv run python -m core.client client_2
uv run python -m core.client client_3
```

### 4. 查看训练状态

- **训练日志**：`logs/` 目录下的 `.log` 文件
- **监控面板**：http://127.0.0.1:9000
- **API 接口**：
  - `GET /health` - 健康检查
  - `GET /logs` - 查看训练日志
  - `GET /summary` - 训练统计摘要
  - `GET /training/status` - 训练状态
  - `POST /training/start` - 启动训练（API 模式）
  - `POST /training/stop` - 停止训练

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

### 运行特定测试文件

```bash
uv run pytest tests/unit/test_server.py -v
```

### 带覆盖率报告

```bash
uv run pytest --cov=core --cov=utils --cov=scripts --cov-report=term-missing
```

### 测试配置说明

测试配置位于 `pyproject.toml`：
- 默认并行运行（8 进程）
- 自动收集覆盖率
- 支持 `integration` 标记的测试（需加 `--run-integration`）

## 配置说明

主要配置位于 `config.json`：

### 训练参数（experiment）

```json
{
  "mode": "centralized",           // 训练模式：centralized 或 splitfed
  "global_epochs": 10,             // 全局训练轮数
  "local_epochs": 1,               // 每轮本地训练轮数
  "seed": 42,                      // 随机种子
  "target_accuracy": 0.99,         // 目标精度
  "dataset_params": {
    "batch_size": 64,              // 批次大小
    "num_workers": 0,              // DataLoader worker 数
    "max_train_samples_per_client": 4000  // 每客户端最大样本数
  },
  "optimization": {
    "client_lr": 0.01,             // 客户端学习率
    "server_lr": 0.01,             // 服务端学习率
    "momentum": 0.9,
    "weight_decay": 0.0005
  }
}
```

### 网络拓扑（topology）

```json
{
  "server": {"host": "127.0.0.1", "port": 8000},
  "clients": [
    {"id": "client_1", "host": "127.0.0.1", "port": 8001},
    {"id": "client_2", "host": "127.0.0.1", "port": 8002},
    {"id": "client_3", "host": "127.0.0.1", "port": 8003}
  ]
}
```

### 网络模拟（network）

```json
{
  "stragglers": {                  // 模拟慢节点/丢包
    "client_2": {"delay": 0.0, "drop_rate": 0.0}
  },
  "server_timeout": 180.0,         // 服务端超时时间（秒）
  "compression": false             // 是否启用压缩
}
```

### 监控配置（monitoring）

```json
{
  "api_host": "127.0.0.1",
  "api_port": 9000,
  "render_mode": "auto"            // 渲染模式：auto、live、plain
}
```

## 详细文档

- [项目总览](doc/overview.md) - 架构设计与运行链路
- [配置说明](doc/configuration.md) - 详细配置项解释
- [数据准备与拆分](doc/data-preparation.md) - 数据生成与格式
- [核心通信与训练流程](doc/core-workflow.md) - 通信协议与训练逻辑
- [监控与可视化](doc/monitoring.md) - 监控 API 与事件上报
- [测试与开发](doc/testing-and-development.md) - 测试说明与开发建议

## 开发提示

1. **Python 解释器**：默认使用当前 Python 解释器（`sys.executable`），可通过 `PYTHON_BIN` 环境变量覆盖
2. **数据文件**：首次运行会自动生成 `data/splits/` 下的数据文件
3. **日志清理**：`logs/` 目录会自动创建，可定期手动清理
4. **端口占用**：确保 8000-8003（训练）和 9000（监控）端口未被占用
5. **调试模式**：设置 `"render_mode": "plain"` 可禁用实时渲染，便于查看原始日志

## 常见问题

### Q: 如何添加新的客户端？
A: 修改 `config.json` 中的 `topology.clients`，然后重新运行数据准备脚本：
```bash
uv run python scripts/prepare_mnist.py --num-clients 5
```

### Q: 如何切换 SplitFed 模式？
A: 修改 `config.json`：
```json
{"experiment": {"mode": "splitfed"}}
```

### Q: 如何修改训练轮数？
A: 直接修改 `config.json` 中的 `global_epochs` 和 `local_epochs`，无需重启监控服务。

### Q: 测试失败怎么办？
A: 先确保依赖已安装：`uv sync`，然后单独运行失败的测试查看详细错误。
