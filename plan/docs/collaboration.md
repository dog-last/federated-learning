# 分工协作

> 本文档定义团队分工、协作规范和开发流程。

---

## 团队分工

### 成员A：模型、数据、客户端

**职责范围：**
- 模型定义与实现
- 数据集加载与划分
- 客户端逻辑开发
- 本地训练实现
- 配置文件维护

**负责文件：**
```
src/model/          # 模型模块
src/data/           # 数据模块
src/client/         # 客户端模块
config/*.yaml       # 配置文件
scripts/download_cifar10.py
scripts/split_dataset.py
scripts/run_client.py
```

**核心任务：**
| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 实现 SimpleCNN 模型 | P0 | core/interfaces.py |
| 实现 CIFAR-10 加载与划分 | P0 | - |
| 实现客户端连接与注册 | P0 | protocol/*, transport/* |
| 实现本地训练逻辑 | P0 | model/*, data/* |
| 实现模型评估 | P1 | model/* |

---

### 成员B：服务端开发

**职责范围：**
- 服务端逻辑开发
- 聚合算法实现
- 客户端连接管理
- 轮次协调与超时控制
- P2P去中心化实现
- 故障检测与恢复

**负责文件：**
```
src/server/         # 服务端模块
src/p2p/            # P2P模块
scripts/run_server.py
scripts/run_p2p_node.py
```

**核心任务：**
| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 实现服务端监听与多线程 | P0 | transport/* |
| 实现客户端连接管理 | P0 | transport/* |
| 实现 FedAvg 聚合 | P0 | - |
| 实现轮次协调器 | P0 | server/client_manager.py |
| 实现超时控制 | P0 | transport/timeout.py |
| 实现环形拓扑管理 | P1 | protocol/* |
| 实现故障检测与跳过 | P1 | p2p/topology.py |
| 实现模型检查点 | P2 | - |

---

### 成员C：网络协议设计

**职责范围：**
- 消息协议设计
- 编解码实现
- 粘包处理
- 序列化/反序列化
- TCP连接封装
- 超时控制

**负责文件：**
```
src/protocol/       # 协议模块
src/transport/      # 传输模块
```

**核心任务：**
| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 定义消息类型与格式 | P0 | core/types.py |
| 实现编解码器 | P0 | protocol/message.py |
| 实现粘包处理 | P0 | protocol/codec.py |
| 实现权重序列化 | P0 | torch, pickle |
| 实现TCP连接封装 | P0 | socket |
| 实现监听器 | P0 | socket |
| 实现超时控制 | P0 | - |

---

### 成员D：测试与日志

**职责范围：**
- 日志系统开发
- 网络状态监控
- 指标统计
- 可视化工具
- 单元测试
- 集成测试
- 文档撰写

**负责文件：**
```
src/utils/          # 工具模块
tests/              # 测试目录
docs/               # 文档目录
```

**核心任务：**
| 任务 | 优先级 | 依赖 |
|------|--------|------|
| 实现结构化日志 | P0 | logging |
| 实现网络状态监控 | P0 | utils/logger.py |
| 实现指标收集 | P0 | - |
| 实现计时工具 | P0 | time |
| 实现准确率曲线绘制 | P1 | matplotlib |
| 编写单元测试 | P1 | pytest, 所有模块 |
| 编写集成测试 | P1 | pytest, 所有模块 |
| 撰写测试报告 | P2 | tests/* |
| 撰写用户手册 | P2 | - |

---

## 协作规范

### 代码规范

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

```toml
# pyproject.toml
[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Git提交规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type:**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

**Scope:**
- `model`: 模型相关
- `data`: 数据相关
- `client`: 客户端
- `server`: 服务端
- `p2p`: P2P模块
- `protocol`: 协议
- `transport`: 传输层
- `utils`: 工具
- `test`: 测试
- `docs`: 文档

**示例:**
```
feat(client): implement federated client connection

- Add TCP connection with timeout
- Implement model receive and send
- Add local training loop

Closes #12
```

### 分支管理

```
main                    # 稳定分支，只接受PR
├── dev                 # 开发分支，日常合并
│   ├── feature/A-*     # 成员A的功能分支
│   ├── feature/B-*     # 成员B的功能分支
│   ├── feature/C-*     # 成员C的功能分支
│   └── feature/D-*     # 成员D的功能分支
```

**工作流:**
1. 从 `dev` 创建功能分支
2. 开发完成后创建 PR 到 `dev`
3. Code Review 通过后合并
4. 测试通过后 `dev` 合并到 `main`

---

## 开发流程

### Phase 1: 基础设施（第1-2天）

**目标：** 完成核心接口和协议层

| 成员 | 任务 | 产出 |
|------|------|------|
| 共同 | 定义核心接口 | `core/interfaces.py` |
| 共同 | 定义核心类型 | `core/types.py` |
| 成员C | 实现消息协议 | `protocol/*` |
| 成员C | 实现TCP传输 | `transport/*` |
| 成员D | 搭建日志框架 | `utils/logger.py` |
| 成员D | 配置pre-commit | `.pre-commit-config.yaml` |

### Phase 2: 核心功能（第3-5天）

**目标：** 完成中心化联邦学习

| 成员 | 任务 | 产出 |
|------|------|------|
| 成员A | 实现SimpleCNN | `model/simple_cnn.py` |
| 成员A | 实现数据加载划分 | `data/*` |
| 成员A | 实现客户端 | `client/*` |
| 成员B | 实现服务端 | `server/*` |
| 成员B | 实现聚合算法 | `server/aggregator.py` |
| 成员D | 实现指标统计 | `utils/metrics.py` |

### Phase 3: 扩展功能（第6-8天）

**目标：** 完成去中心化和掉队者处理

| 成员 | 任务 | 产出 |
|------|------|------|
| 成员B | 实现环形拓扑 | `p2p/topology.py` |
| 成员B | 实现故障检测 | `p2p/failure_detector.py` |
| 成员B | 实现故障恢复 | `p2p/recovery.py` |
| 成员B | 实现超时控制 | `server/round_coordinator.py` |
| 成员D | 编写集成测试 | `tests/integration/*` |

### Phase 4: 测试与文档（第9-10天）

**目标：** 完成测试和文档

| 成员 | 任务 | 产出 |
|------|------|------|
| 成员D | 完成单元测试 | `tests/unit/*` |
| 成员D | 撰写测试报告 | `docs/TEST_REPORT.md` |
| 成员D | 撰写用户手册 | `docs/USER_MANUAL.md` |
| 成员D | 绘制准确率曲线 | `outputs/figures/*` |
| 全体 | Code Review | - |
| 全体 | 最终测试 | - |

---

## 沟通机制

### 日常沟通

- **问题讨论：** 在GitHub Issue中讨论技术问题
- **进度同步：** 每日在群内同步进度
- **阻塞上报：** 遇到阻塞立即在群内上报

### 接口变更

接口变更需遵循以下流程：

1. 在Issue中提出变更请求
2. 说明变更原因和影响范围
3. 获得至少2人同意
4. 更新 `core/interfaces.py`
5. 通知所有相关成员

### 冲突解决

- **代码冲突：** 在PR中解决，优先保留后合并的更改
- **设计冲突：** 在Issue中讨论，投票决定
- **进度冲突：** 调整任务优先级，必要时重新分配

---

## 交付物清单

### 代码交付物

| 成员 | 交付物 | 验收标准 |
|------|--------|----------|
| A | 模型、数据、客户端 | 单元测试通过，可独立运行 |
| B | 服务端、P2P | 单元测试通过，支持多客户端 |
| C | 协议、传输 | 单元测试通过，粘包处理正确 |
| D | 日志、测试、文档 | 覆盖率>80%，文档完整 |

### 文档交付物

| 文档 | 负责人 | 内容 |
|------|--------|------|
| 架构文档 | 成员D | 系统架构图、模块关系 |
| API文档 | 成员D | 接口说明、使用示例 |
| 协议文档 | 成员C | 封包结构、消息格式 |
| 测试报告 | 成员D | 测试结果、准确率曲线 |
| 用户手册 | 成员D | 启动说明、配置说明 |
