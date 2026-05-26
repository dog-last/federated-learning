# 环形拓扑模式（Ring Mode）

## 1. 概述

Ring 模式是一种去中心化的联邦学习架构，节点之间直接形成环形拓扑进行通信，无需中心协调者（Server）。每个节点既是客户端又是服务端，模型权重沿环传递。

## 2. 架构特点

### 去中心化

- 无中心服务端，避免单点故障
- 节点间直接 P2P 通信
- 更符合分布式系统的本质

### 环形拓扑

```
Node 1 → Node 2 → Node 3 → Node 1
   ↑                             ↓
   └─────────────────────────────┘
```

- 每个节点连接到后继节点（successor）
- 每个节点从前驱节点（predecessor）接收数据
- 模型权重沿环单向传递

### 发起者机制

- 节点 1 作为训练发起者（Initiator）
- 控制整个环的生命周期
- 负责全局模型评估
- 检测目标精度并终止训练

## 3. 训练流程

### 启动阶段

1. 所有节点启动监听服务
2. 节点 2、3 向节点 1 发送 `ring_join` 消息
3. 节点 1 等待所有节点加入后开始训练

### 训练轮次

#### 节点 1（发起者）

1. 本地训练模型（使用本地数据）
2. 向节点 2 发送 `ring_pass` 消息（包含模型权重）
3. 等待节点 3 传回的模型权重
4. 加载聚合后的模型权重
5. 在全局测试集上评估模型
6. 检查是否达到目标精度
7. 如达到目标精度，发送 `ring_shutdown` 并退出

#### 节点 2、3（普通节点）

1. 等待前驱节点的 `ring_pass` 消息
2. 加载接收到的模型权重
3. 本地训练模型（使用本地数据）
4. 向后继节点发送 `ring_pass` 消息（包含更新后的模型权重）

### 结束阶段

1. 达到目标精度或完成所有轮次
2. 节点 1 发送 `ring_shutdown` 消息
3. 所有节点关闭监听和连接
4. 监控服务记录训练完成事件

## 4. 消息协议

### ring_join

```python
{
  "type": "ring_join",
  "node_id": 2  # 或 3
}
```

### ring_pass

```python
{
  "type": "ring_pass",
  "round": 1,
  "origin_id": 1,  # 发送节点 ID
  "weights": {...},  # 模型状态字典
  "num_samples": 4000  # 训练样本数
}
```

### ring_shutdown

```python
{
  "type": "ring_shutdown",
  "round": 10,
  "origin_id": 1,
  "reason": "training_finished"
}
```

## 5. 配置要求

### topology.nodes

必须配置节点列表，而不是 server 和 clients：

```json
{
  "topology": {
    "nodes": [
      {"id": 1, "host": "127.0.0.1", "port": 8101},
      {"id": 2, "host": "127.0.0.1", "port": 8102},
      {"id": 3, "host": "127.0.0.1", "port": 8103}
    ]
  }
}
```

### network.stragglers

Ring 模式下使用 `client_{node_id}` 作为键：

```json
{
  "stragglers": {
    "client_1": {"delay": 0.0, "drop_rate": 0.0},
    "client_2": {"delay": 5.0, "drop_rate": 0.0},
    "client_3": {"delay": 0.0, "drop_rate": 0.0}
  }
}
```

## 6. 运行方式

### 使用 manager.py（推荐）

```bash
# 修改 config.json 中 experiment.mode 为 "ring"
uv run python manager.py
```

### 手动启动

```bash
# 启动监控服务
uv run python -m uvicorn utils.monitor_api:app --host 127.0.0.1 --port 9000

# 启动环节点（不同终端）
uv run python -m core.ring_node 1
uv run python -m core.ring_node 2
uv run python -m core.ring_node 3
```

## 7. 监控事件

Ring 模式有专门的监控事件类型：

| 事件类型 | 说明 |
|---------|------|
| `ring_node_startup` | 环节点启动 |
| `ring_node_ready` | 环节点就绪 |
| `ring_all_joined` | 所有节点加入环 |
| `ring_round_start` | 环轮次开始 |
| `ring_local_train_done` | 节点本地训练完成 |
| `ring_send` | 环消息发送 |
| `ring_recv` | 环消息接收 |
| `ring_global_eval` | 全局评估（仅节点 1） |
| `ring_round_end` | 环轮次结束 |
| `ring_pass_failed` | 环传递失败 |
| `ring_recv_timeout` | 环接收超时 |
| `ring_round_dropped` | 环轮次丢弃 |

## 8. 与 Centralized 模式对比

| 特性 | Centralized | Ring |
|------|------------|------|
| 架构 | 有中心服务端 | 去中心化 |
| 通信方式 | 客户端↔服务端 | 节点↔节点 |
| 单点故障 | 服务端 | 无（但环中断会影响后续节点） |
| 聚合策略 | FedAvg 在服务端 | 权重传递（隐式聚合） |
| 全局评估 | 每轮服务端评估 | 每轮节点 1 评估 |
| 复杂度 | 较简单 | 需要节点协调 |

## 9. 扩展建议

### 增加节点数量

1. 修改 `config.json` 中的 `topology.nodes`，添加更多节点
2. 重新运行数据准备脚本生成对应数据文件
3. 调整端口配置避免冲突

### 改进环拓扑

- 可以扩展为更复杂的拓扑（如树形、网状）
- 实现多环并行训练
- 添加节点动态加入/退出机制

### 容错机制

- 实现节点故障检测
- 添加环重构机制（跳过故障节点）
- 实现检查点恢复

## 10. 调试技巧

### 查看节点日志

```bash
# 实时查看节点 1 日志
tail -f logs/<运行时间>/ring-node-1.log

# 查看所有节点日志
tail -f logs/<运行时间>/ring-node-*.log
```

### 网络问题调试

1. 检查端口是否被占用
2. 检查防火墙设置
3. 使用 `tcpdump` 或 `wireshark` 抓包分析
4. 增加日志级别查看详细通信过程

### 训练不收敛

1. 检查数据拆分是否正确
2. 验证学习率是否合适
3. 检查模型权重传递是否完整
4. 对比 Centralized 模式的训练结果
