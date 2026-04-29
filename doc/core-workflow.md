# 核心通信与训练流程

## 1. 核心模块

- `core/communicator.py`：TCP 对象传输协议
- `core/server.py`：训练协调与聚合（Centralized/SplitFed 模式）
- `core/client.py`：本地训练、评估与上报（Centralized/SplitFed 模式）
- `core/ring_node.py`：去中心化环形拓扑节点（Ring 模式）
- `model.py`：集中式模型和 SplitFed 模型定义

## 2. 通信协议

`TCPCommunicator` 使用固定前缀协议传输 Python 对象：

1. 先发送 8 字节大端长度
2. 再发送 4 字节魔数 `SF26`
3. 最后发送 pickle 序列化后的负载

如果启用压缩，则负载会先做 gzip 压缩，再进行传输。

## 3. Centralized/SplitFed 模式流程

### 服务端职责（core.server.py）

- 读取配置并选择设备
- 加载服务端测试集
- 接收客户端更新并进行聚合或 SplitFed 协调
- 记录网络统计并上报监控事件

#### 消息类型

| 消息类型 | 方向 | 说明 |
|---------|------|------|
| `register` | 客户端→服务端 | 客户端注册 |
| `register_ack` | 服务端→客户端 | 注册确认 |
| `round_start_centralized` | 服务端→客户端 | Centralized 模式轮次开始 |
| `round_start_splitfed` | 服务端→客户端 | SplitFed 模式轮次开始 |
| `model_update` | 客户端→服务端 | Centralized 模式模型更新 |
| `split_update` | 客户端→服务端 | SplitFed 模式客户端模型更新 |
| `split_batch` | 客户端→服务端 | SplitFed 批次激活值传输 |
| `split_grad` | 服务端→客户端 | Splitting 激活值梯度 |
| `shutdown` | 服务端→客户端 | 训练结束通知 |

### 客户端职责（core.client.py）

- 加载对应客户端的本地拆分数据
- 构建训练、验证和测试 DataLoader
- 运行本地训练（Centralized 模式）
- 或运行 SplitFed 的前向与反向流程
- 上报训练指标和网络事件

#### Centralized 模式训练流程

1. 接收 `round_start_centralized` 消息
2. 加载全局模型权重
3. 本地训练 `local_epochs` 轮
4. 评估本地模型
5. 发送 `model_update` 消息（包含更新权重和指标）

#### SplitFed 模式训练流程

1. 接收 `round_start_splitfed` 消息
2. 加载客户端和服务端模型权重
3. 对每个训练批次：
   - 客户端前向计算得到激活值
   - 发送 `split_batch` 消息（激活值+标签）
   - 接收 `split_grad` 消息（激活值梯度）
   - 客户端反向传播更新客户端模型
4. 发送 `split_update` 消息（包含客户端更新权重和指标）

## 4. Ring 模式流程

### 环形节点职责（core.ring_node.py）

- 作为 P2P 节点参与去中心化联邦学习
- 监听前驱节点的连接
- 连接到后继节点传递模型
- 本地训练后传递更新模型
- 节点 1 作为发起者，评估全局模型并控制训练生命周期

#### 消息类型

| 消息类型 | 方向 | 说明 |
|---------|------|------|
| `ring_join` | 节点→节点1 | 节点加入环 |
| `ring_pass` | 节点→节点 | 模型权重传递 |
| `ring_shutdown` | 节点→节点 | 训练结束通知 |

### Ring 训练流程

1. **启动同步**：
   - 节点 1 等待其他节点发送 `ring_join` 消息
   - 其他节点向节点 1 发送 `ring_join` 消息

2. **每轮训练**：
   - **节点 1（发起者）**：
     1. 本地训练模型
     2. 发送 `ring_pass` 消息到节点 2
     3. 等待节点 3 传回模型权重
     4. 加载聚合后的模型
     5. 在全局测试集上评估
     6. 检查是否达到目标精度

   - **节点 2 和 3（非发起者）**：
     1. 等待前驱节点传来的 `ring_pass` 消息
     2. 加载模型权重
     3. 本地训练模型
     4. 发送 `ring_pass` 消息到后继节点

3. **训练结束**：
   - 达到目标精度或完成所有轮次
   - 节点 1 发送 `ring_shutdown` 消息
   - 所有节点关闭监听和连接

### Ring 模式特点

- **去中心化**：无中心服务端，无单点故障
- **P2P 通信**：节点间直接 TCP 连接
- **环形拓扑**：模型权重沿环传递
- **发起者控制**：节点 1 控制训练生命周期
- **全局评估**：仅节点 1 在全局测试集上评估

## 5. 模型结构

`model.py` 里提供三类模型：

- `CNN`：集中式完整模型（Centralized/Ring 模式）
- `SplitClientCNN`：SplitFed 的客户端侧模型
- `SplitServerCNN`：SplitFed 的服务端侧模型

`get_model()` 会按模式返回相应模型，后续扩展时也应该从这里统一接入。

## 6. 开发建议

- 调试网络问题时，优先看 `core/communicator.py` 的消息收发是否正常
- 调试训练结果时，优先核对客户端数据和模型输入输出形状
- 若新增消息类型，建议同时更新服务端、客户端和监控上报三处代码
- Ring 模式调试时，注意节点间的端口配置和连接时序
