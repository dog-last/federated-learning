# 核心通信与训练流程

## 1. 核心模块

- `core/communicator.py`：TCP 对象传输协议。
- `core/server.py`：训练协调与聚合。
- `core/client.py`：本地训练、评估与上报。
- `model.py`：集中式模型和 SplitFed 模型定义。

## 2. 通信协议

`TCPCommunicator` 使用固定前缀协议传输 Python 对象：

1. 先发送 8 字节大端长度。
2. 再发送 4 字节魔数 `SF26`。
3. 最后发送 pickle 序列化后的负载。

如果启用压缩，则负载会先做 gzip 压缩，再进行传输。

## 3. 服务端职责

`core.server.Server` 负责：

- 读取配置并选择设备。
- 加载服务端测试集。
- 接收客户端更新并进行聚合或 SplitFed 协调。
- 记录网络统计并上报监控事件。

## 4. 客户端职责

`core.client.Client` 负责：

- 加载对应客户端的本地拆分数据。
- 构建训练、验证和测试 DataLoader。
- 运行本地训练或 SplitFed 的前向与反向流程。
- 上报训练指标和网络事件。

## 5. 模型结构

`model.py` 里提供三类模型：

- `CNN`：集中式完整模型。
- `SplitClientCNN`：SplitFed 的客户端侧模型。
- `SplitServerCNN`：SplitFed 的服务端侧模型。

`get_model()` 会按模式返回相应模型，后续扩展时也应该从这里统一接入。

## 6. 开发建议

- 调试网络问题时，优先看 `core/communicator.py` 的消息收发是否正常。
- 调试训练结果时，优先核对客户端数据和模型输入输出形状。
- 若新增消息类型，建议同时更新服务端、客户端和监控上报三处代码。
