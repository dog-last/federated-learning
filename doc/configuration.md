# 配置说明

## 1. 配置文件位置

项目主配置位于 `config.json`，训练、网络与监控参数都在这里集中管理。

## 2. 主要配置项

### experiment

- `mode`：训练模式，支持 `centralized`、`splitfed` 或 `ring`
- `global_epochs`：全局轮数
- `local_epochs`：本地轮数
- `seed`：随机种子
- `target_accuracy`：实验目标精度
- `device`：计算设备，支持 `auto`、`cuda`、`mps`、`cpu`
- `dataset_params.batch_size`：批大小
- `dataset_params.num_workers`：DataLoader worker 数量
- `dataset_params.val_ratio`：验证集比例
- `dataset_params.test_ratio`：测试集比例
- `dataset_params.max_train_samples_per_client`：每个客户端最多使用的训练样本数
- `optimization.client_lr` 和 `optimization.server_lr`：学习率
- `optimization.momentum` 和 `optimization.weight_decay`：优化器参数

### topology

#### Centralized/SplitFed 模式

- `server`：服务端地址与端口
- `clients`：客户端列表，每个客户端包含 `id`、`host`、`port`

#### Ring 模式

- `nodes`：节点列表，每个节点包含 `id`、`host`、`port`
- 节点间形成环形拓扑：Node 1 → Node 2 → Node 3 → Node 1

### network

- `stragglers`：为指定客户端配置延迟和丢包
- `server_timeout`：服务端等待轮次结果的超时时间（Centralized/SplitFed 模式）
- `min_clients`：最小响应客户端数，用于 straggler 处理
- `compression`：是否对 TCP 载荷启用压缩

### monitoring

- `api_host` 与 `api_port`：监控 API 地址
- `render_mode`：监控显示模式，支持 `auto`、`live`、`plain`、`web`
  - `auto`：自动检测终端能力，终端环境下使用 live 渲染，非终端环境使用 web 模式
  - `live`：强制使用 Rich Live 终端渲染
  - `plain`：普通文本输出，便于日志收集
  - `web`：网页仪表盘模式，通过浏览器访问监控界面，支持实时数据推送、图表、节点拓扑可视化

## 3. 配置示例

### Centralized 模式配置

```json
{
  "experiment": {
    "mode": "centralized",
    "global_epochs": 10,
    "local_epochs": 1
  },
  "topology": {
    "server": {"host": "127.0.0.1", "port": 8000},
    "clients": [
      {"id": "client_1", "host": "127.0.0.1", "port": 8001},
      {"id": "client_2", "host": "127.0.0.1", "port": 8002},
      {"id": "client_3", "host": "127.0.0.1", "port": 8003}
    ]
  }
}
```

### Ring 模式配置

```json
{
  "experiment": {
    "mode": "ring",
    "global_epochs": 10,
    "local_epochs": 1
  },
  "topology": {
    "nodes": [
      {"id": 1, "host": "127.0.0.1", "port": 8101},
      {"id": 2, "host": "127.0.0.1", "port": 8102},
      {"id": 3, "host": "127.0.0.1", "port": 8103}
    ]
  }
}
```

## 4. 修改建议

- 修改训练轮数或学习率时，优先只改 `experiment` 下的参数
- 增加或减少客户端时，同时调整 `topology.clients` 和数据拆分逻辑
- 切换 Ring 模式时，将 `topology.clients` 改为 `topology.nodes`，并调整端口
- 如需调整监控渲染行为，优先改 `monitoring.render_mode`
- Ring 模式下，`network.server_timeout` 和 `network.min_clients` 配置项不适用

## 5. 端口规划建议

为了避免端口冲突，建议按以下方式规划端口：

| 组件 | 起始端口 | 说明 |
|------|----------|------|
| 监控服务 | 9000 | 固定端口 |
| Centralized 服务端 | 8000 | 固定端口 |
| Centralized 客户端 | 8001+ | 根据客户端数量递增 |
| Ring 节点 | 8101+ | 根据节点数量递增 |
| Web 仪表盘 | 9000 | 与监控服务共用端口，路径为 /dashboard |

## 6. Straggler 配置

Straggler 配置用于模拟网络环境中的慢节点和丢包情况：

```json
{
  "stragglers": {
    "client_1": {
      "delay": 0.0,      // 延迟秒数
      "drop_rate": 0.0    // 丢包率 [0, 1]
    },
    "client_2": {
      "delay": 5.0,      // 模拟 5 秒延迟
      "drop_rate": 0.1    // 模拟 10% 丢包率
    }
  }
}
```

- Ring 模式下，使用 `"client_1"`、`"client_2"` 等键名（对应节点 ID)
- Straggler 功能主要用于测试系统的鲁棒性
