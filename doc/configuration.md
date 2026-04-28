# 配置说明

## 1. 配置文件位置

项目主配置位于 `config.json`，训练、网络与监控参数都在这里集中管理。

## 2. 主要配置项

### experiment

- `mode`：训练模式，当前主要使用 `centralized` 或 `splitfed`。
- `global_epochs`：全局轮数。
- `local_epochs`：本地轮数。
- `seed`：随机种子。
- `target_accuracy`：实验目标精度。
- `dataset_params.batch_size`：批大小。
- `dataset_params.num_workers`：DataLoader worker 数量。
- `dataset_params.max_train_samples_per_client`：每个客户端最多使用的训练样本数。
- `optimization.client_lr` 和 `optimization.server_lr`：学习率。
- `optimization.momentum` 和 `optimization.weight_decay`：优化器参数。

### topology

- `server`：服务端地址与端口。
- `clients`：客户端列表，当前配置为 3 个客户端。

### network

- `stragglers`：为指定客户端配置延迟和丢包。
- `server_timeout`：服务端等待轮次结果的超时时间。
- `compression`：是否对 TCP 载荷启用压缩。

### monitoring

- `api_host` 与 `api_port`：监控 API 地址。
- `render_mode`：监控显示模式，支持 `auto`、`live`、`plain`。

## 3. 修改建议

- 修改训练轮数或学习率时，优先只改 `experiment` 下的参数。
- 增加或减少客户端时，同时调整 `topology.clients` 和数据拆分逻辑，确保 `data/splits/` 文件名与客户端 ID 一致。
- 如需调整监控渲染行为，优先改 `monitoring.render_mode`，再考虑改 `utils/monitor_api.py`。
