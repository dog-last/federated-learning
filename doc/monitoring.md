# 监控与可视化

## 1. 监控组件

- `utils/monitor_api.py`：FastAPI 监控服务，对外暴露训练事件接口，并负责终端中的实时渲染
- `utils/monitoring.py`：项目内部的事件上报封装
- `utils/training_controller.py`：若从 API 方式启动训练，负责进程编排、状态管理和停止控制

## 2. 监控服务端口

默认监控服务运行在 `config.json.monitoring.api_host` 与 `config.json.monitoring.api_port` 指定的地址上，当前默认是 `127.0.0.1:9000`。

## 3. 监控内容

监控系统主要记录：

### 通用事件

- 训练开始、结束和停止状态
- 客户端与服务端进程信息
- 网络收发统计
- 每轮训练的损失、精度和阶段进度

### Centralized/SplitFed 模式事件

- `manager_start`：管理器启动
- `process_spawn`：进程创建
- `ring_mode_started`：Ring 模式启动
- `topology_update`：拓扑更新
- `round_start`：轮次开始
- `wait_clients`：等待客户端连接
- `batch_progress`：批次进度
- `round_wait_result`：轮次等待结果
- `local_round_done`：本地轮次完成
- `metric`：指标上报
- `round_end`：轮次结束
- `round_transport`：轮次传输摘要
- `straggler_dropped`：Straggler 节点丢弃
- `round_aborted`：轮次中止
- `target_reached`：目标精度达成
- `shutdown`：训练结束
- `network_io`：网络 I/O 事件
- `send_ack`：发送确认
- `recv_ack`：接收确认
- `client_disconnect`：客户端断开

### Ring 模式事件

- `ring_node_startup`：环节点启动
- `ring_node_ready`：环节点就绪
- `ring_all_joined`：所有节点加入环
- `ring_round_start`：环轮次开始
- `ring_local_train_done`：节点本地训练完成
- `ring_send`：环消息发送
- `ring_recv`：环消息接收
- `ring_global_eval`：全局评估
- `ring_round_end`：环轮次结束
- `ring_pass_failed`：环传递失败
- `ring_recv_timeout`：环接收超时
- `ring_round_dropped`：环轮次丢弃

### 控制事件

- `training_start_requested`：训练启动请求
- `training_started`：训练已启动
- `training_stop_requested`：训练停止请求
- `training_stopped`：训练已停止
- `config_updated`：配置更新

## 4. 实时渲染

监控服务支持三种渲染模式：

- **auto**：自动检测终端能力，支持则使用 live 渲染
- **live**：强制使用 Rich Live 渲染
- **plain**：普通文本输出，便于日志收集

### 实时面板包含

- **主表格**：显示训练模式、当前轮次、平均轮次时间、网络统计、指标
- **客户端表格**：显示每个客户端/节点的状态、进度、损失、精度、网络流量
- **事件面板**：显示最近的关键事件日志

### Web 仪表盘模式

当 `render_mode` 设为 `web` 时，监控服务会在启动时自动挂载网页仪表盘。

> **首次使用前需构建前端**：
> ```bash
> cd web
> npm install
> npm run build
> cd ..
> ```
> 构建产物输出到 `web/static/` 目录。若修改了 `web/src/` 下的前端源码，需重新执行 `npm run build`。

- **访问地址**：`http://{api_host}:{api_port}/dashboard`
- **实时通信**：通过 WebSocket (`/ws`) 推送所有训练事件
- **初始同步**：浏览器连接时自动获取完整状态快照和历史指标数据

#### 仪表盘功能

- **节点拓扑可视化**：实时展示节点状态和数据流，Centralized/SplitFed 模式为星型拓扑，Ring 模式为环形拓扑
- **训练指标图表**：Loss 和 Accuracy 曲线（ECharts），支持缩放、图例切换、保存为图片
- **客户端状态表**：实时显示每个客户端/节点的状态、进度、指标、网络流量
- **事件日志**：滚动显示关键事件，按类型着色
- **图表下载**：每个图表右上角提供"保存为图片"按钮
- **自动保存**：训练结束时自动将图表保存到 `output/` 目录

#### 新增 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/dashboard` | GET | 返回网页仪表盘页面 |
| `/ws` | WebSocket | 实时事件推送 |
| `/api/state` | GET | 完整仪表盘状态快照 |
| `/api/metrics/history` | GET | 历史指标数据（用于图表初始化）|

#### 自动保存的图表

训练结束（正常完成、达到目标精度或手动停止）时，后端自动保存以下文件到 `output/` 目录：

| 文件 | 说明 |
|------|------|
| `output/loss_curve.png` | 训练/验证/测试 Loss 曲线 |
| `output/accuracy_curve.png` | 训练/验证/测试 Accuracy 曲线 |
| `output/client_comparison.png` | 各客户端 Loss/Accuracy 对比 |
| `output/metrics.json` | 原始指标数据 |

## 5. API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/logs` | GET | 查看训练日志（支持筛选） |
| `/summary` | GET | 训练统计摘要 |
| `/config` | GET | 获取当前配置 |
| `/config` | PUT | 更新配置 |
| `/clear` | POST | 清空日志 |
| `/training/status` | GET | 获取训练状态 |
| `/training/start` | POST | 启动训练 |
| `/training/stop` | POST | 停止训练 |

### 查询参数

- `source`：按来源筛选日志（如 `client_1`、`server`）
- `event_type`：按事件类型筛选
- `limit`：限制返回日志数量

## 6. 开发注意点

- `monitor_api.py` 需要同时兼容直接运行和模块方式运行
- 若修改前端或控制台展示逻辑，尽量保持事件结构稳定，否则 `TrainingController` 和上层上报代码都要同步调整
- 如果新增可视化字段，建议先在监控事件里加，再在渲染层消费，避免耦合训练逻辑
- Ring 模式和 Centralized 模式共享监控接口，但事件类型不同
- 监控服务使用 Rich 库进行终端渲染，需要终端支持 ANSI 转义序列
- 前端源码在 `web/src/` 下，构建后输出到 `web/static/`；修改前端源码后需执行 `cd web && npm run build` 重新构建
