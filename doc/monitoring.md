# 监控与可视化

## 1. 监控组件

- `utils/monitor_api.py`：FastAPI 监控服务，对外暴露训练事件接口，并负责终端中的实时渲染。
- `utils/monitoring.py`：项目内部的事件上报封装。
- `utils/training_controller.py`：若从 API 方式启动训练，负责进程编排、状态管理和停止控制。

## 2. 监控服务端口

默认监控服务运行在 `config.json.monitoring.api_host` 与 `config.json.monitoring.api_port` 指定的地址上，当前默认是 `127.0.0.1:9000`。

## 3. 监控内容

监控系统主要记录：

- 训练开始、结束和停止状态。
- 客户端与服务端进程信息。
- 网络收发统计。
- 每轮训练的损失、精度和阶段进度。

## 4. 开发注意点

- `monitor_api.py` 需要同时兼容直接运行和模块方式运行。
- 若修改前端或控制台展示逻辑，尽量保持事件结构稳定，否则 `TrainingController` 和上层上报代码都要同步调整。
- 如果新增可视化字段，建议先在监控事件里加，再在渲染层消费，避免耦合训练逻辑。
