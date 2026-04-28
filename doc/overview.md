# 项目总览

## 1. 项目目标

本项目实现一个用于 MNIST 分类的联邦学习实验框架，同时保留集中式训练路径，便于比较不同训练方式的通信成本、收敛行为与监控表现。

## 2. 运行主链路

一次完整实验的主链路如下：

1. `manager.py` 读取 `config.json`。
2. 如有必要，`scripts/prepare_mnist.py` 生成 `data/splits/` 下的数据文件。
3. 启动 `utils/monitor_api.py` 对外提供监控上报接口。
4. 启动 `core.server` 与多个 `core.client` 进程。
5. 服务端与客户端通过 `core.communicator.TCPCommunicator` 进行 TCP 通信。
6. `utils.monitoring.MonitorReporter` 将事件上报到监控服务。

## 3. 代码分层

- `manager.py`：实验编排层，负责进程启动与回收。
- `core/`：训练节点与网络协议实现。
- `model.py`：集中式与 SplitFed 的模型定义。
- `scripts/`：数据准备脚本。
- `utils/`：监控、训练控制和状态聚合。

## 4. 推荐阅读顺序

如果是第一次接手这个项目，建议按下面顺序看代码和文档：

1. `README.md`
2. `doc/configuration.md`
3. `doc/data-preparation.md`
4. `doc/core-workflow.md`
5. `doc/monitoring.md`
