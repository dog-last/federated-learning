# 计算机网络课程项目

这是一个面向 MNIST 的联邦学习/分布式训练项目，支持集中式训练与 SplitFed 两种模式。项目的主入口是 `manager.py`，它会自动准备数据、启动监控服务、拉起服务端与多个客户端进程，并把运行日志写入 `logs/`。

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动实验

```bash
python manager.py
```

3. 查看输出

- 训练日志：`logs/`
- 监控服务：`http://127.0.0.1:9000`
- 配置文件：`config.json`

## 项目内容

- `core/`：TCP 通信、服务端、客户端等核心运行逻辑。
- `utils/`：训练控制、监控上报、监控 API 等辅助模块。
- `scripts/`：数据准备与拆分脚本。
- `tests/`：基础单测与通信冒烟测试。
- `data/splits/`：MNIST 拆分后的本地训练数据与服务端测试数据。

## 详细文档

- [项目总览](doc/overview.md)
- [配置说明](doc/configuration.md)
- [数据准备与拆分](doc/data-preparation.md)
- [核心通信与训练流程](doc/core-workflow.md)
- [监控与可视化](doc/monitoring.md)
- [测试与后续开发建议](doc/testing-and-development.md)

## 开发提示

- 默认 Python 解释器来自环境变量 `PYTHON_BIN`，未设置时回退到项目当前使用的 Miniconda 环境。
- 运行前请确认 `data/splits/` 下的 4 个数据文件存在且格式正确；`manager.py` 会在缺失时自动调用数据准备脚本。
- 若要修改网络行为、训练超时或监控显示方式，优先检查 `config.json` 和 `utils/monitor_api.py`。
