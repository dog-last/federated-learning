# 数据准备与拆分

## 1. 数据文件约定

训练运行前，项目需要以下文件存在于 `data/splits/`：

- `client_1_data.pt`
- `client_2_data.pt`
- `client_3_data.pt`
- `server_test_data.pt`

`manager.py` 和 `core.server`、`core.client` 都会检查这些文件是否可用。

## 2. 生成入口

数据准备脚本位于 `scripts/prepare_mnist.py`。它会下载 MNIST，按客户端拆分训练样本，再生成服务端测试集。

运行方式：

```bash
python scripts/prepare_mnist.py --data-dir ./data --num-clients 3 --seed 42
```

## 3. 数据格式

客户端数据文件保存为字典，常见键包括：

- `train_images` / `train_labels`
- `val_images` / `val_labels`
- `test_images` / `test_labels`

服务端测试文件保存为：

- `images`
- `labels`

## 4. 处理逻辑

- 输入图像是 MNIST 单通道 28x28 张量。
- 客户端侧和服务端侧都会做 MNIST 归一化处理。
- 拆分策略当前是非 IID 分布，适合联邦学习实验和异常客户端模拟。

## 5. 开发注意点

- 代码里对数据形状有检查，至少要保持 `[N, 1, 28, 28]` 的 4 维张量格式。
- 若未来切换到其他数据集，客户端、服务端和归一化逻辑需要一起改。
- 如新增客户端数量，要同步修改拆分脚本、配置文件和数据文件命名规则。
