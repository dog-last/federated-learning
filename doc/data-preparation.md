# 数据准备与拆分

## 1. 数据文件约定

训练运行前，项目需要以下文件存在于 `data/splits/`：

### Centralized/SplitFed 模式

- `client_1_data.pt`
- `client_2_data.pt`
- `client_3_data.pt`
- `server_test_data.pt`

`manager.py` 和 `core.server`、`core.client` 都会检查这些文件是否可用。

### Ring 模式

- `client_1_data.pt` （对节点 1）
- `client_2_data.pt` （对节点 2）
- `client_3_data.pt` （对节点 3）
- `server_test_data.pt` （全局测试集）

Ring 模式使用相同的数据文件命名约定，节点 ID 直接映射到客户端 ID。

## 2. 生成入口

数据准备脚本位于 `scripts/prepare_mnist.py`。它会下载 MNIST，按客户端拆分训练样本，再生成服务端测试集。

运行方式：

```bash
# 默认配置（3 个客户端）
python scripts/prepare_mnist.py

# 自定义配置
python scripts/prepare_mnist.py --data-dir ./data --num-clients 5 --seed 42
```

### 参数说明

- `--data-dir`：数据根目录（默认 `./data`）
- `--num-clients`：客户端/节点数量（默认 3）
- `--seed`：随机种子（默认 42）

## 3. 数据格式

### 客户端数据文件

保存为字典，包含以下键：

```python
{
  "train_images": torch.Tensor,  # [N, 1, 28, 28]
  "train_labels": torch.Tensor,  # [N]
  "val_images": torch.Tensor,    # [N, 1, 28, 28]
  "val_labels": torch.Tensor,    # [N]
  "test_images": torch.Tensor,   # [N, 1, 28, 28]
  "test_labels": torch.Tensor    # [N]
}
```

### 服务端测试数据文件

保存为字典：

```python
{
  "images": torch.Tensor,  # [N, 1, 28, 28]
  "labels": torch.Tensor   # [N]
}
```

## 4. 处理逻辑

### MNIST 下载

- 使用 `torchvision.datasets.MNIST` 自动下载
- 自动处理缓存，避免重复下载
- 支持训练集和测试集分离

### 数据拆分策略

#### 非 IID 分布

- 将 10 个数字类别分配给不同客户端
- 每个客户端主要拥有某些类别的样本
- 部分样本在其他客户端之间共享，增加联邦学习难度

#### 拆分比例

- 训练集：80%
- 验证集：10%
- 测试集：10%

### 归一化处理

MNIST 归一化参数：

- 均值：0.1307
- 标准差：0.3081

归一化公式：

```python
normalized = (image - 0.1307) / 0.3081
```

## 5. 代码实现细节

### 主要函数

```python
def prepare_mnist_federated(
    root_dir="./data",
    num_clients=3,
    seed=42
):
    """
    准备联邦学习 MNIST 数据集
    
    Args:
        root_dir: 数据根目录
        num_clients: 客户端数量
        seed: 随机种子
    
    Returns:
        dict: 包含客户端数据集、服务端测试数据集和统计信息
    """
```

### 数据拆分算法

1. 按类别组织训练数据索引
2. 为每个客户端分配主要的类别范围
3. 在客户端之间共享部分类别样本（非 IID）
4. 对每个客户端的索引进行 shuffle
5. 按 8:1:1 比例拆分为训练/验证/测试集

## 6. 使用示例

### 在代码中加载

```python
import torch
import os
from torch.utils.data import DataLoader, TensorDataset

# 加载客户端数据
client_data = torch.load("data/splits/client_1_data.pt")
train_loader = DataLoader(
    TensorDataset(client_data["train_images"], client_data["train_labels"]),
    batch_size=64,
    shuffle=True
)

# 加载服务端测试数据
server_data = torch.load("data/splits/server_test_data.pt")
test_loader = DataLoader(
    TensorDataset(server_data["images"], server_data["labels"]),
    batch_size=64,
    shuffle=False
)
```

### 检查数据格式

```python
# 验证数据格式
import torch

client_payload = torch.load("data/splits/client_1_data.pt")
assert client_payload["train_images"].shape[1:] == (1, 28, 28)
assert client_payload["train_images"].ndim == 4
```

## 7. 开发注意点

### 数据形状要求

- 必须保持 `[N, 1, 28, 28]` 的 4 维张量格式
- N 是样本数量
- 1 是通道数（MNIST 单通道）
- 28x28 是图像尺寸

### 兼容性检查

项目代码包含数据格式兼容性检查：

```python
def _is_mnist_split_compatible(required_files):
    """检查数据文件格式是否兼容"""
    try:
        client_payload = torch.load(required_files[0], map_location="cpu")
        server_payload = torch.load(required_files[-1], map_location="cpu")

        client_images = client_payload["train_images"]
        server_images = server_payload["images"]
        
        return (
            client_images.ndim == 4 and 
            client_images.shape[1:] == (1, 28, 28) and 
            server_images.shape[1:] == (1, 28, 28)
        )
    except (OSError, RuntimeError, KeyError, IndexError, ValueError, TypeError):
        return False
```

### 扩展到其他数据集

如果需要支持其他数据集（如 CIFAR-10）：

1. 创建新的数据准备脚本（如 `prepare_cifar10.py`）
2. 调整归一化参数和图像尺寸
3. 保持相同的数据文件格式
4. 更新兼容性检查逻辑
5. 修改 `core/client.py` 和 `core/server.py` 的数据加载代码

## 8. 数据统计

运行数据准备后，会输出统计信息：

```
[CLIENT 1] train=3592 val=449 test=449 -> data/splits/client_1_data.pt
[CLIENT 2] train=3648 val=456 test=456 -> data/splits/client_2_data.pt
[CLIENT 3] train=3488 val=436 test=436 -> data/splits/client_3_data.pt
[SERVER] test=10000 -> data/splits/server_test_data.pt

=== Dataset Preparation Summary ===
clients: [...]
server_test_samples: 10000
total_train_samples: 10728
```

## 9. 故障排查

### 文件缺失

**错误**：`Missing data/splits/client_1_data.pt`

**解决**：运行数据准备脚本
```bash
python scripts/prepare_mnist.py
```

### 格式不匹配

**错误**：`Invalid tensor shape`

**解决**：删除旧数据文件，重新运行准备脚本
```bash
rm -rf data/splits/
python scripts/prepare_mnist.py
```

### 下载失败

**错误**：`Connection refused` 或 `Timeout`

**解决**：
1. 检查网络连接
2. 手动下载 MNIST 数据集
3. 使用镜像或代理

## 10. 性能优化

### 缓存利用

- MNIST 数据集会自动缓存到 `data/` 目录
- 避免重复下载，加快后续运行

### 并行处理

- 数据拆分过程支持并行化优化
- 对大数据集可以考虑使用多线程

### 内存管理

- 使用 `torch.load` 时指定 `map_location="cpu"` 减少显存占用
- 大数据集考虑分批处理
