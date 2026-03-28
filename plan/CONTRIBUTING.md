# 贡献指南

本文档定义项目的协作规范、开发流程和代码标准。

---

## 开发环境设置

### 1. 克隆仓库

```bash
git clone <repository-url>
cd fed
```

### 2. 创建虚拟环境

```bash
# 使用 uv（推荐）
uv venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 同步所有依赖（包括开发依赖）
uv sync

# 仅同步运行时依赖
uv sync --no-dev
```

### 3. 添加新依赖

```bash
# 添加运行时依赖
uv add <package>

# 添加开发依赖
uv add --dev <package>

# 示例
uv add torch torchvision
uv add --dev pytest ruff pre-commit
```

### 3. 安装 Pre-commit Hooks

```bash
pre-commit install
```

---

## 分支策略

### 分支命名

```
main                    # 主分支，稳定代码
├── dev                 # 开发分支
│   ├── feature/A-model     # 成员A: 模型开发
│   ├── feature/A-data      # 成员A: 数据处理
│   ├── feature/A-client    # 成员A: 客户端开发
│   ├── feature/B-server    # 成员B: 服务端开发
│   ├── feature/B-p2p       # 成员B: P2P开发
│   ├── feature/C-protocol  # 成员C: 协议开发
│   ├── feature/C-transport # 成员C: 传输层开发
│   └── feature/D-test      # 成员D: 测试与日志
```

### 工作流程

1. **创建功能分支**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/<member>-<feature-name>
   ```

2. **开发和提交**
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

3. **推送并创建PR**
   ```bash
   git push origin feature/<member>-<feature-name>
   # 在GitHub上创建Pull Request到dev分支
   ```

4. **Code Review后合并**

---

## 提交规范

### 提交消息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

| Type | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug修复 |
| `docs` | 文档更新 |
| `style` | 代码格式（不影响功能） |
| `refactor` | 重构 |
| `test` | 测试相关 |
| `chore` | 构建/工具相关 |

### Scope

| Scope | 说明 |
|-------|------|
| `model` | 模型相关 |
| `data` | 数据相关 |
| `client` | 客户端 |
| `server` | 服务端 |
| `p2p` | P2P模块 |
| `protocol` | 协议 |
| `transport` | 传输层 |
| `utils` | 工具 |
| `test` | 测试 |
| `docs` | 文档 |

### 示例

```
feat(client): implement federated client connection

- Add TCP connection with timeout
- Implement model receive and send
- Add local training loop

Closes #12
```

```
fix(protocol): handle partial recv in codec

The previous implementation could hang when receiving large payloads.
Added explicit length check and loop for complete receive.

Fixes #23
```

---

## 代码规范

### Python代码风格

- 遵循 PEP 8
- 使用 Ruff 进行 Linting 和格式化
- 行长度限制：100字符
- 使用双引号
- 使用空格缩进

### 类型注解

所有公共接口必须添加类型注解：

```python
def train(
    self,
    model: IModel,
    dataloader: DataLoader,
    epochs: int,
    lr: float = 0.01
) -> TrainingResult:
    """训练模型"""
    pass
```

### 文档字符串

使用 Google 风格：

```python
def aggregate(
    self,
    weights_list: List[Weights],
    client_sizes: Optional[List[int]] = None
) -> Weights:
    """
    聚合多个模型权重

    Args:
        weights_list: 模型权重列表
        client_sizes: 各客户端数据量（用于加权平均）

    Returns:
        Weights: 聚合后的权重

    Raises:
        ValueError: 如果weights_list为空
    """
    pass
```

### 导入顺序

```python
# 标准库
import os
import sys
from typing import Dict, List

# 第三方库
import torch
from torch import nn

# 本地模块
from src.core.interfaces import IModel
from src.utils.logger import get_logger
```

---

## 测试规范

### 测试文件命名

- 单元测试：`tests/unit/test_<module>.py`
- 集成测试：`tests/integration/test_<feature>.py`

### 测试类命名

```python
class TestModel:
    """模型测试"""

    def test_get_weights(self):
        """测试获取权重"""
        pass

    def test_set_weights(self):
        """测试设置权重"""
        pass
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_model.py

# 运行特定测试
pytest tests/unit/test_model.py::TestModel::test_get_weights

# 带覆盖率
pytest --cov=src tests/
```

---

## 文件所有权

| 目录 | 负责人 | 其他人 |
|------|--------|--------|
| `src/model/` | 成员A | 只读 |
| `src/data/` | 成员A | 只读 |
| `src/client/` | 成员A | 只读 |
| `src/server/` | 成员B | 只读 |
| `src/p2p/` | 成员B | 只读 |
| `src/protocol/` | 成员C | 只读 |
| `src/transport/` | 成员C | 只读 |
| `src/utils/` | 成员D | 只读 |
| `tests/` | 成员D | 只读 |
| `docs/` | 成员D | 只读 |
| `config/` | 成员A | 只读 |
| `src/core/` | 共同维护 | 需讨论 |

---

## 冲突解决

### 代码冲突

1. 在PR中解决冲突
2. 优先保留后合并的更改
3. 如有疑问，联系相关文件负责人

### 设计冲突

1. 在GitHub Issue中讨论
2. 提供详细的技术分析
3. 投票决定（需至少2人同意）

---

## 发布流程

1. 确保 `dev` 分支所有测试通过
2. 更新版本号（`pyproject.toml`）
3. 合并 `dev` 到 `main`
4. 创建 Git Tag
5. 生成 Release Notes

---

## 联系方式

如有问题，请在GitHub Issue中提出，或联系对应模块负责人。
