# 测试与开发指南

## 测试架构

项目使用 **pytest** 作为测试框架，测试代码位于 `tests/` 目录：

```
tests/
├── conftest.py                    # pytest 配置与共享 fixtures
├── unit/                          # 单元测试
│   ├── test_client.py             # 客户端逻辑测试
│   ├── test_client_extended.py    # 客户端扩展测试
│   ├── test_communicator.py       # TCP 通信协议测试
│   ├── test_config.py             # 配置加载测试
│   ├── test_model.py              # 模型定义测试
│   ├── test_monitor_api.py        # 监控 API 测试
│   ├── test_monitoring.py         # 监控上报测试
│   ├── test_prepare_mnist.py      # 数据准备测试
│   ├── test_scripts.py            # 脚本测试
│   ├── test_server.py             # 服务端逻辑测试
│   ├── test_server_extended.py    # 服务端扩展测试
│   ├── test_training_controller.py          # 训练控制器测试
│   └── test_training_controller_extended.py # 训练控制器扩展测试
│
└── integration/                   # 集成测试
    ├── test_client_server.py      # 客户端-服务端集成测试
    ├── test_data_preparation.py   # 数据准备集成测试
    └── test_tcp.py                # TCP 通信集成测试
    └── test_training_controller_integration.py  # 训练控制器集成测试
```

## 运行测试

### 基础命令

```bash
# 运行所有测试
uv run pytest

# 运行单元测试
uv run pytest tests/unit/

# 运行集成测试
uv run pytest tests/integration/

# 详细输出
uv run pytest -v

# 运行特定测试文件
uv run pytest tests/unit/test_server.py -v

# 运行特定测试函数
uv run pytest tests/unit/test_server.py::TestServer::test_init -v
```

### 覆盖率测试

```bash
# 生成覆盖率报告（默认配置）
uv run pytest

# 仅显示覆盖率摘要
uv run pytest --cov=core --cov=utils --cov=scripts --cov-report=term

# 显示未覆盖的代码行
uv run pytest --cov=core --cov=utils --cov=scripts --cov-report=term-missing

# 生成 HTML 报告
uv run pytest --cov=core --cov=utils --cov=scripts --cov-report=html
```

### 并行测试

```bash
# 使用 8 个进程并行运行（默认配置）
uv run pytest

# 手动指定进程数
uv run pytest -n 4

# 单进程运行（便于调试）
uv run pytest -n 0
```

### 标记与筛选

```bash
# 运行集成测试（需要显式启用）
uv run pytest --run-integration

# 仅运行标记为 integration 的测试
uv run pytest -m integration

# 排除集成测试
uv run pytest -m "not integration"

# 按关键字筛选
uv run pytest -k "test_server"
```

## 测试配置

测试配置位于 `pyproject.toml`：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-n 8 --cov=core --cov=utils --cov=scripts --cov-report=term-missing:skip-covered"
markers = [
    "integration: marks tests as integration tests (need --run-integration to run)",
]
```

### 配置说明

- **并行运行**：默认使用 8 个进程（`-n 8`）
- **覆盖率**：自动收集 core、utils、scripts 模块的覆盖率
- **集成测试**：标记为 `integration` 的测试默认跳过，需加 `--run-integration` 运行

## 编写测试

### 单元测试示例

```python
# tests/unit/test_example.py
import pytest
from core.server import Server

class TestServer:
    def test_server_init(self, tmp_path):
        """测试服务端初始化"""
        config = {"experiment": {"mode": "centralized"}}
        server = Server(config)
        assert server.mode == "centralized"

    def test_server_with_mock(self, mocker):
        """使用 mock 测试"""
        mock_socket = mocker.patch('socket.socket')
        # 测试逻辑...
```

### 集成测试示例

```python
# tests/integration/test_example.py
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_client_server_communication():
    """测试客户端-服务端通信"""
    # 启动服务端
    # 启动客户端
    # 验证通信...
```

### 使用 Fixtures

`conftest.py` 中定义的 fixtures 可在所有测试中共享：

```python
# conftest.py 示例
import pytest
import tempfile
import json

@pytest.fixture
def temp_config():
    """提供临时配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"experiment": {"mode": "centralized"}}, f)
        f.flush()
        yield f.name

@pytest.fixture
def sample_data():
    """提供测试数据"""
    return {
        "train_images": [],
        "train_labels": [],
    }
```

## 开发工作流

### 1. 开发新功能

```bash
# 1. 创建功能分支
git checkout -b feature/new-feature

# 2. 编写代码
# ...

# 3. 编写测试
# tests/unit/test_new_feature.py

# 4. 运行测试确保通过
uv run pytest tests/unit/test_new_feature.py -v

# 5. 运行所有测试确保无回归
uv run pytest

# 6. 提交代码
```

### 2. 调试测试

```bash
# 使用 pdb 调试
uv run pytest tests/unit/test_server.py --pdb

# 在失败处停止
uv run pytest tests/unit/test_server.py -x

# 显示完整的错误信息
uv run pytest tests/unit/test_server.py -v --tb=long

# 仅显示失败的测试
uv run pytest --lf
```

### 3. 性能测试

```bash
# 显示最慢的测试
uv run pytest --durations=10

# 使用 pytest-benchmark 进行基准测试
uv run pytest tests/unit/test_performance.py --benchmark-only
```

## 代码质量

### 类型检查（可选）

```bash
# 安装 mypy
uv add --dev mypy

# 运行类型检查
uv run mypy core/ utils/ scripts/
```

### 代码格式化（可选）

```bash
# 使用 ruff 格式化
uv add --dev ruff
uv run ruff format .

# 使用 ruff 检查
uv run ruff check .
```

## 持续集成建议

### GitHub Actions 示例

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest
```

## 常见问题

### Q: 测试运行很慢怎么办？

A: 
1. 使用 `-n auto` 自动检测 CPU 核心数
2. 使用 `-k` 筛选特定测试
3. 检查是否有耗时的 fixtures，考虑使用 `scope="module"` 或 `scope="session"`

### Q: 覆盖率不准确？

A:
1. 确保 `--cov` 参数包含所有需要统计的模块
2. 使用 `pragma: no cover` 标记不需要测试的代码
3. 检查是否有代码在导入时执行

### Q: 集成测试失败？

A:
1. 确保端口未被占用
2. 确保数据文件已生成
3. 单独运行集成测试查看详细错误

### Q: 如何 mock 外部依赖？

A:
```python
# 使用 pytest-mock
import pytest

def test_with_mock(mocker):
    mock_request = mocker.patch('requests.get')
    mock_request.return_value.status_code = 200
    # 测试代码...
```

## 扩展方向

- **压力测试**：模拟大量客户端并发
- **故障注入**：模拟网络分区、节点宕机
- **性能基准**：记录训练时间、通信开销
- **端到端测试**：完整训练流程验证
