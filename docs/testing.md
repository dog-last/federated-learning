# 测试说明

本文档说明如何运行测试和进行开发。

## 测试结构

```
tests/
├── conftest.py              # pytest 配置和 fixtures
├── unit/                    # 单元测试
│   ├── test_aggregator.py   # 聚合算法测试
│   ├── test_client.py       # 客户端测试
│   ├── test_data.py         # 数据加载测试
│   ├── test_exceptions.py   # 异常处理测试
│   ├── test_model.py        # 模型测试
│   ├── test_model_and_data.py
│   ├── test_p2p.py          # P2P 测试
│   ├── test_protocol.py     # 协议测试
│   ├── test_transport.py    # 传输层测试
│   └── test_utils.py        # 工具函数测试
└── integration/             # 集成测试
    ├── test_centralized.py  # 中心化模式集成测试
    └── test_decentralized.py # 去中心化模式集成测试
```

## 运行测试

### 运行所有测试

```bash
uv run pytest
```

### 运行特定测试文件

```bash
# 运行单元测试
uv run pytest tests/unit/

# 运行集成测试
uv run pytest tests/integration/

# 运行特定测试文件
uv run pytest tests/unit/test_model.py
```

### 运行特定测试函数

```bash
# 运行特定测试函数
uv run pytest tests/unit/test_model.py::test_model_training

# 使用 -k 按名称匹配
uv run pytest -k "test_model"
```

### 详细输出

```bash
# 显示详细输出
uv run pytest -v

# 显示更详细的输出（包括 print 语句）
uv run pytest -v -s
```

### 测试覆盖率

```bash
# 生成覆盖率报告
uv run pytest --cov=src --cov-report=term

# 生成 HTML 覆盖率报告
uv run pytest --cov=src --cov-report=html

# 查看 HTML 报告
open htmlcov/index.html
```

### 并行运行测试

```bash
# 使用 pytest-xdist 并行运行
uv run pytest -n auto
```

## 开发指南

### 添加新测试

#### 单元测试示例

```python
# tests/unit/test_example.py
import pytest
from src.module import MyClass


class TestMyClass:
    """Test cases for MyClass."""

    def test_initialization(self):
        """Test object initialization."""
        obj = MyClass()
        assert obj is not None

    def test_method(self):
        """Test a specific method."""
        obj = MyClass()
        result = obj.my_method(5)
        assert result == expected_value

    @pytest.mark.parametrize("input,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
    ])
    def test_with_parameters(self, input, expected):
        """Test with multiple parameters."""
        obj = MyClass()
        result = obj.square(input)
        assert result == expected
```

#### 集成测试示例

```python
# tests/integration/test_feature.py
import pytest
from src.server.federated_server import FederatedServer
from src.client.federated_client import FederatedClient


@pytest.mark.integration
def test_end_to_end_training():
    """Test complete training workflow."""
    # Setup
    server = FederatedServer()
    client = FederatedClient()

    # Execute
    server.start(port=9000)
    client.connect("127.0.0.1", 9000)

    # Verify
    assert client.is_connected

    # Cleanup
    client.disconnect()
    server.stop()
```

### 使用 Fixtures

```python
# tests/conftest.py
import pytest
from src.model.simple_cnn import SimpleCNN


@pytest.fixture
def model():
    """Create a model instance for testing."""
    return SimpleCNN(input_channels=1, num_classes=10)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    import torch
    return torch.randn(10, 1, 28, 28), torch.randint(0, 10, (10,))
```

使用 fixture：

```python
def test_model_forward(model, sample_data):
    """Test model forward pass."""
    x, y = sample_data
    output = model(x)
    assert output.shape == (10, 10)
```

### 标记测试

```python
import pytest


@pytest.mark.slow
def test_long_running():
    """A slow test that should be run separately."""
    pass


@pytest.mark.integration
def test_integration():
    """An integration test."""
    pass
```

运行特定标记的测试：

```bash
# 只运行慢测试
uv run pytest -m slow

# 排除慢测试
uv run pytest -m "not slow"

# 只运行集成测试
uv run pytest -m integration
```

## 代码质量检查

### 代码格式化

```bash
# 格式化所有代码
uv run ruff format .

# 检查格式化（不修改）
uv run ruff format --check .
```

### 代码检查

```bash
# 运行所有检查
uv run ruff check .

# 自动修复问题
uv run ruff check --fix .

# 检查特定目录
uv run ruff check src/
```

### 类型检查

```bash
# 运行 mypy 类型检查
uv run mypy src/

# 生成类型覆盖率报告
uv run mypy src/ --html-report mypy_report
```

## 调试技巧

### 使用 pdb

```python
def test_something():
    """Test with debugging."""
    result = some_function()
    import pdb; pdb.set_trace()  # 设置断点
    assert result == expected
```

### 使用 pytest 的 --pdb

```bash
# 测试失败时自动进入 pdb
uv run pytest --pdb

# 在第一个失败时进入 pdb
uv run pytest --pdb -x
```

### 日志输出

```python
import logging

logger = logging.getLogger(__name__)

def test_with_logging():
    """Test with log output."""
    logger.info("Starting test")
    result = do_something()
    logger.debug(f"Result: {result}")
    assert result
```

运行测试时显示日志：

```bash
uv run pytest --log-cli-level=DEBUG
```

## 持续集成

### GitHub Actions 示例

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.13"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install uv
      run: curl -LsSf https://astral.sh/uv/install.sh | sh

    - name: Install dependencies
      run: uv sync

    - name: Run tests
      run: uv run pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 性能测试

### 基准测试

```python
# tests/performance/test_benchmark.py
import pytest
import time


def test_training_performance():
    """Benchmark training performance."""
    start = time.time()

    # Run training
    model.train(dataloader, epochs=1)

    duration = time.time() - start
    assert duration < 60  # Should complete within 60 seconds
```

### 使用 pytest-benchmark

```bash
# 安装 pytest-benchmark
uv add --dev pytest-benchmark

# 运行基准测试
uv run pytest tests/performance/ --benchmark-only
```

## 测试最佳实践

1. **独立性**: 每个测试应该独立运行，不依赖其他测试
2. **可重复性**: 测试结果应该可重复，不受外部环境影响
3. **快速性**: 单元测试应该快速执行
4. **清晰性**: 测试名称应该清晰描述测试内容
5. **覆盖性**: 尽量覆盖所有代码路径

### 测试命名规范

```python
# Good
def test_model_training_with_valid_data():
    pass

def test_model_raises_error_with_invalid_input():
    pass

# Bad
def test1():
    pass

def test_model():
    pass
```

### 断言使用

```python
# Good
assert result == expected
assert len(items) == 3
assert error_message in str(exc_info.value)

# Bad
assert result  # 不明确
assert len(items)  # 不明确
```

## 常见问题

### 1. 测试发现失败

**问题**: pytest 找不到测试

**解决方案**:

```bash
# 检查测试文件命名
# 应该是 test_*.py 或 *_test.py

# 检查测试函数命名
# 应该是 test_*

# 显式指定测试路径
uv run pytest tests/unit/test_model.py -v
```

### 2. 导入错误

**问题**: `ModuleNotFoundError`

**解决方案**:

```bash
# 确保已安装包
uv sync

# 使用 editable 安装
uv pip install -e .

# 检查 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 3. 测试超时

**问题**: 测试运行时间过长

**解决方案**:

```bash
# 设置超时
uv run pytest --timeout=60

# 只运行快速测试
uv run pytest -m "not slow"
```

### 4. 覆盖率不准确

**问题**: 覆盖率报告不准确

**解决方案**:

```bash
# 清除缓存
rm -rf .pytest_cache htmlcov

# 重新运行
uv run pytest --cov=src --cov-report=html
```
