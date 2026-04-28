# 监控日志

> 本文档定义网络状态监控、日志输出和指标统计的规范。

---

## 监控指标

### 轮次指标

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `round_id` | 当前轮次ID | - |
| `total_rounds` | 总轮次数 | - |
| `participating_clients` | 参与客户端数量 | - |
| `timeout_clients` | 超时客户端数量 | - |

### 广播阶段指标

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `broadcast_start_time` | 广播开始时间 | timestamp |
| `broadcast_end_time` | 广播结束时间 | timestamp |
| `broadcast_duration` | 广播耗时 | 秒 |
| `broadcast_payload_size` | 广播数据大小 | 字节 |
| `broadcast_success_count` | 成功发送客户端数 | - |
| `broadcast_fail_count` | 发送失败客户端数 | - |

### 训练阶段指标

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `client_id` | 客户端ID | - |
| `training_start_time` | 训练开始时间 | timestamp |
| `training_end_time` | 训练结束时间 | timestamp |
| `training_duration` | 训练耗时 | 秒 |
| `local_loss` | 本地训练损失 | - |
| `local_accuracy` | 本地训练准确率 | % |
| `num_samples` | 训练样本数 | - |

### 聚合阶段指标

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `collect_start_time` | 收集开始时间 | timestamp |
| `collect_end_time` | 收集结束时间 | timestamp |
| `collect_duration` | 收集总耗时 | 秒 |
| `client_collect_times` | 各客户端收集耗时 | Dict[int, float] |
| `aggregate_duration` | 聚合耗时 | 秒 |
| `aggregate_payload_size` | 聚合后模型大小 | 字节 |

### 网络状态指标

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `connection_events` | 连接建立/断开事件 | - |
| `timeout_events` | 超时事件 | - |
| `retry_count` | 重试次数 | - |
| `bytes_sent` | 发送字节数 | 字节 |
| `bytes_recv` | 接收字节数 | 字节 |

### 最终指标

| 指标名称 | 说明 | 单位 |
|----------|------|------|
| `global_accuracy` | 全局模型准确率 | % |
| `global_loss` | 全局模型损失 | - |
| `convergence_round` | 收敛轮次 | - |
| `total_training_time` | 总训练时间 | 秒 |

---

## 日志规范

### 日志级别

| 级别 | 用途 |
|------|------|
| DEBUG | 详细调试信息 |
| INFO | 正常运行信息 |
| WARNING | 警告信息（可恢复） |
| ERROR | 错误信息（影响功能） |
| CRITICAL | 严重错误（系统崩溃） |

### 日志格式

```
[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s
```

**示例：**
```
[2024-03-28 10:30:15,123] [INFO] [Server] Round 1 started
[2024-03-28 10:30:15,135] [INFO] [Server] Broadcasting model to 3 clients...
[2024-03-28 10:30:15,147] [INFO] [Server] Broadcast complete. Size: 256KB, Time: 12ms
[2024-03-28 10:30:18,234] [INFO] [Client-1] Training complete. Loss: 2.31, Acc: 10.2%, Time: 3.1s
[2024-03-28 10:30:21,456] [INFO] [Client-2] Training complete. Loss: 2.28, Acc: 11.5%, Time: 3.2s
[2024-03-28 10:30:25,789] [WARNING] [Server] Client-3 TIMEOUT after 30s, skipping
[2024-03-28 10:30:25,890] [INFO] [Server] Aggregating 2 client updates...
[2024-03-28 10:30:25,901] [INFO] [Server] Aggregation complete. Global Acc: 12.1%
```

### 日志输出位置

| 日志类型 | 输出位置 | 说明 |
|----------|----------|------|
| 控制台日志 | stdout | 实时显示关键信息 |
| 文件日志 | `logs/fed_{date}.log` | 完整日志记录 |
| 网络事件日志 | `logs/network_{date}.log` | 网络状态专用 |
| 指标日志 | `logs/metrics_{date}.log` | 指标数据专用 |

---

## 日志接口

### Logger接口

```python
# utils/logger.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from core.types import RoundStats

class ILogger(ABC):
    """日志接口"""

    @abstractmethod
    def info(self, msg: str, **kwargs) -> None:
        """记录INFO级别日志"""
        pass

    @abstractmethod
    def warning(self, msg: str, **kwargs) -> None:
        """记录WARNING级别日志"""
        pass

    @abstractmethod
    def error(self, msg: str, **kwargs) -> None:
        """记录ERROR级别日志"""
        pass

    @abstractmethod
    def debug(self, msg: str, **kwargs) -> None:
        """记录DEBUG级别日志"""
        pass

    @abstractmethod
    def log_round(self, stats: RoundStats) -> None:
        """
        记录轮次日志

        Args:
            stats: 轮次统计信息
        """
        pass

    @abstractmethod
    def log_network(
        self,
        event: str,
        client_id: Optional[int] = None,
        size: Optional[int] = None,
        duration: Optional[float] = None,
        success: bool = True
    ) -> None:
        """
        记录网络事件日志

        Args:
            event: 事件类型 (connect|disconnect|send|recv|timeout)
            client_id: 客户端ID
            size: 数据大小（字节）
            duration: 耗时（秒）
            success: 是否成功
        """
        pass

    @abstractmethod
    def log_training(
        self,
        client_id: int,
        round_id: int,
        loss: float,
        accuracy: float,
        duration: float
    ) -> None:
        """
        记录训练日志

        Args:
            client_id: 客户端ID
            round_id: 轮次ID
            loss: 损失值
            accuracy: 准确率
            duration: 训练耗时
        """
        pass
```

### 实现示例

```python
import logging
from datetime import datetime
from pathlib import Path

class FedLogger(ILogger):
    """联邦学习日志实现"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建logger
        self.logger = logging.getLogger("FedLearning")
        self.logger.setLevel(logging.DEBUG)

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)

        # 文件handler
        date_str = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(
            self.log_dir / f"fed_{date_str}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s"
        )
        file_handler.setFormatter(file_format)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, msg: str, **kwargs) -> None:
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        self.logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs) -> None:
        self.logger.debug(msg, **kwargs)

    def log_round(self, stats: RoundStats) -> None:
        self.info(f"[Round {stats.round_id}] Summary:")
        self.info(f"  - Participating clients: {stats.participating_clients}")
        self.info(f"  - Timeout clients: {stats.timeout_clients}")
        self.info(f"  - Broadcast time: {stats.broadcast_time:.3f}s")
        self.info(f"  - Aggregate time: {stats.aggregate_time:.3f}s")
        self.info(f"  - Total time: {stats.total_time:.3f}s")
        self.info(f"  - Global accuracy: {stats.global_accuracy:.2f}%")

    def log_network(
        self,
        event: str,
        client_id: Optional[int] = None,
        size: Optional[int] = None,
        duration: Optional[float] = None,
        success: bool = True
    ) -> None:
        client_str = f"Client-{client_id}" if client_id else "Unknown"
        status = "SUCCESS" if success else "FAILED"

        msg_parts = [f"[Network] {event.upper()} {client_str} {status}"]
        if size is not None:
            msg_parts.append(f"Size: {self._format_size(size)}")
        if duration is not None:
            msg_parts.append(f"Time: {duration*1000:.1f}ms")

        if success:
            self.info(" | ".join(msg_parts))
        else:
            self.warning(" | ".join(msg_parts))

    def log_training(
        self,
        client_id: int,
        round_id: int,
        loss: float,
        accuracy: float,
        duration: float
    ) -> None:
        self.info(
            f"[Round {round_id}] Client-{client_id} training complete. "
            f"Loss: {loss:.4f}, Acc: {accuracy:.2f}%, Time: {duration:.2f}s"
        )

    def _format_size(self, size: int) -> str:
        """格式化字节大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
```

---

## 指标收集

### MetricsCollector接口

```python
# utils/metrics.py

from abc import ABC, abstractmethod
from typing import List, Dict
from core.types import RoundStats

class IMetricsCollector(ABC):
    """指标收集器接口"""

    @abstractmethod
    def record_round(self, stats: RoundStats) -> None:
        """记录轮次统计"""
        pass

    @abstractmethod
    def get_accuracy_history(self) -> List[float]:
        """获取准确率历史"""
        pass

    @abstractmethod
    def get_loss_history(self) -> List[float]:
        """获取损失历史"""
        pass

    @abstractmethod
    def get_round_times(self) -> List[float]:
        """获取各轮耗时"""
        pass

    @abstractmethod
    def get_summary(self) -> Dict:
        """获取汇总统计"""
        pass

    @abstractmethod
    def export(self, path: str) -> None:
        """导出指标到文件"""
        pass

    @abstractmethod
    def plot_accuracy(self, path: str) -> None:
        """绘制准确率曲线"""
        pass

    @abstractmethod
    def plot_loss(self, path: str) -> None:
        """绘制损失曲线"""
        pass
```

### 实现示例

```python
import json
from pathlib import Path
from typing import List, Dict

class MetricsCollector(IMetricsCollector):
    """指标收集器实现"""

    def __init__(self):
        self._accuracy_history: List[float] = []
        self._loss_history: List[float] = []
        self._round_times: List[float] = []
        self._round_stats: List[RoundStats] = []

    def record_round(self, stats: RoundStats) -> None:
        self._round_stats.append(stats)
        self._accuracy_history.append(stats.global_accuracy)
        self._round_times.append(stats.total_time)

    def get_accuracy_history(self) -> List[float]:
        return self._accuracy_history

    def get_loss_history(self) -> List[float]:
        return self._loss_history

    def get_round_times(self) -> List[float]:
        return self._round_times

    def get_summary(self) -> Dict:
        if not self._accuracy_history:
            return {}

        return {
            "total_rounds": len(self._accuracy_history),
            "final_accuracy": self._accuracy_history[-1],
            "best_accuracy": max(self._accuracy_history),
            "avg_round_time": sum(self._round_times) / len(self._round_times),
            "total_time": sum(self._round_times),
            "convergence_round": self._find_convergence(),
        }

    def _find_convergence(self, threshold: float = 0.001) -> int:
        """找到收敛轮次"""
        for i in range(1, len(self._accuracy_history)):
            if abs(self._accuracy_history[i] - self._accuracy_history[i-1]) < threshold:
                return i
        return len(self._accuracy_history)

    def export(self, path: str) -> None:
        data = {
            "accuracy_history": self._accuracy_history,
            "loss_history": self._loss_history,
            "round_times": self._round_times,
            "summary": self.get_summary(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def plot_accuracy(self, path: str) -> None:
        """绘制准确率曲线"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self._accuracy_history) + 1), self._accuracy_history)
            plt.xlabel("Round")
            plt.ylabel("Accuracy (%)")
            plt.title("Global Model Accuracy")
            plt.grid(True)
            plt.savefig(path)
            plt.close()
        except ImportError:
            print("matplotlib not installed, skipping plot")

    def plot_loss(self, path: str) -> None:
        """绘制损失曲线"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self._loss_history) + 1), self._loss_history)
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True)
            plt.savefig(path)
            plt.close()
        except ImportError:
            print("matplotlib not installed, skipping plot")
```

---

## 控制台输出示例

### 正常运行

```
================================================================================
                    Federated Learning System v1.0
================================================================================

[2024-03-28 10:30:00] [INFO] [Server] Starting server on port 9000
[2024-03-28 10:30:00] [INFO] [Server] Waiting for 3 clients to connect...

[2024-03-28 10:30:05] [INFO] [Server] Client-1 connected from 127.0.0.1:50001
[2024-03-28 10:30:07] [INFO] [Server] Client-2 connected from 127.0.0.1:50002
[2024-03-28 10:30:09] [INFO] [Server] Client-3 connected from 127.0.0.1:50003

[2024-03-28 10:30:09] [INFO] [Server] All clients connected. Starting training...

================================================================================
                              Round 1 / 10
================================================================================

[2024-03-28 10:30:09] [INFO] [Server] Broadcasting global model...
[2024-03-28 10:30:09] [INFO] [Network] BROADCAST Client-1 SUCCESS | Size: 256.0KB | Time: 8.2ms
[2024-03-28 10:30:09] [INFO] [Network] BROADCAST Client-2 SUCCESS | Size: 256.0KB | Time: 7.9ms
[2024-03-28 10:30:09] [INFO] [Network] BROADCAST Client-3 SUCCESS | Size: 256.0KB | Time: 8.1ms
[2024-03-28 10:30:09] [INFO] [Server] Broadcast complete. Total time: 24.2ms

[2024-03-28 10:30:12] [INFO] [Round 1] Client-1 training complete. Loss: 2.3012, Acc: 10.2%, Time: 3.1s
[2024-03-28 10:30:13] [INFO] [Round 1] Client-2 training complete. Loss: 2.2987, Acc: 11.5%, Time: 3.2s
[2024-03-28 10:30:14] [INFO] [Round 1] Client-3 training complete. Loss: 2.3056, Acc: 9.8%, Time: 3.0s

[2024-03-28 10:30:14] [INFO] [Server] Collecting client updates...
[2024-03-28 10:30:14] [INFO] [Network] RECV Client-1 SUCCESS | Size: 256.0KB | Time: 5.1ms
[2024-03-28 10:30:14] [INFO] [Network] RECV Client-2 SUCCESS | Size: 256.0KB | Time: 4.8ms
[2024-03-28 10:30:14] [INFO] [Network] RECV Client-3 SUCCESS | Size: 256.0KB | Time: 5.3ms

[2024-03-28 10:30:14] [INFO] [Server] Aggregating 3 client updates...
[2024-03-28 10:30:14] [INFO] [Server] Aggregation complete. Time: 15.2ms

--------------------------------------------------------------------------------
[Round 1] Summary:
  - Participating clients: 3
  - Timeout clients: 0
  - Broadcast time: 0.024s
  - Aggregate time: 0.015s
  - Total time: 5.2s
  - Global accuracy: 12.34%
--------------------------------------------------------------------------------
```

### 超时处理

```
================================================================================
                              Round 5 / 10
================================================================================

[2024-03-28 10:35:00] [INFO] [Server] Broadcasting global model...
[2024-03-28 10:35:00] [INFO] [Server] Broadcast complete. Total time: 24.5ms

[2024-03-28 10:35:03] [INFO] [Round 5] Client-1 training complete. Loss: 1.2345, Acc: 56.2%, Time: 3.1s
[2024-03-28 10:35:04] [INFO] [Round 5] Client-2 training complete. Loss: 1.2567, Acc: 54.8%, Time: 3.2s

[2024-03-28 10:35:30] [WARNING] [Server] Client-3 TIMEOUT after 30s, skipping

[2024-03-28 10:35:30] [INFO] [Server] Aggregating 2 client updates (1 timeout)...
[2024-03-28 10:35:30] [INFO] [Server] Aggregation complete. Time: 12.8ms

--------------------------------------------------------------------------------
[Round 5] Summary:
  - Participating clients: 2
  - Timeout clients: [3]
  - Broadcast time: 0.025s
  - Aggregate time: 0.013s
  - Total time: 30.2s
  - Global accuracy: 58.76%
--------------------------------------------------------------------------------
```

---

## 性能目标验证

### 准确率曲线

```
Final Accuracy: 86.5% ✓ (Target: ≥85%)

Accuracy Progress:
Round  1: 12.34%
Round  2: 25.67%
Round  3: 42.18%
Round  4: 58.92%
Round  5: 68.45%
Round  6: 75.23%
Round  7: 80.12%
Round  8: 83.56%
Round  9: 85.34%
Round 10: 86.52%

Convergence: Round 9
```

### 网络性能

```
Network Statistics:
  - Avg broadcast time: 25.3ms
  - Avg collect time: 15.2ms
  - Avg round time: 5.4s
  - Total training time: 54.2s
  - Timeout rate: 5% (3/60 rounds)
```
