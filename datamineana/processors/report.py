# datamineana/processors/report.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ProcessReport:
    """
    单次预处理操作报告。

    用途：
    1. 记录处理前后数据形状
    2. 记录参数
    3. 记录指标变化
    4. 记录警告信息
    5. 方便前端或用户画图分析
    """

    module: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    before_shape: Optional[tuple] = None
    after_shape: Optional[tuple] = None

    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    materialized: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def add_metric(self, key: str, value: Any) -> "ProcessReport":
        self.metrics[key] = value
        return self

    def add_warning(self, message: str) -> "ProcessReport":
        self.warnings.append(message)
        return self

    def add_message(self, message: str) -> "ProcessReport":
        self.messages.append(message)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "method": self.method,
            "params": self.params,
            "before_shape": self.before_shape,
            "after_shape": self.after_shape,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "messages": self.messages,
            "materialized": self.materialized,
            "created_at": self.created_at,
        }
