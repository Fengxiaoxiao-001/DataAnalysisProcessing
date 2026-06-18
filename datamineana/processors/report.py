# datamineana/processors/report.py
"""
一、作用
这是项目早期旧版本的处理报告数据类，用来记录每一次数据预处理操作的执行详情，方便追溯处理参数、数据变化、指标、警告日志，支持序列化字典用于存储、前端展示、问题排查。
二、核心字段说明
业务标识
module：所属模块名（如cleaning、reduction）
method：具体执行的方法名（如select_features、pca）
params：本次调用传入的所有参数
数据变化信息
before_shape：处理前数据集行列尺寸
after_shape：处理后数据集行列尺寸
materialized：是否触发真实数据加载（懒加载场景使用）
日志与指标
metrics：业务统计指标字典（删除列、内存节省量、方差、相关系数等）
warnings：警告信息列表
messages：普通提示消息列表
时间戳
created_at：报告生成的本地时间（ISO 格式化字符串）
三、内置方法
add_metric(key, value)：添加统计指标，存入metrics，支持链式调用
add_warning(message)：追加警告日志
add_message(message)：追加普通提示信息
to_dict()：把对象转为标准字典，方便存入数据集元数据、持久化存储、接口返回
四、和你新版 base 里 ProcessReport 的关系
这是旧版格式：用 module + method 定位操作，指标存在 metrics；
新版 ProcessReport（base.py）：用 processor + step 定位，指标存在 statistics，新增执行耗时、成功状态、起止时间，能力更完善；
你在旧代码（DataReducer）里用的就是这个旧类，新版处理器（DataCleaner、DataIntegrator 等）已经切换到 base 里的新报告类；
之前给新报告加的add_metric方法，就是为了兼容旧代码调用习惯，同时把指标存到新类的statistics里，实现新旧报告平滑兼容。
五、现存问题
项目内同时存在两份ProcessReport定义（report.py旧版、base.py新版），容易引发类型冲突、导入混乱，后续建议：
统一从base.py导入ProcessReport；
逐步把DataReducer这类旧处理器改成继承BaseProcessor，接入新标准报告体系；
最终可废弃report.py里的旧版本类。
"""
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
