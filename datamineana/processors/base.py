# datamineana/processors/base.py
"""
# 一、`ProcessReport` 作用
用于**记录单次数据处理步骤的全量运行日志与统计信息**，实现处理过程可追溯：
1. 基础标识：记录当前处理器名称、方法步骤名、调用入参、处理前后数据集形状；
2. 运行状态：标记执行是否成功、异常提示、警告列表、执行起止时间与耗时；
3. 业务统计：通过 `statistics` 字典存放业务指标（删除行数、缺失值数量等）；
4. 兼容方法 `add_metric`：适配旧代码写入指标的调用方式，最终数据存入 `statistics`；
5. `to_dict()`：将日志转为字典，方便序列化存入数据集元数据、日志持久化。

# 二、`BaseProcessor` 作用
所有数据处理器（清洗、合并、划分等）的**通用父类**，统一规范所有处理器的日志能力：
1. 内置两个属性：`reports` 保存当前处理器所有步骤的全部日志；`last_report` 记录最近一次处理日志，向下兼容旧业务读取；
2. `_new_report()`：统一创建标准日志对象，自动绑定处理器信息，同时存入历史日志列表与最新日志属性；
3. 对外提供方法：批量获取所有日志字典、获取上一步日志字典，方便外部查看处理过程。

# 三、整体设计目的
统一所有数据处理模块的日志规范，不用每个处理器重复写日志创建、存储、查询逻辑；同时兼容新旧两套代码的报告调用方式，实现全链路数据处理过程可追溯、可排查。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import time

Shape = Tuple[int | None, int | None]
ShapeInfo = Shape | Dict[str, Shape] | List[Shape] | None


@dataclass
class ProcessReport:
    step: str
    processor: str
    params: Dict[str, Any] = field(default_factory=dict)

    before_shape: ShapeInfo = None
    after_shape: ShapeInfo = None

    materialized: bool = False
    success: bool = True
    message: str = ""

    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    elapsed_seconds: Optional[float] = None

    def finish(self, success: bool = True, message: str = "") -> "ProcessReport":
        self.success = success
        self.message = message
        self.finished_at = time.time()
        self.elapsed_seconds = self.finished_at - self.started_at
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # 新增：兼容旧版 add_metric 接口
    def add_metric(self, key: str, value: Any) -> None:
        """
        兼容旧版报告接口，将指标存入 statistics 字典。
        旧代码无需修改即可正常调用。
        """
        self.statistics[key] = value


class BaseProcessor:
    name = "base_processor"

    def __init__(self) -> None:
        self.reports: List[ProcessReport] = []
        self.last_report: Optional[ProcessReport] = None

    def _new_report(
            self,
            step: str,
            params: Optional[Dict[str, Any]] = None,
            before_shape: ShapeInfo = None,
            materialized: bool = False,
    ) -> ProcessReport:
        report = ProcessReport(
            step=step,
            processor=self.name,
            params=params or {},
            before_shape=before_shape,
            materialized=materialized,
        )
        self.last_report = report
        self.reports.append(report)
        return report

    def get_reports(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.reports]

    def get_last_report(self) -> Optional[Dict[str, Any]]:
        return None if self.last_report is None else self.last_report.to_dict()
