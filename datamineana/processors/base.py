# datamineana/processors/base.py

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
