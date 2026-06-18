# datamineana/processors/processor.py

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from datamineana.dataobject.DataSelf import TabularData

from .cleaning import DataCleaner
from .integration import DataIntegrator
from .transformation import DataTransformer
from .reduction import DataReducer
from .splitting import DataSplitter
from .timeseries import TimeSeriesProcessor
from .base import ProcessReport
from .utils import ensure_tabular


class DataPreprocessor:
    """
    数据预处理唯一对外主入口。

    设计原则：
    1. 输入至少包含一个 TabularData
    2. 输出至少包含一个 TabularData
    3. 每一步都可以独立运行
    4. 每一步都有报告
    5. 内部按模块分工，外部统一调用
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.integrator = DataIntegrator()
        self.transformer = DataTransformer()
        self.reducer = DataReducer()
        self.splitter = DataSplitter()
        self.timeseries = TimeSeriesProcessor()

        self.reports: List[ProcessReport] = []

    def _collect_report(self, module: Any) -> None:
        report = getattr(module, "last_report", None)
        if report is not None:
            self.reports.append(report)

    def get_reports(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.reports]

    def last_report(self) -> Optional[Dict[str, Any]]:
        if not self.reports:
            return None
        return self.reports[-1].to_dict()

    def get_module_reports(self, module_name: str) -> list[dict[str, Any]]:
        """获取指定子处理器的全部历史报告"""
        module_map = {
            "cleaner": self.cleaner,
            "integrator": self.integrator,
            "transformer": self.transformer,
            "reducer": self.reducer,
            "splitter": self.splitter,
            "timeseries": self.timeseries,
        }
        module = module_map.get(module_name)
        return module.get_reports() if module else []

    def clear_reports(self) -> None:
        self.reports.clear()

    # =========================
    # 数据清洗
    # =========================

    def handle_missing(self, data: TabularData, **kwargs) -> TabularData:
        result = self.cleaner.handle_missing(data, **kwargs)
        self._collect_report(self.cleaner)
        return result

    def drop_duplicates(self, data: TabularData, **kwargs) -> TabularData:
        result = self.cleaner.drop_duplicates(data, **kwargs)
        self._collect_report(self.cleaner)
        return result

    def handle_outliers(self, data: TabularData, **kwargs) -> TabularData:
        result = self.cleaner.handle_outliers(data, **kwargs)
        self._collect_report(self.cleaner)
        return result

    def normalize_text(self, data: TabularData, **kwargs) -> TabularData:
        result = self.cleaner.normalize_text(data, **kwargs)
        self._collect_report(self.cleaner)
        return result

    def convert_types(self, data: TabularData, **kwargs) -> TabularData:
        result = self.cleaner.convert_dtypes_by_mapping(data, **kwargs)
        self._collect_report(self.cleaner)
        return result

    def correct_values(self, data: TabularData, **kwargs) -> TabularData:
        result = self.cleaner.correct_values(data, **kwargs)
        self._collect_report(self.cleaner)
        return result

    # =========================
    # 数据集成
    # =========================

    def concat(self, datasets: Sequence[TabularData], **kwargs) -> TabularData:
        if not datasets:
            raise ValueError("datasets 不能为空")
        for data in datasets:
            ensure_tabular(data)

        result = self.integrator.concat(datasets, **kwargs)
        self._collect_report(self.integrator)
        return result

    def merge(self, left: TabularData, right: TabularData, **kwargs) -> TabularData:
        result = self.integrator.merge(left, right, **kwargs)
        self._collect_report(self.integrator)
        return result

    def remove_redundant_columns(self, data: TabularData, **kwargs) -> TabularData:
        result = self.integrator.remove_redundant_columns(data, **kwargs)
        self._collect_report(self.integrator)
        return result

    # =========================
    # 数据变换
    # =========================

    def scale(self, data: TabularData, **kwargs) -> TabularData:
        result = self.transformer.scale(data, **kwargs)
        self._collect_report(self.transformer)
        return result

    def log_transform(self, data: TabularData, **kwargs) -> TabularData:
        result = self.transformer.log_transform(data, **kwargs)
        self._collect_report(self.transformer)
        return result

    def binning(self, data: TabularData, **kwargs) -> TabularData:
        result = self.transformer.binning(data, **kwargs)
        self._collect_report(self.transformer)
        return result

    def one_hot_encode(self, data: TabularData, **kwargs) -> TabularData:
        result = self.transformer.one_hot_encode(data, **kwargs)
        self._collect_report(self.transformer)
        return result

    def datetime_features(self, data: TabularData, **kwargs) -> TabularData:
        result = self.transformer.datetime_features(data, **kwargs)
        self._collect_report(self.transformer)
        return result

    # =========================
    # 数据规约
    # =========================

    def select_features(self, data: TabularData, **kwargs) -> TabularData:
        result = self.reducer.select_features(data, **kwargs)
        self._collect_report(self.reducer)
        return result

    def pca(self, data: TabularData, **kwargs) -> TabularData:
        result = self.reducer.pca(data, **kwargs)
        self._collect_report(self.reducer)
        return result

    def sample(self, data: TabularData, **kwargs) -> TabularData:
        result = self.reducer.sample(data, **kwargs)
        self._collect_report(self.reducer)
        return result

    def optimize_memory(self, data: TabularData, **kwargs) -> TabularData:
        result = self.reducer.optimize_memory(data, **kwargs)
        self._collect_report(self.reducer)
        return result

    # =========================
    # 数据划分
    # =========================

    def train_valid_test_split(self, data: TabularData, **kwargs) -> Tuple[TabularData, TabularData, TabularData]:
        result = self.splitter.random_split(data, **kwargs)
        self._collect_report(self.splitter)
        return result

    # =========================
    # 时间序列
    # =========================

    def prepare_time_index(self, data: TabularData, **kwargs) -> TabularData:
        result = self.timeseries.prepare_time_index(data, **kwargs)
        self._collect_report(self.timeseries)
        return result

    def resample(self, data: TabularData, **kwargs) -> TabularData:
        result = self.timeseries.resample(data, **kwargs)
        self._collect_report(self.timeseries)
        return result

    def add_lag_features(self, data: TabularData, **kwargs) -> TabularData:
        result = self.timeseries.add_lag_features(data, **kwargs)
        self._collect_report(self.timeseries)
        return result

    def add_rolling_features(self, data: TabularData, **kwargs) -> TabularData:
        result = self.timeseries.add_rolling_features(data, **kwargs)
        self._collect_report(self.timeseries)
        return result

    # =========================
    # 简单流水线
    # =========================

    def run_pipeline(
            self,
            data: TabularData,
            steps: Sequence[Mapping[str, Any]],
    ) -> TabularData:
        """
        按步骤执行预处理流水线。

        示例：
        steps = [
            {"method": "handle_missing", "params": {"strategy": "median"}},
            {"method": "drop_duplicates", "params": {}},
            {"method": "handle_outliers", "params": {"method": "iqr", "action": "cap"}},
            {"method": "scale", "params": {"method": "standard"}},
        ]
        """
        ensure_tabular(data)

        current = data

        for step in steps:
            method_name = step.get("method")
            params = dict(step.get("params", {}))

            if not method_name:
                raise ValueError("流水线步骤缺少 method")

            method = getattr(self, method_name, None)

            if method is None or not callable(method):
                raise ValueError(f"未知预处理方法: {method_name}")

            current = method(current, **params)

            if not isinstance(current, TabularData):
                raise TypeError(
                    f"流水线方法 {method_name} 必须返回 TabularData，实际返回: {type(current)}"
                )

        return current
