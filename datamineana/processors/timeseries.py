# datamineana/processors/timeseries.py
"""
一、类定位
继承 BaseProcessor，属于时间序列专用预处理处理器，针对时序场景做数据预处理与时序特征工程。每次操作自动生成标准化处理报告，同时向下兼容旧代码的 last_report 读取逻辑。
二、四个核心方法功能
prepare_time_index 时间索引预处理
解析时间字段、自动转换为 datetime 类型，可过滤无效时间值、按时间升序排序，也能将时间列设为数据表索引；统计无效时间数据条数，是所有时序操作的前置准备步骤。
resample 时序重采样
支持按小时、日、周、月等时间粒度聚合重采样，提供均值、求和、最值、中位数、计数等常用聚合方式；要求数据必须是时间索引，用来做高频数据降采样、低频数据规整对齐。
add_lag_features 构造滞后特征
对指定数值列生成多阶滞后特征（如 lag1、lag2），把历史时刻数据作为新特征，常用于时序预测建模；记录所有新增的滞后字段名称。
add_rolling_features 构造滚动窗口特征
设置滑动窗口大小，对数值列计算窗口内均值、标准差、最大最小、求和等统计量，生成滚动统计特征，捕捉时序局部变化规律，会记录所有新建的滚动特征列。
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .base import BaseProcessor
from .utils import ensure_tabular, numeric_columns, safe_columns, to_dataframe, ProcessReport
from datamineana.dataobject import TabularData


class TimeSeriesProcessor(BaseProcessor):
    """
    时间序列预处理模块。

    支持：
    1. 时间列解析和排序
    2. 重采样
    3. 时间序列缺失值填充
    4. 滞后特征
    5. 滚动窗口特征
    """

    name = "time_series_processor"

    def prepare_time_index(
            self,
            data: TabularData,
            *,
            time_column: str,
            sort: bool = True,
            drop_invalid_time: bool = True,
            set_index: bool = False,
            name: Optional[str] = None,
    ) -> TabularData:
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        if time_column not in result.columns:
            raise ValueError(f"时间列不存在: {time_column}")

        # 新版体系：标准报告创建
        report = self._new_report(
            step="prepare_time_index",
            params={
                "time_column": time_column,
                "sort": sort,
                "drop_invalid_time": drop_invalid_time,
                "set_index": set_index,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        result[time_column] = pd.to_datetime(result[time_column], errors="coerce")
        invalid_count = int(result[time_column].isna().sum())

        if drop_invalid_time:
            result = result.dropna(subset=[time_column])

        if sort:
            result = result.sort_values(time_column)

        if set_index:
            result = result.set_index(time_column)

        # 新版体系：报告指标与收尾
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "invalid_time_count": invalid_count,
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="timeseries",
            method="prepare_time_index",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("invalid_time_count", invalid_count)

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def resample(
            self,
            data: TabularData,
            *,
            time_column: Optional[str] = None,
            rule: str = "D",
            agg: str = "mean",
            name: Optional[str] = None,
    ) -> TabularData:
        """
        时间序列重采样。

        rule 示例：
        - "D": 日
        - "W": 周
        - "M": 月
        - "H": 小时

        agg:
        - mean, sum, max, min, count, median
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        # 新版体系：标准报告创建
        report = self._new_report(
            step="resample",
            params={
                "time_column": time_column,
                "rule": rule,
                "agg": agg,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        if time_column:
            if time_column not in result.columns:
                raise ValueError(f"时间列不存在: {time_column}")
            result[time_column] = pd.to_datetime(result[time_column], errors="coerce")
            result = result.dropna(subset=[time_column])
            result = result.set_index(time_column)

        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError("重采样要求 DatetimeIndex，或者指定 time_column")

        numeric = result.select_dtypes(include="number")

        if agg == "mean":
            out = numeric.resample(rule).mean()
        elif agg == "sum":
            out = numeric.resample(rule).sum()
        elif agg == "max":
            out = numeric.resample(rule).max()
        elif agg == "min":
            out = numeric.resample(rule).min()
        elif agg == "count":
            out = numeric.resample(rule).count()
        elif agg == "median":
            out = numeric.resample(rule).median()
        else:
            raise ValueError(f"未知聚合方式: {agg}")

        out = out.reset_index()

        # 新版体系：报告指标与收尾
        report.after_shape = (int(out.shape[0]), int(out.shape[1]))
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="timeseries",
            method="resample",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(out.shape[0]), int(out.shape[1])),
            materialized=True,
        )

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, out, report)
        if name is not None:
            result_data.name = name
        return result_data

    def add_lag_features(
            self,
            data: TabularData,
            *,
            columns: Optional[Iterable[str]] = None,
            lags: Iterable[int] = (1,),
            name: Optional[str] = None,
    ) -> TabularData:
        """
        添加滞后特征。
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        target_columns = safe_columns(result, columns) if columns else numeric_columns(result)

        # 新版体系：标准报告创建
        report = self._new_report(
            step="add_lag_features",
            params={
                "columns": target_columns,
                "lags": list(lags),
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        created = []
        for col in target_columns:
            for lag in lags:
                new_col = f"{col}_lag_{lag}"
                result[new_col] = result[col].shift(lag)
                created.append(new_col)

        # 新版体系：报告指标与收尾
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "created_columns": created,
            "created_count": len(created),
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="timeseries",
            method="add_lag_features",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("created_columns", created)

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def add_rolling_features(
            self,
            data: TabularData,
            *,
            columns: Optional[Iterable[str]] = None,
            windows: Iterable[int] = (3, 7),
            funcs: Iterable[str] = ("mean", "std"),
            name: Optional[str] = None,
    ) -> TabularData:
        """
        添加滚动窗口特征。
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        target_columns = safe_columns(result, columns) if columns else numeric_columns(result)

        # 新版体系：标准报告创建
        report = self._new_report(
            step="add_rolling_features",
            params={
                "columns": target_columns,
                "windows": list(windows),
                "funcs": list(funcs),
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        created = []
        for col in target_columns:
            for window in windows:
                roll = result[col].rolling(window=window)
                for func in funcs:
                    new_col = f"{col}_roll_{window}_{func}"

                    if func == "mean":
                        result[new_col] = roll.mean()
                    elif func == "std":
                        result[new_col] = roll.std()
                    elif func == "min":
                        result[new_col] = roll.min()
                    elif func == "max":
                        result[new_col] = roll.max()
                    elif func == "sum":
                        result[new_col] = roll.sum()
                    else:
                        raise ValueError(f"未知 rolling 函数: {func}")

                    created.append(new_col)

        # 新版体系：报告指标与收尾
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "created_columns": created,
            "created_count": len(created),
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="timeseries",
            method="add_rolling_features",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("created_columns", created)

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def _wrap_result(self, source: TabularData, df: pd.DataFrame, report) -> TabularData:
        metadata = dict(source.metadata or {})
        process_reports = list(metadata.get("process_reports", []))
        process_reports.append(report.to_dict())
        metadata["process_reports"] = process_reports

        return TabularData.from_dataframe(
            df,
            name=source.name,
            description=source.description,
            source_path=source.source_path,
            source_type=source.source_type,
            file_type=source.file_type,
            metadata=metadata,
            processed_by=self.name,
        )
