# datamineana/processors/timeseries.py

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from datamineana.dataobject.DataSelf import TabularData
from .report import ProcessReport
from .utils import ensure_tabular, make_tabular_like, numeric_columns, safe_columns, to_dataframe


class TimeSeriesProcessor:
    """
    时间序列预处理模块。

    支持：
    1. 时间列解析和排序
    2. 重采样
    3. 时间序列缺失值填充
    4. 滞后特征
    5. 滚动窗口特征
    """

    def __init__(self):
        self.last_report: Optional[ProcessReport] = None

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

        report = ProcessReport(
            module="timeseries",
            method="prepare_time_index",
            params={
                "time_column": time_column,
                "sort": sort,
                "drop_invalid_time": drop_invalid_time,
                "set_index": set_index,
            },
            before_shape=df.shape,
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

        report.after_shape = result.shape
        report.add_metric("invalid_time_count", invalid_count)

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="TimeSeriesProcessor.prepare_time_index",
            report=report,
        )

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

        report = ProcessReport(
            module="timeseries",
            method="resample",
            params={
                "time_column": time_column,
                "rule": rule,
                "agg": agg,
            },
            before_shape=df.shape,
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

        report.after_shape = out.shape

        self.last_report = report
        return make_tabular_like(
            data,
            out,
            name=name,
            processed_by="TimeSeriesProcessor.resample",
            report=report,
        )

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

        report = ProcessReport(
            module="timeseries",
            method="add_lag_features",
            params={
                "columns": target_columns,
                "lags": list(lags),
            },
            before_shape=df.shape,
            materialized=True,
        )

        created = []

        for col in target_columns:
            for lag in lags:
                new_col = f"{col}_lag_{lag}"
                result[new_col] = result[col].shift(lag)
                created.append(new_col)

        report.after_shape = result.shape
        report.add_metric("created_columns", created)

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="TimeSeriesProcessor.add_lag_features",
            report=report,
        )

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

        report = ProcessReport(
            module="timeseries",
            method="add_rolling_features",
            params={
                "columns": target_columns,
                "windows": list(windows),
                "funcs": list(funcs),
            },
            before_shape=df.shape,
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

        report.after_shape = result.shape
        report.add_metric("created_columns", created)

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="TimeSeriesProcessor.add_rolling_features",
            report=report,
        )
