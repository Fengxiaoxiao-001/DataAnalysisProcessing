# datamineana/processors/transformation.py

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, cast

import numpy as np
import pandas as pd

from datamineana.dataobject.DataSelf import TabularData
from .report import ProcessReport
from .utils import (
    ensure_tabular,
    make_tabular_like,
    numeric_columns,
    safe_columns,
    to_dataframe,
)


def _ensure_series(obj: Any) -> pd.Series:
    """
    保证输入是 Series。
    如果因为重复列名导致拿到 DataFrame，则取第一列。
    """
    if isinstance(obj, pd.DataFrame):
        return cast(pd.Series, obj.iloc[:, 0])
    return cast(pd.Series, obj)


def _dt_part(series: pd.Series, attr: str) -> pd.Series:
    """
    安全获取 datetime accessor 的字段，避免 PyCharm / pandas-stubs
    把 dt.year、dt.month 等识别成 property。
    """
    return cast(pd.Series, getattr(series.dt, attr))


class DataTransformer:
    """
    数据变换和特征工程模块。

    支持：
    1. 归一化
    2. 标准化
    3. 鲁棒缩放
    4. 对数变换
    5. 离散化
    6. One-Hot 编码
    7. 日期时间特征工程
    """
    name = "data_transformer"

    def __init__(self):
        self.last_report: Optional[ProcessReport] = None

    def scale(
            self,
            data: TabularData,
            *,
            method: str = "standard",
            columns: Optional[Iterable[str]] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        数值缩放。

        method:
        - minmax
        - standard
        - robust
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        target_columns = safe_columns(result, columns) if columns else numeric_columns(result)

        report = ProcessReport(
            module="transformation",
            method="scale",
            params={"method": method, "columns": target_columns},
            before_shape=df.shape,
            materialized=True,
        )

        params = {}

        for col in target_columns:
            s = result[col]

            if method == "minmax":
                min_v = s.min()
                max_v = s.max()
                denom = max_v - min_v
                if denom == 0:
                    result[col] = 0
                else:
                    result[col] = (s - min_v) / denom
                params[col] = {"min": float(min_v), "max": float(max_v)}

            elif method == "standard":
                mean = s.mean()
                std = s.std()
                if std == 0:
                    result[col] = 0
                else:
                    result[col] = (s - mean) / std
                params[col] = {"mean": float(mean), "std": float(std)}

            elif method == "robust":
                median = s.median()
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    result[col] = 0
                else:
                    result[col] = (s - median) / iqr
                params[col] = {"median": float(median), "iqr": float(iqr)}

            else:
                raise ValueError(f"未知缩放方法: {method}")

        report.after_shape = result.shape
        report.add_metric("scale_params", params)

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataTransformer.scale",
            report=report,
        )

    def log_transform(
            self,
            data: TabularData,
            *,
            columns: Optional[Iterable[str]] = None,
            offset: float = 1.0,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        对数变换，默认 log1p 风格。
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        target_columns = safe_columns(result, columns) if columns else numeric_columns(result)

        report = ProcessReport(
            module="transformation",
            method="log_transform",
            params={"columns": target_columns, "offset": offset},
            before_shape=df.shape,
            materialized=True,
        )

        skipped = []

        for col in target_columns:
            min_v = result[col].min()
            if min_v + offset <= 0:
                skipped.append(col)
                continue
            result[col] = np.log(result[col] + offset)

        report.after_shape = result.shape
        report.add_metric("skipped_columns", skipped)

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataTransformer.log_transform",
            report=report,
        )

    def binning(
            self,
            data: TabularData,
            *,
            column: str,
            bins,
            labels=None,
            new_column: Optional[str] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        连续变量离散化。

        bins 可以是整数，也可以是边界列表。
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        if column not in result.columns:
            raise ValueError(f"列不存在: {column}")

        report = ProcessReport(
            module="transformation",
            method="binning",
            params={
                "column": column,
                "bins": bins,
                "labels": labels,
                "new_column": new_column,
            },
            before_shape=df.shape,
            materialized=True,
        )

        output_col = new_column or f"{column}_bin"
        result[output_col] = pd.cut(result[column], bins=bins, labels=labels)

        report.after_shape = result.shape
        report.add_metric("output_column", output_col)
        report.add_metric("value_counts", result[output_col].value_counts(dropna=False).to_dict())

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataTransformer.binning",
            report=report,
        )

    def one_hot_encode(
            self,
            data: TabularData,
            *,
            columns: Iterable[str],
            drop_first: bool = False,
            dummy_na: bool = False,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        One-Hot 编码。
        """
        ensure_tabular(data)
        df = to_dataframe(data)

        target_columns = safe_columns(df, columns)

        report = ProcessReport(
            module="transformation",
            method="one_hot_encode",
            params={
                "columns": target_columns,
                "drop_first": drop_first,
                "dummy_na": dummy_na,
            },
            before_shape=df.shape,
            materialized=True,
        )

        result = pd.get_dummies(
            df,
            columns=target_columns,
            drop_first=drop_first,
            dummy_na=dummy_na,
        )

        report.after_shape = result.shape
        report.add_metric("created_columns", [c for c in result.columns if c not in df.columns])

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataTransformer.one_hot_encode",
            report=report,
        )

    def datetime_features(
            self,
            data: TabularData,
            *,
            column: str,
            drop_original: bool = False,
            prefix: Optional[str] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        日期时间特征工程。
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        if column not in result.columns:
            raise ValueError(f"列不存在: {column}")

        report = ProcessReport(
            module="transformation",
            method="datetime_features",
            params={
                "column": column,
                "drop_original": drop_original,
                "prefix": prefix,
            },
            before_shape=df.shape,
            materialized=True,
        )

        dt = pd.to_datetime(result[column], errors="coerce")
        p = prefix or column

        result[f"{p}_year"] = _dt_part(dt, "year")
        result[f"{p}_month"] = _dt_part(dt, "month")
        result[f"{p}_day"] = _dt_part(dt, "day")
        result[f"{p}_dayofweek"] = _dt_part(dt, "dayofweek")
        result[f"{p}_hour"] = _dt_part(dt, "hour")

        if drop_original:
            result = result.drop(columns=[column])

        report.after_shape = result.shape
        report.add_metric("created_columns", [c for c in result.columns if c not in df.columns])

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataTransformer.datetime_features",
            report=report,
        )

    def extract_datetime_features(
            self,
            data: TabularData,
            columns: Optional[Sequence[str]] = None,
            *,
            prefix: Optional[str] = None,
            drop_original: bool = False,
    ) -> TabularData:
        df = to_dataframe(data, materialize=True)

        report = self._new_report(
            step="extract_datetime_features",
            params={
                "columns": list(columns) if columns is not None else None,
                "prefix": prefix,
                "drop_original": drop_original,
            },
            before_shape=df.shape,
            materialized=True,
        )

        result = df.copy()
        target_columns = safe_columns(result, columns)

        created_columns: List[str] = []
        failed_columns: List[str] = []

        for col in target_columns:
            try:
                source = _ensure_series(result[col])
                dt = cast(pd.Series, pd.to_datetime(source, errors="coerce"))

                p = prefix or col

                year_col = f"{p}_year"
                month_col = f"{p}_month"
                day_col = f"{p}_day"
                dayofweek_col = f"{p}_dayofweek"
                hour_col = f"{p}_hour"
                is_month_start_col = f"{p}_is_month_start"
                is_month_end_col = f"{p}_is_month_end"

                result[year_col] = _dt_part(dt, "year")
                result[month_col] = _dt_part(dt, "month")
                result[day_col] = _dt_part(dt, "day")
                result[dayofweek_col] = _dt_part(dt, "dayofweek")
                result[hour_col] = _dt_part(dt, "hour")
                result[is_month_start_col] = _dt_part(dt, "is_month_start")
                result[is_month_end_col] = _dt_part(dt, "is_month_end")

                created_columns.extend([
                    year_col,
                    month_col,
                    day_col,
                    dayofweek_col,
                    hour_col,
                    is_month_start_col,
                    is_month_end_col,
                ])

                if drop_original:
                    result = result.drop(columns=[col])

            except Exception as e:
                failed_columns.append(f"{col}: {repr(e)}")

        report.after_shape = result.shape
        report.statistics = {
            "created_columns": created_columns,
            "created_count": len(created_columns),
            "failed_columns": failed_columns,
            "failed_count": len(failed_columns),
        }
        report.success = len(failed_columns) == 0
        report.finish(success=len(failed_columns) == 0)

        return make_tabular_like(
            data,
            result,
            processed_by=self.name,
            report=report,
        )
