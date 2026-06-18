# datamineana/processors/transformation.py
"""
一、类定位
继承 BaseProcessor，属于数据变换与通用特征工程处理器，负责对数值、类别、时间字段做特征变换，为机器学习建模做特征预处理。每次操作自动生成标准化处理报告，全量留存历史执行日志。
二、核心方法功能
scale 数值标准化缩放
支持三种常用归一化方式：
minmax：最小最大归一化缩放到[0,1]
standard：Z-score 标准化，均值 0 方差 1
robust：基于四分位数的鲁棒缩放，抗异常值；
会保存每一列缩放用的统计参数（均值、极值、四分位数等），方便后续线上推理复用。
log_transform 对数变换
对数值列做对数变换修正偏态分布，设置偏移量避免负数 / 零报错，会跳过不符合取值要求的列并记录跳过字段。
binning 连续特征离散分箱
通过指定分箱边界或分箱数量，将连续数值转为分类类别，支持自定义标签与新列名，统计每个分箱样本分布。
one_hot_encode 类别特征独热编码
对分类字段生成虚拟哑变量，支持丢弃第一列、空值单独编码，自动记录所有新增的编码衍生列。
datetime_features 单时间列特征提取
从单个时间字段拆分出年、月、日、星期、小时特征，可选择删除原时间列，记录所有生成的时间衍生字段。
extract_datetime_features 批量时间特征提取
支持批量处理多个时间列，除常规时间维度外，额外新增月初、月末标记，记录成功 / 失败的字段列表，统一上报执行状态、新增特征数量。
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, cast

import numpy as np
import pandas as pd

from .base import BaseProcessor
from .utils import (
    ensure_tabular,
    numeric_columns,
    safe_columns,
    to_dataframe,
)
from datamineana.dataobject import TabularData


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


class DataTransformer(BaseProcessor):
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

        report = self._new_report(
            step="scale",
            params={"method": method, "columns": target_columns},
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        scale_params = {}
        for col in target_columns:
            s = result[col]

            if method == "minmax":
                min_v = s.min()
                max_v = s.max()
                denom = max_v - min_v
                result[col] = 0 if denom == 0 else (s - min_v) / denom
                scale_params[col] = {"min": float(min_v), "max": float(max_v)}

            elif method == "standard":
                mean = s.mean()
                std = s.std()
                result[col] = 0 if std == 0 else (s - mean) / std
                scale_params[col] = {"mean": float(mean), "std": float(std)}

            elif method == "robust":
                median = s.median()
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                result[col] = 0 if iqr == 0 else (s - median) / iqr
                scale_params[col] = {"median": float(median), "iqr": float(iqr)}

            else:
                raise ValueError(f"未知缩放方法: {method}")

        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {"scale_params": scale_params}
        report.finish()

        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

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

        report = self._new_report(
            step="log_transform",
            params={"columns": target_columns, "offset": offset},
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        skipped = []
        for col in target_columns:
            min_v = result[col].min()
            if min_v + offset <= 0:
                skipped.append(col)
                continue
            result[col] = np.log(result[col] + offset)

        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "skipped_columns": skipped,
            "skipped_count": len(skipped),
        }
        report.finish()

        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

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

        report = self._new_report(
            step="binning",
            params={
                "column": column,
                "bins": bins,
                "labels": labels,
                "new_column": new_column,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        output_col = new_column or f"{column}_bin"
        result[output_col] = pd.cut(result[column], bins=bins, labels=labels)

        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "output_column": output_col,
            "value_counts": result[output_col].value_counts(dropna=False).to_dict(),
        }
        report.finish()

        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

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

        report = self._new_report(
            step="one_hot_encode",
            params={
                "columns": target_columns,
                "drop_first": drop_first,
                "dummy_na": dummy_na,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        result = pd.get_dummies(
            df,
            columns=target_columns,
            drop_first=drop_first,
            dummy_na=dummy_na,
        )
        created_columns = [c for c in result.columns if c not in df.columns]

        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "created_columns": created_columns,
            "created_count": len(created_columns),
        }
        report.finish()

        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

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

        report = self._new_report(
            step="datetime_features",
            params={
                "column": column,
                "drop_original": drop_original,
                "prefix": prefix,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
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

        created_columns = [c for c in result.columns if c not in df.columns]

        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "created_columns": created_columns,
            "created_count": len(created_columns),
        }
        report.finish()

        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

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
            before_shape=(int(df.shape[0]), int(df.shape[1])),
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
                    year_col, month_col, day_col,
                    dayofweek_col, hour_col,
                    is_month_start_col, is_month_end_col,
                ])

                if drop_original:
                    result = result.drop(columns=[col])

            except Exception as e:
                failed_columns.append(f"{col}: {repr(e)}")

        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "created_columns": created_columns,
            "created_count": len(created_columns),
            "failed_columns": failed_columns,
            "failed_count": len(failed_columns),
        }
        report.finish(success=len(failed_columns) == 0)

        return self._wrap_result(data, result, report)

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
