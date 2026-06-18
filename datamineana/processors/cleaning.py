# datamineana/processors/cleaning.py
"""
1. 类定位
继承基类 BaseProcessor，属于数据清洗专用处理器，提供结构化数据集常用的脏数据清洗能力，所有操作都会自动生成标准化处理日志，同时做了新旧代码双向兼容。
2. 核心功能（7 个对外方法）
handle_missing：缺失值处理，支持均值、中位数、众数、常量、行列删除、前后填充、插值等多种填充策略，统计每列处理前后缺失数量。
handle_outliers：异常值检测与处理，提供 IQR、Z-score 两种检测方式，支持截断、置空、删除异常行三种处理动作，记录异常上下限与异常条数。
normalize_text：文本字段统一格式化，去首尾空格、大小写转换、批量字符替换，规整脏字符串。
correct_values：自定义数据修正，支持字典映射批量替换、自定义函数批量修正字段值，统计被修改的数据行数。
drop_duplicates：按指定列删除重复行，统计删除的重复行数，使用新标准报告体系。
convert_dtypes_by_mapping（新规范主方法）：批量字段类型转换，支持日期、数值、字符串、分类、布尔类型，记录转换成功 / 失败字段，是推荐使用的类型转换接口。
convert_types（弃用兼容接口）：旧版类型转换方法，仅做向下兼容，内部直接调用新方法，给出弃用警告，保留旧版报告格式兼容。
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Iterable, Mapping, Optional, Sequence, Union, Callable
import numpy as np
import pandas as pd

from .base import BaseProcessor
from .utils import safe_columns, ensure_tabular, to_dataframe, ProcessReport, numeric_columns
from datamineana.dataobject import TabularData

DuplicateKeep = Literal["first", "last", False]
ConvertErrors = Literal["raise", "coerce"]


class DataCleaner(BaseProcessor):
    name = "data_cleaner"

    def handle_missing(
            self,
            data: TabularData,
            *,
            strategy: str = "mean",
            columns: Optional[Iterable[str]] = None,
            fill_value: Any = None,
            threshold: Optional[float] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        缺失值处理。

        strategy:
        - mean: 数值列均值填充
        - median: 数值列中位数填充
        - mode: 众数填充
        - constant: 常量填充，需要 fill_value
        - drop_rows: 删除包含缺失值的行
        - drop_cols: 删除缺失率超过 threshold 的列
        - ffill: 前向填充
        - bfill: 后向填充
        - interpolate: 插值，主要适合数值列和时间序列
        """
        ensure_tabular(data)
        df = to_dataframe(data)

        # ===== 新体系：标准报告创建 =====
        report = self._new_report(
            step="handle_missing",
            params={
                "strategy": strategy,
                "columns": list(columns) if columns is not None else None,
                "fill_value": fill_value,
                "threshold": threshold,
            },
            before_shape=df.shape,
            materialized=True,
        )

        before_missing = df.isna().sum().to_dict()
        target_columns = safe_columns(df, columns)
        result = df.copy()
        drop_cols = []

        if strategy == "drop_rows":
            result = result.dropna(subset=target_columns)
        elif strategy == "drop_cols":
            if threshold is None:
                threshold = 0.5
            missing_rate = result[target_columns].isna().mean()
            drop_cols = missing_rate[missing_rate > threshold].index.tolist()
            result = result.drop(columns=drop_cols)
        elif strategy == "mean":
            for col in target_columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].mean())
        elif strategy == "median":
            for col in target_columns:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(result[col].median())
        elif strategy == "mode":
            for col in target_columns:
                mode = result[col].mode(dropna=True)
                if not mode.empty:
                    result[col] = result[col].fillna(mode.iloc[0])
        elif strategy == "constant":
            result[target_columns] = result[target_columns].fillna(fill_value)
        elif strategy == "ffill":
            result[target_columns] = result[target_columns].ffill()
        elif strategy == "bfill":
            result[target_columns] = result[target_columns].bfill()
        elif strategy == "interpolate":
            num_cols = [c for c in target_columns if pd.api.types.is_numeric_dtype(result[c])]
            result[num_cols] = result[num_cols].interpolate()
        else:
            raise ValueError(f"未知缺失值处理策略: {strategy}")

        after_missing = result.isna().sum().to_dict()

        # ===== 新体系：报告指标与收尾 =====
        report.after_shape = result.shape
        report.statistics = {
            "dropped_columns": drop_cols,
            "missing_before": before_missing,
            "missing_after": after_missing,
            "missing_reduced": {
                k: before_missing.get(k, 0) - after_missing.get(k, 0)
                for k in before_missing
            }
        }
        report.finish()

        # ===== 兼容旧版 self.last_report =====
        self.last_report = ProcessReport(
            module="cleaning",
            method="handle_missing",
            params=report.params,
            before_shape=df.shape,
            after_shape=result.shape,
            materialized=True,
        )
        for k, v in report.statistics.items():
            self.last_report.add_metric(k, v)

        # ===== 新体系：结果封装 + 兼容name参数 =====
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def handle_outliers(
            self,
            data: TabularData,
            *,
            method: str = "iqr",
            action: str = "cap",
            columns: Optional[Iterable[str]] = None,
            z_threshold: float = 3.0,
            iqr_factor: float = 1.5,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        异常值处理。

        method:
        - iqr
        - zscore

        action:
        - cap: 截尾，将异常值压到上下边界
        - remove: 删除含异常值的行
        - nan: 将异常值置为 NaN
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        # ===== 新体系：标准报告创建 =====
        report = self._new_report(
            step="handle_outliers",
            params={
                "method": method,
                "action": action,
                "columns": list(columns) if columns else None,
                "z_threshold": z_threshold,
                "iqr_factor": iqr_factor,
            },
            before_shape=df.shape,
            materialized=True,
        )

        target_columns = safe_columns(result, columns) if columns else numeric_columns(result)
        outlier_mask = pd.Series(False, index=result.index)
        bounds: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}

        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            s = result[col]

            if method == "iqr":
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
            elif method == "zscore":
                mean = s.mean()
                std = s.std()
                lower = mean - z_threshold * std
                upper = mean + z_threshold * std
            else:
                raise ValueError(f"未知异常值检测方法: {method}")

            mask = (s < lower) | (s > upper)
            outlier_mask = outlier_mask | mask
            counts[col] = int(mask.sum())
            bounds[col] = {"lower": float(lower), "upper": float(upper)}

            if action == "cap":
                result[col] = result[col].clip(lower, upper)
            elif action == "nan":
                result.loc[mask, col] = np.nan
            elif action == "remove":
                pass
            else:
                raise ValueError(f"未知异常值处理动作: {action}")

        if action == "remove":
            result = result.loc[~outlier_mask].copy()

        # ===== 新体系：报告指标与收尾 =====
        report.after_shape = result.shape
        report.statistics = {
            "outlier_counts": counts,
            "bounds": bounds,
            "affected_rows": int(outlier_mask.sum()),
        }
        report.finish()

        # ===== 兼容旧版 self.last_report =====
        self.last_report = ProcessReport(
            module="cleaning",
            method="handle_outliers",
            params=report.params,
            before_shape=df.shape,
            after_shape=result.shape,
            materialized=True,
        )
        for k, v in report.statistics.items():
            self.last_report.add_metric(k, v)

        # ===== 新体系：结果封装 + 兼容name参数 =====
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def normalize_text(
            self,
            data: TabularData,
            *,
            columns: Optional[Iterable[str]] = None,
            strip: bool = True,
            lower: bool = False,
            upper: bool = False,
            replace_map: Optional[Mapping[str, str]] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        统一字符串格式。
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        # ===== 新体系：标准报告创建 =====
        report = self._new_report(
            step="normalize_text",
            params={
                "columns": list(columns) if columns else None,
                "strip": strip,
                "lower": lower,
                "upper": upper,
                "replace_map": dict(replace_map) if replace_map else None,
            },
            before_shape=df.shape,
            materialized=True,
        )

        if columns is None:
            target_columns = result.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            target_columns = safe_columns(result, columns)

        for col in target_columns:
            s = result[col].astype("string")
            if strip:
                s = s.str.strip()
            if lower:
                s = s.str.lower()
            if upper:
                s = s.str.upper()
            if replace_map:
                s = s.replace(replace_map)
            result[col] = s

        # ===== 新体系：报告指标与收尾 =====
        report.after_shape = result.shape
        report.statistics = {"processed_columns": target_columns}
        report.finish()

        # ===== 兼容旧版 self.last_report =====
        self.last_report = ProcessReport(
            module="cleaning",
            method="normalize_text",
            params=report.params,
            before_shape=df.shape,
            after_shape=result.shape,
            materialized=True,
        )
        self.last_report.add_metric("processed_columns", target_columns)

        # ===== 新体系：结果封装 + 兼容name参数 =====
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def convert_types(
            self,
            data: TabularData,
            *,
            type_map: Mapping[str, str],
            errors: ConvertErrors = "coerce",
            name: Optional[str] = None,
    ) -> TabularData:
        """
        类型转换（已弃用，请使用 convert_dtypes_by_mapping 替代）

        type_map 示例：
        {
            "date": "datetime",
            "age": "int",
            "price": "float",
            "label": "category",
            "id": "string"
        }
        """
        warnings.warn(
            "convert_types 已弃用，请迁移至 convert_dtypes_by_mapping，后续版本将移除该方法",
            DeprecationWarning,
            stacklevel=2
        )

        # 核心逻辑转发到统一新实现
        result = self.convert_dtypes_by_mapping(data, dtype_mapping=type_map, errors=errors)

        # 兼容旧版 name 参数
        if name is not None:
            result.name = name

        # 兼容旧版 self.last_report 结构
        df = to_dataframe(data)
        report = ProcessReport(
            module="cleaning",
            method="convert_types",
            params={"type_map": dict(type_map), "errors": errors},
            before_shape=df.shape,
            after_shape=to_dataframe(result).shape,
            materialized=True,
        )
        latest_report = result.metadata.get("process_reports", [])[-1]
        report.add_metric("failed", latest_report["statistics"].get("failed", {}))
        report.add_metric("dtypes_after", to_dataframe(result).dtypes.astype(str).to_dict())
        self.last_report = report

        return result

    def correct_values(
            self,
            data: TabularData,
            *,
            corrections: Mapping[str, Union[Mapping[Any, Any], Callable[[Any], Any]]],
            name: Optional[str] = None,
    ) -> TabularData:
        """
        修正错误数据。

        corrections 示例：
        {
            "gender": {"M ": "M", "male": "M", "female": "F"},
            "age": lambda x: np.nan if x < 0 else x
        }
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        # ===== 新体系：标准报告创建 =====
        report = self._new_report(
            step="correct_values",
            params={"columns": list(corrections.keys())},
            before_shape=df.shape,
            materialized=True,
        )

        changed_counts = {}
        for col, rule in corrections.items():
            if col not in result.columns:
                continue
            before = result[col].copy()
            if callable(rule):
                result[col] = result[col].map(rule)
            else:
                result[col] = result[col].replace(rule)
            changed_counts[col] = int((before != result[col]).fillna(False).sum())

        # ===== 新体系：报告指标与收尾 =====
        report.after_shape = result.shape
        report.statistics = {"changed_counts": changed_counts}
        report.finish()

        # ===== 兼容旧版 self.last_report =====
        self.last_report = ProcessReport(
            module="cleaning",
            method="correct_values",
            params=report.params,
            before_shape=df.shape,
            after_shape=result.shape,
            materialized=True,
        )
        self.last_report.add_metric("changed_counts", changed_counts)

        # ===== 新体系：结果封装 + 兼容name参数 =====
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def drop_duplicates(
            self,
            data: TabularData,
            subset: Optional[Sequence[str]] = None,
            keep: DuplicateKeep = "first",
    ) -> TabularData:
        df = data.to_dataframe()
        report = self._new_report(
            step="drop_duplicates",
            params={
                "subset": list(subset) if subset is not None else None,
                "keep": keep,
            },
            before_shape=df.shape,
            materialized=True,
        )

        valid_subset = safe_columns(df, subset) if subset is not None else None
        result = df.drop_duplicates(
            subset=valid_subset,
            keep=keep,
        )

        report.after_shape = result.shape
        report.statistics = {
            "removed_rows": int(df.shape[0] - result.shape[0]),
        }
        report.finish()

        return self._wrap_result(data, result, report)

    def convert_dtypes_by_mapping(
            self,
            data: TabularData,
            dtype_mapping: Mapping[str, str],
            errors: ConvertErrors = "coerce",
    ) -> TabularData:
        """
        根据映射转换字段类型。

        dtype_mapping 示例：
        {
            "age": "int",
            "price": "float",
            "created_at": "datetime",
            "label": "category",
            "flag": "bool"
        }
        """
        df = data.to_dataframe()
        report = self._new_report(
            step="convert_dtypes_by_mapping",
            params={
                "dtype_mapping": dict(dtype_mapping),
                "errors": errors,
            },
            before_shape=df.shape,
            materialized=True,
        )

        result = df.copy()
        converted: Dict[str, str] = {}
        failed: Dict[str, str] = {}

        for col, target_type in dtype_mapping.items():
            if col not in result.columns:
                failed[col] = "column not found"
                continue

            try:
                target = target_type.lower().strip()

                if target in {"datetime", "date", "time"}:
                    result[col] = pd.to_datetime(result[col], errors=errors)
                elif target in {"int", "integer", "int64"}:
                    result[col] = pd.to_numeric(result[col], errors=errors).astype("Int64")
                elif target in {"float", "double", "float64"}:
                    result[col] = pd.to_numeric(result[col], errors=errors)
                elif target in {"str", "string"}:
                    result[col] = result[col].astype("string")
                elif target == "category":
                    result[col] = result[col].astype("category")
                elif target in {"bool", "boolean"}:
                    result[col] = result[col].astype("boolean")
                else:
                    # 规范写法：转换为标准dtype对象，替代cast暴力绕过
                    result[col] = result[col].astype(pd.api.types.pandas_dtype(target_type))

                converted[col] = target_type
            except Exception as e:
                failed[col] = repr(e)

        report.after_shape = result.shape
        report.statistics = {
            "converted": converted,
            "failed": failed,
        }
        # 移除重复的success赋值，统一通过finish传参
        report.finish(success=len(failed) == 0)

        return self._wrap_result(data, result, report)

    def _wrap_result(
            self,
            source: TabularData,
            df: pd.DataFrame,
            report,
    ) -> TabularData:
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
