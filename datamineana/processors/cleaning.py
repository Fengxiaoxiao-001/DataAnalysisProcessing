from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional, Sequence, cast
import pandas as pd

from .base import BaseProcessor
from .utils import safe_columns
from datamineana.dataobject import TabularData


DuplicateKeep = Literal["first", "last", False]
ConvertErrors = Literal["raise", "coerce"]


class DataCleaner(BaseProcessor):
    name = "data_cleaner"

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
                    result[col] = pd.to_datetime(
                        result[col],
                        errors=errors,
                    )
                elif target in {"int", "integer", "Int64"}:
                    result[col] = pd.to_numeric(
                        result[col],
                        errors=errors,
                    ).astype("Int64")
                elif target in {"float", "double", "float64"}:
                    result[col] = pd.to_numeric(
                        result[col],
                        errors=errors,
                    )
                elif target in {"str", "string"}:
                    result[col] = result[col].astype("string")
                elif target == "category":
                    result[col] = result[col].astype("category")
                elif target in {"bool", "boolean"}:
                    result[col] = result[col].astype("boolean")
                else:
                    # pandas-stubs 对 astype(str变量) 有时提示过严，这里显式忽略或 cast
                    result[col] = result[col].astype(cast(Any, target_type))

                converted[col] = target_type

            except Exception as e:
                failed[col] = repr(e)

        report.after_shape = result.shape
        report.statistics = {
            "converted": converted,
            "failed": failed,
        }
        report.success = len(failed) == 0
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