# datamineana/processors/utils.py

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, cast

import pandas as pd

from datamineana.dataobject.DataSelf import TabularData
from .report import ProcessReport


def ensure_tabular(data: Any) -> TabularData:
    if not isinstance(data, TabularData):
        raise TypeError(f"需要 TabularData 类型，实际收到: {type(data)}")
    return data


def to_dataframe(data: TabularData, materialize: bool = True) -> pd.DataFrame:
    """
    将 TabularData 转为 DataFrame。

    注意：
    对懒加载数据，如果 materialize=True，会触发全量读取。
    """
    ensure_tabular(data)

    if not materialize:
        try:
            return data.view()
        except Exception:
            return data.to_dataframe(cache=False)

    return data.to_dataframe(cache=True)


def copy_metadata(data: TabularData) -> Dict[str, Any]:
    metadata = getattr(data, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def append_report_to_metadata(
        metadata: Dict[str, Any],
        report: ProcessReport,
) -> Dict[str, Any]:
    new_metadata = dict(metadata)
    reports = list(new_metadata.get("process_reports", []))
    reports.append(report.to_dict())
    new_metadata["process_reports"] = reports
    return new_metadata


def make_tabular_like(
        source: TabularData,
        df: pd.DataFrame,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        processed_by: Optional[str] = None,
        report: Optional[ProcessReport] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
) -> TabularData:
    """
    基于已有 TabularData 创建新的 TabularData。

    这里统一保证输出仍然是 TabularData。
    """
    metadata = copy_metadata(source)

    if extra_metadata:
        metadata.update(extra_metadata)

    if report is not None:
        metadata = append_report_to_metadata(metadata, report)

    result = TabularData.from_dataframe(
        df,
        name=name if name is not None else getattr(source, "name", None),
        description=description if description is not None else getattr(source, "description", None),
        source_path=getattr(source, "source_path", None),
        source_type=getattr(source, "source_type", None),
        file_type=getattr(source, "file_type", None),
        metadata=metadata,
        processed_by=processed_by,
    )

    return result


def numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()


def safe_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> List[str]:
    if columns is None:
        return df.columns.tolist()
    return [c for c in columns if c in df.columns]


def estimate_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024 / 1024)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    尽量降低 DataFrame 内存占用：
    1. int 降级
    2. float 降级
    3. 低基数 object 转 category

    注意：
    不要使用 result[col] 取列，因为当 DataFrame 有重复列名时，
    result[col] 可能返回 DataFrame，而不是 Series。
    所以这里使用 iloc 按位置取列，保证 s 是单列 Series。
    """
    result = df.copy()

    for i in range(result.shape[1]):
        s = cast(pd.Series, result.iloc[:, i])

        if pd.api.types.is_integer_dtype(s):
            converted = pd.to_numeric(s, downcast="integer")
            result.iloc[:, i] = converted

        elif pd.api.types.is_float_dtype(s):
            converted = pd.to_numeric(s, downcast="float")
            result.iloc[:, i] = converted

        elif pd.api.types.is_object_dtype(s):
            nunique = int(s.nunique(dropna=True))
            total = int(len(s))

            if total > 0:
                ratio = float(nunique) / float(total)
                if ratio < 0.5:
                    result.iloc[:, i] = s.astype("category")

    return result


def attach_special_dataframe(
        data: TabularData,
        name: str,
        df: pd.DataFrame,
        *,
        description: Optional[str] = None,
        processed_by: Optional[str] = None,
) -> None:
    """
    如果 TabularData 支持 make_special_data，则将中间结果挂载进去。
    """
    if hasattr(data, "make_special_data"):
        try:
            data.make_special_data(
                name=name,
                df=df,
                description=description,
                processed_by=processed_by,
            )
        except Exception:
            pass
