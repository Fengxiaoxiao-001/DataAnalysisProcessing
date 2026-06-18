# datamineana/processors/utils.py
"""
# utils.py 简要概述
## 一、定位
项目**通用工具函数模块**，为所有数据处理器提供底层公共能力，类型校验、格式转换、元数据处理、数据集封装、列筛选、内存优化、子数据集挂载等重复逻辑统一抽离在此，避免各个处理器重复写相同代码，同时支撑新旧两套报告体系运行。

## 二、核心函数功能
### 1. 类型与格式安全校验转换
1. `ensure_tabular`：校验入参必须是`TabularData`类型，否则抛类型异常，所有处理器的入参前置校验工具。
2. `to_dataframe`：将`TabularData`转为原生DataFrame，支持懒加载开关，控制是否全量读取磁盘数据。

### 2. 元数据 & 报告链路管理
1. `copy_metadata`：安全拷贝数据集元数据字典，防止原对象被意外修改。
2. `append_report_to_metadata`：把单次处理报告追加到元数据的`process_reports`列表，实现处理日志跟着数据集流转、全链路可追溯。
3. `make_tabular_like`：旧架构核心封装函数，基于原数据集复制属性、元数据，挂载处理报告，快速生成新的`TabularData`，旧版处理器统一用该方法返回结果。

### 3. 列筛选工具
1. `numeric_columns`/`categorical_columns`/`datetime_columns`：分别筛选数值、分类、时间类型字段名列表。
2. `safe_columns`：过滤掉数据表中不存在的列，防止索引报错，批量列操作必备。

### 4. 内存优化工具
1. `estimate_memory_mb`：统计DataFrame当前占用内存大小（MB）。
2. `reduce_memory_usage`：自动向下精简数值类型、低基数文本列转为分类类型，大幅缩减数据集内存开销。

### 5. 扩展能力
`attach_special_dataframe`：兼容数据集扩展方法，将训练集、验证集这类子数据集挂载到原始`TabularData`对象上，方便后续追溯查看拆分后的子集。

## 三、设计作用
1. **代码复用**：所有清洗、集成、降维、时序、特征工程模块共用一套工具方法，统一编码规范，减少冗余bug。
2. **向下兼容支撑**：旧处理器依赖`make_tabular_like`实现结果封装与报告挂载，新处理器可以选择封装复用该函数，保证新旧代码元数据、报告流转规则完全一致。
3. **安全容错**：内置类型校验、无效列过滤、异常捕获，提升整个预处理框架健壮性。
4. **统一数据流转规则**：所有处理后的数据集属性拷贝、日志挂载逻辑集中管理，后续规则变更仅需要修改当前工具文件即可全局生效。
"""

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
