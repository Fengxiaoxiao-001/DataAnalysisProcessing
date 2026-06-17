from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

import pandas as pd
from pandas.io.parsers import TextFileReader

from .Dtypes import FileType, PathLike
from .special_data import SpecialData, ProcessedTabularData, ProcessedTableData

# =============================================================================
# 内部哨兵对象
# =============================================================================
#
# 用于 copy_with()
#
# 为什么不用 None？
# ---------------------------------------------------------------------------
# 因为 None 本身可能是用户想主动设置的值。
#
# 例如：
#
#   data.copy_with(selected_columns=None)
#
# 这表示用户明确想清除列选择。
#
# 但：
#
#   data.copy_with()
#
# 表示保持 selected_columns 原值不变。
#
# 如果不用 _UNSET，就无法区分这两种情况。
# =============================================================================

_UNSET = object()

# =============================================================================
# 类型别名
# =============================================================================
#
# ColumnSelector:
#     用于表示用户想选择哪些列。
#
#     支持：
#     - None
#     - 单个列名 str
#     - 单个列位置 int
#     - 多个列名 Sequence[str]
#     - 多个列位置 Sequence[int]
#     - 一个 callable，例如 lambda col: col.startswith("user_")
#
# TabularRawData:
#     TabularData 底层可接收的数据类型。
#
#     支持：
#     - pandas.DataFrame
#     - pandas TextFileReader，也就是 read_csv(..., chunksize=xxx)
#     - Iterable[pandas.DataFrame]
#     - None
#
# 注意：
# ---------------------------------------------------------------------------
# 这里使用 TypeAlias + Union 是为了兼容 IDE / 类型检查器。
# 不建议写成复杂的运行时表达式，否则部分检查器会提示：
# “类型提示无效或引用的表达式类型不正确”。
# =============================================================================

ColumnSelector: TypeAlias = Union[
    str,
    int,
    Sequence[str],
    Sequence[int],
    Callable[[str], bool],
    None,
]

RowSelector: TypeAlias = Union[
    str,
    int,
    Sequence[str],
    Sequence[int],
    Callable[[Any], bool],
    None,
]

TabularRawData: TypeAlias = Union[
    pd.DataFrame,
    TextFileReader,
    Iterable[pd.DataFrame],
    None,
]


# =============================================================================
# 时间工具
# =============================================================================

def _now_utc_datetime() -> datetime:
    """
    返回 timezone-aware 的 UTC datetime。

    不使用 datetime.utcnow()，因为它在新版本 Python 中已经被标记为 deprecated。
    """
    return datetime.now(UTC)


def _now_utc_iso() -> str:
    """
    返回 UTC ISO 字符串。

    示例：
        2026-06-17T12:34:56+00:00
    """
    return _now_utc_datetime().isoformat(timespec="seconds")


# =============================================================================
# FileType 兼容工具
# =============================================================================

def _file_type_member(name: str) -> FileType:
    """
    安全获取 FileType 枚举成员。

    这么写是为了兼容你的旧版本 FileType。

    例如你的 FileType 里面可能只有：
        CSV
        XLSX
        XLS
        UNKNOWN

    但没有：
        JSON
        HTML

    如果直接写 FileType.JSON / FileType.HTML，
    IDE 或运行时都可能报错。

    所以统一使用 getattr。
    """
    return getattr(FileType, name, FileType.UNKNOWN)


def _safe_file_type_value(file_type: Any) -> str:
    """
    安全获取 FileType 的字符串值。

    兼容：
    1. FileType 枚举
    2. 普通字符串
    3. None
    """
    if file_type is None:
        return str(getattr(FileType.UNKNOWN, "value", "unknown"))

    if hasattr(file_type, "value"):
        return str(file_type.value)

    return str(file_type)


def _infer_file_type_from_path(path: Optional[Union[str, Path]]) -> FileType:
    """
    根据路径后缀推断文件类型。

    注意：
    ---------------------------------------------------------------------------
    这里只做轻量推断。
    具体读取逻辑应该放在 loader 中。
    """
    if path is None:
        return FileType.UNKNOWN

    suffix = Path(path).suffix.lower().lstrip(".")

    if suffix == "csv":
        return _file_type_member("CSV")

    if suffix == "xlsx":
        return _file_type_member("XLSX")

    if suffix == "xls":
        return _file_type_member("XLS")

    if suffix == "json":
        return _file_type_member("JSON")

    if suffix in {"html", "htm"}:
        return _file_type_member("HTML")

    return FileType.UNKNOWN


# =============================================================================
# JSON 安全工具
# =============================================================================

def _json_safe_dataframe(
        df: pd.DataFrame,
        *,
        max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    将 DataFrame 转成更适合 JSON 输出的 DataFrame。

    主要处理：
    - NaN
    - NaT
    - pandas.NA

    这些值不能被标准 JSON 正确表达，所以统一转成 None。

    大型数据集优化：
    ---------------------------------------------------------------------------
    如果 max_rows 不为 None，只取前 max_rows 行，避免一次性导出巨大数据。

    类型检查说明：
    ---------------------------------------------------------------------------
    有些 IDE 会把 df.where(...) 推断成 Series。
    所以这里使用：
        df.astype(object).mask(pd.isna(df), None)
    并显式 cast 成 DataFrame。
    """
    if max_rows is not None:
        df = df.head(max_rows)

    safe_df = df.astype(object).mask(pd.isna(df), None)
    return safe_df


def _json_default(value: Any) -> Any:
    """
    json.dumps 的 default 回调。

    用于处理 datetime、Path 等标准 JSON 不支持的对象。
    """
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    return str(value)


def _metadata_to_plain_dict(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    将 metadata 转成普通 dict。

    主要是为了兼容：
    - None
    - dict
    - Mapping
    """
    if metadata is None:
        return {}

    return dict(metadata)


# =============================================================================
# 基础数据结构
# =============================================================================

@dataclass
class ColumnInfo:
    """
    单列信息。

    字段：
    ---------------------------------------------------------------------------
    name:
        列名。

    dtype:
        字段类型字符串。
        例如：
        - int64
        - float64
        - object
        - datetime64[ns]
    """

    name: str
    dtype: str


@dataclass
class DataViewInfo:
    """
    数据视图信息。

    这个类用于描述 TabularData 的结构，而不是保存完整数据。

    用途：
    ---------------------------------------------------------------------------
    1. HTML 展示
    2. 数据预处理前后结构查看
    3. 数据分析前的数据概览
    4. 大数据场景下避免直接物化全量数据

    旧字段：
    ---------------------------------------------------------------------------
    row_count:
        行数。
        lazy 模式下未知时为 None。

    column_count:
        列数。
        lazy 模式下如果无法安全获取则为 None。

    columns:
        当前列信息。

    added_columns:
        新增列信息。
        预留给后续数据处理模块记录字段变化。

    dropped_columns:
        删除列名。
        预留给后续数据处理模块记录字段变化。

    source_path:
        数据来源路径。

    file_type:
        文件类型。

    is_lazy:
        是否是懒加载数据。

    新字段：
    ---------------------------------------------------------------------------
    name:
        数据名称。

    description:
        数据说明。

    source_type:
        来源类型字符串。
        例如 csv、xlsx、dataframe、database 等。

    chunksize:
        分块大小。

    shape:
        数据形状。
        lazy 未物化时通常是 (None, None)。

    dtypes:
        字段类型字典。

    index_name:
        索引名。

    processed_by:
        经哪个模块处理。

    created_at:
        创建时间。

    metadata:
        额外元信息。
    """

    row_count: Optional[int]
    column_count: Optional[int]

    columns: List[ColumnInfo] = field(default_factory=list)

    added_columns: List[ColumnInfo] = field(default_factory=list)
    dropped_columns: List[str] = field(default_factory=list)

    source_path: Optional[str] = None
    file_type: FileType = FileType.UNKNOWN

    is_lazy: bool = False

    name: Optional[str] = None
    description: Optional[str] = None
    source_type: Optional[str] = None
    chunksize: Optional[int] = None
    shape: Optional[tuple[Optional[int], Optional[int]]] = None
    dtypes: Dict[str, str] = field(default_factory=dict)
    index_name: Optional[str] = None
    processed_by: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转成普通 dict。
        """
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype,
                }
                for col in self.columns
            ],
            "added_columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype,
                }
                for col in self.added_columns
            ],
            "dropped_columns": self.dropped_columns,
            "source_path": self.source_path,
            "file_type": _safe_file_type_value(self.file_type),
            "is_lazy": self.is_lazy,
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type,
            "chunksize": self.chunksize,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtypes": self.dtypes,
            "index_name": self.index_name,
            "processed_by": self.processed_by,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_json(
            self,
            ensure_ascii: bool = False,
            indent: Optional[int] = 2,
    ) -> str:
        """
        转成 JSON 字符串。
        """
        return json.dumps(
            self.to_dict(),
            ensure_ascii=ensure_ascii,
            indent=indent,
            default=_json_default,
        )


# 兼容新名字。
TabularViewInfo = DataViewInfo


# =============================================================================
# 列选择工具
# =============================================================================

class ColumnNormalizeMixin:
    """
    列选择工具。

    pandas.read_csv 的 usecols 支持：
    - None
    - list[str]
    - list[int]
    - Callable[[str], bool]

    注意：
    ---------------------------------------------------------------------------
    不建议直接把单个 str 传给 pandas usecols。
    因为字符串本身是可迭代对象，可能被错误处理。

    所以：
        "name"
    会被标准化为：
        ["name"]
    """

    @staticmethod
    def _normalize_columns(columns: ColumnSelector = None) -> Union[
        None,
        Callable[[str], bool],
        List[str],
        List[int],
    ]:
        """
        标准化列选择参数。

        支持：
        1. None
        2. str
        3. int
        4. Sequence[str]
        5. Sequence[int]
        6. Callable[[str], bool]
        """
        if columns is None:
            return None

        if callable(columns):
            return columns

        if isinstance(columns, str):
            return [columns]

        if isinstance(columns, int):
            return [columns]

        if isinstance(columns, Sequence):
            return list(columns)

        raise TypeError(
            "columns must be None, str, int, Sequence[str], "
            "Sequence[int], or Callable[[str], bool]"
        )

    @staticmethod
    def _normalize_rows(rows: RowSelector = None) -> Union[
        None,
        Callable[[Any], bool],
        List[Any],
    ]:
        """
        标准化行选择参数，逻辑与列选择对齐。

        支持：
        1. None
        2. str / int 单个行标签/位置
        3. Sequence[str] / Sequence[int] 多个行标签/位置
        4. Callable[[Any], bool] 行索引过滤函数
        """
        if rows is None:
            return None

        if callable(rows):
            return rows

        if isinstance(rows, (str, int)):
            return [rows]

        if isinstance(rows, Sequence):
            return list(rows)

        raise TypeError(
            "rows must be None, str, int, Sequence, or Callable[[Any], bool]"
        )

    @staticmethod
    def _select_dataframe_columns(
            df: pd.DataFrame,
            columns: ColumnSelector = None,
    ) -> pd.DataFrame:
        """
        从 DataFrame 中选择列。
        """
        normalized = ColumnNormalizeMixin._normalize_columns(columns)

        if normalized is None:
            return df

        if callable(normalized):
            selected = [
                col
                for col in df.columns
                if normalized(str(col))
            ]
            return df.loc[:, selected]

        # 整数列表按位置选列，字符串列表按标签选列
        if all(isinstance(col, int) for col in normalized):
            return df.iloc[:, normalized]

        return df.loc[:, normalized]

    @staticmethod
    def _select_dataframe_rows(
            df: pd.DataFrame,
            rows: RowSelector = None,
    ) -> pd.DataFrame:
        """从 DataFrame 中选择行"""
        normalized = ColumnNormalizeMixin._normalize_rows(rows)

        if normalized is None:
            return df

        if callable(normalized):
            selected = [idx for idx in df.index if normalized(idx)]
            return df.loc[selected, :]

        if all(isinstance(row, int) for row in normalized):
            return df.iloc[normalized, :]

        return df.loc[normalized, :]


# =============================================================================
# 分块读取能力
# =============================================================================

class ChunkMixin:
    """
    分块读取相关能力接口。

    TabularData 会实现这两个方法。
    """

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


# =============================================================================
# SpecialData 管理能力
# =============================================================================

class SpecialDataMixin:
    """
    SpecialData 管理能力。

    SpecialData 用于保存：
    - 统计结果
    - 标注信息
    - 模型输出
    - 处理后的图表数据
    - 其他和表格相关但不属于表格本体的数据
    """

    special_data: Dict[str, SpecialData]

    def add_special_data(self, special: SpecialData) -> None:
        """
        添加特殊数据。
        """
        self.special_data[special.name] = special

    def get_special_data(self, name: str) -> Optional[SpecialData]:
        """
        根据名称获取特殊数据。
        """
        return self.special_data.get(name)

    def remove_special_data(self, name: str) -> Optional[SpecialData]:
        """
        删除特殊数据。
        """
        return self.special_data.pop(name, None)

    def list_special_data(self) -> List[str]:
        """
        列出所有特殊数据名称。
        """
        return list(self.special_data.keys())

    def clear_special_data(self) -> None:
        """
        清空特殊数据。
        """
        self.special_data.clear()


# =============================================================================
# TabularData
# =============================================================================

@dataclass
class TabularData(ColumnNormalizeMixin, ChunkMixin, SpecialDataMixin):
    """
    通用表格数据容器。

    支持数据来源：
    ---------------------------------------------------------------------------
    1. pandas.DataFrame
    2. pandas TextFileReader，例如 pd.read_csv(..., chunksize=xxx)
    3. Iterable[pandas.DataFrame]
    4. None

    支持功能：
    ---------------------------------------------------------------------------
    - 普通 DataFrame 数据保存
    - lazy chunk 分块读取
    - 字段选择
    - 元信息保存
    - SpecialData 保存
    - ProcessedTabularData 保存
    - 转 JSON / dict
    - 数据预览
    - 数据结构查看

    大数据集优化：
    ---------------------------------------------------------------------------
    1. view_info() 不强制读取 lazy 数据。
    2. view() 只读取第一个 chunk。
    3. iter_chunks() 支持 reader_factory，可重复迭代。
    4. to_dict() / to_json() 支持 max_rows，避免巨大数据直接转 JSON。
    5. to_dataframe(cache=False) 可避免缓存超大 DataFrame。
    6. iter_dict_records() 支持分块逐行输出 dict，适合流式处理。

    注意：
    ---------------------------------------------------------------------------
    如果 data 是 TextFileReader，而且没有设置 reader_factory，
    那么它通常只能被消费一次。

    对于 CSVLoader，建议在 loader 里面调用：

        tabular.set_reader_factory(lambda: pd.read_csv(..., chunksize=xxx))

    这样 iter_chunks() 可以重复调用。
    """

    data: TabularRawData = None

    name: Optional[str] = None
    description: Optional[str] = None

    source_path: Optional[Union[str, PathLike, Path]] = None

    source_type: Optional[str] = None

    file_type: FileType = FileType.UNKNOWN

    lazy: bool = False
    chunksize: Optional[int] = None

    selected_columns: ColumnSelector = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    processed_data: Optional[ProcessedTabularData] = None
    processed_by: Optional[str] = None

    special_data: Dict[str, SpecialData] = field(default_factory=dict)

    created_at: datetime = field(default_factory=_now_utc_datetime)

    _cached_dataframe: Optional[pd.DataFrame] = field(
        default=None,
        init=False,
        repr=False,
    )

    _reader_factory: Optional[Callable[[], Iterator[pd.DataFrame]]] = field(
        default=None,
        init=False,
        repr=False,
    )

    # 新增：懒加载下列结构缓存，只存列名和类型，不存数据
    _cached_columns: Optional[List[str]] = field(
        default=None,
        init=False,
        repr=False,
    )

    _cached_dtypes: Optional[Dict[str, str]] = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        初始化后处理。
        """
        if isinstance(self.source_path, Path):
            self.source_path = str(self.source_path)

        if self.file_type == FileType.UNKNOWN:
            self.file_type = _infer_file_type_from_path(
                str(self.source_path) if self.source_path is not None else None
            )

        if self.source_type is None:
            self.source_type = _safe_file_type_value(self.file_type)

        if isinstance(self.data, pd.DataFrame):
            if self.selected_columns is not None:
                self.data = self._select_dataframe_columns(
                    self.data,
                    self.selected_columns,
                )

            self.lazy = False
            self._cached_dataframe = self.data

        elif isinstance(self.data, TextFileReader):
            self.lazy = True

            if self.chunksize is None:
                self.chunksize = getattr(self.data, "chunksize", None)

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_dataframe(
            cls,
            df: pd.DataFrame,
            *,
            name: Optional[str] = None,
            description: Optional[str] = None,
            source_path: Optional[Union[str, PathLike, Path]] = None,
            source_type: Optional[str] = None,
            file_type: FileType = FileType.UNKNOWN,
            metadata: Optional[Dict[str, Any]] = None,
            processed_by: Optional[str] = None,
            selected_columns: ColumnSelector = None,
    ) -> TabularData:
        """
        从 pandas.DataFrame 创建 TabularData。
        """
        return cls(
            data=df,
            name=name,
            description=description,
            source_path=source_path,
            source_type=source_type,
            file_type=file_type,
            lazy=False,
            selected_columns=selected_columns,
            metadata=_metadata_to_plain_dict(metadata),
            processed_by=processed_by,
        )

    @classmethod
    def from_chunks(
            cls,
            chunks: Union[TextFileReader, Iterable[pd.DataFrame]],
            *,
            name: Optional[str] = None,
            description: Optional[str] = None,
            source_path: Optional[Union[str, PathLike, Path]] = None,
            source_type: Optional[str] = None,
            file_type: FileType = FileType.UNKNOWN,
            chunksize: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None,
            selected_columns: ColumnSelector = None,
    ) -> TabularData:
        """
        从 chunk reader 或 Iterable[pd.DataFrame] 创建 TabularData。
        """
        return cls(
            data=chunks,
            name=name,
            description=description,
            source_path=source_path,
            source_type=source_type,
            file_type=file_type,
            lazy=True,
            chunksize=chunksize,
            selected_columns=selected_columns,
            metadata=_metadata_to_plain_dict(metadata),
        )

    @classmethod
    def empty(
            cls,
            *,
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> TabularData:
        """
        创建空 TabularData。
        """
        return cls(
            data=pd.DataFrame(),
            name=name,
            description=description,
            metadata=_metadata_to_plain_dict(metadata),
        )

    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> tuple[Optional[int], Optional[int]]:
        """
        数据形状。

        lazy 模式下不强制读取全部数据，所以通常返回：
            (None, None)
        """
        if self._cached_dataframe is not None:
            return self._cached_dataframe.shape

        if isinstance(self.data, pd.DataFrame):
            return self.data.shape

        return None, None

    @property
    def columns(self) -> List[str]:
        """
        列名列表。

        优先级：
        1. 已物化的缓存 DataFrame
        2. 懒加载列名缓存
        3. TextFileReader 内部引擎 names
        4. 读取第一个 chunk 获取（仅懒加载场景，不缓存全量数据）
        """
        if self._cached_dataframe is not None:
            return [str(col) for col in self._cached_dataframe.columns.tolist()]

        if isinstance(self.data, pd.DataFrame):
            return [str(col) for col in self.data.columns.tolist()]

        # 命中缓存直接返回
        if self._cached_columns is not None:
            return self._cached_columns

        # 尝试从 TextFileReader 内部拿
        if isinstance(self.data, TextFileReader):
            names = getattr(getattr(self.data, "_engine", None), "names", None)
            if names is not None:
                self._cached_columns = [str(col) for col in names]
                return self._cached_columns

        # 最后兜底：只读第一个 chunk 拿列名，不缓存全量数据
        try:
            for chunk in self.iter_chunks():
                self._cached_columns = [str(col) for col in chunk.columns.tolist()]
                return self._cached_columns
        except Exception as e:
            raise ValueError(f"读取不到列名，报错{e}")

        return []

    @property
    def dtypes(self) -> Dict[str, str]:
        """
        字段类型。

        懒加载模式下读取第一个 chunk 获取，不物化全量数据。
        """
        if self._cached_dataframe is not None:
            return {
                str(col): str(dtype)
                for col, dtype in self._cached_dataframe.dtypes.items()
            }

        if isinstance(self.data, pd.DataFrame):
            return {
                str(col): str(dtype)
                for col, dtype in self.data.dtypes.items()
            }

        # 命中缓存直接返回
        if self._cached_dtypes is not None:
            return self._cached_dtypes

        # 兜底：读第一个 chunk 拿类型
        try:
            for chunk in self.iter_chunks():
                self._cached_dtypes = {
                    str(col): str(dtype)
                    for col, dtype in chunk.dtypes.items()
                }
                # 顺便同步缓存列名
                if self._cached_columns is None:
                    self._cached_columns = [str(col) for col in chunk.columns.tolist()]
                return self._cached_dtypes

        except Exception as e:
            raise ValueError(f"读取不到类型，报错{e}")
        return {}

    @property
    def row_names(self) -> List[Any]:
        """
        行名。

        lazy 模式下不主动读取数据。
        """
        if self._cached_dataframe is not None:
            return list(self._cached_dataframe.index)

        if isinstance(self.data, pd.DataFrame):
            return list(self.data.index)

        return []

    @property
    def column_names(self) -> List[str]:
        """
        列名。
        """
        return self.columns

    @property
    def values(self) -> List[List[Any]]:
        """
        具体数据。

        注意：
        -----------------------------------------------------------------------
        lazy 模式下会触发完整读取。
        大型数据集请谨慎使用。
        """
        df = self.to_dataframe()
        safe_df = _json_safe_dataframe(df)
        return safe_df.values.tolist()

    # -------------------------------------------------------------------------
    # Reader factory
    # -------------------------------------------------------------------------

    def set_reader_factory(
            self,
            factory: Callable[[], Iterator[pd.DataFrame]],
    ) -> TabularData:
        """
        设置可重复创建 chunk iterator 的工厂。

        这个方法非常适合 CSV 大文件。

        示例：
        -----------------------------------------------------------------------
        data.set_reader_factory(
            lambda: pd.read_csv(path, chunksize=10000)
        )

        这样每次调用 iter_chunks() 都会重新创建 reader，
        避免 TextFileReader 被消费一次后无法再次读取。
        """
        self._reader_factory = factory
        self.lazy = True
        return self

    # -------------------------------------------------------------------------
    # Chunk operations
    # -------------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """
        迭代数据块。

        情况：
        -----------------------------------------------------------------------
        1. 如果设置了 reader_factory，每次都用新的 reader。
        2. 如果是 TextFileReader，直接 yield 它。
        3. 如果是普通 DataFrame，根据 chunksize 切片。
        4. 如果是 Iterable[pd.DataFrame]，直接迭代。

        注意：
        -----------------------------------------------------------------------
        如果 data 是 TextFileReader 且没有 reader_factory，
        它通常只能被消费一次。
        """
        if self._reader_factory is not None:
            for chunk in self._reader_factory():
                yield self._post_process_chunk(chunk)
            return

        if isinstance(self.data, TextFileReader):
            for chunk in self.data:
                yield self._post_process_chunk(chunk)
            return

        if isinstance(self.data, pd.DataFrame):
            df = self._post_process_chunk(self.data)

            if not self.chunksize or self.chunksize <= 0:
                yield df
                return

            total = len(df)
            for start in range(0, total, self.chunksize):
                yield df.iloc[start:start + self.chunksize]
            return

        if self.data is not None:
            for chunk in self.data:
                if not isinstance(chunk, pd.DataFrame):
                    raise TypeError(
                        "lazy iterable must yield pandas.DataFrame objects"
                    )

                yield self._post_process_chunk(chunk)
            return

    def _post_process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        对每个 chunk 做统一后处理。

        当前主要处理：
        - selected_columns
        """
        if self.selected_columns is None:
            return chunk

        return self._select_dataframe_columns(chunk, self.selected_columns)

    def to_dataframe(
            self,
            *,
            cache: bool = True,
            ignore_index: bool = True,
    ) -> pd.DataFrame:
        """
        转成完整 DataFrame。

        注意：
        -----------------------------------------------------------------------
        如果数据是懒加载 chunk，这个方法会读取所有 chunk 到内存。

        参数：
        -----------------------------------------------------------------------
        cache:
            是否缓存读取结果。

            对大型数据集，如果只想临时读取，可以设置：
                cache=False

        ignore_index:
            concat chunk 时是否重置索引。
        """
        if cache and self._cached_dataframe is not None:
            return self._cached_dataframe

        if isinstance(self.data, pd.DataFrame):
            df = self._post_process_chunk(self.data)

            if cache:
                self._cached_dataframe = df

            return df

        chunks = list(self.iter_chunks())

        if not chunks:
            df = pd.DataFrame()
        else:
            df = pd.concat(chunks, ignore_index=ignore_index)

        if cache:
            self._cached_dataframe = df

        return df

    def materialize(
            self,
            *,
            cache: bool = True,
            ignore_index: bool = True,
    ) -> pd.DataFrame:
        """
        显式物化 lazy 数据。

        等价于 to_dataframe()，
        但名字更明确。
        """
        return self.to_dataframe(
            cache=cache,
            ignore_index=ignore_index,
        )

    def iter_dict_records(
            self,
            *,
            max_rows: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        按行流式输出 dict。

        适合大型数据集：
        -----------------------------------------------------------------------
        不需要一次性把整个 DataFrame 转成列表。

        参数：
        -----------------------------------------------------------------------
        max_rows:
            最多输出多少行。
            None 表示不限制。
        """
        emitted = 0

        for chunk in self.iter_chunks():
            safe_chunk = _json_safe_dataframe(chunk)

            for record in safe_chunk.to_dict(orient="records"):
                yield {str(key): value for key, value in record.items()}
                emitted += 1

                if max_rows is not None and emitted >= max_rows:
                    return

    # -------------------------------------------------------------------------
    # View
    # -------------------------------------------------------------------------

    def view_info(self) -> DataViewInfo:
        """
        查看数据结构信息。

        注意：
        -----------------------------------------------------------------------
        lazy 模式下不会强制读取全部数据。
        """
        row_count, column_count = self.shape

        # 列数直接从列名取，懒加载也能拿到
        if column_count is None:
            column_count = len(self.columns) if self.columns else None

        columns = [
            ColumnInfo(name=name, dtype=self.dtypes.get(name, "unknown"))
            for name in self.columns
        ]

        return DataViewInfo(
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            added_columns=[],
            dropped_columns=[],
            source_path=(
                str(self.source_path)
                if self.source_path is not None
                else None
            ),
            file_type=self.file_type,
            is_lazy=self.lazy,
            name=self.name,
            description=self.description,
            source_type=self.source_type,
            chunksize=self.chunksize,
            shape=self.shape,
            dtypes=self.dtypes,
            index_name=self._index_name(),
            processed_by=self.processed_by,
            created_at=self.created_at.isoformat(),
            metadata=self.metadata,
        )

    def _index_name(self) -> Optional[str]:
        """
        获取索引名。
        """
        if self._cached_dataframe is not None:
            return (
                str(self._cached_dataframe.index.name)
                if self._cached_dataframe.index.name is not None
                else None
            )

        if isinstance(self.data, pd.DataFrame):
            return (
                str(self.data.index.name)
                if self.data.index.name is not None
                else None
            )

        return None

    def view(self, n: int = 5) -> pd.DataFrame:
        """
        查看前 n 行。

        对普通 DataFrame：
            直接 head。

        对 lazy 数据：
            尽量只读取第一个 chunk。

        注意：
        -----------------------------------------------------------------------
        如果底层是 TextFileReader 且没有 reader_factory，
        读取第一个 chunk 会消费 reader 的一部分。
        """
        if n < 0:
            raise ValueError("n cannot be negative")

        if self._cached_dataframe is not None:
            return self._cached_dataframe.head(n)

        if isinstance(self.data, pd.DataFrame):
            return self.data.head(n)

        for chunk in self.iter_chunks():
            return chunk.head(n)

        return pd.DataFrame()

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        查看前 n 行。
        """
        return self.view(n)

    def sample_rows(self, n: int = 5) -> pd.DataFrame:
        """
        获取样例行。

        对 lazy 数据不会读取全部，只从第一个 chunk 中取。
        """
        return self.view(n)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def to_dict(
            self,
            *,
            max_rows: Optional[int] = None,
            materialize: bool = True,
    ) -> Dict[str, Any]:
        """
        转成字典。

        参数：
        -----------------------------------------------------------------------
        max_rows:
            最多导出多少行。
            对大型数据集建议设置，例如：
                max_rows=1000

        materialize:
            是否允许完整读取 lazy 数据。

            True:
                会调用 to_dataframe()，可能读取全部数据。

            False:
                只导出 view() 的部分数据，不读取全部数据。
        """
        if materialize:
            df = self.to_dataframe()
        else:
            df = self.view(max_rows or 5)

        safe_df = _json_safe_dataframe(df, max_rows=max_rows)

        return {
            "name": self.name,
            "description": self.description,
            "source_path": (
                str(self.source_path)
                if self.source_path is not None
                else None
            ),
            "source_type": self.source_type,
            "file_type": _safe_file_type_value(self.file_type),
            "lazy": self.lazy,
            "is_lazy": self.lazy,
            "chunksize": self.chunksize,
            "shape": list(df.shape),
            "columns": [str(col) for col in safe_df.columns.tolist()],
            "index": list(safe_df.index),
            "data": safe_df.values.tolist(),
            "dtypes": {
                str(col): str(dtype)
                for col, dtype in df.dtypes.items()
            },
            "metadata": self.metadata,
            "processed_by": self.processed_by,
            "processed_data": (
                self.processed_data.to_dict()
                if self.processed_data is not None
                else None
            ),
            "special_data": {
                name: special.to_dict()
                for name, special in self.special_data.items()
            },
            "created_at": self.created_at.isoformat(),
            "export_limited": max_rows is not None,
            "export_max_rows": max_rows,
            "materialized": materialize,
        }

    def to_json(
            self,
            *,
            ensure_ascii: bool = False,
            indent: Optional[int] = 2,
            max_rows: Optional[int] = None,
            materialize: bool = True,
    ) -> str:
        """
        转成 JSON 字符串。

        大型数据集建议：
        -----------------------------------------------------------------------
        data.to_json(max_rows=1000, materialize=False)
        """
        return json.dumps(
            self.to_dict(
                max_rows=max_rows,
                materialize=materialize,
            ),
            ensure_ascii=ensure_ascii,
            indent=indent,
            default=_json_default,
        )

    # -------------------------------------------------------------------------
    # Processed data
    # -------------------------------------------------------------------------

    def set_processed_data(
            self,
            data: Union[pd.DataFrame, ProcessedTabularData, ProcessedTableData],
            *,
            name: Optional[str] = None,
            processed_by: Optional[str] = None,
            description: Optional[str] = None,
            max_rows: Optional[int] = None,
    ) -> TabularData:
        """
        设置处理后的数据。

        支持：
        -----------------------------------------------------------------------
        1. pandas.DataFrame
        2. ProcessedTabularData
        3. ProcessedTableData 旧别名

        参数：
        -----------------------------------------------------------------------
        name:
            处理后数据名称。

            如果 data 是 DataFrame，则 ProcessedTabularData.from_dataframe()
            需要 name，因此这里会自动兜底：
                name or self.name or "processed_data"

        processed_by:
            处理模块名称。

        description:
            处理后数据说明。

        max_rows:
            从 DataFrame 构造 ProcessedTabularData 时最多保存多少行。

            对大型数据集建议设置，比如：
                max_rows=1000
        """
        final_name = name or self.name or "processed_data"

        if isinstance(data, pd.DataFrame):
            metadata: Dict[str, Any] = {}

            if processed_by is not None:
                metadata["processed_by"] = processed_by

            self.processed_data = ProcessedTabularData.from_dataframe(
                data,
                name=final_name,
                description=description,
                metadata=metadata,
                include_index=True,
                max_rows=max_rows,
            )

        elif isinstance(data, ProcessedTabularData):
            self.processed_data = data

            if processed_by is not None:
                self.processed_data.metadata["processed_by"] = processed_by

        else:
            raise TypeError(
                "data must be pandas.DataFrame or ProcessedTabularData"
            )

        if processed_by is not None:
            self.processed_by = processed_by

        return self

    def make_special_data(
            self,
            *,
            name: str,
            df: Optional[pd.DataFrame] = None,
            processed_by: Optional[str] = None,
            description: Optional[str] = None,
            process_params: Optional[Dict[str, Any]] = None,
            annotations: Optional[Dict[str, Any]] = None,
            statistics: Optional[Dict[str, Any]] = None,
            model_outputs: Optional[Dict[str, Any]] = None,
            extra_metadata: Optional[Dict[str, Any]] = None,
            max_rows: Optional[int] = None,
            materialize: bool = True,
    ) -> SpecialData:
        """
        创建并添加 SpecialData。

        如果 df 不传：
        -----------------------------------------------------------------------
        - materialize=True:
            使用当前完整数据，会触发 lazy 数据物化。

        - materialize=False:
            只使用 view(max_rows or 5)，不会读取全部数据。

        参数说明：
        -----------------------------------------------------------------------
        name:
            SpecialData 名称。

        df:
            要包装成特殊数据的 DataFrame。

        processed_by:
            处理模块名称。

        description:
            特殊数据说明。

        process_params:
            处理参数。

        annotations:
            标注信息。

        statistics:
            统计信息。

        model_outputs:
            模型输出。

        extra_metadata:
            额外元信息。

        max_rows:
            最多保存多少行到 ProcessedTabularData。
            对大型数据集建议设置。

        materialize:
            df 不传时是否允许物化完整数据。
        """
        if df is None:
            if materialize:
                df = self.to_dataframe()
            else:
                df = self.view(max_rows or 5)

        metadata = dict(extra_metadata or {})

        processed = ProcessedTabularData.from_dataframe(
            df,
            name=name,
            description=description,
            metadata=metadata,
            include_index=True,
            max_rows=max_rows,
        )

        special = SpecialData.from_processed_tabular_data(
            processed,
            name=name,
            description=description,
            data_type="processed_result",
            processed_by=processed_by,
            metadata=metadata,
            parameters=process_params or {},
        )

        if annotations:
            special.update_annotations(annotations)

        if statistics:
            special.update_statistics(statistics)

        if model_outputs:
            special.update_model_output(model_outputs)

        self.add_special_data(special)
        return special

    # -------------------------------------------------------------------------
    # Metadata helpers
    # -------------------------------------------------------------------------

    def update_metadata(self, metadata: Mapping[str, Any]) -> TabularData:
        """
        更新 metadata。
        """
        self.metadata.update(dict(metadata))
        return self

    def set_processed_by(self, processed_by: Optional[str]) -> TabularData:
        """
        设置处理模块名称。
        """
        self.processed_by = processed_by
        return self

    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------

    def copy_with(
            self,
            *,
            data: Any = _UNSET,
            name: Any = _UNSET,
            description: Any = _UNSET,
            source_path: Any = _UNSET,
            source_type: Any = _UNSET,
            file_type: Any = _UNSET,
            lazy: Any = _UNSET,
            chunksize: Any = _UNSET,
            selected_columns: Any = _UNSET,
            metadata: Any = _UNSET,
            processed_data: Any = _UNSET,
            processed_by: Any = _UNSET,
            special_data: Any = _UNSET,
    ) -> TabularData:
        """
        复制并修改部分字段。

        使用 _UNSET 的原因：
        -----------------------------------------------------------------------
        copy_with(selected_columns=None)
            表示主动清空列选择。

        copy_with()
            表示保持原值不变。
        """
        copied = replace(
            self,
            data=self.data if data is _UNSET else data,
            name=self.name if name is _UNSET else name,
            description=(
                self.description
                if description is _UNSET
                else description
            ),
            source_path=(
                self.source_path
                if source_path is _UNSET
                else source_path
            ),
            source_type=(
                self.source_type
                if source_type is _UNSET
                else source_type
            ),
            file_type=(
                self.file_type
                if file_type is _UNSET
                else file_type
            ),
            lazy=self.lazy if lazy is _UNSET else lazy,
            chunksize=(
                self.chunksize
                if chunksize is _UNSET
                else chunksize
            ),
            selected_columns=(
                self.selected_columns
                if selected_columns is _UNSET
                else selected_columns
            ),
            metadata=(
                self.metadata.copy()
                if metadata is _UNSET
                else metadata
            ),
            processed_data=(
                self.processed_data
                if processed_data is _UNSET
                else processed_data
            ),
            processed_by=(
                self.processed_by
                if processed_by is _UNSET
                else processed_by
            ),
            special_data=(
                self.special_data.copy()
                if special_data is _UNSET
                else special_data
            ),
        )

        copied._cached_dataframe = None
        copied._reader_factory = self._reader_factory
        # 新增：重置列结构缓存
        copied._cached_columns = None
        copied._cached_dtypes = None

        return copied

    def copy(self) -> TabularData:
        """
        复制当前对象。
        """
        return self.copy_with()

    # -------------------------------------------------------------------------
    # Python protocol helpers
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """
        返回数据行数。

        lazy 数据在未物化之前无法确定长度。
        """
        row_count = self.shape[0]

        if row_count is None:
            raise TypeError(
                "length is unknown for lazy data before materialization"
            )

        return int(row_count)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        """
        默认按 chunk 迭代。
        """
        return self.iter_chunks()

    def __repr__(self) -> str:
        """
        调试显示。
        """
        return (
            "TabularData("
            f"name={self.name!r}, "
            f"source_type={self.source_type!r}, "
            f"file_type={_safe_file_type_value(self.file_type)!r}, "
            f"lazy={self.lazy!r}, "
            f"chunksize={self.chunksize!r}, "
            f"shape={self.shape!r}"
            ")"
        )
