from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

# =============================================================================
# 基础 JSON 类型定义
# =============================================================================
#
# 这些类型主要用于 metadata / parameters / annotations / statistics 等字段。
#
# 可保存的数据类型包括：
# - None
# - bool
# - int
# - float
# - str
# - list
# - dict
#
# 注意：
# 如果你要保存 pandas.DataFrame / numpy.ndarray / datetime 等复杂对象，
# 建议先转换为普通 Python 类型，或者交给本文件提供的 _to_jsonable() 做转换。
# =============================================================================

JSONScalar: TypeAlias = Union[None, bool, int, float, str]
JSONValue: TypeAlias = Union[
    JSONScalar,
    List["JSONValue"],
    Dict[str, "JSONValue"],
]

JSONDict: TypeAlias = Dict[str, JSONValue]

# =============================================================================
# 表格型处理结果相关类型
# =============================================================================
#
# ProcessedTabularData 用于保存“处理后的表格数据”。
#
# 它不是 pandas.DataFrame，但结构类似：
#
# columns:
#     列名，例如：
#     ["feature", "importance"]
#
# index:
#     行名，例如：
#     ["row_1", "row_2"]
#
# data:
#     二维数据，例如：
#     [
#         ["age", 0.82],
#         ["income", 0.75],
#     ]
#
# 这样设计的好处：
# - 不强制依赖 pandas
# - 更容易 JSON 序列化
# - 适合传给画图模块
# - 适合保存模型输出、统计结果、特征重要性、聚合结果等
# =============================================================================

CellValue: TypeAlias = Union[None, bool, int, float, str]
RowValue: TypeAlias = List[CellValue]
TableValue: TypeAlias = List[RowValue]


def _utc_now_iso() -> str:
    """
    返回当前 UTC 时间字符串。

    使用字符串而不是 datetime 对象，主要是为了：
    - 方便 JSON 序列化
    - 避免不同时区对象带来的兼容问题
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_jsonable(value: Any) -> Any:
    """
    将常见 Python / pandas / numpy 对象转换成可 JSON 序列化的对象。

    支持：
    - None / bool / int / float / str
    - list / tuple / set
    - dict
    - datetime
    - dataclass 对象中自定义的 to_dict()
    - pandas.DataFrame / pandas.Series
    - numpy 标量 / ndarray

    如果遇到未知对象，则使用 str(value) 兜底。

    说明：
    这个函数不会原地修改对象。
    """

    if value is None:
        return None

    if isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except TypeError:
            pass

    # pandas.DataFrame / pandas.Series 兼容处理
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.DataFrame):
            return {
                "columns": [str(col) for col in value.columns],
                "index": [str(idx) for idx in value.index],
                "data": _to_jsonable(value.values.tolist()),
            }

        if isinstance(value, pd.Series):
            return {
                "name": None if value.name is None else str(value.name),
                "index": [str(idx) for idx in value.index],
                "data": _to_jsonable(value.tolist()),
            }
    except Exception:
        pass

    # numpy 兼容处理
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return _to_jsonable(value.tolist())

        if isinstance(value, np.generic):
            return _to_jsonable(value.item())
    except Exception:
        pass

    if isinstance(value, Mapping):
        return {
            str(k): _to_jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_jsonable(item)
            for item in value
        ]

    return str(value)


@dataclass
class ProcessedTabularData:
    """
    处理后的表格型数据。

    这个类用于保存已经处理过、方便后续画图或展示的二维表格数据。

    它类似于：
    - Excel 表
    - pandas.DataFrame
    - SQL 查询结果
    - 模型输出表
    - 统计结果表

    但它本身不依赖 pandas。

    常见用途：
    --------------------------------------------------------------------------
    1. 保存特征重要性：

        ProcessedTabularData(
            name="feature_importance",
            columns=["feature", "importance"],
            data=[
                ["age", 0.82],
                ["income", 0.75],
            ],
        )

    2. 保存分组统计：

        ProcessedTabularData(
            name="group_statistics",
            columns=["group", "count", "mean"],
            data=[
                ["A", 120, 0.52],
                ["B", 95, 0.48],
            ],
        )

    3. 保存模型预测结果：

        ProcessedTabularData(
            name="prediction_result",
            columns=["id", "label", "score"],
            data=[
                [1, "positive", 0.91],
                [2, "negative", 0.12],
            ],
        )

    字段说明：
    --------------------------------------------------------------------------
    name:
        表格数据名称。
        例如："feature_importance"、"statistics_result"。

    columns:
        列名列表。
        例如：["feature", "importance"]。

    data:
        二维数据。
        外层列表表示行，内层列表表示单行数据。
        例如：
        [
            ["age", 0.82],
            ["income", 0.75],
        ]

    index:
        可选行名。
        如果不传，则默认为 None。
        例如：["row_1", "row_2"]。

    description:
        数据说明。

    metadata:
        额外元信息。
        例如：
        {
            "source": "model_x",
            "created_by": "feature_importance_module"
        }

    注意：
    --------------------------------------------------------------------------
    对于大型数据集，不建议把完整原始数据都塞进这个类。
    这个类更适合保存：
    - 处理后的结果
    - 聚合结果
    - 统计摘要
    - 绘图需要的小型二维数据

    如果你需要保存大型原始表格，请使用 TabularData 的懒加载能力。
    """

    name: str

    columns: List[str] = field(default_factory=list)

    data: TableValue = field(default_factory=list)

    index: Optional[List[str]] = None

    description: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        """
        初始化后进行基础校验和格式整理。
        """

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ProcessedTabularData.name 必须是非空字符串。")

        self.name = self.name.strip()

        self.columns = [str(col) for col in self.columns]

        if self.index is not None:
            self.index = [str(idx) for idx in self.index]

        self._validate_shape()

    @property
    def n_rows(self) -> int:
        """
        行数。
        """
        return len(self.data)

    @property
    def n_columns(self) -> int:
        """
        列数。
        """
        return len(self.columns)

    @property
    def shape(self) -> Tuple[int, int]:
        """
        表格形状，格式为：

        (行数, 列数)
        """
        return self.n_rows, self.n_columns

    def _validate_shape(self) -> None:
        """
        校验数据形状是否合理。

        规则：
        - 如果 columns 非空，则每一行的长度应该等于 columns 长度。
        - 如果 index 非空，则 index 长度应该等于 data 行数。
        """

        if self.columns:
            expected_columns = len(self.columns)

            for row_number, row in enumerate(self.data):
                if len(row) != expected_columns:
                    raise ValueError(
                        "ProcessedTabularData.data 行长度与 columns 数量不一致："
                        f"第 {row_number} 行长度为 {len(row)}，"
                        f"但 columns 长度为 {expected_columns}。"
                    )

        if self.index is not None and len(self.index) != len(self.data):
            raise ValueError(
                "ProcessedTabularData.index 长度与 data 行数不一致："
                f"index 长度为 {len(self.index)}，"
                f"data 行数为 {len(self.data)}。"
            )

    def iter_rows(self) -> Iterable[RowValue]:
        """
        按行迭代数据。

        对于相对较大的处理结果，可以用这个方法逐行消费，
        避免不必要的数据复制。
        """

        yield from self.data

    def head(self, n: int = 5) -> "ProcessedTabularData":
        """
        返回前 n 行数据。

        常用于预览结果。

        参数：
        ----------------------------------------------------------------------
        n:
            返回的行数，默认 5。
        """

        if n < 0:
            raise ValueError("n 不能为负数。")

        return ProcessedTabularData(
            name=f"{self.name}_head",
            columns=list(self.columns),
            data=[list(row) for row in self.data[:n]],
            index=None if self.index is None else list(self.index[:n]),
            description=self.description,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为普通 dict，方便 JSON 序列化。
        """

        return {
            "name": self.name,
            "columns": _to_jsonable(self.columns),
            "data": _to_jsonable(self.data),
            "index": _to_jsonable(self.index),
            "description": self.description,
            "metadata": _to_jsonable(self.metadata),
            "created_at": self.created_at,
            "shape": self.shape,
        }

    def to_json(
            self,
            *,
            ensure_ascii: bool = False,
            indent: Optional[int] = 2,
    ) -> str:
        """
        转换为 JSON 字符串。

        参数：
        ----------------------------------------------------------------------
        ensure_ascii:
            是否转义非 ASCII 字符。
            默认 False，中文会正常显示。

        indent:
            JSON 缩进。
            默认 2。
            如果希望压缩输出，可传 None。
        """

        return json.dumps(
            self.to_dict(),
            ensure_ascii=ensure_ascii,
            indent=indent,
        )

    def to_dataframe(self) -> Any:
        """
        转换为 pandas.DataFrame。

        注意：
        ----------------------------------------------------------------------
        这个方法需要安装 pandas。

        对于大型数据：
        - 不建议频繁调用此方法
        - 因为它会构造完整 DataFrame
        """

        import pandas as pd

        return pd.DataFrame(
            self.data,
            columns=self.columns if self.columns else None,
            index=self.index,
        )

    @classmethod
    def from_dataframe(
            cls,
            dataframe: Any,
            *,
            name: str,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            include_index: bool = True,
            max_rows: Optional[int] = None,
    ) -> "ProcessedTabularData":
        """
        从 pandas.DataFrame 创建 ProcessedTabularData。

        参数：
        ----------------------------------------------------------------------
        dataframe:
            pandas.DataFrame 对象。

        name:
            结果名称。

        description:
            数据说明。

        metadata:
            额外元信息。

        include_index:
            是否保存 DataFrame 的 index。

        max_rows:
            最大保存行数。
            如果传 None，则保存全部行。
            对大型数据集建议传一个合理值，例如 1000。
        """

        if max_rows is not None:
            if max_rows < 0:
                raise ValueError("max_rows 不能为负数。")
            dataframe = dataframe.head(max_rows)

        columns = [str(col) for col in dataframe.columns]
        data = _to_jsonable(dataframe.values.tolist())

        index = None
        if include_index:
            index = [str(idx) for idx in dataframe.index]

        return cls(
            name=name,
            columns=columns,
            data=data,
            index=index,
            description=description,
            metadata=metadata or {},
        )


# 兼容旧命名。
# 如果其他文件里之前用了 ProcessedTableData，也不会立刻报错。
ProcessedTableData = ProcessedTabularData


@dataclass
class SpecialData:
    """
    特殊数据基础类。

    用于保存和表格数据相关，但不一定属于表格本体的数据。

    例如：
    --------------------------------------------------------------------------
    - 数据名称
    - 数据说明
    - 处理参数
    - 标注信息
    - 统计结果
    - 模型输出
    - 额外元信息
    - 处理后数据

    必填字段：
    --------------------------------------------------------------------------
    name:
        数据名称。

    新增字段：
    --------------------------------------------------------------------------
    processed_data:
        处理后数据。

        推荐使用 ProcessedTabularData。

        它类似 Excel / DataFrame，包含：
        - 行名
        - 列名
        - 具体二维数据

        这样可以方便后续：
        - 画图
        - 展示
        - 导出
        - 传递给其他处理模块

    processed_by:
        经处理模块。

        用于记录这个 SpecialData 是哪个模块处理产生的。

        例如：
        - "missing_value_analyzer"
        - "feature_importance_module"
        - "statistics_processor"
        - "model_predictor"

    字段分类说明：
    --------------------------------------------------------------------------

    1. 基础描述字段

        name:
            数据名称，必填。

        description:
            数据说明，可选。

        data_type:
            特殊数据类型，可选。
            例如：
            - "metadata"
            - "annotation"
            - "statistics"
            - "model_output"
            - "processed_result"
            - "plot_data"

        source:
            数据来源，可选。
            例如：
            - 文件路径
            - 数据库表名
            - 上游模块名称
            - API 地址

        version:
            数据版本，可选。

    2. 元信息字段

        metadata:
            额外元信息。
            适合保存不属于其他字段的补充信息。

            示例：
            {
                "project": "demo",
                "owner": "data_team",
                "stage": "experiment"
            }

        tags:
            标签列表。
            适合用于分类、检索、过滤。

            示例：
            ["train", "cleaned", "important"]

    3. 处理相关字段

        parameters:
            处理参数。
            用于记录生成这个 SpecialData 时使用了哪些参数。

            示例：
            {
                "method": "z_score",
                "threshold": 3.0
            }

        processed_by:
            处理模块名称。
            用于追踪这个特殊数据由哪个模块产生。

        processed_data:
            处理后数据。
            推荐使用 ProcessedTabularData。

    4. 标注与统计字段

        annotations:
            标注信息。
            可以保存人工标注、规则标注、异常标记等。

            示例：
            {
                "label": "abnormal",
                "reviewer": "user_a"
            }

        statistics:
            统计结果。
            适合保存均值、方差、缺失率、分布信息等。

            示例：
            {
                "row_count": 1000,
                "missing_rate": 0.03
            }

    5. 模型输出字段

        model_output:
            模型输出。
            可以保存预测结果、概率、Embedding、模型解释信息等。

            示例：
            {
                "label": "positive",
                "score": 0.91
            }

    6. 时间字段

        created_at:
            创建时间。

        updated_at:
            更新时间。

    可保存的数据类型：
    --------------------------------------------------------------------------

    推荐类型：
    - str
    - int
    - float
    - bool
    - None
    - list
    - dict
    - ProcessedTabularData

    可以兼容但不建议长期直接保存的类型：
    - pandas.DataFrame
    - pandas.Series
    - numpy.ndarray
    - datetime

    原因：
    这些类型不一定适合直接 JSON 序列化。
    本类的 to_dict() 会尽量转换它们，但更推荐你主动转换为普通 Python 类型。
    """

    name: str

    description: Optional[str] = None

    data_type: Optional[str] = None

    source: Optional[str] = None

    version: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    tags: List[str] = field(default_factory=list)

    parameters: Dict[str, Any] = field(default_factory=dict)

    annotations: Dict[str, Any] = field(default_factory=dict)

    statistics: Dict[str, Any] = field(default_factory=dict)

    model_output: Dict[str, Any] = field(default_factory=dict)

    processed_data: Optional[Any] = None

    processed_by: Optional[str] = None

    created_at: str = field(default_factory=_utc_now_iso)

    updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        """
        初始化后进行基础校验。
        """

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("SpecialData.name 必须是非空字符串。")

        self.name = self.name.strip()

        if self.description is not None:
            self.description = str(self.description)

        if self.data_type is not None:
            self.data_type = str(self.data_type)

        if self.source is not None:
            self.source = str(self.source)

        if self.version is not None:
            self.version = str(self.version)

        if self.processed_by is not None:
            self.processed_by = str(self.processed_by)

        self.tags = [str(tag) for tag in self.tags]

    def touch(self) -> None:
        """
        更新 updated_at 时间。

        当你修改 metadata / parameters / processed_data 等内容后，
        可以调用这个方法刷新更新时间。
        """

        self.updated_at = _utc_now_iso()

    def add_tag(self, tag: str) -> None:
        """
        添加标签。

        如果标签已经存在，不会重复添加。
        """

        tag = str(tag)

        if tag not in self.tags:
            self.tags.append(tag)
            self.touch()

    def remove_tag(self, tag: str) -> None:
        """
        删除标签。

        如果标签不存在，则不做任何事情。
        """

        tag = str(tag)

        if tag in self.tags:
            self.tags.remove(tag)
            self.touch()

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        """
        批量更新 metadata。
        """

        self.metadata.update(dict(metadata))
        self.touch()

    def update_parameters(self, parameters: Mapping[str, Any]) -> None:
        """
        批量更新处理参数。
        """

        self.parameters.update(dict(parameters))
        self.touch()

    def update_annotations(self, annotations: Mapping[str, Any]) -> None:
        """
        批量更新标注信息。
        """

        self.annotations.update(dict(annotations))
        self.touch()

    def update_statistics(self, statistics: Mapping[str, Any]) -> None:
        """
        批量更新统计信息。
        """

        self.statistics.update(dict(statistics))
        self.touch()

    def update_model_output(self, model_output: Mapping[str, Any]) -> None:
        """
        批量更新模型输出。
        """

        self.model_output.update(dict(model_output))
        self.touch()

    def set_processed_data(
            self,
            processed_data: Any,
            *,
            processed_by: Optional[str] = None,
    ) -> None:
        """
        设置处理后数据。

        参数：
        ----------------------------------------------------------------------
        processed_data:
            处理后的数据。

            推荐传入：
            - ProcessedTabularData

            也可以传：
            - dict
            - list
            - pandas.DataFrame
            - numpy.ndarray
            - 其他可被 _to_jsonable() 处理的对象

        processed_by:
            处理模块名称。
            如果传入，会同时更新 self.processed_by。
        """

        self.processed_data = processed_data

        if processed_by is not None:
            self.processed_by = str(processed_by)

        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为普通 dict，方便保存、传输或 JSON 序列化。

        注意：
        ----------------------------------------------------------------------
        这个方法会尝试把 processed_data、metadata 等复杂对象
        转成 JSON 友好的结构。
        """

        return {
            "name": self.name,
            "description": self.description,
            "data_type": self.data_type,
            "source": self.source,
            "version": self.version,
            "metadata": _to_jsonable(self.metadata),
            "tags": _to_jsonable(self.tags),
            "parameters": _to_jsonable(self.parameters),
            "annotations": _to_jsonable(self.annotations),
            "statistics": _to_jsonable(self.statistics),
            "model_output": _to_jsonable(self.model_output),
            "processed_data": _to_jsonable(self.processed_data),
            "processed_by": self.processed_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(
            self,
            *,
            ensure_ascii: bool = False,
            indent: Optional[int] = 2,
    ) -> str:
        """
        转换为 JSON 字符串。

        参数：
        ----------------------------------------------------------------------
        ensure_ascii:
            是否转义非 ASCII 字符。
            默认 False，中文会正常显示。

        indent:
            JSON 缩进。
            默认 2。
            如果希望输出紧凑 JSON，可传 None。
        """

        return json.dumps(
            self.to_dict(),
            ensure_ascii=ensure_ascii,
            indent=indent,
        )

    @classmethod
    def from_processed_tabular_data(
            cls,
            processed_data: ProcessedTabularData,
            *,
            name: Optional[str] = None,
            description: Optional[str] = None,
            data_type: str = "processed_result",
            processed_by: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            parameters: Optional[Dict[str, Any]] = None,
    ) -> "SpecialData":
        """
        从 ProcessedTabularData 快速创建 SpecialData。

        适合这种场景：
        - 某个模块已经生成了一个处理后的二维结果
        - 想把它包装成 SpecialData
        - 继续交给后续模块使用
        """

        return cls(
            name=name or processed_data.name,
            description=description or processed_data.description,
            data_type=data_type,
            metadata=metadata or {},
            parameters=parameters or {},
            processed_data=processed_data,
            processed_by=processed_by,
        )

    def copy(self) -> "SpecialData":
        """
        返回当前 SpecialData 的一个浅拷贝。

        注意：
        ----------------------------------------------------------------------
        processed_data 等内部复杂对象不会被深拷贝。
        如果你需要完全独立的数据，请自行使用 copy.deepcopy。
        """

        return SpecialData(
            name=self.name,
            description=self.description,
            data_type=self.data_type,
            source=self.source,
            version=self.version,
            metadata=dict(self.metadata),
            tags=list(self.tags),
            parameters=dict(self.parameters),
            annotations=dict(self.annotations),
            statistics=dict(self.statistics),
            model_output=dict(self.model_output),
            processed_data=self.processed_data,
            processed_by=self.processed_by,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
