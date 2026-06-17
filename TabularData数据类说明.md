### TabularData 全方法说明
TabularData 继承自 `ColumnNormalizeMixin`、`ChunkMixin`、`SpecialDataMixin` 三个混入类，以下为全部方法（含内部方法、内置方法、静态方法）的功能、输入与输出说明。

---

#### 一、构造类方法（类方法，用于创建实例）
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| from_dataframe | 从 pandas DataFrame 创建实例，非懒加载模式 | 必填：`df: pd.DataFrame`<br>可选：`name`/`description`/`source_path`/`source_type`/`file_type`/`metadata`/`processed_by`/`selected_columns` | 返回 `TabularData` 实例，自动缓存底层 DataFrame |
| from_chunks | 从分块读取器/ DataFrame 迭代器创建实例，懒加载模式 | 必填：`chunks: TextFileReader / Iterable[pd.DataFrame]`<br>可选：`name`/`description`/`source_path`/`source_type`/`file_type`/`chunksize`/`metadata`/`selected_columns` | 返回懒加载 `TabularData` 实例 |
| empty | 创建空白 TabularData 实例 | 可选：`name`/`description`/`metadata` | 返回空 `TabularData` 实例，底层为空白 DataFrame |

---

#### 二、属性访问（只读 property）
| 属性名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| shape | 获取数据形状（行数, 列数） | 无 | `tuple[Optional[int], Optional[int]]`，懒加载未物化时行数为 None |
| columns | 获取列名列表 | 无 | `List[str]`，懒加载优先读缓存，无缓存则读取首个分块获取 |
| dtypes | 获取各列字段类型映射 | 无 | `Dict[str, str]`，键为列名，值为字段类型字符串 |
| row_names | 获取行索引列表 | 无 | `List[Any]`，懒加载未物化时返回空列表 |
| column_names | 获取列名列表（columns 别名） | 无 | `List[str]` |
| values | 获取全量数据二维列表 | 无 | `List[List[Any]]`，懒加载会触发全量物化，大数据集慎用 |

---

#### 三、分块与数据物化方法
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| set_reader_factory | 设置可重复生成的分块读取工厂，解决 TextFileReader 只能消费一次的问题 | 必填：`factory: Callable[[], Iterator[pd.DataFrame]]`（无参、返回分块迭代器的函数） | 返回 `TabularData` 自身，支持链式调用 |
| iter_chunks | 按分块迭代返回 DataFrame | 无 | `Iterator[pd.DataFrame]`，自动应用 `selected_columns` 列过滤 |
| to_dataframe | 转换为完整 DataFrame，懒加载触发全量读取 | 可选：`cache: bool = True`（是否缓存结果）、`ignore_index: bool = True`（拼接分块是否重置索引） | 返回完整 `pd.DataFrame` |
| materialize | 显式物化懒加载数据，语义等价于 to_dataframe | 可选：`cache: bool = True`、`ignore_index: bool = True` | 返回完整 `pd.DataFrame` |
| iter_dict_records | 按行流式输出字典记录，无需一次性加载全量数据 | 可选：`max_rows: Optional[int] = None`（最大输出行数） | `Iterator[Dict[str, Any]]`，每行对应字段名到值的映射 |
| _post_process_chunk | 【内部方法】对每个分块做统一后处理（列筛选） | 必填：`chunk: pd.DataFrame` | 返回处理后的 `pd.DataFrame` |

---

#### 四、数据预览方法
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| view_info | 获取数据结构概览，不含全量数据，懒加载不强制物化 | 无 | 返回 `DataViewInfo` 实例，包含行列数、列信息、来源、元数据等结构信息 |
| view | 预览前 n 行数据，懒加载仅读取首个分块 | 必填：`n: int = 5`（预览行数，不可为负） | 返回前 n 行 `pd.DataFrame` |
| head | 预览前 n 行，等价于 view | 必填：`n: int = 5` | 返回前 n 行 `pd.DataFrame` |
| sample_rows | 获取样例行，懒加载仅取首个分块前 n 行 | 必填：`n: int = 5` | 返回 n 行样本 `pd.DataFrame` |
| _index_name | 【内部方法】获取索引名称 | 无 | `Optional[str]`，无索引名则返回 None |

---

#### 五、导出与序列化方法
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| to_dict | 转换为字典格式 | 可选：`max_rows: Optional[int] = None`（最大导出行数）、`materialize: bool = True`（是否允许全量物化） | 返回 `Dict[str, Any]`，包含元数据、结构、数据内容等完整信息 |
| to_json | 转换为 JSON 字符串 | 可选：`ensure_ascii: bool = False`、`indent: Optional[int] = 2`、`max_rows`、`materialize` | 返回 JSON 格式字符串 |

---

#### 六、扩展数据管理方法（含 SpecialDataMixin 混入）
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| set_processed_data | 绑定处理后的结构化结果 | 必填：`data: pd.DataFrame / ProcessedTabularData / ProcessedTableData`<br>可选：`name`/`processed_by`/`description`/`max_rows` | 返回 `TabularData` 自身，支持链式调用 |
| make_special_data | 创建并自动挂载 SpecialData 扩展数据 | 必填：`name: str`（数据名称）<br>可选：`df`/`processed_by`/`description`/`process_params`/`annotations`/`statistics`/`model_outputs`/`extra_metadata`/`max_rows`/`materialize` | 返回创建的 `SpecialData` 实例，同时自动加入 `special_data` 字典 |
| add_special_data | 添加特殊数据对象 | 必填：`special: SpecialData` | 无返回值 |
| get_special_data | 根据名称获取特殊数据 | 必填：`name: str` | `Optional[SpecialData]`，不存在返回 None |
| remove_special_data | 删除指定名称的特殊数据 | 必填：`name: str` | `Optional[SpecialData]`，返回被删除对象，不存在返回 None |
| list_special_data | 列出所有特殊数据的名称 | 无 | `List[str]` |
| clear_special_data | 清空所有特殊数据 | 无 | 无返回值 |

---

#### 七、元数据与对象复制方法
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| update_metadata | 更新元数据字典 | 必填：`metadata: Mapping[str, Any]` | 返回 `TabularData` 自身，支持链式调用 |
| set_processed_by | 设置处理模块名称标记 | 必填：`processed_by: Optional[str]` | 返回 `TabularData` 自身，支持链式调用 |
| copy_with | 复制对象并按需覆写字段，通过 `_UNSET` 区分「未传参」和「主动设为 None」 | 可选：所有类字段参数，未传则保留原值 | 返回新的 `TabularData` 实例，自动重置数据缓存 |
| copy | 完整复制当前对象 | 无 | 返回新的 `TabularData` 实例 |

---

#### 八、列/行选择工具（ColumnNormalizeMixin 混入，内部静态方法）
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| _normalize_columns | 标准化列选择参数格式 | 必填：`columns: ColumnSelector`（支持 None/str/int/序列/可调用对象） | 返回 `None \| Callable[[str], bool] \| List[str] \| List[int]` |
| _normalize_rows | 标准化行选择参数格式 | 必填：`rows: RowSelector`（支持 None/str/int/序列/可调用对象） | 返回 `None \| Callable[[Any], bool] \| List[Any]` |
| _select_dataframe_columns | 按规则从 DataFrame 筛选列 | 必填：`df: pd.DataFrame`、`columns: ColumnSelector` | 返回筛选后的 `pd.DataFrame` |
| _select_dataframe_rows | 按规则从 DataFrame 筛选行 | 必填：`df: pd.DataFrame`、`rows: RowSelector` | 返回筛选后的 `pd.DataFrame` |

---

#### 九、内置特殊方法
| 方法名 | 功能说明 | 输入要求 | 输出要求 |
|--------|----------|----------|----------|
| __post_init__ | dataclass 初始化后自动执行：处理路径、推断文件类型、应用列筛选 | 无（dataclass 自动调用） | 无返回值 |
| __len__ | 支持 `len()` 语法获取行数 | 无 | 返回 `int` 行数；懒加载未物化时抛出 `TypeError` |
| __iter__ | 支持直接迭代对象，默认按分块迭代 | 无 | `Iterator[pd.DataFrame]`，等价于 `iter_chunks()` |
| __repr__ | 调试时的字符串展示 | 无 | 返回摘要字符串，包含名称、来源、加载模式、形状等信息 |