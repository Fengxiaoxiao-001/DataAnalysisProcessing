from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union, List, Tuple

# ==========================================================
# 通用路径类型
# ==========================================================

PathLike = Union[str, Path]

# ==========================================================
# 列选择器类型
# ==========================================================

ColumnSelector = Optional[
    Union[
        str,
        List[str],
        Tuple[str, ...],
    ]
]

# ==========================================================
# 行选择器类型
# ==========================================================

RowSelector = Optional[
    Union[
        int,
        slice,
        List[int],
        Tuple[int, ...],
        range,
    ]
]


class FileType(str, Enum):
    """
    文件类型。
    """

    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    UNKNOWN = "unknown"


class SaveMode(str, Enum):
    """
    数据保存模式。

    NEW_FILE:
        新文件保存。

    MODIFY_ORIGINAL:
        在原文件上覆盖保存。

    PATCH_FILE:
        生成一个新增文件。
        原文件 + 新增文件 = 处理后的文件。

    DATABASE:
        存入数据库。
    """

    NEW_FILE = "new_file"
    MODIFY_ORIGINAL = "modify_original"
    PATCH_FILE = "patch_file"
    DATABASE = "database"


class StoreMode(str, Enum):
    """
    TabularData 内部数据存放模式。

    REPLACE:
        替换原有数据。

    APPEND_ROWS:
        按行追加数据。

    APPEND_COLUMNS:
        按列追加数据。
    """

    REPLACE = "replace"
    APPEND_ROWS = "append_rows"
    APPEND_COLUMNS = "append_columns"
