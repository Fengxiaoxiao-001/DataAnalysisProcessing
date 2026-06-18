# datamineana/dataloader/universal_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from datamineana.dataobject import TabularData
from datamineana.dataobject import ColumnSelector, PathLike, RowSelector
from datamineana.dataloader.base import BaseDataLoader
from datamineana.dataloader.csv_loader import CSVLoader
from datamineana.dataloader.excel_loader import XLSLoader, XLSXLoader


class ExcelLoader:
    """
    通用万能表格读取入口。

    虽然名字叫 ExcelLoader，但是可以支持：
    - csv
    - xlsx
    - xls

    你后续也可以注册：
    - parquet
    - json
    - database
    - 自定义文件格式
    """

    def __init__(self) -> None:
        self.loaders: Dict[str, BaseDataLoader] = {
            ".csv": CSVLoader(),
            ".xlsx": XLSXLoader(),
            ".xls": XLSLoader(),
        }

    def register_loader(
            self,
            suffix: str,
            loader: BaseDataLoader,
    ) -> None:
        """
        注册新的文件读取类。

        示例：
        loader.register_loader(".json", JSONLoader())
        """

        if not suffix.startswith("."):
            suffix = "." + suffix

        self.loaders[suffix.lower()] = loader

    def load(
            self,
            path: PathLike,
            columns: ColumnSelector = None,
            rows: RowSelector = None,
            dtype: Optional[Dict[str, Any]] = None,
            lazy: bool = False,
            chunksize: Optional[int] = None,
            **kwargs: Any,
    ) -> TabularData:
        path = Path(path)
        suffix = path.suffix.lower()

        if not path.exists():
            raise FileNotFoundError(f"目标文件不存在：{path.resolve()}")

        suffix = path.suffix.lower()
        if suffix not in self.loaders:
            raise ValueError(f"不支持的文件类型: {suffix}")

        loader = self.loaders[suffix]

        return loader.load(
            path=path,
            columns=columns,
            rows=rows,
            dtype=dtype,
            lazy=lazy,
            chunksize=chunksize,
            **kwargs,
        )
