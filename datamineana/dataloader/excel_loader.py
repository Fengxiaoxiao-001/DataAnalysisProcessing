# datamineana/dataloader/excel_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import pandas as pd

from datamineana.dataobject import TabularData
from datamineana.dataobject import ColumnSelector, FileType, PathLike, RowSelector
from datamineana.dataloader.base import BaseDataLoader


class BaseExcelFileLoader(BaseDataLoader):
    """
    Excel 文件读取基类。

    XLSXLoader 和 XLSLoader 继承它。
    """

    engine: Optional[str] = None

    def load(
        self,
        path: PathLike,
        columns: ColumnSelector = None,
        rows: RowSelector = None,
        dtype: Optional[Dict[str, Any]] = None,
        lazy: bool = False,
        chunksize: Optional[int] = None,
        sheet_name: Union[str, int] = 0,
        **kwargs: Any,
    ) -> TabularData:
        path = Path(path)

        usecols = TabularData._normalize_columns(columns)

        if lazy:
            if chunksize is None:
                chunksize = 5000

            def factory() -> Iterator[pd.DataFrame]:
                start = 0

                while True:
                    if start == 0:
                        skiprows = None
                    else:
                        # 保留表头，跳过已读取的数据行。
                        skiprows = range(1, start + 1)

                    chunk = pd.read_excel(
                        path,
                        sheet_name=sheet_name,
                        usecols=usecols,
                        dtype=dtype,
                        engine=self.engine,
                        skiprows=skiprows,
                        nrows=chunksize,
                        **kwargs,
                    )

                    if chunk.empty:
                        break

                    yield chunk

                    if len(chunk) < chunksize:
                        break

                    start += chunksize

            return TabularData(
                data=None,
                chunk_factory=factory,
                source_path=path,
                file_type=self.file_type,
            )

        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            usecols=usecols,
            dtype=dtype,
            engine=self.engine,
            **kwargs,
        )

        if rows is not None:
            df = TabularData._select_dataframe(df, rows=rows)

        return TabularData(
            data=df,
            source_path=path,
            file_type=self.file_type,
        )


class XLSXLoader(BaseExcelFileLoader):
    """
    xlsx 文件读取类。
    """

    file_type = FileType.XLSX
    engine = "openpyxl"


class XLSLoader(BaseExcelFileLoader):
    """
    xls 文件读取类。

    注意：
    pandas 读取 xls 一般需要 xlrd。
    """

    file_type = FileType.XLS
    engine = "xlrd"