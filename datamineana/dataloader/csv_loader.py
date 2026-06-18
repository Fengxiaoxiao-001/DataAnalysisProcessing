# datamineana/dataloader/csv_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pandas.io.parsers import TextFileReader

from datamineana.dataobject import TabularData
from datamineana.dataobject import ColumnSelector, FileType, PathLike, RowSelector
from datamineana.dataloader.base import BaseDataLoader


class CSVLoader(BaseDataLoader):
    """
    CSV 文件读取类。
    """

    file_type = FileType.CSV

    def load(
            self,
            path: PathLike,
            columns: ColumnSelector = None,
            rows: RowSelector = None,
            dtype: Optional[Dict[str, Any]] = None,
            lazy: bool = False,
            chunksize: Optional[int] = None,
            encoding: Optional[str] = None,
            **kwargs: Any,
    ) -> TabularData:
        path = Path(path)

        usecols = TabularData._normalize_columns(columns)

        if lazy:
            if chunksize is None:
                chunksize = 10000

            def factory() -> TextFileReader:
                return pd.read_csv(
                    path,
                    usecols=usecols,
                    dtype=dtype,
                    encoding=encoding,
                    chunksize=chunksize,
                    **kwargs,
                )

            # 1. 先构造 TabularData 实例，移除不存在的 chunk_factory 参数
            tabular = TabularData(
                data=None,
                lazy=True,
                chunksize=chunksize,
                selected_columns=columns,
                source_path=path,
                file_type=self.file_type,
            )

            # 2. 通过官方 API 设置可重复迭代的分块读取工厂
            tabular.set_reader_factory(factory)

            return tabular

        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype,
            encoding=encoding,
            **kwargs,
        )

        if rows is not None:
            df = TabularData._select_dataframe_rows(df, rows=rows)

        return TabularData(
            data=df,
            source_path=path,
            file_type=self.file_type,
        )
