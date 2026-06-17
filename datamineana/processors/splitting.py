# datamineana/processors/splitting.py

from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

from .base import BaseProcessor
from datamineana.dataobject import TabularData


class DataSplitter(BaseProcessor):
    name = "data_splitter"

    def random_split(
            self,
            data: TabularData,
            train_size: float = 0.7,
            valid_size: float = 0.15,
            test_size: float = 0.15,
            random_state: Optional[int] = 42,
            shuffle: bool = True,
    ) -> Tuple[TabularData, TabularData, TabularData]:
        total = train_size + valid_size + test_size
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train_size + valid_size + test_size 必须等于 1")

        df = data.to_dataframe()

        report = self._new_report(
            step="random_split",
            params={
                "train_size": train_size,
                "valid_size": valid_size,
                "test_size": test_size,
                "random_state": random_state,
                "shuffle": shuffle,
            },
            before_shape=df.shape,
            materialized=True,
        )

        if shuffle:
            df2 = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        else:
            df2 = df.reset_index(drop=True)

        n = len(df2)
        train_end = int(n * train_size)
        valid_end = train_end + int(n * valid_size)

        train_df = df2.iloc[:train_end].copy()
        valid_df = df2.iloc[train_end:valid_end].copy()
        test_df = df2.iloc[valid_end:].copy()

        report.after_shape = {
            "train": train_df.shape,
            "valid": valid_df.shape,
            "test": test_df.shape,
        }
        report.statistics = {
            "total_rows": int(n),
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "test_rows": int(len(test_df)),
        }
        report.finish()

        return (
            self._wrap_result(data, train_df, report, "train"),
            self._wrap_result(data, valid_df, report, "valid"),
            self._wrap_result(data, test_df, report, "test"),
        )

    def _wrap_result(
            self,
            source: TabularData,
            df: pd.DataFrame,
            report,
            split_name: str,
    ) -> TabularData:
        metadata = dict(source.metadata or {})
        process_reports = list(metadata.get("process_reports", []))
        process_reports.append(report.to_dict())
        metadata["process_reports"] = process_reports
        metadata["split_name"] = split_name

        return TabularData.from_dataframe(
            df,
            name=f"{source.name}_{split_name}" if source.name else split_name,
            description=source.description,
            source_path=source.source_path,
            source_type=source.source_type,
            file_type=source.file_type,
            metadata=metadata,
            processed_by=self.name,
        )
