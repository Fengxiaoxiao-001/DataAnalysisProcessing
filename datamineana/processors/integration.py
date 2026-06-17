# datamineana/processors/integration.py

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple
import pandas as pd

from .base import BaseProcessor
from datamineana.dataobject import TabularData

ConcatAxis = Literal[0, 1, "index", "columns"]
ConcatJoin = Literal["inner", "outer"]
MergeHow = Literal["left", "right", "outer", "inner", "cross"]


class DataIntegrator(BaseProcessor):
    name = "data_integrator"

    def concat(
            self,
            datasets: Sequence[TabularData],
            axis: ConcatAxis = 0,
            join: ConcatJoin = "outer",
            ignore_index: bool = True,
    ) -> TabularData:
        if not datasets:
            raise ValueError("datasets 不能为空")

        dfs = [d.to_dataframe() for d in datasets]

        report = self._new_report(
            step="concat",
            params={
                "axis": axis,
                "join": join,
                "ignore_index": ignore_index,
                "count": len(datasets),
            },
            before_shape=[df.shape for df in dfs],
            materialized=True,
        )

        result = pd.concat(
            dfs,
            axis=axis,
            join=join,
            ignore_index=ignore_index if axis in (0, "index") else False,
        )

        report.after_shape = result.shape
        report.finish()

        return self._wrap_result(datasets[0], result, report)

    def merge(
            self,
            left: TabularData,
            right: TabularData,
            how: MergeHow = "inner",
            on: Optional[Sequence[str]] = None,
            left_on: Optional[Sequence[str]] = None,
            right_on: Optional[Sequence[str]] = None,
            suffixes: Tuple[str | None, str | None] = ("_x", "_y"),
    ) -> TabularData:
        df_left = left.to_dataframe()
        df_right = right.to_dataframe()

        report = self._new_report(
            step="merge",
            params={
                "how": how,
                "on": list(on) if on is not None else None,
                "left_on": list(left_on) if left_on is not None else None,
                "right_on": list(right_on) if right_on is not None else None,
                "suffixes": suffixes,
            },
            before_shape={
                "left": df_left.shape,
                "right": df_right.shape,
            },
            materialized=True,
        )

        result = pd.merge(
            df_left,
            df_right,
            how=how,
            on=list(on) if on is not None else None,
            left_on=list(left_on) if left_on is not None else None,
            right_on=list(right_on) if right_on is not None else None,
            suffixes=suffixes,
        )

        report.after_shape = result.shape
        report.statistics = {
            "left_rows": int(df_left.shape[0]),
            "right_rows": int(df_right.shape[0]),
            "result_rows": int(result.shape[0]),
        }
        report.finish()

        return self._wrap_result(left, result, report)

    def _wrap_result(self, source: TabularData, df: pd.DataFrame, report) -> TabularData:
        metadata = dict(source.metadata or {})
        process_reports = list(metadata.get("process_reports", []))
        process_reports.append(report.to_dict())
        metadata["process_reports"] = process_reports

        return TabularData.from_dataframe(
            df,
            name=source.name,
            description=source.description,
            source_path=source.source_path,
            source_type=source.source_type,
            file_type=source.file_type,
            metadata=metadata,
            processed_by=self.name,
        )
