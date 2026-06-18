# datamineana/processors/integration.py
"""
一、类定位
继承 BaseProcessor，属于数据集成处理器，负责多张表的合并、拼接以及冗余字段清理，每次操作自动生成标准化处理报告，同时向下兼容旧代码的 last_report 读取逻辑。
二、三个核心方法功能
concat 数据集拼接
支持多张 TabularData 纵向（按行堆叠）、横向（按列拼接）合并，可设置内连接 / 外连接、是否重置索引；记录每个输入数据集原始形状、拼接后数据形状，处理报告存入元数据实现链路追溯。
merge 表关联合并
实现两张数据表的左右 / 内 / 外 / 交叉连接，支持指定关联键、左右单独键、重名字段后缀；统计两张表原始行数与合并后结果行数，适合多表主键关联场景。
remove_redundant_columns 删除冗余列
两种删除方式：手动指定待删列、自动删除内容完全重复的列；记录被删除的字段列表，兼容旧版报告格式与自定义数据集名称，用于合并后清理重复冗余字段。
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Iterable
import pandas as pd

from .base import BaseProcessor
from .utils import ensure_tabular, to_dataframe, ProcessReport
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

    def remove_redundant_columns(
            self,
            data: TabularData,
            *,
            columns: Optional[Iterable[str]] = None,
            duplicate_content: bool = True,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        删除冗余字段。

        columns:
        - 手动指定需要删除的列

        duplicate_content:
        - 是否自动删除内容完全相同的重复列
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        # ===== 新体系：标准报告创建 =====
        report = self._new_report(
            step="remove_redundant_columns",
            params={
                "columns": list(columns) if columns else None,
                "duplicate_content": duplicate_content,
            },
            before_shape=df.shape,
            materialized=True,
        )

        dropped = []
        if columns:
            cols = [c for c in columns if c in result.columns]
            result = result.drop(columns=cols)
            dropped.extend(cols)

        if duplicate_content:
            # 优化：用转置+duplicated替代双重循环，性能更优
            duplicated_mask = result.T.duplicated(keep="first")
            duplicated_cols = duplicated_mask[duplicated_mask].index.tolist()
            if duplicated_cols:
                result = result.drop(columns=duplicated_cols)
                dropped.extend(duplicated_cols)

        # ===== 新体系：报告指标与收尾 =====
        report.after_shape = result.shape
        report.statistics = {"dropped_columns": dropped}
        report.finish()

        # ===== 兼容旧版 self.last_report =====
        self.last_report = ProcessReport(
            module="integration",
            method="remove_redundant_columns",
            params=report.params,
            before_shape=df.shape,
            after_shape=result.shape,
            materialized=True,
        )
        self.last_report.add_metric("dropped_columns", dropped)

        # ===== 新体系：结果封装 + 兼容name参数 =====
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

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
