# datamineana/processors/reduction.py
"""
一、类定位
继承 BaseProcessor，属于数据规约处理器，用于从特征、样本、内存三个维度精简数据集，降低数据规模、减少冗余，提升后续建模与运算效率。每次操作自动生成标准化处理报告，同时向下兼容旧代码的 last_report 读取逻辑。
二、四个核心方法功能
select_features 特征筛选
支持三种筛选规则精简特征：
missing_rate：删除缺失率超过阈值的列；
variance：删除数值列中方差过低的弱区分特征；
correlation：剔除高相关性冗余特征，避免多重共线性；
会记录被删除的字段列表。
pca 主成分降维
对数值特征做标准化 + PCA 降维，可按指定维度数或方差保留比例压缩特征；支持保留非数值字段，输出命名为 PC 开头的新特征，记录方差解释率、最终保留主成分数量。
sample 样本抽样
支持按样本数量n、比例frac随机采样，也可基于指定列做分层抽样，保证各类别分布不变；记录抽样后的样本行数，用于缩减大数据量训练集。
optimize_memory 内存优化
调用工具函数自动向下适配数值类型、低基数对象列转 category，统计优化前后内存占用、节省内存大小与压缩率，减少数据集内存开销。
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .base import BaseProcessor
from .utils import ensure_tabular, numeric_columns, safe_columns, to_dataframe, ProcessReport
from datamineana.dataobject import TabularData


class DataReducer(BaseProcessor):
    """
    数据规约模块。

    支持：
    1. 低方差特征删除
    2. 高缺失率特征删除
    3. 高相关特征删除
    4. PCA 降维
    5. 样本采样
    6. 内存优化
    """

    name = "data_reducer"

    def select_features(
            self,
            data: TabularData,
            *,
            method: str = "missing_rate",
            threshold: float = 0.9,
            columns: Optional[Iterable[str]] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        特征筛选。

        method:
        - missing_rate: 删除缺失率大于 threshold 的列
        - variance: 删除方差小于等于 threshold 的数值列
        - correlation: 删除相关系数大于 threshold 的冗余数值列
        """
        ensure_tabular(data)
        df = to_dataframe(data)
        result = df.copy()

        # 新版体系：标准报告创建
        report = self._new_report(
            step="select_features",
            params={
                "method": method,
                "threshold": threshold,
                "columns": list(columns) if columns else None,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        if method == "missing_rate":
            target_columns = safe_columns(result, columns)
            missing_rate = result[target_columns].isna().mean()
            dropped = missing_rate[missing_rate > threshold].index.tolist()
            result = result.drop(columns=dropped)

        elif method == "variance":
            target_columns = safe_columns(result, columns) if columns else numeric_columns(result)
            variances = result[target_columns].var(numeric_only=True)
            dropped = variances[variances <= threshold].index.tolist()
            result = result.drop(columns=dropped)

        elif method == "correlation":
            import numpy as np

            target_columns = safe_columns(result, columns) if columns else numeric_columns(result)
            corr = result[target_columns].corr().abs()
            upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
            upper = corr.where(upper_mask)
            dropped = [
                column for column in upper.columns
                if any(upper[column] > threshold)
            ]
            result = result.drop(columns=dropped)

        else:
            raise ValueError(f"未知特征筛选方法: {method}")

        # 新版体系：报告指标与收尾
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "method": method,
            "threshold": threshold,
            "dropped_columns": dropped,
            "dropped_count": len(dropped),
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="reduction",
            method="select_features",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("dropped_columns", dropped)

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def pca(
            self,
            data: TabularData,
            *,
            columns: Optional[Iterable[str]] = None,
            n_components=0.95,
            prefix: str = "PC",
            keep_non_numeric: bool = True,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        PCA 降维。

        需要安装 scikit-learn:
        pip install scikit-learn
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError("使用 PCA 需要安装 scikit-learn: pip install scikit-learn") from e

        ensure_tabular(data)
        df = to_dataframe(data)

        target_columns = safe_columns(df, columns) if columns else numeric_columns(df)

        # 新版体系：标准报告创建
        report = self._new_report(
            step="pca",
            params={
                "columns": target_columns,
                "n_components": n_components,
                "prefix": prefix,
                "keep_non_numeric": keep_non_numeric,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        x = df[target_columns].copy()
        x = x.fillna(x.mean(numeric_only=True))

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        model = PCA(n_components=n_components)
        comps = model.fit_transform(x_scaled)

        pc_cols = [f"{prefix}{i + 1}" for i in range(comps.shape[1])]
        pc_df = pd.DataFrame(comps, columns=pc_cols, index=df.index)

        if keep_non_numeric:
            other = df.drop(columns=target_columns)
            result = pd.concat([other, pc_df], axis=1)
        else:
            result = pc_df

        # 新版体系：报告指标与收尾
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(model.explained_variance_ratio_.sum()),
            "n_components_": int(model.n_components_),
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="reduction",
            method="pca",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("explained_variance_ratio", model.explained_variance_ratio_.tolist())
        self.last_report.add_metric("total_explained_variance", float(model.explained_variance_ratio_.sum()))
        self.last_report.add_metric("n_components_", int(model.n_components_))

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def sample(
            self,
            data: TabularData,
            *,
            n: Optional[int] = None,
            frac: Optional[float] = None,
            random_state: Optional[int] = None,
            stratify_by: Optional[str] = None,
            name: Optional[str] = None,
    ) -> TabularData:
        """
        样本规约。

        n 和 frac 二选一。
        stratify_by 不为空时做简单分层采样。
        """
        ensure_tabular(data)
        df = to_dataframe(data)

        # 新版体系：标准报告创建
        report = self._new_report(
            step="sample",
            params={
                "n": n,
                "frac": frac,
                "random_state": random_state,
                "stratify_by": stratify_by,
            },
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        if stratify_by:
            if stratify_by not in df.columns:
                raise ValueError(f"分层列不存在: {stratify_by}")

            if frac is None:
                if n is None:
                    raise ValueError("分层采样时需要指定 n 或 frac")
                frac = n / len(df)

            result = (
                df.groupby(stratify_by, group_keys=False)
                .apply(lambda x: x.sample(frac=frac, random_state=random_state))
            )
        else:
            result = df.sample(n=n, frac=frac, random_state=random_state)

        # 新版体系：报告指标与收尾
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "sampled_rows": int(len(result)),
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="reduction",
            method="sample",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("sampled_rows", int(len(result)))

        # 新版体系：结果封装 + 兼容name参数
        result_data = self._wrap_result(data, result, report)
        if name is not None:
            result_data.name = name
        return result_data

    def optimize_memory(
            self,
            data: TabularData,
            *,
            name: Optional[str] = None,
    ) -> TabularData:
        from .utils import estimate_memory_mb, reduce_memory_usage

        ensure_tabular(data)
        df = to_dataframe(data)

        before_mb = estimate_memory_mb(df)

        # 新版体系：标准报告创建
        report = self._new_report(
            step="optimize_memory",
            params={},
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            materialized=True,
        )

        result = reduce_memory_usage(df)
        after_mb = estimate_memory_mb(result)

        # 新版体系：报告指标与收尾
        saved_mb = before_mb - after_mb
        saved_rate = 0 if before_mb == 0 else saved_mb / before_mb
        report.after_shape = (int(result.shape[0]), int(result.shape[1]))
        report.statistics = {
            "memory_before_mb": before_mb,
            "memory_after_mb": after_mb,
            "memory_saved_mb": saved_mb,
            "memory_saved_rate": saved_rate,
        }
        report.finish()

        # 兼容旧版 self.last_report
        self.last_report = ProcessReport(
            module="reduction",
            method="optimize_memory",
            params=report.params,
            before_shape=(int(df.shape[0]), int(df.shape[1])),
            after_shape=(int(result.shape[0]), int(result.shape[1])),
            materialized=True,
        )
        self.last_report.add_metric("memory_before_mb", before_mb)
        self.last_report.add_metric("memory_after_mb", after_mb)
        self.last_report.add_metric("memory_saved_mb", saved_mb)
        self.last_report.add_metric("memory_saved_rate", saved_rate)

        # 新版体系：结果封装 + 兼容name参数
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
