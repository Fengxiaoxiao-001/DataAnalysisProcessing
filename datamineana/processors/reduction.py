# datamineana/processors/reduction.py

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from datamineana.dataobject.DataSelf import TabularData
from .report import ProcessReport
from .utils import ensure_tabular, make_tabular_like, numeric_columns, safe_columns, to_dataframe


class DataReducer:
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

    def __init__(self):
        self.last_report: Optional[ProcessReport] = None

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

        report = ProcessReport(
            module="reduction",
            method="select_features",
            params={
                "method": method,
                "threshold": threshold,
                "columns": list(columns) if columns else None,
            },
            before_shape=df.shape,
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

        report.after_shape = result.shape
        report.add_metric("dropped_columns", dropped)

        import numpy as np
        corr = result[target_columns].corr().abs()
        upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        upper = corr.where(upper_mask)

        dropped = [
            column for column in upper.columns
            if bool((upper[column] > threshold).any())
        ]

        result = result.drop(columns=dropped)

        report.statistics = {
            "method": "correlation",
            "threshold": threshold,
            "dropped_columns": dropped,
            "dropped_count": len(dropped),
        }

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataReducer.select_features",
            report=report,
        )

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

        report = ProcessReport(
            module="reduction",
            method="pca",
            params={
                "columns": target_columns,
                "n_components": n_components,
                "prefix": prefix,
                "keep_non_numeric": keep_non_numeric,
            },
            before_shape=df.shape,
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

        report.after_shape = result.shape
        report.add_metric("explained_variance_ratio", model.explained_variance_ratio_.tolist())
        report.add_metric("total_explained_variance", float(model.explained_variance_ratio_.sum()))
        report.add_metric("n_components_", int(model.n_components_))

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataReducer.pca",
            report=report,
        )

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

        report = ProcessReport(
            module="reduction",
            method="sample",
            params={
                "n": n,
                "frac": frac,
                "random_state": random_state,
                "stratify_by": stratify_by,
            },
            before_shape=df.shape,
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

        report.after_shape = result.shape
        report.add_metric("sampled_rows", int(len(result)))

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataReducer.sample",
            report=report,
        )

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

        report = ProcessReport(
            module="reduction",
            method="optimize_memory",
            params={},
            before_shape=df.shape,
            materialized=True,
        )

        result = reduce_memory_usage(df)
        after_mb = estimate_memory_mb(result)

        report.after_shape = result.shape
        report.add_metric("memory_before_mb", before_mb)
        report.add_metric("memory_after_mb", after_mb)
        report.add_metric("memory_saved_mb", before_mb - after_mb)
        report.add_metric("memory_saved_rate", 0 if before_mb == 0 else (before_mb - after_mb) / before_mb)

        self.last_report = report
        return make_tabular_like(
            data,
            result,
            name=name,
            processed_by="DataReducer.optimize_memory",
            report=report,
        )
