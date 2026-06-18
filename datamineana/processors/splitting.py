# datamineana/processors/splitting.py
"""
# DataSplitter 简要概述
## 一、类定位
继承统一基类 `BaseProcessor`，属于**数据集划分处理器**，使用新版标准化报告体系，专门用来将完整数据集切分为训练集、验证集、测试集，用于机器学习模型训练、验证、测试流程。

## 二、核心方法
### 1. `random_split`（主对外方法）
1. 校验三个数据集划分比例之和必须为1，否则抛出异常；
2. 将 `TabularData` 转为 DataFrame，通过基类 `_new_report` 创建标准处理报告，记录划分参数、原始数据集行列；
3. 支持随机打乱（`shuffle`），通过随机种子保证实验可复现；
4. 按照设定比例切分出训练、验证、测试三份子数据集；
5. 在报告中记录三份数据集各自的行列大小、总样本量、各子集样本行数，调用 `finish()` 结束本次报告；
6. 调用内部封装方法 `_wrap_result`，分别生成三个带完整元数据与处理日志的 `TabularData` 对象并返回。

### 2. `_wrap_result`（内部私有封装方法）
1. 拷贝原始数据集的元数据，将本次划分的处理报告追加到 `process_reports` 实现链路追溯；
2. 在元数据中标记当前子集名称（train/valid/test）；
3. 自动拼接数据集名称，例如 `data_train`，保留原数据集的路径、描述、文件类型等属性；
4. 封装生成新的 `TabularData` 子集并返回。

## 三、设计特点
1. 完全接入新版报告体系：自动归入处理器历史报告列表，支持全流程报告汇总查看；
2. 划分过程可复现：通过 `random_state` 固定随机种子，保证每次切分结果一致；
3. 数据链路可追溯：每个子集都挂载本次划分的完整报告，可回溯划分比例、原始数据规模；
4. 统一封装逻辑：内置 `_wrap_result` 复用封装规则，保证三个子集的元数据、命名、日志格式完全规范统一。

## 四、配套兼容说明
你在顶层入口 `DataPreprocessor` 中对外暴露了同名兼容方法 `train_valid_test_split`，内部实际调用本类的 `random_split`，实现旧调用方式平滑迁移到新架构。
"""
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
