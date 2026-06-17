import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from datamineana.utils import find_project_root

# 全局默认目录
DEFAULT_PATH = find_project_root() / "cache/test_data"


class TestTempCSV:
    # 形参默认给None，用来捕获用户传入None的场景
    def __init__(self,
                 data: pd.DataFrame,
                 save_dir: str | Path | None = None,
                 insert_null: bool = True,
                 null_ratio: float = 0.2
                 ) -> None:
        # 用户传None，就使用预设默认路径
        if save_dir is None:
            save_dir = DEFAULT_PATH

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # 关键：写入前随机插入空值
        if insert_null:
            mask = np.random.choice([True, False], size=data.shape, p=[null_ratio, 1 - null_ratio])
            data = data.mask(mask)

        self._tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            dir=save_dir,
            suffix=".csv",
            delete=False,
            encoding="utf-8"
        )

        self.path = Path(self._tmp_file.name)
        data.to_csv(self._tmp_file, index=False)
        self._tmp_file.close()

    def get_path(self) -> Path:
        return self.path

    def clean(self):
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
                print(f"临时文件已清理：{self.path}")
            except PermissionError:
                print(f"警告：文件被占用，清理失败 {self.path}")

    def __del__(self):
        self.clean()
