from pathlib import Path


def find_project_root(marker_file: str = ".gitignore") -> Path:
    current = Path(__file__).resolve()
    while True:
        if (current / marker_file).exists():
            return current
        if current.parent == current:
            # 到磁盘根目录还没找到，返回脚本所在父两级
            return Path(__file__).parent.parent
        current = current.parent
