import os


def show_tree(path, prefix="", ignore={".venv", "venv", ".idea", ".git", "__pycache__", "tree.py"}):
    items = sorted([i for i in os.listdir(path) if i not in ignore])
    for index, name in enumerate(items):
        full_path = os.path.join(path, name)
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "
        print(prefix + connector + name)
        if os.path.isdir(full_path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            show_tree(full_path, new_prefix, ignore)


if __name__ == "__main__":
    print(os.path.basename(os.getcwd()))
    show_tree(os.getcwd())
