from pathlib import Path
import os
import re


def make_abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    else:
        root_dir = Path(os.getcwd())
        return os.path.join(root_dir, *[s for s in re.split(r"[\|\\|/|//]+", path)])


def check_file_exists(path: str) -> bool:
    return os.path.isfile(path)


def check_dir_exists(path: str) -> bool:
    return os.path.isdir(os.path.dirname(path))


def check_extension(path: str, extension: str) -> bool:
    return path.lower().endswith(f".{extension}")
