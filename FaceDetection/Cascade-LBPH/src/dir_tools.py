import os
from pathlib import Path


def create_dir(dir_path: str) -> None:
    """Create a directory if doesn't exists

    Args:
        dir_path (str): the directory path to be created
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.exists():
        dir_path_obj.mkdir(parents=True, exist_ok=True)
