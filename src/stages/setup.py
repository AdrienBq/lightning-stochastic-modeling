"""Script to set up the directories to store intermediate results
"""
from __init__ import root_path

import os
import shutil

from fire import Fire


def make_dirs(dir_path: str, hard_clean: bool = False) -> None:
    """Create a directory, optionally wiping it clean first.

    Args:
        dir_path: Absolute or relative path of the directory to create.
        hard_clean: If ``True`` and the directory already exists, delete it and
            all its contents before recreating it. Default: ``False``.
    """
    # removes all contents if there are any
    if hard_clean and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

    # create the directories if they do not exist
    os.makedirs(dir_path, exist_ok=True)


def setup(
        hard_clean: bool = False,
        **kwargs
):
    """Simple routine to setup the outputs directory

    Args:
        hard_clean (bool, optional): Whether to remove all pre-existing contents and folders from the outputs
        **kwargs: Additional keyword args are treated as paths where to create empty folders
    """
    for dir_name in kwargs.values():
        dir_path = os.path.join(root_path, dir_name)

        make_dirs(dir_path, hard_clean=hard_clean)


if __name__ == '__main__':
    Fire(setup)
