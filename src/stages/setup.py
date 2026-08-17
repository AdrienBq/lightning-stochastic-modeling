"""Pipeline stage: create the output directories every later stage writes into.

First stage of every pipeline, cheap and idempotent — which is why ``run.py`` never caches it (a cached ``setup`` in a
fresh checkout would skip the one stage that makes the directories exist).

⚠️ It also carries the ONE guard against an unset ``{{$OUTPUT_ROOT}}``. ``parse_config`` substitutes an unset variable
to the EMPTY STRING rather than erroring, so ``'{{$OUTPUT_ROOT}}/mc_dropout/prepared'`` becomes
``/mc_dropout/prepared`` — absolute, at the filesystem root — and ``os.path.join(root_path, ...)`` then discards the
repo root entirely. Without the check the symptom is a bare ``PermissionError`` naming a directory nobody asked for,
several lines into a stage that had nothing to do with the mistake.

The check lives HERE rather than in ``parse_config`` on purpose: ``UPSTREAM_MODEL`` *relies* on the empty-string
behaviour (unset means "no warm start", read that way by both ``tune`` and ``prepare_modeling``), so a blanket raise on
any unset variable would break both stochastic families. And it lives here rather than in each stage because this one
runs first and already receives every directory in the tree.
"""
from __init__ import root_path

import os
import shutil

from fire import Fire


def looks_like_an_unset_root(dir_name: str) -> bool:
    """True when ``dir_name`` looks like the product of an unset ``{{$VAR}}`` rather than a real absolute path.

    The discriminator is the TOP-LEVEL segment. A deliberate absolute path (``/scratch/aburq/lightning-outputs``) sits
    under an existing mount (``/scratch``); an unset variable leaves a first segment that is a project or family name
    and does not exist (``/mc_dropout/prepared`` -> ``/mc_dropout``). Creating a new top-level directory needs root, so
    the false-positive case is one nobody can execute anyway.
    """
    if not os.path.isabs(dir_name):
        return False
    segments = [segment for segment in dir_name.split(os.sep) if segment]
    if not segments:
        return True                                              # the variable was the WHOLE path
    return not os.path.isdir(os.path.join(os.sep, segments[0]))


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
    """Create the pipeline's output directories.

    Args:
        hard_clean (bool, optional): Whether to remove all pre-existing contents and folders from the outputs.
        **kwargs: Additional keyword args are treated as paths where to create empty folders.

    Raises:
        ValueError: If a path looks like the product of an unset environment variable (see
            :func:`looks_like_an_unset_root`) — raised for EVERY offending key at once, since they all have the same
            cause and fixing them one run at a time would be tedious.
    """
    unset = {name: value for name, value in kwargs.items() if looks_like_an_unset_root(str(value))}
    if unset:
        listing = '\n'.join(f'    {name}: {value!r}' for name, value in sorted(unset.items()))
        raise ValueError(
            f'setup: {len(unset)} output path(s) resolve to a non-existent top-level directory:\n{listing}\n'
            f'That is what an UNSET environment variable looks like — `{{{{$VAR}}}}` substitutes to the EMPTY '
            f'STRING rather than erroring, so `\'{{{{$OUTPUT_ROOT}}}}/family/prepared\'` becomes '
            f'`/family/prepared`. Export OUTPUT_ROOT (and DATA_ROOT) before running, e.g.\n'
            f'    export OUTPUT_ROOT=/scratch/$USER/lightning-outputs\n'
            f'Pass an absolute path under an existing mount if this was deliberate.'
        )

    for dir_name in kwargs.values():
        dir_path = os.path.join(root_path, dir_name)

        make_dirs(dir_path, hard_clean=hard_clean)


if __name__ == '__main__':
    Fire(setup)
