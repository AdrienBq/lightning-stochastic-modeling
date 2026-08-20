"""Process-environment helpers for the bootstrap in ``src/__init__.py``: the per-user ``.env`` file, the stage
interpreter, and the bundled cartopy cache.

The three share one shape of bug — something machine-specific that works on the machine it was written on and fails
somewhere unrelated on the next one — which is why they live together rather than beside the code they serve.

``load_env_file`` reads the gitignored ``.env`` at the repo root, so ``DATA_ROOT`` / ``OUTPUT_ROOT`` need not be
exported by hand in every shell. ⚠️ It NEVER overrides a variable that is already set: an explicit ``export``, and the
environment slurm hands a job, must both win over a file the user may have forgotten they wrote.

``prepend_interpreter_to_path`` works around a defect in the layer above. ``mlflow.projects`` builds a stage's command
from a hardcoded literal:

    ext_to_cmd = {".py": "python", ".sh": os.environ.get("SHELL", "bash")}   # mlflow/projects/_project_spec.py

so EVERY stage subprocess resolves its interpreter from ``PATH`` rather than from the interpreter that launched the
pipeline — at both levels, ``run_project.py`` -> ``src/stages/run.py`` and ``run.py`` -> each stage. Launch by absolute
path (``/path/to/venv/bin/python run_project.py ...``) without activating the venv first and the stage dies on
``import mlflow``. The traceback names the stage, so it reads as a broken stage rather than a broken environment.

``use_bundled_cartopy_data`` points cartopy at the coastline shapefile committed under ``data/cartopy`` instead of the
per-user download cache. cartopy resolves ``CARTOPY_DATA_DIR`` into ``config['pre_existing_data_dir']``, which it
consults BEFORE attempting any download and falls back from gracefully, so pointing at the bundle removes the network
call at plot time without pinning the resolution: an extent change that needs a file we did not bundle still downloads.
"""
import os
import re
import sys
from typing import Dict, Optional

# `export FOO=bar` is accepted so a `.env` can double as something you `source` from a shell.
_EXPORT_PREFIX = re.compile(r'^export\s+')
# An unquoted value ends at the first whitespace-preceded `#`, the usual dotenv rule. Requiring the whitespace is what
# lets a path that legitimately contains `#` survive; a quoted value is taken verbatim and keeps its `#` either way.
_INLINE_COMMENT = re.compile(r'\s+#.*$')
_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def parse_env_file(text: str, source: str = '<string>') -> Dict[str, str]:
    """Parse ``KEY=VALUE`` lines into a dict, without touching ``os.environ``.

    Blank lines and whole-line ``#`` comments are skipped. An optional ``export `` prefix is stripped. The FIRST ``=``
    splits, so a value may contain ``=``. A value wrapped in matching single or double quotes is taken verbatim;
    otherwise a trailing whitespace-preceded ``# comment`` is removed and the remainder stripped.

    Args:
        text: Contents of the file.
        source: Path used in error messages — a parse error must name the file, since the caller may be a bootstrap
            with no other context to offer.

    Returns:
        Dict of the names to their values, in file order (later lines win over earlier ones).

    Raises:
        ValueError: On a line that is neither blank, a comment, nor a ``KEY=VALUE`` assignment, and on an invalid
            variable name. Raising rather than skipping is deliberate: a typo in ``.env`` that silently sets nothing is
            exactly the failure this file exists to remove.
    """
    values: Dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _EXPORT_PREFIX.sub('', raw.strip())
        if not line or line.startswith('#'):
            continue

        name, separator, value = line.partition('=')
        name = name.strip()
        if not separator:
            raise ValueError(
                f'{source}:{number}: not a KEY=VALUE assignment: {raw.strip()!r}. Comment the line out with `#` if it '
                f'is a note.'
            )
        if not _NAME.match(name):
            raise ValueError(f'{source}:{number}: {name!r} is not a valid environment variable name.')

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '\'"':
            value = value[1:-1]
        else:
            value = _INLINE_COMMENT.sub('', value).strip()
        values[name] = value
    return values


def load_env_file(path: str) -> Dict[str, str]:
    """Apply a ``.env`` file to ``os.environ``, leaving variables that are already set alone.

    A missing file is not an error — ``.env`` is optional, and the pipeline must stay runnable from a shell that
    exports the variables itself (which is how every existing launch script works).

    Args:
        path: Path to the ``.env`` file.

    Returns:
        Only the variables this call actually SET, so a caller can report what came from the file rather than from the
        shell. An already-set name is absent from the result even when the file names it.
    """
    if not os.path.isfile(path):
        return {}

    with open(path) as handle:
        parsed = parse_env_file(handle.read(), source=path)

    applied = {name: value for name, value in parsed.items() if name not in os.environ}
    os.environ.update(applied)
    return applied


def prepend_interpreter_to_path() -> Optional[str]:
    """Put the running interpreter's directory first on ``PATH`` so mlflow's bare ``python`` resolves to it.

    Idempotent: a second call is a no-op once the directory is already first, so importing ``src`` repeatedly cannot
    grow ``PATH``.

    Returns:
        The directory prepended, or ``None`` when it was already first (or ``sys.executable`` has no directory, as in
        a frozen interpreter).
    """
    directory = os.path.dirname(sys.executable)
    if not directory:
        return None

    current = os.environ.get('PATH', '')
    if current.split(os.pathsep)[0] == directory:
        return None

    os.environ['PATH'] = directory + os.pathsep + current if current else directory
    return directory


def use_bundled_cartopy_data(root: str) -> Optional[str]:
    """Point ``CARTOPY_DATA_DIR`` at the repo's bundled Natural Earth data, if it is present.

    Guarded on the directory existing so this is inert in a checkout that has not fetched the bundle (a clone without
    git-lfs, where the shapefile is a pointer file). An explicit ``CARTOPY_DATA_DIR`` wins, matching ``.env``'s rule.

    Args:
        root: Repository root.

    Returns:
        The directory set, or ``None`` when the variable was already set or the bundle is absent.
    """
    if os.environ.get('CARTOPY_DATA_DIR'):
        return None

    bundled = os.path.join(root, 'data', 'cartopy')
    if not os.path.isdir(bundled):
        return None

    os.environ['CARTOPY_DATA_DIR'] = bundled
    return bundled
