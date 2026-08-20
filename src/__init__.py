"""The process bootstrap: everything that must be true before any ``src.`` code runs, applied once on first import.

Every entry point reaches this module — ``run_project.py`` imports ``console_handler`` from it, each stage reaches it
through ``src/stages/__init__.py``, and ``pytest`` and ``scripts/*`` import ``src.`` packages directly — which is what
makes it the one place to put process-wide setup instead of four. It already mutated ``sys.path``; the three
portability calls below are the same class of side effect, and each is a no-op when it has nothing to do.

Order matters in one place only: ``.env`` is loaded BEFORE anything reads a variable, since it exists precisely so
``DATA_ROOT`` / ``OUTPUT_ROOT`` need not be exported by hand. See ``src/utils/io/environment.py`` for why each call is
here and what breaks without it.
"""
import os
import sys
import logging

# we add to PATH the root folder
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # path to directory containing the script
sys.path.append(root_path)

from src.utils.io.environment import load_env_file, prepend_interpreter_to_path, use_bundled_cartopy_data

load_env_file(os.path.join(root_path, '.env'))          # the per-user paths; an exported variable always wins
prepend_interpreter_to_path()                           # mlflow shells out to a bare `python` (see the module docstring)
use_bundled_cartopy_data(root_path)                     # plot offline from the committed coastline

# The `src` logger owns the console handler for the whole library. Attaching it HERE, once, is what makes
# `src.utils.*` diagnostics visible: those modules define no handler of their own, so before this they propagated to
# the root logger and were written to a file nobody reads. The level lives here too — the loggers themselves are at
# NOTSET, so this is what makes their `logger.info` calls pass the effective-level check at all.
#
# ⚠️ The stage scripts are NOT covered by this. A stage runs as `python src/stages/<stage>.py`, so its `__name__` is
# `__main__` and its logger sits outside the `src.` hierarchy; each therefore keeps its own `addHandler` call. Adding
# one here as well would emit every stage record twice.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

_library_logger = logging.getLogger(__name__)
_library_logger.setLevel(logging.INFO)
_library_logger.addHandler(console_handler)
