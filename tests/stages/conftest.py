"""Fixtures for the stage tests.

⚠️ **Stage modules are NOT importable as ``src.stages.X``.** Every stage begins with ``from __init__ import root_path``,
which resolves only when ``src/stages/`` is itself on ``sys.path`` — because stages are standalone CLI scripts run from
within that directory, not package modules. ``import src.stages.setup`` raises ``ModuleNotFoundError: No module named
'__init__'``.

That is the convention, not a defect: the bare ``__init__`` import is what triggers the automatic ``PIPELINE_SEED``
hook (see ``init_test.py``). So the tests put ``src/stages/`` on the path the same way the orchestrator does, and import
each stage by its bare module name.
"""
import os
import sys

import pytest

STAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'src', 'stages')

# At MODULE scope, not in a fixture: the test files import their stage at module level, which pytest executes during
# COLLECTION — before any fixture, even an autouse session-scoped one, has run.
if STAGES_DIR not in sys.path:
    sys.path.insert(0, STAGES_DIR)


@pytest.fixture(scope='session')
def stages_dir():
    return STAGES_DIR
