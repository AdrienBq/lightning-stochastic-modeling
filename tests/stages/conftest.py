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


# =====================================================================================================================
# A synthetic $DATA_ROOT and split, so a stage can be driven end to end with no real dataset.
#
# The BUILDERS live in `tests/conftest.py` (as plain functions, because `pipeline_e2e_test.py` needs them from a
# session-scoped fixture); these are the `tmp_path`-bound factories the stage tests use. The constants are re-exported
# so `from tests.stages.conftest import HOURS, ...` keeps working.
#
# Shared here rather than in one test file because `prepare_modeling` produces what `evaluate` consumes: the only
# honest way to test the evaluation stage is against a directory the preparation stage actually wrote.
# =====================================================================================================================
from tests.conftest import (                     # noqa: E402 — after the sys.path insert above, by necessity
    FEATURES, HEIGHT, HOURS, VARIABLES, WIDTH, build_dataset_root, write_split_config,
)


@pytest.fixture
def dataset_root(tmp_path):
    """A ``$DATA_ROOT``-shaped directory — see ``tests.conftest.build_dataset_root`` for the layout and the known
    stroke pattern the target derivation is checked against."""
    def build(n_days=6, hours=HOURS):
        root = tmp_path / 'data'
        root.mkdir(parents=True, exist_ok=True)
        return build_dataset_root(str(root), n_days=n_days, hours=hours)
    return build


@pytest.fixture
def split_config(tmp_path):
    """A ``by_sample_id`` split over the synthetic ids — the method the smoke tiers use, and the only one that can
    slice below a year."""
    def build(n_days=6):
        return write_split_config(str(tmp_path / 'split.yaml'), n_days=n_days)
    return build


@pytest.fixture
def prepared(dataset_root, split_config, tmp_path):
    """Run the REAL preparation stage and return ``(output_path, prepared_config, target_stats, split_index)``.

    Deliberately the real stage rather than a hand-built directory: ``evaluate`` reads `split_index.csv`,
    `target_stats.json` and `prepared_config.json` by their exact contracts, so a fixture that guessed at them would
    keep passing after the preparation stage changed one.
    """
    import json

    import pandas as pd
    import prepare_modeling

    def build(mode='daily', n_days=6, hourly_threshold=2, **overrides):
        output = str(tmp_path / f'prepared_{mode}_{hourly_threshold}')
        prepare_modeling.prepare_modeling(
            data_path=dataset_root(n_days=n_days),
            output_path=output,
            mode=mode,
            features=FEATURES,
            split_config=split_config(n_days=n_days),
            hourly_threshold=hourly_threshold,
            **overrides
        )
        with open(os.path.join(output, 'prepared_config.json')) as handle:
            prepared_config = json.load(handle)
        with open(os.path.join(output, 'target_stats.json')) as handle:
            target_stats = json.load(handle)
        split_index = pd.read_csv(os.path.join(output, 'split_index.csv'))
        return output, prepared_config, target_stats, split_index
    return build
