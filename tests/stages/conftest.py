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
# Shared here rather than in one test file because `prepare_modeling` produces what `evaluate` consumes: the only
# honest way to test the evaluation stage is against a directory the preparation stage actually wrote.
# =====================================================================================================================
VARIABLES = ['MU_LI', 'MU_MIXR', 'RH_500850', 'cp', 'lsm', 'lightnings']
FEATURES = 'MU_LI,MU_MIXR,RH_500850,cp,lsm'
HOURS, HEIGHT, WIDTH = 6, 8, 10


@pytest.fixture
def dataset_root(tmp_path):
    """A ``$DATA_ROOT``-shaped directory: ``metadata.json``, ``metadata.csv``, ``samples/sample_XXXXXX.pt``.

    The lightning channel carries a KNOWN per-hour stroke pattern rather than noise, so the target derivation is
    checkable by hand: cell ``(0, 0)`` gets one stroke in 3 hours (sub-threshold at ``hourly_threshold=2``) and cell
    ``(1, 1)`` five strokes in 4 hours (qualifying).
    """
    import json

    import torch

    def build(n_days=6, hours=HOURS):
        root = tmp_path / 'data'
        samples = root / 'samples'
        samples.mkdir(parents=True, exist_ok=True)

        metadata = {'num_variables': len(VARIABLES)}
        metadata.update({f'variable_{position + 1}': name for position, name in enumerate(VARIABLES)})
        (root / 'metadata.json').write_text(json.dumps(metadata))

        rows = ['date,id,num_lightnings,pixels_with_lightning']
        for day in range(n_days):
            payload = {name: torch.randn(hours, HEIGHT, WIDTH) for name in VARIABLES[:-1]}
            lightning = torch.zeros(hours, HEIGHT, WIDTH)
            lightning[:3, 0, 0] = 1.0
            lightning[:4, 1, 1] = 5.0
            payload['lightnings'] = lightning
            torch.save(payload, str(samples / f'sample_{day:06d}.pt'))
            rows.append(f'2015-07-{day + 1:02d},{day},{100 * (day + 1)},{day + 3}')
        (root / 'metadata.csv').write_text('\n'.join(rows) + '\n')
        return str(root)
    return build


@pytest.fixture
def split_config(tmp_path):
    """A ``by_sample_id`` split over the synthetic ids — the method the smoke tiers use, and the only one that can
    slice below a year."""
    def build(n_days=6):
        import yaml

        third = max(n_days // 3, 1)
        spec = {
            'method': 'by_sample_id',
            'cross_check': False,
            'by_sample_id': {
                'train': [[0, third - 1]],
                'valid': [[third, 2 * third - 1]],
                'test': [[2 * third, n_days - 1]],
            },
        }
        path = tmp_path / 'split.yaml'
        path.write_text(yaml.safe_dump(spec))
        return str(path)
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
