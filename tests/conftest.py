"""Shared fixtures for the whole suite.

``tests/`` mirrors ``src/``: every directory under ``src/`` has the same directory here, and every non-``__init__``
module has a ``<module>_test.py`` beside it. ``tests/completeness_test.py`` enforces that, so the mirror stays a
mirror rather than a layout that drifts.

Everything here is SYNTHETIC — no fixture reads ``DATA_ROOT`` or a checkpoint. The trial / ``target_stats`` /
normalization builders are the ones the Step 3 verification gates used to construct all three model families, so
they are known to produce configs the modules actually accept rather than plausible-looking ones.

Heavy imports (torch, lightning) are deferred INTO the fixtures that need them: a numpy-only test file should not pay
for a torch import, and ``--collect-only`` should stay fast.
"""
import os
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:                       # belt and braces: pytest's rootdir insert depends on how it was
    sys.path.insert(0, REPO_ROOT)                   # invoked, and `pytest tests/utils/metrics/scores_test.py` differs

# ---------------------------------------------------------------------------------------------------------------------
# The grid. Most tests use SMALL_GRID for speed; anything asserting geographic behaviour needs the real 101 x 149,
# because the map extents and the 55 N display crop are defined against it.
# ---------------------------------------------------------------------------------------------------------------------
SMALL_H, SMALL_W = 16, 24
FULL_H, FULL_W = 101, 149
IN_CHANNELS = 5                                     # MU_LI, MU_MIXR, RH_500850, cp, lsm (no upstream channel)

UNET = {
    'base_channels': 8, 'depth': 2, 'kernel_size': 3, 'blocks_per_level': 1, 'upsampling': 'bilinear_conv',
    'dropout': 0.0, 'normalization': 'group', 'activation': 'relu', 'bottleneck_attention': False,
}
FLOW = {'n_steps': 4, 'hidden_dim': 128, 'n_blocks': 2, 'num_heads': 4, 'patch_size': 2}
OPTIMIZER = {'name': 'adamw', 'lr': 1e-3, 'weight_decay': 1e-5, 'scheduler': 'cosine', 'finetune_lr_factor': 0.1}
NO_CALIBRATION = {'occurrence': 'none', 'regression': {'structure': 'none', 'objective': 'pointwise'}}


@pytest.fixture(scope='session')
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope='session')
def metrics_config():
    """The shipped shared metric suite, parsed. Tests assert against the REAL config, not a fixture copy of it —
    that is what catches a config/code drift rather than agreeing with itself."""
    with open(os.path.join(REPO_ROOT, 'config/eval/metrics.yaml')) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope='session')
def search_spaces():
    """``{family: parsed search_space.yaml}`` for all three families."""
    spaces = {}
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        with open(os.path.join(REPO_ROOT, f'config/{family}/search_space.yaml')) as handle:
            spaces[family] = yaml.safe_load(handle)
    return spaces


@pytest.fixture
def normalization():
    return {'mean': [0.0] * IN_CHANNELS, 'std': [1.0] * IN_CHANNELS}


@pytest.fixture
def unet_trial():
    """Factory for a deterministic-U-net trial dict, in the Step 2 config shape.

    Note the shape traps this encodes, each of which was a live ``KeyError`` at some point in Step 3: ``batch_size``
    and ``output_activation`` and ``max_hours`` are TOP LEVEL (not under ``unet`` or ``optimizer``), the learning
    rate is ``optimizer.lr`` (not ``learning_rate``), and ``calibration.occurrence`` is a string rather than a
    nested ``enabled`` flag.
    """
    def build(loss=None, calibration=None, **overrides):
        trial = {
            'loss': loss or {'name': 'weighted_mae', 'intensity_weight_gamma': 0.0},
            'unet': dict(UNET),
            'output_activation': 'softplus',
            'max_hours': 24,
            'calibration': calibration or {'occurrence': 'none',
                                           'regression': {'structure': 'none', 'objective': 'pointwise'}},
            'optimizer': dict(OPTIMIZER),
            'batch_size': 4,
            'max_epochs': 2,
        }
        trial.update(overrides)
        return trial
    return build


@pytest.fixture
def mc_trial(unet_trial):
    """Factory for an MC-dropout trial: the U-net trial plus the top-level ``dropout_p`` and the ``finetuning``
    section. ``dropout_p`` is top level and is injected into ``unet.dropout`` by the module — left undone, every MC
    member is identical and the spread is silently zero."""
    def build(finetuning=True, dropout_p=0.2, **overrides):
        return unet_trial(
            dropout_p=dropout_p,
            finetuning={'enabled': finetuning, 'loss': 'almost_fair_crps', 'loss_weight': 0.5, 'beta': 0.7,
                        'samples': 4, 'max_epochs': 2},
            **overrides
        )
    return build


@pytest.fixture
def diffusion_trial():
    """Factory for a flow-matching trial. ``flow.n_blocks`` (not ``depth``) and ``flow.n_steps`` (not
    ``num_sampling_steps``) are the Step 2 names; there is deliberately no ``log_warp`` and no ``residual_target``."""
    def build(**overrides):
        trial = {
            'flow': dict(FLOW),
            'max_hours': 24,
            'optimizer': {key: value for key, value in OPTIMIZER.items() if key != 'finetune_lr_factor'},
            'batch_size': 4,
            'loss': {'name': 'weighted_mae', 'intensity_weight_gamma': 0.0},
        }
        trial.update(overrides)
        return trial
    return build


@pytest.fixture
def target_stats():
    """Factory for the preparation stage's ``target_stats``. ``mode`` is the ONLY key selecting the task, and there is
    no ``gamma_shape`` / ``gamma_scale`` — the F-transform is gone, so training space == evaluation space."""
    def build(mode='daily', residual_target=False, **overrides):
        stats = {'mode': mode, 'residual_target': residual_target}
        stats.update(overrides)
        return stats
    return build


# ---------------------------------------------------------------------------------------------------------------------
# Synthetic fields. The daily target is BOUNDED 0-24 integer lightning-hours and ~99.9 % zero; fixtures that ignore
# either property produce fields no score in this repo was designed for.
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def daily_field():
    """Factory for a sparse bounded ``[N, H, W]`` lightning-hours field: integer valued, 0-24, mostly zero."""
    def build(n=4, height=SMALL_H, width=SMALL_W, seed=0, active_fraction=0.05, max_hours=24):
        rng = np.random.default_rng(seed)
        field = np.zeros((n, height, width), dtype=np.float64)
        active = rng.random((n, height, width)) < active_fraction
        field[active] = rng.integers(1, max_hours + 1, size=int(active.sum()))
        return field
    return build


@pytest.fixture
def hourly_field():
    """Factory for a binary ``[N, H, W]`` occurrence field (the hourly task's observation)."""
    def build(n=4, height=SMALL_H, width=SMALL_W, seed=0, base_rate=0.02):
        rng = np.random.default_rng(seed)
        return (rng.random((n, height, width)) < base_rate).astype(np.float64)
    return build


@pytest.fixture
def batch():
    """Factory for a ``(x, y)`` torch batch, or ``(x, y, upstream)`` in residual mode."""
    def build(batch_size=2, channels=IN_CHANNELS, height=SMALL_H, width=SMALL_W, residual=False, seed=0):
        import torch
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(batch_size, channels, height, width, generator=generator)
        y = torch.clamp(torch.randn(batch_size, height, width, generator=generator) * 3.0, min=0.0)
        if residual:
            upstream = torch.clamp(torch.randn(batch_size, height, width, generator=generator) + 1.0, min=0.0)
            return x, y, upstream
        return x, y
    return build


@pytest.fixture
def prepared_split(tmp_path):
    """Factory for a synthetic PREPARED DIRECTORY: target ``.npy`` files on disk plus the split index that indexes
    them. Returns ``(split_index, prepared_config)``.

    A good part of ``evaluation.py`` is dataset-level rather than array-level — ``build_baselines``,
    ``climatology_brier``, ``climatology_conditional_mae`` and the ``_climatology_tables`` behind them take a split
    index and read the target files themselves — so those cannot be tested with bare arrays at all.

    The split is by YEAR, matching the real one: the train years feed the climatology and the valid year is what gets
    evaluated. Dates are spread across the calendar because the daily climatology is a per-day-of-year mean smoothed
    circularly over a window, so items clustered in one week would leave nearly every day-of-year bin empty.
    """
    import pandas as pd

    def build(mode='daily', train_years=(2010, 2011), eval_year=2016, days_per_year=24,
              height=SMALL_H, width=SMALL_W, hours_per_day=24, seed=0):
        rng = np.random.default_rng(seed)
        target_dir = os.path.join(str(tmp_path), 'targets')
        os.makedirs(target_dir, exist_ok=True)

        rows = []
        for split, years in (('train', train_years), ('valid', (eval_year,))):
            for year in years:
                # spread across the year so the day-of-year climatology has populated bins
                dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', periods=days_per_year)
                for date in dates:
                    shape = (hours_per_day, height, width) if mode == 'hourly' else (height, width)
                    target = np.zeros(shape, dtype=np.float32)
                    active = rng.random(shape) < 0.05
                    ceiling = 2 if mode == 'hourly' else 25       # hourly targets are 0/1 occurrence
                    target[active] = rng.integers(1, ceiling, size=int(active.sum()))
                    path = os.path.join(target_dir, f'{date.date()}.npy')
                    np.save(path, target)
                    if mode == 'hourly':
                        rows.extend({'date': date, 'hour': hour, 'target_file': path, 'split': split}
                                    for hour in range(hours_per_day))
                    else:
                        rows.append({'date': date, 'hour': np.nan, 'target_file': path, 'split': split})

        split_index = pd.DataFrame(rows)
        prepared_config = {'mode': mode, 'hours_per_day': hours_per_day}
        return split_index, prepared_config
    return build


@pytest.fixture
def save_checkpoint(tmp_path):
    """Persist a module the way Lightning would — ``state_dict`` + ``hyper_parameters`` + whatever marker the family
    writes in ``on_save_checkpoint`` — so ``registry.load_model_module`` can round-trip it. Returns the path."""
    def build(module, name='model.ckpt'):
        import lightning
        import torch
        checkpoint = {
            'state_dict': module.state_dict(),
            'hyper_parameters': dict(module.hparams),
            'pytorch-lightning_version': lightning.__version__,
        }
        module.on_save_checkpoint(checkpoint)
        path = os.path.join(str(tmp_path), name)
        torch.save(checkpoint, path)
        return path
    return build
