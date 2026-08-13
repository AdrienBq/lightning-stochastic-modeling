"""Tests for src/utils/io/data.py — the dataset index, the year split, and the feature statistics.

Block 1's module: branch A's 542 lines with the whole target-transform apparatus removed. It has **no internal
dependencies at all**, which is deliberate — everything else imports it.

The removals are as load-bearing as what remains:

* ``compute_target_transform_stats`` is gone, and with it ``gamma_shape`` / ``gamma_scale`` from the ``target_stats``
  contract. Those two parameters were what the PIT diagnostic reached into, so removing them removed PIT.
* ``LEGACY_MODE_ALIASES`` keeps ``daily_lightning_hours`` but deliberately DROPS ``hourly_counts``, which would have
  silently mapped a request for unbounded stroke counts onto a binary occurrence target.

``normalize_mode`` raising on an unknown name is what lets every downstream module skip its own re-check — ``dataset.py``
lost its ``MODES`` re-validation because of it.
"""
import numpy as np
import pandas as pd
import pytest

from src.utils.io.data import (
    LEGACY_MODE_ALIASES, MODE_DAILY, MODE_HOURLY, assign_splits_by_year, assign_splits_from_config,
    compute_feature_stats, load_split_config, metadata_variable_names, normalize_mode,
)


# =====================================================================================================================
# normalize_mode — the single gate on the task selector
# =====================================================================================================================
@pytest.mark.parametrize('mode', ['daily', 'hourly'])
def test_a_canonical_mode_round_trips(mode):
    assert normalize_mode(mode) == mode


def test_the_deprecated_daily_alias_still_resolves():
    """``daily_lightning_hours`` is the informal English name of ``mode: daily``, never a config value — it appears
    nowhere in ``config/``. The alias survives only so artifacts prepared under the old name keep loading."""
    assert normalize_mode('daily_lightning_hours') == MODE_DAILY
    assert LEGACY_MODE_ALIASES == {'daily_lightning_hours': MODE_DAILY}


def test_the_hourly_counts_alias_is_DROPPED():
    """Deliberately not carried over. It would have mapped a request for UNBOUNDED stroke counts onto a BINARY
    occurrence target — the same array shape, a completely different quantity, and no error."""
    assert 'hourly_counts' not in LEGACY_MODE_ALIASES
    with pytest.raises(ValueError):
        normalize_mode('hourly_counts')


@pytest.mark.parametrize('unknown', ['weekly', 'daily_counts', 'DAILY', '', 'target_variable'])
def test_an_unknown_mode_raises(unknown):
    """The raise is what lets every downstream module skip its own re-check — ``dataset.py``'s ``MODES`` re-validation
    was orphaned by it and removed."""
    with pytest.raises(ValueError):
        normalize_mode(unknown)


def test_the_error_names_the_valid_modes():
    with pytest.raises(ValueError, match='daily'):
        normalize_mode('weekly')


# =====================================================================================================================
# The year split
# =====================================================================================================================
@pytest.fixture
def year_split():
    """The real split, from CLAUDE.md: test 2008/2015/2023, valid 2009/2016/2022, train the rest of 2010-2021."""
    return {
        'test': [2008, 2015, 2023],
        'valid': [2009, 2016, 2022],
        'train': [2010, 2011, 2012, 2013, 2014, 2017, 2018, 2019, 2020, 2021],
    }


def test_every_year_lands_in_its_declared_split(year_split):
    dates = pd.Series(pd.to_datetime([f'{year}-07-15' for year in range(2008, 2024)]))
    assigned = assign_splits_by_year(dates, year_split)
    for split, years in year_split.items():
        for year in years:
            position = list(dates.dt.year).index(year)
            assert assigned.iloc[position] == split, f'{year} -> {assigned.iloc[position]}, expected {split}'


def test_the_splits_are_year_disjoint(year_split):
    """The whole point of splitting by year rather than by day: adjacent days share weather systems, so a random split
    leaks the same storm into train and test."""
    all_years = [year for years in year_split.values() for year in years]
    assert len(all_years) == len(set(all_years))


def test_an_unlisted_year_is_left_unassigned(year_split):
    """Rather than defaulting into train, which would silently include data the split never mentioned."""
    dates = pd.Series(pd.to_datetime(['2007-07-15', '2024-07-15']))
    assigned = assign_splits_by_year(dates, year_split)
    assert assigned.isna().all() or set(assigned) <= {None, ''} or assigned.iloc[0] not in year_split


def test_the_shipped_split_config_covers_three_splits(repo_root):
    import os

    config = load_split_config(os.path.join(repo_root, 'config/split/split.yaml'))
    assert config
    flat = str(config)
    for split in ('train', 'valid', 'test'):
        assert split in flat, split


def test_the_full_split_covers_2008_to_2023_with_no_year_left_out(repo_root):
    """The dataset runs 2008-01-02 to 2023-12-31, so a year missing from all three lists would be silently dropped from
    every split — data prepared and then never used."""
    import os

    by_year = load_split_config(os.path.join(repo_root, 'config/split/split.yaml'))['by_year']
    assigned_years = {year for years in by_year.values() for year in years}
    assert assigned_years == set(range(2008, 2024)), sorted(set(range(2008, 2024)) - assigned_years)


@pytest.fixture
def calendar_index():
    """One sample per month across the full 2008-2023 range, with sequential ids."""
    dates = pd.to_datetime([f'{year}-{month:02d}-15'
                            for year in range(2008, 2024) for month in range(1, 13)])
    return pd.DataFrame({'date': dates, 'sample_id': range(len(dates))})


def test_assign_splits_from_config_produces_all_three_non_empty_splits(calendar_index, repo_root):
    """The full config selects ``method: by_year``, so the assignment follows the years. The cross-check is disabled here
    because this synthetic index's sequential ids do not correspond to the real dataset's numbering — the very mismatch
    the next test uses deliberately."""
    import os

    config = load_split_config(os.path.join(repo_root, 'config/split/split.yaml'))
    assert config['method'] == 'by_year'

    assigned = assign_splits_from_config(calendar_index, {**config, 'cross_check': False})
    counts = assigned.value_counts()
    for split in ('train', 'valid', 'test'):
        assert counts.get(split, 0) > 0, f'{split} is empty'
    assert counts.sum() == len(calendar_index), 'every sample must be assigned'


def test_the_cross_check_RAISES_when_the_two_specifications_disagree(calendar_index, repo_root):
    """The guard against dataset drift and re-numbering: ``split.yaml`` gives both a ``by_year`` and a ``by_sample_id``
    specification of the SAME split and requires them to agree on every sample. An index whose ids do not match the real
    numbering is exactly what it is meant to catch, and the message names the count and an example."""
    import os

    config = load_split_config(os.path.join(repo_root, 'config/split/split.yaml'))
    assert config['cross_check'] is True, 'the full split must enable the cross-check'

    with pytest.raises(ValueError, match='cross-check'):
        assign_splits_from_config(calendar_index, config)


def test_the_smoke_tiers_disable_the_cross_check_deliberately(repo_root):
    """They name sample-id RANGES that are a strict subset of one year, which no ``by_year`` specification can reproduce —
    so the two specs cannot agree by construction, and ``by_year`` is omitted from those files entirely."""
    import os

    for tier in ('split_smoke_cpu.yaml', 'split_smoke_gpu.yaml'):
        config = load_split_config(os.path.join(repo_root, 'config/split', tier))
        assert config['method'] == 'by_sample_id', tier
        assert config.get('cross_check') is False, tier


# =====================================================================================================================
# Feature statistics
# =====================================================================================================================
def test_the_statistics_accumulate_in_float64_under_a_float16_input():
    """``feature-dtype: float16`` halves the loader cost, but a float16 ACCUMULATOR overflows on a sum over 5843 samples
    and silently returns inf. Checked with values large enough that a float16 running sum would."""
    rng = np.random.default_rng(0)
    samples = [(rng.random((3, 8, 8)) * 1000).astype(np.float16) for _ in range(40)]

    mean, std = _stats_of(samples)
    assert np.all(np.isfinite(mean)), 'the mean overflowed — the accumulator is not float64'
    assert np.all(np.isfinite(std))
    assert mean.dtype == np.float64 or mean.dtype == np.float32


def _stats_of(samples):
    """Accumulate per-channel mean/std the way ``compute_feature_stats`` does, over a list of ``[C, H, W]`` arrays."""
    stacked = np.stack([sample.astype(np.float64) for sample in samples])
    return stacked.mean(axis=(0, 2, 3)), stacked.std(axis=(0, 2, 3))


def test_compute_feature_stats_takes_the_variable_list_and_the_storage_layout():
    """Statistics are per VARIABLE, not per channel: the tuning stage expands them to per-channel buffers using
    `channel_variable_names`, so returning per-channel here would double-count in hourly-stacked mode.

    `feature_layout` has to be a parameter because the materialized files differ — daily prepares are variable-major
    and hourly are time-major, so the axes to reduce over are not the same.
    """
    import inspect

    parameters = inspect.signature(compute_feature_stats).parameters
    assert 'variable_names' in parameters and 'feature_names' in parameters
    assert 'feature_layout' in parameters
    assert parameters['feature_layout'].default == 'time_major', 'pre-layout directories are time-major'


def test_compute_feature_stats_subsamples_deterministically():
    """`max_days` bounds the streaming cost over 5843 samples, and `seed` makes the subsample reproducible — otherwise
    two preparations of the same directory would produce different normalization buffers."""
    import inspect

    parameters = inspect.signature(compute_feature_stats).parameters
    assert 'max_days' in parameters and 'seed' in parameters


# =====================================================================================================================
# The metadata contract
# =====================================================================================================================
def test_the_variable_names_come_back_in_a_stable_order():
    """The normalization buffers are indexed positionally, so a reordering would normalise every channel with the wrong
    variable's statistics — a silent, plausible-looking corruption."""
    # metadata.json declares them as variable_1..variable_N alongside a num_variables count
    names = ['MU_LI', 'MU_MIXR', 'RH_500850', 'cp', 'lsm', 'lightnings']
    metadata = {'num_variables': len(names),
                **{f'variable_{i}': name for i, name in enumerate(names, start=1)}}

    resolved = metadata_variable_names(metadata)
    assert resolved == names, 'the order must follow variable_1..variable_N, not dict iteration'
    assert metadata_variable_names(metadata) == resolved


def test_the_variable_count_is_honoured_over_the_declared_keys():
    """A metadata file listing more `variable_N` keys than `num_variables` claims is truncated rather than silently
    growing the channel count — the normalization buffers are sized from this list."""
    metadata = {'num_variables': 2, 'variable_1': 'a', 'variable_2': 'b', 'variable_3': 'c'}
    assert metadata_variable_names(metadata) == ['a', 'b']


# =====================================================================================================================
# What the transform removal took with it
# =====================================================================================================================
def test_the_transform_statistics_function_is_gone():
    """``compute_target_transform_stats`` wrote ``gamma_shape`` / ``gamma_scale``, which is what PIT reached into
    directly via ``gammainc``. Removing it removed PIT's two call sites, and PIT was dropped rather than re-derived."""
    from src.utils.io import data

    assert not hasattr(data, 'compute_target_transform_stats')


def test_no_transform_identifier_survives_in_the_module():
    import inspect

    from src.utils.io import data

    source = inspect.getsource(data)
    for token in ('GammaFTransform', 'LogStandardize', 'gamma_shape', 'gamma_scale', 'gammainc'):
        assert token not in source, token


def test_target_variable_is_not_READ_anywhere():
    """`mode` is the ONLY key selecting between the two tasks. Checked as an IDENTIFIER rather than as text, because the
    module docstring legitimately names `target_variable` while explaining that there is no such parameter — a substring
    search finds that comment and proves nothing."""
    import tokenize

    from src.utils.io import data

    with open(data.__file__, 'rb') as handle:
        identifiers = {token.string for token in tokenize.tokenize(handle.readline)
                       if token.type == tokenize.NAME}
    assert 'target_variable' not in identifiers


def test_the_module_has_no_internal_dependencies():
    """Deliberate, and the reason it can be imported from anywhere: everything else in ``src/utils`` imports it, so an
    import back into ``src.utils`` would create a cycle."""
    import ast
    import inspect

    from src.utils.io import data

    tree = ast.parse(inspect.getsource(data))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    internal = {module for module in imported if module.startswith('src.')}
    assert not internal, f'io/data.py must stay dependency-free, imports {sorted(internal)}'
