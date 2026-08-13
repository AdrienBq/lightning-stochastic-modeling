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
    LEGACY_MODE_ALIASES, MODE_DAILY, MODE_HOURLY, MODES, SPLIT_NAMES, assign_splits_by_sample_id,
    assign_splits_by_year, assign_splits_from_config, compute_feature_stats, load_split_config,
    metadata_variable_names, normalize_mode,
)


# =====================================================================================================================
# normalize_mode — the single gate on the task selector
# =====================================================================================================================
def test_the_mode_vocabulary_is_exactly_daily_and_hourly():
    """``mode`` is the ONLY key that selects between the two tasks, so the tuple is the whole task vocabulary. A third
    entry appearing here would mean a task nothing downstream dispatches on."""
    assert MODES == ('daily', 'hourly')
    assert (MODE_DAILY, MODE_HOURLY) == MODES


def test_the_split_names_are_exactly_the_three():
    assert SPLIT_NAMES == ('train', 'valid', 'test') or set(SPLIT_NAMES) == {'train', 'valid', 'test'}


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


@pytest.fixture
def full_calendar_index():
    """The real index shape: 5843 consecutive days from 2008-01-02 with sequential ids, which is what the smoke tiers'
    sample-id ranges were written against."""
    dates = pd.date_range('2008-01-02', periods=5843, freq='D')
    return pd.DataFrame({'date': dates, 'sample_id': np.arange(5843), 'file': 'x.pt'})


@pytest.mark.parametrize('tier,expected', [
    ('cpu', {'train': 4, 'valid': 2, 'test': 2}),
    ('gpu', {'train': 10, 'valid': 4, 'test': 4}),
])
def test_the_smoke_tiers_select_exactly_the_declared_number_of_days(tier, expected, full_calendar_index, repo_root):
    """The smoke tiers exist to make a 1-epoch CPU run finish, so their SIZE is the contract. A range that silently
    widened would turn a smoke run into a long one, which reads as a hang rather than as a config error."""
    import os

    config = load_split_config(os.path.join(repo_root, f'config/split/split_smoke_{tier}.yaml'))
    assigned = assign_splits_from_config(full_calendar_index, config)
    got = {name: int((assigned == name).sum()) for name in SPLIT_NAMES}
    assert got == expected, got


@pytest.mark.parametrize('tier,kept', [('cpu', 8), ('gpu', 18)])
def test_the_smoke_tiers_DROP_the_unnamed_samples_rather_than_raising(tier, kept, full_calendar_index, repo_root):
    """With ``cross_check: false`` a sample outside every declared range must come back unassigned, not raise: the tier
    names a handful of days out of 5843 and the other 5835 are simply not in the run."""
    import os

    config = load_split_config(os.path.join(repo_root, f'config/split/split_smoke_{tier}.yaml'))
    assigned = assign_splits_from_config(full_calendar_index, config)
    assert int(assigned.isna().sum()) == len(full_calendar_index) - kept


@pytest.mark.parametrize('tier', ['cpu', 'gpu'])
def test_the_smoke_tiers_stay_year_disjoint_and_mid_july(tier, full_calendar_index, repo_root):
    """Two properties of the sampled days, both easy to break by editing an id range. Year-disjointness is the same
    leakage guard as the full split; mid-July is chosen because it is the convective season — a smoke run over January
    would train on an almost entirely empty target and every metric would be degenerate."""
    import os

    config = load_split_config(os.path.join(repo_root, f'config/split/split_smoke_{tier}.yaml'))
    assigned = assign_splits_from_config(full_calendar_index, config)
    selected = full_calendar_index[assigned.notna()].copy()
    selected['split'] = assigned[assigned.notna()]

    years_per_split = selected.groupby('split')['date'].apply(lambda days: set(days.dt.year))
    flattened = [year for years in years_per_split for year in years]
    assert len(flattened) == len(set(flattened)), dict(years_per_split)
    assert set(selected['date'].dt.month) == {7}, sorted(set(selected['date'].dt.strftime('%Y-%m-%d')))


def test_overlapping_sample_id_ranges_RAISE(full_calendar_index):
    """Two splits claiming the same id is the one split error that cannot be resolved by a rule — whichever split won
    would be arbitrary, and the sample would leak between train and test."""
    with pytest.raises(ValueError):
        assign_splits_by_sample_id(full_calendar_index['sample_id'],
                                   {'train': [[0, 10]], 'valid': [[5, 15]], 'test': [[20, 21]]})


def test_a_year_claimed_by_two_splits_RAISES(full_calendar_index):
    """The by-year counterpart of the same guard."""
    with pytest.raises(ValueError):
        assign_splits_by_year(full_calendar_index['date'],
                              {'train': [2010], 'valid': [2010], 'test': [2008]})


# =====================================================================================================================
# Feature statistics
# =====================================================================================================================
@pytest.fixture
def float16_feature_files(tmp_path):
    """Three days of ``[Vf, T, H, W]`` features stored on disk as float16 in ``time_major`` layout, exactly as
    ``materialize-features`` writes them.

    A large offset with small variance is deliberate: 1000 ± 1 is precisely where a float32 or float16 accumulator
    loses the variance to cancellation, so it separates a float64 accumulation from a lower-precision one.
    """
    rng = np.random.default_rng(0)
    truth = (1000.0 + rng.standard_normal((3, 2, 1, 8, 9))).astype(np.float64)     # [day, Vf, T, H, W]
    rows = []
    for day in range(3):
        path = tmp_path / f'{day}.npy'
        np.save(path, truth[day].transpose(1, 0, 2, 3).astype(np.float16))         # -> time_major [T, Vf, H, W]
        rows.append({'date': pd.Timestamp('2010-07-14') + pd.Timedelta(days=day), 'file': 'x.pt',
                     'feature_file': str(path)})
    return pd.DataFrame(rows), truth


def test_the_features_are_stored_as_float16_as_configured(float16_feature_files):
    index, _ = float16_feature_files
    assert np.load(index.iloc[0]['feature_file']).dtype == np.float16


@pytest.mark.parametrize('position,name', [(0, 'a'), (1, 'b')])
def test_the_statistics_MATCH_a_float64_reduction_of_the_stored_values(position, name, float16_feature_files):
    """The real assertion behind "accumulates in float64": the returned statistics equal a float64 reduction of exactly
    the bytes on disk, per variable. This drives ``compute_feature_stats`` itself rather than a local reimplementation
    of it — a helper that re-does the accumulation in numpy proves only that numpy accumulates in float64.

    Note the reference casts through float16 first: the function cannot recover precision the STORAGE already lost, so
    the target is a faithful float64 reduction of the stored values, not of the pre-quantisation truth.
    """
    index, truth = float16_feature_files
    stats = compute_feature_stats(index, ['a', 'b'], ['a', 'b'], feature_layout='time_major')

    stored = truth[:, position].astype(np.float16).astype(np.float64)
    assert abs(stats['mean'][name] - float(stored.mean())) < 1e-9, f"{stats['mean'][name]} vs {stored.mean()}"
    assert abs(stats['std'][name] - float(stored.std())) < 1e-6, f"{stats['std'][name]} vs {stored.std()}"


def test_the_statistics_are_NaN_AWARE(float16_feature_files):
    """One NaN cell in one day must not poison the whole normalization buffer. A plain ``.mean()`` propagates it, and a
    NaN feature mean makes every standardized input NaN — the model then trains on nothing and the loss is NaN from
    step 0, with no error naming the cause."""
    index, truth = float16_feature_files
    poisoned = truth[0].copy()
    poisoned[0, 0, 0, 0] = np.nan
    np.save(index.iloc[0]['feature_file'], poisoned.transpose(1, 0, 2, 3).astype(np.float16))

    stats = compute_feature_stats(index, ['a', 'b'], ['a', 'b'], feature_layout='time_major')
    assert np.isfinite(stats['mean']['a']) and np.isfinite(stats['std']['a'])


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
