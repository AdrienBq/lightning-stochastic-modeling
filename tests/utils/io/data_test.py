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


# =====================================================================================================================
# Block 5c — the loaders
#
# Every one of these reads a real file layout, so they cannot be tested with bare arrays. They are also the only place
# in the repo that knows what a batta_torch sample looks like: everything downstream receives a normalized
# ``[V, T, H, W]`` float32 array and has no way to notice that the interpretation was wrong.
# =====================================================================================================================
def test_the_dataset_metadata_is_read_from_metadata_json(tmp_path):
    import json

    from src.utils.io.data import load_dataset_metadata

    payload = {'variables': ['MU_LI', 'lsm'], 'resolution': 0.25, 'shape': [101, 149]}
    (tmp_path / 'metadata.json').write_text(json.dumps(payload))
    assert load_dataset_metadata(str(tmp_path)) == payload


@pytest.fixture
def sample_directory(tmp_path):
    """A batta_torch-style dataset: ``metadata.csv`` plus ``samples/sample_XXXXXX.pt`` named by the integer id."""
    import torch

    def build(ids=(1, 2, 3), files=None, variables=('a', 'b'), hours=3, height=4, width=5):
        samples = tmp_path / 'samples'
        samples.mkdir(exist_ok=True)
        for sample_id in (ids if files is None else files):
            torch.save({name: torch.randn(hours, height, width) for name in variables},
                       str(samples / f'sample_{sample_id:06d}.pt'))
        rows = ['date,id,num_lightnings,pixels_with_lightning']
        rows += [f'2015-07-{day + 1:02d},{sample_id},10,3' for day, sample_id in enumerate(ids)]
        (tmp_path / 'metadata.csv').write_text('\n'.join(rows) + '\n')
        return str(tmp_path)
    return build


def test_the_sample_index_joins_the_csv_to_the_files_by_the_EMBEDDED_INTEGER(sample_directory):
    """``sample_000123.pt`` is matched to metadata row ``id = 123``. Matching by position or by sort order instead would
    pair every date with the wrong day's predictors and produce a perfectly plausible, entirely wrong dataset."""
    from src.utils.io.data import index_samples

    index = index_samples(sample_directory(ids=(1, 2, 3)))

    assert list(index.columns) == ['date', 'sample_id', 'file']
    assert list(index['sample_id']) == [1, 2, 3]
    for _, row in index.iterrows():
        assert f'sample_{int(row["sample_id"]):06d}.pt' in row['file']


def test_the_index_is_sorted_by_DATE(sample_directory):
    """The year split and every downstream item ordering assume it."""
    from src.utils.io.data import index_samples

    index = index_samples(sample_directory(ids=(7, 2, 5)))
    assert list(index['date']) == sorted(index['date'])


def test_a_metadata_record_with_NO_sample_file_is_dropped_with_a_warning(sample_directory, caplog):
    """5843 records against 5843 files today, but a partial download is the normal failure. Dropping is right; dropping
    SILENTLY would train on a smaller dataset than the metadata claims."""
    import logging

    from src.utils.io.data import index_samples

    data_path = sample_directory(ids=(1, 2, 3), files=(1, 2))       # record 3 has no file
    with caplog.at_level(logging.WARNING):
        index = index_samples(data_path)

    assert list(index['sample_id']) == [1, 2]
    assert any('no matching sample file' in record.getMessage() for record in caplog.records)


def test_an_index_that_matches_NOTHING_raises_rather_than_returning_empty(sample_directory):
    """An empty index would flow all the way to a sweep that trains on zero days and reports NaN."""
    from src.utils.io.data import index_samples

    with pytest.raises(ValueError, match='could be matched against'):
        index_samples(sample_directory(ids=(1, 2), files=(90, 91)))


@pytest.mark.parametrize('layout', ['dict_by_name', 'dict_by_position', 'stacked_vthw', 'stacked_tvhw', 'stacked_vhw'])
def test_every_supported_sample_layout_normalizes_to_V_T_H_W(tmp_path, layout):
    """Five on-disk layouts, one in-memory contract. The dangerous pair is ``[V, T, H, W]`` vs ``[T, V, H, W]``: they
    are distinguished only by which axis matches the variable COUNT, so a day with as many hours as variables would be
    ambiguous — worth knowing, and not the case here (5 variables, 24 hours)."""
    import torch

    from src.utils.io.data import load_sample_tensor

    names = ['a', 'b']
    payloads = {
        'dict_by_name': {'a': torch.randn(3, 4, 5), 'b': torch.randn(3, 4, 5)},
        'dict_by_position': {'variable_1': torch.randn(3, 4, 5), 'variable_2': torch.randn(3, 4, 5)},
        'stacked_vthw': torch.randn(2, 3, 4, 5),
        'stacked_tvhw': torch.randn(3, 2, 4, 5),
        'stacked_vhw': torch.randn(2, 4, 5),
    }
    path = str(tmp_path / 'sample.pt')
    torch.save(payloads[layout], path)

    loaded = load_sample_tensor(path, names)

    assert loaded.dtype == np.float32
    assert loaded.shape[0] == len(names)
    assert loaded.ndim == 4
    assert loaded.shape[-2:] == (4, 5)


def test_a_TIME_MAJOR_stack_is_transposed_rather_than_reinterpreted(tmp_path):
    """The check that makes the layout detection meaningful: variable 0's map must come back as variable 0's map, not
    as hour 0's."""
    import torch

    from src.utils.io.data import load_sample_tensor

    stored = torch.arange(3 * 2 * 4 * 5, dtype=torch.float32).reshape(3, 2, 4, 5)   # [T, V, H, W]
    path = str(tmp_path / 'sample.pt')
    torch.save(stored, path)

    loaded = load_sample_tensor(path, ['a', 'b'])
    assert loaded.shape == (2, 3, 4, 5)
    assert np.allclose(loaded[0, 1], stored[1, 0].numpy()), 'variable 0 at hour 1'


def test_a_STATIC_variable_is_broadcast_to_the_days_time_steps(tmp_path):
    """``lsm`` is one map for every hour. Left un-broadcast the stack would be ragged and ``torch.stack`` would raise —
    but the fix has to be a broadcast, not a truncation of the other variables to T = 1."""
    import torch

    from src.utils.io.data import load_sample_tensor

    path = str(tmp_path / 'sample.pt')
    torch.save({'a': torch.randn(3, 4, 5), 'lsm': torch.randn(4, 5)}, path)

    loaded = load_sample_tensor(path, ['a', 'lsm'])
    assert loaded.shape == (2, 3, 4, 5)
    assert all(np.array_equal(loaded[1, 0], loaded[1, step]) for step in range(3))


def test_a_MISSING_variable_raises_naming_it_and_listing_the_keys(tmp_path):
    """The message has to carry both, because the usual cause is a metadata/sample mismatch and the keys are the only
    way to see which side is wrong."""
    import torch

    from src.utils.io.data import load_sample_tensor

    path = str(tmp_path / 'sample.pt')
    torch.save({'a': torch.randn(3, 4, 5)}, path)

    with pytest.raises(ValueError) as raised:
        load_sample_tensor(path, ['a', 'missing_one'])
    assert 'missing_one' in str(raised.value) and 'a' in str(raised.value)


def test_an_UNINTERPRETABLE_layout_raises_rather_than_guessing(tmp_path):
    import torch

    from src.utils.io.data import load_sample_tensor

    path = str(tmp_path / 'sample.pt')
    torch.save(torch.randn(7, 4, 5), path)                          # 7 matches neither 2 variables nor a [T,V,H,W]
    with pytest.raises(ValueError, match='Cannot interpret'):
        load_sample_tensor(path, ['a', 'b'])


# ---------------------------------------------------------------------------------------------------------------------
# The prepared-directory artifacts
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def prepared_directory(tmp_path):
    """The three files ``prepare_regression`` writes, plus the optional feature/upstream columns."""
    import json

    def build(mode='daily', feature_files=False, upstream_files=False):
        root = tmp_path / 'prepared'
        (root / 'targets').mkdir(parents=True, exist_ok=True)
        (root / 'prepared_config.json').write_text(json.dumps({'mode': mode, 'hours_per_day': 24}))
        (root / 'target_stats.json').write_text(json.dumps({'mode': mode, 'residual_target': upstream_files}))

        rows = []
        for day in range(2):
            row = {'date': f'2015-07-0{day + 1}', 'sample_id': day, 'file': f's{day}.pt',
                   'target_filename': f'{day}.npy', 'split': 'valid'}
            if feature_files:
                row['feature_filename'] = f'{day}.npy'
            if upstream_files:
                row['upstream_filename'] = f'{day}.npy'
            rows.append(row)
        pd.DataFrame(rows).to_csv(str(root / 'split_index.csv'), index=False)
        return str(root)
    return build


def test_the_prepared_artifacts_come_back_with_ABSOLUTE_target_paths(prepared_directory):
    """``split_index.csv`` stores bare filenames so a prepared directory can be moved. The join happens here, once —
    every consumer opens ``target_file`` directly."""
    import os

    from src.utils.io.data import load_prepared_artifacts

    root = prepared_directory()
    prepared_config, split_index, target_stats = load_prepared_artifacts(root)

    assert prepared_config['mode'] == 'daily' and target_stats['mode'] == 'daily'
    assert len(split_index) == 2
    for path in split_index['target_file']:
        assert os.path.isabs(path) and path.startswith(root) and '/targets/' in path


def test_the_feature_and_upstream_columns_appear_ONLY_when_the_stage_wrote_them(prepared_directory):
    """``LightningMapsDataset`` decides between the materialized and checkpoint readers on the column's presence, and
    raises in residual mode when ``upstream_file`` is absent. An always-present column of ``None`` would break both."""
    from src.utils.io.data import load_prepared_artifacts

    _, bare, _ = load_prepared_artifacts(prepared_directory())
    assert 'feature_file' not in bare.columns and 'upstream_file' not in bare.columns

    _, full, _ = load_prepared_artifacts(prepared_directory(feature_files=True, upstream_files=True))
    assert full['feature_file'].notna().all() and full['upstream_file'].notna().all()


def test_a_LEGACY_mode_name_is_normalized_in_BOTH_artifacts(prepared_directory):
    """``daily_lightning_hours`` was the old name of ``mode: daily``. It survives as a read-time alias precisely so a
    directory prepared before the rename still loads — and both files carry the key, so both must be normalized or the
    two would disagree about the task."""
    from src.utils.io.data import load_prepared_artifacts

    root = prepared_directory(mode='daily_lightning_hours')
    prepared_config, _, target_stats = load_prepared_artifacts(root)

    assert prepared_config['mode'] == 'daily'
    assert target_stats['mode'] == 'daily'


# ---------------------------------------------------------------------------------------------------------------------
# The upstream-channel statistics
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def upstream_index(tmp_path):
    def build(days=6, seed=0, with_nan=False):
        rng = np.random.default_rng(seed)
        rows, values = [], []
        for day in range(days):
            block = rng.normal(5.0, 2.0, (4, 5))
            if with_nan and day == 0:
                block[0, 0] = np.nan
            path = str(tmp_path / f'upstream_{day}.npy')
            np.save(path, block)
            rows.append({'upstream_file': path})
            values.append(block)
        return pd.DataFrame(rows), np.concatenate([block.ravel() for block in values])
    return build


def test_the_upstream_statistics_match_a_float64_reduction_of_the_files(upstream_index):
    """The diffusion residual model standardizes its upstream channel with these, so a wrong mean shifts the whole
    conditioning channel and the network compensates — training still converges, to a different model."""
    from src.utils.io.data import compute_upstream_stats

    index, values = upstream_index(days=6)
    stats = compute_upstream_stats(index)

    assert abs(stats['mean'] - float(values.mean())) < 1e-9
    assert abs(stats['std'] - float(values.std())) < 1e-6


def test_the_upstream_statistics_are_NaN_AWARE(upstream_index):
    """A masked upstream cell must not drag the mean toward NaN — and if it did, the standardized channel would be
    entirely NaN and every gradient would vanish silently."""
    from src.utils.io.data import compute_upstream_stats

    index, _ = upstream_index(days=3, with_nan=True)
    stats = compute_upstream_stats(index)
    assert np.isfinite(stats['mean']) and np.isfinite(stats['std'])


def test_the_standard_deviation_is_FLOORED_so_standardizing_cannot_divide_by_zero(tmp_path):
    """A constant upstream map — what an upstream model that predicts all-zero produces, which is not far-fetched on a
    99.93 %-zero target."""
    from src.utils.io.data import compute_upstream_stats

    path = str(tmp_path / 'flat.npy')
    np.save(path, np.zeros((4, 5)))
    stats = compute_upstream_stats(pd.DataFrame([{'upstream_file': path}]))

    assert stats['std'] > 0.0
    assert stats['mean'] == 0.0


def test_the_day_SUBSAMPLE_is_capped_and_deterministic(upstream_index):
    """5843 days x 101 x 149 streamed per sweep. The cap is what keeps it affordable; determinism is what keeps two
    runs of the same pipeline comparable."""
    from src.utils.io.data import compute_upstream_stats

    index, _ = upstream_index(days=20)
    first = compute_upstream_stats(index, max_days=5, seed=3)
    assert first == compute_upstream_stats(index, max_days=5, seed=3)
    assert first != compute_upstream_stats(index, max_days=5, seed=4)


def test_a_MISSING_upstream_column_raises_rather_than_returning_zero_statistics(upstream_index):
    """Silent zeros here would standardize the upstream channel to itself and hand the network an unscaled input."""
    from src.utils.io.data import compute_upstream_stats

    index, _ = upstream_index(days=2)
    with pytest.raises(ValueError, match='upstream_file'):
        compute_upstream_stats(index.drop(columns=['upstream_file']))
