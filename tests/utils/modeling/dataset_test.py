"""Tests for src/utils/modeling/dataset.py — the prepared-artifact reader and the day-grouped sampler.

Branch A's version needed no change for the residual channel: it already appends the upstream prediction LAST, returns
the third batch item, and names the channel ``upstream`` last. So the tests here mostly pin what block 3a-doc had to
document because the naming reads backwards:

* ``feature_aggregation`` is DAILY-MODE ONLY — in hourly mode an item is already a single hour and the key is never
  read, so a nonsense value there changes nothing;
* ``hourly_stack`` is the option that does NO aggregation (``Vf * T`` channels), and ``daily_mean`` is the aggregating
  one. The value name is kept for on-disk compatibility despite reading the wrong way round;
* ``DayGroupedShuffleSampler`` IS called, from ``tuning.py`` at both fit sites, under a guard that is DORMANT today
  because every config sets ``materialize-features: true``. It reads as dead code and must not be dropped.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.utils.modeling.dataset import DayGroupedShuffleSampler, LightningMapsDataset, build_split_datasets

VARIABLES = ['MU_LI', 'MU_MIXR', 'RH_500850', 'cp', 'lsm']
HEIGHT, WIDTH = 8, 10
HOURS = 4


@pytest.fixture
def prepared(tmp_path):
    """A synthetic prepared directory: per-day target and materialized feature files, plus the split index."""
    def build(mode='daily', n_days=6, residual=False, feature_aggregation='hourly_stack', hours=HOURS,
              materialize=True, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for day in range(n_days):
            date = pd.Timestamp('2015-06-01') + pd.Timedelta(days=day)
            target_shape = (hours, HEIGHT, WIDTH) if mode == 'hourly' else (HEIGHT, WIDTH)
            target_path = str(tmp_path / f'target_{day}.npy')
            np.save(target_path, rng.integers(0, 25, target_shape).astype(np.float32))

            row = {'date': date, 'split': 'train', 'target_file': target_path}
            if materialize:
                # daily prepares are variable-major [Vf, T, H, W]; hourly are time-major [T, Vf, H, W]
                shape = (hours, len(VARIABLES), HEIGHT, WIDTH) if mode == 'hourly' \
                    else (len(VARIABLES), hours, HEIGHT, WIDTH)
                feature_path = str(tmp_path / f'feature_{day}.npy')
                np.save(feature_path, rng.standard_normal(shape).astype(np.float16))
                row['feature_file'] = feature_path
            if residual:
                upstream_shape = (hours, HEIGHT, WIDTH) if mode == 'hourly' else (HEIGHT, WIDTH)
                upstream_path = str(tmp_path / f'upstream_{day}.npy')
                np.save(upstream_path, rng.random(upstream_shape).astype(np.float32) * 4)
                row['upstream_file'] = upstream_path
            rows.append(row)

        prepared_config = {
            'mode': mode,
            'features': VARIABLES,
            'variable_names': VARIABLES,
            'hours_per_day': hours,
            'feature_aggregation': feature_aggregation,
            'feature_layout': 'time_major' if mode == 'hourly' else 'variable_major',
            'residual_target': residual,
        }
        return pd.DataFrame(rows), prepared_config
    return build


# =====================================================================================================================
# Items: one per day, or one per (day, hour)
# =====================================================================================================================
def test_daily_mode_yields_one_item_per_day(prepared):
    index, config = prepared(mode='daily', n_days=6)
    assert len(LightningMapsDataset(index, config)) == 6


def test_hourly_mode_yields_one_item_per_day_hour_pair(prepared):
    index, config = prepared(mode='hourly', n_days=6)
    assert len(LightningMapsDataset(index, config)) == 6 * HOURS


def test_the_items_frame_carries_the_hour_in_hourly_mode(prepared):
    index, config = prepared(mode='hourly', n_days=3)
    frame = LightningMapsDataset(index, config).items_frame()
    assert len(frame) == 3 * HOURS
    assert set(frame['hour']) == set(range(HOURS))


def test_the_items_frame_hour_is_nan_in_daily_mode(prepared):
    index, config = prepared(mode='daily', n_days=3)
    frame = LightningMapsDataset(index, config).items_frame()
    assert frame['hour'].isna().all()


# =====================================================================================================================
# feature_aggregation: daily-mode only, and hourly_stack does NOT aggregate
# =====================================================================================================================
def test_hourly_stack_does_NOT_aggregate(prepared):
    """It concatenates the T hourly maps onto the CHANNEL axis — 5 variables x 4 hours = 20 channels — so the network
    sees the diurnal cycle and learns its own weighting. A rename to ``daily_sum`` would name an operation that does not
    happen."""
    index, config = prepared(mode='daily', feature_aggregation='hourly_stack')
    dataset = LightningMapsDataset(index, config)
    assert dataset.in_channels == len(VARIABLES) * HOURS
    x, _ = dataset[0]
    assert x.shape == (len(VARIABLES) * HOURS, HEIGHT, WIDTH)


def test_daily_mean_is_the_aggregating_option(prepared):
    index, config = prepared(mode='daily', feature_aggregation='daily_mean')
    dataset = LightningMapsDataset(index, config)
    assert dataset.in_channels == len(VARIABLES)
    x, _ = dataset[0]
    assert x.shape == (len(VARIABLES), HEIGHT, WIDTH)


def test_the_channel_names_are_variable_major_when_stacked(prepared):
    """The tuning stage expands per-VARIABLE normalization statistics to per-CHANNEL buffers using this list, so the
    order has to match how the channels are actually laid out — a transposed list would normalise every channel with
    the wrong variable's mean."""
    index, config = prepared(mode='daily', feature_aggregation='hourly_stack')
    names = LightningMapsDataset(index, config).channel_variable_names
    assert names[:HOURS] == [VARIABLES[0]] * HOURS
    assert len(names) == len(VARIABLES) * HOURS


def test_feature_aggregation_is_IGNORED_in_hourly_mode(prepared):
    """The key is daily-mode only: in hourly mode an item is already a single hour. A nonsense value must be accepted
    and change nothing about the item — which is the claim block 3a-doc makes and could not otherwise support."""
    index, config = prepared(mode='hourly')
    sensible = LightningMapsDataset(index, config)
    nonsense = LightningMapsDataset(index, {**config, 'feature_aggregation': 'not_an_aggregation'})

    assert nonsense.in_channels == sensible.in_channels
    assert len(nonsense) == len(sensible)
    assert torch.equal(nonsense[0][0], sensible[0][0])


def test_an_unknown_aggregation_DOES_raise_in_daily_mode(prepared):
    index, config = prepared(mode='daily')
    with pytest.raises(ValueError, match='aggregation'):
        LightningMapsDataset(index, {**config, 'feature_aggregation': 'not_an_aggregation'})


# =====================================================================================================================
# Residual mode: the upstream channel is LAST and comes back as a third item
# =====================================================================================================================
def test_residual_mode_appends_the_upstream_as_the_last_channel(prepared):
    index, config = prepared(mode='daily', feature_aggregation='daily_mean', residual=True)
    dataset = LightningMapsDataset(index, config)
    assert dataset.in_channels == len(VARIABLES) + 1
    assert dataset.channel_variable_names[-1] == 'upstream'


def test_residual_mode_returns_three_batch_items(prepared):
    index, config = prepared(mode='daily', feature_aggregation='daily_mean', residual=True)
    item = LightningMapsDataset(index, config)[0]
    assert len(item) == 3
    x, target, upstream = item
    assert x.shape[0] == len(VARIABLES) + 1
    assert upstream.shape == (HEIGHT, WIDTH)


def test_the_appended_channel_IS_the_upstream_returned_separately(prepared):
    """Both, deliberately: as a conditioning channel for the network and as a separate tensor so the module can build
    the residual target and add it back at inference. They must be the same array."""
    index, config = prepared(mode='daily', feature_aggregation='daily_mean', residual=True)
    x, _, upstream = LightningMapsDataset(index, config)[0]
    assert torch.allclose(x[-1].float(), upstream, atol=1e-3)


def test_a_non_residual_dataset_returns_two_items(prepared):
    index, config = prepared(mode='daily', feature_aggregation='daily_mean', residual=False)
    assert len(LightningMapsDataset(index, config)[0]) == 2


def test_residual_mode_without_an_upstream_column_raises(prepared):
    """This is the check that catches "residual data, non-residual preparation" BEFORE the model's channel-count
    assertion does — and it names the missing column rather than reporting a shape error."""
    index, config = prepared(mode='daily', residual=False)
    with pytest.raises(ValueError, match='upstream_file'):
        LightningMapsDataset(index, {**config, 'residual_target': True})


# =====================================================================================================================
# Storage: features stay raw and narrow, the target is float32
# =====================================================================================================================
def test_features_keep_their_on_disk_dtype(prepared):
    """Deliberate: keeping them float16 halves the collate / pin-memory / host-to-device copy, which is the loader's
    dominant cost. The model casts to float32 inside its normalization step."""
    index, config = prepared(mode='daily', feature_aggregation='daily_mean')
    x, _ = LightningMapsDataset(index, config)[0]
    assert x.dtype == torch.float16


def test_the_target_is_always_float32(prepared):
    index, config = prepared(mode='daily', feature_aggregation='daily_mean')
    _, target = LightningMapsDataset(index, config)[0]
    assert target.dtype == torch.float32


def test_the_target_is_the_hour_slice_in_hourly_mode(prepared):
    index, config = prepared(mode='hourly', n_days=1)
    dataset = LightningMapsDataset(index, config)
    on_disk = np.load(index.iloc[0]['target_file'])
    for hour in range(HOURS):
        _, target = dataset[hour]
        assert np.array_equal(target.numpy(), on_disk[hour])


# =====================================================================================================================
# DayGroupedShuffleSampler — dormant today, and must not be dropped
# =====================================================================================================================
def test_the_sampler_visits_every_item_exactly_once(prepared):
    index, config = prepared(mode='hourly', n_days=5)
    dataset = LightningMapsDataset(index, config)
    order = list(DayGroupedShuffleSampler(dataset))
    assert sorted(order) == list(range(len(dataset)))
    assert len(DayGroupedShuffleSampler(dataset)) == len(dataset)


def test_the_sampler_keeps_each_day_contiguous(prepared):
    """The point of it: reading one day's hours together makes each ``.pt`` file a single sequential read instead of 24
    scattered ones. Days are shuffled, hours within a day are not."""
    index, config = prepared(mode='hourly', n_days=5)
    dataset = LightningMapsDataset(index, config)
    order = list(DayGroupedShuffleSampler(dataset))

    days = [dataset.items[position][0] for position in order]
    # each day appears as one unbroken run
    runs = [day for i, day in enumerate(days) if i == 0 or day != days[i - 1]]
    assert len(runs) == len(set(runs)), f'a day was split across runs: {days}'


def test_the_sampler_shuffles_the_day_order(prepared):
    index, config = prepared(mode='hourly', n_days=8)
    dataset = LightningMapsDataset(index, config)
    orders = {tuple(DayGroupedShuffleSampler(dataset)) for _ in range(6)}
    assert len(orders) > 1, 'the sampler must shuffle across epochs'


def test_the_sampler_is_still_referenced_by_the_tuning_stage():
    """Its guard — ``mode == hourly and not uses_materialized_features`` — is DORMANT today, because every config sets
    ``materialize-features: true``. So it reads as dead code, and this is the assertion that stops it being deleted in a
    cleanup pass."""
    import inspect

    from src.utils.modeling import tuning

    source = inspect.getsource(tuning)
    assert source.count('DayGroupedShuffleSampler') >= 2, 'both fit sites must still pass the sampler'


# =====================================================================================================================
# build_split_datasets
# =====================================================================================================================
def test_build_split_datasets_returns_one_dataset_per_split(prepared):
    index, config = prepared(mode='daily', n_days=9)
    index.loc[3:5, 'split'] = 'valid'
    index.loc[6:8, 'split'] = 'test'

    datasets = build_split_datasets(index, config)
    assert set(datasets) == {'train', 'valid', 'test'}
    assert [len(datasets[split]) for split in ('train', 'valid', 'test')] == [3, 3, 3]


def test_build_split_datasets_honours_a_requested_subset(prepared):
    index, config = prepared(mode='daily', n_days=6)
    index.loc[3:5, 'split'] = 'valid'
    assert set(build_split_datasets(index, config, splits=['valid'])) == {'valid'}


def test_no_sampler_is_CONSTRUCTED_here():
    """It is a LOADER concern and applies to the train split only, while this returns a dataset per split — so the
    sampler belongs to whoever builds the DataLoader.

    Checked by AST rather than by substring: the docstring legitimately NAMES the sampler to explain its absence, so a
    text search finds it and proves nothing.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(build_split_datasets)))
    constructed = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert 'DayGroupedShuffleSampler' not in constructed, f'built here: {sorted(constructed)}'
