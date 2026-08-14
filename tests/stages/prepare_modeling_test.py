"""Tests for src/stages/prepare_modeling.py — the one preparation stage, both tasks.

Base is branch ``aru-probabilistic-eval``'s ``prepare_regression.py`` (760 lines). Renamed because the hourly task is
a classification, so "regression" was wrong for half of its job.

**The hourly 0/1 target is the only genuinely new code in this stage**, and the invariant that makes the two tasks one
pipeline rather than two is worth stating up front: the daily 0-24 count is *exactly* the per-cell sum of the hourly
0/1 field prepared from the same ``hourly_threshold``. Both go through ``_qualifying_hours``, which is why. A test
below asserts that composition directly — it is what "sharing the cutoff keeps the denoising consistent" means, and it
would be easy to break by thresholding one side differently.

What LEFT branch A, and is therefore asserted absent: the ``target_variable`` parameter (with its ``lightning_counts``
/ ``lightning_peak`` unbounded-count aggregations) and the ``gamma_shape`` / ``gamma_scale`` fit, which existed to
condition the removed F-transform.

Everything here is synthetic — the fixture builds a ``$DATA_ROOT``-shaped directory on a small grid, so no test reads
the real dataset.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

import prepare_modeling                                          # bare name: see conftest.py
from tests.conftest import executable_source

# the synthetic $DATA_ROOT / split fixtures live in tests/stages/conftest.py: `evaluate` consumes what this stage
# produces, so both files need them
from tests.stages.conftest import FEATURES, HEIGHT, HOURS, VARIABLES, WIDTH


# =====================================================================================================================
# ⭐ The target derivation — the one thing that branches on the mode
# =====================================================================================================================
def test_the_DAILY_target_is_the_number_of_QUALIFYING_hours():
    """Bounded 0-T by construction: it counts hours, so it cannot exceed the day's length however many strokes fall.
    That boundedness is what the whole classification-first scope rests on."""
    lightning = np.zeros((6, 2, 2))
    lightning[:3, 0, 0] = 1.0                                    # 3 single-stroke hours
    lightning[:4, 1, 1] = 5.0                                    # 4 qualifying hours
    lightning[:, 0, 1] = 9.0                                     # every hour qualifies

    target = prepare_modeling._derive_target(lightning, 'daily', hourly_threshold=2)

    assert target.shape == (2, 2) and target.dtype == np.float32
    assert target[0, 0] == 0.0, 'single-stroke hours must not count at threshold 2'
    assert target[1, 1] == 4.0
    assert target[0, 1] == 6.0, 'the ceiling is the day length, not a stroke count'
    assert target.max() <= lightning.shape[0]


def test_the_HOURLY_target_is_a_0_1_OCCURRENCE_field():
    """New code in this block. Branch A returned raw uint16 stroke COUNTS here, which is the unbounded heavy-tailed
    regime this project removed — the hourly task is a classification, so the target is the event itself."""
    lightning = np.zeros((6, 2, 2))
    lightning[:3, 0, 0] = 1.0
    lightning[:4, 1, 1] = 5.0

    target = prepare_modeling._derive_target(lightning, 'hourly', hourly_threshold=2)

    assert target.shape == (6, 2, 2)
    assert set(np.unique(target)) <= {0, 1}, 'the hourly target is an event, not a count'
    assert target[:4, 1, 1].tolist() == [1, 1, 1, 1]
    assert target[4:, 1, 1].tolist() == [0, 0]
    assert target[:, 0, 0].sum() == 0, 'sub-threshold hours are not events'


def test_the_DAILY_target_is_EXACTLY_the_per_cell_sum_of_the_HOURLY_one():
    """⭐ The invariant that makes one stage serve both tasks. Both derivations go through ``_qualifying_hours``, so
    the daily count is the hourly indicator summed over the day — not merely a similar quantity computed another way.

    Break it by thresholding one side differently and both targets stay individually plausible: the daily field is
    still bounded 0-24, the hourly field is still 0/1, and nothing downstream would notice that a model trained on one
    is being compared against a climatology built from the other."""
    rng = np.random.default_rng(0)
    lightning = rng.poisson(0.7, size=(HOURS, HEIGHT, WIDTH)).astype(np.float64)

    for threshold in (1, 2, 3):
        daily = prepare_modeling._derive_target(lightning, 'daily', threshold)
        hourly = prepare_modeling._derive_target(lightning, 'hourly', threshold)
        assert np.array_equal(daily, hourly.sum(axis=0).astype(np.float32)), f'threshold {threshold}'


@pytest.mark.parametrize('threshold,expected_hours', [(1, 3), (2, 0), (5, 0)])
def test_the_threshold_IS_the_minimum_stroke_count_not_an_offset(threshold, expected_hours):
    """``hourly_threshold=2`` means "an hour needs >= 2 strokes", so 3 single-stroke hours count as 3 at threshold 1
    and as 0 at threshold 2. Reading it as an exclusive cut (``> 2``) would shift every band by one hour."""
    lightning = np.zeros((6, 1, 1))
    lightning[:3, 0, 0] = 1.0
    assert prepare_modeling._derive_target(lightning, 'daily', threshold)[0, 0] == expected_hours


def test_the_hourly_target_is_stored_as_UINT8():
    """It is exactly 0/1, so uint8 is lossless and four times smaller than float32 — and an hourly prepared directory
    holds 24x the items of a daily one, so the factor is real. Every reader casts to float32 on load."""
    lightning = np.ones((6, 2, 2)) * 5
    assert prepare_modeling._derive_target(lightning, 'hourly', 2).dtype == np.uint8


def test_the_qualifying_hours_helper_is_shared_by_BOTH_derivations():
    """A source check, because it is the mechanism rather than an output: if either branch stopped calling it, the two
    targets could drift apart while every value-level test above still passed on its own side."""
    import inspect

    daily_source = inspect.getsource(prepare_modeling._daily_aggregation)
    hourly_path = inspect.getsource(prepare_modeling._derive_target)
    assert '_qualifying_hours' in daily_source
    assert '_qualifying_hours' in hourly_path


# =====================================================================================================================
# What branch A carried and this project removed
# =====================================================================================================================
def test_there_is_no_target_variable_PARAMETER():
    """A offered ``lightning_hours`` / ``lightning_counts`` / ``lightning_peak``. The latter two are unbounded
    heavy-tailed counts — the regime removed with the gamma F-transform — so the parameter is gone and `mode` is the
    only key that selects the target."""
    import inspect

    parameters = inspect.signature(prepare_modeling.prepare_modeling).parameters
    assert 'target_variable' not in parameters
    assert 'mode' in parameters


@pytest.mark.source_invariant
def test_no_unbounded_count_aggregation_survives_in_the_EXECUTABLE_stage():
    executable = executable_source(prepare_modeling)
    for banned in ('lightning_counts', 'lightning_peak', 'TARGET_VARIABLES', 'target_variable'):
        assert banned not in executable, banned


@pytest.mark.source_invariant
def test_the_GAMMA_FIT_is_gone_with_the_transform_it_conditioned():
    """``gamma_shape`` / ``gamma_scale`` existed only to parameterize the removed F-transform. The scipy import went
    with them, which is the observable trace — and an import is executable, so it is caught with docstrings stripped."""
    executable = executable_source(prepare_modeling)
    assert 'gamma_shape' not in executable and 'gamma_scale' not in executable
    assert 'scipy' not in executable


def test_the_written_target_stats_carry_no_transform_parameters(prepared):
    """The contract downstream reads. A stray gamma key would not break anything — which is exactly why it is worth
    pinning: it would sit in every prepared directory suggesting a transform that no longer exists."""
    _, _, target_stats, _ = prepared(mode='daily', n_days=6)

    assert 'gamma_shape' not in target_stats and 'gamma_scale' not in target_stats
    assert 'target_variable' not in target_stats
    assert target_stats['mode'] == 'daily'
    assert target_stats['hourly_threshold'] == 2


def test_positive_quantiles_SURVIVE_because_the_threshold_resolver_still_reads_them(prepared):
    """The one part of the old statistics block that stays: ``evaluation.resolve_threshold``'s generic
    ``train_positive_quantile`` kind reads it. Removing it with the gamma fit would break that resolver kind."""
    _, _, target_stats, _ = prepared(mode='daily', n_days=6)
    assert set(target_stats['positive_quantiles']) == {'0.5', '0.9', '0.95', '0.99', '0.999'}


# =====================================================================================================================
# The artifacts, end to end
# =====================================================================================================================
def test_the_stage_writes_the_five_documented_artifacts(prepared):
    output, prepared_config, _, split_index = prepared(mode='daily', n_days=6)

    assert os.path.isdir(os.path.join(output, 'targets'))
    assert os.path.isdir(os.path.join(output, 'features'))
    assert os.path.exists(os.path.join(output, 'split_index.csv'))
    assert os.path.exists(os.path.join(output, 'target_stats.json'))
    assert os.path.exists(os.path.join(output, 'prepared_config.json'))
    assert not os.path.exists(os.path.join(output, 'upstream')), 'no upstream without an upstream model'

    assert list(split_index.columns) == ['date', 'sample_id', 'file', 'target_filename', 'split', 'feature_filename']
    assert set(split_index['split']) == {'train', 'valid', 'test'}


def test_the_prepared_config_records_what_the_DATASET_decided_not_what_was_configured(prepared):
    """``hours_per_day`` and ``grid_shape`` are discovered from the data at prepare time and appear nowhere in
    ``config/`` — which is why the dataset reads them from here rather than taking them as arguments."""
    _, prepared_config, _, _ = prepared(mode='daily', n_days=6)

    assert prepared_config['hours_per_day'] == HOURS
    assert prepared_config['grid_shape'] == [HEIGHT, WIDTH]
    assert prepared_config['variable_names'] == VARIABLES
    assert prepared_config['features'] == FEATURES.split(',')


@pytest.mark.parametrize('mode,expected_layout', [('daily', 'variable_major'), ('hourly', 'time_major')])
def test_the_feature_LAYOUT_follows_the_modes_access_pattern(mode, expected_layout, prepared):
    """Pure I/O optimisation, derived from the mode and never configured: hourly reads one hour (a contiguous slice in
    time-major), daily reads the whole day and stacks channels (a free reshape in variable-major)."""
    _, prepared_config, _, _ = prepared(mode=mode, n_days=6)
    assert prepared_config['feature_layout'] == expected_layout


def test_the_hourly_mode_writes_one_TIME_STACKED_target_per_day(prepared):
    output, prepared_config, _, split_index = prepared(mode='hourly', n_days=6)
    target = np.load(os.path.join(output, 'targets', split_index.iloc[0]['target_filename']))

    assert target.shape == (HOURS, HEIGHT, WIDTH), 'one file per DAY, holding the day\'s hours'
    assert target.dtype == np.uint8
    assert prepared_config['mode'] == 'hourly'


def test_features_are_stored_in_the_requested_DTYPE(prepared):
    output, prepared_config, _, split_index = prepared(mode='daily', n_days=6, feature_dtype='float16')
    stored = np.load(os.path.join(output, 'features', split_index.iloc[0]['feature_filename']))
    assert stored.dtype == np.float16
    assert prepared_config['feature_dtype'] == 'float16'


def test_the_sparsity_diagnostic_reports_BOTH_raw_numbers_not_one_signed_difference(prepared):
    """⚠️ This test corrected the source. The docstring first claimed the sign was fixed per mode — daily negative,
    hourly positive — and the daily half is FALSE: ``raw_hourly`` counts the untouched field, so the target differs
    from it by the aggregation (pushing the zero proportion down) AND the threshold (pushing it up), and either can
    win. On this fixture, where every active hour of cell (0, 0) is single-stroke, the threshold wins and the daily
    difference comes out POSITIVE.

    So only the guaranteed direction is asserted here; the two tests below isolate each effect.
    """
    _, _, daily_stats, _ = prepared(mode='daily', n_days=6)
    _, _, hourly_stats, _ = prepared(mode='hourly', n_days=6)

    for stats in (daily_stats, hourly_stats):
        block = stats['zero_proportion']
        assert {'target', 'raw_hourly', 'target_minus_raw_hourly'} == set(block)
        assert abs(block['target_minus_raw_hourly'] - (block['target'] - block['raw_hourly'])) < 1e-12

    assert hourly_stats['zero_proportion']['target_minus_raw_hourly'] > 0, \
        'hourly: ONLY the threshold acts, and it can only zero more cells — this direction IS guaranteed'


def test_AGGREGATION_ALONE_makes_the_daily_target_LESS_sparse(prepared):
    """The effect isolated: at ``hourly_threshold=1`` nothing is denoised, so the only difference from the raw field is
    the aggregation, and a cell is zero for the day only if none of its hours had any stroke."""
    _, _, stats, _ = prepared(mode='daily', n_days=6, hourly_threshold=1)
    assert stats['zero_proportion']['target_minus_raw_hourly'] < 0


def test_the_THRESHOLD_ALONE_makes_the_hourly_target_SPARSER(prepared):
    """The other effect isolated: hourly mode does no aggregation, so a rising threshold can only add zeros."""
    _, _, permissive, _ = prepared(mode='hourly', n_days=6, hourly_threshold=1)
    _, _, strict, _ = prepared(mode='hourly', n_days=6, hourly_threshold=2)

    assert permissive['zero_proportion']['target_minus_raw_hourly'] == 0.0, 'threshold 1 keeps every non-zero hour'
    assert strict['zero_proportion']['target_minus_raw_hourly'] > 0


def test_the_sparsity_diagnostic_never_divides_by_zero():
    report = prepare_modeling._zero_proportion_report(0.5, raw_hourly_zero_cells=0, raw_hourly_total_cells=0)
    assert report['raw_hourly'] == 0.0 and np.isfinite(report['target_minus_raw_hourly'])


# =====================================================================================================================
# Validation and the staleness guard
# =====================================================================================================================
@pytest.mark.parametrize('threshold', [0, -1])
def test_a_threshold_below_ONE_raises_because_zero_would_count_EMPTY_hours(threshold, dataset_root, split_config,
                                                                          tmp_path):
    with pytest.raises(ValueError, match='must be >= 1'):
        prepare_modeling.prepare_modeling(
            data_path=dataset_root(), output_path=str(tmp_path / 'out'), split_config=split_config(),
            features=FEATURES, hourly_threshold=threshold)


def test_an_unknown_mode_raises(dataset_root, split_config, tmp_path):
    with pytest.raises(ValueError, match='Unknown mode'):
        prepare_modeling.prepare_modeling(
            data_path=dataset_root(), output_path=str(tmp_path / 'out'), split_config=split_config(),
            features=FEATURES, mode='weekly')


def test_an_unknown_FEATURE_raises_listing_the_datasets_variables(dataset_root, split_config, tmp_path):
    with pytest.raises(ValueError, match='Unknown feature variables'):
        prepare_modeling.prepare_modeling(
            data_path=dataset_root(), output_path=str(tmp_path / 'out'), split_config=split_config(),
            features='MU_LI,not_a_variable')


def test_an_unknown_feature_DTYPE_raises(dataset_root, split_config, tmp_path):
    with pytest.raises(ValueError, match='Unknown feature dtype'):
        prepare_modeling.prepare_modeling(
            data_path=dataset_root(), output_path=str(tmp_path / 'out'), split_config=split_config(),
            features=FEATURES, feature_dtype='float64')


def test_an_EMPTY_split_raises_rather_than_preparing_a_directory_nothing_can_train_on(dataset_root, tmp_path):
    import yaml

    spec = {'method': 'by_sample_id', 'cross_check': False,
            'by_sample_id': {'train': [[0, 2]], 'valid': [[3, 4]], 'test': [[90, 95]]}}
    path = tmp_path / 'bad_split.yaml'
    path.write_text(yaml.safe_dump(spec))

    with pytest.raises(ValueError, match='"test" split is empty'):
        prepare_modeling.prepare_modeling(
            data_path=dataset_root(), output_path=str(tmp_path / 'out'), split_config=str(path), features=FEATURES)


@pytest.mark.parametrize('key,changed', [('mode', 'hourly'), ('hourly_threshold', 3)])
def test_re_preparing_with_a_DIFFERENT_TARGET_parameter_RAISES_rather_than_skipping(
        key, changed, dataset_root, split_config, tmp_path):
    """⚠️ The guard that matters most in this stage. ``overwrite=false`` normally skips to a feature backfill — but the
    targets on disk depend on ``mode`` and ``hourly_threshold``, and neither is encoded in the output path. Skipping
    would silently train on the previous target while the config says otherwise."""
    data = dataset_root()
    split = split_config()
    output = str(tmp_path / 'reused')
    arguments = dict(data_path=data, output_path=output, split_config=split, features=FEATURES,
                     mode='daily', hourly_threshold=2)
    prepare_modeling.prepare_modeling(**arguments)

    with pytest.raises(ValueError, match='STALE'):
        prepare_modeling.prepare_modeling(**{**arguments, key: changed})


def test_re_preparing_with_the_SAME_parameters_is_a_no_op(dataset_root, split_config, tmp_path, caplog):
    """The lazy-cache-friendly path: an unchanged request only backfills features, and preparation is the expensive
    stage the cache exists for."""
    import logging

    data, split = dataset_root(), split_config()
    output = str(tmp_path / 'reused')
    arguments = dict(data_path=data, output_path=output, split_config=split, features=FEATURES)
    prepare_modeling.prepare_modeling(**arguments)

    with caplog.at_level(logging.INFO, logger='prepare_modeling'):
        prepare_modeling.prepare_modeling(**arguments)
    assert any('nothing to do' in record.getMessage() for record in caplog.records)


def test_OVERWRITE_re_derives_the_target(dataset_root, split_config, tmp_path):
    data, split = dataset_root(), split_config()
    output = str(tmp_path / 'overwritten')
    prepare_modeling.prepare_modeling(data_path=data, output_path=output, split_config=split, features=FEATURES,
                                      hourly_threshold=1)
    first = np.load(os.path.join(output, 'targets', '2015-07-01.npy')).copy()

    prepare_modeling.prepare_modeling(data_path=data, output_path=output, split_config=split, features=FEATURES,
                                      hourly_threshold=2, overwrite=True)
    second = np.load(os.path.join(output, 'targets', '2015-07-01.npy'))

    assert first[0, 0] == 3.0, 'threshold 1 counts the three single-stroke hours'
    assert second[0, 0] == 0.0, 'threshold 2 drops them'


# =====================================================================================================================
# Residual mode — the diffusion-only upstream pass
# =====================================================================================================================
def test_no_upstream_path_leaves_a_PLAIN_full_target_directory(prepared, caplog):
    """An unset ``{{$UPSTREAM_MODEL}}`` substitutes to the EMPTY STRING, not None — the documented `{{$VAR}}` footgun —
    so both must read as "no upstream"."""
    import logging

    with caplog.at_level(logging.INFO, logger='prepare_modeling'):
        _, prepared_config, _, split_index = prepared(mode='daily', n_days=6, upstream_model_path='')

    assert not prepared_config.get('residual_target')
    assert 'upstream_filename' not in split_index.columns
    assert any('No upstream model' in record.getMessage() for record in caplog.records)


def test_a_MISSING_upstream_checkpoint_raises_before_any_prediction(prepared):
    with pytest.raises(FileNotFoundError, match='not found'):
        prepared(mode='daily', n_days=6, upstream_model_path='outputs/does/not/exist.ckpt')


def test_the_upstream_pass_materializes_per_day_maps_and_flags_the_directory(
        prepared, unet_trial, normalization, target_stats, save_checkpoint):
    """The whole residual mechanism: one prediction map per day beside the untouched targets, the
    ``upstream_filename`` column, and the ``residual_target`` flag the dataset and the module both read."""
    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output, prepared_config, _, split_index = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS                # hourly_stack: Vf * T
    module = DeterministicUnetModule(
        unet_trial(), in_channels,
        target_stats(mode='daily', hourly_threshold=2, features=FEATURES.split(','),
                     feature_aggregation='hourly_stack'),
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})
    checkpoint = save_checkpoint(module, name='upstream.ckpt')

    prepare_modeling._materialize_upstream(
        output_path=output, upstream_model_path=checkpoint, overwrite=False,
        accelerator='cpu', devices=1, num_workers=0, batch_size=2)

    with open(os.path.join(output, 'prepared_config.json')) as handle:
        updated = json.load(handle)
    reloaded = pd.read_csv(os.path.join(output, 'split_index.csv'))

    assert updated['residual_target'] is True
    assert updated['upstream_model_signature']
    assert reloaded['upstream_filename'].notna().all()
    for name in reloaded['upstream_filename']:
        stored = np.load(os.path.join(output, 'upstream', name))
        assert stored.shape == (HEIGHT, WIDTH) and stored.dtype == np.float32

    # the raw targets are untouched: the residual is formed on the fly, and the baselines / metric suite stay in
    # target space precisely because this file was not overwritten
    assert np.load(os.path.join(output, 'targets', '2015-07-01.npy'))[1, 1] == 4.0


def test_an_upstream_with_the_WRONG_CHANNEL_COUNT_raises(prepared, unet_trial, normalization, target_stats,
                                                         save_checkpoint):
    """The upstream standardizes its conditioning positionally, so a channel-count mismatch cannot be recovered
    from — and a silent partial load would mis-standardize every channel."""
    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output, _, _, _ = prepared(mode='daily', n_days=6)
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(mode='daily'), normalization)

    with pytest.raises(ValueError, match='input channels'):
        prepare_modeling._materialize_upstream(
            output_path=output, upstream_model_path=save_checkpoint(module, name='narrow.ckpt'), overwrite=False,
            accelerator='cpu', devices=1, num_workers=0, batch_size=2)


def test_an_upstream_trained_at_a_DIFFERENT_THRESHOLD_raises(prepared, unet_trial, target_stats, save_checkpoint):
    """The residual is ``prepared target - upstream prediction``, so the two must live in the same target space —
    including the same hourly-count denoising. Mixing them would be invisible in every shape check."""
    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output, _, _, _ = prepared(mode='daily', n_days=6, hourly_threshold=2)
    in_channels = len(FEATURES.split(',')) * HOURS
    module = DeterministicUnetModule(
        unet_trial(), in_channels,
        target_stats(mode='daily', hourly_threshold=1),          # <- prepared at 2, upstream at 1
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})

    with pytest.raises(ValueError, match='hourly_threshold'):
        prepare_modeling._materialize_upstream(
            output_path=output, upstream_model_path=save_checkpoint(module, name='stale.ckpt'), overwrite=False,
            accelerator='cpu', devices=1, num_workers=0, batch_size=2)


def test_an_upstream_trained_on_REORDERED_features_raises(prepared, unet_trial, target_stats, save_checkpoint):
    """A reorder keeps the channel count identical, so nothing else catches it — and every channel would then be
    standardized with another channel's mean and std. This is why `tuning.run_sweep` records the ordered feature list
    into `target_stats` in the first place."""
    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output, _, _, _ = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS
    reordered = list(reversed(FEATURES.split(',')))
    module = DeterministicUnetModule(
        unet_trial(), in_channels,
        target_stats(mode='daily', hourly_threshold=2, features=reordered, feature_aggregation='hourly_stack'),
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})

    with pytest.raises(ValueError, match='set/order mismatch'):
        prepare_modeling._materialize_upstream(
            output_path=output, upstream_model_path=save_checkpoint(module, name='reordered.ckpt'), overwrite=False,
            accelerator='cpu', devices=1, num_workers=0, batch_size=2)


def test_an_upstream_with_NO_feature_provenance_only_WARNS(prepared, unet_trial, target_stats, save_checkpoint,
                                                           caplog):
    """A checkpoint predating the provenance recording. Refusing it would make old checkpoints unusable; warning says
    what cannot be verified."""
    import logging

    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output, _, _, _ = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS
    module = DeterministicUnetModule(
        unet_trial(), in_channels, target_stats(mode='daily', hourly_threshold=2),   # no `features` key
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})

    with caplog.at_level(logging.WARNING, logger='prepare_modeling'):
        prepare_modeling._materialize_upstream(
            output_path=output, upstream_model_path=save_checkpoint(module, name='old.ckpt'), overwrite=False,
            accelerator='cpu', devices=1, num_workers=0, batch_size=2)

    assert any('no feature provenance' in record.getMessage() for record in caplog.records)


def test_the_upstream_pass_is_IDEMPOTENT_for_the_same_checkpoint(prepared, unet_trial, target_stats,
                                                                 save_checkpoint, caplog):
    """It is a full forward pass over every item — 5843 days on the real dataset — so re-running the pipeline must not
    redo it. The signature is path + size + mtime, so an in-place retune of the same path DOES invalidate."""
    import logging

    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output, _, _, _ = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS
    module = DeterministicUnetModule(
        unet_trial(), in_channels,
        target_stats(mode='daily', hourly_threshold=2, features=FEATURES.split(','),
                     feature_aggregation='hourly_stack'),
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})
    checkpoint = save_checkpoint(module, name='upstream.ckpt')
    arguments = dict(output_path=output, upstream_model_path=checkpoint, overwrite=False,
                     accelerator='cpu', devices=1, num_workers=0, batch_size=2)

    prepare_modeling._materialize_upstream(**arguments)
    with caplog.at_level(logging.INFO, logger='prepare_modeling'):
        prepare_modeling._materialize_upstream(**arguments)
    assert any('already materialized' in record.getMessage() for record in caplog.records)


# =====================================================================================================================
# The feature backfill / rewrite path
# =====================================================================================================================
def test_a_feature_DTYPE_change_rewrites_every_file_even_under_overwrite_false(dataset_root, split_config, tmp_path):
    """Documented behaviour worth pinning: the fast path normally skips, but a dtype change has to touch every file or
    the directory would carry a mix of dtypes while ``prepared_config`` claimed one."""
    data, split = dataset_root(), split_config()
    output = str(tmp_path / 'redtype')
    prepare_modeling.prepare_modeling(data_path=data, output_path=output, split_config=split, features=FEATURES,
                                      feature_dtype='float32')
    assert np.load(os.path.join(output, 'features', '2015-07-01.npy')).dtype == np.float32

    prepare_modeling.prepare_modeling(data_path=data, output_path=output, split_config=split, features=FEATURES,
                                      feature_dtype='float16')
    assert np.load(os.path.join(output, 'features', '2015-07-01.npy')).dtype == np.float16
    with open(os.path.join(output, 'prepared_config.json')) as handle:
        assert json.load(handle)['feature_dtype'] == 'float16'


def test_materialize_features_FALSE_writes_no_feature_files(prepared):
    output, prepared_config, _, split_index = prepared(mode='daily', n_days=6, materialize_features=False)

    assert not os.path.exists(os.path.join(output, 'features'))
    assert 'feature_filename' not in split_index.columns
    assert prepared_config['feature_dtype'] is None
    assert prepared_config['feature_layout'] is None, \
        'stored as literal None, which is why the dataset reads it with `or "time_major"`'


def test_the_written_features_ARE_the_requested_variables_in_order(prepared, dataset_root):
    """Positional standardization again: the stored channel order must be the configured feature order, or the model's
    per-channel buffers line up with the wrong variables."""
    output, prepared_config, _, split_index = prepared(mode='daily', n_days=6, feature_dtype='float32')
    stored = np.load(os.path.join(output, 'features', split_index.iloc[0]['feature_filename']))

    assert stored.shape == (len(FEATURES.split(',')), HOURS, HEIGHT, WIDTH), 'variable-major for daily'
    source = torch.load(split_index.iloc[0]['file'], map_location='cpu', weights_only=False)
    for position, name in enumerate(FEATURES.split(',')):
        assert np.allclose(stored[position], source[name].numpy(), atol=1e-5), name


# =====================================================================================================================
# The stage wiring
# =====================================================================================================================
def test_the_stage_is_wrapped_with_fire_and_names_the_entry_point(repo_root):
    source = open(os.path.join(repo_root, 'src/stages/prepare_modeling.py')).read()
    assert 'Fire(prepare_modeling)' in source


def test_the_stage_imports_root_path_BEFORE_any_src_import(repo_root):
    """Stages are standalone scripts run from inside ``src/stages/``; that first line is what puts the repo root on
    the path, so any ``src.`` import above it fails at collection time."""
    lines = [line for line in open(os.path.join(repo_root, 'src/stages/prepare_modeling.py'))
             if line.startswith('from ') or line.startswith('import ')]
    root_position = next(index for index, line in enumerate(lines) if '__init__ import root_path' in line)
    src_positions = [index for index, line in enumerate(lines) if line.startswith('from src.')]
    assert src_positions and min(src_positions) > root_position


def test_every_parameter_the_configs_pass_is_accepted(repo_root):
    """The contract check. ``run.py`` forwards every YAML key to the stage's fire CLI, so a key the signature lacks
    aborts the stage — and the twelve shipped configs are the authority on what is passed."""
    import inspect

    from src.utils.io.parse_config import parse_config

    accepted = set(inspect.signature(prepare_modeling.prepare_modeling).parameters)
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        config = parse_config(os.path.join(repo_root, f'config/{family}/{family}.yaml'))
        block = next(parameters for stage in config['stages']
                     for name, parameters in stage.items() if name == 'prepare_modeling')
        passed = {key.replace('-', '_') for key in block}
        assert passed <= accepted, f'{family} passes unknown parameters: {sorted(passed - accepted)}'


# =====================================================================================================================
# The helpers underneath, driven directly
#
# Every one of these runs on any `prepare_modeling()` call, so the end-to-end tests above already execute them — the
# completeness gate flagged them anyway, and correctly: through the stage the only observable is "a directory came
# out", so a layout chosen backwards or a positives cap that dropped the wrong sample would look like success.
# =====================================================================================================================
@pytest.mark.parametrize('value,expected', [
    ('MU_LI,cp,lsm', ['MU_LI', 'cp', 'lsm']),
    ('MU_LI, cp , lsm', ['MU_LI', 'cp', 'lsm']),                 # YAML-ish spacing
    (['MU_LI', 'cp'], ['MU_LI', 'cp']),                          # already a list
    (('MU_LI', 'cp'), ['MU_LI', 'cp']),                          # the TUPLE fire parses `--features a,b` into
    ('MU_LI,,cp,', ['MU_LI', 'cp']),                             # a trailing comma must not yield an empty name
])
def test_the_feature_list_accepts_every_form_the_CLI_boundary_produces(value, expected):
    """``features`` arrives from a YAML scalar through fire, which may hand over a string OR a tuple. An empty name
    surviving would index ``variable_names`` for '' and raise a ValueError blaming the dataset."""
    assert prepare_modeling._as_name_list(value) == expected


@pytest.mark.parametrize('mode,expected', [('daily', 'variable_major'), ('hourly', 'time_major')])
def test_the_layout_is_chosen_by_the_modes_ACCESS_PATTERN(mode, expected):
    """Getting this backwards costs a strided transpose-copy per item and nothing else — no error, no wrong number,
    just a loader that is the dominant cost of every epoch."""
    assert prepare_modeling._feature_layout_for_mode(mode) == expected


@pytest.mark.parametrize('layout,expected_shape', [
    ('variable_major', (2, HOURS, HEIGHT, WIDTH)),
    ('time_major', (HOURS, 2, HEIGHT, WIDTH)),
])
def test_a_feature_file_is_written_in_the_requested_layout_and_stays_CONTIGUOUS(layout, expected_shape, tmp_path):
    """Contiguity is the point of the ``ascontiguousarray``: the file is memory-mapped and sliced per item, so a
    non-contiguous store would make every read a gather."""
    sample = np.arange(3 * HOURS * HEIGHT * WIDTH, dtype=np.float32).reshape(3, HOURS, HEIGHT, WIDTH)

    prepare_modeling._write_feature_file(str(tmp_path), 'day.npy', sample, [0, 2],
                                         np.dtype('float16'), layout)

    stored = np.load(str(tmp_path / 'day.npy'))
    assert stored.shape == expected_shape
    assert stored.dtype == np.float16
    assert stored.flags['C_CONTIGUOUS']
    # the SELECTED variables, in the requested order — positions 0 and 2, never 0 and 1
    reference = sample[[0, 2]] if layout == 'variable_major' else sample[[0, 2]].transpose(1, 0, 2, 3)
    assert np.allclose(stored.astype(np.float32), reference, rtol=1e-2)


def test_an_accumulator_starts_EMPTY_in_all_four_fields():
    accumulator = prepare_modeling._new_accumulator()
    assert accumulator == {'positives': [], 'n_zero': 0, 'n_total': 0, 'max': 0.0}


def test_the_accumulator_counts_zeros_and_totals_over_EVERY_cell():
    """These become the zero proportion — the project's headline sparsity number — so they must count the whole grid,
    not just the positives it also collects."""
    accumulator = prepare_modeling._new_accumulator()
    rng = np.random.default_rng(0)

    prepare_modeling._accumulate(accumulator, np.array([[0.0, 3.0], [0.0, 0.0]]), rng)
    prepare_modeling._accumulate(accumulator, np.array([[0.0, 0.0], [7.0, 24.0]]), rng)

    assert accumulator['n_total'] == 8
    assert accumulator['n_zero'] == 5
    assert accumulator['max'] == 24.0
    assert sorted(np.concatenate(accumulator['positives'])) == [3.0, 7.0, 24.0]


def test_the_accumulator_SUBSAMPLES_a_day_with_too_many_positives():
    """The reservoir feeds the positive quantiles over 5843 train days x 15k cells. Uncapped it would hold the whole
    positive marginal in memory; the cap is per day so no single day can dominate it either."""
    accumulator = prepare_modeling._new_accumulator()
    dense = np.full((200, 200), 5.0)                             # 40 000 positives, twice the cap

    prepare_modeling._accumulate(accumulator, dense, np.random.default_rng(0))

    assert accumulator['positives'][0].size == prepare_modeling.MAX_POSITIVES_PER_DAY
    assert accumulator['n_total'] == 40_000, 'the COUNTS are exact even when the sample is capped'
    assert accumulator['n_zero'] == 0


def test_the_accumulator_tolerates_an_ALL_ZERO_day():
    """Most of the calendar. A winter day contributes zeros and no positives, and must not make the reservoir ragged."""
    accumulator = prepare_modeling._new_accumulator()
    prepare_modeling._accumulate(accumulator, np.zeros((4, 4)), np.random.default_rng(0))

    assert accumulator['n_zero'] == 16 and accumulator['max'] == 0.0
    assert np.concatenate(accumulator['positives']).size == 0


def test_the_sparsity_LOG_LINE_reads_its_direction_off_the_MEASURED_sign(caplog):
    """⚠️ This is the other half of the source correction above: the line used to name the direction from the MODE,
    which is wrong in daily mode where the aggregation and the threshold pull opposite ways. It now reports what was
    measured, so the log cannot contradict the numbers beside it."""
    import logging

    with caplog.at_level(logging.INFO, logger='prepare_modeling'):
        prepare_modeling._log_zero_proportion(
            'daily', {'target': 0.98, 'raw_hourly': 0.99, 'target_minus_raw_hourly': -0.01})
        prepare_modeling._log_zero_proportion(
            'daily', {'target': 0.99, 'raw_hourly': 0.98, 'target_minus_raw_hourly': 0.01})

    messages = [record.getMessage() for record in caplog.records]
    assert 'less sparse' in messages[0], messages[0]
    assert 'sparser' in messages[1], messages[1]


def test_the_BASE_preparation_does_no_upstream_pass(dataset_root, split_config, tmp_path):
    """The split that keeps residual mode a diffusion-only concern: ``_prepare_base`` is what every family runs, and
    the upstream pass is bolted on afterwards by the entry point only when a checkpoint was supplied."""
    output = str(tmp_path / 'base_only')
    prepare_modeling._prepare_base(
        data_path=dataset_root(), output_path=output, mode='daily', feature_aggregation='hourly_stack',
        features=FEATURES, split_config=split_config(), overwrite=False, max_samples=None,
        materialize_features=True, feature_dtype='float16', hourly_threshold=2)

    with open(os.path.join(output, 'prepared_config.json')) as handle:
        prepared_config = json.load(handle)
    assert not os.path.exists(os.path.join(output, 'upstream'))
    assert 'residual_target' not in prepared_config
    assert 'upstream_model_signature' not in prepared_config


def _base(dataset_root, split_path, output, **overrides):
    arguments = dict(data_path=dataset_root, output_path=output, mode='daily', feature_aggregation='hourly_stack',
                     features=FEATURES, split_config=split_path, overwrite=False, max_samples=None,
                     materialize_features=False, feature_dtype='float16', hourly_threshold=2)
    arguments.update(overrides)
    return prepare_modeling._prepare_base(**arguments)


def test_MAX_SAMPLES_caps_the_days_processed(dataset_root, tmp_path):
    """Debug-only. It applies AFTER the split assignment and unassigned-day drop, so it caps what survives."""
    import yaml

    spec = {'method': 'by_sample_id', 'cross_check': False,
            'by_sample_id': {'train': [[0, 1]], 'valid': [[2, 3]], 'test': [[4, 5]]}}   # ids 6-8 unassigned
    split = tmp_path / 'six_of_nine.yaml'
    split.write_text(yaml.safe_dump(spec))

    output = str(tmp_path / 'capped')
    _base(dataset_root(n_days=9), str(split), output, max_samples=6)

    written = pd.read_csv(os.path.join(output, 'split_index.csv'))
    assert len(written) == 6
    assert set(written['split']) == {'train', 'valid', 'test'}


def test_MAX_SAMPLES_can_TRUNCATE_A_SPLIT_AWAY_and_that_raises(dataset_root, split_config, tmp_path):
    """⚠️ The documented footgun, made executable — and it is what a naive use of this flag actually does. The cap takes
    the EARLIEST surviving days, so on any split whose test years come last it removes them entirely. It raises rather
    than preparing a directory with an empty split, which is the useful outcome: an empty test split would only be
    noticed at the evaluation stage, several hours of training later.

    This is also the reason the smoke tiers use a ``by_sample_id`` split instead of ``max_samples`` — see
    ``config/split/split_smoke_cpu.yaml``'s header, which says so.
    """
    with pytest.raises(ValueError, match='"test" split is empty'):
        _base(dataset_root(n_days=9), split_config(n_days=9), str(tmp_path / 'truncated'), max_samples=6)


def test_the_feature_BACKFILL_materializes_into_a_directory_prepared_without_features(
        dataset_root, split_config, tmp_path):
    """The resumable path for a directory prepared with ``materialize-features: false``, or interrupted mid-write. It
    must leave the targets and the split assignment untouched — only features appear."""
    data, split = dataset_root(), split_config()
    output = str(tmp_path / 'backfilled')
    prepare_modeling._prepare_base(
        data_path=data, output_path=output, mode='daily', feature_aggregation='hourly_stack', features=FEATURES,
        split_config=split, overwrite=False, max_samples=None, materialize_features=False,
        feature_dtype='float16', hourly_threshold=2)
    target_before = np.load(os.path.join(output, 'targets', '2015-07-01.npy')).copy()
    assert not os.path.exists(os.path.join(output, 'features'))

    prepare_modeling._backfill_features(output, 'float16')

    reloaded = pd.read_csv(os.path.join(output, 'split_index.csv'))
    assert reloaded['feature_filename'].notna().all()
    assert np.load(os.path.join(output, 'features', '2015-07-01.npy')).dtype == np.float16
    assert np.array_equal(np.load(os.path.join(output, 'targets', '2015-07-01.npy')), target_before)
    with open(os.path.join(output, 'prepared_config.json')) as handle:
        assert json.load(handle)['feature_layout'] == 'variable_major'


def test_the_backfill_is_a_NO_OP_when_up_to_date_features_are_present(dataset_root, split_config, tmp_path, caplog):
    import logging

    output = str(tmp_path / 'already')
    prepare_modeling._prepare_base(
        data_path=dataset_root(), output_path=output, mode='daily', feature_aggregation='hourly_stack',
        features=FEATURES, split_config=split_config(), overwrite=False, max_samples=None,
        materialize_features=True, feature_dtype='float16', hourly_threshold=2)

    with caplog.at_level(logging.INFO, logger='prepare_modeling'):
        prepare_modeling._backfill_features(output, 'float16')
    assert any('nothing to do' in record.getMessage() for record in caplog.records)


def test_a_stored_LAYOUT_that_mismatches_the_mode_is_REWRITTEN(dataset_root, split_config, tmp_path):
    """A directory materialized before layouts were mode-specific carries time-major files in daily mode. The rewrite
    has to touch every file — a partial rewrite would leave the directory holding two layouts while
    ``prepared_config`` names one, and the dataset would transpose half of it wrongly."""
    output = str(tmp_path / 'relayout')
    prepare_modeling._prepare_base(
        data_path=dataset_root(), output_path=output, mode='daily', feature_aggregation='hourly_stack',
        features=FEATURES, split_config=split_config(), overwrite=False, max_samples=None,
        materialize_features=True, feature_dtype='float16', hourly_threshold=2)

    # forge the legacy state: claim time-major in a daily directory
    config_path = os.path.join(output, 'prepared_config.json')
    with open(config_path) as handle:
        prepared_config = json.load(handle)
    prepared_config['feature_layout'] = 'time_major'
    with open(config_path, 'w') as handle:
        json.dump(prepared_config, handle)

    prepare_modeling._backfill_features(output, 'float16')

    with open(config_path) as handle:
        assert json.load(handle)['feature_layout'] == 'variable_major'
    stored = np.load(os.path.join(output, 'features', '2015-07-01.npy'))
    assert stored.shape == (len(FEATURES.split(',')), HOURS, HEIGHT, WIDTH)
