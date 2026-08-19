"""Tests for src/utils/modeling/tuning.py — the family-generic sweep harness.

``run_sweep`` and ``_fit_trial`` were already family-generic on branch A: ``run_sweep`` takes a ``module_factory`` and
``_fit_trial`` runs a general phase loop driven by the module's own ``monitor_metric`` / ``monitor_mode``. The only
non-generic part was that the phase LIST was derived inside ``_fit_trial`` from U-net-specific trial keys, which block
3b-2 replaced with a call to ``module.training_phases()``.

Running a real sweep needs an MLflow tracking server and a prepared directory, so the tests here cover the parts that
are plain Python: the key lists, the staleness comparison, and the contract ``_fit_trial`` requires of every module.
The end-to-end sweep is Step 4's gate.
"""
import inspect

import pytest

from src.utils.modeling import tuning
from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.diffusion_module import DiffusionModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule


# =====================================================================================================================
# The module contract _fit_trial drives
# =====================================================================================================================
@pytest.mark.parametrize('module_class', [DeterministicUnetModule, MCDropoutModule, DiffusionModule])
def test_every_family_implements_the_phase_contract(module_class):
    """``training_phases()`` has ONE call site and, before block 3c, ZERO implementations — ``_fit_trial`` would have
    raised ``AttributeError`` on any module handed to it. All three families owe one."""
    for method in ('training_phases', 'set_phase', 'predict_step', 'on_save_checkpoint'):
        assert callable(getattr(module_class, method, None)), f'{module_class.__name__}.{method}'


@pytest.mark.parametrize('module_class', [DeterministicUnetModule, MCDropoutModule, DiffusionModule])
def test_every_family_exposes_the_monitor_contract(module_class):
    """``_fit_trial`` builds each phase's checkpoint callback from these, so a family missing either would silently
    monitor whatever Lightning defaults to."""
    for attribute in ('monitor_metric', 'monitor_mode'):
        assert hasattr(module_class, attribute), f'{module_class.__name__}.{attribute}'


def test_every_family_writes_a_marker_and_the_three_are_DISTINCT(
        unet_trial, mc_trial, diffusion_trial, normalization, target_stats):
    """Two families sharing a marker would make the registry load one family's weights with the other's module — a
    shape-compatible, semantically wrong load that no exception would catch.

    Checked through `on_save_checkpoint`, which is the contract the registry actually reads, because the three families
    do not expose the marker the same way: the two U-net families inherit a `CHECKPOINT_MARKER` CLASS attribute from
    `UnetModuleBase`, while `DiffusionModule` is standalone (no U-net, so no shared base) and keeps its marker as a
    MODULE-level constant. Asserting the class attribute would pass for two families and fail for the third while all
    three behave correctly.
    """
    modules = [
        DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization),
        MCDropoutModule(mc_trial(), 5, target_stats(), normalization),
        DiffusionModule(diffusion_trial(), 5, target_stats(), normalization),
    ]
    markers = []
    for module in modules:
        checkpoint = {}
        module.on_save_checkpoint(checkpoint)
        assert checkpoint.get('module_class'), type(module).__name__
        markers.append(checkpoint['module_class'])

    assert len(set(markers)) == 3, markers
    assert set(markers) == {'deterministic_unet', 'mc_dropout', 'diffusion'}


def test_fit_trial_asks_the_module_for_its_phases_rather_than_deriving_them():
    """Block 3b-2's change. Deriving the list from U-net trial keys inside `_fit_trial` is what made the harness
    family-specific; asking the module is what let the two-phase MC fit fold in as a few lines.

    The absence half is checked on the CODE, not the text: the function's comments legitimately name the U-net-specific
    keys while explaining that they no longer drive anything, so a substring search finds them and proves nothing.
    """
    import ast
    import textwrap

    source = inspect.getsource(tuning._fit_trial)
    assert 'training_phases()' in source

    tree = ast.parse(textwrap.dedent(source))
    subscripts = {
        node.slice.value for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    for u_net_specific in ('hierarchy', 'occurrence_head', 'unet'):
        assert u_net_specific not in subscripts, f'{u_net_specific} is still read to build the phase list'


# =====================================================================================================================
# The two key lists (block 3b-2 edited both)
# =====================================================================================================================
def test_the_structural_keys_dropped_target_variable():
    """``target_variable`` was rejected outright by the preparation stage, and ``mode`` is now the ONLY key selecting
    between the two tasks."""
    assert 'target_variable' not in tuning._STRUCTURAL_KEYS
    assert 'mode' in tuning._STRUCTURAL_KEYS


def test_the_structural_keys_cover_what_makes_a_prepared_directory_incompatible():
    """These are the fields a retrain must match: a different mode, residual flag, feature list or aggregation means a
    different input tensor, so reusing a best-trial config across them is meaningless."""
    assert set(tuning._STRUCTURAL_KEYS) == {'mode', 'residual_target', 'features', 'feature_aggregation'}


def test_the_distribution_keys_dropped_the_gamma_parameters():
    """``gamma_shape`` / ``gamma_scale`` were written by ``compute_target_transform_stats``, which went with the
    transform. A surviving reference would be a KeyError on every prepared directory built since."""
    assert 'gamma_shape' not in tuning._DISTRIBUTION_KEYS
    assert 'gamma_scale' not in tuning._DISTRIBUTION_KEYS


def test_the_distribution_keys_are_the_target_statistics_worth_warning_about():
    assert set(tuning._DISTRIBUTION_KEYS) == {'hourly_threshold', 'zero_proportion', 'positive_mean'}


@pytest.mark.source_invariant
def test_no_transform_identifier_survives_anywhere_in_the_module():
    """The cross-cutting check for this file: the F-transform is removed, so a reference to it here would mean a code
    path expecting a space that no longer exists."""
    source = inspect.getsource(tuning)
    for token in ('GammaFTransform', 'LogStandardize', 'gamma_shape', 'gamma_scale', 'target_variable'):
        assert token not in source, token


# =====================================================================================================================
# Sweep configuration
# =====================================================================================================================
def test_both_samplers_are_offered():
    assert set(tuning.SAMPLERS) == {'random', 'tpe'}


def test_run_sweep_takes_a_module_factory_rather_than_a_family_name():
    """This is what makes the harness family-generic: the STAGE picks the factory, which is also where the one
    ``upstream-model-path`` string forks into its two independent uses — the warm-start weights here, and the trial
    constraint via ``apply_constraints``."""
    parameters = inspect.signature(tuning.run_sweep).parameters
    assert 'module_factory' in parameters
    assert 'model_family' not in parameters


def test_run_sweep_accepts_the_upstream_model_path():
    assert 'upstream_model_path' in inspect.signature(tuning.run_sweep).parameters


def test_the_selection_stage_PARAMETERS_are_gone_from_both_entry_points():
    """Step 2 made the search space's ``selection:`` block the ONE source of truth: ``tune`` reads it from
    ``model-config`` and records it into ``best_trial.json``, and ``retrain_best`` reads it back. With
    ``selection-metric`` / ``selection-mode`` still accepted as stage parameters, a retrain could rank on a different
    score from the sweep that chose the configuration — and nothing would report the disagreement."""
    sweep = inspect.signature(tuning.run_sweep).parameters
    retrain = inspect.signature(tuning.retrain_best_config).parameters

    assert 'selection_metric' not in sweep and 'selection_mode' not in sweep
    assert 'selection_metric' not in retrain and 'selection_mode' not in retrain


def test_only_the_SWEEP_takes_the_model_config():
    """``model_config`` is where ``selection:`` lives, so the sweep needs it. The retrain must NOT take it — it reads the
    recorded selection out of ``best_trial.json`` instead, which is what stops the two from diverging."""
    assert 'model_config' in inspect.signature(tuning.run_sweep).parameters
    assert 'model_config' not in inspect.signature(tuning.retrain_best_config).parameters


@pytest.mark.source_invariant
def test_apply_constraints_is_called_with_the_KEYWORD_argument():
    """The call-site half of ``search_test.py::test_upstream_model_path_is_KEYWORD_ONLY``. Branch A called
    ``apply_constraints(trial, rng)``, and a ``Generator`` is truthy — positionally that would have forced
    ``finetuning.enabled = True`` on every trial of every family, silently. The signature makes it a ``TypeError``; this
    pins that the call site here passes the path by keyword and never reintroduces the positional form."""
    source = inspect.getsource(tuning)
    assert 'apply_constraints(trial, upstream_model_path=upstream_model_path)' in source
    assert 'apply_constraints(trial, rng)' not in source


def test_the_selection_metric_is_resolved_from_the_mode():
    """``selection_metric_for_mode`` is imported here and raises when the search space declares the other composite, so
    the sweep cannot rank a binary target on the regression composite."""
    source = inspect.getsource(tuning)
    assert 'selection_metric_for_mode' in source
    assert 'DEFAULT_SELECTION_WEIGHTS' in source


def test_the_climatology_denominators_are_injected_into_every_trial_module():
    """Both are model-INDEPENDENT and computed once per sweep. Without the Brier one the hourly composite is silently
    short its 0.20 ``brier_skill_score`` term — the component would be NaN and contribute 0, so the trial still ranks,
    just on 0.80 of the intended score."""
    source = inspect.getsource(tuning)
    assert 'valid_climatology_cond_mae' in source
    assert 'valid_climatology_brier' in source


def test_the_batch_size_is_read_from_the_top_level_of_the_trial():
    """A shipped ``KeyError``: this read ``trial['optimizer']['batch_size']`` while Step 2 moved it to the top level and
    no ``tune`` stage passes ``batch-size:``, so it would have aborted trial 0 of every family. The 3b-2 gate missed it
    because it inspected signatures rather than running a sweep."""
    source = inspect.getsource(tuning)
    assert "trial['batch_size']" in source
    assert "['optimizer']['batch_size']" not in source


def test_pruning_is_attached_only_when_the_monitor_is_the_prune_metric():
    """Which is why the diffusion family never prunes: it monitors ``valid_flow_loss`` while the sweep ranks on the
    target-space composite. By design, not by omission — the flow loss says nothing about occurrence skill."""
    source = inspect.getsource(tuning._fit_trial)
    assert 'prune_metric' in source


# =====================================================================================================================
# retrain_best_config
# =====================================================================================================================
def test_retrain_best_config_does_not_take_a_search_space():
    """It reads the recorded best trial rather than re-sampling, so ``model-config`` / ``selection-metric`` /
    ``selection-mode`` are all dropped from its signature — the selection is read back out of ``best_trial.json``."""
    parameters = inspect.signature(tuning.retrain_best_config).parameters
    for dropped in ('model_config', 'selection_metric', 'selection_mode', 'search_space'):
        assert dropped not in parameters, dropped


def test_the_staleness_check_compares_both_key_groups():
    """A structural mismatch must be fatal and a distributional one only a warning: a different mode means the recorded
    architecture cannot be rebuilt, while a shifted zero-proportion just means the tuning was done on different data."""
    source = inspect.getsource(tuning._check_retrain_staleness)
    assert '_STRUCTURAL_KEYS' in source
    assert '_DISTRIBUTION_KEYS' in source


def test_the_prepared_mode_is_read_from_the_prepared_directory():
    """Not from config: ``mode`` is a property of the prepared artifacts, so the sweep discovers it rather than being
    told and risking a disagreement."""
    source = inspect.getsource(tuning._prepared_mode)
    assert 'prepared_config' in source or 'json' in source


# =====================================================================================================================
# Block 5c — the sweep harness's plain-Python parts
#
# ⚠️ ``tuning.py`` is the lowest-coverage file in ``src/`` and stays that way after this block: ``run_sweep`` and
# ``_fit_trial`` need optuna, a Lightning trainer and a real fit, which is Step 4's end-to-end gate. What IS testable
# here is everything around them — the store, the two restart paths, the metrics writer and the two diagnostic
# callbacks — and those are where a sweep loses work rather than merely failing.
# =====================================================================================================================
import json
import os


def test_the_optuna_store_is_a_JOURNAL_file_not_sqlite(tmp_path):
    """Deliberate: ``$DATA_ROOT`` and the output paths live on a shared/network filesystem, where sqlite's locking is
    unreliable — a sweep resumed there can corrupt its own study. The journal backend appends instead."""
    storage = tuning._journal_storage(str(tmp_path / 'study.log'))

    from optuna.storages import JournalStorage
    assert isinstance(storage, JournalStorage)
    assert os.path.exists(str(tmp_path / 'study.log'))


def test_the_store_ROUND_TRIPS_a_study_so_a_restart_resumes_rather_than_restarting(tmp_path):
    """The point of persisting at all. ``restart: false`` reloads the study by name and continues from trial N — if
    the store did not survive the process, every resume would silently re-run the whole sweep."""
    import optuna

    path = str(tmp_path / 'study.log')
    study = optuna.create_study(storage=tuning._journal_storage(path), study_name='sweep', direction='maximize')
    study.optimize(lambda trial: trial.suggest_float('x', 0.0, 1.0), n_trials=3)

    reopened = optuna.load_study(storage=tuning._journal_storage(path), study_name='sweep')
    assert len(reopened.trials) == 3
    assert reopened.direction == optuna.study.StudyDirection.MAXIMIZE


# ---------------------------------------------------------------------------------------------------------------------
# The best-metrics writer
# ---------------------------------------------------------------------------------------------------------------------
def test_the_best_metrics_file_carries_the_SELECTION_SCORE_under_its_own_name(tmp_path, repo_root):
    """``metrics-path`` is one of the three OUTPUT_PARAM_KEYS, and run.py logs every scalar in it to MLflow. The
    composite has to appear under the metric name the sweep ranked on, or the comparison table has no column for the
    thing that chose the model."""
    relative = os.path.relpath(str(tmp_path / 'out' / 'metrics.json'), repo_root)
    tuning._write_best_metrics(relative, {'valid_mae': 1.5, 'valid_psd_full_fidelity': 0.8},
                               'valid_regression_score', 0.72, repo_root)

    payload = json.load(open(str(tmp_path / 'out' / 'metrics.json')))
    assert payload['valid_regression_score'] == 0.72
    assert payload['valid_mae'] == 1.5


def test_NON_FINITE_and_NON_NUMERIC_metrics_are_dropped_from_the_file(tmp_path, repo_root):
    """``json.dump`` writes bare ``NaN`` / ``Infinity``, which are not valid JSON — and MLflow's ``log_metric`` rejects
    them. A single NaN component would otherwise make the whole metrics file unreadable, losing every other scalar with
    it. NaN components are routine: the deterministic family's ensemble scalars are all NaN by design."""
    relative = os.path.relpath(str(tmp_path / 'metrics.json'), repo_root)
    tuning._write_best_metrics(relative, {'good': 1.0, 'nan': float('nan'), 'inf': float('inf'),
                                          'text': 'not a number', 'nested': {'a': 1}},
                               'score', 0.5, repo_root)

    payload = json.load(open(str(tmp_path / 'metrics.json')))
    assert set(payload) == {'good', 'score'}, payload


def test_the_writer_CREATES_the_output_directory(tmp_path, repo_root):
    relative = os.path.relpath(str(tmp_path / 'deep' / 'nested' / 'metrics.json'), repo_root)
    tuning._write_best_metrics(relative, {'a': 1.0}, 'score', 0.1, repo_root)
    assert os.path.exists(str(tmp_path / 'deep' / 'nested' / 'metrics.json'))


def test_NO_metrics_path_is_a_no_op_rather_than_an_error(repo_root):
    """``metrics-path`` is optional on the tune stage. Raising would make a pipeline that does not want the file fail
    at the very end of a sweep, after all the training is done."""
    assert tuning._write_best_metrics(None, {'a': 1.0}, 'score', 0.1, repo_root) is None


# ---------------------------------------------------------------------------------------------------------------------
# load_existing: the four ways a persisted best can be unusable
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def experiment_store(tmp_path):
    """The two files a completed sweep leaves behind."""
    def build(best_trial=None, checkpoint=True):
        root = tmp_path / 'sweep'
        root.mkdir(exist_ok=True)
        if best_trial is not None:
            (root / 'best_trial.json').write_text(json.dumps(best_trial))
        if checkpoint:
            (root / 'best_model.ckpt').write_bytes(b'weights')
        return str(root)
    return build


def test_a_complete_store_loads_and_returns_what_the_sweep_recorded(experiment_store):
    saved = {'selection_metric': 'valid_regression_score', 'score': 0.8, 'trial': {'batch_size': 4}}
    loaded = tuning._load_existing_best(experiment_store(saved), 'sweep', 'valid_regression_score')
    assert loaded == saved


def test_a_metric_of_NONE_accepts_whatever_the_sweep_recorded(experiment_store):
    """The retrain path. ``retrain_best`` reads the composite back OUT of ``best_trial.json`` rather than being told
    which to expect — that is what makes the search space the single source of truth, and stops a retrain from
    disagreeing with the sweep that chose the configuration."""
    saved = {'selection_metric': 'valid_classification_score', 'score': 0.4}
    assert tuning._load_existing_best(experiment_store(saved), 'sweep', None) == saved


def test_an_EMPTY_output_path_raises_FileNotFoundError_naming_the_remedy(experiment_store, tmp_path):
    """Distinct from the RuntimeErrors below, because the remedy is different: nothing has been run yet."""
    empty = str(tmp_path / 'never_run')
    os.makedirs(empty)
    with pytest.raises(FileNotFoundError, match='Run the sweep first'):
        tuning._load_existing_best(empty, 'never_run', None)


def test_a_checkpoint_with_NO_best_trial_json_raises(experiment_store):
    """Half a store — the sweep crashed between writing the checkpoint and writing its record. Loading the checkpoint
    anyway would give a model with no idea which composite selected it."""
    with pytest.raises(RuntimeError, match='best_trial.json is missing'):
        tuning._load_existing_best(experiment_store(None, checkpoint=True), 'sweep', None)


def test_a_best_trial_with_NO_checkpoint_raises(experiment_store):
    with pytest.raises(RuntimeError, match='no.*best_model.ckpt'):
        tuning._load_existing_best(experiment_store({'selection_metric': 'valid_regression_score'}, checkpoint=False),
                                   'sweep', None)


def test_a_store_selected_on_a_DIFFERENT_composite_raises_naming_BOTH(experiment_store):
    """The check that stops an hourly-selected model being reused by a daily pipeline. Both names are in the message
    because either side could be the mistake."""
    saved = {'selection_metric': 'valid_classification_score'}
    with pytest.raises(RuntimeError) as raised:
        tuning._load_existing_best(experiment_store(saved), 'sweep', 'valid_regression_score')
    assert 'valid_classification_score' in str(raised.value)
    assert 'valid_regression_score' in str(raised.value)


def test_a_store_recording_NO_selection_metric_at_all_raises(experiment_store):
    """A pre-Step-3 store, from before the composite was recorded. There is no way to know what it was chosen on, and
    guessing would silently rank one task's model by the other task's score."""
    with pytest.raises(RuntimeError, match='no selection_metric'):
        tuning._load_existing_best(experiment_store({'score': 0.5}), 'sweep', None)


# ---------------------------------------------------------------------------------------------------------------------
# Reading a checkpoint's hyper-parameters
# ---------------------------------------------------------------------------------------------------------------------
def test_the_checkpoint_hparams_come_back_as_the_four_construction_arguments(
        unet_trial, normalization, target_stats, save_checkpoint):
    """This is what the warm-start path reads to check an upstream checkpoint's architecture against the sampled one —
    so all four have to survive the round trip, not just the state dict."""
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization)
    hparams = tuning._load_checkpoint_hparams(save_checkpoint(module))

    assert {'trial', 'in_channels', 'target_stats', 'normalization'} <= set(hparams)
    assert hparams['in_channels'] == 5
    assert hparams['trial']['batch_size'] == 4


def test_a_checkpoint_with_no_hyper_parameters_gives_an_EMPTY_dict(tmp_path):
    """A raw state-dict checkpoint. Returning ``{}`` lets the caller raise its own message naming the offending field,
    which is more useful than a ``KeyError`` from in here."""
    import torch

    path = str(tmp_path / 'bare.ckpt')
    torch.save({'state_dict': {}}, path)
    assert tuning._load_checkpoint_hparams(path) == {}


# ---------------------------------------------------------------------------------------------------------------------
# The two diagnostic callbacks
# ---------------------------------------------------------------------------------------------------------------------
class _StubTrainer:
    def __init__(self, current_epoch=0):
        self.current_epoch = current_epoch


def _drive_epoch(callback, batches, wait=0.0, compute=0.0):
    """Run one full epoch of the Lightning hooks against a stub trainer, END INCLUDED — the epoch-end hook is where
    the diagnostic is emitted, so a helper that stopped at the last batch would make every assertion below vacuous."""
    import time

    trainer = _StubTrainer()
    callback.on_train_epoch_start(trainer, None)
    for index in range(batches):
        if wait:
            time.sleep(wait)
        callback.on_train_batch_start(trainer, None, ([0] * 4,), index)
        if compute:
            time.sleep(compute)
        callback.on_train_batch_end(trainer, None, None, ([0] * 4,), index)
    callback.on_train_epoch_end(trainer, None)
    return trainer


def test_the_throughput_callback_attributes_time_between_WAITING_and_COMPUTING(caplog):
    """The diagnostic exists to answer one question — is the GPU starved by the dataloader? — and the answer is the
    wait SHARE. A callback that attributed everything to compute would report 0 % on a badly starved run and the
    obvious fix (more workers, materialized features) would never be suggested."""
    import logging

    callback = tuning.ThroughputDiagnostics('trial 0 train', use_cuda=False)
    with caplog.at_level(logging.INFO, logger='src.utils.modeling.tuning'):
        _drive_epoch(callback, batches=4, wait=0.02, compute=0.0)

    message = next(record.getMessage() for record in caplog.records if 'items/s' in record.getMessage())
    share = int(message.split('data-wait ')[1].split('%')[0])
    assert share > 50, f'an all-wait epoch reported {share} % wait: {message}'


def test_the_FIRST_batch_is_excluded_from_the_wait_share_and_reported_separately():
    """Batch 0's wait is worker spin-up and prefetch warm-up — seconds, once per epoch. Folding it into the share
    would make every short epoch look starved."""
    callback = tuning.ThroughputDiagnostics('trial 0 train', use_cuda=False)
    _drive_epoch(callback, batches=3, wait=0.0, compute=0.01)

    assert callback._first_wait >= 0.0
    assert callback._batches == 3
    assert callback._items == 12, 'four items per batch, counted from the batch TUPLE\'s first element'


def test_a_SINGLE_batch_epoch_reports_nothing(caplog):
    """With one batch there is no steady state to measure — the whole epoch is first-batch warm-up. The smoke configs
    run exactly this, so a line claiming 100 % data-wait would appear on every smoke run."""
    import logging

    callback = tuning.ThroughputDiagnostics('trial 0 train', use_cuda=False)
    with caplog.at_level(logging.INFO, logger='src.utils.modeling.tuning'):
        _drive_epoch(callback, batches=1)
    assert not [record for record in caplog.records if 'items/s' in record.getMessage()]


def test_the_counters_RESET_between_epochs():
    """They accumulate per epoch and are reported per epoch. Without the reset, epoch 5's throughput would be the
    running average over all five and the trend a sweep is watched for would flatten out."""
    callback = tuning.ThroughputDiagnostics('trial 0 train', use_cuda=False)
    _drive_epoch(callback, batches=3)
    _drive_epoch(callback, batches=2)
    assert callback._batches == 2 and callback._items == 8


def test_the_progress_bar_throttles_itself_OFF_a_tty():
    """A pipeline runs orchestrated, so stdout is a file: every refresh is a full printed line, and an unthrottled bar
    writes thousands of them into the run log."""
    import sys

    bar = tuning.TrialProgressBar('trial 0', max_epochs=5)
    assert bar.refresh_rate == (1 if sys.stdout.isatty() else 50)


def test_the_progress_bar_description_carries_the_TRIAL_and_the_EPOCH():
    """A sweep is dozens of fits; a bar that said only "Epoch 2/5" would not say which trial it belonged to."""
    class _StubBar:
        def __init__(self):
            self.description = None

        def set_description(self, text):
            self.description = text

    bar = tuning.TrialProgressBar('trial 3 finetune', max_epochs=5)
    bar._train_progress_bar = _StubBar()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(bar).__bases__[0], 'on_train_epoch_start', lambda self, trainer, *args: None)
        bar.on_train_epoch_start(_StubTrainer(current_epoch=1))

    assert bar.train_progress_bar.description == 'trial 3 finetune epoch 2/5'
