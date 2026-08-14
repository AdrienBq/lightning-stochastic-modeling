"""Tests for src/stages/retrain_best.py — refit the winning configuration without a new sweep.

The same shape as ``tune``: a family dispatch over a shared harness, so these are contract tests.

⭐ The property that carries most of the value is that **a warm-started sweep is retrained warm-started**, inherited
from ``best_trial.json`` with no config key required. It is not a nicety: the sweep's hyperparameters were chosen
under a one-phase fit from the upstream's weights, so a from-scratch retrain runs two phases from random weights and
answers a different question. ``tuning.retrain_best_config`` states the requirement at its warm-start branch; this
stage is where it is discharged.

The other property is negative and unchanged: no selection metric, because the sweep's is authoritative.
"""
import ast
import inspect
import os

import pytest

import retrain_best                                              # bare name: see conftest.py
import tune
from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule


def test_it_reuses_TUNES_dispatch_table_rather_than_declaring_its_own():
    """Two tables would be two places for a family to be added, and the failure of forgetting one is a stage that
    cannot retrain a family the sweep can tune — discovered only after the sweep has finished."""
    assert retrain_best.MODULE_FACTORIES is tune.MODULE_FACTORIES


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_every_family_the_sweep_can_tune_can_also_be_retrained(family):
    assert family in retrain_best.MODULE_FACTORIES


def test_an_unknown_family_raises_BEFORE_the_expensive_fit():
    """The check is at the top of the stage, ahead of ``retrain_best_config``, which loads the prepared data and runs
    a staleness comparison before training. Failing late would waste the load."""
    with pytest.raises(ValueError, match='Unknown model family'):
        retrain_best.retrain_best(model_family='unet', source_path='a', input_path='b', output_path='c')


# =====================================================================================================================
# ⭐ Warm start: inherited from the sweep, overridable, never silently dropped
# =====================================================================================================================
@pytest.fixture
def experiment_store(tmp_path, repo_root):
    """A ``source_path`` holding a ``best_trial.json``, relative to the repo root the way the stage resolves it."""
    import json

    def build(upstream=None, checkpoint_exists=True):
        root = tmp_path / 'sweep'
        root.mkdir(exist_ok=True)
        recorded = None
        if upstream is not None:
            checkpoint = tmp_path / 'upstream.ckpt'
            if checkpoint_exists:
                checkpoint.write_bytes(b'weights')
            recorded = str(checkpoint)
        (root / 'best_trial.json').write_text(json.dumps({
            'selection_metric': 'valid_regression_score', 'score': 0.7, 'trial': {},
            'upstream_model_path': recorded,
        }))
        return os.path.relpath(str(root), repo_root), recorded
    return build


def test_the_sweeps_warm_start_is_INHERITED_with_no_config_key(experiment_store, monkeypatch, caplog):
    """⭐ The behaviour this stage owes ``retrain_best_config``. The shipped ``retrain_best`` block carries no
    ``upstream-model-path``, so if this did not read the record back, a warm-started MC-dropout sweep would be
    retrained from scratch: two phases from random weights, under hyperparameters chosen for one phase from the
    upstream's. Nothing would raise, and the retrained model would simply be a different model."""
    import logging
    from functools import partial

    from src.utils.modeling.mc_dropout_module import MCDropoutModule

    source, recorded = experiment_store(upstream='outputs/det/best_model.ckpt')
    captured = {}
    monkeypatch.setattr(retrain_best, 'retrain_best_config',
                        lambda **kwargs: captured.update(kwargs))

    with caplog.at_level(logging.INFO, logger='retrain_best'):
        retrain_best.retrain_best(model_family='mc_dropout', source_path=source,
                                  input_path='prepared', output_path='out')

    factory = captured['module_factory']
    assert isinstance(factory, partial), 'a recorded upstream must produce the warm-start constructor'
    assert factory.func.__func__ is MCDropoutModule.from_upstream.__func__
    assert factory.args == (recorded,)
    assert any('WARM-STARTED' in record.getMessage() for record in caplog.records)


def test_a_FROM_SCRATCH_sweep_is_retrained_from_scratch(experiment_store, monkeypatch):
    """The converse, and what makes the test above mean something: no recorded upstream gives the plain class."""
    from src.utils.modeling.mc_dropout_module import MCDropoutModule

    source, _ = experiment_store(upstream=None)
    captured = {}
    monkeypatch.setattr(retrain_best, 'retrain_best_config', lambda **kwargs: captured.update(kwargs))

    retrain_best.retrain_best(model_family='mc_dropout', source_path=source,
                             input_path='prepared', output_path='out')

    assert captured['module_factory'] is MCDropoutModule


def test_an_EXPLICIT_upstream_overrides_the_record_and_WARNS(experiment_store, monkeypatch, caplog, tmp_path):
    """The case the override exists for: the upstream itself was retrained on the new data. It warns because the
    retrained model then starts from different weights than the swept one — deliberate, but worth saying out loud."""
    import logging
    from functools import partial

    source, recorded = experiment_store(upstream='outputs/det/best_model.ckpt')
    newer = tmp_path / 'newer.ckpt'
    newer.write_bytes(b'newer weights')
    captured = {}
    monkeypatch.setattr(retrain_best, 'retrain_best_config', lambda **kwargs: captured.update(kwargs))

    with caplog.at_level(logging.WARNING, logger='retrain_best'):
        retrain_best.retrain_best(model_family='mc_dropout', source_path=source, input_path='prepared',
                                  output_path='out', upstream_model_path=str(newer))

    assert isinstance(captured['module_factory'], partial)
    assert captured['module_factory'].args == (str(newer),)
    assert any('DIFFERENT weights' in record.getMessage() for record in caplog.records)


def test_a_recorded_upstream_that_is_GONE_raises_rather_than_fitting_from_scratch(experiment_store, monkeypatch):
    """⚠️ The silent-regime-change guard. Falling back to a from-scratch fit here would be the worst outcome: the run
    succeeds, writes a checkpoint, reports finite metrics, and is a different model from the one that was selected.
    The message names where the path came from, because the fix differs (restore the file vs pass a new one)."""
    monkeypatch.setattr(retrain_best, 'retrain_best_config', lambda **kwargs: None)
    source, _ = experiment_store(upstream='outputs/det/best_model.ckpt', checkpoint_exists=False)

    with pytest.raises(FileNotFoundError, match='best_trial.json'):
        retrain_best.retrain_best(model_family='mc_dropout', source_path=source,
                                  input_path='prepared', output_path='out')


def test_an_explicit_upstream_that_is_GONE_names_the_FLAG_instead(experiment_store, monkeypatch):
    monkeypatch.setattr(retrain_best, 'retrain_best_config', lambda **kwargs: None)
    source, _ = experiment_store(upstream=None)

    with pytest.raises(FileNotFoundError, match='upstream-model-path'):
        retrain_best.retrain_best(model_family='mc_dropout', source_path=source, input_path='prepared',
                                  output_path='out', upstream_model_path='outputs/gone.ckpt')


def test_a_MISSING_best_trial_json_defers_to_the_harnesss_better_message(experiment_store, monkeypatch, tmp_path,
                                                                        repo_root):
    """Duplicating ``_load_existing_best``'s diagnosis here would only make the worse message arrive first."""
    monkeypatch.setattr(retrain_best, 'retrain_best_config', lambda **kwargs: None)
    empty = tmp_path / 'never_run'
    empty.mkdir()

    assert retrain_best._recorded_upstream(os.path.relpath(str(empty), repo_root)) is None


def test_the_warm_start_resolution_is_SHARED_with_the_sweep():
    """One definition of what a warm start means. Two would let the sweep and the retrain disagree about which family
    may warm-start, or about how the checkpoint is bound to the constructor."""
    assert retrain_best._module_factory is tune._module_factory


def test_the_stage_takes_NO_selection_metric_or_mode():
    """Read back from ``source-path/best_trial.json`` instead, so a retrain cannot rank on a different composite than
    the sweep. ``retrain_best_config`` is called with no metric argument at all — ``None`` means "whatever the sweep
    recorded is authoritative", and a store recording none raises rather than guessing."""
    parameters = set(inspect.signature(retrain_best.retrain_best).parameters)
    assert 'selection_metric' not in parameters and 'selection_mode' not in parameters

    call = _retrain_call()
    assert 'selection_metric' not in {keyword.arg for keyword in call.keywords}


def _retrain_call():
    tree = ast.parse(inspect.getsource(retrain_best))
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'retrain_best_config')


def test_every_argument_the_stage_forwards_is_ACCEPTED_by_the_harness():
    passed = {keyword.arg for keyword in _retrain_call().keywords}
    accepted = set(inspect.signature(retrain_best.retrain_best_config).parameters)
    assert passed <= accepted, f'not accepted: {sorted(passed - accepted)}'


def test_every_REQUIRED_argument_of_the_harness_is_supplied():
    passed = {keyword.arg for keyword in _retrain_call().keywords}
    required = {name for name, parameter in
                inspect.signature(retrain_best.retrain_best_config).parameters.items()
                if parameter.default is inspect.Parameter.empty}
    assert required <= passed, f'not supplied: {sorted(required - passed)}'


def test_every_parameter_the_shipped_configs_pass_is_accepted(repo_root):
    from src.utils.io.parse_config import parse_config

    accepted = set(inspect.signature(retrain_best.retrain_best).parameters)
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        for tier in ('', '_smoke_cpu', '_smoke_gpu'):
            config = parse_config(os.path.join(repo_root, f'config/{family}/{family}{tier}.yaml'))
            block = next(parameters for stage in config['stages']
                         for name, parameters in stage.items() if name == 'retrain_best')
            passed = {key.replace('-', '_') for key in block}
            assert passed <= accepted, f'{family}{tier}: unknown {sorted(passed - accepted)}'


def test_the_training_knobs_MIRROR_tunes_so_the_fit_is_identical():
    """A retrain is meant to reproduce the sweep's fit on new data. A knob present in one stage and not the other
    would make the two fits differ in a way no artifact records."""
    tune_parameters = set(inspect.signature(tune.tune).parameters)
    retrain_parameters = set(inspect.signature(retrain_best.retrain_best).parameters)

    sweep_only = {'model_config', 'n_trials', 'sampler', 'pruning', 'pruning_startup_trials',
                  'pruning_warmup_epochs', 'restart', 'load_existing', 'input_path', 'output_path'}
    retrain_only = {'source_path', 'staleness_max_age_days', 'input_path', 'output_path'}

    assert tune_parameters - sweep_only == retrain_parameters - retrain_only


def test_the_stage_is_wrapped_with_fire():
    assert 'Fire(retrain_best)' in inspect.getsource(retrain_best)


def test_the_stage_imports_root_path_BEFORE_any_src_import(repo_root):
    lines = [line for line in open(os.path.join(repo_root, 'src/stages/retrain_best.py'))
             if line.startswith('from ') or line.startswith('import ')]
    root_position = next(index for index, line in enumerate(lines) if '__init__ import root_path' in line)
    src_positions = [index for index, line in enumerate(lines) if line.startswith('from src.')]
    assert src_positions and min(src_positions) > root_position
