"""Tests for src/stages/tune.py — the unified sweep stage.

Branch A had two of these and branch D another two, each of D's with its own copy of the per-trial fit. All of that
merged into ``src.utils.modeling.tuning`` in Step 3, so what is left here is a family dispatch and a CLI — which makes
this file almost entirely CONTRACT tests. That is the right shape: a thin stage's failure modes are all about what it
forwards, and a real sweep needs optuna, a Lightning trainer and a fit (Step 4's end-to-end gate).

⭐ The one piece of real logic is ``_module_factory``, and it is where one ``upstream-model-path`` string forks into
two independent uses — the weights loaded here, and the trial constraint that ``run_sweep`` applies separately. Its
tests are the substantial ones below.
"""
import ast
import inspect
import os

import pytest

import tune                                                      # bare name: see conftest.py
from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.diffusion_module import DiffusionModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule


# =====================================================================================================================
# The family dispatch
# =====================================================================================================================
@pytest.mark.parametrize('family,expected', [
    ('deterministic_unet', DeterministicUnetModule),
    ('mc_dropout', MCDropoutModule),
    ('diffusion', DiffusionModule),
])
def test_every_family_resolves_to_its_module_class(family, expected):
    assert tune._module_factory(family, upstream_model_path=None) is expected


def test_the_dispatch_covers_exactly_the_three_families():
    """No more and no fewer. An extra key here would be a family no pipeline can reach; a missing one fails at the
    stage boundary, after preparation has already run."""
    from src.utils.modeling.registry import FAMILY_NAMES

    assert set(tune.MODULE_FACTORIES) == set(FAMILY_NAMES)


def test_an_unknown_family_raises_LISTING_the_valid_ones(caplog):
    with pytest.raises(ValueError, match='Unknown model family') as raised:
        tune._module_factory('unet', upstream_model_path=None)
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        assert family in str(raised.value)


def test_the_dispatch_is_NOT_the_registry_which_carries_a_LEGACY_MARKER():
    """``registry.MODULE_REGISTRY`` maps CHECKPOINT MARKERS to classes and includes ``distr_regression`` so
    pre-rename checkpoints still load. That is a loading concern — a legacy marker is not a family a pipeline may ask
    to tune, and accepting it here would let a config name a family that no search space exists for."""
    from src.utils.modeling.registry import MODULE_REGISTRY

    assert 'distr_regression' in MODULE_REGISTRY
    assert 'distr_regression' not in tune.MODULE_FACTORIES
    with pytest.raises(ValueError, match='Unknown model family'):
        tune._module_factory('distr_regression', upstream_model_path=None)


# =====================================================================================================================
# ⭐ The warm start — one string, two independent uses
# =====================================================================================================================
def test_a_warm_start_returns_the_ALTERNATE_CONSTRUCTOR_bound_to_the_checkpoint(caplog):
    """``from_upstream`` is the only way to reach ``warm_started = True``, and it loads the weights on the line above
    setting it. Returning the plain class here instead would give a RANDOMLY INITIALISED net that then skips phase 1 —
    the only phase that would have trained it. It fits nothing, checkpoints, scores finitely-but-badly, and the sweep
    merely ranks it low: no error, no warning."""
    import logging
    from functools import partial

    with caplog.at_level(logging.INFO, logger='tune'):
        factory = tune._module_factory('mc_dropout', upstream_model_path='outputs/det/best_model.ckpt')

    assert isinstance(factory, partial)
    # `.__func__` because `from_upstream` is a CLASSMETHOD: every attribute access builds a fresh bound method, so
    # identity on the bound objects is false even when they wrap the same function.
    assert factory.func.__func__ is MCDropoutModule.from_upstream.__func__
    assert factory.args == ('outputs/det/best_model.ckpt',)
    assert any('WARM START' in record.getMessage() for record in caplog.records)


def test_the_factory_signature_still_matches_what_run_sweep_CALLS():
    """``run_sweep`` calls ``module_factory(trial, in_channels, target_stats, normalization)``. The partial binds the
    checkpoint as the FIRST positional of ``from_upstream``, so the remaining four line up — bind it anywhere else and
    the trial dict would arrive as the checkpoint path."""
    from functools import partial

    factory = tune._module_factory('mc_dropout', upstream_model_path='a.ckpt')
    remaining = list(inspect.signature(MCDropoutModule.from_upstream).parameters)[1:]
    assert remaining == ['trial', 'in_channels', 'target_stats', 'normalization']
    assert isinstance(factory, partial) and len(factory.args) == 1


@pytest.mark.parametrize('empty', [None, ''])
def test_an_UNSET_upstream_leaves_the_plain_constructor(empty):
    """An unset ``{{$UPSTREAM_MODEL}}`` substitutes to the EMPTY STRING, not None — the documented `{{$VAR}}` footgun —
    so both must read as "fit from scratch", giving the full two-phase run."""
    assert tune._module_factory('mc_dropout', upstream_model_path=empty) is MCDropoutModule


@pytest.mark.parametrize('family', ['deterministic_unet', 'diffusion'])
def test_an_upstream_given_to_the_WRONG_FAMILY_raises_and_says_where_it_belongs(family):
    """⚠️ Both families read ``UPSTREAM_MODEL``, at different stages and for different things. Diffusion's is consumed
    by ``prepare_modeling`` as a conditioning channel; if one reached this stage it would silently do nothing here
    while ``apply_constraints`` applied an MC-dropout rule to a diffusion trial. Raising names the right block."""
    with pytest.raises(ValueError) as raised:
        tune._module_factory(family, upstream_model_path='outputs/det/best_model.ckpt')

    message = str(raised.value)
    assert 'prepare_modeling' in message, 'the error must say where the key belongs'
    assert family in message


def test_only_MC_DROPOUT_declares_an_upstream_in_the_shipped_tune_blocks(repo_root):
    """The config side of the same claim, driven from the real pipelines rather than a fixture."""
    from src.utils.io.parse_config import parse_config

    declaring = set()
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        config = parse_config(os.path.join(repo_root, f'config/{family}/{family}.yaml'))
        block = next(parameters for stage in config['stages']
                     for name, parameters in stage.items() if name == 'tune')
        if 'upstream-model-path' in block:
            declaring.add(family)
    assert declaring == {'mc_dropout'}, declaring


def test_the_upstream_string_is_ALSO_forwarded_to_run_sweep():
    """The second, independent use: ``run_sweep`` hands it to ``apply_constraints``, which forces
    ``finetuning.enabled`` true. Without that a warm-started trial that sampled ``false`` would have NO fitting phase
    at all — ``set_phase`` raises on finetune, and there is no train phase to fall back on. Consuming the string only
    for the weights would leave that constraint unapplied."""
    source = inspect.getsource(tune.tune)
    assert 'upstream_model_path=upstream_model_path' in source


# =====================================================================================================================
# What the stage forwards
# =====================================================================================================================
def _run_sweep_call():
    """The `run_sweep(...)` call node inside `tune`, so the keywords can be inspected without running a sweep."""
    tree = ast.parse(inspect.getsource(tune))
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'run_sweep')


def test_every_argument_the_stage_forwards_is_ACCEPTED_by_run_sweep():
    """The stage is a passthrough, so a renamed harness parameter shows up here as a TypeError on trial 0 — after the
    prepared data has been loaded and the study created."""
    passed = {keyword.arg for keyword in _run_sweep_call().keywords}
    accepted = set(inspect.signature(tune.run_sweep).parameters)
    assert passed <= accepted, f'not accepted by run_sweep: {sorted(passed - accepted)}'


def test_every_REQUIRED_argument_of_run_sweep_is_supplied():
    """``run_sweep`` is keyword-only with many non-defaulted parameters, so an omission is a TypeError rather than a
    quiet default."""
    passed = {keyword.arg for keyword in _run_sweep_call().keywords}
    required = {name for name, parameter in inspect.signature(tune.run_sweep).parameters.items()
                if parameter.default is inspect.Parameter.empty}
    assert required <= passed, f'not supplied: {sorted(required - passed)}'


def test_every_parameter_the_shipped_configs_pass_is_accepted(repo_root):
    """``run.py`` forwards every YAML key to the stage's fire CLI, so a key the signature lacks aborts the stage. The
    twelve configs are the authority on what is passed; all three tiers are checked because the smoke tiers add knobs
    the full pipelines do not (``batch-size``, ``limit-*-batches``)."""
    from src.utils.io.parse_config import parse_config

    accepted = set(inspect.signature(tune.tune).parameters)
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        for tier in ('', '_smoke_cpu', '_smoke_gpu'):
            config = parse_config(os.path.join(repo_root, f'config/{family}/{family}{tier}.yaml'))
            block = next(parameters for stage in config['stages']
                         for name, parameters in stage.items() if name == 'tune')
            passed = {key.replace('-', '_') for key in block}
            assert passed <= accepted, f'{family}{tier}: unknown {sorted(passed - accepted)}'


def test_the_study_name_is_PER_FAMILY():
    """Three families can share an output root, and optuna resumes a study BY NAME. One shared name would make a
    diffusion sweep resume a U-net's journal and try to read its trial vocabulary."""
    keywords = {keyword.arg: keyword.value for keyword in _run_sweep_call().keywords}
    study_name = keywords['study_name']
    assert isinstance(study_name, ast.JoinedStr), 'the study name must interpolate the family'
    assert 'model_family' in ast.unparse(study_name)


# =====================================================================================================================
# The selection metric is NOT a stage parameter
# =====================================================================================================================
def test_the_stage_takes_NO_selection_metric_or_mode():
    """The search space's ``selection:`` block is the single source of truth: ``run_sweep`` reads it from
    ``model-config`` and records it into ``best_trial.json``, and ``retrain_best`` reads it back. A stage parameter
    would let a retrain rank on a different composite than the sweep that chose the configuration."""
    parameters = set(inspect.signature(tune.tune).parameters)
    assert 'selection_metric' not in parameters
    assert 'selection_mode' not in parameters


def test_no_shipped_config_passes_a_selection_metric(repo_root):
    import glob

    for path in glob.glob(os.path.join(repo_root, 'config/*/*.yaml')):
        text = open(path).read()
        assert 'selection-metric' not in text, path
        assert 'selection-mode' not in text, path


def test_the_model_config_is_required_because_it_carries_the_selection_block():
    """It is the only non-defaulted path argument besides the input/output pair — a sweep with no search space has
    nothing to sample AND no composite to rank on."""
    parameters = inspect.signature(tune.tune).parameters
    assert parameters['model_config'].default is inspect.Parameter.empty


# =====================================================================================================================
# Stage wiring
# =====================================================================================================================
def test_the_stage_is_wrapped_with_fire():
    assert 'Fire(tune)' in inspect.getsource(tune)


def test_the_stage_imports_root_path_BEFORE_any_src_import(repo_root):
    lines = [line for line in open(os.path.join(repo_root, 'src/stages/tune.py'))
             if line.startswith('from ') or line.startswith('import ')]
    root_position = next(index for index, line in enumerate(lines) if '__init__ import root_path' in line)
    src_positions = [index for index, line in enumerate(lines) if line.startswith('from src.')]
    assert src_positions and min(src_positions) > root_position
