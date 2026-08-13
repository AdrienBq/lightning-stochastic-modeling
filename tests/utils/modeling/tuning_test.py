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
