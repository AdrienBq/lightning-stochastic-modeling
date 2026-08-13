"""Tests for src/utils/modeling/search.py — sampling a trial from a search space, and repairing it.

The file is nearly identical on both source branches; branch A carried one extra ``apply_constraints`` rule keyed on
the gaussianized target transform, which is dead under this scope, so branch D was the base.

``apply_constraints`` is the interesting part, and the thing to hold onto is what it does NOT do: it repairs the sampled
DICT and never loads a checkpoint. The weights are loaded by the stage's ``module_factory``. Rule 2 also LOGS a
behaviour it does not implement — the ``unet`` block being ignored is an obligation owed by
``MCDropoutModule.from_upstream``, not something enforced here — and the byte-identity test below is what keeps that
docstring honest.
"""
import copy

import numpy as np
import pytest

from src.utils.modeling.search import (
    apply_constraints, flatten_trial, is_parameter_node, sample_trial, suggest_trial_optuna,
)


# =====================================================================================================================
# Sampling
# =====================================================================================================================
def test_a_parameter_node_is_recognised_by_its_type_key():
    assert is_parameter_node({'type': 'float', 'low': 0.0, 'high': 1.0})
    assert is_parameter_node({'type': 'categorical', 'choices': ['a', 'b']})
    assert not is_parameter_node({'nested': {'type': 'int', 'low': 1, 'high': 2}})
    assert not is_parameter_node(4)
    assert not is_parameter_node('softplus')


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_every_shipped_search_space_samples_a_usable_trial(family, search_spaces):
    """Driven from the REAL YAML rather than a fixture: this is the check whose absence let two ``KeyError``s ship, one
    on ``optimizer.learning_rate`` and one on ``optimizer.batch_size``."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        trial = apply_constraints(sample_trial(search_spaces[family], rng))
        assert isinstance(trial['batch_size'], int)
        assert 'lr' in trial['optimizer']
        assert trial['max_hours'] == 24


def test_fixed_scalars_pass_through_unsampled(search_spaces):
    """A plain value in the YAML is a FIXED choice, not a one-element distribution. ``kernel_size: 3`` must come back as
    3 rather than as a dict."""
    rng = np.random.default_rng(1)
    trial = sample_trial(search_spaces['deterministic_unet'], rng)
    assert trial['unet']['kernel_size'] == 3
    assert trial['max_epochs'] == search_spaces['deterministic_unet']['max_epochs']


def test_sampling_is_reproducible_under_a_seed(search_spaces):
    space = search_spaces['deterministic_unet']
    assert sample_trial(space, np.random.default_rng(7)) == sample_trial(space, np.random.default_rng(7))


def test_sampling_actually_varies_across_seeds(search_spaces):
    space = search_spaces['deterministic_unet']
    trials = [flatten_trial(sample_trial(space, np.random.default_rng(seed))) for seed in range(8)]
    assert len({tuple(sorted(t.items())) for t in trials}) > 1


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_a_sampled_float_respects_its_bounds(family, search_spaces):
    rng = np.random.default_rng(2)
    space = search_spaces[family]
    for _ in range(20):
        trial = sample_trial(space, rng)
        low = space['optimizer']['lr']['low']
        high = space['optimizer']['lr']['high']
        assert low <= trial['optimizer']['lr'] <= high


def test_a_sampled_categorical_is_one_of_its_choices(search_spaces):
    rng = np.random.default_rng(3)
    space = search_spaces['deterministic_unet']
    choices = space['loss']['name']['choices']
    for _ in range(20):
        assert sample_trial(space, rng)['loss']['name'] in choices


def test_the_intensity_gamma_range_reaches_zero(search_spaces):
    """``intensity_weights(y, 0) = 1``, so gamma = 0 is how ``weighted_mae`` covers a plain MAE. If the space excluded
    it, the unweighted loss would be unreachable."""
    for family, space in search_spaces.items():
        assert space['loss']['intensity_weight_gamma']['low'] == 0.0, family


# =====================================================================================================================
# apply_constraints
# =====================================================================================================================
def test_the_focal_bce_rule_zeroes_the_intensity_gamma():
    """``focal_bce`` brings its OWN class reweighting via ``positive_class_weight``, and on a binary target
    ``(1 + y)^gamma`` takes values in {1, 2^gamma} — a second multiplicative positive-class weight. Two independent
    knobs for one effect make a sweep's results uninterpretable, so one is pinned."""
    trial = apply_constraints({'loss': {'name': 'focal_bce', 'intensity_weight_gamma': 3.0,
                                        'positive_class_weight': 4.0}})
    assert trial['loss']['intensity_weight_gamma'] == 0.0


def test_the_rule_leaves_a_distance_loss_alone():
    """The rule is keyed on the loss NAME, and gamma is the reweighting knob for every other loss — including the
    binary ones that bring no weight of their own."""
    for name in ('weighted_mae', 'brier', 'dice', 'crps_binary'):
        trial = apply_constraints({'loss': {'name': name, 'intensity_weight_gamma': 3.0}})
        assert trial['loss']['intensity_weight_gamma'] == 3.0, name


def test_an_upstream_model_forces_finetuning_on():
    """Load-bearing rather than cosmetic: ``set_phase`` RAISES when the finetune phase is requested with
    ``finetuning.enabled`` false. Without the rule, a warm-started trial that sampled ``false`` would either raise or
    run phase 1 alone — deterministically re-training the warm-started net with no MC calibration, which is the entire
    point of the run."""
    trial = apply_constraints({'finetuning': {'enabled': False}}, upstream_model_path='/tmp/upstream.ckpt')
    assert trial['finetuning']['enabled'] is True


def test_no_upstream_leaves_finetuning_as_sampled():
    trial = apply_constraints({'finetuning': {'enabled': False}}, upstream_model_path=None)
    assert trial['finetuning']['enabled'] is False
    trial = apply_constraints({'finetuning': {'enabled': False}}, upstream_model_path='')
    assert trial['finetuning']['enabled'] is False


def test_the_unet_block_comes_back_BYTE_IDENTICAL_under_a_warm_start():
    """Rule 2's docstring says the sampled ``unet`` block "is ignored", and that is a LOG LINE about an obligation owed
    by ``MCDropoutModule.from_upstream`` — the body of ``if 'unet' in trial:`` is a bare ``logger.info``. Nothing is
    removed, overwritten or flagged here.

    Pinned so the docstring cannot start claiming otherwise, and so the reader is not misled into thinking the override
    happens at this layer.
    """
    unet = {'base_channels': 16, 'depth': 3, 'activation': 'relu', 'normalization': 'group'}
    trial = {'unet': copy.deepcopy(unet), 'finetuning': {'enabled': False}}
    repaired = apply_constraints(trial, upstream_model_path='/tmp/upstream.ckpt')
    assert repaired['unet'] == unet


def test_apply_constraints_never_touches_the_filesystem():
    """It holds the checkpoint path only as a string it tests for truthiness. The loading lives in the stage that builds
    the ``module_factory`` — so a nonexistent path must pass through without an error here."""
    trial = apply_constraints({'finetuning': {'enabled': False}},
                              upstream_model_path='/nonexistent/definitely/not/a/file.ckpt')
    assert trial['finetuning']['enabled'] is True


def test_apply_constraints_repairs_IN_PLACE_as_well_as_returning():
    """It mutates the dict it is given. Harmless at the one call site — ``tuning.py`` does
    ``trial = apply_constraints(trial, ...)`` and records ``flatten_trial(trial)`` into ``trials.csv`` AFTERWARDS, so the
    table shows what was actually trained rather than what was sampled.

    ⚠️ It does mean the OPTUNA path diverges from its own record: ``study.ask()`` registers the suggested
    ``intensity_weight_gamma`` in optuna's trial params, the repair then forces it to 0, and ``study.tell`` attributes
    the resulting score to the value that was never used — so the TPE surrogate learns from a phantom. Currently
    unreachable, because the rule only fires for ``focal_bce`` and no shipped search space offers it (see
    losses_test.py). Documented here rather than fixed, since making the function copy would not close it: the
    divergence is inherent to repairing a trial after optuna has recorded it.
    """
    original = {'loss': {'name': 'focal_bce', 'intensity_weight_gamma': 3.0}}
    returned = apply_constraints(original)
    assert returned['loss']['intensity_weight_gamma'] == 0.0
    assert original['loss']['intensity_weight_gamma'] == 0.0, 'documented in-place behaviour'


def test_the_transform_conditioned_rule_is_gone():
    """Branch A carried an extra rule disabling monotone regression calibration under the gaussianized target
    transform. The transform is removed, so the rule is dead — branch D was taken as the base precisely because it
    never had it."""
    import inspect

    source = inspect.getsource(apply_constraints)
    for token in ('gaussian', 'transform', 'log_warp'):
        assert token not in source.lower(), f'a {token}-conditioned rule reappeared'


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_constraints_are_idempotent(family, search_spaces):
    rng = np.random.default_rng(4)
    trial = sample_trial(search_spaces[family], rng)
    once = apply_constraints(trial)
    assert apply_constraints(once) == once


# =====================================================================================================================
# flatten_trial — what reaches the trials table
# =====================================================================================================================
def test_flatten_trial_dots_the_nested_keys():
    flat = flatten_trial({'unet': {'depth': 3}, 'batch_size': 8, 'loss': {'name': 'weighted_mae'}})
    assert flat['unet.depth'] == 3
    assert flat['batch_size'] == 8
    assert flat['loss.name'] == 'weighted_mae'


def test_flatten_trial_is_flat():
    flat = flatten_trial({'a': {'b': {'c': 1}}})
    assert flat == {'a.b.c': 1}
    assert not any(isinstance(value, dict) for value in flat.values())


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_a_flattened_real_trial_has_no_dict_values(family, search_spaces):
    """The trials table is a CSV, so a surviving dict would be stringified into an unusable column."""
    flat = flatten_trial(sample_trial(search_spaces[family], np.random.default_rng(5)))
    assert not any(isinstance(value, dict) for value in flat.values())


# =====================================================================================================================
# The optuna path samples the same space
# =====================================================================================================================
@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_the_optuna_suggester_produces_the_same_trial_shape(family, search_spaces):
    """Two samplers over one space: ``sample_trial`` for the random path and ``suggest_trial_optuna`` for the guided
    one. If their key sets diverged, a sweep would silently train a different parameterisation than a smoke run."""
    optuna = pytest.importorskip('optuna')

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    suggested = suggest_trial_optuna(search_spaces[family], study.ask())
    randomly = sample_trial(search_spaces[family], np.random.default_rng(0))
    assert set(flatten_trial(suggested)) == set(flatten_trial(randomly))
