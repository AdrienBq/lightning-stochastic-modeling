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

from src.utils.modeling import search
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


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_no_occurrence_head_block_survives(family, search_spaces):
    """The gate/occurrence-head was dropped in block 3c: the bounded 0-24 target needs no dry-cell gate, and the
    report-only probability head went with it. A surviving block would be sampled into every trial and silently
    ignored, costing search budget on a dimension nothing reads."""
    assert 'occurrence_head' not in search_spaces[family]


@pytest.mark.source_invariant
@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_no_space_still_carries_the_dangling_average_precision_WEIGHT(family, repo_root):
    """It was a commented-out weight left behind when the composite was re-decided, sitting directly under the live
    weights — read as authoritative by anyone skimming the block. ``average_precision_occurrence`` IS still returned, as
    an unweighted diagnostic, which is the recorded mitigation for the regression composite having no false-alarm term.
    """
    import os

    text = open(os.path.join(repo_root, f'config/{family}/search_space_daily.yaml')).read()
    assert 'average_precision_occurrence: 0.40' not in text


@pytest.mark.source_invariant
def test_the_prose_no_longer_documents_a_THIRD_weighting(repo_root):
    """The comment above the ``selection:`` block used to describe ``0.40 AP + 0.30 + 0.30`` — a weighting the block
    below never had, and a different one again from the 0.50/0.50 the weights themselves said. Three sources, three
    answers; the prose is the one a reader trusts first."""
    import os

    text = open(os.path.join(repo_root, 'config/deterministic_unet/search_space_daily.yaml')).read()
    assert '0.40 * average_precision_occurrence' not in text


@pytest.mark.source_invariant
def test_the_mc_dropout_prose_names_the_RENAMED_ensemble_builder(repo_root):
    """Its comment claimed ``build_finetune_loss`` was removed and folded into ``build_regression_loss``. It was not:
    ``mc_dropout_module`` computes ``loss_reg + weight * loss_crps``, so the two are used TOGETHER in one expression and
    have different signatures. The builder was RENAMED to ``build_ensemble_loss`` for what it actually is."""
    import os

    text = open(os.path.join(repo_root, 'config/mc_dropout/search_space_daily.yaml')).read()
    assert 'build_finetune_loss is removed' not in text
    assert 'build_ensemble_loss' in text


def test_upstream_model_path_is_KEYWORD_ONLY():
    """The signature exists to prevent one specific silent bug. Branch A's call site was
    ``apply_constraints(trial, rng)`` — and a ``numpy`` ``Generator`` is TRUTHY, so passed positionally into this
    parameter it would have forced ``finetuning.enabled = True`` on every trial of every family, with no error and no
    log line. Keyword-only makes that call a ``TypeError`` instead."""
    import inspect

    parameter = inspect.signature(apply_constraints).parameters['upstream_model_path']
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_a_positional_second_argument_RAISES():
    """The executable half of the guard above, with the exact value that made it necessary."""
    import numpy as np

    with pytest.raises(TypeError):
        apply_constraints({'finetuning': {'enabled': False}}, np.random.default_rng(0))


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


# =====================================================================================================================
# Block 5c — the primitive sampler
#
# ⚠️ ``_sample_value`` already had 100 % LINE coverage before this section existed: every test above drives it through
# ``sample_trial``. What it lacked was a test of its own, and the two are not the same thing — through the sweep the
# only observable is "a trial dict came out", so a log-uniform silently sampling uniformly, or an int range excluding
# its upper bound, would look exactly like correct behaviour.
# =====================================================================================================================
def test_a_categorical_draws_only_from_its_declared_choices():
    rng = np.random.default_rng(0)
    drawn = {search._sample_value({'type': 'categorical', 'choices': ['a', 'b', 'c']}, rng) for _ in range(200)}
    assert drawn == {'a', 'b', 'c'}, 'every choice must be reachable, and nothing else'


def test_an_int_range_is_INCLUSIVE_at_both_ends():
    """``numpy``'s ``integers`` is half-open, so the ``+ 1`` is what makes ``high`` reachable. Silently excluding it
    would mean ``depth: {low: 2, high: 4}`` never samples the deepest U-net the space declares."""
    rng = np.random.default_rng(0)
    drawn = {search._sample_value({'type': 'int', 'low': 2, 'high': 4}, rng) for _ in range(300)}
    assert drawn == {2, 3, 4}
    assert all(isinstance(value, int) for value in drawn)


def test_a_float_range_stays_inside_its_bounds_and_returns_a_python_float():
    rng = np.random.default_rng(0)
    for _ in range(200):
        value = search._sample_value({'type': 'float', 'low': 0.1, 'high': 0.5}, rng)
        assert isinstance(value, float) and 0.1 <= value <= 0.5


def test_a_LOG_scaled_float_samples_uniformly_in_the_EXPONENT():
    """The learning rate spans 1e-5 to 1e-2 — three decades. Sampled linearly, ~90 % of trials would land in the top
    decade and the sweep would barely explore the small rates. The observable difference is the MEDIAN: log-uniform
    puts it at the geometric mean, linear at the arithmetic one."""
    rng = np.random.default_rng(0)
    node = {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True}
    drawn = np.array([search._sample_value(node, rng) for _ in range(4000)])

    assert drawn.min() >= 1e-5 and drawn.max() <= 1e-2
    geometric_mean = np.sqrt(1e-5 * 1e-2)
    assert abs(np.median(drawn) - geometric_mean) < 0.35 * geometric_mean
    assert np.median(drawn) < 0.1 * 0.5 * (1e-5 + 1e-2), 'a linear sample would sit near the arithmetic mean'


def test_the_absence_of_a_log_flag_means_LINEAR():
    rng = np.random.default_rng(0)
    drawn = np.array([search._sample_value({'type': 'float', 'low': 1e-5, 'high': 1e-2}, rng)
                      for _ in range(4000)])
    assert abs(np.median(drawn) - 0.5 * (1e-5 + 1e-2)) < 0.1 * (1e-2)


def test_the_sampler_is_reproducible_under_a_fixed_generator():
    """``PIPELINE_SEED`` makes a sweep reproducible only if every draw is a pure function of the generator state."""
    node = {'type': 'float', 'low': 0.0, 'high': 1.0}
    assert search._sample_value(node, np.random.default_rng(11)) == \
        search._sample_value(node, np.random.default_rng(11))


# =====================================================================================================================
# The HOURLY search space  (Step 4 block 4f)
#
# The daily spaces above are proven usable by SAMPLING them. That is not enough here, and the difference is the point:
# the hourly space is the first to reach `build_binary_loss`, whose `focal_bce` branch reads `positive_class_weight`
# and `focal_gamma` with `[]` rather than `.get`, and the first whose loss names are checked against the MODE at module
# construction (`UnetModuleBase` raises for a binary loss on daily and for `crps_binary` on a single-pass family). A
# sampled dict that looks fine is exactly what shipped the two `KeyError`s the test above commemorates, so this one
# builds the module and takes a training step.
# =====================================================================================================================
def test_the_hourly_space_samples_a_usable_trial(search_space_hourly):
    rng = np.random.default_rng(0)
    for _ in range(10):
        trial = apply_constraints(sample_trial(search_space_hourly, rng))
        assert isinstance(trial['batch_size'], int)
        assert 'lr' in trial['optimizer']
        assert trial['loss']['name'] in ('focal_bce', 'dice', 'brier', 'wmse_psd')
        # ⚠️ NOT `trial['max_hours'] == 24` like the daily spaces: the key is absent here by decision (a 0/1 target has
        # no hour ceiling), and the module defaults it. Asserting its absence is asserting the decision.
        assert 'max_hours' not in trial and 'output_activation' not in trial


def test_the_focal_bce_rule_FIRES_on_the_hourly_space(search_space_hourly):
    """``apply_constraints``' first rule is documented as *"inert in the three daily search spaces, whose loss.name
    choices are all distance losses; it fires only where focal_bce is reachable, i.e. an hourly pipeline"*. This is that
    pipeline, so the rule stops being unreachable-by-configuration — and gamma must come back exactly 0 whenever
    focal_bce is sampled, because focal BCE already carries its own positive_class_weight."""
    rng = np.random.default_rng(3)
    fired = 0
    for _ in range(60):
        trial = apply_constraints(sample_trial(search_space_hourly, rng))
        if trial['loss']['name'] == 'focal_bce':
            fired += 1
            assert trial['loss']['intensity_weight_gamma'] == 0.0
        else:
            assert 'intensity_weight_gamma' in trial['loss']       # the reweighting knob the distance losses need
    assert fired, 'focal_bce was never sampled in 60 draws -- the rule is still unreachable'


@pytest.mark.parametrize('loss_name', ['focal_bce', 'dice', 'brier', 'wmse_psd'])
def test_every_hourly_LOSS_CHOICE_builds_a_module_and_takes_a_step(
        loss_name, search_space_hourly, normalization, target_stats
):
    """⭐ The end-to-end check on the space: each of the four reachable losses, built into a real hourly module, one
    forward + backward on a 0/1 target. Two of the four take LOGITS through ``build_binary_loss`` and two take the
    PROBABILITY through ``build_regression_loss``, and the module dispatches on the NAME — so this is also the test that
    the mixed list is admissible rather than merely plausible.
    """
    import torch

    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    rng = np.random.default_rng(11)
    trial = None
    for _ in range(200):                                   # draw until this loss comes up, keeping every other key
        candidate = apply_constraints(sample_trial(search_space_hourly, rng))
        if candidate['loss']['name'] == loss_name:
            trial = candidate
            break
    assert trial is not None, f'{loss_name} never sampled'
    trial['unet'] = {**trial['unet'], 'base_channels': 8, 'depth': 2}          # keep the step cheap

    module = DeterministicUnetModule(trial, 5, target_stats(mode='hourly'), normalization)
    assert module.hourly
    assert module.loss_takes_logits is (loss_name in ('focal_bce', 'dice', 'brier'))

    x = torch.randn(2, 5, 16, 24)
    y = (torch.rand(2, 16, 24) < 0.05).float()              # a 0/1 occurrence target at a plausible base rate
    loss = module.training_step((x, y), 0)
    assert torch.isfinite(loss), loss
    loss.backward()


def test_an_hourly_module_reports_its_PREDICTION_as_the_probability(search_space_hourly, normalization, target_stats):
    """The one line that makes the reliability diagram, ``explained_deviance`` and ``dice`` appear on this task and be
    absent on the daily one. Checked through the shipped space rather than a hand-built trial, so it covers the
    configuration a real hourly sweep would produce."""
    import torch

    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    trial = apply_constraints(sample_trial(search_space_hourly, np.random.default_rng(5)))
    trial['unet'] = {**trial['unet'], 'base_channels': 8, 'depth': 2}
    module = DeterministicUnetModule(trial, 5, target_stats(mode='hourly'), normalization).eval()

    x = torch.randn(2, 5, 16, 24)
    y = (torch.rand(2, 16, 24) < 0.05).float()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert output['probability'] is not None
    assert torch.equal(output['probability'], output['prediction'])
    assert float(output['prediction'].min()) >= 0.0 and float(output['prediction'].max()) <= 1.0


def test_the_hourly_space_can_fit_PLATT_calibration(search_space_hourly, normalization, target_stats):
    """``calibration.occurrence: platt`` has never been fitted by a shipped pipeline — it is inert in all three daily
    spaces, where the head emits hours and there is no logit to scale. Here it is the ONE calibrator the task admits, so
    the space must be able to reach it and the module must schedule its phase."""
    import torch

    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    assert 'platt' in search_space_hourly['calibration']['occurrence']['choices']

    trial = apply_constraints(sample_trial(search_space_hourly, np.random.default_rng(2)))
    trial['calibration'] = {'occurrence': 'platt'}
    trial['unet'] = {**trial['unet'], 'base_channels': 8, 'depth': 2}
    module = DeterministicUnetModule(trial, 5, target_stats(mode='hourly'), normalization)

    assert module.training_phases() == ('train', 'occurrence_calibration')
    assert module.net.output_calibration is not None, 'the Platt layer must live in the net, so it travels in state_dict'
    module.set_phase('occurrence_calibration')
    loss = module.training_step((torch.randn(2, 5, 16, 24), (torch.rand(2, 16, 24) < 0.05).float()), 0)
    assert torch.isfinite(loss), loss
