"""Tests for src/utils/modeling/unet_module_base.py — the base the two U-net families share.

Extracted in block 3d so the deterministic and MC-dropout modules share one implementation of the phase machinery,
the loss dispatch and the validation accumulation. It exists on neither source branch.

Two of its class attributes were made **required rather than conventional**, which turned two live hazards into
impossibilities. Those are the tests that matter most here:

* ``CHECKPOINT_MARKER = None`` on the base and ``on_save_checkpoint`` raising if it is unset — so a new family cannot
  inherit another family's marker and be silently loaded as the wrong class by the registry;
* ``predict_step`` abstract — so a stochastic family cannot silently inherit a POINT prediction, satisfy the contract,
  and report no ensemble metrics at all.
"""
import inspect

import pytest
import torch

from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.unet_module_base import UnetModuleBase


# =====================================================================================================================
# The two required class attributes
# =====================================================================================================================
def test_the_base_declares_no_checkpoint_marker():
    assert UnetModuleBase.CHECKPOINT_MARKER is None


def test_a_family_without_a_marker_cannot_save(unet_trial, normalization, target_stats):
    """The hazard this closes: a new family that forgot its marker would inherit the base's, and the registry would
    load its checkpoints as whichever class that names — scoring one family's weights with another's module."""
    class MarkerLess(UnetModuleBase):
        def predict_step(self, batch, batch_idx):
            return {}

    module = MarkerLess(unet_trial(), 5, target_stats(), normalization)
    with pytest.raises((ValueError, AssertionError, TypeError)):
        module.on_save_checkpoint({})


def test_predict_step_is_abstract_on_the_base():
    """A stochastic family that forgot to override it would return a POINT prediction, satisfy the shared contract, and
    report no ensemble metrics — CRPS, spread-skill and the rank histogram all silently absent."""
    with pytest.raises(NotImplementedError):
        UnetModuleBase.predict_step(object(), (None, None), 0)


def test_the_base_does_not_support_the_ensemble_loss():
    """Opt-in, not opt-out: the deterministic family must not be able to train ``crps_binary``, whose spread term would
    be identically zero from a single forward pass."""
    assert UnetModuleBase.SUPPORTS_ENSEMBLE_LOSS is False


def test_the_base_phases_are_train_plus_the_two_calibrations():
    assert UnetModuleBase.PHASES[0] == 'train'
    assert set(UnetModuleBase.PHASES[1:]) == {'occurrence_calibration', 'regression_calibration'}


# =====================================================================================================================
# Loss dispatch is by NAME, not by mode
# =====================================================================================================================
def test_a_binary_loss_on_the_daily_target_raises(unet_trial, normalization, target_stats):
    """The 0-24 target is a bounded REGRESSION, so a binary loss there is a category error rather than an odd choice.
    Raising at construction beats training a model whose loss silently treats 7 lightning-hours as a logit."""
    with pytest.raises(ValueError):
        DeterministicUnetModule(unet_trial(loss={'name': 'brier'}), 5, target_stats(mode='daily'), normalization)


def test_a_distance_loss_on_the_hourly_target_is_ALLOWED(unet_trial, normalization, target_stats):
    """Deliberate, and the reason the dispatch is on the loss NAME rather than the mode: the hourly prediction is a
    PROBABILITY, and `rmse(p, y) ** 2 == brier_score(p, y)` exactly — so a distance loss on it is proper, not a
    mistake. Forbidding it would rule out a legitimate configuration."""
    module = DeterministicUnetModule(unet_trial(loss={'name': 'weighted_mae', 'intensity_weight_gamma': 0.0}),
                                     5, target_stats(mode='hourly'), normalization)
    assert module.loss_takes_logits is False


def test_a_binary_loss_on_the_hourly_target_takes_logits(unet_trial, normalization, target_stats):
    module = DeterministicUnetModule(unet_trial(loss={'name': 'brier'}), 5, target_stats(mode='hourly'),
                                     normalization)
    assert module.loss_takes_logits is True


def test_an_ensemble_loss_on_a_family_that_cannot_provide_members_raises(
        unet_trial, normalization, target_stats):
    """``crps_binary`` needs a genuine sample axis. The deterministic family's single forward pass gives N=1, a zero
    spread term and a silent MAE-on-probabilities — so it is refused rather than quietly degraded."""
    with pytest.raises(ValueError):
        DeterministicUnetModule(unet_trial(loss={'name': 'crps_binary'}), 5, target_stats(mode='hourly'),
                                normalization)


# =====================================================================================================================
# Mode-derived state
# =====================================================================================================================
def test_the_mode_comes_from_target_stats_not_a_constructor_argument():
    """``prepare_regression`` writes ``mode`` into ``target_stats.json``, so the factory signature
    ``(trial, in_channels, target_stats, normalization)`` needed no new argument for the task split."""
    parameters = list(inspect.signature(UnetModuleBase.__init__).parameters)
    assert parameters == ['self', 'trial', 'in_channels', 'target_stats', 'normalization']


def test_a_missing_mode_raises(unet_trial, normalization):
    """Better than defaulting to daily: an artifact with no recorded mode is one this code cannot interpret, and
    guessing would silently train the wrong head."""
    with pytest.raises((KeyError, ValueError)):
        DeterministicUnetModule(unet_trial(), 5, {}, normalization)


def test_the_residual_channel_count_is_validated_against_the_normalization(
        unet_trial, normalization, target_stats):
    """``feature_mean`` is a checkpoint buffer the dataset never sees, so the channel-count check has to live on the
    model side. A mismatch means the prepared directory and the checkpoint disagree about the upstream channel."""
    with pytest.raises(ValueError):
        DeterministicUnetModule(unet_trial(), 6, target_stats(), normalization)


# =====================================================================================================================
# Phases
# =====================================================================================================================
def test_an_unknown_phase_raises(unet_trial, normalization, target_stats):
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization)
    with pytest.raises(ValueError):
        module.set_phase('pretrain')


def test_the_monitor_metric_equals_the_composite_the_sweep_ranks_on(unet_trial, normalization, target_stats):
    """If the checkpoint monitor and ``run_sweep``'s prune metric diverge, the sweep prunes on one quantity and keeps
    the best checkpoint by another — silently, and only visible as a trials table whose best row is not the best
    checkpoint. Both are derived from one source of truth."""
    from src.utils.modeling.validation import selection_metric_for_mode

    for mode in ('daily', 'hourly'):
        loss = {'name': 'brier'} if mode == 'hourly' else {'name': 'weighted_mae', 'intensity_weight_gamma': 0.0}
        module = DeterministicUnetModule(unet_trial(loss=loss), 5, target_stats(mode=mode), normalization)
        assert module.monitor_metric == selection_metric_for_mode(mode), mode
    assert module.monitor_mode == 'max'


def test_the_learning_rate_is_read_from_lr_not_learning_rate(unet_trial, normalization, target_stats):
    """A live `KeyError` when Step 3 started: branch A read `optimizer.learning_rate` while Step 2's config writes
    `optimizer.lr`. Read through `_learning_rate()` rather than `configure_optimizers()`, which needs an attached
    Trainer for its scheduler's step count."""
    trial = unet_trial()
    trial['optimizer'] = {**trial['optimizer'], 'lr': 3e-4}
    module = DeterministicUnetModule(trial, 5, target_stats(), normalization)
    module.set_phase('train')
    assert module._learning_rate() == pytest.approx(3e-4)


@pytest.mark.source_invariant
def test_the_output_activation_is_read_from_the_TOP_level_of_the_trial():
    """The third key-contract mismatch of the same family: branch A read ``trial['unet']['output_activation']`` while
    Step 2 put it at the top level. Unlike the other two this one did NOT raise — it silently defaulted, so the key the
    user chose to keep was never read and the activation was whatever the default happened to be."""
    source = inspect.getsource(UnetModuleBase)
    assert "trial.get('output_activation'" in source
    assert "['unet']['output_activation']" not in source


def test_a_dozen_SAMPLED_trials_all_build_a_working_module(search_spaces, normalization, target_stats):
    """The check whose absence let two ``KeyError``s ship. Building from a hand-written fixture cannot catch a config
    contract mismatch; building from what ``sample_trial`` actually produces can, and twelve draws cover the categorical
    combinations that a single draw misses.

    ``configure_optimizers`` is exercised through ``_learning_rate`` because the real call needs an attached Trainer for
    its scheduler's step count.
    """
    import numpy as np

    from src.utils.modeling.search import apply_constraints, sample_trial

    space = search_spaces['deterministic_unet']
    for attempt in range(12):
        trial = apply_constraints(sample_trial(space, np.random.default_rng(attempt)))
        module = DeterministicUnetModule(trial, 5, target_stats(), normalization)
        module.set_phase('train')
        assert module._learning_rate() > 0, attempt
        assert int(trial['batch_size']) > 0, attempt


def test_the_batch_size_is_read_from_the_TOP_level(search_spaces):
    """The other live ``KeyError``: ``tuning.py`` read ``trial['optimizer']['batch_size']`` while Step 2 moved it to the
    top level and no ``tune`` stage passes ``batch-size:``. It would have aborted trial 0 of every family."""
    for family, space in search_spaces.items():
        assert 'batch_size' in space, family
        assert 'batch_size' not in space.get('optimizer', {}), family


# =====================================================================================================================
# The calibration phases: each trains ONLY its own layer, and monitors its own metric
# =====================================================================================================================
@pytest.fixture
def platt_module(unet_trial, normalization, target_stats):
    """An hourly module with Platt calibration enabled, advanced into its calibration phase."""
    trial = unet_trial(loss={'name': 'brier'},
                       calibration={'occurrence': 'platt', 'regression': {'structure': 'none'}})
    module = DeterministicUnetModule(trial, 5, target_stats(mode='hourly'), normalization)
    module.set_phase('occurrence_calibration')
    return module


def test_a_calibration_phase_monitors_its_OWN_metric(platt_module):
    """Not the composite. The calibration phase fits one or two scalars on a frozen backbone, so ranking it by the full
    composite would compare it against the train phase's checkpoint on a quantity it barely moves."""
    assert platt_module.monitor_metric == 'valid_occurrence_calibration'


def test_a_calibration_phase_FREEZES_everything_but_its_own_layer(platt_module):
    """The point of the phase. If the backbone stayed trainable, the "calibration" step would be a second training run
    at the calibration learning rate — which fits, writes a checkpoint, and scores plausibly."""
    frozen = [name for name, parameter in platt_module.net.named_parameters()
              if parameter.requires_grad and 'output_calibration' not in name]
    assert not frozen, f'still trainable: {frozen}'
    assert all(parameter.requires_grad for parameter in platt_module.net.output_calibration_parameters())


def test_the_platt_layer_is_owned_by_the_NET_not_the_module(platt_module):
    """Block 3d-0 rewired it. A module-level Platt is not in ``net.state_dict()``, so ``from_upstream``'s
    ``load_state_dict`` would silently discard fitted Platt weights when warm-starting an hourly MC-dropout run from an
    hourly deterministic upstream. Rewiring closed the hole rather than documenting it."""
    assert not isinstance(getattr(platt_module, 'output_calibration', None), torch.nn.Module)
    assert any('output_calibration' in key for key in platt_module.net.state_dict())


# =====================================================================================================================
# Which SPACE the loss is fed, per mode — the 3b-1r contract, end to end
# =====================================================================================================================
def test_an_hourly_BINARY_loss_receives_the_raw_logit(unet_trial, normalization, target_stats):
    """The uniform logit contract reaching the module layer: the head emits a logit and every binary loss sigmoids
    internally, so the module must pass the head output through untouched. Passing the probability instead is finite and
    plausible — a double sigmoid — which is why this is checked numerically rather than by a flag."""
    from src.utils.modeling import losses

    module = DeterministicUnetModule(unet_trial(loss={'name': 'brier'}), 5,
                                     target_stats(mode='hourly'), normalization).eval()
    torch.manual_seed(0)
    logit = torch.randn(2, 8, 8)
    labels = (torch.rand(2, 8, 8) < 0.3).float()

    got = module._fitting_loss(logit, labels)
    expected = losses.brier_loss(logit, labels)
    wrong = losses.brier_loss(torch.sigmoid(logit), labels)

    assert float(got) == pytest.approx(float(expected), abs=1e-6)
    assert float(got) != pytest.approx(float(wrong), abs=1e-6), 'a double sigmoid would not raise'


def test_an_hourly_DISTANCE_loss_receives_the_PROBABILITY_and_is_the_brier_score(
        unet_trial, normalization, target_stats):
    """The other side of the dispatch, and the reason it is safe: a distance loss brings no sigmoid of its own, so the
    module has to apply it. ``rmse`` on a probability is exactly ``sqrt(brier_score)`` and therefore PROPER — which is
    what makes offering a distance loss on the hourly task legitimate rather than a mistake."""
    from src.utils.metrics import scores

    module = DeterministicUnetModule(unet_trial(loss={'name': 'weighted_mse', 'intensity_weight_gamma': 0.0}), 5,
                                     target_stats(mode='hourly'), normalization).eval()
    torch.manual_seed(0)
    logit = torch.randn(2, 8, 8)
    labels = (torch.rand(2, 8, 8) < 0.3).float()

    got = float(module._fitting_loss(logit, labels))
    assert got == pytest.approx(float(scores.brier_score(torch.sigmoid(logit).numpy(), labels.numpy())), abs=1e-6)


def test_the_numpy_VALIDATION_MIRROR_agrees_with_the_torch_calibration_objective(
        unet_trial, normalization, target_stats):
    """The second hand-written copy of one formula, and the one no invariant covered. ``_validation_reg_calibration``
    recomputes ``log1p_huber`` in numpy for the validation monitor; nothing checked the two agree. This is the
    ``crps_ensemble`` divergence risk in a place the merge did not look — if they drift, the calibration phase is
    monitored on a quantity it is not optimising.

    Asserted for ``pointwise`` only, deliberately: the torch loss sorts per BATCH and the mirror per EPOCH, so the
    ``quantile`` pair is not comparable and pretending otherwise would need a tolerance wide enough to hide a real
    divergence.
    """
    from src.utils.modeling.losses import log1p_huber

    delta = 1.0
    trial = unet_trial(calibration={'occurrence': 'none',
                                    'regression': {'structure': 'power_law', 'objective': 'pointwise',
                                                   'num_sigmoids': 4, 'huber_delta': delta}})
    module = DeterministicUnetModule(trial, 5, target_stats(), normalization)

    torch.manual_seed(0)
    observed = torch.randint(0, 25, (4, 8, 8)).float()
    predicted = (observed + torch.randn_like(observed)).clamp(0, 24)
    mask = torch.ones_like(observed, dtype=torch.bool)

    mirror = module._validation_reg_calibration(predicted.numpy(), observed.numpy(), mask.numpy())
    assert mirror == pytest.approx(float(log1p_huber(predicted, observed, mask, delta)), abs=1e-5)


def test_the_legacy_mode_alias_resolves_at_the_MODULE_layer(unet_trial, normalization, target_stats):
    """A checkpoint prepared under the old name must still build a module, not just load a dataset."""
    stats = {**target_stats(), 'mode': 'daily_lightning_hours'}
    module = DeterministicUnetModule(unet_trial(), 5, stats, normalization)
    assert not module.hourly


# =====================================================================================================================
# The training path never clamps
# =====================================================================================================================
def test_the_prediction_path_clamps_but_the_training_path_does_not(unet_trial, normalization, target_stats, batch):
    """``softplus`` gives the 0-hour floor structurally and is unbounded above, so the ceiling is applied in
    ``_to_prediction``. It must NOT be applied during training: clamping a live gradient zeroes it for exactly the
    over-predicting cells the loss needs to move."""
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization).eval()
    with torch.no_grad():
        module.net.head.bias.fill_(100.0)
        x, y = batch()
        raw = module._head_output(module(x))
        predicted = module._to_prediction(raw)

    assert float(raw.max()) > 24.0, 'the raw head must be able to exceed the ceiling'
    assert float(predicted.max()) <= 24.0 + 1e-5, 'the prediction path must clamp'


def test_the_prediction_is_detached(unet_trial, normalization, target_stats):
    """``_to_prediction`` detaches, so a validation pass cannot accidentally retain a graph over a whole epoch's
    accumulated predictions."""
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization)
    head = torch.randn(2, 8, 8, requires_grad=True)
    assert not module._to_prediction(head).requires_grad
