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


def test_the_batch_size_is_read_from_the_TOP_level(search_spaces):
    """The other live ``KeyError``: ``tuning.py`` read ``trial['optimizer']['batch_size']`` while Step 2 moved it to the
    top level and no ``tune`` stage passes ``batch-size:``. It would have aborted trial 0 of every family."""
    for family, space in search_spaces.items():
        assert 'batch_size' in space, family
        assert 'batch_size' not in space.get('optimizer', {}), family


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
