"""Tests for src/utils/modeling/mc_dropout_module.py.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_probabilistic_eval_compat.py`` (the module constructs and
its MC forward produces distinct members), adapted to the Step 3 module, plus new coverage for the warm-start
mechanism — which exists on neither source branch.

⚠️ **One ported assertion changes meaning and is deliberately inverted.** A's trial set
``output_activation: scaled_sigmoid`` = ``max_hours * sigmoid(raw)``, so ``mc_forward`` was bounded in ``[0, 24]`` and
A asserted that. Step 3 correction 3 replaced both of D's ceiling-in-the-activation options with plain ``softplus`` and
moved the ceiling to a clamp in ``predict_step``, **per member**. So the bound is now asserted on ``predict_step``, and
the converse — that ``mc_forward`` may exceed 24 — is asserted too. Porting A's assertion unchanged would fail;
deleting it would leave the clamp untested.
"""
import numpy as np
import pytest
import torch

from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.mc_dropout_module import WARM_START_ARCHITECTURE_KEYS, MCDropoutModule

MEMBERS = 4


@pytest.fixture
def make_mc(mc_trial, normalization, target_stats):
    def build(mode='daily', **kwargs):
        return MCDropoutModule(mc_trial(**kwargs), 5, target_stats(mode=mode), normalization).eval()
    return build


# =====================================================================================================================
# Construction and the MC forward  (ported)
# =====================================================================================================================
def test_the_module_constructs_and_mc_forward_returns_a_member_stack(make_mc, batch):
    module = make_mc()
    x, _ = batch()
    with torch.no_grad():
        members = module.mc_forward(x, MEMBERS)
    assert members.shape == (MEMBERS, x.shape[0], x.shape[2], x.shape[3])


def test_mc_members_actually_differ(make_mc, batch):
    """The silent failure this guards: ``dropout_p`` is a TOP-LEVEL trial key while ``UNetBackbone`` reads
    ``unet['dropout']``, whose config value is a ``0.0`` placeholder. Left un-injected, every MC forward pass is
    deterministic, every member identical, the spread zero, and ``spread_skill_sums`` NaN through ``ddof=1`` — with no
    exception anywhere."""
    module = make_mc(dropout_p=0.3)
    x, _ = batch()
    with torch.no_grad():
        members = module.mc_forward(x, MEMBERS)
    spread = members.std(dim=0)
    assert float(spread.max()) > 0.0, 'members are identical: dropout_p never reached unet.dropout'


def test_dropout_p_is_injected_into_the_unet_config(make_mc):
    """``save_hyperparameters`` must record the EFFECTIVE architecture, so the injection happens before
    ``super().__init__`` — otherwise a reloaded checkpoint rebuilds the net with the 0.0 placeholder."""
    module = make_mc(dropout_p=0.25)
    assert module.hparams['trial']['unet']['dropout'] == pytest.approx(0.25)


def test_a_non_positive_dropout_is_rejected(mc_trial, normalization, target_stats):
    """A zero dropout makes the family pointless — it becomes the deterministic model wearing an ensemble contract,
    reporting a zero spread and NaN spread-skill. Raising is better than a silently degenerate run."""
    with pytest.raises(ValueError):
        MCDropoutModule(mc_trial(dropout_p=0.0), 5, target_stats(), normalization)


def test_mc_forward_restores_the_previous_training_mode(make_mc, batch):
    """``mc_forward`` flips only the dropout layers on and must put the net back as it found it, or a validation pass
    would silently leave the whole net in train mode for the next epoch."""
    module = make_mc()
    x, _ = batch()
    assert not module.net.training
    with torch.no_grad():
        module.mc_forward(x, 2)
    assert not module.net.training, 'mc_forward left the net in training mode'


# =====================================================================================================================
# The clamp moved from the activation to predict_step  (the inverted assertion)
# =====================================================================================================================
def test_predict_step_clamps_every_member_to_the_hour_ceiling(make_mc, batch):
    module = make_mc()
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0, ensemble_size=MEMBERS)
    members = output['ensemble_members']
    assert members.shape == (x.shape[0], MEMBERS, x.shape[2], x.shape[3])
    assert float(members.min()) >= 0.0
    assert float(members.max()) <= 24.0 + 1e-5
    # the MEAN being in range is not enough: a single over-predicting member would still corrupt the spread
    assert float(output['prediction'].max()) <= 24.0 + 1e-5


def test_the_ceiling_comes_from_the_clamp_and_not_from_the_activation(make_mc):
    """A's bound assertion still HOLDS but for a different reason, and this pins the difference.

    ``softplus`` gives the 0-hour floor structurally and is UNBOUNDED above, so driving the head hard makes the raw
    head output exceed 24 — while ``mc_forward``, which maps each member through ``_to_prediction``, comes back
    clamped. If the ceiling ever moved back into the activation this test would still pass; what it rules out is the
    ceiling being absent, which is what a bare ``softplus`` with no clamp would give.
    """
    module = make_mc()
    x = torch.zeros(1, 5, 16, 24)
    with torch.no_grad():
        module.net.head.bias.fill_(100.0)
        raw_head = module._head_output(module(x))
        members = module.mc_forward(x, 2)

    assert float(raw_head.max()) > 24.0, 'softplus is unbounded above, so the raw head must be able to exceed 24'
    assert float(members.max()) <= 24.0 + 1e-5, 'mc_forward must clamp each member'


def test_the_differentiable_prediction_is_not_clamped(make_mc):
    """Deliberate: a clamp zeroes the gradient of exactly the over-predicting cells the CRPS spread term must move, so
    the training path must NOT clamp even though the prediction path does."""
    module = make_mc()
    head_output = torch.full((2, 4, 4), 100.0, requires_grad=True)
    differentiable = module._to_prediction_differentiable(head_output)
    assert float(differentiable.max()) > 24.0
    differentiable.sum().backward()
    assert head_output.grad is not None and float(head_output.grad.abs().min()) > 0.0


# =====================================================================================================================
# NEW: the warm start — from_upstream is the whole mechanism
# =====================================================================================================================
@pytest.fixture
def upstream_checkpoint(unet_trial, normalization, target_stats, save_checkpoint):
    module = DeterministicUnetModule(unet_trial(), 5, target_stats(), normalization)
    return save_checkpoint(module, 'upstream.ckpt')


def test_from_upstream_loads_the_weights_and_flags_the_warm_start(
        upstream_checkpoint, mc_trial, normalization, target_stats):
    warm = MCDropoutModule.from_upstream(upstream_checkpoint, mc_trial(), 5, target_stats(), normalization)
    assert warm.warm_started is True


def test_from_upstream_takes_its_ARCHITECTURE_from_the_checkpoint(
        upstream_checkpoint, mc_trial, normalization, target_stats):
    """The override and the compatibility check are two halves of one thing. Without the override the factory builds
    from the SAMPLED architecture, and the check then turns every warm-start trial into a hard failure whenever the
    sampled base_channels/depth/activation differs from the upstream's — 26 times in 27 on a 3x3x3 grid.

    This is also what makes ``apply_constraints``' "the unet block is ignored" log line TRUE.
    """
    divergent = mc_trial()
    divergent['unet'] = {**divergent['unet'], 'base_channels': 32, 'depth': 3}

    warm = MCDropoutModule.from_upstream(upstream_checkpoint, divergent, 5, target_stats(), normalization)
    assert warm.hparams['trial']['unet']['base_channels'] == 8, 'the checkpoint architecture must win'
    assert warm.hparams['trial']['unet']['depth'] == 2


def test_an_architecture_difference_is_OVERRIDDEN_rather_than_rejected(
        upstream_checkpoint, mc_trial, normalization, target_stats, caplog):
    """Deliberate, and the opposite of what an early draft of the plan specified: the sampled architecture is
    DISCARDED in favour of the checkpoint's, not treated as an error. Rejecting it would fail 26 warm-start trials in
    27 on a 3x3x3 sampled grid, since the sweep has no reason to resample the upstream's exact architecture.

    The discard is logged so a trials table showing varying ``unet.*`` columns against identical fitted architectures
    is not mystifying.
    """
    import logging

    divergent = mc_trial()
    divergent['unet'] = {**divergent['unet'], 'base_channels': 32}

    with caplog.at_level(logging.INFO):
        warm = MCDropoutModule.from_upstream(upstream_checkpoint, divergent, 5, target_stats(), normalization)

    assert warm.hparams['trial']['unet']['base_channels'] == 8
    assert 'base_channels' in caplog.text, 'the discarded sampled value must be reported'


def test_a_channel_count_mismatch_raises_naming_in_channels(
        upstream_checkpoint, mc_trial, normalization, target_stats):
    """``in_channels`` comes from the DATA rather than the trial, so it cannot be overridden — a mismatch means the two
    preparations differ in their feature list, aggregation or residual flag, and the weights genuinely do not fit."""
    with pytest.raises(ValueError, match='in_channels'):
        MCDropoutModule.from_upstream(upstream_checkpoint, mc_trial(), 6, target_stats(),
                                      {'mean': [0.0] * 6, 'std': [1.0] * 6})


def test_a_mode_mismatch_raises_naming_mode(upstream_checkpoint, mc_trial, normalization, target_stats):
    """The head means a different thing in each task — bounded hours versus an occurrence logit — so its weights do
    not transfer even though the shapes match exactly. Shape-compatible and semantically wrong is precisely the case a
    ``load_state_dict`` cannot catch."""
    with pytest.raises(ValueError, match='mode'):
        MCDropoutModule.from_upstream(upstream_checkpoint, mc_trial(), 5, target_stats(mode='hourly'), normalization)


def test_a_checkpoint_without_hyperparameters_raises(tmp_path, mc_trial, normalization, target_stats):
    """Without the recorded trial the architecture cannot be read, so a warm start would silently build a DIFFERENT
    network and then fail the strict state-dict load — or worse, match by accident."""
    import os

    path = os.path.join(str(tmp_path), 'bare.ckpt')
    torch.save({'state_dict': {}}, path)
    with pytest.raises(ValueError, match='hyper_parameters'):
        MCDropoutModule.from_upstream(path, mc_trial(), 5, target_stats(), normalization)


def test_the_mc_dropout_rate_is_ours_not_the_upstreams(upstream_checkpoint, mc_trial, normalization, target_stats):
    """The upstream is deterministic (dropout 0.0), and the MC rate is exactly what this family's finetuning phase has
    to calibrate — so ``dropout`` is the one architecture field the override must NOT take from the checkpoint."""
    warm = MCDropoutModule.from_upstream(upstream_checkpoint, mc_trial(dropout_p=0.3), 5, target_stats(),
                                         normalization)
    assert warm.hparams['trial']['unet']['dropout'] == pytest.approx(0.3)


def test_the_architecture_key_list_covers_every_unet_field_that_matters():
    """The tuple drives the override log. A key missing from it means a discarded sampled value is never reported,
    which is the difference between a puzzling trials table and a readable one."""
    assert set(WARM_START_ARCHITECTURE_KEYS) == {
        'base_channels', 'depth', 'activation', 'normalization', 'bottleneck_attention',
        'kernel_size', 'blocks_per_level', 'upsampling',
    }
    assert 'dropout' not in WARM_START_ARCHITECTURE_KEYS, 'dropout is ours, not compared against the upstream'


# =====================================================================================================================
# NEW: training_phases — the part with a silent failure mode
# =====================================================================================================================
def test_a_warm_started_module_runs_the_finetune_phase_alone(
        upstream_checkpoint, mc_trial, normalization, target_stats):
    """Phase 1 already happened — it IS the upstream checkpoint. ``finetuning.enabled`` only makes phase 2 LEGAL
    (``set_phase`` raises otherwise); it does not skip phase 1, which is why the module has to be TOLD it was warm
    started. After ``apply_constraints`` has run, a warm-started trial and a from-scratch trial that sampled
    ``enabled = true`` are byte-identical dicts, so nothing is derivable from the trial."""
    warm = MCDropoutModule.from_upstream(upstream_checkpoint, mc_trial(), 5, target_stats(), normalization)
    assert warm.training_phases() == ('finetune',)


def test_from_scratch_runs_train_then_finetune(make_mc):
    assert make_mc(finetuning=True).training_phases()[:2] == ('train', 'finetune')


def test_finetuning_disabled_runs_the_train_phase_alone(make_mc):
    assert make_mc(finetuning=False).training_phases() == ('train',)


def test_warm_started_is_not_reachable_through_the_constructor(make_mc):
    """Why ``from_upstream`` is a CLASSMETHOD rather than an ``__init__`` flag: with a kwarg, setting the flag and
    loading the weights are independent, so a factory could set one without the other — yielding a RANDOMLY
    INITIALISED net that skips the only phase which would have trained it. It fits nothing, writes a checkpoint,
    scores finitely-but-badly, and the sweep merely ranks it low. No error, no warning.

    In the classmethod ``warm_started = True`` is unreachable without the ``load_state_dict`` above it.
    """
    import inspect

    assert 'warm_started' not in inspect.signature(MCDropoutModule.__init__).parameters
    assert make_mc().warm_started is False


def test_requesting_the_finetune_phase_without_enabling_it_raises(make_mc):
    module = make_mc(finetuning=False)
    with pytest.raises(ValueError):
        module.set_phase('finetune')


def test_the_family_supports_the_ensemble_loss(make_mc):
    """MC-dropout is the family that CAN train ``crps_binary`` / the CRPS finetune loss, because a forward pass
    yields real members. The deterministic family cannot, and the flag is what enforces that split."""
    assert MCDropoutModule.SUPPORTS_ENSEMBLE_LOSS is True


def test_the_checkpoint_marker_is_the_family_name(make_mc):
    checkpoint = {}
    make_mc().on_save_checkpoint(checkpoint)
    assert checkpoint['module_class'] == 'mc_dropout'


def test_hourly_mode_returns_a_probability(make_mc, batch):
    module = make_mc(mode='hourly', loss={'name': 'brier'})
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0, ensemble_size=MEMBERS)
    assert output['probability'] is not None
    assert float(output['prediction'].min()) >= 0.0 and float(output['prediction'].max()) <= 1.0


def test_the_finetune_phase_uses_a_reduced_learning_rate(make_mc):
    """``finetune_lr_factor`` exists so the CRPS phase refines rather than undoing the point-fit phase."""
    module = make_mc()
    module.set_phase('train')
    base = module._learning_rate()
    module.set_phase('finetune')
    assert module._learning_rate() < base
