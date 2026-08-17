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


def test_the_warm_start_works_at_the_SHIPPED_blocks_per_level(
        unet_trial, mc_trial, normalization, target_stats, save_checkpoint):
    """🐛 The Step 4 block 4e gate failure, pinned at the level it actually happened.

    Every test above builds its upstream from the shared UNET fixture, whose ``blocks_per_level`` is **1**. Every
    shipped search space FIXES it at **2**. At 1 the conditional ``Dropout2d`` landed after the last layer and shifted
    nothing, so a dropout-0 checkpoint loaded into a dropout-0.2 net and the whole suite was green — while on real data
    the warm start failed on its first trial, every time, for every family combination. The fix was to emit the dropout
    layer unconditionally (`unet.ConvBlock`); this test is what would have caught it.
    """
    deterministic = DeterministicUnetModule(unet_trial(unet={**unet_trial()['unet'], 'blocks_per_level': 2}),
                                            5, target_stats(), normalization)
    checkpoint = save_checkpoint(deterministic, 'upstream_blocks2.ckpt')

    warm = MCDropoutModule.from_upstream(checkpoint, mc_trial(dropout_p=0.2), 5, target_stats(), normalization)

    assert warm.warm_started is True
    assert warm.hparams['trial']['unet']['blocks_per_level'] == 2, 'the architecture comes from the checkpoint'
    # the weights are the upstream's, not a fresh initialisation
    upstream_weights = deterministic.net.state_dict()
    for key, tensor in warm.net.state_dict().items():
        assert torch.allclose(tensor, upstream_weights[key]), key


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


# =====================================================================================================================
# training_step, per phase — the two-term finetune loss is what this family exists for
# =====================================================================================================================
def test_the_train_phase_returns_a_finite_loss_carrying_a_gradient(make_mc, batch):
    """Phase 1 is an ordinary pointwise fit, identical to the deterministic family's."""
    module = make_mc()
    module.train()
    module.set_phase('train')
    loss = module.training_step(batch(), 0)
    assert torch.isfinite(loss) and loss.requires_grad


def test_the_finetune_phase_returns_a_finite_loss_carrying_a_gradient(make_mc, batch):
    """Phase 2 draws an MC ensemble inside the training step, so it is the one place a live gradient flows through
    ``_to_prediction_differentiable`` — the unclamped path. A clamp there would zero the gradient for exactly the
    over-predicting members the ensemble term needs to move."""
    module = make_mc()
    module.train()
    module.set_phase('finetune')
    loss = module.training_step(batch(), 0)
    assert torch.isfinite(loss) and loss.requires_grad


def test_the_finetune_loss_ADDS_the_ensemble_term_to_the_pointwise_one(make_mc, batch):
    """``loss_reg + finetune_loss_weight * loss_crps`` — the expression that settled the ``build_ensemble_loss``
    question: the two builders are used TOGETHER, so folding the ensemble loss into ``build_regression_loss`` was never
    possible. If the ensemble term were dropped, phase 2 would be phase 1 at a lower learning rate: it would fit, write a
    checkpoint, and score plausibly, with no calibration of the spread at all.
    """
    module = make_mc()
    module.train()
    torch.manual_seed(0)
    one_batch = batch()

    module.set_phase('train')
    pointwise = float(module.training_step(one_batch, 0))
    module.set_phase('finetune')
    combined = float(module.training_step(one_batch, 0))

    assert abs(combined - pointwise) > 1e-6, 'the ensemble term contributed nothing'


def test_only_the_MC_family_reduces_its_learning_rate_for_a_second_phase(make_mc, mc_trial, normalization,
                                                                        target_stats):
    """``finetune_lr_factor`` exists because phase 2 starts from converged weights. The deterministic family has one
    fitting phase and therefore no such reduction — checked here so the factor cannot leak into the shared base."""
    module = make_mc()
    module.set_phase('train')
    train_lr = module._learning_rate()

    deterministic = DeterministicUnetModule(mc_trial(), 5, target_stats(), normalization)
    deterministic.set_phase('train')
    assert deterministic._learning_rate() == pytest.approx(train_lr)


def test_the_composite_is_logged_under_the_selection_metric_name(make_mc):
    """The same monitor/prune agreement the deterministic family needs, driven through this family's own validation
    hooks — which draw an ensemble per batch rather than a single pass."""
    module = make_mc()
    module.valid_climatology_cond_mae = 2.0
    module.on_validation_epoch_start()
    generator = torch.Generator().manual_seed(0)
    for index in range(2):
        features = torch.randn(2, 5, 16, 16, generator=generator)
        target = torch.randint(0, 25, (2, 16, 16), generator=generator).float()
        module.validation_step((features, target), index)
    module.on_validation_epoch_end()

    assert module.selection_metric in module.last_val_metrics, sorted(module.last_val_metrics)


def test_the_hourly_selection_metric_is_the_CLASSIFICATION_composite(make_mc):
    """The mode drives it here exactly as in the deterministic family — the composite is not a per-family choice."""
    module = make_mc(mode='hourly', loss={'name': 'brier'})
    assert module.selection_metric == 'valid_classification_score'


@pytest.mark.source_invariant
def test_the_target_space_prediction_is_the_MC_MEAN_not_a_single_pass(make_mc):
    """The override that makes the shared validation accumulation correct for this family. Inheriting the base's single
    forward pass would score MC-dropout's validation on one dropout draw — noisier than the deterministic baseline and
    not what the family reports at test time."""
    import inspect

    source = inspect.getsource(MCDropoutModule._target_space_prediction)
    assert 'mc_forward' in source


@pytest.mark.source_invariant
def test_the_shared_machinery_is_NOT_redefined_by_either_family():
    """Block 3d extracted ``UnetModuleBase`` so the phase machinery, the loss dispatch and the validation accumulation
    have ONE implementation. A family redefining one of them would drift silently — the base's version would still exist
    and still look authoritative.

    ⚠️ ``_check_phase_available`` and ``_learning_rate`` are deliberately NOT in this list: MC-dropout overrides both, to
    reject the finetune phase when ``finetuning.enabled`` is false and to apply ``finetune_lr_factor``. Overriding a hook
    is the mechanism; re-implementing the shared body is the defect.
    """
    import ast
    import inspect

    from src.utils.modeling import deterministic_module, mc_dropout_module

    shared = ('forward', '_head_output', '_to_prediction', 'on_validation_epoch_end',
              '_validation_reg_calibration', 'set_phase', 'configure_optimizers')
    for module in (deterministic_module, mc_dropout_module):
        tree = ast.parse(inspect.getsource(module))
        defined = {node.name for klass in tree.body if isinstance(klass, ast.ClassDef)
                   for node in klass.body if isinstance(node, ast.FunctionDef)}
        for name in shared:
            assert name not in defined, f'{module.__name__} redefines the shared {name}'


def test_a_warm_started_module_writes_ITS_OWN_marker_not_the_upstreams(
        upstream_checkpoint, mc_trial, normalization, target_stats):
    """It loaded a ``deterministic_unet`` checkpoint's weights, so the marker is the one thing it must NOT inherit — the
    registry would then load the finetuned MC-dropout weights as a deterministic module and report no ensemble metrics
    at all."""
    module = MCDropoutModule.from_upstream(upstream_checkpoint, mc_trial(), 5, target_stats(), normalization)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint['module_class'] == 'mc_dropout'


def test_the_calibration_phase_is_APPENDED_after_the_finetune_phase(make_mc):
    """Three phases, in order. A calibration phase inserted before the finetune one would fit the calibrator against
    weights the finetune phase then moves."""
    module = make_mc(calibration={'occurrence': 'none',
                                  'regression': {'structure': 'power_law', 'objective': 'pointwise',
                                                 'num_sigmoids': 4, 'huber_delta': 1.0}})
    assert module.training_phases() == ('train', 'finetune', 'regression_calibration')


def test_the_finetune_phase_belongs_to_THIS_family_alone(make_mc):
    assert 'finetune' in MCDropoutModule.PHASES
    assert 'finetune' not in DeterministicUnetModule.PHASES


def test_the_two_families_are_SIBLINGS_not_parent_and_child():
    """Both derive from the shared base, and neither from the other. If MC-dropout inherited from the deterministic
    family it would inherit its POINT ``predict_step`` — satisfying the ensemble contract while reporting no members."""
    from src.utils.modeling.unet_module_base import UnetModuleBase

    assert issubclass(MCDropoutModule, UnetModuleBase)
    assert issubclass(DeterministicUnetModule, UnetModuleBase)
    assert not issubclass(MCDropoutModule, DeterministicUnetModule)
    assert not issubclass(DeterministicUnetModule, MCDropoutModule)


def test_the_finetune_phase_uses_a_reduced_learning_rate(make_mc):
    """``finetune_lr_factor`` exists so the CRPS phase refines rather than undoing the point-fit phase."""
    module = make_mc()
    module.set_phase('train')
    base = module._learning_rate()
    module.set_phase('finetune')
    assert module._learning_rate() < base


# =====================================================================================================================
# Block 5c — the family's own phase guard
# =====================================================================================================================
def test_the_FINETUNE_phase_requires_finetuning_to_be_enabled(mc_trial, normalization, target_stats):
    """``finetuning.enabled`` is what builds the ensemble loss; without it ``self.finetune_loss`` is None and the phase
    would run a second fitting pass with no objective. The message names ``apply_constraints`` because that is what
    forces the flag true on a warm start — a warm-started run has no train phase to fall back on."""
    module = MCDropoutModule(mc_trial(finetuning=False), 5, target_stats(), normalization)

    assert 'finetune' in module.PHASES, 'the phase must be NAMED, or this tests nothing'
    with pytest.raises(ValueError, match='finetuning.enabled'):
        module._check_phase_available('finetune')


def test_the_finetune_phase_IS_available_once_enabled(mc_trial, normalization, target_stats):
    module = MCDropoutModule(mc_trial(finetuning=True), 5, target_stats(), normalization)
    assert module._check_phase_available('finetune') is None
    assert 'finetune' in module.training_phases()


def test_the_guard_still_DELEGATES_to_the_shared_calibration_checks(mc_trial, normalization, target_stats):
    """It is an override, not a replacement — the ``super()`` call is what keeps the two calibration phases guarded for
    this family too. Dropping it would let an MC-dropout trial enter a calibration phase whose layer does not exist."""
    module = MCDropoutModule(mc_trial(finetuning=True), 5, target_stats(mode='daily'), normalization)
    with pytest.raises(ValueError, match='monotone'):
        module._check_phase_available('regression_calibration')
