"""Tests for src/utils/modeling/diffusion_module.py — flow matching, optionally residual.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_residual_diagnostics.py`` (the unclamped-residual plumbing
and the ``trainer.predict`` wiring), adapted to the Step 2 trial vocabulary: ``transformer:`` became ``flow:``,
``depth`` became ``n_blocks``, ``num_sampling_steps`` became ``n_steps``, ``optimizer.learning_rate`` became ``lr``, and
both ``flow.log_warp`` and ``target_stats['diffusion_transform']`` are gone with the transform — generation happens in
the RAW target space, so there is no back-transform anywhere.
"""
import inspect

import numpy as np
import pytest
import torch

from src.utils.modeling.diffusion_module import CHECKPOINT_MARKER, PHASES, DiffusionModule

MEMBERS = 4
OCCURRENCE_EVENT = (0.0, True)
COND_CHANNELS = 5


@pytest.fixture
def make_diffusion(diffusion_trial, target_stats):
    def build(residual=False, mode='daily', channels=None, **overrides):
        channels = channels if channels is not None else (COND_CHANNELS + 1 if residual else COND_CHANNELS)
        normalization = {'mean': [0.0] * channels, 'std': [1.0] * channels}
        module = DiffusionModule(diffusion_trial(**overrides), channels,
                                 target_stats(mode=mode, residual_target=residual), normalization).eval()
        module.eval_ensemble_size = MEMBERS
        module.eval_occurrence_event = OCCURRENCE_EVENT
        module.eval_sampling_steps = 2
        return module
    return build


# =====================================================================================================================
# residual_target comes from the DATA, never from a hyperparameter
# =====================================================================================================================
def test_residual_mode_is_read_from_the_prepared_data(make_diffusion):
    """It is a property of the prepared DIRECTORY: the mode exists only if ``prepare_regression`` was given
    ``upstream-model-path``, which is what materialises the upstream channel and the third batch item. Sampling it per
    trial would make half of every sweep ask for a mode the data cannot provide."""
    assert make_diffusion(residual=True).residual_target is True
    assert make_diffusion(residual=False).residual_target is False


def test_the_constructor_takes_no_residual_target_argument():
    assert 'residual_target' not in inspect.signature(DiffusionModule.__init__).parameters


def test_the_search_space_carries_no_residual_target_key(search_spaces):
    assert 'residual_target' not in search_spaces['diffusion']


def test_residual_mode_unpacks_three_batch_items(make_diffusion, batch):
    module = make_diffusion(residual=True)
    x, y, upstream = batch(channels=COND_CHANNELS + 1, residual=True)
    assert module._unpack((x, y, upstream))[2] is not None


def test_the_generation_target_is_the_discrepancy_in_residual_mode(make_diffusion, batch):
    """The model learns ``y - upstream``, not ``y`` — that is the whole of "residual"."""
    module = make_diffusion(residual=True)
    _, y, upstream = batch(channels=COND_CHANNELS + 1, residual=True)
    assert torch.equal(module._generation_target(y, upstream), y - upstream)


def test_the_generation_target_is_the_target_itself_otherwise(make_diffusion, batch):
    module = make_diffusion(residual=False)
    _, y = batch()
    assert torch.equal(module._generation_target(y, None), y)


# =====================================================================================================================
# The reconstruction clamp — at BOTH ends, and per draw
# =====================================================================================================================
def test_the_residual_reconstruction_is_clamped_at_both_ends(make_diffusion):
    """``clamp(upstream + residual, 0, max_hours)``. A's version clamped only the floor, which was right for an
    unbounded count target and wrong for a bounded 0-24 one: an upstream near 24 plus a positive correction would
    otherwise report more lightning-hours than a day has."""
    module = make_diffusion(residual=True)
    upstream = torch.full((1, 8, 8), 23.0)
    huge = torch.full((1, 8, 8), 50.0)
    negative = torch.full((1, 8, 8), -50.0)

    assert float((upstream + huge).clamp(0.0, module.max_hours).max()) == pytest.approx(24.0)
    assert float((upstream + negative).clamp(0.0, module.max_hours).min()) == pytest.approx(0.0)


def test_every_ensemble_member_is_in_range_not_just_the_mean(make_diffusion, batch):
    """Clamping the mean would let members above the ceiling pull it up, and the members are what the spread metrics
    read — so the clamp has to be per draw."""
    module = make_diffusion()
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    members = output['ensemble_members']
    assert float(members.min()) >= 0.0
    assert float(members.max()) <= module.max_hours + 1e-4


# =====================================================================================================================
# The unclamped residual survives for the diagnostics  (ported)
# =====================================================================================================================
def test_the_unclamped_discrepancy_is_exposed_for_the_diagnostics(make_diffusion, batch):
    """``residual_diagnostics`` needs the RAW predicted discrepancy: the clamped reconstruction has already lost the
    over/under-correction information the surprise panes categorise."""
    module = make_diffusion(residual=True)
    module.eval_return_residual = True
    x, y, upstream = batch(channels=COND_CHANNELS + 1, residual=True)

    with torch.no_grad():
        output = module.predict_step((x, y, upstream), 0)

    assert output['ensemble_residual_members'].shape == (x.shape[0], MEMBERS, x.shape[2], x.shape[3])
    assert output['upstream'].shape == (x.shape[0], x.shape[2], x.shape[3])
    # the clamped members must equal clamp(U + r), i.e. r really is the unclamped discrepancy
    reconstructed = (upstream[:, None] + output['ensemble_residual_members']).clamp(0.0, module.max_hours)
    assert torch.allclose(output['ensemble_members'], reconstructed, atol=1e-5)


def test_the_residual_keys_are_absent_when_not_requested(make_diffusion, batch):
    module = make_diffusion(residual=True)                       # eval_return_residual stays False
    x, y, upstream = batch(channels=COND_CHANNELS + 1, residual=True)
    with torch.no_grad():
        output = module.predict_step((x, y, upstream), 0)
    assert 'ensemble_residual_members' not in output and 'upstream' not in output


def test_requesting_the_residual_on_a_full_target_run_is_ignored(make_diffusion, batch):
    """``eval_return_residual and self.residual_target``: there is no discrepancy to report when the model predicts the
    target directly, so the flag is inert rather than producing a meaningless array."""
    module = make_diffusion(residual=False)
    module.eval_return_residual = True
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert 'ensemble_residual_members' not in output


# =====================================================================================================================
# Trial-shape translation and the sweep contract
# =====================================================================================================================
# =====================================================================================================================
# The objective is the FLOW loss, and the trial's loss: block is deliberately not read
# =====================================================================================================================
@pytest.mark.source_invariant
@pytest.mark.parametrize('builder', ['build_regression_loss', 'build_binary_loss'])
def test_the_module_never_reaches_for_a_POINTWISE_loss_builder(builder):
    """This family's objective is the flow-matching velocity MSE — a loss on the velocity field, not on the target. The
    ``loss:`` block in its search space exists so the three spaces stay structurally comparable, and is NOT the training
    objective. A pointwise builder appearing here would mean the flow loss had been quietly supplemented."""
    from src.utils.modeling import diffusion_module

    source = open(diffusion_module.__file__).read()
    assert builder not in source


@pytest.mark.source_invariant
def test_the_search_space_SAYS_its_loss_block_is_not_the_objective(repo_root):
    """Because the key is there and reads like every other family's. Without the warning next to it, the obvious
    conclusion from the YAML is that ``loss.name`` selects this family's objective too."""
    import os

    text = open(os.path.join(repo_root, 'config/diffusion/search_space.yaml')).read()
    assert 'NOT THE TRAINING OBJECTIVE' in text


def test_the_training_step_returns_a_finite_loss_carrying_a_gradient(make_diffusion, batch):
    """One phase, one objective. The velocity MSE is computed at a randomly sampled time per item, so this also exercises
    ``_sample_time`` and ``_standardize_conditioning``."""
    module = make_diffusion()
    module.train()
    loss = module.training_step(batch(), 0)
    assert torch.isfinite(loss) and loss.requires_grad


def test_the_flow_loss_is_DETERMINISTIC_under_a_seeded_generator(make_diffusion, batch):
    """The loss draws both the noise and the time, so without seeding two evaluations of the same batch disagree — which
    would make the validation curve unreadable and the monitor's "best" epoch arbitrary."""
    module = make_diffusion()
    module.train()
    features, target = batch()

    conditioning = module._standardize_conditioning(features)
    first = module._flow_loss(target, conditioning, generator=torch.Generator().manual_seed(0))
    again = module._flow_loss(target, conditioning, generator=torch.Generator().manual_seed(0))
    assert float(first.detach()) == pytest.approx(float(again.detach()), abs=1e-9)


# =====================================================================================================================
# The validation epoch: the flow loss is monitored, the composite is what the sweep ranks on
# =====================================================================================================================
@pytest.fixture
def validated_diffusion(make_diffusion):
    def run(batches=2, scoring=True):
        module = make_diffusion()
        module.valid_climatology_cond_mae = 2.0
        if scoring:
            module.prepare_full_validation()
        module.on_validation_epoch_start()
        generator = torch.Generator().manual_seed(0)
        for index in range(batches):
            features = torch.randn(2, COND_CHANNELS, 16, 16, generator=generator)
            target = torch.randint(0, 25, (2, 16, 16), generator=generator).float()
            module.validation_step((features, target), index)
        module.on_validation_epoch_end()
        return module
    return run


def test_the_flow_loss_is_ALWAYS_reported(validated_diffusion):
    """Every epoch, scoring pass or not: it is the monitored metric, so a missing value means the checkpoint callback has
    nothing to compare."""
    assert 'valid_flow_loss' in validated_diffusion(scoring=False).last_val_metrics


def test_the_composite_is_reported_on_a_SCORING_pass(validated_diffusion):
    """Generating an ensemble per validation batch is expensive, so the target-space composite is computed only on
    scoring epochs. It must still land under the selection metric's name — the sweep ranks on the composite and never on
    ``valid_flow_loss``, which is a velocity-space quantity and not comparable across architectures."""
    module = validated_diffusion(scoring=True)
    assert module.selection_metric in module.last_val_metrics
    assert 'valid_tail_score' not in module.last_val_metrics


def test_prepare_full_validation_FORCES_the_scoring_pass(make_diffusion):
    """The hook ``run_sweep`` calls before the final validation of a trial, so the recorded composite is a real scoring
    pass rather than whatever the epoch schedule happened to land on."""
    module = make_diffusion()
    module.prepare_full_validation()
    assert module._force_full_validation


def test_the_validation_and_evaluation_seeds_DIFFER(make_diffusion):
    """Both are fixed, deliberately, and they must not be the same fixed value: reusing the validation seed at evaluation
    would score the test split on the same noise draws the model was selected under."""
    module = make_diffusion()
    assert module.valid_seed != module.eval_ensemble_seed


def test_the_ensemble_is_REPRODUCIBLE_at_a_fixed_seed(make_diffusion, batch):
    """Two evaluation runs of one checkpoint must report the same CRPS. Each ODE draw starts from fresh noise, so without
    a fixed seed the number moves run to run."""
    module = make_diffusion()
    one_batch = batch()
    first = module.predict_step(one_batch, 0)
    repeated = module.predict_step(one_batch, 0)
    assert torch.allclose(first['prediction'], repeated['prediction'], atol=1e-6)


def test_an_unknown_phase_raises(make_diffusion):
    module = make_diffusion()
    with pytest.raises(ValueError):
        module.set_phase('finetune')


def test_n_blocks_is_translated_to_the_networks_depth():
    """``flow.n_blocks`` is named that way because ``depth`` already means the down/upsampling level count in the other
    families' ``unet`` block. The module translates rather than the config compromising."""
    translated = DiffusionModule._net_config({'n_steps': 4, 'hidden_dim': 128, 'n_blocks': 3, 'num_heads': 4,
                                              'patch_size': 2})
    assert translated['depth'] == 3
    assert translated['num_heads'] == 4
    assert 'n_blocks' not in translated


def test_the_family_has_a_single_training_phase(make_diffusion):
    assert make_diffusion().training_phases() == PHASES
    assert len(PHASES) == 1


def test_it_monitors_the_flow_loss_but_ranks_on_the_target_space_composite(make_diffusion):
    """Deliberate divergence: the checkpoint monitor is the validation flow loss, while the SWEEP ranks trials on the
    target-space composite. ``_fit_trial`` only attaches the optuna pruning callback when the monitor equals the prune
    metric, so this family simply never prunes — by design, because the flow loss lives in the generation space and
    says nothing about occurrence skill or structure fidelity."""
    module = make_diffusion()
    assert module.monitor_metric == 'valid_flow_loss'
    assert module.monitor_mode == 'min'


def test_the_checkpoint_marker_is_the_family_name(make_diffusion):
    checkpoint = {}
    make_diffusion().on_save_checkpoint(checkpoint)
    assert checkpoint['module_class'] == CHECKPOINT_MARKER == 'diffusion'


def test_a_single_draw_run_returns_the_deterministic_dict(make_diffusion, batch):
    module = make_diffusion()
    module.eval_ensemble_size = 1
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert set(output) == {'prediction', 'probability', 'observation'}


def test_this_family_never_reports_a_probability(make_diffusion, batch):
    """There is no occurrence-probability head, so ``probability`` is None in both modes — the family's hourly story is
    that each ODE draw is a member, not that it emits a calibrated probability."""
    module = make_diffusion()
    x, y = batch()
    with torch.no_grad():
        assert module.predict_step((x, y), 0)['probability'] is None


def test_a_residual_channel_count_mismatch_raises_naming_the_upstream_channel(diffusion_trial, target_stats):
    """The upstream prediction is appended as the LAST conditioning channel, so residual data and a non-residual
    normalization disagree by exactly one. Naming the channel is what makes the cause obvious."""
    with pytest.raises(ValueError):
        DiffusionModule(diffusion_trial(), COND_CHANNELS + 1,
                        target_stats(residual_target=True),
                        {'mean': [0.0] * COND_CHANNELS, 'std': [1.0] * COND_CHANNELS})


# (the straight-path interpolant itself is a module-level function in diffusion.py — see diffusion_test.py)


def test_trainer_predict_drives_the_module_end_to_end(make_diffusion):
    """The eval stage's actual call path: ``trainer.predict(module, loader)`` then concatenate the per-batch dicts."""
    import lightning
    from torch.utils.data import DataLoader, TensorDataset

    module = make_diffusion(residual=True)
    module.eval_return_residual = True
    n, height, width = 4, 16, 24
    x = torch.randn(n, COND_CHANNELS + 1, height, width)
    y = torch.clamp(torch.randn(n, height, width) * 3.0, min=0.0)
    upstream = torch.clamp(torch.randn(n, height, width) + 1.0, min=0.0)
    loader = DataLoader(TensorDataset(x, y, upstream), batch_size=2, shuffle=False)

    trainer = lightning.Trainer(accelerator='cpu', devices=1, logger=False, enable_progress_bar=False)
    outputs = trainer.predict(module, loader)

    residual_members = torch.cat([b['ensemble_residual_members'] for b in outputs]).numpy()
    assert residual_members.shape == (n, MEMBERS, height, width)
