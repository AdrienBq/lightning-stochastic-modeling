"""LightningModule for the MC-dropout U-net — the same network as the deterministic family, sampled stochastically.

Everything about the model itself is inherited from :class:`~src.utils.modeling.unet_module_base.UnetModuleBase`;
read that docstring first. This file adds only what makes the family stochastic.

TWO WAYS TO GET HERE, decided by whether the ``tune`` stage was given an ``upstream-model-path``:

  FROM SCRATCH (``UPSTREAM_MODEL`` unset) — the full two-phase fit:
    phase 1 ``train``    : a single deterministic forward pass, the ``loss:`` section; dropout is a plain regularizer.
    phase 2 ``finetune`` : MC-dropout active, and the probabilistic ``finetuning.loss`` is ADDED to the pointwise
                           loss so the ensemble DISTRIBUTION is calibrated, at a reduced learning rate.

  WARM START (``UPSTREAM_MODEL`` set) — phase 1 is REPLACED by the deterministic U-net:
    :meth:`MCDropoutModule.from_upstream` loads that checkpoint's weights, and ONLY ``finetune`` runs. This is a
    WEIGHT initialization, not a conditioning channel — the crucial difference from the diffusion family, whose
    residual mode needs the upstream PREDICTION materialised by ``prepare_regression``.

⚠️ ``from_upstream`` IS THE WHOLE WARM-START MECHANISM, and it is a classmethod on purpose. It sets
``warm_started`` on the same object it just loaded weights into, so the flag is unreachable without them. Were it a
constructor flag, a caller could set it while skipping the load — yielding a randomly-initialised net that skips the
only phase which would have trained it. That fits nothing, writes a checkpoint, scores badly-but-finitely, and the
sweep merely ranks it low: no error, no warning.

WHY GROUP NORM IS MANDATORY (fixed in the search space, not searched). MC inference re-enables dropout while the
rest of the net stays in ``eval()``. Batch norm keeps running mean/var statistics that are updated in ``train()``
and frozen in ``eval()``, so putting the model back into ``train()`` to resample dropout would also unfreeze those
statistics and shift them at test time. Group norm is invariant to the switch, which is what makes
:func:`enable_mc_dropout` safe — and it is also why every deterministic checkpoint is a valid upstream.

⚠️ ``dropout_p`` IS INJECTED into ``unet.dropout``. The search space carries the MC rate as a top-level
``dropout_p`` while ``UNetBackbone`` reads ``unet.dropout`` (a ``0.0`` placeholder there). Skip the injection and
every MC pass is deterministic, every member identical, the spread zero, and ``spread_skill_sums`` returns NaN
through its ``ddof=1`` — a silent failure with no exception anywhere.
"""
import logging
from typing import Optional, Tuple

import torch

from src.utils.modeling.losses import build_ensemble_loss
from src.utils.modeling.unet import enable_mc_dropout
from src.utils.modeling.unet_module_base import UnetModuleBase

logger = logging.getLogger(__name__)

# The `unet` fields whose SAMPLED value `from_upstream` reports as discarded when it differs from the upstream
# checkpoint's. Nothing here is a compatibility requirement: the checkpoint's architecture always wins, and this tuple
# drives only the log line that says so.
#
# ⚠️ Do NOT turn a mismatch here into an error. The sweep samples base_channels x depth x activation independently of
# the frozen upstream, so exactly one of ~27 combinations matches it — rejecting the rest would fail 26 warm-start
# trials in 27 for a reason unrelated to the model. The override is also what makes `search.apply_constraints`'s
# "the sampled unet block is ignored" log line true.
#
# What DOES raise, because it genuinely cannot be overridden (see `from_upstream`): `in_channels` (it comes from the
# DATA, not the trial, so the weights really do not fit), `mode` (the shapes match but the head means a different
# thing in each task, which is precisely what `load_state_dict` cannot catch), and a checkpoint with no recorded
# `hyper_parameters.trial` (the architecture cannot be read at all).
#
# `dropout` is deliberately absent: the MC rate is OURS, not the upstream's — the upstream is deterministic, and
# calibrating that rate is what this family's finetuning phase exists to do.
WARM_START_ARCHITECTURE_KEYS = (
    'base_channels', 'depth', 'kernel_size', 'blocks_per_level', 'upsampling', 'normalization', 'activation',
    'bottleneck_attention'
)


class MCDropoutModule(UnetModuleBase):
    """MC-dropout U-net for one sampled trial (see the module docstring for the two entry points).

    Constructor arguments are :class:`UnetModuleBase`'s. ``dropout_p`` is read from the trial's top level and
    injected into the ``unet`` section before the network is built.

    Raises:
        ValueError: On ``dropout_p <= 0`` (a zero-dropout model cannot produce an ensemble at all), in addition to
            everything the base raises.
    """

    CHECKPOINT_MARKER = 'mc_dropout'
    PHASES = ('train', 'finetune') + UnetModuleBase.PHASES[1:]
    #: MC-dropout has a genuine sample axis, so an ensemble loss (``crps_binary``) is admissible here.
    SUPPORTS_ENSEMBLE_LOSS = True

    def __init__(self, trial: dict, in_channels: int, target_stats: dict, normalization: dict):
        dropout_p = float(trial.get('dropout_p', 0.0))
        if dropout_p <= 0.0:
            raise ValueError(
                f'dropout_p is {dropout_p}, but MC inference needs stochastic units: at 0 every member is identical, '
                f'the ensemble spread is exactly 0 and spread_skill_sums returns NaN through its ddof=1. The search '
                f'space bounds it strictly above 0.'
            )
        # inject the MC rate where UNetBackbone reads it. Done BEFORE super().__init__ so save_hyperparameters()
        # records the EFFECTIVE architecture -- a checkpoint must not claim dropout 0.0 for a net that has it.
        trial = {**trial, 'unet': {**trial['unet'], 'dropout': dropout_p}}
        super().__init__(trial, in_channels, target_stats, normalization)
        self.dropout_p = dropout_p

        finetuning = trial.get('finetuning', {})
        self.finetuning_enabled = bool(finetuning.get('enabled', False))
        self.ensemble_loss = build_ensemble_loss(finetuning) if self.finetuning_enabled else None
        self.ensemble_loss_weight = float(finetuning.get('loss_weight', 1.0))
        # MC samples per step of the finetune phase, and per validation pass. One knob: the search space has no
        # separate mc_inference block, and the number that calibrates the spread is the number that measures it.
        self.mc_samples = int(finetuning.get('samples', 8))

        # set ONLY by from_upstream, never by a caller -- see the module docstring.
        self.warm_started = False

    # ------------------------------------------------------------------------------------------------------
    # warm start
    # ------------------------------------------------------------------------------------------------------
    @classmethod
    def from_upstream(cls, checkpoint_path: str, trial: dict, in_channels: int, target_stats: dict,
                      normalization: dict) -> 'MCDropoutModule':
        """Build an MC-dropout module initialised from a DETERMINISTIC U-net checkpoint's weights.

        Discharges three things at once, which is why it is one function:

        1. the architecture is taken from the CHECKPOINT, overriding the sampled ``unet`` block — this is what
           makes ``search.apply_constraints``'s "the sampled unet block is ignored" log actually true;
        2. a compatibility check that RAISES naming the offending field, rather than letting
           ``load_state_dict`` partially match;
        3. ``warm_started = True``, on an object whose weights are already loaded.

        Args:
            checkpoint_path: A ``deterministic_unet`` checkpoint (its ``best_model.ckpt``).
            trial: The sampled trial; its ``unet`` block is replaced by the checkpoint's.
            in_channels: Input channels of the CURRENT data — must match what the upstream was fitted on.
            target_stats: Current train-target statistics.
            normalization: Current per-channel feature normalization.

        Raises:
            ValueError: On a checkpoint with no recorded hyperparameters, an ``in_channels`` mismatch, or a
                ``mode`` mismatch — each naming the field.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved = checkpoint.get('hyper_parameters') or {}
        if not saved.get('trial'):
            raise ValueError(
                f'The upstream checkpoint "{checkpoint_path}" records no hyper_parameters.trial, so its architecture '
                f'cannot be read and a warm start would silently build a different network. Re-train the upstream '
                f'with the current code.'
            )

        saved_in_channels = int(saved.get('in_channels', in_channels))
        if saved_in_channels != int(in_channels):
            raise ValueError(
                f'Warm-start mismatch on "in_channels": the upstream was fitted on {saved_in_channels} input '
                f'channels but the current data provides {in_channels}. The two preparations differ in their feature '
                f'list, aggregation or residual flag — warm-start from an upstream trained on THIS prepared '
                f'directory.'
            )
        saved_mode = (saved.get('target_stats') or {}).get('mode')
        if saved_mode is not None and saved_mode != target_stats.get('mode'):
            raise ValueError(
                f'Warm-start mismatch on "mode": the upstream was trained on "{saved_mode}" but the current data is '
                f'"{target_stats.get("mode")}". The head means a different thing in each task, so its weights do not '
                f'transfer.'
            )

        # (1) the architecture comes from the checkpoint. Report what the sweep sampled and lost, so a trials table
        # showing varying `unet.*` columns against identical architectures is not mystifying.
        upstream_unet = dict(saved['trial']['unet'])
        sampled_unet = trial.get('unet', {})
        overridden = {
            key: (sampled_unet.get(key), upstream_unet.get(key))
            for key in WARM_START_ARCHITECTURE_KEYS
            if key in sampled_unet and sampled_unet.get(key) != upstream_unet.get(key)
        }
        if overridden:
            logger.info(
                'Warm start: taking the architecture from the upstream checkpoint and discarding the sampled values '
                + ', '.join(f'{key} {was!r} -> {now!r}' for key, (was, now) in overridden.items()) + '.'
            )
        # dropout is OURS, not the upstream's: the upstream is deterministic (dropout 0.0), and the MC rate is
        # precisely what this family's finetuning phase has to calibrate.
        upstream_unet['dropout'] = float(trial.get('dropout_p', 0.0))
        module = cls({**trial, 'unet': upstream_unet}, in_channels, target_stats, normalization)

        # (2)+(3) load the weights, then flag. strict=True: Fp32BilinearUpsample is parameter-free and the head was
        # renamed in lockstep, so the two state dicts must match exactly -- a partial load is the failure to prevent.
        net_state = {
            key[len('net.'):]: value for key, value in checkpoint['state_dict'].items() if key.startswith('net.')
        }
        module.net.load_state_dict(net_state, strict=True)
        module.warm_started = True
        logger.info(f'Warm-started from "{checkpoint_path}"; the train phase is replaced by that checkpoint.')
        return module

    # ------------------------------------------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------------------------------------------
    def training_phases(self) -> Tuple[str, ...]:
        """``('finetune',)`` after a warm start — phase 1 already happened, it IS the upstream checkpoint. Otherwise
        the two-phase fit, or ``train`` alone when fine-tuning is disabled."""
        if self.warm_started:
            fitting = ('finetune',)
        else:
            fitting = ('train', 'finetune') if self.finetuning_enabled else ('train',)
        return fitting + self._calibration_phases()

    def _check_phase_available(self, phase: str) -> None:
        if phase == 'finetune' and not self.finetuning_enabled:
            raise ValueError(
                'The finetune phase requires finetuning.enabled, which builds the ensemble loss. A warm-started run '
                'has no train phase to fall back on, which is why search.apply_constraints forces it true.'
            )
        super()._check_phase_available(phase)

    # ------------------------------------------------------------------------------------------------------
    # MC sampling
    # ------------------------------------------------------------------------------------------------------
    def mc_forward(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """``n_samples`` stochastic passes, returned in the TARGET SPACE as ``[N, B, H, W]``.

        The whole net runs in ``eval()`` with only ``Dropout2d`` resampling, so group norm stays frozen. Each member
        goes through :meth:`_to_prediction` INDIVIDUALLY — clamping the mean instead would let members above
        ``max_hours`` pull it upward, and the members are what the spread metrics read.
        """
        was_training = self.net.training
        self.net.eval()
        enable_mc_dropout(self.net)
        try:
            with torch.no_grad():
                members = [self._to_prediction(self._head_output(self(x))) for _ in range(max(int(n_samples), 1))]
        finally:
            self.net.train(was_training)             # restore, so a validation pass cannot leave the net in eval()
        return torch.stack(members, dim=0)

    def _target_space_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Validation accumulates the MC ENSEMBLE MEAN, in both phases. Phase 1 does not need the ensemble, but
        using it there too keeps the selection score measuring the same quantity across phases — otherwise the
        phase-1 and phase-2 numbers of one trial are not comparable, nor are two trials that stopped in different
        phases."""
        return self.mc_forward(x, self.mc_samples).mean(dim=0)

    # ------------------------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.phase != 'finetune':
            return super().training_step(batch, batch_idx)       # train + the calibration phases, unchanged

        x, y = batch[0], batch[1]
        head = self._head_output(self(x))
        pointwise = self._fitting_loss(head, y)                  # carries the gradient

        # SPLIT ESTIMATOR. CRPS = E|X - y| - 0.5 E|X - X'|. The first term is approximated by the current
        # grad-carrying prediction; the spread term uses stop-gradient MC samples, so calibrating dispersion costs
        # one sort rather than N backward passes.
        with torch.no_grad():
            detached = self.mc_forward(x, self.mc_samples)       # [N, B, H, W], target space, no grad
        member = self._to_prediction_differentiable(head)
        samples = torch.cat([member.unsqueeze(0), detached[1:]], dim=0)
        ensemble = self.ensemble_loss(samples, y)

        loss = pointwise + self.ensemble_loss_weight * ensemble
        self.log('train_pointwise_loss', pointwise, on_epoch=True, on_step=False)
        self.log('train_ensemble_loss', ensemble, on_epoch=True, on_step=False)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def _to_prediction_differentiable(self, head_output: torch.Tensor) -> torch.Tensor:
        """Target-space map that KEEPS the gradient, for slot 0 of the CRPS sample stack.

        :meth:`UnetModuleBase._to_prediction` detaches by design (it serves the metric paths). Here the same map has
        to stay differentiable, and it must be the SAME map or the ensemble loss would score a different quantity
        than the one evaluated. Daily is deliberately NOT clamped: a hard clamp zeroes the gradient of every
        over-predicting cell, which is exactly the population the CRPS spread term needs to move."""
        return torch.sigmoid(head_output) if self.hourly else head_output

    # ------------------------------------------------------------------------------------------------------
    # prediction (evaluation stage)
    # ------------------------------------------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, ensemble_size: Optional[int] = None):
        """The shared evaluation contract, with ``ensemble_members`` because this family is stochastic.

        ⚠️ ``ensemble_size`` must be >= 2: ``spread_skill_sums`` uses ``ddof=1``, so a one-member ensemble yields
        NaN rather than raising. The evaluation stage normally drives this through
        :class:`~src.utils.modeling.mc_dropout_eval.MCDropoutEnsembleModule`, which owns the eval-time knobs and the
        streaming partials; this method is the standalone path.
        """
        x, y = batch[0], batch[1]
        members = self.mc_forward(x, ensemble_size or self.mc_samples).cpu().float()   # [M, B, H, W]
        prediction = members.mean(dim=0)
        return {
            'prediction': prediction,
            'ensemble_members': members.movedim(0, 1),           # [B, M, H, W] -- the contract's layout
            'probability': prediction if self.hourly else None,
            'observation': y.detach().cpu().float()
        }

    # ------------------------------------------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------------------------------------------
    def _learning_rate(self) -> float:
        """The base LR, reduced by ``finetune_lr_factor`` while fine-tuning so phase 2 calibrates the dispersion
        without catastrophically forgetting the phase-1 fit."""
        lr = super()._learning_rate()
        if self.phase == 'finetune':
            lr *= float(self.trial['optimizer'].get('finetune_lr_factor', 0.05))
        return lr
