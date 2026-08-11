"""Shared LightningModule base for the two U-net families — the deterministic baseline and MC-dropout.

Everything that does not depend on HOW the model is sampled lives here, so the two families cannot drift apart on
it: the feature normalization, the task dispatch, the loss dispatch, the head-to-prediction maps, the calibration
phases, the validation composite and the optimizer. What a subclass supplies is small and explicit:

===============================  ==============================================================================
class attribute / hook           what it decides
===============================  ==============================================================================
``CHECKPOINT_MARKER``            the family name written into every checkpoint. **Required** — a class attribute
                                 rather than an overridable method so a family cannot silently inherit another's
                                 marker and be evaluated as the wrong family.
``PHASES``                       the phases ``set_phase`` accepts (MC-dropout adds ``finetune``).
``SUPPORTS_ENSEMBLE_LOSS``       whether a loss needing a sample axis (``crps_binary``) is admissible. False for
                                 a single forward pass, where the spread term is identically zero.
``predict_step``                 the evaluation contract. Abstract here: the two families differ (a point
                                 prediction vs an ensemble), and there is no safe default.
``_target_space_prediction``     what validation accumulates — one forward pass by default, the ensemble mean for
                                 a stochastic family.
``_learning_rate``               the base LR by default; MC-dropout scales it down in its finetune phase.
``_check_phase_available``       extra per-phase guards beyond the two calibration ones.
===============================  ==============================================================================

ONE OUTPUT HEAD, SELECTED BY ``mode``. The model emits a single map whose meaning is fixed by the prepared data's
mode, read from ``target_stats['mode']`` — it is a property of the DATA, not a hyperparameter:

===================  ==========================================  =============================================
                     ``mode: daily``                             ``mode: hourly``
===================  ==========================================  =============================================
target               lightning-hours per day, bounded ``0-24``   occurrence, ``0`` or ``1``
head output          ``softplus(raw)`` — non-negative hours      the RAW LOGIT (no activation)
prediction path      ``clamp(head, 0, max_hours)``               ``sigmoid(head)`` — a probability in ``[0, 1]``
calibration phase    ``regression_calibration`` (monotone)       ``occurrence_calibration`` (Platt, on the logit)
===================  ==========================================  =============================================

THE LOSS IS SELECTED BY ITS NAME, NOT BY THE MODE, because the hourly task admits both families. Both builders read
the same ``loss:`` section and the same ``name`` key inside it:

- ``name in BINARY_LOSSES`` → :func:`losses.build_binary_loss`, called on the **LOGIT** (every binary loss applies
  its own sigmoid — the uniform contract in ``losses.py``). Hourly only; a binary loss on a 0-24 target raises.
- ``name in REGRESSION_LOSSES`` → :func:`losses.build_regression_loss`, called on the value in the TARGET SPACE:
  the activated hours in daily mode, and the **PROBABILITY** in hourly mode.

⚠️ That second row is not a fallback but a legitimate configuration: a distance loss on a probability field is
proper — ``rmse(p, y) ** 2`` is exactly ``brier_score(p, y)``. And since no distance loss carries a positive-class
weight of its own, ``intensity_weight_gamma`` stays meaningful there: on a binary target
``(1 + y)^gamma ∈ {1, 2^gamma}`` IS a tunable positive-class weight. ``search.apply_constraints`` therefore zeroes
gamma for ``focal_bce`` alone, which brings its own ``positive_class_weight``, and leaves it alone otherwise.

TRAINING SPACE == EVALUATION SPACE. There is no target transform and no back-transform anywhere: whatever
:meth:`_head_output` returns is already in the space the metrics are computed in. The only difference between the
training and prediction paths is the ceiling — ``max_hours`` is enforced on the PREDICTION path only, never during
training, because clamping a live gradient would zero it for every over-predicting cell.

Metric accumulation assumes single-device training (``devices: 1``); a multi-GPU run would need a gather step.
"""
import logging
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io.data import MODE_HOURLY, normalize_mode
from src.utils.modeling.losses import (
    BINARY_LOSSES, REGRESSION_LOSSES, build_binary_loss, build_regression_loss, intensity_weights,
    log1p_huber, log1p_huber_quantile
)
from src.utils.modeling.unet import DeterministicUnetNet
from src.utils.modeling.validation import (
    DEFAULT_SELECTION_WEIGHTS, compute_selection_components, selection_metric_for_mode, selection_score
)

logger = logging.getLogger(__name__)

CALIBRATION_PHASES = ('occurrence_calibration', 'regression_calibration')


class UnetModuleBase(L.LightningModule):
    """Shared behaviour of the U-net families (see the module docstring for the subclass contract).

    Args:
        trial: Fully-sampled trial configuration. Sections read: ``loss``, ``unet``, ``calibration``, ``optimizer``,
            ``selection``, plus the top-level ``output_activation`` and ``max_hours``.
        in_channels: Number of input feature channels (the dataset's, including the appended ``upstream`` channel in
            residual mode).
        target_stats: Train-split target statistics from the preparation stage. ``mode`` is read from here.
        normalization: Per-CHANNEL feature normalization fitted on the train split at tuning time,
            ``{'mean': [C floats], 'std': [C floats]}``. Stored as checkpoint buffers and applied (followed by NaN
            imputation, i.e. mean-imputation in the standardized space) inside :meth:`forward`, so inference is
            self-contained — no external scaler files.

    Raises:
        ValueError: If ``target_stats`` carries no ``mode``; if the normalization channel count disagrees with
            ``in_channels`` (the residual-mode channel check); or if the sampled loss is inadmissible for the task
            or for this family's sampling.
    """

    #: Family name written into every checkpoint and read back by ``registry.load_model_module``. Subclasses MUST
    #: set it: two families sharing a marker would be dispatched to the wrong evaluation path.
    CHECKPOINT_MARKER = None
    #: Phases :meth:`set_phase` accepts. A stochastic family extends this.
    PHASES = ('train',) + CALIBRATION_PHASES
    #: Whether a loss needing a sample axis (``crps_binary``) can be trained by this family.
    SUPPORTS_ENSEMBLE_LOSS = False

    def __init__(self, trial: dict, in_channels: int, target_stats: dict, normalization: dict):
        super().__init__()
        self.save_hyperparameters()

        self.trial = trial
        self.target_stats = target_stats

        # --- feature normalization, baked into the checkpoint -------------------------------------------------
        mean = torch.as_tensor(normalization['mean'], dtype=torch.float32).view(-1, 1, 1)
        std = torch.as_tensor(normalization['std'], dtype=torch.float32).clamp(min=1e-6).view(-1, 1, 1)
        # this IS the residual-mode channel check: in residual mode the dataset appends `upstream` as the last
        # conditioning channel, so a normalization fitted without it would be silently misaligned per channel
        if mean.shape[0] != in_channels:
            raise ValueError(
                f'Normalization carries {mean.shape[0]} channels but the model expects {in_channels}.'
            )
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)

        # --- the task, from the DATA ------------------------------------------------------------------------
        if target_stats.get('mode') is None:
            raise ValueError(
                'target_stats carries no "mode", so the task is undetermined. It is written by the preparation '
                'stage into target_stats.json; re-prepare the data or point at a complete prepared directory.'
            )
        self.mode = normalize_mode(target_stats['mode'])            # raises on an unknown name
        self.hourly = self.mode == MODE_HOURLY
        self.max_hours = float(trial.get('max_hours', 24))
        self.output_activation = trial.get('output_activation', 'softplus')

        # --- calibration: at most one layer, chosen by the mode ---------------------------------------------
        calibration = trial.get('calibration', {})
        occurrence_calibration = calibration.get('occurrence', 'none')
        if occurrence_calibration not in ('none', 'platt'):
            raise ValueError(
                f'Unknown occurrence calibration "{occurrence_calibration}" (expected "none" or "platt").'
            )
        # Platt scaling recalibrates a LOGIT, so it only exists in hourly mode. In daily mode the head emits hours
        # and there is no logit to scale -- say so once rather than silently dropping a requested hyperparameter.
        # The LAYER lives in the net (built below), not here, so it travels inside net.state_dict() and a warm start
        # carries it; this flag only records whether to ask for it.
        output_calibration_enabled = self.hourly and occurrence_calibration == 'platt'
        if occurrence_calibration == 'platt' and not self.hourly:
            logger.info(
                'calibration.occurrence is "platt" but the mode is daily, where the head emits hours rather than '
                'a logit; the key is hourly-only and is ignored for this trial.'
            )

        # Conversely the monotone calibrator is a zero-preserving warp of a NON-NEGATIVE target, so it is daily-only.
        regression_calibration = calibration.get('regression', {})
        structure = regression_calibration.get('structure', 'none')
        self.regression_calibration_enabled = (not self.hourly) and structure not in (None, 'none')
        self.regression_calibration_huber_delta = float(regression_calibration.get('huber_delta', 1.0))
        # the SAME monotone calibrator is fitted with one of two objectives (a hyperparameter):
        #  * pointwise: per-cell log1p-Huber residual (corrects the conditional mean of the prediction);
        #  * quantile : log1p-Huber between the SORTED marginals (classical quantile-mapping bias correction).
        self.regression_calibration_objective = regression_calibration.get('objective', 'pointwise')
        if self.regression_calibration_objective not in ('pointwise', 'quantile'):
            raise ValueError(
                f'Unknown regression calibration objective "{self.regression_calibration_objective}" '
                f'(expected "pointwise" or "quantile").'
            )

        # --- network: ONE head, plus at most one calibration layer, both owned by the net ---------------------
        self.net = DeterministicUnetNet(
            in_channels, trial['unet'],
            output_calibration=output_calibration_enabled,
            regression_calibration={
                'structure': structure, 'num_sigmoids': int(regression_calibration.get('num_sigmoids', 4))
            } if self.regression_calibration_enabled else None
        )

        # --- loss: the NAME picks the builder, over the same `loss` section ---------------------------------
        # The hourly task admits both families -- a distance loss on the predicted probability is proper -- so this
        # cannot be a dispatch on the mode. `loss_takes_logits` records which space training_step must hand over.
        self.intensity_weight_gamma = float(trial['loss'].get('intensity_weight_gamma', 0.0))
        loss_name = trial['loss']['name']
        self.loss_takes_logits = loss_name in BINARY_LOSSES
        if self.loss_takes_logits:
            if not self.hourly:
                raise ValueError(
                    f'Loss "{loss_name}" is a BINARY loss but the mode is daily, whose target is lightning-hours '
                    f'in 0-{self.max_hours:g} rather than an occurrence label. Pick one of {REGRESSION_LOSSES}, or '
                    f'prepare the data with mode "hourly".'
                )
            binary_loss = build_binary_loss(trial['loss'])
            if binary_loss.needs_ensemble and not self.SUPPORTS_ENSEMBLE_LOSS:
                raise ValueError(
                    f'Loss "{loss_name}" needs an ensemble [N, *spatial], but {type(self).__name__} predicts with a '
                    f'single forward pass: N = 1, so its spread term is identically zero and it degrades silently '
                    f'to a mean absolute error on probabilities. Use it with mc_dropout or diffusion, or pick a '
                    f'pointwise binary loss.'
                )
            self.binary_loss_needs_ensemble = binary_loss.needs_ensemble
            self.loss_fn = binary_loss.fn
        else:
            self.binary_loss_needs_ensemble = False
            self.loss_fn = build_regression_loss(trial['loss'])          # raises on a name in neither family

        # --- trial selection -------------------------------------------------------------------------------
        selection = trial.get('selection', {})
        # ONE source of truth with the sweep: run_sweep ranks trials on selection_metric_for_mode(mode, declared)
        # and passes it to _fit_trial as the prune metric, while _fit_trial checkpoints on monitor_metric. If the
        # two names differed, checkpointing and ranking would optimise different quantities without erroring.
        self.selection_metric = selection_metric_for_mode(self.mode, selection.get('metric'))
        self.selection_components = dict(
            selection.get('components') or DEFAULT_SELECTION_WEIGHTS[self.selection_metric]
        )
        # model-independent denominators of two components, injected by the tuning stage before the sweep (they are
        # properties of the validation split, not of the model). None outside a sweep -> that component is NaN.
        self.valid_climatology_cond_mae = None
        self.valid_climatology_brier = None
        self.selection_occurrence_event = None

        self.phase = 'train'
        self.last_val_metrics = {}
        self._reset_validation_buffers()

    # ------------------------------------------------------------------------------------------------------
    # phases
    # ------------------------------------------------------------------------------------------------------
    def _calibration_phases(self) -> Tuple[str, ...]:
        """The trailing calibration phases, at most one — WHICH is determined by the mode, not configured: Platt
        scaling applies to the hourly logit, the monotone warp to the daily hours."""
        if self.hourly:
            return ('occurrence_calibration',) if self.net.output_calibration is not None else ()
        return ('regression_calibration',) if self.regression_calibration_enabled else ()

    def training_phases(self) -> Tuple[str, ...]:
        """The phase sequence :func:`tuning._fit_trial` fits, in order. One fitting phase, then the calibration."""
        return ('train',) + self._calibration_phases()

    def _check_phase_available(self, phase: str) -> None:
        """Raise if ``phase`` is in :data:`PHASES` but not constructible for this trial. Subclass hook."""
        if phase == 'occurrence_calibration' and self.net.output_calibration is None:
            raise ValueError(
                'The occurrence-calibration phase requires the Platt layer, which exists only in hourly mode with '
                'calibration.occurrence = "platt".'
            )
        if phase == 'regression_calibration' and not self.regression_calibration_enabled:
            raise ValueError(
                'The regression-calibration phase requires the monotone calibrator, which exists only in daily '
                'mode with calibration.regression.structure != "none".'
            )

    def set_phase(self, phase: str) -> None:
        """Enter a training phase. A calibration phase freezes the WHOLE backbone and trains only its own layer;
        both calibration layers are identity-initialised and frozen otherwise, so they are exact no-ops until their
        own phase."""
        if phase not in self.PHASES:
            raise ValueError(f'Unknown phase "{phase}" (expected one of {self.PHASES}).')
        self._check_phase_available(phase)
        self.phase = phase

        if phase in CALIBRATION_PHASES:
            for parameter in self.net.parameters():
                parameter.requires_grad = False
            trainable = self.net.output_calibration_parameters() if phase == 'occurrence_calibration' \
                else self.net.regression_calibration_parameters()
            for parameter in trainable:
                parameter.requires_grad = True
            return

        # any fitting phase (train, and finetune for MC-dropout): the backbone trains, calibration stays frozen
        for parameter in self.net.parameters():
            parameter.requires_grad = True
        for parameter in self.net.regression_calibration_parameters():
            parameter.requires_grad = False
        for parameter in self.net.output_calibration_parameters():
            parameter.requires_grad = False

    @property
    def monitor_metric(self) -> str:
        """Validation metric to checkpoint / early-stop on in the current phase. Outside the calibration phases it
        is the sweep's selection composite, so the checkpoint kept is the one the trial is ranked by."""
        if self.phase == 'occurrence_calibration':
            return 'valid_occurrence_calibration'       # negative validation BCE of the calibrated probabilities
        if self.phase == 'regression_calibration':
            return 'valid_reg_calibration'              # negative validation log1p-Huber on observed-positive cells
        return self.selection_metric

    @property
    def monitor_mode(self) -> str:
        """Every metric this module monitors is MAXIMIZED (the composite, and the negated calibration losses), so
        the direction is constant across its phases."""
        return 'max'

    # ------------------------------------------------------------------------------------------------------
    # forward, head output and prediction
    # ------------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw single-head output, ``[B, H, W]``. Standardizes through the train-fitted checkpoint buffers, then
        imputes NaNs with the (standardized) mean, so inference needs no external scaler."""
        x = x.float()                                # items may arrive in the dataset's storage dtype (e.g. float16)
        x = (x - self.feature_mean) / self.feature_std
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(x)                           # single tensor: Platt (hourly) is applied inside the net

    def _head_output(self, raw: torch.Tensor) -> torch.Tensor:
        """Map the raw head output into the space the LOSS consumes — hours in daily mode, a LOGIT in hourly mode.

        Daily applies the non-negative activation and then the monotone calibration (identity-initialised and
        frozen outside its own phase, so a no-op during training). Hourly is a pass-through: the net has already
        applied Platt scaling inside its own ``forward``, affine in the logit and therefore BEFORE the sigmoid. No
        sigmoid here either way — every binary loss takes logits and applies its own."""
        if self.hourly:
            return raw                               # already Platt-scaled by the net when hourly calibration is on

        if self.output_activation == 'softplus':
            activated = F.softplus(raw)
        elif self.output_activation == 'relu':
            activated = F.relu(raw)
        else:
            raise ValueError(f'Unknown output activation "{self.output_activation}".')
        if self.net.regression_calibration is not None:
            activated = self.net.regression_calibration(activated)
        return activated

    def _to_prediction(self, head_output: torch.Tensor) -> torch.Tensor:
        """Map the head output to the PREDICTION, in the target space. Detached, on the input's device.

        This is the only place the two task-specific final maps live: the ``max_hours`` ceiling for daily (softplus
        is unbounded above, so the ceiling cannot come from the activation) and the sigmoid for hourly. Never called
        on the training path — clamping a live gradient would zero it for every over-predicting cell.

        A stochastic family applies this PER MEMBER, before averaging: clamping the mean would let members above the
        ceiling pull it up, and the members are what the spread metrics read."""
        head_output = head_output.detach()
        if self.hourly:
            return torch.sigmoid(head_output)
        return head_output.clamp(min=0.0, max=self.max_hours)

    def _target_space_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """The prediction validation accumulates, ``[B, H, W]`` in the target space. One forward pass here; a
        stochastic family overrides this with its ensemble mean so the selection score is consistent across
        phases."""
        return self._to_prediction(self._head_output(self(x)))

    # ------------------------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------------------------
    def _fitting_loss(self, head: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The loss of a fitting phase, given the head output. Shared by ``train`` and (as one of its two terms)
        MC-dropout's ``finetune``."""
        if self.loss_takes_logits:
            return self.loss_fn(head, y)                        # head is a LOGIT; the loss sigmoids internally
        # A distance loss consumes the TARGET SPACE, so hourly hands it the probability -- the one sigmoid, applied
        # here as it is on the prediction path. Daily hands over the activated hours unchanged.
        prediction = torch.sigmoid(head) if self.hourly else head
        # every cell contributes: there is no classifier level owning the zeros any more, and the imbalance is
        # carried by the intensity weighting instead (gamma = 0 leaves the loss unweighted). On a binary target
        # that weighting IS a positive-class weight, which is why it stays a useful knob in hourly mode.
        weights = intensity_weights(y, self.intensity_weight_gamma) \
            if self.intensity_weight_gamma > 0 else torch.ones_like(y)
        return self.loss_fn(prediction, y, weights, torch.ones_like(y))

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        head = self._head_output(self(x))

        # Platt phase (hourly): recalibrate the frozen-backbone logit with PLAIN BCE on the occurrence labels,
        # independent of which binary loss trained the head. Rank-preserving, so it moves the probabilities
        # (Brier / reliability / deviance) without moving a hard decision mask.
        if self.phase == 'occurrence_calibration':
            loss = F.binary_cross_entropy_with_logits(head, (y > 0).float())
            self.log('train_occurrence_calibration_loss', loss, on_epoch=True, on_step=False)
        # monotone phase (daily): fit the frozen-backbone calibrator with a PLAIN, symmetric log1p-Huber on
        # observed-positive cells -- a neutral, extremes-robust objective, decoupled from the backbone's weighted
        # loss so the monotone map corrects its systematic distortion instead of relearning the identity.
        elif self.phase == 'regression_calibration':
            calibrate = log1p_huber_quantile if self.regression_calibration_objective == 'quantile' else log1p_huber
            loss = calibrate(head, y, (y > 0), self.regression_calibration_huber_delta)
            self.log('train_reg_calibration_loss', loss, on_epoch=True, on_step=False)
        else:
            loss = self._fitting_loss(head, y)

        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    # ------------------------------------------------------------------------------------------------------
    # validation: the selection composite, in the target space
    # ------------------------------------------------------------------------------------------------------
    def _reset_validation_buffers(self):
        self._val_prediction = []
        self._val_observation = []

    def on_validation_epoch_start(self):
        self._reset_validation_buffers()

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        self._val_prediction.append(self._target_space_prediction(x).cpu().float().numpy())
        self._val_observation.append(y.detach().cpu().float().numpy())

    def on_validation_epoch_end(self):
        prediction = np.concatenate(self._val_prediction)
        observation = np.concatenate(self._val_observation)
        occurrence = observation > 0

        # calibration monitors: negated so they are MAXIMIZED like every other monitored metric, and each mirrors
        # the objective its phase fits so the phase selects the matching checkpoint
        if self.net.output_calibration is not None:
            clipped = np.clip(prediction, 1e-7, 1.0 - 1e-7)     # hourly: the prediction IS the probability
            bce = -np.mean(occurrence * np.log(clipped) + (~occurrence) * np.log(1.0 - clipped))
            self.log('valid_occurrence_calibration', -float(bce))
        if self.regression_calibration_enabled:
            self.log('valid_reg_calibration', -self._validation_reg_calibration(prediction, observation, occurrence))

        # In hourly mode the prediction IS the occurrence probability (evaluation.py documents the same contract),
        # which is what makes brier_skill_score computable there. In daily mode there is no probabilistic output --
        # ACCEPTED COST of dropping the second head -- so brier_skill_score is NaN and the ranking components fall
        # back to ranking on the predicted hours, which is exact: AP and AUC are invariant to any monotone map.
        components = compute_selection_components(
            prediction, observation,
            climatology_cond_mae=self.valid_climatology_cond_mae,
            climatology_brier=self.valid_climatology_brier,
            occurrence_probability=prediction if self.hourly else None,
            occurrence_event=self.selection_occurrence_event
        )
        score = selection_score(components, self.selection_components)

        for name, value in components.items():
            self.log(f'valid_{name}', value if np.isfinite(value) else 0.0)
        self.log(self.selection_metric, score, prog_bar=True)

        self.last_val_metrics = {self.selection_metric: float(score)}
        self.last_val_metrics.update({
            f'valid_{name}': float(value) if np.isfinite(value) else float('nan')
            for name, value in components.items()
        })
        self._reset_validation_buffers()

    def _validation_reg_calibration(self, prediction: np.ndarray, observation: np.ndarray,
                                    occurrence: np.ndarray) -> float:
        """Validation log1p-Huber on observed-positive cells — the numpy mirror of :func:`losses.log1p_huber`, whose
        torch original the regression-calibration phase minimizes. Mirrors its pointwise-vs-quantile pairing too, so
        the monitored metric selects the checkpoint that phase actually improved.

        ⚠️ TWO IMPLEMENTATIONS OF ONE FORMULA, which is the divergence risk this repo names elsewhere (the CRPS in
        losses.py vs scores.py). numpy is needed here because validation accumulates numpy arrays. The gate asserts
        this agrees with ``losses.log1p_huber`` on the POINTWISE objective; the quantile pair is deliberately NOT
        asserted equal, because the torch loss sorts per BATCH while this sorts the whole EPOCH."""
        if not occurrence.any():
            return 0.0
        delta = self.regression_calibration_huber_delta
        predicted, observed = np.clip(prediction[occurrence], 0, None), observation[occurrence]
        if self.regression_calibration_objective == 'quantile':
            predicted, observed = np.sort(predicted), np.sort(observed)
        residual = np.log1p(predicted) - np.log1p(observed)
        absolute = np.abs(residual)
        return float(np.where(absolute <= delta, 0.5 * residual ** 2, delta * (absolute - 0.5 * delta)).mean())

    # ------------------------------------------------------------------------------------------------------
    # prediction (evaluation stage) -- ABSTRACT: the families differ and there is no safe default
    # ------------------------------------------------------------------------------------------------------
    def predict_step(self, batch, batch_idx):
        """Return the evaluation contract: ``observation`` and ``prediction`` both ``[B, H, W]``, plus
        ``ensemble_members`` ``[B, M, H, W]`` only for a stochastic family, and ``probability`` (the prediction
        itself in hourly mode, ``None`` in daily).

        Deliberately not implemented here: silently inheriting a point prediction would make a stochastic family
        report no ensemble metrics at all, with nothing raising."""
        raise NotImplementedError(
            f'{type(self).__name__} must implement predict_step (see UnetModuleBase.predict_step for the contract).'
        )

    # ------------------------------------------------------------------------------------------------------
    # checkpoint marker (read by registry.load_model_module to dispatch the shared evaluation stage)
    # ------------------------------------------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.CHECKPOINT_MARKER is None:
            raise ValueError(
                f'{type(self).__name__} sets no CHECKPOINT_MARKER, so its checkpoints could not be dispatched to '
                f'an evaluation path. Set it as a class attribute.'
            )
        checkpoint['module_class'] = self.CHECKPOINT_MARKER

    # ------------------------------------------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------------------------------------------
    def _learning_rate(self) -> float:
        """Learning rate for the current phase. Subclass hook — MC-dropout scales it down while fine-tuning."""
        return float(self.trial['optimizer']['lr'])

    def configure_optimizers(self):
        optimizer_config = self.trial['optimizer']
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self._learning_rate(),
            weight_decay=float(optimizer_config['weight_decay'])
        )
        scheduler_name = optimizer_config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.trainer.max_epochs or 1, 1)
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        if scheduler_name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor_metric}
            }
        raise ValueError(f'Unknown scheduler "{scheduler_name}".')
