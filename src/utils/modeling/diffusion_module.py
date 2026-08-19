"""LightningModule for the conditional flow-matching (diffusion) family.

Unlike the other two families this one does NOT share ``UnetModuleBase``: it has no U-net, no output activation, no
head whose meaning flips with the mode, and its training objective lives in a different space from its prediction.
What it does share is the contract everything downstream depends on — ``training_phases`` / ``set_phase`` /
``monitor_metric`` / ``monitor_mode`` for the tuning loop, and ``predict_step``'s evaluation dict.

TWO MODES, decided by the PREPARED DATA (``residual_target`` in ``target_stats``), never by a hyperparameter:

  FULL TARGET  (``UPSTREAM_MODEL`` unset at prepare time): the flow generates the lightning-hours map directly.
  RESIDUAL     (``UPSTREAM_MODEL`` set):                   it generates the DISCREPANCY ``y - upstream`` and the
                                                           prediction is ``clamp(upstream + residual, 0, max_hours)``.
  The upstream prediction is appended as the LAST conditioning channel and arrives as a third batch item,
  ``(x_cond, y, upstream)``. Sampling this per trial is impossible — half the trials would ask for a mode the
  prepared directory cannot provide — which is why the search space carries no ``residual_target`` key.

⚠️ GENERATION IS IN THE RAW TARGET SPACE. No warp, no standardization, no inverse: training space == evaluation
space. The source branch generated in a log1p-standardized space (``flow.log_warp``), which was the removed target
transform under another name. The honest consequence to know: the flow transports a standard-normal prior onto a
field that is 95.30 % exactly zero (daily; 99.57 % hourly), which is a hard transport problem — a plain affine standardization would NOT
have helped, since it rescales that spike rather than spreading it; only the log warp did, and it is out of scope.

⚠️ THE ``loss:`` BLOCK OF THE SEARCH SPACE IS NOT READ HERE. This family trains on the flow-matching velocity MSE
alone. A target-space term would need a full ODE integration (``flow.n_steps`` forward passes) per training step,
roughly an order of magnitude slower, with a gradient threaded through the whole Euler chain. The block stays in the
config as shared skeleton and the config says so too, so neither file implies an objective the other lacks.

WHAT IS MONITORED VS WHAT IS RANKED, and why they differ here alone. Checkpointing and early stopping monitor the
cheap ``valid_flow_loss`` (minimized, one velocity-MSE pass per batch, NO sampling), while ``run_sweep`` ranks
trials on the target-space composite. The composite needs the ODE sampler, so it is computed only every
``SCORE_EVERY_N_EPOCHS`` epochs (0 = never during the fit) and once at the end of each trial, when the tuning loop
calls :meth:`prepare_full_validation` on the restored best checkpoint. So the kept checkpoint is chosen by a PROXY
for the ranking metric. That is a deliberate cost trade, not an oversight, and ``tuning._fit_trial`` already accounts
for it: it attaches the optuna pruning callback only when ``monitor_metric == prune_metric``, so this family simply
never prunes rather than pruning on a metric it does not log every epoch.

Metric accumulation assumes single-device training (``devices: 1``); a multi-GPU run would need a gather step.
"""
import logging
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io.data import normalize_mode
from src.utils.metrics import scores
from src.utils.modeling.diffusion import FlowVelocityNet, flow_matching_targets, sample
from src.utils.modeling.validation import (
    DEFAULT_SELECTION_WEIGHTS, compute_selection_components, selection_metric_for_mode, selection_score
)

logger = logging.getLogger(__name__)

PHASES = ('train',)
CHECKPOINT_MARKER = 'diffusion'

# Validation-scoring cadence. The composite costs an ODE integration per item, so it is NOT computed every epoch:
# 0 means "only at the end of the trial", via prepare_full_validation(). Not a search-space key -- it trades trial
# wall-clock against how early a bad trial is visible, and the end-of-trial pass is what the trials table needs.
SCORE_EVERY_N_EPOCHS = 0
# Independent ODE draws per item on a scoring pass. 1 keeps validation cheap; the pointwise components then score
# that single draw and the structure components are measured on it too.
VALID_ENSEMBLE_SIZE = 1


class DiffusionModule(L.LightningModule):
    """Conditional flow-matching model for one sampled trial.

    Args:
        trial: Fully-sampled trial configuration. Sections read: ``flow`` (architecture + ODE steps), ``optimizer``,
            ``selection``, plus the top-level ``max_hours``. ``loss:`` is deliberately NOT read — see the module
            docstring.
        in_channels: Conditioning channels (standardized ERA5 predictors, plus the upstream-prediction channel in
            residual mode). The single noisy-target channel is added inside the velocity net.
        target_stats: Train-split target statistics from the preparation stage. ``mode`` and ``residual_target`` are
            read from here — both are properties of the DATA, not hyperparameters.
        normalization: Per-CHANNEL conditioning normalization fitted on the train split at tuning time,
            ``{'mean': [C floats], 'std': [C floats]}`` (the upstream channel is the last entry in residual mode).
            Stored as checkpoint buffers, so inference is self-contained.

    Raises:
        ValueError: If ``target_stats`` carries no ``mode``, or the normalization channel count disagrees with
            ``in_channels`` (which in residual mode is the check that the upstream channel is actually present).
    """

    def __init__(self, trial: dict, in_channels: int, target_stats: dict, normalization: dict):
        super().__init__()
        self.save_hyperparameters()

        self.trial = trial
        self.target_stats = target_stats
        self.in_channels = in_channels

        mean = torch.as_tensor(normalization['mean'], dtype=torch.float32).view(-1, 1, 1)
        std = torch.as_tensor(normalization['std'], dtype=torch.float32).clamp(min=1e-6).view(-1, 1, 1)
        # in residual mode the dataset appends `upstream` LAST, so this also validates that the normalization was
        # fitted on the residual-mode channel set rather than the plain one
        if mean.shape[0] != in_channels:
            raise ValueError(
                f'Normalization carries {mean.shape[0]} channels but the model expects {in_channels}'
                + (' (residual mode appends the upstream prediction as the last channel).'
                   if target_stats.get('residual_target') else '.')
            )
        self.register_buffer('feature_mean', mean)
        self.register_buffer('feature_std', std)

        if target_stats.get('mode') is None:
            raise ValueError(
                'target_stats carries no "mode", so the task is undetermined. It is written by the preparation '
                'stage into target_stats.json; re-prepare the data or point at a complete prepared directory.'
            )
        self.mode = normalize_mode(target_stats['mode'])
        # a property of the prepared directory, NOT a sampled hyperparameter -- see the module docstring
        self.residual_target = bool(target_stats.get('residual_target', False))
        self.max_hours = float(trial.get('max_hours', 24))

        flow = trial.get('flow', {})
        self.net = FlowVelocityNet(in_channels, self._net_config(flow))
        self.num_sampling_steps = int(flow.get('n_steps', 16))
        # optional evaluation-time override of the ODE step count (set by the evaluation stage); None = the trial's
        self.eval_sampling_steps: Optional[int] = None
        # evaluation-time ensemble knobs, set by the stage. Defaults keep predict_step a single deterministic draw.
        # eval_ensemble_size > 1 turns on the probabilistic suite; eval_occurrence_event conditions the tail CRPS and
        # the rank histogram; eval_ensemble_seed seeds the per-member noise so the evaluation is reproducible.
        self.eval_ensemble_size: int = 1
        self.eval_occurrence_event: Optional[Tuple[float, bool]] = None
        self.eval_ensemble_seed: int = 0
        # residual-mode diagnostics: when set by the stage, an ensemble predict_step ALSO returns the UNCLAMPED
        # generated discrepancy per member plus the upstream prediction, so diagnostics.residual_diagnostics sees the
        # model's true correction rather than the censored clamp(P) - U. Free: the raw draw already exists.
        self.eval_return_residual: bool = False

        selection = trial.get('selection', {})
        self.selection_metric = selection_metric_for_mode(self.mode, selection.get('metric'))
        self.selection_components = dict(
            selection.get('components') or DEFAULT_SELECTION_WEIGHTS[self.selection_metric]
        )
        self.valid_climatology_cond_mae = None
        self.valid_climatology_brier = None
        self.selection_occurrence_event = None

        # fixed seed for the validation noise/time, so valid_flow_loss is a deterministic function of the weights
        # (a stable signal across epochs, independent of global-RNG drift during the fit)
        self.valid_seed = 1234
        self.phase = 'train'
        self._force_full_validation = False
        self._scoring_epoch = False
        self.last_val_metrics = {}
        self._reset_validation_buffers()

    @staticmethod
    def _net_config(flow: dict) -> dict:
        """Translate the trial's ``flow`` section into :class:`FlowVelocityNet`'s keys.

        Only one name differs, deliberately: the config calls the DiT-block count ``n_blocks`` because ``depth``
        already means the down/upsampling level count in the ``unet`` block, and one word meaning two things across
        families is worse than one translation here."""
        return {
            'hidden_dim': int(flow['hidden_dim']),
            'depth': int(flow['n_blocks']),
            'num_heads': int(flow['num_heads']),
            'patch_size': int(flow['patch_size']),
        }

    # ------------------------------------------------------------------------------------------------------
    # phase / monitoring interface (the contract tuning._fit_trial drives)
    # ------------------------------------------------------------------------------------------------------
    def training_phases(self) -> Tuple[str, ...]:
        """One phase. There is no calibration phase here: both calibrators belong to a head this family has not
        got, and neither is meaningful on an ODE draw."""
        return PHASES

    def set_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f'Unknown phase "{phase}" (the diffusion model only supports {PHASES}).')
        self.phase = phase

    @property
    def monitor_metric(self) -> str:
        """The cheap flow-matching loss — a PROXY for the composite the sweep ranks on (see the module docstring)."""
        return 'valid_flow_loss'

    @property
    def monitor_mode(self) -> str:
        """The flow-matching loss is MINIMIZED — the one family whose monitored metric is not maximized."""
        return 'min'

    def prepare_full_validation(self) -> None:
        """Force the next validation pass to run the ODE sampler and compute the target-space composite. Called by
        the tuning loop before each trial's final, best-checkpoint validation, so every trial reports its composite
        to the trials table even when nothing was scored during the fit."""
        self._force_full_validation = True

    # ------------------------------------------------------------------------------------------------------
    # conditioning / generation helpers
    # ------------------------------------------------------------------------------------------------------
    def _unpack(self, batch):
        """``(conditioning maps, target, upstream or None)`` for either dataset layout."""
        if self.residual_target:
            return batch[0], batch[1], batch[2]
        return batch[0], batch[1], None

    def _standardize_conditioning(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize the CONDITIONING maps through the train-fitted buffers, then impute NaNs with the
        (standardized) mean — the same mechanism as the U-net families' feature handling.

        ⚠️ This standardizes the INPUTS only. The generation target is left in the raw target space."""
        x = x.float()                                # conditioning may arrive in the dataset's dtype (e.g. float16)
        x = (x - self.feature_mean) / self.feature_std
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _generation_target(self, y: torch.Tensor, upstream: Optional[torch.Tensor]) -> torch.Tensor:
        """The map the velocity field learns to generate, in the RAW target space: the signed discrepancy
        ``y - upstream`` in residual mode, or the target ``y`` itself."""
        return y - upstream if self.residual_target else y

    def _sample_time(self, batch_size: int, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Logit-normal time sampling: it concentrates on mid-path times, where the velocity is hardest to predict
        and the flow-matching literature finds the training signal most valuable."""
        normal = torch.randn(batch_size, device=device, generator=generator)
        return torch.sigmoid(normal)

    def _flow_loss(self, x1: torch.Tensor, cond: torch.Tensor,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """The training objective: MSE between the predicted velocity and the straight-path target ``x1 - z`` at a
        sampled time. ONE forward pass, and the whole objective of this family."""
        t = self._sample_time(x1.shape[0], x1.device, generator)
        noise = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        x_t, target_velocity = flow_matching_targets(x1, t, noise)
        return F.mse_loss(self.net(x_t, t, cond), target_velocity)

    def _predict_target_space(self, x: torch.Tensor, upstream: Optional[torch.Tensor], num_steps: int,
                              generator: Optional[torch.Generator] = None, return_residual: bool = False):
        """Integrate the ODE and reconstruct the prediction, ``[B, H, W]`` on the CPU.

        ``clamp(upstream + residual, 0, max_hours)`` in residual mode, ``clamp(draw, 0, max_hours)`` in full mode.
        Both bounds matter: the flow is unconstrained, so nothing else keeps a draw inside the target's range.

        With ``return_residual`` also returns the UNCLAMPED generated discrepancy, which is what the residual
        diagnostics need — the censored ``clamp(P) - upstream`` would hide exactly the corrections that overshoot."""
        cond = self._standardize_conditioning(x)
        generated = sample(self.net, cond, x.shape[-2:], num_steps, generator=generator).detach().cpu().float()
        reconstructed = generated + upstream.detach().cpu().float() if self.residual_target else generated
        prediction = reconstructed.clamp(min=0.0, max=self.max_hours)
        return (prediction, generated) if return_residual else prediction

    def _draw_ensemble(self, x: torch.Tensor, upstream: Optional[torch.Tensor], num_steps: int, batch_idx: int,
                       members: int, seed: int, return_residual: bool = False):
        """``members`` independent ODE integrations sharing the conditioning, each seeded ``seed + batch_idx*M + i``
        so the whole pass is reproducible and collision-free across batches. Returns ``[M, B, H, W]`` (CPU float),
        or that plus the matching unclamped-discrepancy stack.

        Validation and evaluation pass DIFFERENT seeds on purpose: trial selection must not draw the same noise the
        final evaluation will, or the selected trial is the one that got lucky on the evaluation's own draws."""
        drawn, residuals = [], []
        for member in range(members):
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed + batch_idx * members + member)
            result = self._predict_target_space(x, upstream, num_steps, generator=generator,
                                                return_residual=return_residual)
            if return_residual:
                drawn.append(result[0])
                residuals.append(result[1])
            else:
                drawn.append(result)
        if return_residual:
            return torch.stack(drawn, dim=0), torch.stack(residuals, dim=0)
        return torch.stack(drawn, dim=0)

    # ------------------------------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------------------------------
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(x_t, t, cond)

    def training_step(self, batch, batch_idx):
        x, y, upstream = self._unpack(batch)
        loss = self._flow_loss(self._generation_target(y, upstream), self._standardize_conditioning(x))
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    # ------------------------------------------------------------------------------------------------------
    # validation: the cheap flow loss every epoch, the target-space composite on a scoring pass
    # ------------------------------------------------------------------------------------------------------
    def _reset_validation_buffers(self):
        self._val_flow_loss_sum = 0.0
        self._val_flow_loss_count = 0
        self._val_prediction = []           # per-batch ensemble means [B, H, W] -> pointwise / categorical scoring
        self._val_members = []              # per-batch member stacks [B, M, H, W] -> pooled structure scoring
        self._val_observation = []

    def on_validation_epoch_start(self):
        self._reset_validation_buffers()
        self._scoring_epoch = bool(self._force_full_validation) or (
            SCORE_EVERY_N_EPOCHS > 0 and (self.current_epoch + 1) % SCORE_EVERY_N_EPOCHS == 0
        )

    def validation_step(self, batch, batch_idx):
        x, y, upstream = self._unpack(batch)
        cond = self._standardize_conditioning(x)
        x1 = self._generation_target(y, upstream)

        # the monitored metric: cheap, deterministic, NO sampling
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.valid_seed + batch_idx)
        flow_loss = self._flow_loss(x1, cond, generator=generator)
        self._val_flow_loss_sum += float(flow_loss) * x1.shape[0]
        self._val_flow_loss_count += x1.shape[0]

        if self._scoring_epoch:
            members = self._draw_ensemble(x, upstream, self.num_sampling_steps, batch_idx,
                                          members=VALID_ENSEMBLE_SIZE, seed=self.valid_seed)
            self._val_prediction.append(members.mean(dim=0).numpy())        # [B, H, W]
            self._val_members.append(members.movedim(0, 1).numpy())         # [B, M, H, W]
            self._val_observation.append(y.detach().cpu().float().numpy())

    def on_validation_epoch_end(self):
        flow_loss = self._val_flow_loss_sum / max(self._val_flow_loss_count, 1)
        self.log('valid_flow_loss', flow_loss, prog_bar=True)
        self.last_val_metrics = {'valid_flow_loss': float(flow_loss)}

        if self._scoring_epoch and self._val_prediction:
            prediction = np.concatenate(self._val_prediction)               # [N, H, W] ensemble means
            observation = np.concatenate(self._val_observation)             # [N, H, W]
            # POOLED member stack for the structure components: every item-member pair as one map, the observation
            # replicated per member (np.repeat interleaves per item, matching the [N*M, H, W] reshape). The pooled
            # estimator averages per-map spectra, never the maps, so the texture terms are not measured on a mean
            # that is smoother than any member. Mirrors run_metric_suite's mean-vs-pooled split.
            members = np.concatenate(self._val_members)                     # [N, M, H, W]
            n_items, n_members = members.shape[0], members.shape[1]
            components = compute_selection_components(
                prediction, observation,
                climatology_cond_mae=self.valid_climatology_cond_mae,
                climatology_brier=self.valid_climatology_brier,
                occurrence_probability=None,        # no probabilistic head; AP/AUC rank on the prediction itself
                occurrence_event=self.selection_occurrence_event,
                prediction_structure=members.reshape(n_items * n_members, *members.shape[2:]),
                observation_structure=np.repeat(observation, n_members, axis=0)
            )
            score = selection_score(components, self.selection_components)
            for name, value in components.items():
                self.log(f'valid_{name}', value if np.isfinite(value) else 0.0)
            self.log(self.selection_metric, score)
            self.last_val_metrics[self.selection_metric] = float(score)
            self.last_val_metrics.update({
                f'valid_{name}': float(value) if np.isfinite(value) else float('nan')
                for name, value in components.items()
            })

        self._force_full_validation = False
        self._reset_validation_buffers()

    # ------------------------------------------------------------------------------------------------------
    # prediction (evaluation stage)
    # ------------------------------------------------------------------------------------------------------
    def predict_step(self, batch, batch_idx):
        """The shared evaluation contract. ``ensemble_members`` only when the stage asked for an ensemble, so a
        single-draw run is treated exactly like a deterministic family's point run."""
        x, y, upstream = self._unpack(batch)
        num_steps = self.eval_sampling_steps or self.num_sampling_steps
        observation = y.detach().cpu().float()

        if self.eval_ensemble_size <= 1:
            return {
                'prediction': self._predict_target_space(x, upstream, num_steps),
                'probability': None,                # no occurrence-probability head in this family
                'observation': observation
            }

        want_residual = bool(self.eval_return_residual and self.residual_target)
        drawn = self._draw_ensemble(x, upstream, num_steps, batch_idx, members=int(self.eval_ensemble_size),
                                    seed=self.eval_ensemble_seed, return_residual=want_residual)
        members, residual_members = drawn if want_residual else (drawn, None)
        rng = np.random.default_rng(self.eval_ensemble_seed + batch_idx)
        output = {
            'prediction': members.mean(dim=0),                   # ensemble mean -> point/skill/categorical scores
            'ensemble_members': members.movedim(0, 1),           # [B, M, H, W] -> pooled structure scores
            'probability': None,
            'observation': observation,
            'ensemble_partials': scores.ensemble_partials(
                members.numpy(), observation.numpy(),            # member-FIRST [M, ...]: crps_sums' layout
                occurrence_event=self.eval_occurrence_event, rng=rng
            )
        }
        if want_residual:
            output['ensemble_residual_members'] = residual_members.movedim(0, 1)   # [B, M, H, W] unclamped r
            output['upstream'] = upstream.detach().cpu().float()                   # [B, H, W] U
        return output

    # ------------------------------------------------------------------------------------------------------
    # checkpoint marker (read by registry.load_model_module to dispatch the shared evaluation stage)
    # ------------------------------------------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint['module_class'] = CHECKPOINT_MARKER

    # ------------------------------------------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer_config = self.trial['optimizer']
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=float(optimizer_config['lr']),
            weight_decay=float(optimizer_config['weight_decay'])
        )
        scheduler_name = optimizer_config.get('scheduler', 'cosine')
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.trainer.max_epochs or 1, 1)
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        if scheduler_name == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=self.monitor_mode, patience=3      # 'min': this family's monitor is minimized
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': self.monitor_metric}
            }
        raise ValueError(f'Unknown scheduler "{scheduler_name}".')
