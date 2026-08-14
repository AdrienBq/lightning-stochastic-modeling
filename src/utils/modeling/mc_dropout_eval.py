"""Evaluation adapter that lets an MC-dropout checkpoint be scored by THE shared evaluation stage, identically to a
diffusion model.

``evaluate`` drives every family through one duck-typed contract:
  * eval-time knobs set on the loaded module — ``eval_ensemble_size`` (``> 1`` enables the probabilistic suite),
    ``eval_occurrence_event`` (the metrics.yaml occurrence event), ``eval_ensemble_seed``, ``target_stats``;
  * ``predict_step(batch, batch_idx)`` returning, in the target space:
      - single-sample run: ``{'prediction': [B,H,W], 'probability': ..., 'observation': [B,H,W]}``;
      - ensemble run:      ``{'prediction'` (ensemble mean) ``[B,H,W], 'ensemble_members' [B,M,H,W],
                              'probability': ..., 'observation' [B,H,W],
                              'ensemble_partials': scores.ensemble_partials(...)}``.

:class:`MCDropoutModule` already produces an M-member ensemble in target space via ``mc_forward``, but it has no
``eval_*`` knobs and does not compute the streaming partials. This thin wrapper adds exactly those two things, and
reuses THIS repo's :func:`scores.ensemble_partials` so CRPS, almost-fair CRPS, spread-skill and the rank histogram
come from the same code path as the diffusion family's. That is what makes the two comparable: only the per-batch
ensemble generator differs (MC-dropout passes vs ODE sampling).

⚠️ The partials are SUMS, not means (``crps_sums`` / ``spread_skill_sums`` return ``(sum, ..., n_cells)``), because
the full ``[N, M, H, W]`` stack cannot be held in memory. They are additive across batches; means and ratios are
not. The stage divides exactly once, at the end.
"""
import logging
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch

from src.utils.metrics import scores
from src.utils.modeling.mc_dropout_module import MCDropoutModule

logger = logging.getLogger(__name__)


class MCDropoutEnsembleModule(L.LightningModule):
    """Adapt a loaded :class:`MCDropoutModule` to the shared evaluation ``predict_step`` contract.

    The wrapped module is registered as a submodule, so ``trainer.predict`` moves it to the chosen accelerator and
    its baked feature-normalization buffers travel with it. Nothing here changes the model's numerics: the members
    come straight from :meth:`MCDropoutModule.mc_forward`.

    Args:
        wrapped: A constructed or loaded :class:`MCDropoutModule`.
    """

    def __init__(self, wrapped: MCDropoutModule):
        super().__init__()
        self.wrapped = wrapped

        # eval-time knobs the stage sets (mirroring the diffusion module). The default of 1 keeps predict_step a
        # single MC draw, i.e. behaviour analogous to a deterministic point run.
        self.eval_ensemble_size: int = 1
        self.eval_occurrence_event: Optional[Tuple[float, bool]] = None
        self.eval_ensemble_seed: int = 0
        # accepted for interface parity with the diffusion module (the number of ODE steps has no MC analogue); the
        # stage only sets it when present, and it is intentionally ignored here.
        self.eval_sampling_steps: Optional[int] = None

    @property
    def target_stats(self) -> dict:
        """Proxy the wrapped model's train-target statistics (read by the stage for the occurrence-threshold sanity
        check and available to downstream code)."""
        return getattr(self.wrapped, 'target_stats', {})

    @property
    def expected_in_channels(self) -> int:
        """Input channels the loaded checkpoint was fitted on (the width of its baked normalization)."""
        return int(self.wrapped.feature_mean.shape[0])

    def _unpack(self, batch):
        """Conditioning maps ``x`` and target ``y``. A residual-mode dataset also yields an upstream channel, which
        a full-target MC model cannot consume — caught here rather than by a shape error deep in the backbone."""
        x, y = batch[0], batch[1]
        if x.shape[1] != self.expected_in_channels:
            raise ValueError(
                f'MC-dropout checkpoint expects {self.expected_in_channels} input channels but the prepared data '
                f'provides {x.shape[1]}. Evaluate it against a prepared directory built with the same feature list '
                f'and aggregation it was trained on (and non-residual, unless the checkpoint itself is residual).'
            )
        return x, y

    def predict_step(self, batch, batch_idx):
        x, y = self._unpack(batch)
        observation = y.detach().cpu().float()
        members_wanted = int(self.eval_ensemble_size)

        # per-batch deterministic seeding so the ensemble is reproducible across runs (the diffusion module seeds its
        # per-member generation noise for the same reason). MC-dropout draws its masks from the global RNG.
        torch.manual_seed(self.eval_ensemble_seed + batch_idx)
        if x.is_cuda:
            torch.cuda.manual_seed_all(self.eval_ensemble_seed + batch_idx)

        members = self.wrapped.mc_forward(x, max(members_wanted, 1)).cpu().float()      # [M, B, H, W], target space

        # single-sample (pre-ensemble) behaviour: one draw, no probabilistic suite -- mirrors the diffusion module's
        # deterministic branch, so a non-ensemble MC run is treated exactly like a non-ensemble diffusion run
        if members_wanted <= 1:
            prediction = members[0]
            return {
                'prediction': prediction,
                'probability': prediction if self.wrapped.hourly else None,
                'observation': observation
            }

        prediction = members.mean(dim=0)
        rng = np.random.default_rng(self.eval_ensemble_seed + batch_idx)
        return {
            'prediction': prediction,                            # ensemble mean -> point/skill/categorical/calibration
            'ensemble_members': members.movedim(0, 1),           # [B, M, H, W] -> pooled spatial-structure scores
            'probability': prediction if self.wrapped.hourly else None,
            'observation': observation,
            'ensemble_partials': scores.ensemble_partials(
                members.numpy(), observation.numpy(),            # member-FIRST [M, ...]: crps_sums' layout
                occurrence_event=self.eval_occurrence_event, rng=rng
            )
        }
