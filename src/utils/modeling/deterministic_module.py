"""LightningModule for the deterministic U-net — the baseline, and the UPSTREAM model for the other two families.

Almost everything lives in :class:`~src.utils.modeling.unet_module_base.UnetModuleBase`, shared with MC-dropout:
the feature normalization, the ``mode``-selected head, the loss dispatch, the calibration phases, the validation
composite and the optimizer. Read that module's docstring for the model itself.

What is specific to this family, and all that is left here:

- ``CHECKPOINT_MARKER = 'deterministic_unet'`` — the family name the evaluation stage dispatches on.
- ``predict_step`` returns a POINT prediction with no ``ensemble_members`` key, so the shared evaluation stage
  emits no ensemble metrics for it.
- ``SUPPORTS_ENSEMBLE_LOSS`` stays False (inherited): one forward pass gives N = 1, where an ensemble loss has an
  identically-zero spread term.

That thinness is the point. This family and MC-dropout differ only in how the model is SAMPLED, so anything else
that differed between them would be drift, not design — and the warm start relies on it, loading these weights
straight into an MC-dropout net.
"""
import logging

from src.utils.modeling.unet_module_base import UnetModuleBase

logger = logging.getLogger(__name__)

class DeterministicUnetModule(UnetModuleBase):
    """Deterministic U-net for one sampled trial. See :class:`UnetModuleBase` for the constructor arguments."""

    CHECKPOINT_MARKER = 'deterministic_unet'

    def predict_step(self, batch, batch_idx):
        """The shared evaluation contract: ``observation`` and ``prediction`` both ``[B, H, W]``, and NO
        ``ensemble_members`` — this family is deterministic, so the evaluation stage emits no ensemble metrics.

        ``probability`` is the prediction itself in hourly mode (where it IS a probability, which is what lets the
        threshold-free scores read a genuine probability field) and ``None`` in daily mode."""
        x, y = batch[0], batch[1]
        prediction = self._target_space_prediction(x).cpu().float()
        return {
            'prediction': prediction,
            'probability': prediction if self.hourly else None,
            'observation': y.detach().cpu().float()
        }


# Module-level aliases for callers that import these from here rather than from the class / the base.
CHECKPOINT_MARKER = DeterministicUnetModule.CHECKPOINT_MARKER
PHASES = DeterministicUnetModule.PHASES
