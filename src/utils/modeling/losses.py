"""Training losses for both tasks, shared by all three model families.

Every option is designed against the extreme class imbalance of the target (~99.93 % of cells are zero):
- intensity weighting ``w(y) = (1 + y)^gamma`` pulls gradients towards high-hour cells;
- the asymmetric Huber penalizes under-prediction ``tau / (1 - tau)`` times more than over-prediction
  (``tau > 0.5`` encodes the preference for conservative models);
- the PSD composites add a differentiable whole-spectrum structure penalty, so a model cannot buy pointwise
  accuracy with blur;
- focal BCE with a positive-class weight drives the high-recall occurrence classifier of the hierarchy.

EVERY BINARY LOSS TAKES LOGITS. Each applies its own sigmoid internally, and the probability is formed exactly once
per loss, so there is no configuration in which the head has to know which space the sampled loss wants:

| builder                   | returns                              | callable signature                      |
|---------------------------|--------------------------------------|-----------------------------------------|
| ``build_regression_loss`` | a callable                           | ``(pred, target, weights, mask)``       |
| ``build_binary_loss``     | ``BinaryLoss(fn, needs_ensemble)``   | ``(logits, target)``                    |
| ``build_ensemble_loss``   | a callable                           | ``(samples [N, *spatial], target)``     |

The occurrence head therefore emits a RAW LOGIT and never sigmoids it for training; the sigmoid happens once, on
the inference path, where the reported probability is formed. That makes the worst mismatch structurally impossible
rather than merely documented: there is no probability-taking binary loss left to hand a logit to, and no
logit-taking loss left to hand a probability to (which would sigmoid it a second time, squashing [0, 1] into
[0.5, 0.73] — a monotone map that trains without ever erroring).

One difference survives, and it is about SHAPE rather than space: ``crps_binary`` needs an ensemble
``[N, *spatial]``. Given a plain ``[B, H, W]`` batch it would read the batch as a B-member ensemble and return a
number, so it cannot be caught by shape alone — hence ``BinaryLoss.needs_ensemble``.

The regression and ensemble builders are SEPARATE because the MC-dropout finetuning phase uses both at once:
``loss = regression_loss(...) + loss_weight * ensemble_loss(...)``. They are not alternatives, and their signatures
differ, so there is no single builder to fold them into.

Every pointwise loss reduces through :func:`_weighted_masked_mean`, which normalises by the SUM OF EFFECTIVE
WEIGHTS rather than the cell count. Inlining a reduction risks a loss on a different scale from its siblings, which
makes tuning results incomparable across trials.

Deliberately absent (do not reintroduce):
- ``mae`` / ``rmse`` — absorbed at ``gamma = 0``, where ``(1 + y)^0 = 1`` makes ``weighted_mae`` identically MAE.
  This is why ``intensity_weight_gamma``'s lower bound must stay 0.0 in the search spaces: raising it makes the
  unweighted losses unreachable.
- ``tweedie_deviance`` / ``poisson_nll`` — parameterized for an UNBOUNDED zero-inflated target. Poisson is actively
  wrong on a bounded 0-24 target: it places probability mass above 24 hours per day.
- ``TRANSFORM_COMPATIBLE_LOSSES`` — there is no target transform, so every loss is compatible with the only space
  there is.
"""
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F

# exactly the `loss.name` choices of the three DAILY search spaces (config/<family>/search_space_daily.yaml). An
# hourly space names a SUBSET — a distance loss on the predicted probability is proper (`rmse ** 2` IS the Brier
# score) — but never `weighted_mae` / `wmae_psd`, which are IMPROPER against a 0/1 observation.
REGRESSION_LOSSES = ('weighted_mae', 'weighted_rmse', 'weighted_mse', 'asymmetric_huber', 'wmae_psd', 'wmse_psd')
# the `loss.name` choices of an HOURLY search space; see BinaryLoss for the input space of each.
# ⚠️ These were once `occurrence_head.loss`, and that block is GONE by decision (search_space_daily.yaml records it):
# they are the MAIN loss of the hourly classification task, not an auxiliary head's, read from the same `loss:` section
# as the regression names above. config/deterministic_unet/search_space_hourly.yaml offers three of the four —
# `crps_binary` needs a genuine ensemble, so it belongs to an hourly mc_dropout or diffusion pipeline.
BINARY_LOSSES = ('focal_bce', 'dice', 'brier', 'crps_binary')
# exactly the `finetuning.loss` choices (MC-dropout phase 2)
ENSEMBLE_LOSSES = ('crps', 'almost_fair_crps', 'afcrps_psd')
_EPS = 1e-8


def intensity_weights(y_raw: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-cell weights ``(1 + y)^gamma`` computed from the RAW targets; ``gamma = 0`` is unweighted."""
    return (1.0 + y_raw.clamp(min=0.0)) ** gamma


def _weighted_masked_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` normalised by the SUM OF EFFECTIVE WEIGHTS (``weights * mask``), not the cell count.

    Every pointwise loss reduces through here so that all of them live on the same scale and their tuning results
    stay comparable. Normalising by the count instead would make a heavily-weighted loss numerically larger purely
    because of its weights.
    """
    effective = weights * mask
    return (values * effective).sum() / effective.sum().clamp(min=_EPS)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _weighted_masked_mean((pred - target) ** 2, weights, mask)


def weighted_mae(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _weighted_masked_mean((pred - target).abs(), weights, mask)


def weighted_rmse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    return _weighted_masked_mean((pred - target) ** 2, weights, mask).sqrt()


def asymmetric_huber(
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        tau: float = 0.8,
        delta: float = 1.0
) -> torch.Tensor:
    """Huber loss with quantile-style asymmetry: residuals where the model UNDER-predicts (target > pred) are
    weighted ``tau``, over-predictions ``1 - tau``; ``tau > 0.5`` prefers conservative models."""
    residual = target - pred
    abs_residual = residual.abs()
    huber = torch.where(
        abs_residual <= delta,
        0.5 * residual ** 2,
        delta * (abs_residual - 0.5 * delta)
    )
    asymmetry = torch.where(residual > 0, torch.full_like(residual, tau), torch.full_like(residual, 1.0 - tau))
    return _weighted_masked_mean(huber * asymmetry, weights, mask)


def psd_penalty(pred: torch.Tensor, target: torch.Tensor, num_bins: int = None) -> torch.Tensor:
    """Differentiable whole-spectrum power-spectrum fidelity penalty in ``[0, 1]`` (0 = perfect match).

    Torch counterpart of ``scores.radial_psd`` + ``scores.psd_fidelity`` applied over the WHOLE spectrum
    rather than a single band: the predicted and observed mean 2-D power spectra are radially binned into
    ``num_bins`` frequency bins (DC component excluded), and a scale-invariant power ratio -> fidelity gap
    is formed per bin and averaged over all populated bins

        ratio_b   = mean predicted power in bin b / mean observed power in bin b
        fidelity_b = clip(1 - |1 - ratio_b|, 0, 1)       (== scores.psd_fidelity, per bin)
        penalty    = 1 - mean_b(fidelity_b)

    Averaging per-bin (rather than over raw coefficients) gives every spatial scale equal weight, so the
    penalty rewards faithful structure across the whole spectrum instead of only the low-frequency power
    that dominates the raw spectrum. Each ratio is scale-invariant, so no normalisation of ``pred``/
    ``target`` is needed — and equally, the penalty carries NO magnitude signal, which is why every composite
    below keeps the pointwise term dominant.

    Args:
        pred: Predicted maps, shape ``[..., H, W]`` (leading dims are flattened and averaged over).
        target: Ground-truth maps, same shape as ``pred``.
        num_bins: Number of radial frequency bins; defaults to ``min(H, W) // 2`` to match ``scores.radial_psd``.

    Returns:
        Scalar penalty in ``[0, 1]``; minimising it maximises whole-spectrum fidelity.
    """
    pred = pred.reshape(-1, *pred.shape[-2:])
    target = target.reshape(-1, *target.shape[-2:]).to(pred.dtype)
    height, width = pred.shape[-2:]

    pred_power = (torch.fft.fft2(pred).abs() ** 2).mean(dim=0)       # [H, W]
    obs_power = (torch.fft.fft2(target).abs() ** 2).mean(dim=0)      # [H, W]

    ky = torch.fft.fftfreq(height, device=pred.device, dtype=pred.dtype).unsqueeze(1)
    kx = torch.fft.fftfreq(width, device=pred.device, dtype=pred.dtype).unsqueeze(0)
    radial_frequency = torch.sqrt(kx ** 2 + ky ** 2).reshape(-1)    # [H*W]; 0 at the DC component

    num_bins = num_bins or (min(height, width) // 2)
    edges = torch.linspace(0.0, 0.5, num_bins + 1, device=pred.device, dtype=pred.dtype)

    # keep non-DC coefficients up to the Nyquist radius (mirrors scores.radial_psd, which drops the corners)
    keep = (radial_frequency > 0) & (radial_frequency <= 0.5)
    freq = radial_frequency[keep]
    pred_flat = pred_power.reshape(-1)[keep]
    obs_flat = obs_power.reshape(-1)[keep]

    # bin b collects coefficients with frequency in (edges[b], edges[b + 1]]
    bin_index = (torch.bucketize(freq, edges, right=False) - 1).clamp(0, num_bins - 1)
    counts = torch.zeros(num_bins, device=pred.device, dtype=pred.dtype).scatter_add_(
        0, bin_index, torch.ones_like(freq))
    pred_bin = torch.zeros(num_bins, device=pred.device, dtype=pred.dtype).scatter_add_(0, bin_index, pred_flat)
    obs_bin = torch.zeros(num_bins, device=pred.device, dtype=pred.dtype).scatter_add_(0, bin_index, obs_flat)

    populated = counts > 0
    ratio = (pred_bin[populated] / counts[populated]) / (obs_bin[populated] / counts[populated] + _EPS)
    fidelity = (1.0 - (1.0 - ratio).abs()).clamp(0.0, 1.0)
    return 1.0 - fidelity.mean()


def wmae_psd(
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.8
) -> torch.Tensor:
    """Composite: ``alpha * weighted_mae + (1 - alpha) * psd_penalty``.

    Blends the pointwise weighted-MAE error with the differentiable whole-spectrum PSD penalty so the model is
    rewarded for both accurate magnitudes and faithful structure at every spatial scale. ``alpha`` keeps the
    pointwise term dominant: the PSD ratios are scale-invariant and carry no magnitude signal, so a PSD-dominated
    loss would happily admit a well-textured field with the wrong values. ``alpha = 1`` recovers ``weighted_mae``.
    """
    return alpha * weighted_mae(pred, target, weights, mask) + (1.0 - alpha) * psd_penalty(pred, target)


def wmse_psd(
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.8
) -> torch.Tensor:
    """Composite: ``alpha * weighted_mse + (1 - alpha) * psd_penalty``.

    The squared-error sibling of :func:`wmae_psd` — same structure term, but the pointwise term penalizes large
    misses quadratically, so it is the harsher option on the high-hour cells. ``alpha = 1`` recovers
    ``weighted_mse``.
    """
    return alpha * weighted_mse(pred, target, weights, mask) + (1.0 - alpha) * psd_penalty(pred, target)


def focal_bce_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        focal_gamma: float = 2.0,
        positive_class_weight: float = 1.0
) -> torch.Tensor:
    """Focal binary cross-entropy with a positive-class weight, for the occurrence classifier head. Takes LOGITS."""
    pos_weight = torch.as_tensor(positive_class_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    return ((1.0 - p_t) ** focal_gamma * bce).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice / F1 loss for binary classification: ``1 - 2*TP / (2*TP + FP + FN)``.

    Takes LOGITS and applies its own sigmoid, like every other binary loss here.

    Its eval-time complement is ``scores.dice_coefficient`` on ``sigmoid(logits)``, so the reported
    ``dice_<threshold>`` measures exactly what this optimized.

    Args:
        logits: Raw model outputs (pre-sigmoid), any shape.
        target: Binary ground-truth labels (0/1), same shape.
        smooth: Laplace smoothing to avoid 0/0 on empty batches.
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)


def brier_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Brier score: mean squared error of the predicted probability vs binary labels. Takes LOGITS.

    Equals ``scores.brier_score(sigmoid(logits), target)`` — the same number, entered from the training side.

    Args:
        logits: Raw model outputs (pre-sigmoid), any shape.
        target: Binary ground-truth labels (0/1), same shape.
    """
    return ((torch.sigmoid(logits) - target.float()) ** 2).mean()


def crps(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Continuous ranked probability score for continuous non-negative targets.

    CRPS(F, y) = E[|X - y|] - (1/2) E[|X - X'|]

    The spread term is estimated in O(N log N) via the order-statistics formula:
        E[|X - X'|] = (2 / N²) Σ_{k=0}^{N-1} (2k - N + 1) * X_{(k)}
    i.e. spread_term = Σ(2k-N+1)*X_(k) / N² = E[|X-X'|] / 2, so CRPS = mae_term - spread_term.

    ⚠️ Must agree numerically with ``scores.crps_ensemble``, the reference contract. The two implementations
    exist because one is a torch training loss and the other a numpy verification score; the gate pins their
    agreement, because a silent divergence would mean training against a different quantity than the one reported.

    Args:
        samples: MC prediction samples, shape [N, *spatial].
        target: Continuous ground-truth values, shape [*spatial].

    Returns:
        Scalar CRPS averaged over all spatial positions.
    """
    n = samples.shape[0]
    mae_term = (samples - target.unsqueeze(0)).abs().mean(dim=0)

    sorted_s, _ = torch.sort(samples, dim=0)
    k = torch.arange(n, dtype=samples.dtype, device=samples.device)
    weights = (2.0 * k - n + 1).view(-1, *([1] * (samples.ndim - 1)))
    spread_term = (weights * sorted_s).sum(dim=0) / (n * n)  # = E[|X-X'|] / 2

    return (mae_term - spread_term).mean()


def almost_fair_crps(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bias-corrected CRPS estimator for continuous targets (Ferro 2014).

    The standard CRPS estimator is negatively biased in the spread term because the N MC samples
    are used to estimate both E[|X - y|] and E[|X - X'|] from the same draw. Ferro (2014) shows
    the bias-corrected "almost fair" CRPS replaces the spread factor 1/2 with (N-1)/(2N):

        CRPS_af = E[|X - y|] - ((N-1)/(2N)) * E[|X - X'|]
                = mae_term - ((N-1)/N) * spread_term

    where spread_term = E[|X-X'|]/2 (see crps). For large N the correction vanishes; it matters
    most when finetune_samples is small (< 32).

    Args:
        samples: MC prediction samples, shape [N, *spatial].
        target: Continuous ground-truth values, shape [*spatial].

    Returns:
        Scalar almost-fair CRPS averaged over all spatial positions.
    """
    n = samples.shape[0]
    mae_term = (samples - target.unsqueeze(0)).abs().mean(dim=0)

    sorted_s, _ = torch.sort(samples, dim=0)
    k = torch.arange(n, dtype=samples.dtype, device=samples.device)
    weights = (2.0 * k - n + 1).view(-1, *([1] * (samples.ndim - 1)))
    spread_term = (weights * sorted_s).sum(dim=0) / (n * n)  # = E[|X-X'|] / 2

    fair_factor = (n - 1) / n  # (N-1)/N; approaches 1 as N -> inf
    return (mae_term - fair_factor * spread_term).mean()


def afcrps_psd(samples: torch.Tensor, target: torch.Tensor, beta: float = 0.7) -> torch.Tensor:
    """Composite: ``beta * almost_fair_crps + (1 - beta) * psd_penalty``.

    Adds the differentiable whole-spectrum PSD penalty (evaluated on the ensemble mean) to the
    almost-fair CRPS, so fine-tuning calibrates the ensemble without buying calibration with blur.

    Gradient note: in the phase-2 split estimator only slot 0 of ``samples`` carries gradient
    (see the MC-dropout module's ``training_step``), so the penalty on ``samples.mean(dim=0)`` back-props
    through that slot scaled ~1/N — consistent with how the CRPS MAE term is already estimated.

    Args:
        samples: MC prediction samples, shape ``[N, B, H, W]``.
        target: Ground-truth maps, shape ``[B, H, W]``.
        beta: Weight on the CRPS term; ``1 - beta`` weights the PSD penalty.
    """
    return beta * almost_fair_crps(samples, target) + (1.0 - beta) * psd_penalty(samples.mean(dim=0), target)


def crps_binary(logit_samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Continuous ranked probability score for binary events (Ferro & Fricker 2012). Takes LOGIT samples.

    CRPS(F, y) = E[|P - y|] - (1/2) E[|P - P'|]

    The spread term is computed in O(N log N) via the order-statistics formula:
        E[|P - P'|] = (2 / N²) Σ_{k=0}^{N-1} (2k - N + 1) * P_{(k)}
    where P_{(0)} ≤ ... ≤ P_{(N-1)} are the sorted MC samples.

    ⚠️ The only binary loss needing a SAMPLE AXIS, and the one place the uniform logit contract does not make a
    mismatch impossible: handed a plain ``[B, H, W]`` batch it would read the BATCH as a B-member ensemble and
    return a number. That is what ``BinaryLoss.needs_ensemble`` flags.

    ⚠️ With ``N = 1`` the spread term is identically zero and this degrades to a mean absolute error on
    probabilities — which is why it is not offered to the deterministic family, whose single forward pass has no
    sample axis.

    Args:
        logit_samples: MC samples of the raw model output (pre-sigmoid), shape [N, *spatial].
        target: Binary ground-truth labels (0/1), shape [*spatial].

    Returns:
        Scalar CRPS averaged over all spatial positions.
    """
    samples = torch.sigmoid(logit_samples)
    n = samples.shape[0]
    mae_term = (samples - target.unsqueeze(0)).abs().mean(dim=0)        # [*spatial]

    sorted_s, _ = torch.sort(samples, dim=0)                            # [N, *spatial]
    k = torch.arange(n, dtype=samples.dtype, device=samples.device)
    weights = (2.0 * k - n + 1).view(-1, *([1] * (samples.ndim - 1)))  # [N, 1, ...]
    spread_term = (weights * sorted_s).sum(dim=0) / (n * n)             # [*spatial]

    return (mae_term - 0.5 * spread_term).mean()


# ----------------------------------------------------------------------------------------------------------------
# CALIBRATION OBJECTIVES — reached by `calibration.regression.objective`, NOT by `loss.name`
#
# These fit the MonotoneCalibration layer (unet.py) in its own final phase with the backbone frozen. They are
# deliberately absent from REGRESSION_LOSSES: they are not selectable as a backbone loss, and that tuple is asserted
# set-equal to the search spaces' `loss.name` choices.
#
# ⚠️ They take (pred, target, mask, delta) and NOT the (pred, target, weights, mask) of every other pointwise loss
# ⚠️ here. The missing `weights` is the point, not an oversight: the calibrator is fitted with a PLAIN, SYMMETRIC
# ⚠️ objective precisely so it is decoupled from the intensity-weighted backbone loss — sharing that weighting would
# ⚠️ let the monotone map relearn the identity instead of correcting the backbone's systematic distortion. A
# ⚠️ `weights` parameter would invite the one thing the design rules out.
# ----------------------------------------------------------------------------------------------------------------
def log1p_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float) -> torch.Tensor:
    """Mean Huber loss on the ``log1p`` residual over the masked cells.

    The log1p compression makes the loss scale-robust across the target's wide dynamic range; Huber makes it
    extremes-robust. log1p space is valid here BECAUSE the target is non-negative — there is no signed transformed
    space any more, so the earlier count-regression objection to this objective no longer applies.

    Args:
        pred: Predicted target, non-negative (clamped defensively).
        target: Observed target, non-negative.
        mask: Cells that contribute; typically ``target > 0`` (the calibrator is fitted on observed positives).
        delta: Huber transition width in log1p space (~a factor e at ``delta = 1``).
    """
    residual = torch.log1p(pred.clamp(min=0.0)) - torch.log1p(target.clamp(min=0.0))
    absolute = residual.abs()
    elementwise = torch.where(absolute <= delta, 0.5 * residual ** 2, delta * (absolute - 0.5 * delta))
    # reduces through the shared helper like every other pointwise loss, at unit weights: the normalizer is then the
    # masked cell count, which is what a plain (unweighted) objective wants
    return _weighted_masked_mean(elementwise, torch.ones_like(elementwise), mask.to(elementwise.dtype))


def log1p_huber_quantile(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                         delta: float) -> torch.Tensor:
    """Marginal quantile-matching variant of :func:`log1p_huber`: the same log1p-Huber, but between the SORTED
    masked prediction and the SORTED masked observation.

    Sorting each marginal independently and pairing by rank is 1-D quantile mapping (discrete optimal transport), so
    minimizing it realigns the prediction's marginal QUANTILES onto the observation's — classical bias correction —
    rather than correcting each cell pointwise. Same signature as :func:`log1p_huber`, so the two are interchangeable
    as the ``calibration.regression.objective``.

    ⚠️ The sort population is whatever tensor it is handed, so a per-batch training call and a per-epoch validation
    call do NOT return the same number. That is intended — each realigns the marginals it can see — but it means only
    the pointwise objective is comparable across the two.
    """
    mask = mask.bool()
    pred_sorted = torch.sort(pred[mask])[0]
    target_sorted = torch.sort(target[mask])[0]
    return log1p_huber(pred_sorted, target_sorted, torch.ones_like(pred_sorted), delta)


# ----------------------------------------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------------------------------------
class BinaryLoss(NamedTuple):
    """An occurrence-head loss together with the one thing about it the caller still has to know.

    Every binary loss takes LOGITS, so the input SPACE needs no flag — the head emits one tensor and hands it to
    whichever loss was sampled. What still differs is the SHAPE: ``crps_binary`` needs an ensemble
    ``[N, *spatial]`` while the other three take the head output as-is. That distinction cannot be made structural,
    because a plain ``[B, H, W]`` batch handed to ``crps_binary`` is a valid B-member ensemble as far as the tensor
    shapes are concerned — it would return a number, computed over the wrong axis.
    """
    fn: Callable
    needs_ensemble: bool            # crps_binary only: the callable's first argument is [N, *spatial]


def build_regression_loss(loss_config: dict) -> Callable:
    """Build the pointwise regression loss ``loss(pred, target, weights, mask)`` from a trial's ``loss`` section.

    Args:
        loss_config: Sampled ``loss`` section — ``name``, plus ``asymmetry_tau`` / ``huber_delta`` for
            ``asymmetric_huber`` and ``alpha`` for the two PSD composites. ``intensity_weight_gamma`` is NOT read
            here: it parameterizes :func:`intensity_weights`, which the module applies to produce ``weights``.

    Raises:
        ValueError: On a name outside :data:`REGRESSION_LOSSES`.
    """
    name = loss_config['name']
    if name not in REGRESSION_LOSSES:
        raise ValueError(f'Unknown regression loss "{name}"; expected one of {REGRESSION_LOSSES}.')

    if name == 'weighted_mae':
        return weighted_mae
    if name == 'weighted_rmse':
        return weighted_rmse
    if name == 'weighted_mse':
        return weighted_mse
    if name == 'asymmetric_huber':
        tau, delta = float(loss_config['asymmetry_tau']), float(loss_config['huber_delta'])
        return lambda pred, target, weights, mask: asymmetric_huber(pred, target, weights, mask, tau=tau, delta=delta)
    alpha = float(loss_config['alpha'])
    composite = wmae_psd if name == 'wmae_psd' else wmse_psd
    return lambda pred, target, weights, mask: composite(pred, target, weights, mask, alpha=alpha)


def build_binary_loss(loss_config: dict) -> BinaryLoss:
    """Build the binary occurrence loss from a trial's ``loss`` section. Every option takes LOGITS.

    This is the MAIN loss of the hourly classification task, not an auxiliary head's: it reads the same ``loss``
    section and the same ``name`` key as :func:`build_regression_loss`, and ``mode`` selects between the two
    builders (see ``unet_module_base.py``). Keeping the two signatures aligned is what lets the module dispatch on
    the mode alone, with no second config key.

    Args:
        loss_config: Sampled ``loss`` section — ``name``, plus ``positive_class_weight`` / ``focal_gamma`` for
            ``focal_bce`` and ``dice_smooth`` for ``dice``.

    Returns:
        :class:`BinaryLoss` — the callable and whether its first argument is an ensemble.

    Raises:
        ValueError: On a name outside :data:`BINARY_LOSSES`.
    """
    name = loss_config['name']
    if name not in BINARY_LOSSES:
        raise ValueError(f'Unknown binary loss "{name}"; expected one of {BINARY_LOSSES}.')

    if name == 'focal_bce':
        pos_weight = float(loss_config['positive_class_weight'])
        gamma = float(loss_config['focal_gamma'])
        return BinaryLoss(
            lambda logits, target: focal_bce_with_logits(
                logits, target.float(), focal_gamma=gamma, positive_class_weight=pos_weight
            ),
            needs_ensemble=False
        )
    if name == 'dice':
        smooth = float(loss_config.get('dice_smooth', 1.0))
        return BinaryLoss(
            lambda logits, target: dice_loss(logits, target.float(), smooth=smooth), needs_ensemble=False
        )
    if name == 'brier':
        return BinaryLoss(brier_loss, needs_ensemble=False)
    return BinaryLoss(crps_binary, needs_ensemble=True)


def build_ensemble_loss(finetuning_config: dict) -> Callable:
    """Build the MC fine-tuning loss ``loss(samples, target)`` from a trial's ``finetuning`` section.

    Kept separate from :func:`build_regression_loss` because the finetuning phase uses BOTH — its step is
    ``regression_loss(...) + loss_weight * ensemble_loss(...)`` — and the two signatures differ.

    Args:
        finetuning_config: Sampled ``finetuning`` section — ``loss``, and ``beta`` when ``loss == afcrps_psd``.
            ``loss_weight`` is read by the module, not here: it scales this loss against the pointwise one.

    Returns:
        Callable ``loss(samples, target)`` with samples ``[N, *spatial]`` and target ``[*spatial]``.

    Raises:
        ValueError: On a name outside :data:`ENSEMBLE_LOSSES`.
    """
    name = finetuning_config.get('loss', 'almost_fair_crps')
    if name not in ENSEMBLE_LOSSES:
        raise ValueError(f'Unknown fine-tuning loss "{name}"; expected one of {ENSEMBLE_LOSSES}.')
    if name == 'crps':
        return crps
    if name == 'almost_fair_crps':
        return almost_fair_crps
    beta = float(finetuning_config['beta'])
    return lambda samples, target: afcrps_psd(samples, target, beta=beta)
