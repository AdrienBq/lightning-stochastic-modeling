"""Residual-space diagnostics for the diffusion model in RESIDUAL (knowledge-guided) mode.

The evaluation suite (scores.py / evaluation.py) scores everything in the ORIGINAL target space. These diagnostics
are a complementary sanity check on the learnt DISCREPANCY model: do the residuals the diffusion model predicts
look like the true discrepancy ``observed - upstream``? They are computed ONLY for a residual-mode diffusion
ensemble run and use the model's UNCLAMPED predicted residual (``r = P_unclamped - upstream``; the eval stage gets
it from DiffusionModule via ``eval_return_residual``), never the censored ``clamp(P) - upstream``.

Notation (all per-cell unless noted):
  * ``O`` observed target, ``U`` upstream prediction, ``P`` clamped ensemble-mean prediction (target space);
  * ``D_pred = r`` the model's predicted discrepancy (ensemble mean of the unclamped residual), ``[N, H, W]``;
  * ``D_pred_members`` the per-member unclamped residual ``[N, M, H, W]`` (pooled for the marginal hist/QQ);
  * ``D_true = O - U`` the true discrepancy, ``[N, H, W]``.

Returns ``(flat, curves)``: flat ``resid_*`` scalars (merged into the metrics JSON, every one NON-redundant with
the target-space suite) and a ``curves['residual']`` payload of compact arrays the report renders (bias / surprise
maps, overlaid histograms + KDE, residual QQ, scatters, heteroscedasticity curves). Point quantities use the
ensemble MEAN; the marginal-distribution diagnostics (histograms, QQ) pool the ensemble members.
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-9                          # magnitudes at/below this count as "zero" for the surprise 0/0 and +/-inf cases
# surprise-map category codes (shared by the magnitude and direction panes)
SURPRISE_FINITE, SURPRISE_OVER, SURPRISE_UNDER = 0, 1, -1     # finite log/ratio, +inf "overcorrected", -inf "failed"


def _occurrence_mask(observation: np.ndarray, occurrence_event: Tuple[float, bool]) -> np.ndarray:
    """Boolean mask of the occurrence (positive-target) cells, per the evaluation-side occurrence event."""
    value, strict = occurrence_event
    return observation > value if strict else observation >= value


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two finite-aligned 1-D arrays (NaN if degenerate)."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float('nan')
    x, y = x[finite], y[finite]
    if x.std() < _EPS or y.std() < _EPS:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks), scipy-free."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float('nan')
    from scipy.stats import rankdata
    return _pearson(rankdata(x[finite]), rankdata(y[finite]))


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of the ordinary least-squares fit y ~ a + b x (NaN if degenerate)."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2 or x[finite].std() < _EPS:
        return float('nan')
    return float(np.polyfit(x[finite], y[finite], 1)[0])


def _subsample(rng: np.random.Generator, n: int, cap: int) -> np.ndarray:
    """Indices for a uniform subsample of ``n`` items capped at ``cap`` (all of them when n <= cap)."""
    return np.arange(n) if n <= cap else rng.choice(n, size=cap, replace=False)


def _surprise_pane(numerator: np.ndarray, denominator: np.ndarray, signed: bool) -> Dict[str, np.ndarray]:
    """One surprise pane on the ratio ``numerator / denominator`` with the 0/0 = 0 and +/-inf conventions.

    Two cases (the magnitude and direction panes are structurally different):

    * ``signed=False`` — MAGNITUDE pane, ``num, den >= 0`` (e.g. ``mean|D_pred|`` over ``mean|D_true|``). Finite
      cells get ``log(num/den)``; ``num>0, den=0`` is +inf "overcorrected" (corrects where nothing is needed);
      ``num=0, den>0`` is -inf "failed" (no correction where one is needed); ``num=den=0`` is 0 (white).
    * ``signed=True`` — DIRECTION pane, signed ``num, den`` (e.g. ``mean sign(D_pred)`` over ``mean sign(D_true)``).
      Finite cells get the raw ratio ``num/den`` (den != 0); a directionless truth (``den=0``) with a net
      predicted direction is +inf "overcorrected" (``num>0``) or -inf "failed" (``num<0``); ``num=den=0`` is 0.

    Returns ``{'value': finite-value map (0 where inf / 0-0), 'category': code map}``."""
    num_zero = np.abs(numerator) <= _EPS
    den_zero = np.abs(denominator) <= _EPS
    category = np.zeros(numerator.shape, dtype=np.int8)
    value = np.zeros(numerator.shape, dtype=np.float64)
    if not signed:                                                   # magnitude: num, den >= 0
        over = (~num_zero) & den_zero                               # correction where none is needed
        under = num_zero & (~den_zero)                             # no correction where one is needed
        finite = (~num_zero) & (~den_zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            value[finite] = np.log(numerator[finite] / denominator[finite])
    else:                                                           # direction: signed num, den
        over = den_zero & (numerator > _EPS)                       # net predicted direction over a directionless truth
        under = den_zero & (numerator < -_EPS)
        finite = ~den_zero
        with np.errstate(divide='ignore', invalid='ignore'):
            value[finite] = numerator[finite] / denominator[finite]
    category[over] = SURPRISE_OVER
    category[under] = SURPRISE_UNDER
    return {'value': value, 'category': category}


def _surprise_scalars(prefix: str, pane: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Median/mean over finite cells + over-/under-correction cell fractions of a surprise pane."""
    category = pane['category']
    finite = category == SURPRISE_FINITE
    finite_values = pane['value'][finite]
    total = max(category.size, 1)
    return {
        f'{prefix}_median': float(np.median(finite_values)) if finite_values.size else float('nan'),
        f'{prefix}_mean': float(finite_values.mean()) if finite_values.size else float('nan'),
        f'{prefix}_overcorrect_frac': float((category == SURPRISE_OVER).sum() / total),
        f'{prefix}_failed_frac': float((category == SURPRISE_UNDER).sum() / total),
    }


def _decile_heteroscedasticity(error: np.ndarray, conditioning: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
    """Per-bin RMS of the correction error ``error = D_pred - D_true`` across equal-count bins (deciles by default)
    of the POSITIVE conditioning values — the heteroscedasticity curve. Returns bin centers + RMS per bin."""
    err = error.ravel()
    cond = conditioning.ravel()
    positive = np.isfinite(cond) & np.isfinite(err) & (cond > 0)
    err, cond = err[positive], cond[positive]
    if cond.size < n_bins:
        return {'bin_center': np.array([]), 'rms_error': np.array([])}
    edges = np.quantile(cond, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)                                          # collapse ties (a near-constant conditioner)
    centers, rms = [], []
    # half-open interior bins [low, high), the top bin closed [low, high]: a DISCRETE conditioner (the integer
    # target O) puts many cells exactly on a quantile edge, and a both-inclusive test would double-count them.
    n_intervals = len(edges) - 1
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (cond >= low) & (cond <= high) if index == n_intervals - 1 else (cond >= low) & (cond < high)
        if mask.any():
            centers.append(float(np.median(cond[mask])))
            rms.append(float(np.sqrt((err[mask] ** 2).mean())))
    return {'bin_center': np.array(centers), 'rms_error': np.array(rms)}


def _overlaid_histogram(pred: np.ndarray, true: np.ndarray, bins: int, rng: np.random.Generator,
                        kde_cap: int) -> Dict[str, np.ndarray]:
    """Shared-bin histogram densities of two samples + a Gaussian-KDE curve for each (on a subsample), for the
    superimposed report histogram. Robust extent (1st-99th percentile of the pooled sample) clips heavy tails."""
    pred = pred[np.isfinite(pred)]
    true = true[np.isfinite(true)]
    if pred.size == 0 or true.size == 0:
        return {}
    pooled = np.concatenate([pred, true])
    lo, hi = np.percentile(pooled, [1.0, 99.0])
    if hi - lo < _EPS:
        lo, hi = float(pooled.min()), float(pooled.max() + 1.0)
    edges = np.linspace(lo, hi, bins + 1)
    pred_density, _ = np.histogram(pred, bins=edges, density=True)
    true_density, _ = np.histogram(true, bins=edges, density=True)
    out = {'edges': edges, 'pred_density': pred_density, 'true_density': true_density}
    try:                                                             # KDE is optional (scipy); skip if it fails
        from scipy.stats import gaussian_kde
        grid = np.linspace(lo, hi, 256)
        pred_kde = gaussian_kde(pred[_subsample(rng, pred.size, kde_cap)])(grid)
        true_kde = gaussian_kde(true[_subsample(rng, true.size, kde_cap)])(grid)
        # assign all three together so a partial failure never leaves kde_grid without both curves (which would
        # crash the report's histogram builder)
        out['kde_grid'], out['pred_kde'], out['true_kde'] = grid, pred_kde, true_kde
    except Exception as error:
        logger.warning(f'residual diagnostics: KDE skipped ({error}).')
    return out


def residual_diagnostics(
        observation: np.ndarray,
        prediction: np.ndarray,
        upstream: np.ndarray,
        residual_mean: np.ndarray,
        residual_members: np.ndarray,
        occurrence_event: Tuple[float, bool] = (0.0, True),
        n_deciles: int = 10,
        pixel_sample: int = 200_000,
        hist_bins: int = 60,
        qq_points: int = 99,
        seed: int = 0
) -> Tuple[Dict[str, float], dict]:
    """Compute the residual-space diagnostics suite (see the module docstring).

    Args:
        observation, prediction, upstream, residual_mean: ``[N, H, W]`` — the observed target, the clamped
            ensemble-mean prediction, the upstream prediction, and the UNCLAMPED ensemble-mean residual ``D_pred``.
        residual_members: ``[N, M, H, W]`` — the per-member unclamped residual (pooled for the histograms/QQ).
        occurrence_event: evaluation-side occurrence event ``(value, strict)`` — the positive-target cells used for
            the original-vs-positive-only conditioning whenever a diagnostic is taken against ``O``.
        n_deciles, pixel_sample, hist_bins, qq_points, seed: binning / subsampling / RNG controls.

    Returns:
        ``(flat, curves)``: ``flat`` the ``resid_*`` scalar metrics, ``curves`` a dict with a ``'residual'`` block.
    """
    rng = np.random.default_rng(seed)
    O, P, U = observation, prediction, upstream
    D_pred = residual_mean                                            # unclamped predicted discrepancy (ens mean)
    D_true = O - U                                                   # true discrepancy
    error = D_pred - D_true                                          # correction error
    flat: Dict[str, float] = {}
    residual: dict = {}

    # --- 1) discrepancy bias map + scalars ---
    bias_map = D_pred.mean(axis=0)                                   # [H, W] mean predicted correction
    residual['bias_map'] = bias_map
    flat['resid_bias_mean'] = float(D_pred.mean())
    flat['resid_bias_spatial_rms'] = float(np.sqrt((bias_map ** 2).mean()))

    # --- 2) surprise maps (magnitude + direction) ---
    mag = _surprise_pane(np.abs(D_pred).mean(axis=0), np.abs(D_true).mean(axis=0), signed=False)
    direction = _surprise_pane(np.sign(D_pred).mean(axis=0), np.sign(D_true).mean(axis=0), signed=True)
    residual['surprise_magnitude'] = mag
    residual['surprise_direction'] = direction
    flat.update(_surprise_scalars('resid_surprise_mag', mag))
    flat.update(_surprise_scalars('resid_surprise_dir', direction))

    # --- 3) scale skill (correction magnitude vs the upstream error) + C1 upstream skill + C2 direction ---
    abs_true_mean = float(np.abs(D_true).mean())
    rms_true = float(np.sqrt((D_true ** 2).mean()))
    flat['resid_scale_absmean_ratio'] = float(np.abs(D_pred).mean() / abs_true_mean) if abs_true_mean > _EPS else float('nan')
    flat['resid_scale_rms_ratio'] = float(np.sqrt((D_pred ** 2).mean()) / rms_true) if rms_true > _EPS else float('nan')
    mse_up, mae_up = float(((U - O) ** 2).mean()), float(np.abs(U - O).mean())
    flat['resid_mse_skill_vs_upstream'] = 1.0 - float(((P - O) ** 2).mean()) / mse_up if mse_up > _EPS else float('nan')
    flat['resid_mae_skill_vs_upstream'] = 1.0 - float(np.abs(P - O).mean()) / mae_up if mae_up > _EPS else float('nan')
    flat['resid_dir_corr'] = _pearson(D_pred.ravel(), D_true.ravel())
    flat['resid_sign_agreement'] = float((np.sign(D_pred) == np.sign(D_true)).mean())

    # --- 4) histograms (pixel-wise member-pooled + per-image magnitudes) + distribution-distance scalars ---
    # strip non-finite values once (an unconverged ODE sample can be non-finite): np.quantile / KS / Wasserstein
    # would otherwise return NaN and silently corrupt resid_qq_* / resid_hist_*
    members_flat = residual_members.reshape(-1)
    members_flat = members_flat[np.isfinite(members_flat)]
    dtrue_flat = D_true.reshape(-1)
    dtrue_flat = dtrue_flat[np.isfinite(dtrue_flat)]
    residual['hist_pixel'] = _overlaid_histogram(members_flat, dtrue_flat, hist_bins, rng, kde_cap=pixel_sample)
    per_image = {
        'pred_mae': np.abs(D_pred).mean(axis=(-2, -1)), 'true_mae': np.abs(D_true).mean(axis=(-2, -1)),
        'pred_rms': np.sqrt((D_pred ** 2).mean(axis=(-2, -1))), 'true_rms': np.sqrt((D_true ** 2).mean(axis=(-2, -1))),
    }
    residual['hist_image'] = per_image
    pred_sub = members_flat[_subsample(rng, members_flat.size, pixel_sample)]
    true_sub = dtrue_flat[_subsample(rng, dtrue_flat.size, pixel_sample)]
    try:
        from scipy.stats import ks_2samp, wasserstein_distance
        flat['resid_hist_ks'] = float(ks_2samp(pred_sub, true_sub).statistic)
        flat['resid_hist_wasserstein'] = float(wasserstein_distance(pred_sub, true_sub))
    except Exception as exc:                                          # NB: don't name this `error` — Python clears the
        logger.warning(f'residual diagnostics: KS/Wasserstein skipped ({exc}).')  # `as` target, wiping the D_pred-D_true `error` array used below

    # --- 5) residual QQ (member-pooled D_pred quantiles vs D_true quantiles) ---
    if members_flat.size and dtrue_flat.size:                         # guard the all-non-finite degenerate case
        levels = np.linspace(0.01, 0.99, qq_points)
        pred_q, true_q = np.quantile(members_flat, levels), np.quantile(dtrue_flat, levels)
        residual['qq'] = {'levels': levels, 'pred': pred_q, 'true': true_q}
        flat['resid_qq_pearson'] = _pearson(true_q, pred_q)
        flat['resid_qq_slope'] = _ols_slope(true_q, pred_q)

    # --- 6) scatters: D_pred (ens mean) vs O / D_true / U, pixel (subsampled) + per-image, with correlations ---
    occ = _occurrence_mask(O, occurrence_event)
    pixel_idx = _subsample(rng, O.size, pixel_sample)
    dpred_pix = D_pred.ravel()[pixel_idx]
    image_pred = np.abs(D_pred).mean(axis=(-2, -1))                  # per-image |D_pred| (MAE-style)
    scatter: dict = {}
    for name, field, image_field in (
        ('obs', O, O.mean(axis=(-2, -1))),
        ('discrepancy', D_true, np.abs(D_true).mean(axis=(-2, -1))),
        ('upstream', U, U.mean(axis=(-2, -1))),
    ):
        field_pix = field.ravel()[pixel_idx]
        scatter[name] = {
            'pixel': {'x': field_pix, 'y': dpred_pix},
            'image': {'x': image_field, 'y': image_pred},
        }
        flat[f'resid_scatter_{name}_pixel_pearson'] = _pearson(field_pix, dpred_pix)
        flat[f'resid_scatter_{name}_pixel_spearman'] = _spearman(field_pix, dpred_pix)
        flat[f'resid_scatter_{name}_image_pearson'] = _pearson(image_field, image_pred)
    # against the observed target: also the positive-only (occurrence) version (pixel level)
    occ_idx = np.where(occ.ravel())[0]
    occ_idx = occ_idx[_subsample(rng, occ_idx.size, pixel_sample)]
    if occ_idx.size:
        ox, oy = O.ravel()[occ_idx], D_pred.ravel()[occ_idx]
        scatter['obs']['pixel_positive'] = {'x': ox, 'y': oy}
        flat['resid_scatter_obs_pixel_positive_pearson'] = _pearson(ox, oy)
        flat['resid_scatter_obs_pixel_positive_spearman'] = _spearman(ox, oy)
    residual['scatter'] = scatter

    # --- 7) heteroscedasticity: per-decile RMS of (D_pred - D_true) vs U and vs O (+ a single Spearman each) ---
    residual['heteroscedasticity'] = {
        'upstream': _decile_heteroscedasticity(error, U, n_deciles),
        'obs': _decile_heteroscedasticity(error, O, n_deciles),
    }
    flat['resid_hetero_upstream_spearman'] = _spearman(np.abs(error).ravel(), U.ravel())
    flat['resid_hetero_obs_spearman'] = _spearman(np.abs(error).ravel(), O.ravel())

    curves = {'residual': residual}
    return flat, curves
