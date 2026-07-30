"""Verification scores for rare-event map regression and occurrence classification.

Two metric families address the two evaluation challenges of the project (see config/eval/metrics.yaml):

Imbalance-aware (challenge A):
- occurrence-conditional and intensity-stratified continuous errors;
- categorical exceedance scores from pooled contingency counts, including the chance-corrected ETS and the
  base-rate-robust SEDI (non-degenerate as the base rate -> 0), plus the frequency bias (> 1 = conservative);
- threshold-free discrimination (ROC-AUC and PR-AUC) and the probabilistic occurrence scores (Brier, Dice,
  Bernoulli explained deviance);
- skill scores vs trivial baselines.

Structure-aware (challenge B):
- radially-averaged power spectral density (PSD) band ratios and the log-spectral distance (blur detectors);
- fractions skill score (FSS) over thresholds x neighborhood scales with the standard useful-scale criterion;
- sharpness and spatial-variance ratios.

Every score is computed in the TARGET SPACE. There is no target transform and no back-transform anywhere in this
project, so there is never a question of which space a tensor is in.

Deliberately absent (do not reintroduce):
- ``tweedie_deviance_score`` — parameterized for an unbounded zero-inflated target, so it misstates error on the
  bounded 0-24 one. The likelihood-based score for this project is ``explained_deviance``, on the binary head.
- ``uniform_histogram_ks`` — the PIT test. It pushed targets through a train-fitted gamma CDF whose parameters
  came from the deleted target-transform statistics.
- ``quantile_ratios`` / ``quantile_quantile`` — quantiles of the positive marginal collapse on a 0-24 integer
  target (the 0.99 and 0.999 quantiles land on the same hour value), which is also why the metric suite's
  thresholds are absolute hour bands.

All functions take plain numpy arrays; map stacks are ``[N, H, W]``.
"""
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

_EPS = 1e-12


# ----------------------------------------------------------------------------------------------------------------
# events and contingency tables
# ----------------------------------------------------------------------------------------------------------------
def exceedance(values: np.ndarray, threshold: float, strict: bool = False) -> np.ndarray:
    """Binary exceedance field: ``values > threshold`` if strict else ``values >= threshold``."""
    return values > threshold if strict else values >= threshold


def contingency_counts(
        pred: np.ndarray,
        obs: np.ndarray,
        threshold: float,
        strict: bool = False
) -> Tuple[float, float, float, float]:
    """Pooled (hits, misses, false alarms, correct negatives) at an exceedance threshold."""
    pred_event = exceedance(pred, threshold, strict)
    obs_event = exceedance(obs, threshold, strict)
    hits = float(np.sum(pred_event & obs_event))
    misses = float(np.sum(~pred_event & obs_event))
    false_alarms = float(np.sum(pred_event & ~obs_event))
    correct_negatives = float(np.sum(~pred_event & ~obs_event))
    return hits, misses, false_alarms, correct_negatives


def categorical_scores(hits: float, misses: float, false_alarms: float, correct_negatives: float) -> Dict[str, float]:
    """Categorical verification scores from a pooled contingency table.

    Returns:
        Dict with pod, far, csi, ets, hss, sedi and frequency_bias (NaN where undefined).
    """
    n = hits + misses + false_alarms + correct_negatives

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator > 0 else np.nan

    pod = safe_divide(hits, hits + misses)
    far = safe_divide(false_alarms, hits + false_alarms)
    csi = safe_divide(hits, hits + misses + false_alarms)
    frequency_bias = safe_divide(hits + false_alarms, hits + misses)

    hits_random = safe_divide((hits + misses) * (hits + false_alarms), n)
    ets = safe_divide(hits - hits_random, hits + misses + false_alarms - hits_random)

    hss_denominator = (hits + misses) * (misses + correct_negatives) \
        + (hits + false_alarms) * (false_alarms + correct_negatives)
    hss = safe_divide(2.0 * (hits * correct_negatives - misses * false_alarms), hss_denominator)

    # SEDI from the hit rate H and false alarm RATE F, both clipped away from {0, 1}
    hit_rate = np.clip(safe_divide(hits, hits + misses), 1e-6, 1.0 - 1e-6)
    false_rate = np.clip(safe_divide(false_alarms, false_alarms + correct_negatives), 1e-6, 1.0 - 1e-6)
    if np.isnan(hit_rate) or np.isnan(false_rate):
        sedi = np.nan
    else:
        log_f, log_h = np.log(false_rate), np.log(hit_rate)
        log_1f, log_1h = np.log(1.0 - false_rate), np.log(1.0 - hit_rate)
        sedi = (log_f - log_h - log_1f + log_1h) / (log_f + log_h + log_1f + log_1h)

    return {
        'pod': float(pod), 'far': float(far), 'csi': float(csi), 'ets': float(ets),
        'hss': float(hss), 'sedi': float(sedi), 'frequency_bias': float(frequency_bias)
    }


# ----------------------------------------------------------------------------------------------------------------
# continuous errors
# ----------------------------------------------------------------------------------------------------------------
def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean error (pred - obs); >= 0 indicates the preferred conservative behavior."""
    return float(np.mean(pred - obs))


def r2_score(pred: np.ndarray, obs: np.ndarray, condition: Optional[np.ndarray] = None) -> float:
    """Coefficient of determination ``1 - SS_res / SS_tot`` on the conditioning cells (default: all cells).

    SS_tot is the observed variance over the conditioning cells, so the score is the fraction of that variance
    explained by the predictions (it can go negative for predictions worse than the conditional mean — exactly
    the regime where an imbalance-exploiting model lands on an exceedance subgroup). NaN on an empty subgroup or
    one with no observed variance (e.g. a constant obs).

    Args:
        condition: Boolean mask of the cells to score (e.g. an observed-exceedance subgroup); defaults to all.
    """
    mask = np.ones(obs.shape, dtype=bool) if condition is None else condition
    if not mask.any():
        return float('nan')
    o = obs[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    ss_tot = float(np.sum((o - o.mean()) ** 2))
    if ss_tot <= _EPS:
        return float('nan')
    return float(1.0 - float(np.sum((o - p) ** 2)) / ss_tot)


def estimation_tendency(
        pred: np.ndarray,
        obs: np.ndarray,
        condition: Optional[np.ndarray] = None,
        tolerance: float = 0.0
) -> Dict[str, float]:
    """Proportions of under- and over-estimated cells among the conditioning cells (default: observed-positive).

    A cell is under-estimated when ``pred - obs < -tolerance`` and over-estimated when ``pred - obs > tolerance``;
    cells within +/- tolerance are counted as on-target. Complements the signed ``bias`` with the *direction*
    of the error mass: a conservative model should carry more over- than under-estimation on the event sets.

    Args:
        condition: Boolean mask of the cells to score (e.g. an observed-exceedance subgroup); defaults to obs > 0.
        tolerance: Absolute dead-band around zero error counted as on-target (in target units).

    Returns:
        Dict with ``under``, ``over`` and ``on_target`` proportions (NaN on an empty subgroup).
    """
    mask = obs > 0 if condition is None else condition
    if not mask.any():
        return {'under': float('nan'), 'over': float('nan'), 'on_target': float('nan')}
    diff = pred[mask].astype(np.float64) - obs[mask].astype(np.float64)
    n = diff.size
    under = float(np.count_nonzero(diff < -tolerance) / n)
    over = float(np.count_nonzero(diff > tolerance) / n)
    return {'under': under, 'over': over, 'on_target': float(1.0 - under - over)}


def rank_correlation(
        pred: np.ndarray,
        obs: np.ndarray,
        condition: Optional[np.ndarray] = None,
        method: str = 'spearman',
        max_samples: int = 2_000_000
) -> float:
    """Rank (ordering) agreement between predictions and observations on the conditioning cells.

    Measures whether the model orders cells the same way the observations do — robust to tail outliers. Evaluate
    it within obs subgroups (the zero mass makes the all-cells coefficient near-degenerate through ties). NaN on
    fewer than two cells or a constant side.

    Args:
        condition: Boolean mask of the cells to score (e.g. an observed-exceedance subgroup); defaults to obs > 0.
        method: ``spearman`` (default; rank Pearson, O(n log n)) or ``kendall`` (concordance; far costlier).
        max_samples: Random-subsample cap for very large pixel populations (seeded, reproducible).
    """
    from scipy.stats import kendalltau, spearmanr

    mask = obs > 0 if condition is None else condition
    if int(np.count_nonzero(mask)) < 2:
        return float('nan')
    p = pred[mask].astype(np.float64)
    o = obs[mask].astype(np.float64)
    if p.size > max_samples:
        chosen = np.random.default_rng(0).choice(p.size, size=max_samples, replace=False)
        p, o = p[chosen], o[chosen]
    if np.ptp(p) == 0 or np.ptp(o) == 0:                # a constant side has no ordering to agree with
        return float('nan')
    correlation = kendalltau(p, o).correlation if method == 'kendall' else spearmanr(p, o).correlation
    return float(correlation)


def conditional_error(
        pred: np.ndarray,
        obs: np.ndarray,
        kind: str = 'mae',
        condition: Optional[np.ndarray] = None
) -> float:
    """RMSE/MAE restricted to the conditioning cells (where a trivially-zero predictor collapses).

    Args:
        condition: Boolean mask of the cells to score (e.g. the evaluation-side occurrence event);
            defaults to the observed-positive cells ``obs > 0``.
    """
    positive = obs > 0 if condition is None else condition
    if not positive.any():
        return float('nan')
    return mae(pred[positive], obs[positive]) if kind == 'mae' else rmse(pred[positive], obs[positive])


def stratified_mae(pred: np.ndarray, obs: np.ndarray, bin_edges: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    """MAE within observed-intensity bins delimited by named ascending edges (last bin is open above).

    Args:
        bin_edges: Sequence of (name, value) pairs. Under the current metric suite these are the ABSOLUTE hour
            bands, e.g. [('occurrence', 0), ('h3', 3.0), ('h6', 6.0), ('h12', 12.0)] — they used to be quantiles
            of the positive marginal, which collapse on a bounded integer target. Bin i collects cells with obs in
            (edge_i, edge_{i+1}]; the first edge is exclusive (occurrence = obs > 0).

    Returns:
        Dict ``mae_bin_<name_i>_<name_i+1>`` (and ``mae_bin_<last>_inf``) -> MAE.
    """
    results = {}
    for i, (name, lower) in enumerate(bin_edges):
        if i + 1 < len(bin_edges):
            upper_name, upper = bin_edges[i + 1]
            in_bin = (obs > lower) & (obs <= upper)
            key = f'mae_bin_{name}_{upper_name}'
        else:
            in_bin = obs > lower
            key = f'mae_bin_{name}_inf'
        results[key] = mae(pred[in_bin], obs[in_bin]) if in_bin.any() else float('nan')
    return results


def skill_score(model_error: float, baseline_error: float) -> float:
    """Generic reduction-of-error skill score ``1 - e_model / e_baseline`` (positive = beats the baseline)."""
    if not np.isfinite(baseline_error) or baseline_error <= 0:
        return float('nan')
    return float(1.0 - model_error / baseline_error)


# ----------------------------------------------------------------------------------------------------------------
# neighborhood and spectral structure scores
# ----------------------------------------------------------------------------------------------------------------
def _fractions_skill(
        pred_event: np.ndarray,
        obs_event: np.ndarray,
        scale: int
) -> Tuple[float, float]:
    """FSS numerator/denominator contributions of ONE map pair at one neighborhood scale.

    Split out of :func:`fss` so that :func:`fss_useful_scale` can compute the binary exceedance fields once per
    map and reuse them across every scale, instead of re-thresholding the map for each scale.
    """
    pred_fraction = uniform_filter(pred_event.astype(np.float64), size=scale, mode='constant', cval=0.0)
    obs_fraction = uniform_filter(obs_event.astype(np.float64), size=scale, mode='constant', cval=0.0)
    return (
        float(np.sum((pred_fraction - obs_fraction) ** 2)),
        float(np.sum(pred_fraction ** 2 + obs_fraction ** 2))
    )


def fss(
        pred: np.ndarray,
        obs: np.ndarray,
        threshold: float,
        scale: int,
        strict: bool = False,
        progress: Optional[Callable[[], None]] = None
) -> float:
    """Fractions skill score at one exceedance threshold and neighborhood scale, aggregated over time.

    Fractions are neighborhood means of the binary exceedance fields; the numerator/denominator are accumulated
    over all maps before the final ratio (the standard aggregation for multi-day verification).

    Args:
        pred, obs: Map stacks ``[N, H, W]``.
        threshold: Exceedance threshold.
        scale: Side (in pixels) of the square neighborhood; 1 reduces to gridpoint verification.
        strict: Strict (>) vs non-strict (>=) exceedance.
        progress: Optional no-arg callback invoked once per map (drives an evaluation-stage progress bar over the
            pooled ensemble stack); ``None`` disables it.
    """
    numerator, denominator = 0.0, 0.0
    for index in range(pred.shape[0]):
        map_numerator, map_denominator = _fractions_skill(
            exceedance(pred[index], threshold, strict), exceedance(obs[index], threshold, strict), scale
        )
        numerator += map_numerator
        denominator += map_denominator
        if progress is not None:
            progress()
    if denominator <= 0:
        return float('nan')
    return float(1.0 - numerator / denominator)


def fss_useful_scale(
        pred: np.ndarray,
        obs: np.ndarray,
        threshold: float,
        scales: Sequence[int],
        strict: bool = False,
        progress: Optional[Callable[[], None]] = None
) -> Tuple[float, Dict[int, float]]:
    """Smallest scale whose FSS exceeds the useful-skill criterion ``0.5 + base_rate / 2``, plus every scale's FSS.

    Every requested scale is computed in ONE pass over the stack: the binary exceedance fields are derived once
    per map and reused across scales, where calling :func:`fss` per scale would re-threshold the whole stack for
    each one. The returned per-scale dict is what the evaluation suite emits as ``fss_<threshold>_s<scale>``, so
    the useful scale costs nothing beyond those values.

    ``progress`` still ticks once per (map, scale) pair, so an evaluation-stage progress bar sized as
    ``n_thresholds * n_scales * n_maps`` stays exact.

    Returns:
        Tuple (useful scale or NaN if never reached, dict scale -> FSS).
    """
    scales = [int(scale) for scale in scales]
    numerators = {scale: 0.0 for scale in scales}
    denominators = {scale: 0.0 for scale in scales}
    for index in range(pred.shape[0]):
        pred_event = exceedance(pred[index], threshold, strict)
        obs_event = exceedance(obs[index], threshold, strict)
        for scale in scales:
            map_numerator, map_denominator = _fractions_skill(pred_event, obs_event, scale)
            numerators[scale] += map_numerator
            denominators[scale] += map_denominator
            if progress is not None:
                progress()

    by_scale = {
        scale: (float(1.0 - numerators[scale] / denominators[scale]) if denominators[scale] > 0 else float('nan'))
        for scale in scales
    }

    base_rate = float(np.mean(exceedance(obs, threshold, strict)))
    target_skill = 0.5 + base_rate / 2.0
    for scale in sorted(by_scale):
        if np.isfinite(by_scale[scale]) and by_scale[scale] >= target_skill:
            return float(scale), by_scale
    return float('nan'), by_scale


def mean_power_spectrum(fields: np.ndarray, progress: Optional[Callable[[], None]] = None) -> np.ndarray:
    """Mean 2-D power spectrum ``|FFT|^2`` over a stack of maps ``[N, H, W]`` (``progress`` ticks once per map)."""
    spectrum = np.zeros(fields.shape[-2:], dtype=np.float64)
    for index in range(fields.shape[0]):
        spectrum += np.abs(np.fft.fft2(fields[index])) ** 2
        if progress is not None:
            progress()
    return spectrum / max(fields.shape[0], 1)


def _wavelength_grid(height: int, width: int) -> np.ndarray:
    """Per-Fourier-coefficient wavelength in pixels (inf at the DC component)."""
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.fftfreq(width)[None, :]
    radial_frequency = np.sqrt(kx ** 2 + ky ** 2)
    with np.errstate(divide='ignore'):
        return np.where(radial_frequency > 0, 1.0 / radial_frequency, np.inf)


def radial_psd(
        fields: np.ndarray,
        num_bins: Optional[int] = None,
        progress: Optional[Callable[[], None]] = None,
        spectrum: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectral density of a stack of maps.

    Args:
        spectrum: Optional precomputed ``mean_power_spectrum(fields)`` to reuse (skips the expensive FFT loop);
            ``None`` computes it. The caller must pass the spectrum of *these* ``fields`` — it is only used for the
            radial binning, whose grid is taken from ``fields.shape``.

    Returns:
        Tuple (wavelengths in pixels per bin center, mean power per bin), high frequencies last.
    """
    height, width = fields.shape[-2:]
    if spectrum is None:
        spectrum = mean_power_spectrum(fields, progress=progress)
    radial_frequency = 1.0 / _wavelength_grid(height, width)

    num_bins = num_bins or (min(height, width) // 2)
    edges = np.linspace(0.0, 0.5, num_bins + 1)
    centers, power = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (radial_frequency > low) & (radial_frequency <= high)
        if mask.any():
            centers.append(1.0 / (0.5 * (low + high)))
            power.append(float(spectrum[mask].mean()))
    return np.array(centers), np.array(power)


def radial_psd_per_map(
        fields: np.ndarray,
        num_bins: Optional[int] = None,
        progress: Optional[Callable[[], None]] = None,
        return_mean_spectrum: bool = False
):
    """Radially-averaged PSD of EACH map in a stack ``[N, H, W]`` (one row per map), for an ensemble-spread band.

    The radial bins are identical to :func:`radial_psd`, so the per-map rows share its wavelength axis and their
    column mean reproduces ``radial_psd(fields)`` exactly. ``progress`` ticks once per map.

    Args:
        return_mean_spectrum: also return the mean 2-D ``|FFT|^2`` (identical to :func:`mean_power_spectrum`), so a
            caller that needs BOTH the per-map curves (for the band) and the cached 2-D spectrum (for the band-ratio
            / log-spectral-distance scalars) gets them from a SINGLE FFT pass.

    Returns:
        ``(wavelengths, per_map[N, n_bins])``, or ``(wavelengths, per_map, mean_spectrum[H, W])`` when
        ``return_mean_spectrum`` is set; high frequencies last (matching :func:`radial_psd`).
    """
    height, width = fields.shape[-2:]
    radial_frequency = 1.0 / _wavelength_grid(height, width)
    num_bins = num_bins or (min(height, width) // 2)
    edges = np.linspace(0.0, 0.5, num_bins + 1)
    masks, centers = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (radial_frequency > low) & (radial_frequency <= high)
        if mask.any():
            masks.append(mask)
            centers.append(1.0 / (0.5 * (low + high)))

    per_map = np.empty((fields.shape[0], len(masks)), dtype=np.float64)
    mean_spectrum = np.zeros((height, width), dtype=np.float64) if return_mean_spectrum else None
    for index in range(fields.shape[0]):
        spectrum = np.abs(np.fft.fft2(fields[index])) ** 2
        if mean_spectrum is not None:
            mean_spectrum += spectrum
        per_map[index] = [spectrum[mask].mean() for mask in masks]
        if progress is not None:
            progress()

    if return_mean_spectrum:
        return np.array(centers), per_map, mean_spectrum / max(fields.shape[0], 1)
    return np.array(centers), per_map


def psd_band_ratios(
        pred: np.ndarray,
        obs: np.ndarray,
        bands: Dict[str, Sequence[float]],
        progress: Optional[Callable[[], None]] = None,
        pred_spectrum: Optional[np.ndarray] = None,
        obs_spectrum: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Predicted/observed mean spectral power ratio within wavelength bands (in pixels).

    Args:
        bands: Dict band name -> (lower wavelength, upper wavelength), upper possibly inf; the DC component is
            always excluded. The ``full`` band [2, inf] is low+mid+high combined.
        progress: Optional no-arg callback invoked once per map of each stack (pred then obs).
        pred_spectrum, obs_spectrum: Optional precomputed ``mean_power_spectrum`` of ``pred`` / ``obs`` to reuse
            (skips the FFT loop); ``None`` computes them.

    Returns:
        Dict band name -> power ratio (1 = structure-faithful, -> 0 in the high band = over-smoothed).
    """
    wavelengths = _wavelength_grid(*pred.shape[-2:])
    if pred_spectrum is None:
        pred_spectrum = mean_power_spectrum(pred, progress=progress)
    if obs_spectrum is None:
        obs_spectrum = mean_power_spectrum(obs, progress=progress)

    ratios = {}
    for name, (low, high) in bands.items():
        mask = (wavelengths >= low) & (wavelengths < high) & np.isfinite(wavelengths)
        obs_power = float(obs_spectrum[mask].mean()) if mask.any() else 0.0
        pred_power = float(pred_spectrum[mask].mean()) if mask.any() else 0.0
        ratios[name] = pred_power / obs_power if obs_power > _EPS else float('nan')
    return ratios


def psd_fidelity(ratio: float) -> float:
    """Scalar fidelity ``clip(1 - |1 - ratio|, 0, 1)`` of a spectral band power ratio.

    ONE function serves every band: the metric keys ``psd_full_fidelity`` and ``psd_high_fidelity`` are produced by
    the evaluation suite passing this the corresponding band's ratio (the ``band:`` argument of each
    ``psd_*_fidelity`` entry in metrics.yaml). There is deliberately no separate ``psd_full_fidelity`` function.
    """
    if not np.isfinite(ratio):
        return 0.0
    return float(np.clip(1.0 - abs(1.0 - ratio), 0.0, 1.0))


def log_spectral_distance(
        pred: np.ndarray,
        obs: np.ndarray,
        progress: Optional[Callable[[], None]] = None,
        pred_spectrum: Optional[np.ndarray] = None,
        obs_spectrum: Optional[np.ndarray] = None
) -> float:
    """RMS distance (in dB) between the radially-averaged log-spectra of predictions and observations
    (``pred_spectrum`` / ``obs_spectrum`` optionally reuse a precomputed ``mean_power_spectrum``)."""
    _, pred_power = radial_psd(pred, progress=progress, spectrum=pred_spectrum)
    _, obs_power = radial_psd(obs, progress=progress, spectrum=obs_spectrum)
    valid = (pred_power > _EPS) & (obs_power > _EPS)
    if not valid.any():
        return float('nan')
    log_ratio = 10.0 * np.log10(pred_power[valid] / obs_power[valid])
    return float(np.sqrt(np.mean(log_ratio ** 2)))


def sharpness_ratio(pred: np.ndarray, obs: np.ndarray) -> float:
    """Ratio of the pooled standard deviations of the spatial gradient magnitude (pred / obs); < 1 = blurry."""
    def gradient_std(fields):
        gy, gx = np.gradient(fields, axis=(-2, -1))
        return float(np.std(np.sqrt(gy ** 2 + gx ** 2)))

    obs_value = gradient_std(obs)
    return gradient_std(pred) / obs_value if obs_value > _EPS else float('nan')


def variance_ratio(pred: np.ndarray, obs: np.ndarray) -> float:
    """Ratio of the mean per-map spatial variances (pred / obs); < 1 = under-dispersed fields."""
    obs_value = float(np.mean(np.var(obs, axis=(-2, -1))))
    return float(np.mean(np.var(pred, axis=(-2, -1)))) / obs_value if obs_value > _EPS else float('nan')


# ----------------------------------------------------------------------------------------------------------------
# probabilistic / classification scores
# ----------------------------------------------------------------------------------------------------------------
def reliability_curve(
        probabilities: np.ndarray,
        occurrences: np.ndarray,
        bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability diagram data for the occurrence classifier.

    Returns:
        Tuple (mean forecast probability per bin, observed frequency per bin, cell counts per bin).
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.clip(np.digitize(probabilities, edges) - 1, 0, bins - 1)
    mean_probability = np.full(bins, np.nan)
    observed_frequency = np.full(bins, np.nan)
    counts = np.zeros(bins)
    for index in range(bins):
        in_bin = bin_index == index
        counts[index] = in_bin.sum()
        if counts[index] > 0:
            mean_probability[index] = float(probabilities[in_bin].mean())
            observed_frequency[index] = float(occurrences[in_bin].mean())
    return mean_probability, observed_frequency, counts


def brier_score(probabilities: np.ndarray, occurrences: np.ndarray) -> float:
    return float(np.mean((probabilities - occurrences.astype(np.float64)) ** 2))


def bernoulli_logloss(probabilities: np.ndarray, occurrences: np.ndarray, eps: float = 1e-7) -> float:
    """Mean Bernoulli negative log-likelihood (log loss) of probabilistic occurrence forecasts.

    Probabilities are clipped to ``[eps, 1 - eps]`` because the log loss is unbounded at a confident mistake: a
    single cell forecast at exactly 0 where the event occurred would otherwise make the whole score infinite.
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(occurrences, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def explained_deviance(
        probabilities: np.ndarray,
        occurrences: np.ndarray,
        baseline_probabilities: np.ndarray,
        eps: float = 1e-7
) -> float:
    """Bernoulli explained deviance of an occurrence forecast against a baseline: ``1 - logloss / logloss_base``.

    The likelihood-based counterpart of :func:`brier_score`'s quadratic skill score, and the reason it is scoped to
    the BINARY head: with the Tweedie/Poisson machinery gone, the bounded 0-24 target has no likelihood left to
    take a deviance of. 1 is a perfect forecast, 0 is baseline-equivalent, negative is worse than the baseline.

    Args:
        probabilities: Model occurrence probabilities.
        occurrences: Binary observed occurrence (0/1), same shape.
        baseline_probabilities: Baseline occurrence probabilities (the per-cell climatological frequency).
        eps: Probability clip, see :func:`bernoulli_logloss`.

    Returns:
        The explained deviance, or NaN when the baseline log loss is not usable as a denominator.
    """
    return skill_score(
        bernoulli_logloss(probabilities, occurrences, eps),
        bernoulli_logloss(baseline_probabilities, occurrences, eps)
    )


def dice_coefficient(pred_binary: np.ndarray, obs_binary: np.ndarray, smooth: float = 1.0) -> float:
    """Dice / F1 coefficient: 2*TP / (2*TP + FP + FN), pooled over all pixels and time steps.

    Args:
        pred_binary: Predicted binary map(s), any shape.
        obs_binary: Observed binary map(s), same shape.
        smooth: Laplace smoothing to avoid 0/0 on empty batches.

    Returns:
        Dice coefficient in [0, 1]; NaN if both pred and obs are all-zero (undefined overlap).
    """
    p = pred_binary.ravel().astype(np.float64)
    o = obs_binary.ravel().astype(np.float64)
    intersection = (p * o).sum()
    denominator = p.sum() + o.sum()
    if denominator == 0 and smooth == 0.0:
        return float('nan')
    return float((2.0 * intersection + smooth) / (denominator + smooth))


def _subsampled_ranking_inputs(
        ranking_score: np.ndarray,
        occurrences: np.ndarray,
        max_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten and (seeded) random-subsample a score/label pair, for the threshold-free ranking metrics."""
    ranking_score = np.asarray(ranking_score).ravel()
    occurrences = np.asarray(occurrences).ravel()
    if ranking_score.size > max_samples:
        chosen = np.random.default_rng(0).choice(ranking_score.size, size=max_samples, replace=False)
        ranking_score, occurrences = ranking_score[chosen], occurrences[chosen]
    return ranking_score, occurrences


def average_precision(probabilities: np.ndarray, occurrences: np.ndarray, max_samples: int = 2_000_000) -> float:
    """Average precision (area under the precision-recall curve; random-subsampled for large populations).

    THE discrimination measure at this base rate: precision-recall is computed from hits, misses and false alarms
    only, so it is not flattered by the huge correct-negative mass the way ROC-AUC is. NaN when the event never
    occurs (no positive class to rank).
    """
    from sklearn.metrics import average_precision_score

    probabilities, occurrences = _subsampled_ranking_inputs(probabilities, occurrences, max_samples)
    if occurrences.sum() == 0:
        return float('nan')
    return float(average_precision_score(occurrences.astype(int), probabilities))


def roc_auc(probabilities: np.ndarray, occurrences: np.ndarray, max_samples: int = 2_000_000) -> float:
    """Area under the ROC curve (random-subsampled for large populations).

    Reported ALONGSIDE :func:`average_precision`, not instead of it: ROC-AUC is the familiar cross-study number
    but is optimistic when negatives dominate, so at a ~0.07 % base rate a model with little practical skill can
    still score well. A high ``roc_auc`` next to a low ``average_precision`` is the signature of
    imbalance-exploitation, which is exactly why both are emitted.

    NaN when only one class is present (the ROC curve is undefined without both).
    """
    from sklearn.metrics import roc_auc_score

    probabilities, occurrences = _subsampled_ranking_inputs(probabilities, occurrences, max_samples)
    labels = occurrences.astype(int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float('nan')
    return float(roc_auc_score(labels, probabilities))


def roc_curve_points(
        probabilities: np.ndarray,
        occurrences: np.ndarray,
        max_samples: int = 2_000_000
) -> Tuple[np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate) of the ROC curve, for the ``roc_pr_curves`` report figure."""
    from sklearn.metrics import roc_curve

    probabilities, occurrences = _subsampled_ranking_inputs(probabilities, occurrences, max_samples)
    labels = occurrences.astype(int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return np.array([]), np.array([])
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    return false_positive_rate, true_positive_rate


def precision_recall_curve_points(
        probabilities: np.ndarray,
        occurrences: np.ndarray,
        max_samples: int = 2_000_000
) -> Tuple[np.ndarray, np.ndarray]:
    """(recall, precision) of the precision-recall curve, for the ``roc_pr_curves`` report figure."""
    from sklearn.metrics import precision_recall_curve

    probabilities, occurrences = _subsampled_ranking_inputs(probabilities, occurrences, max_samples)
    labels = occurrences.astype(int)
    if labels.sum() == 0:
        return np.array([]), np.array([])
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return recall, precision


# ----------------------------------------------------------------------------------------------------------------
# ensemble / probabilistic scores (stochastic-family ensemble evaluation)
# ----------------------------------------------------------------------------------------------------------------
# These operate on an ENSEMBLE of M map members stacked on axis 0: members ``[M, *cells]``, obs ``[*cells]``. Each
# helper returns either a finished scalar (convenience / unit tests) or summable PARTIALS, so the evaluation stage
# can STREAM the scores batch-by-batch over the held-out split without ever materializing the full
# ``[N_items, M, H, W]`` stack (prohibitive in hourly mode). Every helper accepts an optional boolean ``condition``
# mask (same shape as obs) restricting the score to a cell subset, e.g. the evaluation-side occurrence event (the
# zero mass otherwise dominates the rank histogram).
#
# ⚠️ The CRPS contract here is THE reference: ``crps_ensemble`` / ``almost_fair_crps_ensemble`` return a FLOAT and
# accept ``condition=``. The MC-dropout branch had same-named functions returning a per-element array — merging by
# name would have failed silently. src/utils/modeling/losses.py must agree with this module, not the other way round.


def _crps_terms(members: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell CRPS decomposition terms for an ensemble.

    ``CRPS(F, y) = E|X - y| - (1/2) E|X - X'|``; the spread term uses the O(M log M) order-statistic estimator
        ``(1/2) E|X - X'| = sum_k (2k - M + 1) * X_(k) / M^2``   (k = 0 .. M-1, X_(k) sorted ascending).

    Args:
        members: Ensemble members, shape ``[M, *cells]``.
        obs: Observations, shape ``[*cells]``.

    Returns:
        Tuple (mae_term, spread_term) of per-cell arrays (shape ``[*cells]``); ``spread_term = (1/2) E|X - X'|``.
    """
    members = members.astype(np.float64)
    obs = obs.astype(np.float64)
    m = members.shape[0]
    mae_term = np.abs(members - obs[None]).mean(axis=0)

    sorted_members = np.sort(members, axis=0)
    k = np.arange(m, dtype=np.float64).reshape((m,) + (1,) * obs.ndim)
    spread_term = ((2.0 * k - m + 1.0) * sorted_members).sum(axis=0) / (m * m)
    return mae_term, spread_term


def crps_sums(
        members: np.ndarray, obs: np.ndarray, condition: Optional[np.ndarray] = None
) -> Tuple[float, float, int]:
    """Summed (fair, almost-fair) CRPS over the scored cells plus the cell count — the streaming partials.

    The almost-fair correction (Ferro 2014) replaces the spread factor ``1/2`` by ``(M-1)/(2M)`` (equivalently a
    ``(M-1)/M`` factor on ``spread_term``), removing the negative bias of the finite-ensemble spread estimator;
    it matters most for small M.

    Returns SUMS, not means, because the full ``[N, M, H, W]`` stack does not fit in memory: sums are additive
    across batches, means and ratios are not. Divide exactly ONCE, at the end, in ``finalize_ensemble_metrics``.

    Returns:
        Tuple (crps_sum, almost_fair_crps_sum, n_cells) summed over the scored cells.
    """
    m = members.shape[0]
    mae_term, spread_term = _crps_terms(members, obs)
    crps = mae_term - spread_term
    almost_fair = mae_term - ((m - 1.0) / m) * spread_term
    if condition is not None:
        crps, almost_fair = crps[condition], almost_fair[condition]
    return float(crps.sum()), float(almost_fair.sum()), int(crps.size)


def crps_ensemble(members: np.ndarray, obs: np.ndarray, condition: Optional[np.ndarray] = None) -> float:
    """Fair CRPS averaged over the scored cells (scalar convenience wrapper over :func:`crps_sums`)."""
    crps_sum, _, n = crps_sums(members, obs, condition)
    return crps_sum / n if n else float('nan')


def almost_fair_crps_ensemble(members: np.ndarray, obs: np.ndarray, condition: Optional[np.ndarray] = None) -> float:
    """Almost-fair (bias-corrected) CRPS averaged over the scored cells."""
    _, af_sum, n = crps_sums(members, obs, condition)
    return af_sum / n if n else float('nan')


def spread_skill_sums(
        members: np.ndarray, obs: np.ndarray, condition: Optional[np.ndarray] = None
) -> Tuple[float, float, int]:
    """Summed ensemble variance and squared error of the ensemble mean over the scored cells (streaming partials).

    The spread-skill ratio is ``sqrt(mean ensemble variance) / sqrt(mean squared error of the ensemble mean)``: a
    well-calibrated ensemble has ratio ~ 1; < 1 is under-dispersed (over-confident), > 1 over-dispersed.

    ⚠️ The per-cell variance uses ``ddof=1``, so an ensemble of ONE member yields NaN rather than an error. That is
    why ``ensemble-size`` must always be >= 2, smoke configs included.

    Returns:
        Tuple (variance_sum, squared_error_sum, n_cells) over the scored cells; the per-cell variance is the
        sample variance (ddof=1, matching the torch ``var(dim=0)`` of the MC-dropout reference).
    """
    members = members.astype(np.float64)
    obs = obs.astype(np.float64)
    variance = members.var(axis=0, ddof=1)                          # per-cell ensemble (sample) variance
    squared_error = (members.mean(axis=0) - obs) ** 2               # squared error of the ensemble mean
    if condition is not None:
        variance, squared_error = variance[condition], squared_error[condition]
    return float(variance.sum()), float(squared_error.sum()), int(variance.size)


def spread_skill_ratio(members: np.ndarray, obs: np.ndarray, condition: Optional[np.ndarray] = None) -> float:
    """Ensemble spread-skill ratio ``sqrt(mean variance) / sqrt(mean squared error of the mean)`` (scalar)."""
    variance_sum, squared_error_sum, n = spread_skill_sums(members, obs, condition)
    if n == 0:
        return float('nan')
    spread = np.sqrt(variance_sum / n)
    skill_rmse = np.sqrt(squared_error_sum / n)
    return float(spread / skill_rmse) if skill_rmse > _EPS else float('nan')


def rank_histogram_counts(
        members: np.ndarray,
        obs: np.ndarray,
        condition: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Talagrand rank histogram counts: bin ``i`` counts cells whose observation has rank ``i`` among the members.

    The rank is the number of members strictly below the observation, with TIES broken at random (the standard
    treatment; otherwise the huge mass of all-zero cells, where the observation ties every zero member, would pile
    into bin 0 and the diagram would be meaningless on this sparse target). A calibrated ensemble gives a flat
    histogram over the M+1 bins; a U-shape signals under-dispersion, a dome over-dispersion, a slope a bias.

    Args:
        members: Ensemble members, shape ``[M, *cells]``.
        obs: Observations, shape ``[*cells]``.
        condition: Optional boolean mask of cells to score (e.g. the occurrence event).
        rng: numpy Generator for the tie-breaking (seeded by the caller for reproducibility); defaults to seed 0.

    Returns:
        Integer counts array of length M+1.
    """
    m = members.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    obs_broadcast = obs[None]
    below = (members < obs_broadcast).sum(axis=0)                   # members strictly below obs -> minimum rank
    equal = (members == obs_broadcast).sum(axis=0)                  # tied members -> spread the rank uniformly
    if condition is not None:
        below, equal = below[condition], equal[condition]
    # randomized rank in [below, below + equal]: insert obs at a uniformly random position among its ties
    rank = below + (rng.random(below.shape) * (equal + 1)).astype(np.int64)
    rank = np.clip(rank, 0, m)
    return np.bincount(rank.ravel(), minlength=m + 1).astype(np.int64)


def rank_histogram_reliability(counts: np.ndarray) -> float:
    """Scalar flatness of a rank histogram: ``sum_i |c_i / N - 1/(M+1)|`` (0 = perfectly flat / calibrated).

    A single number summarising the Talagrand diagram for the metrics JSON (the full counts go to the report
    figure); larger means a more U-shaped / domed / sloped — i.e. mis-calibrated — ensemble.
    """
    total = counts.sum()
    if total == 0:
        return float('nan')
    frequencies = counts.astype(np.float64) / total
    uniform = 1.0 / counts.size
    return float(np.abs(frequencies - uniform).sum())


def ensemble_partials(
        members: np.ndarray,
        obs: np.ndarray,
        occurrence_event: Optional[Tuple[float, bool]] = None,
        rng: Optional[np.random.Generator] = None
) -> Dict[str, object]:
    """Bundle one batch's summable ensemble-metric partials (the streaming unit of the ensemble suite).

    Computes the CRPS / almost-fair CRPS sums (over all cells AND restricted to the occurrence event, a
    tail-focused variant), the spread/skill sums, and the occurrence-conditioned rank histogram counts. The
    caller sums these dicts across batches (scalars add, ``rank_counts`` arrays add elementwise) and passes the
    total to :func:`src.utils.metrics.evaluation.finalize_ensemble_metrics`.

    This is the SHARED accumulator: MC-dropout reaches it through ``MCDropoutEnsembleModule`` rather than through
    a bespoke accumulator of its own, which is what keeps the two stochastic families' numbers comparable.

    Args:
        members: Ensemble members for the batch, shape ``[M, B, H, W]``.
        obs: Observations for the batch, shape ``[B, H, W]``.
        occurrence_event: ``(value, strict)`` restricting the tail-conditional CRPS and the rank histogram to the
            evaluation-side occurrence cells; ``None`` falls back to ``obs > 0``.
        rng: numpy Generator for the rank-histogram tie-breaking (seeded by the caller for reproducibility).

    Returns:
        Dict of summable partials (keys: ``crps_sum``, ``af_crps_sum``, ``crps_n`` and their ``_occ`` variants,
        ``var_sum``, ``sqerr_sum``, ``ss_n``, ``rank_counts``).
    """
    if occurrence_event is None:
        occurrence = obs > 0
    else:
        value, strict = occurrence_event
        occurrence = exceedance(obs, value, strict)

    crps_sum, af_crps_sum, crps_n = crps_sums(members, obs)
    crps_sum_occ, af_crps_sum_occ, crps_n_occ = crps_sums(members, obs, condition=occurrence)
    var_sum, sqerr_sum, ss_n = spread_skill_sums(members, obs)
    rank_counts = rank_histogram_counts(members, obs, condition=occurrence, rng=rng)
    return {
        'crps_sum': crps_sum, 'af_crps_sum': af_crps_sum, 'crps_n': crps_n,
        'crps_sum_occ': crps_sum_occ, 'af_crps_sum_occ': af_crps_sum_occ, 'crps_n_occ': crps_n_occ,
        'var_sum': var_sum, 'sqerr_sum': sqerr_sum, 'ss_n': ss_n,
        'rank_counts': rank_counts,
    }
