"""Evaluation-suite runner: trivial baselines and the full metric suite of the active metrics config
(config/eval/metrics_daily.yaml for the daily task, config/eval/metrics_hourly.yaml for the hourly one).

ONE evaluation for all three model families. ``run_metric_suite`` is the single path; there is no family-specific
evaluation, which is what lets a deterministic U-net, an MC-dropout ensemble and a flow-matching model be put in
one table with identical metric-key columns.

Baselines (the yardsticks an imbalance-exploiting model cannot beat):
- ``zero``: the all-zero predictor;
- ``climatology``: per-cell day-of-year mean within a +/- window (daily mode) or per-cell month x hour-of-day
  mean (hourly mode), estimated on the train split.

There is deliberately NO ``persistence`` baseline. Persistence ("the previous time step's observation") presumes a
forecasting task with access to past observations; this project is a diagnostic ERA5 -> lightning parameterization
that sees reanalysis predictors only, so persistence is not an admissible competitor.

All metrics are computed in the TARGET SPACE — training space == evaluation space, since there is no target
transform and no back-transform anywhere.

Occurrence is ``target > 0``, unconditionally. Single-stroke observational-noise denoising lives entirely in the
PREPARATION stage (its ``hourly_threshold`` filters the hourly stroke counts before the target is assembled), so it
is baked into the stored target and any positive value has already survived it. There is no eval-side
``occurrence_threshold`` knob — it would be a second, competing definition of "this cell had lightning" —
and ``resolve_occurrence_event`` hard-asserts that none has reappeared.
"""
import logging
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io.data import MODE_HOURLY
from src.utils.metrics import scores

logger = logging.getLogger(__name__)

# Categorical entries that are NOT read off a contingency table: they need the observed EVENT but no cut on the
# prediction, so they are computed from the continuous field directly.
#   * RANKING_SCORES score the ordering of that field over all cells rather than at one decision cut.
#   * `dice` is soft Dice — the same formula as the hard coefficient, evaluated on the probability (see
#     scores.dice_coefficient). It is the eval-time complement of the `dice` training loss.
RANKING_SCORES = ('roc_auc', 'average_precision')
SOFT_SCORES = ('dice',)
THRESHOLD_FREE_SCORES = RANKING_SCORES + SOFT_SCORES


# ----------------------------------------------------------------------------------------------------------------
# threshold resolution
# ----------------------------------------------------------------------------------------------------------------
class EventThreshold(NamedTuple):
    """A resolved threshold, carrying the OBSERVATION side and the PREDICTION side separately.

    They are separate because the two tasks cut different sides. In the daily REGRESSION task prediction and
    observation are the same quantity in the same units, so both sides hold the same hour band — that is the
    symmetric case every ``kind: absolute`` entry produces. In the hourly CLASSIFICATION task the observation is
    already a 0/1 event while the prediction is a probability, so a shared level is meaningless: the prediction side
    is a DECISION threshold on the probability and the observation side just reads the labels.

    Every metric that conditions on *observed* intensity (the stratified MAE bins, the r2 and rank-correlation
    subgroups, FSS) uses the obs side. Only the contingency table uses both.
    """
    obs_value: float
    obs_strict: bool
    pred_value: float
    pred_strict: bool

    @property
    def obs_event(self) -> Tuple[float, bool]:
        return self.obs_value, self.obs_strict

    @property
    def is_symmetric(self) -> bool:
        return (self.obs_value, self.obs_strict) == (self.pred_value, self.pred_strict)


def resolve_occurrence_event(metrics_config: dict, target_stats: dict) -> Tuple[float, bool]:
    """Resolve the evaluation-side occurrence event into an exceedance spec ``(value, strict)``.

    The occurrence event is unconditionally ``target > 0`` -> ``(0.0, strict=True)``: a cell counts as having
    lightning when its (already noise-filtered) target is positive. Single-stroke observational-noise denoising
    lives entirely in the PREPARATION stage, so there is no eval-side stroke-count threshold any more. The legacy
    ``occurrence_threshold`` knob has been removed; a HARD ASSERT here fails loudly if a stray one ever reappears
    in the metrics config, guaranteeing no thresholding is applied unintentionally.

    ``target_stats`` is accepted for signature stability (callers still pass it) but is no longer consulted.
    """
    assert 'occurrence_threshold' not in metrics_config, (
        "Stray 'occurrence_threshold' in the metrics config: this eval-side knob was REMOVED. Hourly-count "
        "denoising now lives in the preparation stage (its hourly_threshold), and the occurrence event is "
        "unconditionally `target > 0`. Remove 'occurrence_threshold' from the metrics YAML."
    )
    return 0.0, True


def resolve_threshold(spec: dict, target_stats: dict, occurrence_event: Tuple[float, bool]) -> EventThreshold:
    """Resolve one named threshold spec from the metrics config into an :class:`EventThreshold`.

    Supported kinds:

    * ``absolute`` (the DEFAULT, and what the daily suite uses: the hour bands h3/h6/h12) — a plain value in target
      units, applied SYMMETRICALLY to both sides.
    * ``occurrence`` — the evaluation-side occurrence event on the OBSERVATION side, with an optional ``pred_value``
      giving the prediction side its own cut (``pred_strict`` optional, default inclusive).

      ⚠️ **Supply ``pred_value`` for any REGRESSION run.** The occurrence event is ``target > 0``, and applied
      symmetrically that asks whether the *predicted hours* are ``> 0`` — which a softplus/ReLU head satisfies at
      essentially every cell, so the contingency table degenerates to hits + false alarms with zero misses and zero
      correct negatives (POD = 1, frequency bias = 1/base-rate). ``pred_value: 1`` reads as "the model forecasts at
      least one lightning-hour", which is the same ``pred >= k`` rule the h3/h6/h12 bands already use, at ``k = 1``.
      Omitting it keeps the symmetric behaviour, which is right only where the prediction is genuinely 0/1.
    * ``probability`` — for the HOURLY CLASSIFICATION task: the prediction side is a decision threshold on the
      predicted probability (``value``, e.g. 0.5) and the observation side is the occurrence event, so the 0/1
      labels are read as they are rather than re-thresholded. An hourly pipeline's ``metrics.categorical.thresholds``
      must use this kind — see the warning in :func:`run_metric_suite` for what an ``occurrence`` entry would do to
      a probability field.
    * ``train_positive_quantile`` — a quantile of the positive train target marginal, symmetric. Kept because the
      resolver is generic, NOT because the suite uses it: quantile levels of the positive marginal collapse on the
      bounded 0-24 integer target (the 0.99 and 0.999 quantiles land on the same hour value), which is exactly why
      the suite names absolute levels instead.
    """
    kind = spec.get('kind', 'absolute')
    strict = bool(spec.get('strict', False))

    def symmetric(value: float) -> EventThreshold:
        return EventThreshold(obs_value=value, obs_strict=strict, pred_value=value, pred_strict=strict)

    if kind == 'occurrence':
        # the OBSERVATION side is always the occurrence event, so every obs-conditioned metric (the conditional
        # errors, the r2 / rank-correlation subgroups, FSS, the climatology Brier) is unaffected by `pred_value`
        if 'pred_value' in spec:
            return EventThreshold(occurrence_event[0], occurrence_event[1],
                                  float(spec['pred_value']), bool(spec.get('pred_strict', False)))
        return EventThreshold(occurrence_event[0], occurrence_event[1], occurrence_event[0], occurrence_event[1])
    if kind == 'absolute':
        return symmetric(float(spec['value']))
    if kind == 'probability':
        return EventThreshold(
            obs_value=occurrence_event[0], obs_strict=occurrence_event[1],
            pred_value=float(spec['value']), pred_strict=strict
        )
    if kind == 'train_positive_quantile':
        quantiles = target_stats['positive_quantiles']
        key = str(spec['value'])
        if key not in quantiles:
            # tolerate float-formatting differences between YAML and JSON keys (0.9 vs 0.90)
            key = str(float(spec['value']))
        return symmetric(float(quantiles[key]))
    raise ValueError(f'Unknown threshold kind "{kind}".')


def resolve_thresholds(metrics_config: dict, target_stats: dict) -> Dict[str, EventThreshold]:
    """Resolve every named threshold of the metric suite into an :class:`EventThreshold`."""
    occurrence_event = resolve_occurrence_event(metrics_config, target_stats)
    return {
        name: resolve_threshold(spec, target_stats, occurrence_event)
        for name, spec in metrics_config.get('thresholds', {}).items()
    }


# ----------------------------------------------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------------------------------------------
def _load_target(target_file: str) -> np.ndarray:
    return np.load(target_file).astype(np.float32)


def _climatology_tables(
        train_index: pd.DataFrame,
        mode: str,
        hours_per_day: int,
        window_days: int,
        occurrence_event: Tuple[float, bool]
):
    """Accumulate train-split climatological means: per day-of-year (daily mode, smoothed circularly over the
    window) or per (month, hour-of-day) (hourly mode). Also accumulates frequencies of the evaluation-side
    occurrence event (the Brier / explained-deviance baseline). Returns lookup_fn(date, hour) -> (mean_map,
    occurrence_map)."""
    first = _load_target(train_index.iloc[0]['target_file'])
    grid_shape = first.shape[-2:]
    occurrence_value, occurrence_strict = occurrence_event

    if mode == MODE_HOURLY:
        sums = np.zeros((12, hours_per_day) + grid_shape)
        occurrence_sums = np.zeros_like(sums)
        counts = np.zeros((12, hours_per_day))
        for _, row in train_index.iterrows():
            target = _load_target(row['target_file'])                       # [T, H, W]
            month = row['date'].month - 1
            for hour in range(min(hours_per_day, target.shape[0])):
                sums[month, hour] += target[hour]
                occurrence_sums[month, hour] += scores.exceedance(target[hour], occurrence_value, occurrence_strict)
                counts[month, hour] += 1
        safe_counts = np.maximum(counts, 1)[..., None, None]
        means, frequencies = sums / safe_counts, occurrence_sums / safe_counts

        def lookup(date: pd.Timestamp, hour: Optional[int]):
            return means[date.month - 1, int(hour)], frequencies[date.month - 1, int(hour)]

        return lookup

    n_doy = 366
    sums = np.zeros((n_doy,) + grid_shape)
    occurrence_sums = np.zeros_like(sums)
    counts = np.zeros(n_doy)
    for _, row in train_index.iterrows():
        target = _load_target(row['target_file'])                           # [H, W]
        doy = row['date'].dayofyear - 1
        sums[doy] += target
        occurrence_sums[doy] += scores.exceedance(target, occurrence_value, occurrence_strict)
        counts[doy] += 1

    # circular day-of-year window average
    half_window = window_days // 2
    offsets = np.arange(-half_window, half_window + 1)
    window_means = np.zeros_like(sums)
    window_frequencies = np.zeros_like(sums)
    for doy in range(n_doy):
        neighbors = (doy + offsets) % n_doy
        neighbor_count = max(counts[neighbors].sum(), 1.0)
        window_means[doy] = sums[neighbors].sum(axis=0) / neighbor_count
        window_frequencies[doy] = occurrence_sums[neighbors].sum(axis=0) / neighbor_count

    def lookup(date: pd.Timestamp, hour: Optional[int]):
        return window_means[date.dayofyear - 1], window_frequencies[date.dayofyear - 1]

    return lookup


def build_baselines(
        baseline_names,
        baselines_config: dict,
        split_index: pd.DataFrame,
        eval_items: pd.DataFrame,
        mode: str,
        hours_per_day: int,
        occurrence_event: Optional[Tuple[float, bool]] = None
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Build baseline prediction stacks aligned with the evaluation items.

    Args:
        baseline_names: Names to build (subset of zero / climatology).
        baselines_config: The ``baselines`` section of the metrics config (the climatology window).
        split_index: Full split index (all splits; train rows feed the climatology).
        eval_items: One row per evaluated item, in prediction order (columns ``date``, ``hour``, ``target_file``).
        mode: Preparation mode.
        hours_per_day: Hourly steps per day.
        occurrence_event: Evaluation-side occurrence event as an exceedance spec (value, strict) — see
            :func:`resolve_occurrence_event` — used for the climatological occurrence frequency behind the Brier
            and explained-deviance baselines; defaults to plain ``target > 0``.

    Returns:
        Tuple (dict baseline name -> stack [N, H, W], climatological occurrence probability stack or None).
    """
    # validate before touching the filesystem, so an unsupported name fails with its own message rather than with
    # whatever the first target load happens to raise
    unknown = [name for name in baseline_names if name not in ('zero', 'climatology')]
    if unknown:
        raise ValueError(
            f'Unknown baseline(s) {unknown}. This project supports "zero" and "climatology" only; in particular '
            f'"persistence" was removed, since a diagnostic parameterization has no past observation at inference.'
        )

    occurrence_event = occurrence_event if occurrence_event is not None else (0.0, True)
    grid_shape = _load_target(eval_items.iloc[0]['target_file']).shape[-2:]
    n_items = len(eval_items)
    baselines: Dict[str, np.ndarray] = {}
    occurrence_probability = None

    if 'zero' in baseline_names:
        baselines['zero'] = np.zeros((n_items,) + grid_shape, dtype=np.float32)

    if 'climatology' in baseline_names:
        window_days = int(baselines_config.get('climatology', {}).get('window_days', 31))
        train_index = split_index[split_index['split'] == 'train']
        lookup = _climatology_tables(train_index, mode, hours_per_day, window_days, occurrence_event)
        climatology = np.zeros((n_items,) + grid_shape, dtype=np.float32)
        occurrence_probability = np.zeros((n_items,) + grid_shape, dtype=np.float32)
        for position, (_, item) in enumerate(eval_items.iterrows()):
            mean_map, frequency_map = lookup(item['date'], item.get('hour'))
            climatology[position] = mean_map
            occurrence_probability[position] = frequency_map
        baselines['climatology'] = climatology

    return baselines, occurrence_probability


def _observation_stack(eval_items: pd.DataFrame, mode: str) -> np.ndarray:
    """Load the observed target maps for the evaluation items into a ``[N, H, W]`` stack (in item order)."""
    maps = []
    for _, item in eval_items.iterrows():
        target = _load_target(item['target_file'])
        if mode == MODE_HOURLY:
            target = target[int(item['hour'])]
        maps.append(target)
    return np.stack(maps)


def _climatology_reference(
        split_index: pd.DataFrame,
        eval_items: pd.DataFrame,
        prepared_config: dict,
        metrics_config: dict,
        target_stats: dict
):
    """Shared setup of the model-INDEPENDENT selection denominators: the climatology baseline stacks, the
    observations, and the resolved occurrence event. Factored so the two denominators below cannot drift apart in
    how they build the climatology or condition the cells."""
    occurrence_event = resolve_occurrence_event(metrics_config, target_stats)
    baselines, occurrence_probability = build_baselines(
        ['climatology'], metrics_config.get('baselines', {}), split_index, eval_items,
        prepared_config['mode'], int(prepared_config['hours_per_day']), occurrence_event=occurrence_event
    )
    observation = _observation_stack(eval_items, prepared_config['mode'])
    occurrence_value, occurrence_strict = occurrence_event
    occurrence = scores.exceedance(observation, occurrence_value, occurrence_strict)
    return baselines, occurrence_probability, observation, occurrence, occurrence_event


def climatology_conditional_mae(
        split_index: pd.DataFrame,
        eval_items: pd.DataFrame,
        prepared_config: dict,
        metrics_config: dict,
        target_stats: dict
) -> Tuple[float, Tuple[float, bool]]:
    """Model-INDEPENDENT denominator of the ``mae_cond_ss_climatology`` selection component: the conditional MAE
    of the climatology baseline over the evaluation items, on the evaluation occurrence cells.

    The climatology baseline (built from the train split exactly as in the eval suite) and the observations are
    fixed, so this scalar is constant across trials; the tuning stage computes it ONCE and injects it into every
    trial's module, which then forms ``mae_cond_ss_climatology = 1 - mae_cond(model) / denominator`` during
    validation (a properly normalized skill term, commensurate with the [0, 1] PSD-fidelity term it is combined
    with). Returns the denominator and the resolved occurrence event ``(value, strict)`` so the module conditions
    its numerator on the same cells the eval suite uses.

    Args:
        split_index: Full split index (train rows feed the climatology).
        eval_items: One row per validation item (columns ``date``, ``hour``, ``target_file``), in item order.
        prepared_config: Contents of prepared_config.json (mode, hours_per_day).
        metrics_config: The parsed metrics config (the climatology window).
        target_stats: Train target statistics (accepted for signature stability).
    """
    baselines, _, observation, occurrence, occurrence_event = _climatology_reference(
        split_index, eval_items, prepared_config, metrics_config, target_stats
    )
    denominator = scores.conditional_error(baselines['climatology'], observation, kind='mae', condition=occurrence)
    return float(denominator), occurrence_event


def climatology_brier(
        split_index: pd.DataFrame,
        eval_items: pd.DataFrame,
        prepared_config: dict,
        metrics_config: dict,
        target_stats: dict
) -> Tuple[float, Tuple[float, bool]]:
    """Model-INDEPENDENT denominator of the ``brier_skill_score`` selection component: the Brier score of the
    CLIMATOLOGICAL occurrence frequency against the observed occurrence event.

    The sibling of :func:`climatology_conditional_mae`, and needed for the same reason: ``brier_skill_score`` is a
    component of the CLASSIFICATION selection composite (``valid_classification_score``), so each trial needs a
    fixed denominator computed once per sweep rather than re-deriving the climatology per trial.

    Returns:
        Tuple (baseline Brier score, resolved occurrence event) — NaN if no climatological probability could be
        built (no train rows).
    """
    _, occurrence_probability, _, occurrence, occurrence_event = _climatology_reference(
        split_index, eval_items, prepared_config, metrics_config, target_stats
    )
    if occurrence_probability is None:
        return float('nan'), occurrence_event
    return float(scores.brier_score(occurrence_probability, occurrence)), occurrence_event


# ----------------------------------------------------------------------------------------------------------------
# metric suite
# ----------------------------------------------------------------------------------------------------------------
def run_metric_suite(
        metrics_config: dict,
        prediction: np.ndarray,
        observation: np.ndarray,
        probability: Optional[np.ndarray],
        baselines: Dict[str, np.ndarray],
        occurrence_probability: Optional[np.ndarray],
        target_stats: dict,
        prediction_structure: Optional[np.ndarray] = None,
        observation_structure: Optional[np.ndarray] = None,
        progress: bool = False
) -> Tuple[Dict[str, float], dict]:
    """Compute the full metric suite. THE single evaluation path, for every family and both tasks.

    Args:
        metrics_config: The parsed metrics config — `metrics_daily.yaml` or `metrics_hourly.yaml`. This function
            is mode-agnostic and decides what to compute from the ARRAYS it is handed, never from a mode key.
        prediction: Model predictions in the target space, ``[N, H, W]``. The point/skill/categorical/calibration
            scores are computed on this stack (the ensemble MEAN for a stochastic family, the single deterministic
            prediction otherwise).
        observation: Observed targets, ``[N, H, W]``.
        probability: Occurrence PROBABILITIES, ``[N, H, W]`` — the occurrence head's output in daily mode, and the
            model's own output in hourly classification mode (where the prediction IS a probability). ``None`` when
            no probabilistic occurrence forecast exists, which disables the reliability diagram and the explained
            deviance and makes the Brier skill score fall back to a binarized prediction.
        baselines: Baseline prediction stacks.
        occurrence_probability: Climatological occurrence probability stack (Brier / deviance baseline) or None.
        target_stats: Train target statistics (threshold resolution).
        prediction_structure: Optional separate stack for the SPATIAL-structure scores (PSD band ratios, FSS,
            log-spectral distance, sharpness/variance ratios and the PSD curves). For an ensemble run this is the
            POOLED ``[N*M, H, W]`` member stack (every item-member pair as one map), so the structure scores
            aggregate the per-member texture (averaging spectra / FSS fractions, never the maps) instead of the
            over-smoothed ensemble mean. Defaults to ``prediction``.
        observation_structure: Observation stack paired with ``prediction_structure`` for the spatial scores. For an
            ensemble run this is the observation REPLICATED M times (each member paired with its item's obs), which
            FSS needs and which leaves the ratio-of-means scores (PSD/sharpness/variance) unchanged. Defaults to
            ``observation``.
        progress: Show a tqdm progress bar over the spatial-structure section (PSD/FSS per map) — the dominant cost
            once the structure stack is the pooled ``[N*M, H, W]`` ensemble. Requires tqdm; silently skipped if it
            is unavailable.

    Returns:
        Tuple (flat scalar metrics, curves payload for the reporting figures).
    """
    config = metrics_config.get('metrics', {})
    thresholds = resolve_thresholds(metrics_config, target_stats)
    flat: Dict[str, float] = {}
    curves: dict = {'thresholds': {name: spec.obs_value for name, spec in thresholds.items()}}

    # A probability field is detected structurally rather than passed in, because run_metric_suite is deliberately
    # mode-agnostic: it decides what to do from the arrays it is handed. Used for two things below — choosing the
    # probabilistic FSS form, and warning about a decision threshold that would fire everywhere.
    prediction_is_probability = bool(
        prediction.size and np.nanmin(prediction) >= 0.0 and np.nanmax(prediction) <= 1.0
        and np.all(np.isin(observation, (0.0, 1.0)))
    )

    # spatial-structure scores read from a (possibly) distinct prediction/observation pair: the POOLED
    # ``[N*M, H, W]`` ensemble stack (obs replicated per member) for an ensemble run, the point
    # prediction/observation otherwise. Pooled formulae (member m of item i: prediction y^_im, obs y_i):
    #   PSD ratio  : mean_{i,m} |FFT(y^_im)|^2  /  mean_i |FFT(y_i)|^2      (average the spectra, not the maps)
    #   sharpness  : std_{i,m,cells} grad(y^_im) / std_{i,cells} grad(y_i)
    #   variance   : mean_{i,m} Var_cells(y^_im)  / mean_i Var_cells(y_i)
    #   FSS_{th,s} : 1 - [sum_{i,m} ||f(y^_im)-f(y_i)||^2] / [sum_{i,m} ||f(y^_im)||^2 + M sum_i ||f(y_i)||^2]
    # The PSD/sharpness/variance ratios are invariant to the M-fold obs replication; FSS requires the pairing.
    # Point/skill/categorical/calibration scores always read `prediction`/`observation` (the mean, or the single
    # prediction, vs the un-replicated obs).
    structure = prediction_structure if prediction_structure is not None else prediction
    structure_obs = observation_structure if observation_structure is not None else observation

    # the occurrence event is `target > 0`: the stored target has already been noise-filtered in preparation
    occurrence_value, occurrence_strict = resolve_occurrence_event(metrics_config, target_stats)
    occurrence = scores.exceedance(observation, occurrence_value, occurrence_strict)

    # --- continuous -------------------------------------------------------------------------------------------
    continuous = config.get('continuous', {})
    if 'rmse' in continuous:
        flat['rmse'] = scores.rmse(prediction, observation)
    if 'mae' in continuous:
        flat['mae'] = scores.mae(prediction, observation)
    if 'bias' in continuous:
        flat['bias'] = scores.bias(prediction, observation)
    if 'rmse_conditional_positive' in continuous:
        flat['rmse_cond_pos'] = scores.conditional_error(prediction, observation, kind='rmse',
                                                         condition=occurrence)
    if 'mae_conditional_positive' in continuous:
        flat['mae_cond_pos'] = scores.conditional_error(prediction, observation, kind='mae',
                                                        condition=occurrence)
    if 'mae_stratified' in continuous:
        bin_edges = [(name, thresholds[name].obs_value) for name in continuous['mae_stratified']['bins']]
        stratified = scores.stratified_mae(prediction, observation, bin_edges)
        flat.update(stratified)
        curves['error_by_bin'] = {
            'model': stratified,
            **{
                name: scores.stratified_mae(stack, observation, bin_edges)
                for name, stack in baselines.items()
            }
        }

    # observed-exceedance subgroups reused by the threshold-conditioned metrics below (the `occurrence` name
    # resolves to the evaluation occurrence event, identical to the `occurrence` mask computed above)
    def observed_subgroup(threshold_name: str) -> np.ndarray:
        spec = thresholds[threshold_name]
        return scores.exceedance(observation, spec.obs_value, spec.obs_strict)

    if 'r2' in continuous:                              # overall and within each observed-exceedance subgroup
        flat['r2'] = scores.r2_score(prediction, observation)
        for threshold_name in continuous['r2'].get('thresholds', []):
            flat[f'r2_{threshold_name}'] = scores.r2_score(
                prediction, observation, condition=observed_subgroup(threshold_name)
            )
    if 'estimation_tendency' in continuous:             # under-/over-estimation mass per event set
        tolerance = float(continuous['estimation_tendency'].get('tolerance', 0.0))
        for threshold_name in continuous['estimation_tendency'].get('thresholds', []):
            tendency = scores.estimation_tendency(
                prediction, observation, condition=observed_subgroup(threshold_name), tolerance=tolerance
            )
            flat[f'under_frac_{threshold_name}'] = tendency['under']
            flat[f'over_frac_{threshold_name}'] = tendency['over']

    # --- categorical ------------------------------------------------------------------------------------------
    # Two kinds of entry share the `<score>_<threshold>` grammar. The contingency scores (pod/far/csi/ets/hss/sedi/
    # frequency_bias) come off a 2x2 table at one decision cut. The THRESHOLD-FREE scores (roc_auc,
    # average_precision, dice) cut only the OBSERVATION and read the prediction as a continuous field: the ordering
    # of it for the ranking metrics, the overlap ratio for soft Dice. For the `occurrence` event that field is the
    # occurrence PROBABILITY when the model emits one, and the regression prediction otherwise; for the hour bands it
    # is always the prediction, since the occurrence head says nothing about intensity.
    categorical = config.get('categorical', {})
    requested_scores = list(categorical.get('scores', []))
    table_scores = [name for name in requested_scores if name not in THRESHOLD_FREE_SCORES]
    ranking_scores = [name for name in requested_scores if name in RANKING_SCORES]
    soft_scores = [name for name in requested_scores if name in SOFT_SCORES]
    ranking_edges = scores.ranking_bin_edges()
    curves['roc_pr'] = {}
    curves['confusion'] = {}
    for threshold_name in categorical.get('thresholds', []):
        spec = thresholds[threshold_name]
        # obs and pred sides are cut SEPARATELY: symmetric for the daily hour bands, and for the hourly task a
        # decision threshold on the probability against labels that are read as they are (see EventThreshold).
        counts = scores.contingency_counts(
            prediction, observation, spec.pred_value, spec.pred_strict,
            obs_threshold=spec.obs_value, obs_strict=spec.obs_strict
        )
        # A probability field admits only a cut strictly INSIDE (0, 1); anything else is degenerate in one direction
        # or the other. Both ends matter, and the second only became reachable when the daily `occurrence` threshold
        # gained `pred_value: 1`: at <= 0 every cell with any probability counts as a predicted event (pod ~ 1), and
        # at >= 1 essentially none does (pod ~ 0). Neither raises, and both produce a full table of nonsense.
        if prediction_is_probability and not (0.0 < spec.pred_value < 1.0):
            direction = ('at > 0, so every cell with any non-zero probability counts as a predicted event (pod ~ 1, '
                         'far ~ 1 - base_rate)' if spec.pred_value <= 0.0 else
                         f'at {spec.pred_value:g}, which a probability essentially never reaches, so almost nothing '
                         f'counts as a predicted event (pod ~ 0)')
            logger.warning(
                f'Threshold "{threshold_name}" cuts the PREDICTION {direction} — but the prediction looks like a '
                f'probability field, so the contingency scores for this threshold are meaningless. An hourly '
                f'(classification) pipeline must use `kind: probability` thresholds in '
                f'metrics.categorical.thresholds, e.g. `p50: {{kind: probability, value: 0.5}}` — not the daily '
                f'suite\'s `occurrence` / hour-band entries.'
            )
        table = scores.categorical_scores(*counts)
        for score_name in table_scores:
            flat[f'{score_name}_{threshold_name}'] = table[score_name]
        hits, misses, false_alarms, correct_negatives = counts
        curves['confusion'][threshold_name] = {
            'hits': hits, 'misses': misses,
            'false_alarms': false_alarms, 'correct_negatives': correct_negatives
        }

        if not (ranking_scores or soft_scores):
            continue
        # The threshold-free scores need a continuous field: the occurrence probability where the model emits one,
        # the regression prediction otherwise (the occurrence head says nothing about intensity, so the hour bands
        # always read the prediction). score_max maps hours into [0, 1] for the ranking binning; being a fixed
        # monotone map it leaves the ranking metrics exactly invariant.
        # Keyed on the OBSERVED event alone, deliberately not on `is_symmetric`. The threshold-free scores sweep every
        # cut, so a threshold's prediction-side DECISION level is irrelevant to them — what matters is whether the
        # event being ranked against is the occurrence event. Requiring symmetry here meant the daily `occurrence`
        # threshold silently stopped using the probability the moment it gained a `pred_value`, dropping
        # `dice_occurrence` and re-ranking `roc_auc` / `average_precision` off the hours field instead.
        is_occurrence = spec.obs_event == (occurrence_value, occurrence_strict)
        use_probability = (probability is not None) and (is_occurrence or prediction_is_probability)
        continuous_field = probability if use_probability else prediction
        score_max = 1.0 if (use_probability or prediction_is_probability) else max(float(np.nanmax(prediction)), 1.0)
        event = observed_subgroup(threshold_name)

        # Soft Dice is a RATIO of the field to the event, not an ordering of it, so unlike the ranking metrics it is
        # not invariant to score_max: on a lightning-hours field 2*sum(p*o)/(sum(p)+sum(o)) mixes units and means
        # nothing. Emitted only where the field is a genuine probability — absent on the hour bands of a regression
        # run, which is a deliberate absence rather than a NaN.
        if 'dice' in soft_scores and use_probability:
            flat[f'dice_{threshold_name}'] = scores.dice_coefficient(continuous_field, event)

        if ranking_scores:
            ranking = scores.finalize_ranking_metrics(
                scores.ranking_partials(continuous_field, event, ranking_edges, score_max=score_max), ranking_edges
            )
            for score_name in ranking_scores:
                flat[f'{score_name}_{threshold_name}'] = float(ranking[score_name])
            curves['roc_pr'][threshold_name] = {
                'fpr': ranking['fpr'], 'tpr': ranking['tpr'],
                'recall': ranking['recall'], 'precision': ranking['precision'],
                'base_rate': float(np.mean(event)),
                'roc_auc': ranking['roc_auc'],
                'average_precision': ranking['average_precision'],
                'from_probability': bool(use_probability)
            }

    # --- skill vs trivial baselines ---------------------------------------------------------------------------
    skill = config.get('skill', {})
    if 'mse_skill_score' in skill:
        model_mse = scores.rmse(prediction, observation) ** 2
        for baseline_name in skill['mse_skill_score'].get('baselines', []):
            if baseline_name in baselines:
                baseline_mse = scores.rmse(baselines[baseline_name], observation) ** 2
                flat[f'mse_ss_{baseline_name}'] = scores.skill_score(model_mse, baseline_mse)
    if 'mae_conditional_skill_score' in skill:
        model_error = scores.conditional_error(prediction, observation, kind='mae', condition=occurrence)
        for baseline_name in skill['mae_conditional_skill_score'].get('baselines', []):
            if baseline_name in baselines:
                baseline_error = scores.conditional_error(
                    baselines[baseline_name], observation, kind='mae', condition=occurrence
                )
                flat[f'mae_cond_ss_{baseline_name}'] = scores.skill_score(model_error, baseline_error)
    if 'brier_skill_score' in skill and occurrence_probability is not None:
        # with a probabilistic occurrence forecast this is the genuine Brier skill score; without one we binarize
        # the prediction at the occurrence event (a deterministic 0/1 forecast), so the score stays defined for
        # every model type
        forecast = probability if probability is not None else \
            scores.exceedance(prediction, occurrence_value, occurrence_strict).astype(np.float64)
        model_brier = scores.brier_score(forecast, occurrence)
        baseline_brier = scores.brier_score(occurrence_probability, occurrence)
        flat['brier_skill_score'] = scores.skill_score(model_brier, baseline_brier)
    if 'explained_deviance' in skill and occurrence_probability is not None and probability is not None:
        # BERNOULLI explained deviance, and deliberately probability-only: unlike the Brier score, the log loss is
        # unbounded at a confident mistake, so a binarized 0/1 forecast would be scored at the clip floor and the
        # deviance would be a large negative number that says nothing about the model. No probabilistic occurrence
        # forecast => the key is simply absent.
        flat['explained_deviance'] = scores.explained_deviance(probability, occurrence, occurrence_probability)

    # --- calibration ------------------------------------------------------------------------------------------
    calibration = config.get('calibration', {})
    if 'rank_correlation' in calibration:               # ordering agreement within each observed obs subgroup
        method = calibration['rank_correlation'].get('method', 'spearman')
        for threshold_name in calibration['rank_correlation'].get('thresholds', []):
            flat[f'rank_corr_{threshold_name}'] = scores.rank_correlation(
                prediction, observation, condition=observed_subgroup(threshold_name), method=method
            )
    if 'reliability_diagram' in calibration and probability is not None:
        # meaningful only for a probabilistic forecast: a classifier-less regressor would yield a degenerate
        # two-point curve, so the figure self-skips instead (curves['reliability'] is simply not populated)
        bins = int(calibration['reliability_diagram'].get('bins', 10))
        mean_probability, observed_frequency, counts = scores.reliability_curve(
            probability.ravel(), occurrence.ravel(), bins
        )
        curves['reliability'] = {
            'mean_probability': mean_probability, 'observed_frequency': observed_frequency, 'counts': counts
        }

    # --- spatial structure / high-frequency fidelity ----------------------------------------------------------
    spatial = config.get('spatial', {})
    # optional progress bar over the spatial section — the dominant evaluation cost once the structure stack is the
    # pooled [N*M, H, W] ensemble (PSD/FSS run per map). Sized by the per-map passes: the two power spectra
    # (computed once below and reused) + the FSS passes (fss_useful_scale ticks once per map AND scale). The FFT and
    # uniform-filter passes have different per-map costs, so the ETA is approximate but converges; sharpness /
    # variance are vectorized and do not tick.
    spatial_progress, spatial_bar = None, None
    if progress and spatial and structure.shape[0]:
        passes = 2                                                          # the structure + obs power spectra
        if 'fss' in spatial:
            # one FSS evaluation per threshold, or a single threshold-free one in the probabilistic form
            n_fss_events = 1 if prediction_is_probability else len(spatial['fss']['thresholds'])
            passes += n_fss_events * len(spatial['fss']['scales'])
        try:
            from tqdm import tqdm
            spatial_bar = tqdm(total=passes * structure.shape[0], unit='map',
                               desc=f'spatial metrics over {structure.shape[0]} maps')
            spatial_progress = spatial_bar.update
        except Exception:                                                   # tqdm optional: compute without a bar
            spatial_progress = None

    # the FFT power spectrum is the spatial section's expensive shared input — compute it ONCE per stack and reuse
    # it across the band ratios, the log-spectral distance and the report PSD curve (instead of recomputing it ~3x
    # each). Identical numbers, fewer FFT passes; the curves block below always needs both, so this is never wasted.
    # For an ENSEMBLE run (pooled [N*M, H, W] structure stack) we additionally keep each map's radial PSD — in the
    # SAME FFT pass — to draw the probabilistic models' +/-1 std ensemble-spread band on the report PSD curve.
    is_ensemble_structure = observation_structure is not None
    structure_per_map = None
    if is_ensemble_structure:
        _psd_centers, structure_per_map, structure_spectrum = scores.radial_psd_per_map(
            structure, progress=spatial_progress, return_mean_spectrum=True
        )
    else:
        structure_spectrum = scores.mean_power_spectrum(structure, progress=spatial_progress)
    obs_spectrum = scores.mean_power_spectrum(structure_obs, progress=spatial_progress)

    if 'psd_band_ratio' in spatial:
        bands = {
            name: (float(low), float(high))
            for name, (low, high) in spatial['psd_band_ratio']['bands'].items()
        }
        ratios = scores.psd_band_ratios(structure, structure_obs, bands,
                                        pred_spectrum=structure_spectrum, obs_spectrum=obs_spectrum)
        for band_name, ratio in ratios.items():
            flat[f'psd_ratio_{band_name}'] = ratio
        # scalar band fidelities ``clip(1 - |1 - ratio|, 0, 1)``: every ``psd_<name>_fidelity: {band: <band>}``
        # spatial entry emits its own key through ONE psd_fidelity function, so the suite carries both the
        # high-band over-smoothing detector (``psd_high_fidelity``) and the full-band term (``psd_full_fidelity``)
        # used in the selection score
        for key, spec in spatial.items():
            if key.endswith('_fidelity') and isinstance(spec, dict):
                flat[key] = scores.psd_fidelity(ratios[spec.get('band', 'high')])
    if 'log_spectral_distance' in spatial:
        flat['log_spectral_distance'] = scores.log_spectral_distance(
            structure, structure_obs, pred_spectrum=structure_spectrum, obs_spectrum=obs_spectrum
        )
    if 'fss' in spatial:
        fss_config = spatial['fss']
        scales = [int(scale) for scale in fss_config['scales']]
        curves['fss'] = {}
        if prediction_is_probability:
            # CLASSIFICATION: prediction and observation are already fractions (a probability and a 0/1 event), so
            # FSS needs no threshold and IS the fractions Brier skill score at each scale. Which form applies
            # follows from the data, not from a config switch — a thresholded FSS on a probability field would just
            # be one arbitrary decision cut, and the threshold-free version strictly dominates it.
            useful_scale, by_scale = scores.fss_useful_scale(structure, structure_obs, None, scales,
                                                            progress=spatial_progress)
            for scale, fss_value in by_scale.items():
                flat[f'fss_s{scale}'] = fss_value
            if fss_config.get('report_useful_scale', False):
                flat['fss_useful_scale'] = useful_scale
            curves['fss']['probabilistic'] = by_scale
        else:
            # REGRESSION: neither side is a probability, so the fractions have to come from an hour band. The
            # threshold-free form would compare neighbourhood means of HOURS, a scale-dependent MSE skill score
            # rather than a Brier one.
            for threshold_name in fss_config['thresholds']:
                spec = thresholds[threshold_name]
                useful_scale, by_scale = scores.fss_useful_scale(
                    structure, structure_obs, spec.obs_value, scales, spec.obs_strict, progress=spatial_progress
                )
                for scale, fss_value in by_scale.items():
                    flat[f'fss_{threshold_name}_s{scale}'] = fss_value
                if fss_config.get('report_useful_scale', False):
                    flat[f'fss_useful_scale_{threshold_name}'] = useful_scale
                curves['fss'][threshold_name] = by_scale
    if 'sharpness_ratio' in spatial:
        flat['sharpness_ratio'] = scores.sharpness_ratio(structure, structure_obs)
    if 'variance_ratio' in spatial:
        flat['variance_ratio'] = scores.variance_ratio(structure, structure_obs)

    # radially-averaged PSD curves for the report (model, observations and baselines) — reuse the cached structure
    # / obs spectra (baselines have their own stacks, so they are computed fresh)
    wavelengths, observation_power = scores.radial_psd(structure_obs, spectrum=obs_spectrum)
    curves['psd'] = {'wavelengths': wavelengths, 'obs': observation_power,
                     'model': scores.radial_psd(structure, spectrum=structure_spectrum)[1]}
    # probabilistic models only: +/-1 std ENSEMBLE-spread band around the model PSD. The per-map radial spectra are
    # grouped back by item (the structure stack is item-major, M members per item) so the band is the typical
    # member-to-member spread of the spatial spectrum (mean over items of the per-item across-member std) — the
    # ensemble's spatial-structure uncertainty, not inter-day variability.
    if structure_per_map is not None:
        n_items = observation.shape[0]
        if n_items > 0 and structure_per_map.shape[0] % n_items == 0:
            members = structure_per_map.shape[0] // n_items
            per_item = structure_per_map.reshape(n_items, members, -1)       # [N, M, n_bins]
            curves['psd']['model_std'] = per_item.std(axis=1).mean(axis=0)   # mean ensemble spread per band
    for baseline_name, stack in baselines.items():
        if baseline_name != 'zero':
            curves['psd'][baseline_name] = scores.radial_psd(stack)[1]
    if spatial_bar is not None:
        spatial_bar.close()

    return flat, curves


# ----------------------------------------------------------------------------------------------------------------
# ensemble metric suite (stochastic-family ensemble evaluation)
# ----------------------------------------------------------------------------------------------------------------
def merge_ensemble_partials(accumulator: Optional[dict], batch: dict) -> dict:
    """Sum one batch's ensemble partials into the running accumulator (scalars add, ``rank_counts`` elementwise)."""
    if accumulator is None:
        return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in batch.items()}
    for key, value in batch.items():
        accumulator[key] = accumulator[key] + value
    return accumulator


def finalize_ensemble_metrics(
        partials: Optional[dict], ensemble_config: dict, n_members: int
) -> Tuple[Dict[str, float], dict]:
    """Reduce the summed ensemble partials into flat scalar metrics + the rank-histogram curve.

    This is the ONE place the streaming sums are divided. ``crps_sums`` / ``spread_skill_sums`` return sums
    precisely because the full ``[N, M, H, W]`` stack cannot be held in memory: sums are additive across batches,
    means and ratios are not, so dividing anywhere but here would give a mean-of-ratios.

    Args:
        partials: Accumulated partials from :func:`src.utils.metrics.scores.ensemble_partials` summed over every
            batch (None when the model produced no ensemble, e.g. the deterministic U-net); then this is a no-op.
        ensemble_config: The ``ensemble`` section of the metrics config; its keys select which metrics are emitted
            (``crps``, ``almost_fair_crps``, ``spread_skill_ratio``, ``rank_histogram``).
        n_members: Ensemble size M (the rank histogram has M+1 bins).

    Returns:
        Tuple (flat scalar metrics, curves payload — ``rank_histogram`` counts for the report figure). The CRPS
        scores come in an all-cells form and an occurrence-conditioned ``_occ`` (tail-focused) form.
    """
    flat: Dict[str, float] = {}
    curves: dict = {}
    if not partials or not ensemble_config:
        return flat, curves

    def average(total: float, count: float) -> float:
        return float(total / count) if count else float('nan')

    if 'crps' in ensemble_config:
        flat['crps'] = average(partials['crps_sum'], partials['crps_n'])
        flat['crps_occ'] = average(partials['crps_sum_occ'], partials['crps_n_occ'])
    if 'almost_fair_crps' in ensemble_config:
        flat['almost_fair_crps'] = average(partials['af_crps_sum'], partials['crps_n'])
        flat['almost_fair_crps_occ'] = average(partials['af_crps_sum_occ'], partials['crps_n_occ'])
    if 'spread_skill_ratio' in ensemble_config:
        count = partials['ss_n']
        if count:
            spread = np.sqrt(partials['var_sum'] / count)
            skill_rmse = np.sqrt(partials['sqerr_sum'] / count)
            flat['spread_skill_ratio'] = float(spread / skill_rmse) if skill_rmse > 1e-12 else float('nan')
    if 'rank_histogram' in ensemble_config:
        counts = np.asarray(partials['rank_counts'], dtype=np.int64)
        flat['rank_histogram_reliability'] = scores.rank_histogram_reliability(counts)
        curves['rank_histogram'] = {'counts': counts, 'n_members': int(n_members)}

    return flat, curves
