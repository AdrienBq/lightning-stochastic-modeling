"""Trial selection scoring, shared by all three model families.

Every family ranks and checkpoints trials on ONE composite score computed in the target space, so their trials
tables are directly comparable. There are two composites — one per task — and **the task picks the composite, not
the config**: a search space may only *declare* which one it expects, and :func:`selection_metric_for_mode` raises
if that declaration disagrees with the prepared data's mode. A composite chosen by config would be one more way for
a run to score itself on the wrong thing.

    valid_regression_score      = 0.60 * mae_cond_ss_climatology       (magnitude, anti-trivial)
                                + 0.40 * psd_full_fidelity             (structure, anti-smoothing)

    valid_classification_score  = 0.50 * average_precision_occurrence  (discrimination)
                                + 0.20 * brier_skill_score             (calibration)
                                + 0.30 * psd_full_fidelity             (structure)

The weights live in each search space's ``selection`` block, not here, so they can be retuned without touching
code; this module supplies the components and the default weights, and the gate checks that the two agree.

⚠️ TWO KNOWN PROPERTIES OF THESE COMPOSITES, both deliberate:

1. The REGRESSION composite has **no false-alarm term**. ``mae_cond_ss_climatology`` is conditioned on
   observed-positive cells only, and ``psd_full_fidelity`` is a location-insensitive power ratio, so neither sees a
   cell where the model predicted lightning and none occurred. A spatially-textured over-forecast is not penalized.
   ``average_precision_occurrence`` IS computed and returned as an unweighted diagnostic so the gap is visible in
   the trials table — read it when comparing trials with similar composites.
2. The CLASSIFICATION composite's ``psd_full_fidelity`` term is **biased against calibration**. A calibrated
   probability field is intrinsically smoother than the 0/1 field it is compared with (spreading probability mass
   is what calibration MEANS at a 0.07 % base rate), so a correct model reads low here. Weighted 0.30 anyway, by
   decision; read it against the reliability diagram before concluding a model is over-smoothed.
"""
from typing import Dict, Optional, Tuple

import numpy as np

from src.utils.io.data import MODE_DAILY, MODE_HOURLY, normalize_mode
from src.utils.metrics.scores import (
    brier_score,
    categorical_scores,
    conditional_error,
    contingency_counts,
    exceedance,
    finalize_ranking_metrics,
    fss,
    psd_band_ratios,
    psd_fidelity,
    ranking_bin_edges,
    ranking_partials,
    skill_score
)

# wavelength band (in pixels) of the high-frequency PSD fidelity component; keep consistent with the `high` band
# of config/eval/metrics.yaml
SELECTION_PSD_HIGH_BAND = (2.0, 8.0)
# wavelength band (in pixels) of the full-band PSD fidelity component: all wavelengths >= 2 px (the low/mid/high
# bands combined; the DC component and the sub-2 px diagonal-corner coefficients are excluded). Keep consistent
# with the `full` band of config/eval/metrics.yaml
SELECTION_PSD_FULL_BAND = (2.0, np.inf)
# FSS neighborhood scale (pixels) of the fss_occurrence_scale3 diagnostic
SELECTION_FSS_SCALE = 3
# absolute hour band of the ets_h6 diagnostic. These used to be quantiles of the positive marginal (`ets_p99`,
# `fss_p90_scale3`), which collapse on a bounded 0-24 integer target -- the 0.99 and 0.999 quantiles land on the
# same hour. Absolute bands also make the trials table use the SAME threshold definition as metrics.yaml.
SELECTION_ETS_THRESHOLD = 6.0

MODE_SELECTION_METRICS = {
    MODE_DAILY: 'valid_regression_score',
    MODE_HOURLY: 'valid_classification_score',
}
DEFAULT_SELECTION_WEIGHTS = {
    'valid_regression_score': {'mae_cond_ss_climatology': 0.60, 'psd_full_fidelity': 0.40},
    'valid_classification_score': {
        'average_precision_occurrence': 0.50, 'brier_skill_score': 0.20, 'psd_full_fidelity': 0.30
    },
}


def selection_metric_for_mode(mode: str, declared_metric: Optional[str] = None) -> str:
    """The composite name implied by the prepared data's ``mode``, checked against the search space's declaration.

    Args:
        mode: Preparation mode from ``prepared_config.json``.
        declared_metric: ``selection.metric`` from the search space, when present.

    Returns:
        The composite name for this mode.

    Raises:
        ValueError: If ``declared_metric`` names a different composite. The mode wins by construction — the point
            of raising rather than overriding is that a search space asking for the regression composite on a
            binary target has its WEIGHTS wrong too, and silently swapping the name would leave those in place.
    """
    metric = MODE_SELECTION_METRICS[normalize_mode(mode)]
    if declared_metric is not None and declared_metric != metric:
        raise ValueError(
            f'The search space declares selection.metric "{declared_metric}", but mode "{mode}" implies '
            f'"{metric}". The task determines the composite, so fix the search space: its `selection.components` '
            f'weights are for the wrong composite as well.'
        )
    return metric


def compute_selection_components(
        prediction: np.ndarray,
        observation: np.ndarray,
        climatology_cond_mae: Optional[float] = None,
        climatology_brier: Optional[float] = None,
        occurrence_probability: Optional[np.ndarray] = None,
        occurrence_event: Optional[Tuple[float, bool]] = None,
        prediction_structure: Optional[np.ndarray] = None,
        observation_structure: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Components of both composite validation scores, in the target space.

    Every key is returned on every call, whichever composite is active: the two PSD fidelities share a single FFT
    pass, and the ranking metrics one binned accumulation, so computing the superset costs almost nothing and keeps
    the trials table comparable across families. Components:

    - ``mae_cond_ss_climatology``: conditional-MAE skill versus the climatology baseline over the occurrence cells
      (> 0 beats climatology; <= 0 is the imbalance-exploiting regime). Needs the model-INDEPENDENT denominator
      ``climatology_cond_mae``, computed once per sweep; ``NaN`` without it.
    - ``average_precision_occurrence``: PR-AUC of the occurrence event, ranked on the occurrence probability when
      the model emits one and on the prediction otherwise. Threshold-free, and the only component that sees a false
      alarm.
    - ``brier_skill_score``: Brier skill versus the climatology Brier score (``climatology_brier``); ``NaN``
      without a probabilistic forecast or that denominator.
    - ``psd_full_fidelity`` / ``psd_high_fidelity``: ``clip(1 - |1 - ratio|, 0, 1)`` of the full-band and
      high-band radial PSD ratios. ``full`` is the structure term of both composites; being a ratio of MEAN
      in-band power it is dominated by the large scales of the red lightning spectrum, so it penalizes
      over-smoothing more weakly than ``high`` — keep ``high`` as the sharper diagnostic.
    - ``ets_h6`` / ``fss_occurrence_scale3``: categorical and neighbourhood diagnostics at absolute bands, not in
      either composite.

    Args:
        prediction: Predicted maps in the target space, ``[N, H, W]`` (the ensemble MEAN for a stochastic family).
        observation: Observed target maps, ``[N, H, W]``.
        climatology_cond_mae: Conditional MAE of the climatology baseline on the occurrence cells.
        climatology_brier: Brier score of the climatological occurrence frequency, the skill denominator.
        occurrence_probability: The model's occurrence probabilities ``[N, H, W]`` when it has a probabilistic
            head; in the classification task this IS the prediction. ``None`` ranks on ``prediction`` instead and
            leaves ``brier_skill_score`` NaN.
        occurrence_event: Evaluation occurrence event ``(value, strict)``; ``None`` falls back to ``obs > 0``.
        prediction_structure: Optional separate stack for the STRUCTURE components, ``[N', H, W]`` — the pooled
            ``[N*M, H, W]`` member stack for an ensemble, so the texture terms are not measured on a mean that is
            smoother than any member. Mirrors ``run_metric_suite``'s split. ``None`` -> ``prediction``.
        observation_structure: Observation stack paired with ``prediction_structure``. ``None`` -> ``observation``.
    """
    structure_pred = prediction if prediction_structure is None else prediction_structure
    structure_obs = observation if observation_structure is None else observation_structure

    if occurrence_event is None:
        occurrence = observation > 0
    else:
        occurrence_value, occurrence_strict = occurrence_event
        occurrence = exceedance(observation, occurrence_value, occurrence_strict)

    model_cond_mae = conditional_error(prediction, observation, kind='mae', condition=occurrence)
    mae_cond_ss_climatology = skill_score(model_cond_mae, climatology_cond_mae) \
        if climatology_cond_mae is not None else float('nan')

    # ranked on the probability where there is one, else on the prediction itself (ordering hours against the
    # occurrence event). Computed through the SAME streaming primitives the metric suite uses -- one
    # implementation, so a trial's selection AP and its reported AP cannot drift.
    ranking_field = prediction if occurrence_probability is None else occurrence_probability
    score_max = 1.0 if occurrence_probability is not None else max(float(np.nanmax(prediction)), 1.0)
    edges = ranking_bin_edges()
    ranking = finalize_ranking_metrics(
        ranking_partials(ranking_field, occurrence, edges, score_max=score_max), edges
    )

    brier_skill = float('nan')
    if occurrence_probability is not None and climatology_brier is not None:
        brier_skill = skill_score(brier_score(occurrence_probability, occurrence), climatology_brier)

    # the high-band and full-band PSD fidelities share one FFT pass: psd_band_ratios computes the mean power
    # spectra of pred/obs once and only re-masks per band, so requesting both bands is no costlier than one
    psd_ratios = psd_band_ratios(
        structure_pred, structure_obs, {'high': SELECTION_PSD_HIGH_BAND, 'full': SELECTION_PSD_FULL_BAND}
    )
    return {
        'mae_cond_ss_climatology': mae_cond_ss_climatology,
        'average_precision_occurrence': float(ranking['average_precision']),
        'brier_skill_score': brier_skill,
        'psd_full_fidelity': psd_fidelity(psd_ratios['full']),
        'psd_high_fidelity': psd_fidelity(psd_ratios['high']),
        'ets_h6': categorical_scores(
            *contingency_counts(prediction, observation, SELECTION_ETS_THRESHOLD)
        )['ets'],
        'fss_occurrence_scale3': fss(
            structure_pred, structure_obs, 0.0, strict=True, scale=SELECTION_FSS_SCALE
        )
    }


def selection_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted sum of the selection components.

    A non-finite component contributes 0 rather than propagating NaN, so a degenerate model that makes a component
    undefined cannot be rewarded for it — but note it is not punished either: it simply forfeits that term.
    """
    score = 0.0
    for name, weight in weights.items():
        value = components.get(name, float('nan'))
        score += float(weight) * (value if np.isfinite(value) else 0.0)
    return float(score)
