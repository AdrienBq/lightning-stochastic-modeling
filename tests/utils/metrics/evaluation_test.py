"""Tests for src/utils/metrics/evaluation.py — THE single evaluation path, for every family and both tasks.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_ensemble_scores.py`` (the merge/finalize round trip and
the PSD-cache integration) and ``tests/test_probabilistic_eval_compat.py`` (the shared ensemble suite running
identically on both stochastic families), plus new coverage for the Block 2r threshold routing.

A's versions of these guarded their imports with ``try/except ImportError: return`` because that branch's environment
might lack torch or pandas. Ours has everything in ``minimal_requirements.txt``, so the imports are hard here: a test
that prints "skipped" and passes is the exact failure mode this repo keeps closing, and it would let a genuine
``evaluation.py`` import break read as green.
"""
import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import scores
from src.utils.metrics.evaluation import (
    build_baselines, climatology_brier, climatology_conditional_mae, finalize_ensemble_metrics,
    merge_ensemble_partials, resolve_occurrence_event, resolve_threshold, resolve_thresholds, run_metric_suite,
)


# =====================================================================================================================
# Streaming ensemble metrics: merge the per-batch partials, divide ONCE at the end  (ported)
# =====================================================================================================================
def test_finalize_round_trip_over_batches():
    """The whole streaming contract end to end: three batches of partials merged, finalized once, and the result
    identical to the single-shot score over the same members. Sums are additive; means and ratios are not, which is
    why the division happens exactly once here."""
    rng = np.random.default_rng(6)
    m, k = 12, 150
    members = rng.gamma(1.0, 1.0, size=(m, k))
    obs = rng.gamma(1.0, 1.0, size=k)
    occurrence = (0.5, False)

    accumulator = None
    for chunk in (slice(0, 50), slice(50, 100), slice(100, 150)):
        partials = scores.ensemble_partials(members[:, chunk], obs[chunk], occurrence_event=occurrence,
                                            rng=np.random.default_rng(0))
        accumulator = merge_ensemble_partials(accumulator, partials)

    ensemble_spec = {'crps': {}, 'almost_fair_crps': {}, 'spread_skill_ratio': {}, 'rank_histogram': {}}
    flat, curves = finalize_ensemble_metrics(accumulator, ensemble_spec, m)

    for key in ('crps', 'crps_occ', 'almost_fair_crps', 'almost_fair_crps_occ', 'spread_skill_ratio',
                'rank_histogram_reliability'):
        assert key in flat and np.isfinite(flat[key]), key
    assert np.isclose(flat['crps'], scores.crps_ensemble(members, obs), rtol=1e-9, atol=1e-9)
    assert curves['rank_histogram']['counts'].shape == (m + 1,)


def test_merge_ensemble_partials_starts_from_none():
    """The accumulator is seeded with ``None`` by every caller, so merging into ``None`` must return the partials
    themselves rather than raising."""
    rng = np.random.default_rng(7)
    partials = scores.ensemble_partials(rng.gamma(1.0, 1.0, size=(4, 20)), rng.gamma(1.0, 1.0, size=20),
                                        occurrence_event=(0.0, True), rng=np.random.default_rng(0))
    merged = merge_ensemble_partials(None, partials)
    assert set(merged) == set(partials)
    assert merged['crps_sum'] == partials['crps_sum']


def test_metrics_yaml_ensemble_section_is_reachable(metrics_config):
    """Regression guard for a real gating bug: the evaluation stage enables the probabilistic suite from
    ``metrics_spec['metrics'].get('ensemble') or metrics_spec.get('ensemble')``. The section was once written at the
    TOP level while the code read it from under ``metrics:``, so CRPS, almost-fair CRPS, spread-skill, the rank
    histogram AND the 3-panel ensemble maps were all silently disabled with no error anywhere.

    Asserted against the SHIPPED config rather than a fixture copy, which is the only version that can drift.
    """
    ensemble_spec = metrics_config.get('metrics', {}).get('ensemble') or metrics_config.get('ensemble', {})
    assert ensemble_spec, 'ensemble section not reachable -> the probabilistic suite would be silently disabled'
    for key in ('crps', 'almost_fair_crps', 'spread_skill_ratio', 'rank_histogram'):
        assert key in ensemble_spec, f'{key} missing from the metrics.yaml ensemble section'
    assert 'ensemble' in metrics_config.get('metrics', {}), 'the ensemble section must live under `metrics:`'


# =====================================================================================================================
# The PSD cache must not change any number  (ported)
# =====================================================================================================================
def test_run_metric_suite_psd_cache_matches_fresh_computation():
    """``run_metric_suite``'s cached-spectrum spatial metrics must equal computing them the plain way on the DEFAULT
    (non-ensemble) path, where structure == prediction — i.e. exactly the deterministic U-net branch. Guards that the
    right cached spectrum reaches each function, which a single mis-wired argument would break silently."""
    rng = np.random.default_rng(13)
    n, h, w = 6, 16, 16
    prediction = rng.gamma(1.0, 1.0, size=(n, h, w))
    observation = rng.gamma(1.0, 1.0, size=(n, h, w))
    spec = {
        'thresholds': {'occurrence': {'kind': 'occurrence'}},
        'metrics': {'spatial': {
            'psd_band_ratio': {'bands': {'full': [2, np.inf], 'low': [4, np.inf], 'high': [2, 4]}},
            'psd_high_fidelity': {'band': 'high'}, 'psd_full_fidelity': {'band': 'full'},
            'log_spectral_distance': {}, 'fss': {'thresholds': ['occurrence'], 'scales': [1, 3]},
            'sharpness_ratio': {}, 'variance_ratio': {},
        }},
    }
    flat, _ = run_metric_suite(spec, prediction, observation, None, {}, None, {})

    bands = {'full': (2.0, np.inf), 'low': (4.0, np.inf), 'high': (2.0, 4.0)}
    ratios = scores.psd_band_ratios(prediction, observation, bands)
    assert flat['psd_ratio_high'] == ratios['high']
    assert flat['psd_ratio_full'] == ratios['full']
    # both the selection term (full) and the sharper diagnostic (high) must be wired through the fidelity loop
    assert flat['psd_full_fidelity'] == scores.psd_fidelity(ratios['full'])
    assert flat['psd_high_fidelity'] == scores.psd_fidelity(ratios['high'])
    assert flat['log_spectral_distance'] == scores.log_spectral_distance(prediction, observation)
    assert flat['sharpness_ratio'] == scores.sharpness_ratio(prediction, observation)


def test_psd_ensemble_band_is_added_for_an_ensemble_run_only():
    """``run_metric_suite`` adds a +/-1 sigma ensemble PSD band for a pooled structure stack and NOT for a point run,
    which is what lets ``_psd_curves`` draw the band without knowing the family."""
    rng = np.random.default_rng(2)
    n, m, h, w = 5, 4, 20, 28
    structure = np.abs(rng.normal(size=(n * m, h, w)))          # pooled [N*M, H, W], item-major blocks of M
    prediction = np.abs(rng.normal(size=(n, h, w)))
    observation = np.abs(rng.normal(size=(n, h, w)))
    spec = {'metrics': {'spatial': {'psd_band_ratio': {'bands': {'high': [2.0, 8.0]}}}}}
    baselines = {'zero': np.zeros_like(prediction)}

    _, curves = run_metric_suite(spec, prediction, observation, None, baselines, None, {},
                                 prediction_structure=structure,
                                 observation_structure=np.repeat(observation, m, axis=0))
    band = curves['psd'].get('model_std')
    assert band is not None, 'an ensemble run must add curves["psd"]["model_std"]'
    assert len(band) == len(curves['psd']['model']) == len(curves['psd']['wavelengths'])
    assert np.all(np.asarray(band) >= 0.0)

    _, point_curves = run_metric_suite(spec, prediction, observation, None, baselines, None, {})
    assert 'model_std' not in point_curves['psd'], 'a deterministic run must NOT add a band'


# =====================================================================================================================
# NEW: threshold routing — the two tasks cut different sides  (Block 2r)
# =====================================================================================================================
OCCURRENCE_EVENT = (0.0, True)


def test_resolve_threshold_defaults_to_an_absolute_symmetric_band():
    """The daily form: one level, both sides. ``kind: absolute`` is the default so an hour band needs no ceremony."""
    event = resolve_threshold({'value': 6.0}, {}, OCCURRENCE_EVENT)
    assert event.pred_value == 6.0 and event.obs_value == 6.0
    assert event.pred_strict == event.obs_strict


def test_resolve_threshold_probability_kind_splits_the_two_sides():
    """The hourly form: a DECISION cut on the prediction, and an observation read as-is. Without this the daily
    ``occurrence`` level (> 0) applied to a probability field gives POD ~ 1 and a table of nonsense."""
    event = resolve_threshold({'kind': 'probability', 'value': 0.5}, {}, OCCURRENCE_EVENT)
    assert event.pred_value == 0.5
    assert (event.obs_value, event.obs_strict) == OCCURRENCE_EVENT   # the occurrence event, not the probability cut


def test_resolve_threshold_occurrence_kind_is_symmetric_WITHOUT_a_pred_value():
    event = resolve_threshold({'kind': 'occurrence'}, {}, OCCURRENCE_EVENT)
    assert (event.pred_value, event.pred_strict) == OCCURRENCE_EVENT
    assert (event.obs_value, event.obs_strict) == OCCURRENCE_EVENT


def test_the_occurrence_kind_takes_a_SEPARATE_prediction_cut():
    """``pred_value`` gives the prediction side its own level while the observation side stays the occurrence event —
    which is what keeps every observation-conditioned metric (mae_cond_pos, r2_occurrence, FSS, the climatology Brier)
    unchanged by a decision-threshold choice."""
    event = resolve_threshold({'kind': 'occurrence', 'pred_value': 1}, {}, OCCURRENCE_EVENT)
    assert (event.obs_value, event.obs_strict) == OCCURRENCE_EVENT
    assert (event.pred_value, event.pred_strict) == (1.0, False)      # pred >= 1, inclusive by default
    assert not event.is_symmetric


def test_a_REGRESSION_occurrence_cut_at_zero_makes_the_table_DEGENERATE(daily_field):
    """🐛 The bug found by reading the block 4e gate's confusion matrix, pinned as the reason `pred_value` exists.

    A softplus/ReLU head emits small POSITIVE hours almost everywhere, so cutting the prediction at ``> 0`` marks every
    cell a predicted event: zero misses, zero correct negatives, POD = 1, frequency bias = 1/base-rate. Nothing raises
    — the table is simply meaningless, and the confusion figure reads as "lightning everywhere".
    """
    import numpy as np

    from src.utils.metrics import scores

    observation = daily_field(n=2, seed=3)                            # sparse, mostly zero
    prediction = np.full_like(observation, 0.2)                       # never exactly zero, never a whole hour
    prediction[observation > 0] = 4.0                                 # right where it matters

    symmetric = resolve_threshold({'kind': 'occurrence'}, {}, OCCURRENCE_EVENT)
    hits, misses, false_alarms, correct_negatives = scores.contingency_counts(
        prediction, observation, symmetric.pred_value, symmetric.pred_strict,
        obs_threshold=symmetric.obs_value, obs_strict=symmetric.obs_strict
    )
    assert misses == 0 and correct_negatives == 0, 'the degeneracy this parameter exists to remove'
    assert false_alarms == observation.size - hits

    decided = resolve_threshold({'kind': 'occurrence', 'pred_value': 1}, {}, OCCURRENCE_EVENT)
    hits, misses, false_alarms, correct_negatives = scores.contingency_counts(
        prediction, observation, decided.pred_value, decided.pred_strict,
        obs_threshold=decided.obs_value, obs_strict=decided.obs_strict
    )
    assert correct_negatives > 0, 'a sub-hour prediction over a dry cell must be a CORRECT NEGATIVE'
    assert false_alarms == 0 and misses == 0                          # this fixture predicts the event exactly


def test_the_SHIPPED_daily_suite_cuts_the_occurrence_PREDICTION_at_one_hour(metrics_config):
    """The decision itself, pinned against the real config rather than a fixture: `pred >= 1` is the same rule the
    h3/h6/h12 bands use at k=1, so the four thresholds form one consistent family (`obs >= k` vs `pred >= k`)."""
    resolved = resolve_thresholds(metrics_config, {})

    occurrence = resolved['occurrence']
    assert (occurrence.obs_value, occurrence.obs_strict) == OCCURRENCE_EVENT
    assert occurrence.pred_value == 1.0 and occurrence.pred_strict is False

    for name, level in (('h3', 3.0), ('h6', 6.0), ('h12', 12.0)):
        assert resolved[name].pred_value == level, f'{name} must keep its symmetric band'


def test_resolve_occurrence_event_rejects_a_reintroduced_threshold():
    """``occurrence_threshold`` was removed deliberately: occurrence is unconditionally ``target > 0``, and hourly
    denoising lives in the preparation stage. The HARD ASSERT is what stops it drifting back in as a config knob."""
    assert resolve_occurrence_event({}, {}) == OCCURRENCE_EVENT
    with pytest.raises(AssertionError, match='occurrence_threshold'):
        resolve_occurrence_event({'occurrence_threshold': 2.0}, {})


# =====================================================================================================================
# NEW: the two model-independent selection denominators — dataset-level, so they need a prepared directory
# =====================================================================================================================
@pytest.mark.source_invariant
def test_the_suite_reaches_the_SHARED_modules_rather_than_reimplementing_them():
    """``evaluation.py`` is the single eval stage for all three families, so every score it reports must come from
    ``scores.py`` and every mode/split decision from ``io.data``. A local reimplementation here is how the reported number
    and the trained objective drift apart — the ``crps_ensemble`` collision this merge exists to prevent."""
    import ast

    from src.utils.metrics import evaluation as evaluation_module

    tree = ast.parse(open(evaluation_module.__file__).read())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert any(name.startswith('src.utils.io.data') for name in imported), sorted(imported)
    assert any(name.startswith('src.utils.metrics') for name in imported), sorted(imported)


@pytest.mark.source_invariant
def test_the_suite_RECORDS_which_scores_are_hourly_only(repo_root):
    """``brier_skill_score``, ``explained_deviance`` and ``dice`` need a calibrated probability, which a daily run has
    no head for. They were annotated "occurrence head, when enabled" until block 3c dropped that head — so the note had
    to become "hourly mode only", with the daily consequence stated where a reader meets the key."""
    import os

    text = open(os.path.join(repo_root, 'config/eval/metrics.yaml')).read()
    assert 'HOURLY ONLY' in text


@pytest.mark.source_invariant
def test_the_suite_records_that_the_RANKING_scores_survive_a_daily_run(repo_root):
    """The other half of that note, and the reason dropping the probability head was acceptable: AP and ROC-AUC are
    invariant to any monotone rescaling, so ranking on predicted HOURS is exact rather than approximate. Without this
    written down beside the hourly-only note, a reader concludes the whole categorical group is lost on daily."""
    import os

    text = open(os.path.join(repo_root, 'config/eval/metrics.yaml')).read()
    assert 'monotone' in text and 'roc_auc' in text


def test_build_baselines_aligns_with_the_evaluation_items(prepared_split, metrics_config):
    split_index, prepared_config = prepared_split()
    eval_items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    baselines, occurrence_probability = build_baselines(
        ['zero', 'climatology'], metrics_config.get('baselines', {}), split_index, eval_items,
        prepared_config['mode'], int(prepared_config['hours_per_day']), occurrence_event=OCCURRENCE_EVENT
    )
    assert set(baselines) == {'zero', 'climatology'}
    for name, stack in baselines.items():
        assert stack.shape[0] == len(eval_items), name
    assert not baselines['zero'].any(), 'the zero baseline must be all zeros'
    assert occurrence_probability is not None
    assert occurrence_probability.min() >= 0.0 and occurrence_probability.max() <= 1.0


def test_climatology_brier_is_a_real_reference_a_skilful_forecast_beats(prepared_split, metrics_config):
    """The sibling of ``climatology_conditional_mae``, computed ONCE per sweep so every trial shares a fixed
    denominator. Directional check: a forecast that actually discriminates must score BELOW it, or
    ``brier_skill_score`` would be negative for a good model."""
    split_index, prepared_config = prepared_split()
    eval_items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    reference, occurrence_event = climatology_brier(
        split_index, eval_items, prepared_config, metrics_config, {}
    )
    assert occurrence_event == OCCURRENCE_EVENT
    assert np.isfinite(reference) and 0.0 < reference < 0.25          # a Brier score of a probability field

    observation = np.stack([np.load(path) for path in eval_items['target_file']])
    occurrence = scores.exceedance(observation, *OCCURRENCE_EVENT)
    skilful = np.where(occurrence, 0.8, 0.05)
    assert scores.brier_score(skilful, occurrence.astype(float)) < reference


def test_climatology_conditional_mae_is_finite_and_positive(prepared_split, metrics_config):
    """The regression composite's denominator. It conditions on the SAME occurrence cells the eval suite uses, which
    is what makes ``1 - mae_cond(model)/denominator`` a properly normalized skill term."""
    split_index, prepared_config = prepared_split()
    eval_items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    denominator, occurrence_event = climatology_conditional_mae(
        split_index, eval_items, prepared_config, metrics_config, {}
    )
    assert np.isfinite(denominator) and denominator > 0.0
    assert occurrence_event == OCCURRENCE_EVENT


def test_climatology_denominators_are_model_independent(prepared_split, metrics_config):
    """Both denominators must be reproducible from the data alone: the tuning stage computes them once and injects
    them into every trial, so a value that depended on anything per-trial would make trials incomparable."""
    split_index, prepared_config = prepared_split()
    eval_items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    first = (climatology_brier(split_index, eval_items, prepared_config, metrics_config, {})[0],
             climatology_conditional_mae(split_index, eval_items, prepared_config, metrics_config, {})[0])
    second = (climatology_brier(split_index, eval_items, prepared_config, metrics_config, {})[0],
              climatology_conditional_mae(split_index, eval_items, prepared_config, metrics_config, {})[0])
    assert first == second


# =====================================================================================================================
# NEW: the suite is mode-agnostic — the same call scores hours and probabilities
# =====================================================================================================================
@pytest.mark.parametrize('task', ['daily', 'hourly'])
def test_run_metric_suite_produces_the_continuous_group_for_both_tasks(task, daily_field, hourly_field):
    """Block 2r2: the continuous block is deliberately mode-agnostic and is fed whatever ``prediction`` holds, which
    on hourly is the probability field. The scores apply to both tasks — only the *interpretation* differs."""
    if task == 'daily':
        observation = daily_field(n=6, seed=1)
        prediction = np.clip(observation + np.random.default_rng(2).integers(-2, 3, observation.shape), 0, 24)
    else:
        observation = hourly_field(n=6, seed=1)
        prediction = np.clip(observation * 0.7 + np.random.default_rng(2).random(observation.shape) * 0.1, 0, 1)

    spec = {'thresholds': {'occurrence': {'kind': 'occurrence'}},
            'metrics': {'continuous': {'rmse': {}, 'mae': {}, 'bias': {}}}}
    flat, _ = run_metric_suite(spec, prediction, observation, None, {}, None, {})

    for key in ('rmse', 'mae', 'bias'):
        assert key in flat and np.isfinite(flat[key]), f'{key} missing for the {task} task'


# =====================================================================================================================
# The SHIPPED suite, end to end — the one call every family's evaluation makes
# =====================================================================================================================
@pytest.fixture(scope='module')
def suite_arrays():
    """A sparse bounded observation, a noisy prediction, an occurrence probability and both baselines — the argument set
    ``evaluate`` assembles. Module-scoped: the PSD pass over it is the expensive part."""
    rng = np.random.default_rng(0)
    n, h, w = 6, 24, 28
    observation = np.zeros((n, h, w))
    for index in range(n):
        active = rng.random((h, w)) < 0.02
        observation[index][active] = rng.integers(1, 20, size=active.sum())
    prediction = np.clip(observation + rng.normal(0, 1.5, observation.shape), 0, 24)
    probability_map = np.clip(0.5 * (observation > 0) + 0.2 * rng.random(observation.shape), 0, 1)
    occurrence_probability = np.full(observation.shape, float((observation > 0).mean()))
    baselines = {'zero': np.zeros_like(observation),
                 'climatology': np.full_like(observation, observation.mean())}
    return prediction, observation, probability_map, baselines, occurrence_probability


@pytest.fixture(scope='module')
def daily_suite(suite_arrays, metrics_config):
    prediction, observation, probability, baselines, occurrence = suite_arrays
    return run_metric_suite(metrics_config, prediction, observation, probability, baselines, occurrence, {})


EXPECTED_DAILY_KEYS = (
    'rmse', 'mae', 'bias', 'rmse_cond_pos', 'mae_cond_pos', 'r2', 'r2_occurrence',
    'pod_occurrence', 'far_occurrence', 'csi_occurrence', 'ets_occurrence', 'hss_occurrence', 'sedi_occurrence',
    'frequency_bias_occurrence', 'roc_auc_occurrence', 'average_precision_occurrence',
    'roc_auc_h3', 'average_precision_h3',
    'mse_ss_zero', 'mse_ss_climatology', 'mae_cond_ss_zero', 'mae_cond_ss_climatology',
    'brier_skill_score', 'explained_deviance',
    'rank_corr_occurrence', 'psd_ratio_full', 'psd_full_fidelity', 'psd_high_fidelity',
    'log_spectral_distance', 'fss_occurrence_s1', 'fss_useful_scale_occurrence',
    'sharpness_ratio', 'variance_ratio', 'mae_bin_occurrence_h3',
)


def test_the_shipped_suite_emits_every_expected_key(daily_suite):
    """Driven from ``config/eval/metrics.yaml`` itself, so this is the test that catches a score silently dropping out
    of the report — the failure mode where every number present is correct and one is simply absent."""
    flat, _ = daily_suite
    missing = [key for key in EXPECTED_DAILY_KEYS if key not in flat]
    assert not missing, missing


def test_every_selection_component_the_sweep_WEIGHTS_is_in_the_report(daily_suite):
    """The trials table and the report have to name the same quantities, or a trial's winning score cannot be located in
    its own report."""
    flat, _ = daily_suite
    for key in ('mae_cond_ss_climatology', 'psd_full_fidelity', 'average_precision_occurrence',
                'brier_skill_score'):
        assert key in flat, key


def test_no_removed_metric_family_LEAKS_back_into_the_report(daily_suite):
    """PIT needed ``target_stats['gamma_shape']``, the quantile ratios needed the positive marginal's tail, and Tweedie
    is unbounded-count machinery. All three went with the transform and the scope change — a key reappearing means one
    of those paths was reconnected."""
    flat, _ = daily_suite
    leaked = [key for key in flat if key.startswith(('pit_', 'q_ratio_', 'tweedie'))]
    assert not leaked, leaked


def test_the_curves_carry_the_roc_pr_and_confusion_payloads(daily_suite):
    """Both figures were added in Block 2 and both are driven off ``curves`` rather than recomputed, so a missing payload
    makes the figure self-skip — which the report only records as a warning."""
    _, curves = daily_suite
    assert 'roc_pr' in curves and 'confusion' in curves
    assert 'occurrence' in curves['roc_pr']


def test_the_occurrence_ranking_uses_the_PROBABILITY_and_the_hour_bands_do_not(daily_suite):
    """The ranking field is chosen per threshold: the occurrence event gets the model's probability when it has one, and
    an hour band has no probability to use, so it ranks on the predicted hours. Both are exact — AP and ROC-AUC are
    invariant to any monotone rescaling — and the flag is recorded so the report can say which was used."""
    _, curves = daily_suite
    assert curves['roc_pr']['occurrence']['from_probability'] is True
    assert curves['roc_pr']['h3']['from_probability'] is False


def test_the_confusion_counts_sum_to_the_CELL_COUNT(daily_suite, suite_arrays):
    """A contingency table that does not partition the grid means some cells were dropped by a mask or double-counted by
    an overlapping cut."""
    _, curves = daily_suite
    _, observation, _, _, _ = suite_arrays
    counts = curves['confusion']['occurrence']
    assert sum(counts.values()) == pytest.approx(observation.size, abs=1e-6), counts


def test_the_daily_FSS_keeps_its_PER_THRESHOLD_keys(daily_suite):
    """Daily FSS is the thresholded hour-band form, so its keys carry the threshold. The threshold-FREE keys belong to
    the hourly task, where the prediction is already a fraction — emitting those here would mean neighbourhood means of
    HOURS were being read as a Brier skill score."""
    flat, _ = daily_suite
    assert 'fss_occurrence_s1' in flat
    assert 'fss_s1' not in flat


@pytest.fixture(scope='module')
def suite_without_a_probability(suite_arrays, metrics_config):
    prediction, observation, _, baselines, occurrence = suite_arrays
    return run_metric_suite(metrics_config, prediction, observation, None, baselines, occurrence, {})


def test_explained_deviance_is_ABSENT_without_a_probabilistic_forecast(suite_without_a_probability):
    """It is a Bernoulli log-loss skill score, so it needs a calibrated probability — which a daily run has no head for.
    Absent is the right answer; a number computed from lightning-hours would not be."""
    flat, _ = suite_without_a_probability
    assert 'explained_deviance' not in flat


def test_brier_skill_score_is_STILL_defined_without_one(suite_without_a_probability):
    """The asymmetry that is easy to get wrong: ``brier_skill_score`` falls back to the occurrence probability the
    caller supplies (a climatological rate), so it survives where ``explained_deviance`` does not."""
    flat, _ = suite_without_a_probability
    assert 'brier_skill_score' in flat


def test_the_reliability_curve_is_absent_without_a_probabilistic_forecast(suite_without_a_probability):
    _, curves = suite_without_a_probability
    assert 'reliability' not in curves


# =====================================================================================================================
# The HOURLY configuration of the same suite — kind: probability, and the FSS form switch
# =====================================================================================================================
def _hourly_config(metrics_config):
    """``metrics.yaml`` retargeted for the hourly task, exactly as its own note prescribes: the categorical group cuts
    the PROBABILITY at 0.5, and every group that bins or conditions on observed intensity is pointed at the occurrence
    event instead of an hour band."""
    import copy

    config = copy.deepcopy(metrics_config)
    config['thresholds'] = {'occurrence': {'kind': 'occurrence'},
                            'p50': {'kind': 'probability', 'value': 0.5}}
    config['metrics']['categorical']['thresholds'] = ['p50']
    for group, key in (('continuous', 'r2'), ('continuous', 'estimation_tendency'),
                       ('calibration', 'rank_correlation')):
        config['metrics'][group][key]['thresholds'] = ['occurrence']
    config['metrics']['continuous']['mae_stratified']['bins'] = ['occurrence']
    config['metrics']['spatial']['fss']['thresholds'] = ['occurrence']
    return config


@pytest.fixture(scope='module')
def hourly_suite(metrics_config):
    rng = np.random.default_rng(0)
    observation = (rng.random((6, 24, 28)) < 0.05).astype(np.float64)
    prediction = np.clip(0.35 * observation + 0.1 * rng.random(observation.shape), 0, 1)
    baselines = {'zero': np.zeros_like(observation),
                 'climatology': np.full_like(observation, observation.mean())}
    flat, curves = run_metric_suite(_hourly_config(metrics_config), prediction, observation, prediction,
                                    baselines, np.full_like(observation, observation.mean()), {})
    return flat, curves, prediction, observation


def test_the_hourly_run_switches_FSS_to_its_THRESHOLD_FREE_form(hourly_suite):
    """When the prediction is already a probability its neighbourhood mean IS a fraction, so no cut is needed — and in
    that form FSS is exactly a fractions Brier skill score at that scale, which carries strictly more information than
    committing to a threshold. Derived from the MODE, not from a config key, so the meaningless combination cannot be
    requested."""
    flat, _, _, _ = hourly_suite
    assert 'fss_s1' in flat
    assert not [key for key in flat if key.startswith('fss_occurrence')], \
        [key for key in flat if key.startswith('fss')]


def test_the_hourly_categorical_group_uses_the_PROBABILITY_decision_cut(hourly_suite):
    """The whole point of ``kind: probability``. Under a shared ``occurrence`` cut the prediction side becomes ``p > 0``,
    which fires on every cell with any non-zero probability: POD would be ~1 and FAR ~ the base rate — a full contingency
    table of nonsense, with no error raised. A real decision cut gives POD < 1."""
    flat, _, _, _ = hourly_suite
    assert flat['pod_p50'] < 1.0, flat['pod_p50']
    assert np.isfinite(flat['csi_p50'])


def test_the_hourly_run_emits_the_ranking_metrics_and_explained_deviance(hourly_suite):
    """``explained_deviance`` appears here and is absent on daily for one reason: on the hourly task the prediction IS
    the calibrated probability, so the Bernoulli log-loss skill score is well defined."""
    flat, _, _, _ = hourly_suite
    assert 'average_precision_p50' in flat and 'roc_auc_p50' in flat
    assert 'explained_deviance' in flat


def test_a_shared_occurrence_cut_on_a_probability_field_WARNS(metrics_config, caplog):
    """The runtime guard for the degenerate configuration above, caught at the point it would produce garbage rather than
    left to be noticed in a report. It names the fix — ``kind: probability`` — because the config edit is not obvious from
    the symptom."""
    import copy
    import logging

    rng = np.random.default_rng(0)
    observation = (rng.random((4, 16, 16)) < 0.05).astype(np.float64)
    prediction = np.clip(0.35 * observation + 0.1 * rng.random(observation.shape), 0, 1)

    config = copy.deepcopy(_hourly_config(metrics_config))
    config['metrics']['categorical']['thresholds'] = ['occurrence']       # the degenerate choice

    with caplog.at_level(logging.WARNING, logger='src.utils.metrics.evaluation'):
        run_metric_suite(config, prediction, observation, prediction,
                         {'zero': np.zeros_like(observation)}, None, {})

    assert any('kind: probability' in record.message for record in caplog.records), \
        [record.message for record in caplog.records]


# =====================================================================================================================
# dice_coefficient, wired as SOFT dice and guarded on the probability
# =====================================================================================================================
def test_dice_occurrence_is_emitted_from_the_OCCURRENCE_PROBABILITY(daily_suite, suite_arrays):
    """It was unreachable before Block 2r2 — no config requested it and ``evaluation.py`` never called it — while ``dice``
    was already a trainable loss alias. Wiring it makes the reported score the eval-time complement of ``dice_loss``."""
    from src.utils.metrics import scores

    flat, _ = daily_suite
    _, observation, probability, _, _ = suite_arrays
    assert 'dice_occurrence' in flat

    expected = scores.dice_coefficient(probability, (observation > 0).astype(float))
    assert flat['dice_occurrence'] == pytest.approx(expected, abs=1e-9)


def test_dice_is_ABSENT_on_the_hour_bands(daily_suite):
    """Guarded on ``use_probability``: on an hour band the field is in HOURS, where ``2*sum(p*o) / (sum(p) + sum(o))``
    mixes units and means nothing. The key is simply not emitted there."""
    flat, _ = daily_suite
    assert not [key for key in flat if key.startswith('dice_h')], \
        [key for key in flat if key.startswith('dice')]


def test_dice_is_absent_entirely_without_a_probability_forecast(suite_without_a_probability):
    flat, _ = suite_without_a_probability
    assert not [key for key in flat if key.startswith('dice')]


def test_the_hourly_dice_reads_the_probability_with_NO_decision_cut(hourly_suite):
    """``dice`` is in the threshold-free group: it needs the observed EVENT but no prediction cut, because soft Dice is
    defined directly on the probability. Binarising it first would throw away the calibration the score is measuring."""
    from src.utils.metrics import scores

    flat, _, prediction, observation = hourly_suite
    assert 'dice_p50' in flat
    expected = scores.dice_coefficient(prediction, (observation > 0).astype(float))
    assert flat['dice_p50'] == pytest.approx(expected, abs=1e-9)


def test_the_persistence_baseline_is_REJECTED(suite_arrays, metrics_config):
    """Removed deliberately: this is a diagnostic ERA5 -> lightning mapping, not a temporal forecast, so "yesterday's
    field" is not a baseline the task admits. Rejecting it by name beats silently scoring against something meaningless.
    """
    import pandas as pd

    from src.utils.metrics.evaluation import build_baselines

    items = pd.DataFrame({'date': pd.date_range('2010-07-14', periods=6), 'hour': [np.nan] * 6})
    with pytest.raises(ValueError, match='persistence'):
        build_baselines(['persistence'], {}, pd.DataFrame(), items, 'daily', 24)


# =====================================================================================================================
# Block 5c — the threshold value object and the dataset-level helpers
# =====================================================================================================================
def test_the_two_sides_of_a_SYMMETRIC_threshold_are_reported_as_one():
    """``is_symmetric`` is what tells a caller whether it may reuse one cut for both fields. Every ``kind: absolute``
    entry — the whole daily suite's h3/h6/h12 bands — is symmetric, because prediction and observation are the same
    quantity in the same units there."""
    resolved = resolve_threshold({'kind': 'absolute', 'value': 6.0, 'strict': False}, {}, (0.0, True))

    assert resolved.is_symmetric
    assert resolved.obs_event == (6.0, False)
    assert (resolved.pred_value, resolved.pred_strict) == (6.0, False)


def test_a_PROBABILITY_threshold_is_asymmetric_and_reads_the_labels_unchanged():
    """The hourly classification case, and the reason ``EventThreshold`` carries two sides at all. The observation is
    already a 0/1 event so its side is the occurrence event; the prediction is a probability so its side is a DECISION
    cut. A shared cut of ``> 0`` on a probability field fires on every cell with any non-zero probability."""
    resolved = resolve_threshold({'kind': 'probability', 'value': 0.5}, {}, (0.0, True))

    assert not resolved.is_symmetric
    assert resolved.obs_event == (0.0, True), 'the labels are read as they are, not re-thresholded'
    assert resolved.pred_value == 0.5


def test_obs_event_hands_back_exactly_the_pair_the_score_functions_take():
    """``scores.exceedance`` and ``diagnostics._occurrence_mask`` both take ``(value, strict)``. The property exists so
    the obs side can be passed straight through rather than unpacked at each of the six call sites."""
    resolved = resolve_threshold({'kind': 'occurrence'}, {}, (0.0, True))
    value, strict = resolved.obs_event
    assert (value, strict) == (0.0, True)
    assert scores.exceedance(np.array([0.0, 1.0]), value, strict).tolist() == [False, True]


def test_every_threshold_in_the_SHIPPED_suite_resolves(metrics_config):
    """Driven from the real ``config/eval/metrics.yaml`` rather than a fixture: an unresolvable kind raises inside the
    evaluation stage, long after the model has been trained."""
    resolved = resolve_thresholds(metrics_config, {'mode': 'daily'})

    assert set(resolved) == {'occurrence', 'h3', 'h6', 'h12'}, sorted(resolved)
    for name, threshold in resolved.items():
        assert isinstance(threshold.obs_value, float), name

    # The hour bands are symmetric; `occurrence` is NOT, and that asymmetry is the point. Its observed event is
    # `> 0` while its prediction side cuts at one whole hour, because a regression head emits small positive hours
    # nearly everywhere and a `> 0` prediction cut marks every cell an event (see
    # test_a_REGRESSION_occurrence_cut_at_zero_makes_the_table_DEGENERATE).
    for name in ('h3', 'h6', 'h12'):
        assert resolved[name].is_symmetric, f'{name}: an hour band cuts both sides at the same level'
    assert not resolved['occurrence'].is_symmetric


def test_the_resolver_keeps_the_suites_own_threshold_NAMES():
    """The names become metric-key suffixes (``ets_h6``, ``fss_h6_s3``), so a renamed threshold silently renames every
    metric that uses it and breaks the cross-family comparison table."""
    config = {'thresholds': {'h3': {'kind': 'absolute', 'value': 3.0},
                             'occurrence': {'kind': 'occurrence'}}}
    assert set(resolve_thresholds(config, {})) == {'h3', 'occurrence'}


def test_a_suite_with_no_thresholds_resolves_to_an_empty_mapping():
    """The continuous-only configuration. Raising here would make a suite that asks for no categorical scores
    unusable."""
    assert resolve_thresholds({}, {}) == {}


# ---------------------------------------------------------------------------------------------------------------------
# The dataset-level helpers — these read the prepared .npy files, so they cannot be tested with bare arrays
# ---------------------------------------------------------------------------------------------------------------------
def test_a_target_file_is_loaded_as_FLOAT32(tmp_path):
    """The prepared targets are written as float32 and some as smaller dtypes. The cast is what stops an integer-typed
    target from making every downstream mean an integer division."""
    from src.utils.metrics.evaluation import _load_target

    path = str(tmp_path / 'target.npy')
    np.save(path, np.array([[0, 3], [24, 1]], dtype=np.int16))

    loaded = _load_target(path)
    assert loaded.dtype == np.float32
    assert loaded.tolist() == [[0.0, 3.0], [24.0, 1.0]]


def test_the_observation_stack_preserves_ITEM_ORDER(prepared_split):
    """Predictions arrive in item order from the dataloader, and the stack is compared against them cell by cell. A
    reordering here would score every day against a different day's observation and still produce finite numbers."""
    from src.utils.metrics.evaluation import _load_target, _observation_stack

    split_index, _ = prepared_split(mode='daily', days_per_year=6)
    items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    stack = _observation_stack(items, 'daily')

    assert stack.shape[0] == len(items)
    for position in range(len(items)):
        assert np.array_equal(stack[position], _load_target(items.iloc[position]['target_file']))


def test_the_observation_stack_SLICES_THE_HOUR_in_hourly_mode(prepared_split):
    """An hourly item is one hour of a ``[T, H, W]`` day file. Without the slice the stack would be ``[N, T, H, W]``
    and every score would broadcast against the wrong shape."""
    from src.utils.metrics.evaluation import _load_target, _observation_stack

    split_index, config = prepared_split(mode='hourly', days_per_year=2, hours_per_day=4)
    items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    stack = _observation_stack(items, 'hourly')

    assert stack.ndim == 3, stack.shape
    day = _load_target(items.iloc[3]['target_file'])
    assert np.array_equal(stack[3], day[int(items.iloc[3]['hour'])])


def test_the_daily_climatology_is_a_per_DAY_OF_YEAR_mean_smoothed_over_a_window(prepared_split):
    """Not a single split-wide mean. Lightning is strongly seasonal, so a flat climatology would be an easy baseline in
    July and an impossible one in January — and the skill scores built on it would be meaningless in both."""
    from src.utils.metrics.evaluation import _climatology_tables

    split_index, _ = prepared_split(mode='daily', days_per_year=40)
    train = split_index[split_index['split'] == 'train']

    lookup = _climatology_tables(train, 'daily', 24, window_days=15, occurrence_event=(0.0, True))

    summer, _ = lookup(pd.Timestamp('2016-07-15'), None)
    winter, _ = lookup(pd.Timestamp('2016-01-15'), None)
    assert summer.shape == (16, 24)
    assert not np.array_equal(summer, winter), 'a flat climatology would return the same map for both'


def test_the_climatology_window_is_CIRCULAR_across_the_year_boundary(prepared_split):
    """31 December and 1 January are one day apart, not 364. Without the wrap the two ends of the year would each be
    averaged over half a window and be noisier than every other day."""
    from src.utils.metrics.evaluation import _climatology_tables

    split_index, _ = prepared_split(mode='daily', days_per_year=60)
    train = split_index[split_index['split'] == 'train']
    lookup = _climatology_tables(train, 'daily', 24, window_days=31, occurrence_event=(0.0, True))

    last, _ = lookup(pd.Timestamp('2016-12-31'), None)
    first, _ = lookup(pd.Timestamp('2016-01-01'), None)
    assert np.corrcoef(last.ravel(), first.ravel())[0, 1] > 0.5, \
        'the two ends of the year should share most of their window'


def test_the_climatology_also_accumulates_the_OCCURRENCE_FREQUENCY(prepared_split):
    """The second return value is the Brier / explained-deviance denominator. It is a FREQUENCY, so it must lie in
    ``[0, 1]`` — accumulating the target values instead would give a "probability" in hours and a skill score against
    a baseline that is not a probability at all."""
    from src.utils.metrics.evaluation import _climatology_tables

    split_index, _ = prepared_split(mode='daily', days_per_year=30)
    train = split_index[split_index['split'] == 'train']
    lookup = _climatology_tables(train, 'daily', 24, window_days=15, occurrence_event=(0.0, True))

    mean_map, frequency_map = lookup(pd.Timestamp('2016-07-15'), None)
    assert frequency_map.min() >= 0.0 and frequency_map.max() <= 1.0
    assert mean_map.max() > 1.0, 'the mean map is in hours, so the two are genuinely different quantities'


def test_the_hourly_climatology_keys_on_MONTH_AND_HOUR_OF_DAY(prepared_split):
    """The diurnal cycle is the dominant signal in hourly lightning — afternoon convection. A day-of-year climatology
    would average it away and hand every hour of the day the same baseline."""
    from src.utils.metrics.evaluation import _climatology_tables

    split_index, _ = prepared_split(mode='hourly', days_per_year=12, hours_per_day=4)
    train = split_index[split_index['split'] == 'train']

    lookup = _climatology_tables(train, 'hourly', 4, window_days=15, occurrence_event=(0.0, True))

    hour_zero, _ = lookup(pd.Timestamp('2016-07-15'), 0)
    hour_three, _ = lookup(pd.Timestamp('2016-07-15'), 3)
    assert hour_zero.shape == (16, 24)
    assert not np.array_equal(hour_zero, hour_three)


def test_the_shared_selection_reference_builds_every_denominator_from_ONE_climatology(prepared_split, metrics_config):
    """``_climatology_reference`` exists so ``climatology_brier`` and ``climatology_conditional_mae`` cannot drift apart
    in how they build the climatology or condition the cells — the two halves of a composite whose weights only mean
    something if their denominators agree."""
    from src.utils.metrics.evaluation import _climatology_reference

    split_index, prepared_config = prepared_split(mode='daily', days_per_year=20)
    eval_items = split_index[split_index['split'] == 'valid'].reset_index(drop=True)

    baselines, occurrence_probability, observation, occurrence, occurrence_event = _climatology_reference(
        split_index, eval_items, prepared_config, metrics_config['metrics'], {'mode': 'daily'})

    assert 'climatology' in baselines
    assert baselines['climatology'].shape == observation.shape
    assert occurrence_probability.shape == observation.shape
    assert occurrence.dtype == bool and occurrence.shape == observation.shape
    assert occurrence_event == (0.0, True)
    assert np.array_equal(occurrence, observation > 0.0), 'the mask must be the occurrence event applied to the obs'
