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
import pytest

from src.utils.metrics import scores
from src.utils.metrics.evaluation import (
    build_baselines, climatology_brier, climatology_conditional_mae, finalize_ensemble_metrics,
    merge_ensemble_partials, resolve_occurrence_event, resolve_threshold, run_metric_suite,
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


def test_resolve_threshold_occurrence_kind_is_symmetric():
    event = resolve_threshold({'kind': 'occurrence'}, {}, OCCURRENCE_EVENT)
    assert (event.pred_value, event.pred_strict) == OCCURRENCE_EVENT
    assert (event.obs_value, event.obs_strict) == OCCURRENCE_EVENT


def test_resolve_occurrence_event_rejects_a_reintroduced_threshold():
    """``occurrence_threshold`` was removed deliberately: occurrence is unconditionally ``target > 0``, and hourly
    denoising lives in the preparation stage. The HARD ASSERT is what stops it drifting back in as a config knob."""
    assert resolve_occurrence_event({}, {}) == OCCURRENCE_EVENT
    with pytest.raises(AssertionError, match='occurrence_threshold'):
        resolve_occurrence_event({'occurrence_threshold': 2.0}, {})


# =====================================================================================================================
# NEW: the two model-independent selection denominators — dataset-level, so they need a prepared directory
# =====================================================================================================================
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
