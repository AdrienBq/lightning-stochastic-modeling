"""Tests for src/utils/modeling/validation.py — the two composite validation scores trials are ranked on.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_selection_components.py``, which pinned the
mean-vs-pooled partition the diffusion scoring pass relies on. Adapted to the Step 3 signature: ``target_stats`` was
dropped (it existed only to read ``positive_quantiles``, whose last consumer went with the transform), and
``climatology_brier`` / ``occurrence_probability`` / ``occurrence_event`` were added for the classification composite.
``ets_p99`` is re-keyed to the absolute band ``ets_h6``.

The routing test is the important one: feeding an over-smoothed ensemble MEAN as the pointwise prediction but the
textured MEMBERS as the pooled structure stack must keep the PSD fidelity high. Scoring the structure on the mean
collapses it, and that collapse is exactly what the pooled split exists to avoid.
"""
import numpy as np
import pytest

from src.utils.modeling.validation import (
    DEFAULT_SELECTION_WEIGHTS, MODE_SELECTION_METRICS, compute_selection_components, selection_metric_for_mode,
    selection_score,
)

COMPONENT_KEYS = {
    'mae_cond_ss_climatology', 'average_precision_occurrence', 'brier_skill_score',
    'psd_full_fidelity', 'psd_high_fidelity', 'ets_h6', 'fss_occurrence_scale3',
}


@pytest.fixture
def fixture():
    """Textured bounded observations, a heavily over-smoothed per-map-constant 'mean' prediction, and a pooled member
    stack that retains the observed texture.

    The target is 0-24 integer lightning-hours, not A's unbounded gamma draws: the composite is computed in the target
    space and the whole point of the classification-first scope is that the space is bounded.
    """
    rng = np.random.default_rng(0)
    n, h, w = 8, 32, 32
    observation = np.clip(np.round(rng.gamma(1.2, 2.0, size=(n, h, w))), 0, 24).astype(np.float32)

    # the ensemble MEAN is over-smoothed: a per-map constant has ~zero alternating-current power, so scoring the PSD
    # on it collapses the full-band ratio toward 0 (fidelity -> 0)
    mean_prediction = np.broadcast_to(
        observation.mean(axis=(1, 2), keepdims=True), observation.shape
    ).astype(np.float32).copy()

    members = 4
    pooled_prediction = np.repeat(observation, members, axis=0) + \
        0.05 * rng.standard_normal((n * members, h, w)).astype(np.float32)
    pooled_observation = np.repeat(observation, members, axis=0)

    climatology_cond_mae = float(np.mean(np.abs(observation)))       # a finite, model-independent denominator
    return mean_prediction, observation, pooled_prediction, pooled_observation, climatology_cond_mae


# =====================================================================================================================
# The component set
# =====================================================================================================================
def test_every_component_key_is_returned_on_every_call(fixture):
    """Every key on every call, whichever composite is active: the two PSD fidelities share one FFT pass and the
    ranking metrics one binned accumulation, so computing the superset costs almost nothing and keeps the trials table
    comparable across families."""
    prediction, observation, _, _, climatology = fixture
    components = compute_selection_components(prediction, observation, climatology_cond_mae=climatology)
    assert set(components) == COMPONENT_KEYS


def test_the_two_default_weightings_are_the_DECIDED_ones():
    """These numbers resolved a three-way disagreement in the repo: the YAML's weights said 0.50/0.50, the YAML's own
    prose said ``0.40 AP + 0.30 + 0.30``, and ``validation.py``'s docstring said 0.60/0.40. The docstring was right.

    ⚠️ The classification composite keeps ``psd_full_fidelity`` at its full 0.30 despite the Block 2r2 finding that the
    structure scores are BIASED against a calibrated probability field — a calibrated map is intrinsically smoother than
    the 0/1 field it is compared against. Accepted deliberately, so the weight is pinned rather than left to drift.
    """
    assert DEFAULT_SELECTION_WEIGHTS == {
        'valid_regression_score': {'mae_cond_ss_climatology': 0.60, 'psd_full_fidelity': 0.40},
        'valid_classification_score': {'average_precision_occurrence': 0.50, 'brier_skill_score': 0.20,
                                       'psd_full_fidelity': 0.30},
    }


def test_every_shipped_space_declares_the_regression_composite_and_its_weights(search_spaces):
    """The three ``search_space_daily.yaml`` belong to the nine DAILY pipelines, so all three must name the regression
    composite — and their weights must equal the module default for it, or a family would rank on a different score from
    its siblings while the comparison table presented the numbers as equivalent.

    ``search_space_hourly.yaml`` is deliberately outside the ``search_spaces`` fixture and is checked below: it is the
    one shipped space that must name the OTHER composite.
    """
    for family, space in search_spaces.items():
        selection = space['selection']
        assert selection['metric'] == 'valid_regression_score', family
        assert selection['components'] == DEFAULT_SELECTION_WEIGHTS['valid_regression_score'], family
        assert selection_metric_for_mode('daily', selection['metric']) == 'valid_regression_score', family


def test_the_HOURLY_space_declares_the_classification_composite_and_its_weights(search_space_hourly):
    """⭐ The hourly counterpart (Step 4 block 4f), and the one search space where getting this wrong RAISES rather than
    drifting: ``selection_metric_for_mode`` rejects a declaration that disagrees with the prepared mode, on the grounds
    that a file naming the wrong composite has the wrong component weights too.

    Both directions are asserted, because the raise is only half the protection — a space could name the right composite
    and still carry the regression weights, which nothing in the code would notice.
    """
    selection = search_space_hourly['selection']
    assert selection['metric'] == 'valid_classification_score'
    assert selection['mode'] == 'max'
    assert selection['components'] == DEFAULT_SELECTION_WEIGHTS['valid_classification_score']
    assert selection_metric_for_mode('hourly', selection['metric']) == 'valid_classification_score'
    with pytest.raises(ValueError):
        selection_metric_for_mode('daily', selection['metric'])


def test_the_hourly_space_weights_are_exactly_the_emitted_keys(search_space_hourly):
    """Same check as the daily one below, on the composite whose components were dead until block 4f gave them a
    pipeline: ``brier_skill_score`` needs a calibrated probability, which no daily run produces."""
    weights = set(search_space_hourly['selection']['components'])
    assert weights <= COMPONENT_KEYS, f'unknown components: {sorted(weights - COMPONENT_KEYS)}'


def test_search_space_weights_are_exactly_the_emitted_keys(search_spaces):
    """The check that would have caught the 0.50/0.50-vs-prose drift: a weight naming a component the function does
    not emit contributes nothing silently, and the trial ranks on a composite missing that term."""
    for family, space in search_spaces.items():
        weights = set(space['selection']['components'])
        assert weights <= COMPONENT_KEYS, f'{family} weights unknown components: {sorted(weights - COMPONENT_KEYS)}'


def test_every_component_is_FINITE_when_all_its_inputs_are_supplied(fixture):
    """The complement of the NaN tests below: given the probability and the climatology denominators, nothing is NaN. A
    silently NaN component contributes 0 to the composite, so a trial would rank on a partial score without any signal
    that a term dropped out."""
    prediction, observation, _, _, climatology = fixture
    probability = np.clip(prediction / 24.0, 0.0, 1.0)
    occurrence = (observation > 0).astype(np.float32)

    components = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology,
        occurrence_probability=probability, occurrence_event=(0.0, True),
        climatology_brier=float(np.mean((occurrence.mean() - occurrence) ** 2)),
    )
    not_finite = {key: value for key, value in components.items() if not np.isfinite(value)}
    assert not not_finite, not_finite


def test_the_average_precision_IS_finalize_ranking_metrics_on_the_same_inputs(fixture):
    """One implementation, not two — the ``crps_ensemble`` divergence lesson applied to the ranking path. AP carries
    0.50 of the classification composite, so a second implementation drifting from the reported one would change which
    trial wins without changing any reported number."""
    from src.utils.metrics import scores

    prediction, observation, _, _, climatology = fixture
    probability = np.clip(prediction / 24.0, 0.0, 1.0)
    occurrence = (observation > 0).astype(np.float32)

    components = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology,
        occurrence_probability=probability, occurrence_event=(0.0, True),
    )

    edges = scores.ranking_bin_edges()
    partials = scores.ranking_partials(probability.ravel(), occurrence.ravel(), edges)
    expected = scores.finalize_ranking_metrics(partials, edges)['average_precision']
    assert components['average_precision_occurrence'] == pytest.approx(expected, abs=1e-12)


def test_the_brier_skill_score_IS_one_minus_the_ratio_to_its_denominator(fixture):
    """The definition, pinned against the arithmetic. The denominator is the per-cell day-of-year climatology, which is
    STRONGER than the constant base rate ``r2_score`` implicitly uses — so ``r2 > 0`` with ``brier_skill_score < 0``
    means the model beats the base rate but not the climatology."""
    from src.utils.metrics import scores

    prediction, observation, _, _, climatology = fixture
    probability = np.clip(prediction / 24.0, 0.0, 1.0)
    occurrence = (observation > 0).astype(np.float32)
    denominator = 0.05

    components = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology,
        occurrence_probability=probability, occurrence_event=(0.0, True),
        climatology_brier=denominator,
    )
    expected = 1.0 - scores.brier_score(probability, occurrence) / denominator
    assert components['brier_skill_score'] == pytest.approx(expected, abs=1e-9)


def test_ets_h6_matches_an_INDEPENDENT_contingency_table_at_six_hours(fixture):
    """``ets_h6`` was re-keyed from A's ``ets_p99`` so that the trials table uses the same threshold definition as
    ``config/eval/metrics_daily.yaml`` — an absolute 6-hour band rather than a distribution quantile. This computes the table
    from scratch at that threshold, so the re-keying is checked rather than assumed."""
    from src.utils.metrics import scores

    prediction, observation, _, _, _ = fixture
    components = compute_selection_components(prediction, observation)

    expected = scores.categorical_scores(*scores.contingency_counts(prediction, observation, 6.0))['ets']
    assert components['ets_h6'] == pytest.approx(expected, abs=1e-9)


def test_compute_selection_components_takes_NO_target_stats(fixture):
    """It dropped the argument in block 3b-2 along with ``positive_quantiles`` — the last consumer of that block of
    ``target_stats.json``. A caller still passing it positionally would land the dict in the next parameter."""
    import inspect

    parameters = inspect.signature(compute_selection_components).parameters
    assert 'target_stats' not in parameters, sorted(parameters)


def test_mae_cond_skill_is_nan_without_its_denominator(fixture):
    """The skill term needs the model-INDEPENDENT climatology denominator. Without it the component is NaN rather
    than silently falling back to an un-normalized MAE, which would be on a different scale from the [0, 1] PSD term
    it is combined with."""
    prediction, observation, _, _, _ = fixture
    components = compute_selection_components(prediction, observation)
    assert np.isnan(components['mae_cond_ss_climatology'])


def test_brier_skill_is_nan_without_a_probabilistic_forecast(fixture):
    """``brier_skill_score`` carries 0.20 of the classification composite, and a daily run has no probability head —
    so it must be NaN there rather than a number computed from lightning-hours."""
    prediction, observation, _, _, climatology = fixture
    components = compute_selection_components(prediction, observation, climatology_cond_mae=climatology)
    assert np.isnan(components['brier_skill_score'])


def test_average_precision_ranks_on_the_probability_when_one_is_given(fixture):
    """AP is the only component that sees a FALSE ALARM, which is the recorded mitigation for the regression
    composite having no false-alarm term. It ranks on the occurrence probability when the model emits one and on the
    prediction otherwise — and AP is invariant to any monotone rescaling, so ranking on predicted hours is exact
    rather than approximate."""
    prediction, observation, _, _, climatology = fixture
    occurrence = (observation > 0).astype(np.float32)

    on_prediction = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology
    )['average_precision_occurrence']
    # a monotone rescaling of the ranking field must not move AP at all
    rescaled = compute_selection_components(
        prediction * 3.0 + 1.0, observation, climatology_cond_mae=climatology
    )['average_precision_occurrence']
    assert np.isclose(on_prediction, rescaled, rtol=1e-9)

    # a probability that ranks the events perfectly must beat the over-smoothed prediction
    perfect = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology, occurrence_probability=occurrence
    )['average_precision_occurrence']
    assert perfect > on_prediction


# =====================================================================================================================
# Structure routing: the pointwise and structure pairs are scored separately  (ported)
# =====================================================================================================================
def test_structure_default_is_the_pointwise_pair(fixture):
    """Omitting the structure arguments == passing structure equal to the pointwise pair, which is the contract the
    deterministic U-net and a single-draw diffusion rely on."""
    prediction, observation, _, _, climatology = fixture
    default = compute_selection_components(prediction, observation, climatology_cond_mae=climatology)
    explicit = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology,
        prediction_structure=prediction, observation_structure=observation
    )
    assert default.keys() == explicit.keys()
    for key in default:
        assert np.isclose(default[key], explicit[key], rtol=1e-12, atol=1e-12, equal_nan=True), key


def test_pointwise_components_ignore_the_structure_stack(fixture):
    """The conditional-MAE skill and ETS depend only on the pointwise pair, so routing a different structure stack
    must not move them."""
    prediction, observation, pooled_prediction, pooled_observation, climatology = fixture
    without = compute_selection_components(prediction, observation, climatology_cond_mae=climatology)
    with_structure = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology,
        prediction_structure=pooled_prediction, observation_structure=pooled_observation
    )
    for key in ('mae_cond_ss_climatology', 'ets_h6'):
        assert np.isclose(without[key], with_structure[key], rtol=1e-12, atol=1e-12), key


def test_pooled_structure_is_texture_faithful(fixture):
    """Scoring the structure on the textured pooled members keeps the PSD fidelity high; scoring it on the
    over-smoothed mean collapses it. This is the over-smoothing the pooled split exists to avoid."""
    prediction, observation, pooled_prediction, pooled_observation, climatology = fixture
    on_mean = compute_selection_components(prediction, observation, climatology_cond_mae=climatology)
    on_pooled = compute_selection_components(
        prediction, observation, climatology_cond_mae=climatology,
        prediction_structure=pooled_prediction, observation_structure=pooled_observation
    )
    assert on_pooled['psd_full_fidelity'] > on_mean['psd_full_fidelity'] + 0.2
    assert on_pooled['psd_high_fidelity'] > on_mean['psd_high_fidelity']


# =====================================================================================================================
# NEW: the composite is chosen by the MODE, and a disagreeing search space raises
# =====================================================================================================================
def test_the_mode_selects_the_composite():
    assert selection_metric_for_mode('daily') == 'valid_regression_score'
    assert selection_metric_for_mode('hourly') == 'valid_classification_score'
    assert set(MODE_SELECTION_METRICS.values()) == set(DEFAULT_SELECTION_WEIGHTS)


def test_the_deprecated_mode_alias_still_resolves():
    """Artifacts prepared under the old name must keep loading — ``daily_lightning_hours`` survives only as an alias
    in ``normalize_mode`` and appears nowhere in config/."""
    assert selection_metric_for_mode('daily_lightning_hours') == 'valid_regression_score'


def test_a_search_space_declaring_the_other_composite_raises():
    """The mode wins by construction. Raising rather than silently overriding is the point: a search space asking for
    the regression composite on a binary target has its WEIGHTS wrong too, and swapping just the name would leave
    those in place."""
    assert selection_metric_for_mode('daily', declared_metric='valid_regression_score') == 'valid_regression_score'
    with pytest.raises(ValueError, match='valid_classification_score'):
        selection_metric_for_mode('hourly', declared_metric='valid_regression_score')


@pytest.mark.parametrize('metric', ['valid_regression_score', 'valid_classification_score'])
def test_the_default_weights_sum_to_one(metric):
    """Not cosmetic: the composite is a weighted mean of terms that are each roughly [0, 1], so weights summing to
    something else would make the two composites incomparable in magnitude."""
    assert np.isclose(sum(DEFAULT_SELECTION_WEIGHTS[metric].values()), 1.0)


def test_selection_score_is_the_hand_computed_weighted_sum():
    components = {'mae_cond_ss_climatology': 0.5, 'psd_full_fidelity': 0.25, 'average_precision_occurrence': 0.9}
    weights = {'mae_cond_ss_climatology': 0.6, 'psd_full_fidelity': 0.4}
    assert np.isclose(selection_score(components, weights), 0.6 * 0.5 + 0.4 * 0.25)


def test_a_nan_component_contributes_zero_rather_than_poisoning_the_score():
    """A NaN term must not turn the whole composite into NaN — a daily trial has no ``brier_skill_score`` and would
    otherwise be unrankable, so optuna would see NaN for every trial and prune nothing."""
    components = {'mae_cond_ss_climatology': 0.5, 'psd_full_fidelity': float('nan')}
    weights = {'mae_cond_ss_climatology': 0.6, 'psd_full_fidelity': 0.4}
    score = selection_score(components, weights)
    assert np.isfinite(score) and np.isclose(score, 0.6 * 0.5)


def test_a_missing_component_contributes_zero():
    assert np.isclose(selection_score({'mae_cond_ss_climatology': 1.0},
                                      {'mae_cond_ss_climatology': 0.6, 'psd_full_fidelity': 0.4}), 0.6)
