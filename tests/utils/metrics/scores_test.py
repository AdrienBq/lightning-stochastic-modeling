"""Tests for src/utils/metrics/scores.py.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_ensemble_scores.py`` (the ensemble estimators), plus new
coverage for the scores the merge added or re-derived.

The three imported ensemble estimators are checked against their O(M^2) brute-force definitions rather than against a
golden number, because the streaming order-statistic implementation must reproduce the definition EXACTLY — branch A
and branch D shipped two ``crps_ensemble`` functions with different return types, and picking one silently was the
merge's headline risk (rebuild-plan risk #9).

⚠️ The classification prediction is a PROBABILITY, not a 0/1 field (Block 2r2). Several tests below pin identities
that only make sense under that reading, and one pins that ``mae`` is IMPROPER on it — the sharpest consequence.
"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.metrics import scores


# =====================================================================================================================
# Ensemble estimators, against the brute-force definition  (ported: test_ensemble_scores.py)
# =====================================================================================================================
def _brute_crps_cell(members_cell, y):
    """Reference CRPS / almost-fair CRPS for one cell from the definition E|X-y| - (1/2) E|X-X'|."""
    m = members_cell.shape[0]
    mae = float(np.mean(np.abs(members_cell - y)))
    pairwise = float(np.abs(members_cell[:, None] - members_cell[None, :]).sum())
    spread_term = 0.5 * pairwise / (m * m)
    return mae - spread_term, mae - ((m - 1.0) / m) * spread_term


def test_crps_matches_bruteforce():
    rng = np.random.default_rng(0)
    m, k = 11, 60
    members = rng.gamma(1.2, 1.0, size=(m, k))
    obs = rng.gamma(1.2, 1.0, size=k)

    crps_cells, af_cells = zip(*(_brute_crps_cell(members[:, c], obs[c]) for c in range(k)))
    assert np.isclose(scores.crps_ensemble(members, obs), np.mean(crps_cells), rtol=1e-10, atol=1e-12)
    assert np.isclose(scores.almost_fair_crps_ensemble(members, obs), np.mean(af_cells), rtol=1e-10, atol=1e-12)


def test_crps_ensemble_returns_a_scalar_float():
    """Rebuild-plan risk #9, and the one thing no other test would catch.

    Branch A's ``crps_ensemble`` returns a ``float`` and accepts ``condition=``; branch D's returns a per-element
    ARRAY. Both are named identically, so importing the wrong one type-checks, runs, and silently turns every
    downstream mean into a mean-of-means over a different denominator. A's contract is the one this repo kept.
    """
    rng = np.random.default_rng(1)
    members, obs = rng.gamma(1.0, 1.0, size=(5, 40)), rng.gamma(1.0, 1.0, size=40)
    value = scores.crps_ensemble(members, obs)
    assert isinstance(value, float) and not isinstance(value, np.ndarray)
    assert np.ndim(value) == 0
    assert isinstance(scores.almost_fair_crps_ensemble(members, obs), float)


def test_crps_conditioning_selects_cells():
    rng = np.random.default_rng(7)
    m, k = 9, 80
    members = rng.gamma(1.0, 1.0, size=(m, k))
    obs = rng.gamma(1.0, 1.0, size=k)
    condition = obs >= 1.0

    crps_sum, _, n = scores.crps_sums(members, obs, condition=condition)
    assert n == int(condition.sum())
    expected = np.mean([_brute_crps_cell(members[:, c], obs[c])[0] for c in range(k) if condition[c]])
    assert np.isclose(crps_sum / n, expected, rtol=1e-10, atol=1e-12)


def test_spread_skill_ratio_formula_and_monotonicity():
    rng = np.random.default_rng(1)
    m, k = 16, 4000
    obs = rng.normal(0.0, 1.0, size=k)
    # FIXED zero-mean perturbations (sum(z) == 0): the ensemble mean is obs + bias exactly, so the skill (RMSE of the
    # mean) is the constant bias and is INDEPENDENT of the spread amplitude. With unbiased iid noise, spread and
    # mean-error scale together and the ratio is fixed, so the monotonicity below could not be observed at all.
    z = (np.arange(m, dtype=float) - (m - 1) / 2.0)[:, None]
    bias = 0.5
    members_narrow = obs[None, :] + bias + 0.3 * z
    members_wide = obs[None, :] + bias + 0.6 * z

    var_sum, sqerr_sum, n = scores.spread_skill_sums(members_narrow, obs)
    spread, skill = np.sqrt(var_sum / n), np.sqrt(sqerr_sum / n)
    assert np.isclose(scores.spread_skill_ratio(members_narrow, obs), spread / skill, rtol=1e-12)
    assert np.isclose(skill, bias, rtol=1e-6)
    assert np.isclose(scores.spread_skill_ratio(members_wide, obs),
                      2.0 * scores.spread_skill_ratio(members_narrow, obs), rtol=1e-6)


def test_spread_skill_is_nan_for_a_single_member():
    """CLAUDE.md invariant: ``ensemble-size`` must be >= 2 because ``spread_skill_sums`` uses ``ddof=1``, so a
    one-member ensemble yields NaN rather than erroring — which is what bites smoke configs.

    The RuntimeWarning is asserted rather than silenced: it pins that the NaN comes from the ddof=1 variance being
    undefined, not from some other degeneracy that happens to produce the same NaN.
    """
    rng = np.random.default_rng(2)
    obs = rng.gamma(1.0, 1.0, size=50)
    with pytest.warns(RuntimeWarning, match='Degrees of freedom'):
        assert np.isnan(scores.spread_skill_ratio(obs[None, :], obs))


def test_rank_histogram_flat_for_calibrated_ensemble():
    rng = np.random.default_rng(2)
    m, k = 20, 40000
    pooled = rng.normal(0.0, 1.0, size=(m + 1, k))          # obs + members exchangeable -> uniform rank
    members, obs = pooled[:m], pooled[m]

    counts = scores.rank_histogram_counts(members, obs, rng=np.random.default_rng(3))
    assert counts.shape == (m + 1,)
    assert int(counts.sum()) == k
    reliability_flat = scores.rank_histogram_reliability(counts)
    assert reliability_flat < 0.1

    members_tight = rng.normal(0.0, 0.02, size=(m, k))      # under-dispersed -> U-shaped
    obs_wide = rng.normal(0.0, 1.0, size=k)
    counts_under = scores.rank_histogram_counts(members_tight, obs_wide, rng=np.random.default_rng(4))
    assert scores.rank_histogram_reliability(counts_under) > reliability_flat


def test_streaming_partials_are_additive():
    """The invariant the whole streaming design rests on: partials are SUMS, so they add across batches. Means and
    ratios do not, which is why ``ensemble_partials`` returns sums and the division happens once at the end."""
    rng = np.random.default_rng(5)
    m, k = 8, 240
    members = rng.gamma(1.0, 1.0, size=(m, k))
    obs = rng.gamma(1.0, 1.0, size=k)
    occurrence = (0.5, False)

    whole = scores.ensemble_partials(members, obs, occurrence_event=occurrence, rng=np.random.default_rng(0))
    split = k // 2
    left = scores.ensemble_partials(members[:, :split], obs[:split], occurrence_event=occurrence,
                                   rng=np.random.default_rng(0))
    right = scores.ensemble_partials(members[:, split:], obs[split:], occurrence_event=occurrence,
                                     rng=np.random.default_rng(0))

    for key in ('crps_sum', 'af_crps_sum', 'crps_n', 'crps_sum_occ', 'af_crps_sum_occ', 'crps_n_occ',
                'var_sum', 'sqerr_sum', 'ss_n'):
        assert np.isclose(whole[key], left[key] + right[key], rtol=1e-9, atol=1e-9), key
    # gamma draws have no ties, so ranks are deterministic and the counts add exactly
    assert np.array_equal(whole['rank_counts'], left['rank_counts'] + right['rank_counts'])


# =====================================================================================================================
# Pooled-over-members structure scores  (ported)
# =====================================================================================================================
def test_pooled_fss_matches_user_formula():
    """The pooled-over-members FSS (``fss`` on the ``[N*M, H, W]`` stack with obs replicated M times) equals
    ``1 - sum||f(pred)-f(obs)||^2 / (sum||f(pred)||^2 + M*sum||f(obs)||^2)``."""
    from scipy.ndimage import uniform_filter

    rng = np.random.default_rng(10)
    n, m, h, w = 4, 3, 12, 12
    members = rng.gamma(1.0, 1.0, size=(n, m, h, w))
    obs = rng.gamma(1.0, 1.0, size=(n, h, w))
    threshold, scale = 1.0, 3

    fss_pooled = scores.fss(members.reshape(n * m, h, w), np.repeat(obs, m, axis=0), threshold, scale)

    def fraction(field):
        return uniform_filter((field >= threshold).astype(np.float64), size=scale, mode='constant', cval=0.0)

    numerator, den_pred, den_obs = 0.0, 0.0, 0.0
    for i in range(n):
        f_obs = fraction(obs[i])
        den_obs += float((f_obs ** 2).sum())
        for member in range(m):
            f_pred = fraction(members[i, member])
            numerator += float(((f_pred - f_obs) ** 2).sum())
            den_pred += float((f_pred ** 2).sum())

    assert np.isclose(fss_pooled, 1.0 - numerator / (den_pred + m * den_obs), rtol=1e-10, atol=1e-12)


def test_pooled_ratio_scores_are_invariant_to_obs_replication():
    """The ratio-of-means structure scores are unchanged whether obs is passed once ``[N]`` or replicated to
    ``[N*M]`` — replication only matters for the PAIRED score (FSS)."""
    rng = np.random.default_rng(11)
    n, m, h, w = 5, 4, 16, 16
    members = rng.gamma(1.0, 1.0, size=(n, m, h, w))
    obs = rng.gamma(1.0, 1.0, size=(n, h, w))
    pooled_pred = members.reshape(n * m, h, w)
    obs_replicated = np.repeat(obs, m, axis=0)
    bands = {'low': (8.0, np.inf), 'high': (2.0, 8.0)}

    for score in (lambda o: scores.psd_band_ratios(pooled_pred, o, bands)['high'],
                  lambda o: scores.sharpness_ratio(pooled_pred, o),
                  lambda o: scores.variance_ratio(pooled_pred, o)):
        assert np.isclose(score(obs), score(obs_replicated), rtol=1e-10, atol=1e-12)

    # pooling over members is a distinct (lower-variance) estimate from a single member's structure
    assert not np.isclose(scores.psd_band_ratios(pooled_pred, obs_replicated, bands)['high'],
                          scores.psd_band_ratios(members[:, 0], obs, bands)['high'], rtol=1e-12, atol=1e-12)


def test_psd_cache_matches_fresh_computation():
    """The precomputed-spectrum fast path must be NUMERICALLY IDENTICAL to recomputing, which is what makes the
    ``run_metric_suite`` PSD caching safe for every caller rather than just the diffusion ensemble."""
    rng = np.random.default_rng(12)
    k, h, w = 9, 16, 16
    pred = rng.gamma(1.0, 1.0, size=(k, h, w))
    obs = rng.gamma(1.0, 1.0, size=(k, h, w))
    bands = {'low': (8.0, np.inf), 'high': (2.0, 8.0)}
    pred_spectrum, obs_spectrum = scores.mean_power_spectrum(pred), scores.mean_power_spectrum(obs)

    fresh_wavelengths, fresh_power = scores.radial_psd(pred)
    cached_wavelengths, cached_power = scores.radial_psd(pred, spectrum=pred_spectrum)
    assert np.array_equal(fresh_wavelengths, cached_wavelengths)
    assert np.array_equal(fresh_power, cached_power)

    fresh = scores.psd_band_ratios(pred, obs, bands)
    cached = scores.psd_band_ratios(pred, obs, bands, pred_spectrum=pred_spectrum, obs_spectrum=obs_spectrum)
    assert fresh.keys() == cached.keys()
    for key in fresh:
        assert (np.isnan(fresh[key]) and np.isnan(cached[key])) or fresh[key] == cached[key], key

    assert scores.log_spectral_distance(pred, obs) == scores.log_spectral_distance(
        pred, obs, pred_spectrum=pred_spectrum, obs_spectrum=obs_spectrum
    )


def test_psd_fidelity_peaks_at_a_unit_ratio():
    assert scores.psd_fidelity(1.0) == 1.0
    assert scores.psd_fidelity(0.5) == scores.psd_fidelity(1.5)      # symmetric in |1 - ratio|
    assert scores.psd_fidelity(0.0) == 0.0
    assert scores.psd_fidelity(5.0) == 0.0                           # clipped, not negative


def test_fss_useful_scale_reuses_the_per_scale_curve():
    """``fss_useful_scale`` derives each map's fields ONCE and reuses them across scales, where calling ``fss`` per
    scale re-derives the whole stack each time. Same numbers, one pass — so the useful scale costs nothing beyond
    the per-scale values the suite already emits."""
    rng = np.random.default_rng(13)
    obs = (rng.random((6, 20, 20)) < 0.1).astype(np.float64)
    pred = np.clip(obs + rng.normal(0, 0.3, obs.shape), 0, 1)
    scales = [1, 3, 5, 9]

    useful, per_scale = scores.fss_useful_scale(pred, obs, 0.5, scales)
    assert set(per_scale) == set(scales)
    for scale in scales:
        assert np.isclose(per_scale[scale], scores.fss(pred, obs, 0.5, scale), rtol=1e-12), scale

    criterion = 0.5 + float(obs.mean()) / 2.0
    if np.isnan(useful):
        assert all(per_scale[scale] < criterion for scale in scales)
    else:
        assert per_scale[useful] >= criterion
        assert all(per_scale[scale] < criterion for scale in scales if scale < useful)   # SMALLEST such scale


# =====================================================================================================================
# NEW: the ranking metrics are streaming and binned, but must agree with sklearn  (Block 2r)
# =====================================================================================================================
@pytest.mark.parametrize('regime,seed', [('calibrated_sparse', 0), ('over_confident', 1), ('mid_range', 2)])
def test_roc_auc_and_average_precision_match_sklearn(regime, seed):
    """The implementation accumulates per-bin label counts (summable across batches) instead of holding the whole
    score array, so agreement with sklearn is close but not exact. 4000 mirrored-log bins keep AUC within ~1e-4 and
    AP within ~1 % across the regimes this project actually sees."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(40000) < 0.02).astype(int)
    if regime == 'calibrated_sparse':                       # nearly all probability mass below 0.01
        probability = np.clip(rng.beta(0.3, 60.0, size=labels.size) + labels * 0.05, 0, 1)
    elif regime == 'over_confident':
        probability = np.clip(rng.beta(2.0, 2.0, size=labels.size) + labels * 0.2, 0, 1)
    else:
        probability = 1.0 / (1.0 + np.exp(-rng.normal(labels * 1.6, 1.0)))

    assert np.isclose(scores.roc_auc(probability, labels), roc_auc_score(labels, probability), atol=2e-3)
    assert np.isclose(scores.average_precision(probability, labels),
                      average_precision_score(labels, probability), rtol=3e-2)


def test_ranking_metrics_are_exact_on_a_discrete_hour_field():
    """The DAILY task is exact at any bin count: a 25-value integer field has no within-bin mixing, so the binned
    estimator and sklearn agree to floating point."""
    rng = np.random.default_rng(3)
    hours = rng.integers(0, 25, size=20000).astype(np.float64)
    labels = (hours > 0).astype(int)
    prediction = np.clip(hours + rng.integers(-2, 3, size=hours.size), 0, 24).astype(np.float64)

    assert np.isclose(scores.roc_auc(prediction, labels, score_max=24.0),
                      roc_auc_score(labels, prediction), atol=1e-9)
    assert np.isclose(scores.average_precision(prediction, labels, score_max=24.0),
                      average_precision_score(labels, prediction), rtol=1e-6)


def test_ranking_partials_are_additive_and_batch_invariant():
    rng = np.random.default_rng(4)
    labels = (rng.random(3000) < 0.05).astype(int)
    probability = np.clip(rng.beta(0.5, 20.0, size=labels.size) + labels * 0.1, 0, 1)
    edges = scores.ranking_bin_edges()

    whole = scores.ranking_partials(probability, labels, edges)
    accumulated = None
    for chunk in (slice(0, 1000), slice(1000, 2000), slice(2000, 3000)):
        part = scores.ranking_partials(probability[chunk], labels[chunk], edges)
        accumulated = part if accumulated is None else {
            key: accumulated[key] + part[key] for key in part
        }
    for key in ('positive_counts', 'negative_counts'):
        assert np.array_equal(whole[key], accumulated[key]), key

    single = scores.finalize_ranking_metrics(whole, edges)
    batched = scores.finalize_ranking_metrics(accumulated, edges)
    assert single['roc_auc'] == batched['roc_auc']
    assert single['average_precision'] == batched['average_precision']


def test_ranking_bin_edges_are_dense_at_both_ends():
    """Mirrored-geometric spacing. Uniform bins fail exactly where this project lives: a calibrated forecast on a
    0.07 % base rate puts nearly all its probabilities below 0.01, which uniform bins barely resolve."""
    edges = scores.ranking_bin_edges(n_bins=1000)
    assert edges[0] == 0.0 and np.isclose(edges[-1], 1.0)
    assert np.all(np.diff(edges) > 0)
    widths = np.diff(edges)
    assert widths[0] < widths[len(widths) // 2]                  # dense near 0
    assert widths[-1] < widths[len(widths) // 2]                 # and near 1


# =====================================================================================================================
# NEW: the scores the merge added
# =====================================================================================================================
def test_explained_deviance_is_zero_at_climatology_and_one_at_perfection():
    rng = np.random.default_rng(5)
    labels = (rng.random(5000) < 0.1).astype(float)
    base_rate = float(labels.mean())

    climatology = np.full_like(labels, base_rate)
    assert abs(scores.explained_deviance(climatology, labels, base_rate)) < 1e-9

    near_perfect = np.where(labels > 0, 1.0 - 1e-9, 1e-9)
    assert scores.explained_deviance(near_perfect, labels, base_rate) > 0.99


def test_dice_coefficient_is_soft_dice_on_probabilities():
    """``2*sum(p*o)/(sum(p)+sum(o))`` needs no binarization, so on a probability field it IS soft Dice — the
    eval-time complement of ``dice_loss``, which is what makes the reported score match the trained objective."""
    rng = np.random.default_rng(6)
    obs = (rng.random((4, 12, 12)) < 0.1).astype(np.float64)
    probability = np.clip(obs * 0.7 + rng.random(obs.shape) * 0.2, 0, 1)

    smooth = 1.0
    expected = (2.0 * (probability * obs).sum() + smooth) / (probability.sum() + obs.sum() + smooth)
    assert np.isclose(scores.dice_coefficient(probability, obs, smooth=smooth), expected, rtol=1e-12)

    # unchanged on already-binarized inputs (the score is a generalisation, not a replacement)
    hard = (probability >= 0.5).astype(np.float64)
    expected_hard = (2.0 * (hard * obs).sum() + smooth) / (hard.sum() + obs.sum() + smooth)
    assert np.isclose(scores.dice_coefficient(hard, obs, smooth=smooth), expected_hard, rtol=1e-12)


# =====================================================================================================================
# NEW: the Block 2r2 identities — the classification prediction is a PROBABILITY
# =====================================================================================================================
@pytest.fixture
def sparse_probability_forecast():
    """A calibrated probability field against binary labels at a realistic base rate."""
    rng = np.random.default_rng(7)
    labels = (rng.random(50000) < 0.02).astype(np.float64)
    probability = np.clip(rng.beta(0.4, 40.0, size=labels.size) + labels * 0.3, 0.0, 1.0)
    return probability, labels


def test_rmse_squared_equals_the_brier_score(sparse_probability_forecast):
    """On a probability field ``rmse`` is exactly ``sqrt(brier_score)``, hence PROPER. This is why the continuous
    group is tagged for both tasks rather than "degenerate on a 0/1 target"."""
    probability, labels = sparse_probability_forecast
    assert np.isclose(scores.rmse(probability, labels) ** 2, scores.brier_score(probability, labels), rtol=1e-12)


def test_r2_is_the_brier_skill_score_against_the_base_rate(sparse_probability_forecast):
    probability, labels = sparse_probability_forecast
    expected = 1.0 - scores.brier_score(probability, labels) / labels.var()
    assert np.isclose(scores.r2_score(probability, labels), expected, rtol=1e-10)


def test_mae_is_improper_on_a_binary_observation_while_rmse_is_not(sparse_probability_forecast):
    """The sharpest statement of the retag, made executable. ``E|p-y| = pi(1-p) + (1-pi)p`` is LINEAR in p, so it is
    minimized at p = 0 whenever the base rate is below 0.5: the all-zero forecast BEATS an honest calibrated one.
    ``rmse`` ranks them the other way. Report mae, never select on it."""
    probability, labels = sparse_probability_forecast
    all_zero = np.zeros_like(probability)

    assert scores.mae(all_zero, labels) < scores.mae(probability, labels)
    assert scores.rmse(all_zero, labels) > scores.rmse(probability, labels)


def test_rank_correlation_is_nan_under_its_configured_condition(sparse_probability_forecast):
    """Every configured subgroup is ``obs > 0``, which on a binary target is CONSTANT, so Spearman is undefined.
    That is why ``rank_correlation`` stays tagged regression-only despite being affine in ROC-AUC."""
    probability, labels = sparse_probability_forecast
    assert np.isnan(scores.rank_correlation(probability, labels, condition=labels > 0))


# =====================================================================================================================
# NEW: contingency_counts thresholds the two sides differently  (Block 2r)
# =====================================================================================================================
def test_contingency_counts_is_symmetric_by_default():
    """The regression behaviour: one threshold cuts BOTH fields, which is right for "did model and observation both
    reach >= 6 lightning-hours?"."""
    prediction = np.array([0.0, 3.0, 7.0, 12.0])
    observation = np.array([0.0, 8.0, 6.0, 2.0])
    hits, misses, false_alarms, correct_negatives = scores.contingency_counts(prediction, observation, 6.0)
    assert hits == 1                                             # 7 vs 6, both >= 6
    assert misses == 1                                           # 3 vs 8
    assert false_alarms == 1                                     # 12 vs 2
    assert correct_negatives == 1                                # 0 vs 0


def test_contingency_counts_thresholds_the_two_sides_independently():
    """The classification form: a DECISION cut on the probability against an observation that is not re-thresholded.
    A shared cut here is meaningless, and concretely `occurrence` resolves to `> 0`, so `pred > 0` fires on every
    cell with any non-zero probability — POD ~ 1 and a full table of nonsense, with no error raised."""
    probability = np.array([0.01, 0.02, 0.80, 0.60])
    labels = np.array([0.0, 1.0, 1.0, 0.0])

    hits, misses, false_alarms, correct_negatives = scores.contingency_counts(
        probability, labels, 0.5, obs_threshold=0.0, obs_strict=True
    )
    assert (hits, misses, false_alarms, correct_negatives) == (1, 1, 1, 1)

    # the degenerate configuration: the daily `occurrence` level (> 0) applied to BOTH sides of a probability field
    _, degenerate_misses, degenerate_false_alarms, degenerate_negatives = scores.contingency_counts(
        probability, labels, 0.0, strict=True
    )
    assert degenerate_misses == 0                                 # POD == 1: every positive is "detected"
    assert degenerate_false_alarms == 2                           # and every negative is a false alarm
    assert degenerate_negatives == 0


# =====================================================================================================================
# NEW: threshold-free FSS is a fractions Brier skill score  (Block 2r)
# =====================================================================================================================
def test_threshold_free_fss_equals_the_fractions_brier_skill_score():
    """When the prediction is already a probability its neighbourhood mean IS a fraction, so no cut is needed — and
    in that form FSS is exactly a Brier skill score at that neighbourhood scale."""
    from scipy.ndimage import uniform_filter

    rng = np.random.default_rng(8)
    obs = (rng.random((5, 16, 16)) < 0.1).astype(np.float64)
    probability = np.clip(obs * 0.6 + rng.random(obs.shape) * 0.3, 0, 1)
    scale = 3

    # PER MAP: the neighbourhood is 2-D. Filtering the [N, H, W] stack in one call would also smooth across the item
    # axis, i.e. average one day's field into the next day's — a mistake that lands within ~1 % of the right answer
    # and so would not look wrong.
    numerator = denominator = 0.0
    for index in range(obs.shape[0]):
        f_pred = uniform_filter(probability[index], size=scale, mode='constant', cval=0.0)
        f_obs = uniform_filter(obs[index], size=scale, mode='constant', cval=0.0)
        numerator += float(((f_pred - f_obs) ** 2).sum())
        denominator += float((f_pred ** 2 + f_obs ** 2).sum())

    assert np.isclose(scores.fss(probability, obs, None, scale), 1.0 - numerator / denominator, rtol=1e-10)


def test_threshold_free_fss_reproduces_the_thresholded_value_on_a_binarised_field():
    rng = np.random.default_rng(9)
    obs = (rng.random((4, 14, 14)) < 0.15).astype(np.float64)
    prediction = np.clip(obs + rng.normal(0, 0.4, obs.shape), 0, None)

    binarised = (prediction >= 0.5).astype(np.float64)
    assert np.isclose(scores.fss(binarised, obs, None, 3), scores.fss(prediction, obs, 0.5, 3), rtol=1e-12)
