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


# =====================================================================================================================
# The Task: taxonomy — merge guards on the docstrings, because a wrong tag makes a reader SKIP a working score
# =====================================================================================================================
def _task_tags():
    """``{public score name: its Task: tag}``, parsed from ``scores.py``'s docstrings."""
    import ast
    import re

    tree = ast.parse(open(scores.__file__).read())
    tags = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            match = re.search(r'^Task: (\w+)', ast.get_docstring(node) or '', flags=re.MULTILINE)
            tags[node.name] = match.group(1) if match else None
    return tags


@pytest.mark.source_invariant
def test_every_public_score_declares_its_TASK():
    """The two tasks have genuinely different valid score sets and nothing else says so."""
    untagged = [name for name, tag in _task_tags().items() if tag is None]
    assert not untagged, untagged


@pytest.mark.source_invariant
def test_the_regression_only_set_is_exactly_the_THREE_limited_by_their_bins_or_conditions():
    """Not "has a tag" — WHICH functions are regression-only. Block 2r got this wrong by assuming the classification
    prediction was a 0/1 field, and tagged five working scores as degenerate. That is the worst kind of error: a reader
    trusts the tag and skips a score that applies.

    These three are limited by their BINS or CONDITIONS, not by the prediction: ``estimation_tendency`` conditions on
    ``obs > 0``, ``stratified_mae`` bins by observed intensity, and ``rank_correlation`` does both and is misleadingly
    scaled besides.
    """
    tags = _task_tags()
    regression_only = {name for name, tag in tags.items() if tag == 'regression'}
    assert regression_only == {'estimation_tendency', 'rank_correlation', 'stratified_mae'}, sorted(regression_only)


@pytest.mark.source_invariant
def test_the_five_probability_valid_continuous_scores_are_tagged_BOTH():
    """The retag Block 2r2 made. Reverting any one of them fails here rather than passing a "has a tag" check — and the
    sharpest pair is ``rmse`` and ``mae``, two functions apart in the file and completely divergent on a binary
    observation: ``rmse ** 2`` IS the Brier score (proper), while ``mae`` is minimised by the all-zero forecast."""
    tags = _task_tags()
    for name in ('rmse', 'mae', 'bias', 'r2_score', 'conditional_error'):
        assert tags[name] == 'both', f'{name} is tagged {tags[name]!r}'


@pytest.mark.source_invariant
def test_dice_coefficient_is_tagged_for_the_CLASSIFICATION_task():
    """It reads a probability directly, as soft Dice, so it belongs to the occurrence head rather than to the hour
    bands."""
    assert _task_tags()['dice_coefficient'] == 'classification'


@pytest.mark.source_invariant
@pytest.mark.parametrize('module_name', ['scores', 'evaluation', 'reporting', 'diagnostics'])
def test_no_removed_metric_identifier_survives_as_CODE(module_name):
    """The merge guard for the metric suite. Every one of these was a real function that a decision removed: the target
    transform's PIT block, the quantile-ratio pair behind the deleted figures, Tweedie and the unbounded-count scores,
    and branch D's rival ensemble accumulator.

    Tokenized rather than grepped: these files' comments legitimately NAME what was removed in order to explain the
    absence, so a text search flags the prose that documents the decision.
    """
    import importlib
    import tokenize

    module = importlib.import_module(f'src.utils.metrics.{module_name}')
    with open(module.__file__, 'rb') as handle:
        identifiers = {token.string for token in tokenize.tokenize(handle.readline)
                       if token.type == tokenize.NAME}

    banned = {'tweedie_deviance_score', 'uniform_histogram_ks', 'quantile_ratios', 'quantile_quantile',
              'psd_full_fidelity', 'gamma_shape', 'gamma_scale', 'GammaFTransform', 'LogStandardizeTransform',
              'EnsembleProbabilisticAccumulator', 'regression_metric_suite', 'target_variable'}
    survivors = identifiers & banned
    assert not survivors, sorted(survivors)


def test_the_ranking_metrics_are_NaN_when_the_observation_has_one_class():
    """Both are undefined without a positive and a negative. NaN propagates into the composite as a 0 contribution,
    which is the intended degradation — a returned 0.5 or 0.0 would read as "no skill" rather than "not computable"."""
    rng = np.random.default_rng(0)
    probability = rng.random(2000)
    assert np.isnan(scores.roc_auc(probability, np.zeros(2000)))
    assert np.isnan(scores.average_precision(probability, np.zeros(2000)))


def test_explained_deviance_is_NEGATIVE_for_an_anti_informative_forecast():
    """It is a skill score against the climatology, so a forecast that inverts the signal must score below zero rather
    than clipping at it — the sign is what tells a reader the model is worse than saying nothing."""
    rng = np.random.default_rng(1)
    labels = (rng.random(20000) < 0.02).astype(float)
    probability = 1.0 / (1.0 + np.exp(-rng.normal(labels * 1.6, 1.0)))
    climatology = np.full_like(probability, labels.mean())

    assert scores.explained_deviance(1.0 - probability, labels, climatology) < 0


def test_a_DISCRIMINATING_but_miscalibrated_forecast_scores_well_on_AUC_and_badly_here():
    """The pair that makes reporting both worthwhile. ROC-AUC sees only the RANKING, so an inverted-but-informative
    forecast still ranks; ``explained_deviance`` is a proper log-loss skill score and sees the calibration. A model can
    therefore look strong on the AUCs and be useless as a probability — which is precisely the failure the Platt phase
    exists to fix."""
    rng = np.random.default_rng(1)
    labels = (rng.random(20000) < 0.02).astype(float)
    probability = 1.0 / (1.0 + np.exp(-rng.normal(labels * 1.6, 1.0)))
    climatology = np.full_like(probability, labels.mean())

    assert scores.roc_auc(probability, labels) > 0.5
    assert scores.explained_deviance(probability, labels, climatology) < 0


def test_roc_auc_EXCEEDS_average_precision_on_a_sparse_target():
    """The imbalance signature, and the reason the classification composite weights AP at 0.50 and not ROC-AUC: on a
    ~2 % base rate ROC-AUC is flattered by the enormous true-negative count, while AP is not."""
    rng = np.random.default_rng(1)
    labels = (rng.random(20000) < 0.02).astype(float)
    # a logistic forecast with real OVERLAP between the classes. A perfectly separating one pins both metrics at 1.0,
    # which is where the imbalance signature disappears.
    probability = 1.0 / (1.0 + np.exp(-rng.normal(labels * 1.6, 1.0)))

    assert scores.roc_auc(probability, labels) > scores.average_precision(probability, labels)


def test_score_max_is_a_MONOTONE_RESCALE_so_the_metric_is_invariant():
    """``score_max`` maps a 0-24 hour prediction into [0, 1] so the binned accumulator can use its fixed edges. Any fixed
    monotone map leaves AUC and AP exactly invariant and changes only bin resolution — which is what makes the rescale a
    free implementation detail rather than a modelling choice."""
    rng = np.random.default_rng(2)
    hours = rng.integers(0, 25, 20000).astype(float)
    labels = (hours > 0).astype(float)

    assert scores.roc_auc(hours / 24.0, labels) == pytest.approx(
        scores.roc_auc(hours, labels, score_max=24.0), abs=1e-12)


def test_the_SUBSAMPLING_helpers_are_gone():
    """``_subsampled_ranking_inputs`` capped the ranking path at 2e6 cells — 12 % of the daily test split and 0.51 % of
    the hourly one. So ``average_precision_occurrence``, weight 0.50 in the classification composite, was a random
    sample. The binned accumulator replaced it, and the sklearn curve wrappers went with it: one implementation, not an
    exact one for tests and a binned one for production."""
    for name in ('_subsampled_ranking_inputs', 'roc_curve_points', 'precision_recall_curve_points'):
        assert not hasattr(scores, name), name


def test_finalize_ranking_metrics_is_NaN_with_one_class_and_returns_empty_curves():
    """The streaming counterpart of the wrapper behaviour above, since the accumulator can legitimately see a batch with
    no positives."""
    edges = scores.ranking_bin_edges()
    partials = scores.ranking_partials(np.random.default_rng(0).random(500), np.zeros(500), edges)
    finalized = scores.finalize_ranking_metrics(partials, edges)

    assert np.isnan(finalized['roc_auc']) and np.isnan(finalized['average_precision'])


def test_psd_full_fidelity_is_a_KEY_not_a_function():
    """It is the ``psd_fidelity`` of the full-band ratio, computed in the evaluation loop. A second function of that name
    is exactly how the two would drift — and this key carries 0.40 of the regression composite."""
    assert not hasattr(scores, 'psd_full_fidelity')


def test_bernoulli_logloss_stays_FINITE_at_a_confident_mistake():
    """A probability of exactly 0 on an observed event is an infinite log-loss, which would poison
    ``explained_deviance`` for the whole split. The implementation clips, so one over-confident cell degrades the score
    instead of destroying it."""
    assert np.isfinite(scores.bernoulli_logloss(np.array([0.0, 1.0]), np.array([1.0, 0.0])))


def test_dice_coefficient_is_ONE_on_a_perfect_overlap_and_low_when_disjoint():
    """The two anchors of the coefficient, on binary fields where it reduces to ``2TP / (2TP + FP + FN)``."""
    labels = (np.random.default_rng(0).random(500) < 0.2).astype(float)
    assert scores.dice_coefficient(labels, labels) == pytest.approx(1.0, abs=1e-6)
    assert scores.dice_coefficient(np.array([1.0, 0.0]), np.array([0.0, 1.0])) < 0.5


def test_estimation_tendency_DEGENERATES_on_a_binary_observation(sparse_probability_forecast):
    """Why it stays tagged ``regression``: it is conditioned on ``obs > 0``, so on a binary target ``pred - obs = p - 1``
    is never positive. ``under`` is ~1 and ``over`` is 0 for ANY model — degenerate by construction of the CONDITION, not
    of the prediction."""
    probability, labels = sparse_probability_forecast
    tendency = scores.estimation_tendency(probability, labels)
    assert tendency['over'] == 0.0
    assert tendency['under'] > 0.99


def test_stratified_mae_leaves_every_hour_band_but_the_first_EMPTY(sparse_probability_forecast):
    """The other bin-driven reason for a ``regression`` tag: it bins by OBSERVED intensity, and a 0/1 target has exactly
    one non-empty band. The remaining bands are NaN rather than 0, so they read as absent rather than as perfect."""
    probability, labels = sparse_probability_forecast
    bands = scores.stratified_mae(probability, labels, [('occurrence', 0.0), ('h3', 3.0), ('h6', 6.0)])
    populated = sum(not np.isnan(value) for value in bands.values())
    assert populated == 1, bands


def test_the_unconditioned_spearman_is_the_AFFINE_IMAGE_of_roc_auc(sparse_probability_forecast):
    """Not merely redundant with ``roc_auc`` — actively misleading. The scaling is ~``sqrt(base_rate)``, so a
    near-perfect AUC of 0.998 reads as a Spearman of 0.049. That is the second reason it stays tagged ``regression``."""
    probability, labels = sparse_probability_forecast
    spearman = scores.rank_correlation(probability, labels, condition=np.ones(labels.shape, dtype=bool))

    positives, negatives = labels.sum(), labels.size - labels.sum()
    auc = scores.roc_auc(probability, labels)
    expected = (2.0 * auc - 1.0) * np.sqrt(3.0 * positives * negatives / labels.size ** 2)
    assert spearman == pytest.approx(expected, rel=1e-5)


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


def test_the_threshold_free_useful_scale_reuses_the_threshold_free_fss_at_every_scale():
    """``fss_useful_scale`` reuses the per-scale ``fss`` it already computed rather than recomputing — the same reuse the
    thresholded form gets. Checked in the threshold-free form too, because that path takes different branches in both
    functions and could reuse the wrong one silently."""
    rng = np.random.default_rng(0)
    observation = (rng.random((4, 24, 24)) < 0.08).astype(np.float64)
    probability = np.clip(0.4 * observation + 0.15 * rng.random(observation.shape), 0, 1)
    scale_list = [1, 3, 5]

    _, by_scale = scores.fss_useful_scale(probability, observation, None, scale_list)
    for scale in scale_list:
        assert by_scale[scale] == pytest.approx(scores.fss(probability, observation, None, scale), abs=1e-12), scale


def test_threshold_free_fss_reproduces_the_thresholded_value_on_a_binarised_field():
    rng = np.random.default_rng(9)
    obs = (rng.random((4, 14, 14)) < 0.15).astype(np.float64)
    prediction = np.clip(obs + rng.normal(0, 0.4, obs.shape), 0, None)

    binarised = (prediction >= 0.5).astype(np.float64)
    assert np.isclose(scores.fss(binarised, obs, None, 3), scores.fss(prediction, obs, 0.5, 3), rtol=1e-12)


# =====================================================================================================================
# Block 5c — the internals the public scores are built from
#
# These are reached through their callers everywhere above. Tested directly here because each one is a place where a
# shared quantity is computed ONCE and reused, so an error in it propagates identically into every score that depends
# on it — and a suite where every CRPS-derived number is wrong by the same factor looks self-consistent.
# =====================================================================================================================
def test_the_crps_spread_term_equals_the_BRUTE_FORCE_pairwise_expectation():
    """``_crps_terms`` uses the O(M log M) order-statistic estimator of ``(1/2) E|X - X'|`` because the ensemble stack
    cannot be held pairwise in memory. That optimisation is the thing worth checking: it must agree exactly with the
    definition on a small ensemble, or every CRPS in the suite is wrong by a consistent amount and nothing looks odd.
    """
    rng = np.random.default_rng(0)
    members = rng.gamma(2.0, 2.0, size=(7, 5))                     # [M, cells]
    observation = rng.gamma(2.0, 2.0, size=5)

    mae_term, spread_term = scores._crps_terms(members, observation)

    brute_mae = np.abs(members - observation[None]).mean(axis=0)
    brute_spread = 0.5 * np.abs(members[:, None, :] - members[None, :, :]).mean(axis=(0, 1))

    assert np.allclose(mae_term, brute_mae)
    assert np.allclose(spread_term, brute_spread), f'{spread_term} vs {brute_spread}'


def test_the_crps_terms_are_ORDER_INVARIANT_in_the_members():
    """The ensemble has no member ordering — draw 3 is not "after" draw 2. A term that depended on it would make CRPS
    depend on the order ``predict_step`` happened to stack the draws in."""
    rng = np.random.default_rng(1)
    members = rng.normal(size=(6, 4))
    observation = rng.normal(size=4)

    straight = scores._crps_terms(members, observation)
    shuffled = scores._crps_terms(members[[3, 0, 5, 1, 4, 2]], observation)
    assert np.allclose(straight[0], shuffled[0]) and np.allclose(straight[1], shuffled[1])


def test_a_ONE_MEMBER_ensemble_has_zero_spread_so_crps_degenerates_to_mae():
    """Not an error, which is the hazard CLAUDE.md flags for ``ensemble-size``: a single member gives a finite CRPS
    that is silently just the MAE, with no sign the spread term contributed nothing."""
    members = np.array([[2.0, 5.0]])
    mae_term, spread_term = scores._crps_terms(members, np.array([1.0, 1.0]))
    assert np.allclose(spread_term, 0.0)
    assert np.allclose(mae_term, [1.0, 4.0])


# ---------------------------------------------------------------------------------------------------------------------
# The FSS internals
# ---------------------------------------------------------------------------------------------------------------------
def test_identical_fields_contribute_a_ZERO_fss_numerator():
    """FSS is ``1 - num/den``, so a perfect forecast has to give ``num = 0`` exactly at every scale."""
    rng = np.random.default_rng(0)
    field = (rng.random((16, 16)) < 0.1).astype(np.float64)
    for scale in (1, 3, 5):
        numerator, denominator = scores._fractions_skill(field, field, scale)
        assert numerator == 0.0
        assert denominator > 0.0


def test_the_fractions_helper_accepts_BOOLEAN_and_FLOAT_fields_interchangeably():
    """The documented reason it casts: the thresholded form passes boolean exceedance masks and the probabilistic form
    passes float fractions. A boolean neighbourhood mean that stayed boolean would collapse every fraction to 0 or 1
    and destroy the whole point of FSS."""
    rng = np.random.default_rng(0)
    boolean = rng.random((12, 12)) < 0.2
    observed = rng.random((12, 12)) < 0.2

    from_bool = scores._fractions_skill(boolean, observed, 3)
    from_float = scores._fractions_skill(boolean.astype(np.float64), observed.astype(np.float64), 3)
    assert from_bool == from_float


def test_a_larger_neighbourhood_SMOOTHS_the_fraction_mismatch():
    """The property the useful-scale search rests on: a displaced forecast is penalised less as the neighbourhood grows
    past the displacement. A numerator that did not fall with scale would make ``fss_useful_scale`` meaningless."""
    prediction = np.zeros((21, 21))
    observation = np.zeros((21, 21))
    prediction[10, 10] = 1.0
    observation[10, 13] = 1.0                                       # the same event, displaced by 3 pixels

    numerators = [scores._fractions_skill(prediction, observation, scale)[0] for scale in (1, 3, 9)]
    assert numerators[0] > numerators[1] > numerators[2]


def test_a_threshold_BINARISES_both_sides_and_None_passes_them_through():
    """The switch between the regression and classification forms of FSS. ``None`` is not "no threshold, use zero" — it
    means the fields ARE already fractions, which is what makes the probabilistic form a fractions Brier skill score."""
    prediction = np.array([[0.2, 0.8], [6.0, 0.0]])
    observation = np.array([[0.0, 1.0], [7.0, 0.0]])

    passthrough = scores._fss_fields(prediction, observation, None, strict=True)
    assert passthrough[0] is prediction and passthrough[1] is observation

    binarised = scores._fss_fields(prediction, observation, 0.5, strict=True)
    assert binarised[0].dtype == bool and binarised[1].dtype == bool
    assert binarised[0].tolist() == [[False, True], [True, False]]
    assert binarised[1].tolist() == [[False, True], [True, False]]


# ---------------------------------------------------------------------------------------------------------------------
# The spectral internals
# ---------------------------------------------------------------------------------------------------------------------
def test_the_wavelength_grid_is_INFINITE_at_the_dc_component():
    """The DC term is the field's mean — a structure of infinite wavelength. Left as a zero frequency it would divide
    by zero and put a NaN into every radial bin that touches it."""
    grid = scores._wavelength_grid(8, 8)
    assert np.isinf(grid[0, 0])
    assert np.all(np.isfinite(grid.ravel()[1:]))
    assert np.all(grid.ravel()[1:] > 0)


def test_the_shortest_resolvable_wavelength_along_an_axis_is_TWO_pixels():
    """Nyquist. It is what makes the report's kilometre axis honest: two pixels is 55.5 km, and nothing below it is
    resolved by the grid at all."""
    grid = scores._wavelength_grid(8, 8)
    assert abs(grid[0, 4] - 2.0) < 1e-12
    assert abs(grid[4, 0] - 2.0) < 1e-12


def test_the_per_map_psd_rows_AVERAGE_to_the_pooled_spectrum():
    """The stated contract, and the reason the ensemble band and the pooled curve can be drawn on one axis: they share
    a wavelength axis AND the rows' column mean is the pooled curve, so the band is centred on the line it surrounds."""
    rng = np.random.default_rng(0)
    fields = rng.normal(size=(6, 16, 16))

    pooled_wavelengths, pooled = scores.radial_psd(fields)
    per_map_wavelengths, per_map = scores.radial_psd_per_map(fields)

    assert np.allclose(pooled_wavelengths, per_map_wavelengths)
    assert per_map.shape == (6, pooled.size)
    assert np.allclose(per_map.mean(axis=0), pooled)


def test_the_per_map_psd_can_return_the_pooled_2d_spectrum_from_ONE_fft_pass():
    """The optimisation that exists so the band-ratio scalars and the per-map band do not each pay for a full FFT sweep
    of the split. It has to give exactly what the separate function gives."""
    rng = np.random.default_rng(1)
    fields = rng.normal(size=(5, 16, 16))

    _, _, mean_spectrum = scores.radial_psd_per_map(fields, return_mean_spectrum=True)
    assert np.allclose(mean_spectrum, scores.mean_power_spectrum(fields))


def test_the_per_map_psd_ticks_its_progress_callback_once_per_map():
    """The evaluation stage's progress bar is driven by this, and a tick per BIN rather than per map would make the bar
    finish long before the sweep does."""
    ticks = []
    scores.radial_psd_per_map(np.random.default_rng(0).normal(size=(4, 8, 8)), progress=lambda: ticks.append(1))
    assert len(ticks) == 4


# ---------------------------------------------------------------------------------------------------------------------
# reliability_curve and skill_score
# ---------------------------------------------------------------------------------------------------------------------
def test_a_perfectly_calibrated_forecast_puts_the_reliability_curve_ON_the_diagonal():
    """The definition of calibration: among the cells given probability p, a fraction p occur. Drawn against the
    diagonal in the report, so a curve that could not reach it would make every model look miscalibrated."""
    rng = np.random.default_rng(0)
    probabilities = rng.random(200_000)
    occurrences = (rng.random(200_000) < probabilities).astype(np.float64)

    mean_probability, observed_frequency, counts = scores.reliability_curve(probabilities, occurrences, bins=10)

    populated = counts > 0
    assert np.allclose(mean_probability[populated], observed_frequency[populated], atol=0.02)
    assert counts.sum() == 200_000


def test_an_UNPOPULATED_reliability_bin_is_NaN_rather_than_zero():
    """At a 0.07 % base rate nearly every high-probability bin is empty. A zero there would draw a curve plunging to the
    floor and read as catastrophic over-forecasting."""
    probabilities = np.array([0.01, 0.02, 0.03])
    occurrences = np.array([0.0, 0.0, 1.0])

    mean_probability, observed_frequency, counts = scores.reliability_curve(probabilities, occurrences, bins=10)

    assert counts[0] == 3 and counts[1:].sum() == 0
    assert np.isnan(mean_probability[5]) and np.isnan(observed_frequency[5])
    assert np.isfinite(mean_probability[0])


def test_a_probability_of_exactly_ONE_lands_in_the_LAST_bin_not_past_the_end():
    """``np.digitize`` would put it at index ``bins``, one past the array. The clip is what keeps a confident forecast
    from being dropped from the diagram entirely."""
    _, _, counts = scores.reliability_curve(np.array([1.0]), np.array([1.0]), bins=10)
    assert counts[-1] == 1 and counts.sum() == 1


@pytest.mark.parametrize('model,baseline,expected', [
    (0.5, 1.0, 0.5),                                                # half the baseline's error
    (1.0, 1.0, 0.0),                                                # no skill
    (2.0, 1.0, -1.0),                                               # worse than the baseline
    (0.0, 1.0, 1.0),                                                # perfect
])
def test_the_skill_score_is_one_minus_the_error_ratio(model, baseline, expected):
    assert abs(scores.skill_score(model, baseline) - expected) < 1e-12


@pytest.mark.parametrize('baseline', [0.0, -1.0, float('nan'), float('inf')])
def test_a_DEGENERATE_baseline_gives_NaN_rather_than_an_infinite_skill(baseline):
    """A climatology with zero error on the evaluated cells is possible on a sparse subgroup, and ``1 - x/0`` would
    report an infinite skill for a model that is merely being compared against nothing."""
    assert np.isnan(scores.skill_score(1.0, baseline))
