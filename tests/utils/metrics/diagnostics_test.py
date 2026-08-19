"""Tests for src/utils/metrics/diagnostics.py — the residual-mode diffusion diagnostics.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_residual_diagnostics.py`` (the computation half; the
module plumbing moved to ``modeling/diffusion_module_test.py`` and the figures to ``reporting_test.py``).

These answer "is the learnt correction actually tracking the true discrepancy?", which no scalar metric asks: the
residual model can score well on the reconstructed target while its correction is uncorrelated with the discrepancy it
was supposed to learn.

The fixture is adapted to the bounded 0-24 target: the reconstruction is ``clamp(upstream + residual, 0, max_hours)``
at BOTH ends, where A's clamped only the floor.
"""
import numpy as np
import pytest

from src.utils.metrics import diagnostics

H = W = 16
MEMBERS = 4
OCCURRENCE_EVENT = (0.0, True)
MAX_HOURS = 24.0

CURVE_BLOCKS = {'bias_map', 'surprise_magnitude', 'surprise_direction', 'hist_pixel', 'hist_image', 'qq', 'scatter',
                'heteroscedasticity'}
SCALAR_KEYS = ('resid_bias_mean', 'resid_scale_rms_ratio', 'resid_mse_skill_vs_upstream', 'resid_dir_corr',
               'resid_sign_agreement', 'resid_qq_pearson', 'resid_scatter_obs_pixel_positive_pearson',
               'resid_hetero_upstream_spearman', 'resid_surprise_mag_overcorrect_frac')


@pytest.fixture
def residual_run():
    """``(observation, prediction, upstream, residual_mean, residual_members)`` for a model whose correction genuinely
    tracks the true discrepancy — so the direction correlation and the upstream skill must both come out positive."""
    def build(n_items=10, seed=0):
        rng = np.random.default_rng(seed)
        upstream = np.clip(np.abs(rng.normal(6, 3, (n_items, H, W))), 0, MAX_HOURS)
        true_residual = rng.normal(0, 3.0, (n_items, H, W))
        observation = np.clip(upstream + true_residual, 0, MAX_HOURS)
        members = true_residual[:, None] + rng.normal(0, 1.2, (n_items, MEMBERS, H, W))
        residual_mean = members.mean(axis=1)
        prediction = np.clip(upstream + residual_mean, 0, MAX_HOURS)
        return observation, prediction, upstream, residual_mean, members
    return build


def test_scalars_and_curve_blocks_are_all_produced(residual_run):
    observation, prediction, upstream, residual_mean, members = residual_run()
    flat, curves = diagnostics.residual_diagnostics(
        observation, prediction, upstream, residual_mean, members, occurrence_event=OCCURRENCE_EVENT
    )
    assert set(curves['residual']) == CURVE_BLOCKS
    for key in SCALAR_KEYS:
        assert key in flat, key


def test_a_correction_that_tracks_the_discrepancy_scores_positively(residual_run):
    """The diagnostic's whole purpose: a model whose predicted discrepancy correlates with the true one must show a
    positive direction correlation and positive skill against the raw upstream."""
    observation, prediction, upstream, residual_mean, members = residual_run()
    flat, _ = diagnostics.residual_diagnostics(
        observation, prediction, upstream, residual_mean, members, occurrence_event=OCCURRENCE_EVENT
    )
    assert flat['resid_dir_corr'] > 0.7
    assert flat['resid_mse_skill_vs_upstream'] > 0.0


def test_an_uncorrelated_correction_shows_no_skill(residual_run):
    """The converse, which is what makes the test above mean something: a correction drawn independently of the true
    discrepancy must NOT show direction correlation, and must not beat the upstream."""
    observation, _, upstream, _, _ = residual_run()
    rng = np.random.default_rng(99)
    noise_members = rng.normal(0, 3.0, (observation.shape[0], MEMBERS, H, W))
    noise_mean = noise_members.mean(axis=1)
    prediction = np.clip(upstream + noise_mean, 0, MAX_HOURS)

    flat, _ = diagnostics.residual_diagnostics(
        observation, prediction, upstream, noise_mean, noise_members, occurrence_event=OCCURRENCE_EVENT
    )
    assert abs(flat['resid_dir_corr']) < 0.2, 'a correction independent of the truth must not correlate with it'
    # adding noise to the upstream can only make the reconstruction worse, so the skill must be NEGATIVE
    assert flat['resid_mse_skill_vs_upstream'] < 0.0


def test_surprise_categories_are_flagged(residual_run):
    """The +/-inf surprise panes: a correction where the truth needs none (overcorrected) and no correction where the
    truth needs one (failed). Both are ratios with a zero denominator, so they are categorised rather than computed."""
    observation, _, upstream, residual_mean, members = residual_run()
    observation = observation.copy()
    residual_mean = residual_mean.copy()
    members = members.copy()

    observation[:, :6, :6] = upstream[:, :6, :6]        # D_true = 0 here -> overcorrect (+inf)
    members[:, :, 9:14, 9:14] = 0.0
    residual_mean[:, 9:14, 9:14] = 0.0                  # D_pred = 0 here -> failed (-inf)
    prediction = np.clip(upstream + residual_mean, 0, MAX_HOURS)

    flat, _ = diagnostics.residual_diagnostics(
        observation, prediction, upstream, residual_mean, members, occurrence_event=OCCURRENCE_EVENT
    )
    assert flat['resid_surprise_mag_overcorrect_frac'] > 0.0
    assert flat['resid_surprise_mag_failed_frac'] > 0.0


def test_nonfinite_members_are_stripped_without_corrupting_the_suite(residual_run):
    """An unconverged ODE sample can leave NaN/inf members. They must be stripped rather than silently corrupting the
    QQ block."""
    observation, prediction, upstream, _, members = residual_run()
    members = members.copy()
    members[0, 0, 0, 0] = np.inf
    members[1, 1, 2, 3] = np.nan

    flat, curves = diagnostics.residual_diagnostics(
        observation, prediction, upstream, members.mean(axis=1), members, occurrence_event=OCCURRENCE_EVENT
    )
    assert 'qq' in curves['residual']
    assert np.isfinite(flat['resid_qq_pearson'])


def test_all_nonfinite_members_degrade_rather_than_crash(residual_run):
    """The degenerate case, and a specific regression guard: the KS call raises on an empty input, and an
    ``except ... as error`` there once shadowed the ``error`` array the heteroscedasticity block needs. So the QQ
    block must be absent while the rest of the suite still runs."""
    observation, prediction, upstream, _, members = residual_run()
    nan_members = np.full_like(members, np.nan)

    # the KS warning is asserted, not silenced: it is the evidence that the empty-pool path was actually taken
    with pytest.warns(Warning, match='too small'):
        _, curves = diagnostics.residual_diagnostics(
            observation, prediction, upstream, nan_members.mean(axis=1), nan_members,
            occurrence_event=OCCURRENCE_EVENT
        )
    assert 'qq' not in curves['residual'], 'no quantiles can come from an empty pool'
    assert 'heteroscedasticity' in curves['residual'], 'the rest of the suite must still have run'


def test_bias_map_has_the_grid_shape(residual_run):
    """The bias map is a per-cell mean over items, so it must collapse the item axis and keep the grid — it is drawn
    directly as a map by ``residual_bias_map``."""
    observation, prediction, upstream, residual_mean, members = residual_run()
    _, curves = diagnostics.residual_diagnostics(
        observation, prediction, upstream, residual_mean, members, occurrence_event=OCCURRENCE_EVENT
    )
    assert curves['residual']['bias_map'].shape == (H, W)


def test_diagnostics_are_deterministic_under_a_fixed_seed(residual_run):
    """The suite subsamples pixels for the scatter and histogram blocks, so it must be seeded — two runs over the same
    arrays reporting different numbers would make a trials table incomparable."""
    arrays = residual_run()
    first, _ = diagnostics.residual_diagnostics(*arrays, occurrence_event=OCCURRENCE_EVENT, seed=0)
    second, _ = diagnostics.residual_diagnostics(*arrays, occurrence_event=OCCURRENCE_EVENT, seed=0)
    assert first == second


# =====================================================================================================================
# Block 5c — the private helpers, driven directly
#
# Everything above goes through ``residual_diagnostics``, which computes 30-odd scalars at once: a helper returning
# something subtly wrong shows up there as one number among thirty, and only if a test happens to assert on it. These
# pin the helpers' own contracts — most of which are about DEGENERATE input, since that is what a sparse 95.3 %-zero
# field produces constantly and what silently turns a metric into NaN.
# =====================================================================================================================
@pytest.mark.parametrize('event,expected', [
    ((0.0, True), [False, True, True]),                    # strict: > 0, the default occurrence event
    ((0.0, False), [True, True, True]),                    # non-strict: >= 0 is EVERY cell on a non-negative target
    ((6.0, True), [False, False, True]),
])
def test_the_occurrence_mask_honours_the_STRICTNESS_of_the_event(event, expected):
    """``occurrence_event`` is a ``(value, strict)`` pair, not a bare threshold, and the difference matters exactly at
    the value: ``>= 0`` selects the whole domain on a non-negative target, which would make every "positive-only"
    diagnostic a domain-wide one with no error."""
    observation = np.array([0.0, 1.0, 24.0])
    assert list(diagnostics._occurrence_mask(observation, event)) == expected


def test_the_correlations_agree_with_numpy_and_scipy_on_a_clean_sample():
    """Pinned against the reference implementations rather than against themselves — ``_spearman`` exists to avoid a
    scipy dependency at the top of the module, so it has to give scipy's answer."""
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.7 * x + rng.normal(size=200) * 0.5

    assert abs(diagnostics._pearson(x, y) - pearsonr(x, y)[0]) < 1e-9
    assert abs(diagnostics._spearman(x, y) - spearmanr(x, y)[0]) < 1e-9


def test_spearman_is_invariant_under_a_MONOTONE_rescaling_where_pearson_is_not():
    """The reason both are reported. ``resid_scatter_*_pixel_spearman`` is the one that survives the target's
    nonlinearity; a Spearman that moved under ``exp`` would be a Pearson in disguise."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = 0.8 * x + rng.normal(size=300) * 0.4

    assert abs(diagnostics._spearman(x, y) - diagnostics._spearman(x, np.exp(y))) < 1e-9
    assert abs(diagnostics._pearson(x, y) - diagnostics._pearson(x, np.exp(y))) > 1e-3


@pytest.mark.parametrize('function', ['_pearson', '_spearman', '_ols_slope'])
@pytest.mark.parametrize('x,y', [
    (np.array([1.0]), np.array([2.0])),                              # a single point
    (np.array([]), np.array([])),
    (np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 3.0])),          # a constant x — no slope to estimate
    (np.array([1.0, np.nan, 3.0]), np.array([np.nan, np.nan, np.nan])),
])
def test_every_correlation_returns_NAN_rather_than_raising_on_degenerate_input(function, x, y):
    """These are called on the positive-only subsets of a field that is 95.3 % zero, so an empty or constant sample is
    routine. Raising would abort the whole diagnostics block; ``numpy`` alone would emit a runtime warning and return a
    silent NaN through a divide-by-zero."""
    result = getattr(diagnostics, function)(x, y)
    assert np.isnan(result)


def test_a_constant_Y_is_still_degenerate_for_correlation_but_NOT_for_a_slope():
    """The asymmetry is deliberate: a correlation of a constant with anything is undefined, while the OLS slope of a
    flat response is a perfectly meaningful zero."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    flat = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.isnan(diagnostics._pearson(x, flat))
    assert abs(diagnostics._ols_slope(x, flat)) < 1e-9


def test_the_ols_slope_recovers_a_known_line():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(diagnostics._ols_slope(x, 3.0 * x - 7.0) - 3.0) < 1e-9


def test_the_slope_ignores_cells_where_EITHER_side_is_non_finite():
    """An unconverged ODE draw makes some members non-finite. Dropping the pair rather than the whole array is what
    keeps the QQ slope computable on a partially-bad ensemble."""
    x = np.array([0.0, 1.0, np.nan, 2.0, 3.0])
    y = np.array([1.0, 3.0, 99.0, 5.0, np.inf])
    assert abs(diagnostics._ols_slope(x, y) - 2.0) < 1e-9


def test_a_subsample_returns_EVERY_index_below_the_cap_and_exactly_the_cap_above_it():
    """The cap is 200 000 cells against a real split of 16.5 M, so almost every call subsamples. Below the cap it must
    return all of them IN ORDER — the pixel scatter pairs two fields by the same index array, and a shuffled 'all'
    would silently decorrelate them."""
    rng = np.random.default_rng(0)

    assert list(diagnostics._subsample(rng, n=5, cap=10)) == [0, 1, 2, 3, 4]
    assert list(diagnostics._subsample(rng, n=10, cap=10)) == list(range(10))

    drawn = diagnostics._subsample(rng, n=1000, cap=10)
    assert drawn.size == 10
    assert len(set(drawn.tolist())) == 10, 'sampled without replacement'


def test_a_subsample_is_reproducible_under_a_fixed_generator():
    first = diagnostics._subsample(np.random.default_rng(7), n=1000, cap=20)
    second = diagnostics._subsample(np.random.default_rng(7), n=1000, cap=20)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------------------------------------------------
# The surprise panes — the 0/0 and +/-inf conventions
# ---------------------------------------------------------------------------------------------------------------------
def test_the_magnitude_pane_encodes_the_three_categories_and_the_log_ratio():
    """``+inf`` = a correction where none was needed, ``-inf`` = no correction where one was, ``0/0`` = matched
    (white). The finite cells carry ``log(num/den)``, so a correction twice too large reads as ``log 2`` and one twice
    too small as ``-log 2`` — symmetric about zero, which is what makes the diverging colour axis honest."""
    numerator = np.array([2.0, 1.0, 0.0, 0.0])
    denominator = np.array([1.0, 2.0, 3.0, 0.0])

    pane = diagnostics._surprise_pane(numerator, denominator, signed=False)

    assert list(pane['category']) == [diagnostics.SURPRISE_FINITE, diagnostics.SURPRISE_FINITE,
                                      diagnostics.SURPRISE_UNDER, diagnostics.SURPRISE_FINITE]
    assert abs(pane['value'][0] - np.log(2.0)) < 1e-12
    assert abs(pane['value'][1] + np.log(2.0)) < 1e-12
    assert pane['value'][2] == 0.0, 'an infinite cell carries no finite value to colour'
    assert pane['value'][3] == 0.0, '0/0 is "nothing to correct, nothing corrected" — matched, not surprising'


def test_the_magnitude_pane_flags_a_correction_where_NONE_was_needed():
    pane = diagnostics._surprise_pane(np.array([3.0]), np.array([0.0]), signed=False)
    assert pane['category'][0] == diagnostics.SURPRISE_OVER


def test_the_direction_pane_treats_a_DIRECTIONLESS_truth_differently_from_the_magnitude_pane():
    """Structurally different, and the difference is the point: here the numerator is a mean SIGN in ``[-1, 1]``, so
    its own zero is not special — only a zero denominator is, and its category follows the numerator's SIGN rather
    than its magnitude."""
    numerator = np.array([0.5, -0.5, 0.0, 1.0])
    denominator = np.array([0.0, 0.0, 0.0, 0.5])

    pane = diagnostics._surprise_pane(numerator, denominator, signed=True)

    assert list(pane['category']) == [diagnostics.SURPRISE_OVER, diagnostics.SURPRISE_UNDER,
                                      diagnostics.SURPRISE_FINITE, diagnostics.SURPRISE_FINITE]
    assert abs(pane['value'][3] - 2.0) < 1e-12, 'the finite cells carry the raw ratio, not its log'


def test_a_surprise_pane_never_emits_a_non_finite_VALUE():
    """The value map is what the diverging norm's percentile is computed over, and a single ``inf`` there would set the
    colour axis to infinity and blank the whole panel. The categories carry the infinities instead."""
    numerator = np.array([1.0, 0.0, 5.0, 0.0])
    denominator = np.array([0.0, 0.0, 2.0, 4.0])
    for signed in (False, True):
        pane = diagnostics._surprise_pane(numerator, denominator, signed=signed)
        assert np.all(np.isfinite(pane['value'])), signed


def test_the_surprise_scalars_summarise_only_the_FINITE_cells_and_count_the_rest():
    """Mixing the infinite cells into the median would be undefined; dropping them silently would hide the model's most
    interesting failures. Reported as fractions alongside instead."""
    pane = {
        'value': np.array([np.log(2.0), -np.log(2.0), 0.0, 0.0]),
        'category': np.array([diagnostics.SURPRISE_FINITE, diagnostics.SURPRISE_FINITE,
                              diagnostics.SURPRISE_OVER, diagnostics.SURPRISE_UNDER], dtype=np.int8),
    }
    scalars = diagnostics._surprise_scalars('mag', pane)

    assert abs(scalars['mag_median']) < 1e-12, 'the median of +log2 and -log2'
    assert abs(scalars['mag_mean']) < 1e-12
    assert scalars['mag_overcorrect_frac'] == 0.25
    assert scalars['mag_failed_frac'] == 0.25


def test_the_surprise_scalars_report_NAN_when_no_cell_is_finite():
    """A model that corrects nothing anywhere. The fractions still have to be real numbers — they are what says so."""
    pane = {'value': np.zeros(4), 'category': np.full(4, diagnostics.SURPRISE_UNDER, dtype=np.int8)}
    scalars = diagnostics._surprise_scalars('mag', pane)

    assert np.isnan(scalars['mag_median']) and np.isnan(scalars['mag_mean'])
    assert scalars['mag_failed_frac'] == 1.0 and scalars['mag_overcorrect_frac'] == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# The decile heteroscedasticity curve
# ---------------------------------------------------------------------------------------------------------------------
def test_the_heteroscedasticity_curve_recovers_a_GROWING_error_with_the_conditioner():
    """The diagnostic's whole purpose: does the correction get less reliable where the upstream is large? An error
    built to scale with the conditioner has to come back as a rising RMS curve."""
    rng = np.random.default_rng(0)
    conditioning = rng.uniform(0.1, 10.0, 5000)
    error = rng.normal(0.0, 1.0, 5000) * conditioning

    curve = diagnostics._decile_heteroscedasticity(error, conditioning, n_bins=10)

    assert curve['bin_center'].size == 10
    assert np.all(np.diff(curve['bin_center']) > 0), 'bin centres ascend with the conditioner'
    assert curve['rms_error'][-1] > 3 * curve['rms_error'][0]


def test_only_POSITIVE_conditioning_cells_enter_the_curve():
    """Both conditioners are non-negative fields that are overwhelmingly zero. Including the zeros would put ~99 % of
    the domain in the first decile and collapse the other nine."""
    conditioning = np.concatenate([np.zeros(1000), np.arange(1.0, 101.0)])
    error = np.ones(1100)

    curve = diagnostics._decile_heteroscedasticity(error, conditioning, n_bins=5)
    assert curve['bin_center'].min() >= 1.0


def test_too_few_positive_cells_gives_EMPTY_arrays_rather_than_a_ragged_curve():
    """What the report's heteroscedasticity builder checks for before drawing. Fewer positives than bins cannot be
    binned at all."""
    curve = diagnostics._decile_heteroscedasticity(np.ones(3), np.array([1.0, 2.0, 3.0]), n_bins=10)
    assert curve['bin_center'].size == 0 and curve['rms_error'].size == 0


def test_a_DISCRETE_conditioner_does_not_double_count_the_cells_sitting_on_a_bin_EDGE():
    """The documented reason the interior bins are half-open. The observed target is an integer count of hours, so many
    cells land exactly on a quantile edge; with both-inclusive tests those cells would be counted in two bins and the
    per-bin RMS would be a weighted average of neighbours rather than a bin statistic."""
    conditioning = np.repeat(np.arange(1.0, 6.0), 200)                # 5 distinct integer values, 200 cells each
    error = np.repeat(np.array([1.0, 1.0, 1.0, 1.0, 10.0]), 200)      # only the TOP value carries a large error

    curve = diagnostics._decile_heteroscedasticity(error, conditioning, n_bins=5)

    assert abs(curve['rms_error'][-1] - 10.0) < 1e-9, 'the top bin must be pure, not diluted by its neighbour'
    assert all(abs(value - 1.0) < 1e-9 for value in curve['rms_error'][:-1])


def test_a_NEAR_CONSTANT_conditioner_collapses_its_tied_bin_edges():
    """``np.unique`` on the quantile edges. Without it a constant conditioner produces zero-width bins whose masks are
    empty, and the curve would carry however many bins happened to be non-degenerate with no relation to ``n_bins``."""
    conditioning = np.concatenate([np.full(500, 2.0), np.array([5.0] * 20)])
    curve = diagnostics._decile_heteroscedasticity(np.ones(520), conditioning, n_bins=10)
    assert 0 < curve['bin_center'].size < 10


# ---------------------------------------------------------------------------------------------------------------------
# The overlaid histogram
# ---------------------------------------------------------------------------------------------------------------------
def test_the_two_histograms_share_one_set_of_bin_edges():
    """They are drawn superimposed, so separate binnings would put the two distributions on different axes and make the
    comparison meaningless while looking fine."""
    rng = np.random.default_rng(0)
    block = diagnostics._overlaid_histogram(rng.normal(0, 1, 5000), rng.normal(0.5, 2, 5000),
                                            bins=30, rng=rng, kde_cap=1000)

    assert block['edges'].size == 31
    assert block['pred_density'].size == block['true_density'].size == 30


def test_the_histogram_extent_is_ROBUST_to_a_heavy_tail():
    """A single extreme residual would otherwise define the axis and put every real value in the first bin."""
    rng = np.random.default_rng(0)
    sample = np.concatenate([rng.normal(0, 1, 5000), np.array([1e6])])
    block = diagnostics._overlaid_histogram(sample, rng.normal(0, 1, 5000), bins=30, rng=rng, kde_cap=500)
    assert block['edges'][-1] < 100.0


def test_an_EMPTY_side_gives_an_empty_block_rather_than_a_partial_one():
    """The report's histogram builder tests the block for truthiness, so a half-populated dict would reach it and fail
    on a missing key inside the try/except that only warns."""
    rng = np.random.default_rng(0)
    assert diagnostics._overlaid_histogram(np.array([]), np.ones(10), 30, rng, 100) == {}
    assert diagnostics._overlaid_histogram(np.array([np.nan, np.inf]), np.ones(10), 30, rng, 100) == {}


def test_a_DEGENERATE_extent_still_produces_usable_bins():
    """An all-identical sample (a model that predicts one constant correction) gives ``lo == hi``, and
    ``np.linspace(lo, lo, n)`` would make every bin zero-width."""
    rng = np.random.default_rng(0)
    block = diagnostics._overlaid_histogram(np.full(100, 3.0), np.full(100, 3.0), bins=10, rng=rng, kde_cap=50)
    assert block['edges'][-1] > block['edges'][0]


def test_the_kde_curves_arrive_TOGETHER_or_not_at_all(monkeypatch):
    """The documented reason all three keys are assigned in one statement: the report indexes ``pred_kde`` and
    ``true_kde`` whenever ``kde_grid`` is present, so a partial failure would crash the figure rather than skip it."""
    rng = np.random.default_rng(0)
    block = diagnostics._overlaid_histogram(rng.normal(size=500), rng.normal(size=500), 20, rng, kde_cap=200)
    assert {'kde_grid', 'pred_kde', 'true_kde'} <= set(block), 'scipy is installed, so the KDE must be present'
    assert block['pred_kde'].shape == block['kde_grid'].shape

    import scipy.stats

    def explode(*args, **kwargs):
        raise ValueError('singular covariance')

    monkeypatch.setattr(scipy.stats, 'gaussian_kde', explode)
    degraded = diagnostics._overlaid_histogram(rng.normal(size=500), rng.normal(size=500), 20, rng, kde_cap=200)
    assert 'edges' in degraded
    assert not ({'kde_grid', 'pred_kde', 'true_kde'} & set(degraded)), 'a KDE failure must drop all three'
