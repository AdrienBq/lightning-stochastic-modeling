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
