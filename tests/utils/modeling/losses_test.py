"""Tests for src/utils/modeling/losses.py — training losses for both tasks, shared by all three families.

Neither source branch tested this file, and it is where the merge did the most reconciling: branch D's 408 lines and
branch A's 130 had two different ``crps`` implementations, three different binary-loss key sets, and a
``build_finetune_loss`` that A's vendored copy could not satisfy.

The headline contract is the **uniform logit contract** (Step 3 block 3b-1r): every binary loss takes LOGITS and
sigmoids internally, so the three input spaces D and A between them assumed collapse to one and a mismatch becomes
impossible rather than tag-guarded. ``needs_ensemble`` survives only because ``crps_binary``'s sample axis is a SHAPE
distinction a ``[B, H, W]`` batch silently satisfies.
"""
import numpy as np
import pytest
import torch

from src.utils.metrics import scores
from src.utils.modeling import losses


def BINARY_CONFIG(name):
    """Every key any binary loss reads. ``focal_bce`` needs BOTH ``focal_gamma`` and ``positive_class_weight`` —
    it is the one binary loss that brings its own class reweighting, which is why ``apply_constraints`` forces
    ``intensity_weight_gamma = 0`` alongside it."""
    return {'name': name, 'focal_gamma': 2.0, 'positive_class_weight': 4.0, 'smooth': 1.0, 'beta': 0.7}


PRESENT = ('intensity_weights', '_weighted_masked_mean', 'weighted_mse', 'weighted_mae', 'weighted_rmse',
           'asymmetric_huber', 'psd_penalty', 'wmae_psd', 'wmse_psd', 'focal_bce_with_logits', 'dice_loss',
           'brier_loss', 'crps', 'almost_fair_crps', 'afcrps_psd', 'crps_binary',
           'build_regression_loss', 'build_binary_loss', 'build_ensemble_loss', 'BinaryLoss')

REMOVED = ('mae', 'rmse',                       # absorbed by their weighted forms at gamma = 0
           'tweedie_deviance', 'poisson_nll',   # unbounded-count machinery; Poisson puts mass above 24 h/day
           'TRANSFORM_COMPATIBLE_LOSSES',       # the gamma F-transform is gone
           'build_finetune_loss')               # renamed build_ensemble_loss for what it actually is


# =====================================================================================================================
# The loss name sets are exactly what the search spaces offer
# =====================================================================================================================
@pytest.mark.parametrize('name', PRESENT)
def test_the_merged_surface_is_all_present(name):
    """The union both branches had to produce. ``wmse_psd`` is in the list because the search spaces offered a name
    NEITHER branch implemented, so it was written for this merge."""
    assert hasattr(losses, name)


@pytest.mark.parametrize('name', REMOVED)
def test_a_removed_loss_stays_removed(name):
    """Each removal is a decision, not a cleanup. ``poisson_nll`` is the sharpest: it is UNBOUNDED, so on a target
    capped at 24 hours/day it puts probability mass above the physical ceiling — actively wrong here, not merely
    unnecessary. ``mae`` / ``rmse`` are absorbed at ``gamma = 0``, which is why the search space must allow that value.
    """
    assert not hasattr(losses, name)


def test_the_regression_loss_set_matches_the_search_spaces(search_spaces):
    """The tuple and the YAML must agree in BOTH directions: a name only in the YAML raises at trial 0, and a name only
    in the tuple is dead code that looks supported."""
    for family, space in search_spaces.items():
        offered = set(space['loss']['name']['choices'])
        assert offered <= set(losses.REGRESSION_LOSSES) | set(losses.BINARY_LOSSES), family


@pytest.mark.parametrize('family', ['deterministic_unet', 'mc_dropout', 'diffusion'])
def test_no_shipped_search_space_offers_a_binary_loss(family, search_spaces):
    """All three shipped spaces are DAILY spaces, so all three offer only the six distance losses. This is the state,
    not an omission: ``deterministic_unet/search_space_daily.yaml`` says an hourly space "just sets ``loss.name`` to one of
    focal_bce / dice / brier / crps_binary and changes nothing else", and no hourly space exists yet.

    Pinned so that when one is added, whoever adds it sees this test and updates it deliberately.
    """
    offered = set(search_spaces[family]['loss']['name']['choices'])
    assert not offered & set(losses.BINARY_LOSSES), f'{family} now offers a binary loss — is it an hourly space?'
    assert offered <= set(losses.REGRESSION_LOSSES)


def test_the_binary_loss_names_are_the_four_the_design_settled_on():
    """``crps_binary`` is the one that must never reach the deterministic family: it needs a genuine ensemble, and a
    single forward pass gives N=1, a zero spread term and a silent MAE-on-probabilities."""
    assert set(losses.BINARY_LOSSES) == {'focal_bce', 'dice', 'brier', 'crps_binary'}


def test_the_calibration_objectives_are_not_backbone_losses():
    """``log1p_huber`` / ``log1p_huber_quantile`` are reached by ``calibration.regression.objective``, never by
    ``loss.name``. Offering them as a backbone loss would be a category error."""
    assert 'log1p_huber' not in losses.REGRESSION_LOSSES
    assert 'log1p_huber_quantile' not in losses.REGRESSION_LOSSES
    with pytest.raises((ValueError, KeyError)):
        losses.build_regression_loss({'name': 'log1p_huber', 'intensity_weight_gamma': 0.0})


# =====================================================================================================================
# THE uniform logit contract
# =====================================================================================================================
@pytest.mark.parametrize('name', ['focal_bce', 'dice', 'brier', 'crps_binary'])
def test_every_binary_loss_takes_logits(name):
    """The contract in one assertion per loss: pass an UNBOUNDED logit and the loss must be finite, because it applies
    the sigmoid itself. A loss expecting a probability would either clip to a degenerate value or produce inf/nan on a
    logit of -20."""
    built = losses.build_binary_loss(BINARY_CONFIG(name))
    logits = torch.tensor([[-20.0, -2.0, 0.0, 2.0, 20.0]])
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
    value = built.fn(logits, target)
    assert torch.isfinite(value).all(), f'{name} did not survive an unbounded logit'


@pytest.mark.parametrize('name', ['focal_bce', 'dice', 'brier', 'crps_binary'])
def test_every_binary_loss_prefers_a_correct_logit_to_a_wrong_one(name):
    """The direction check that makes the finiteness check above mean something: a loss can be finite and still be
    wired backwards."""
    built = losses.build_binary_loss(BINARY_CONFIG(name))
    target = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    confident_right = torch.tensor([[-4.0, 4.0, 4.0, -4.0]])
    confident_wrong = torch.tensor([[4.0, -4.0, -4.0, 4.0]])
    assert float(built.fn(confident_right, target)) < float(built.fn(confident_wrong, target)), name


def test_only_crps_binary_needs_an_ensemble():
    """The one distinction the logit contract could NOT remove: ``crps_binary``'s first argument is ``[N, *spatial]``,
    a sample axis that a ``[B, H, W]`` batch satisfies by accident. So it stays flagged rather than inferred."""
    assert losses.build_binary_loss(BINARY_CONFIG('crps_binary')).needs_ensemble is True
    for name in ('focal_bce', 'dice', 'brier'):
        built = losses.build_binary_loss(BINARY_CONFIG(name))
        assert built.needs_ensemble is False, name


def test_the_binary_builder_reads_the_name_key(search_spaces):
    """It once read ``loss_config['loss']`` while the search spaces write ``loss.name`` — a KeyError on every hourly
    trial. Driving it from the real YAML key is what catches that class of drift."""
    for family, space in search_spaces.items():
        for name in space['loss']['name']['choices']:
            if name in losses.BINARY_LOSSES:
                assert losses.build_binary_loss(BINARY_CONFIG(name)) is not None


def test_brier_loss_on_logits_EQUALS_the_evaluation_brier_score_on_probabilities():
    """The training/evaluation agreement for the binary head, the counterpart of the CRPS agreement below. The loss takes
    logits and sigmoids internally; ``scores.brier_score`` takes probabilities. They must be the same number, or the
    reported score does not measure the objective that was optimised."""
    torch.manual_seed(3)
    logits = torch.randn(3, 8, 9) * 2
    labels = (torch.rand(3, 8, 9) < 0.3).float()

    assert float(losses.brier_loss(logits, labels)) == pytest.approx(
        float(scores.brier_score(torch.sigmoid(logits).numpy(), labels.numpy())), abs=1e-6
    )


def test_crps_binary_SIGMOIDS_its_samples_before_the_crps_formula():
    """The logit contract reaching inside the ensemble axis: the members arrive as logits and the CRPS has to be computed
    on probabilities. Applying the formula to raw logits would give a finite, plausible, wrong number.

    Compared against the energy form written out independently — ``E|p - y| - 0.5 E|p - p'|`` via the sorted-sample
    identity — so the assertion pins the whole computation, not just that a sigmoid happened somewhere.
    """
    torch.manual_seed(4)
    members = 8
    logit_samples = torch.randn(members, 3, 5, 6)                # [N, B, H, W]
    labels = (torch.rand(3, 5, 6) < 0.1).float()

    probabilities = torch.sigmoid(logit_samples)
    absolute = (probabilities - labels.unsqueeze(0)).abs().mean(dim=0).mean()
    coefficients = (2.0 * torch.arange(float(members)) - members + 1).view(-1, 1, 1, 1)
    spread = (coefficients * torch.sort(probabilities, dim=0).values).sum(dim=0).div(members ** 2).mean()
    expected = float(absolute - 0.5 * spread)

    assert float(losses.crps_binary(logit_samples, labels)) == pytest.approx(expected, abs=1e-6)


def test_crps_binary_at_one_member_degenerates_to_a_MAE_on_probabilities():
    """Why ``crps_binary`` is not offered to the deterministic family: a single forward pass gives N = 1, the spread term
    vanishes, and the loss silently becomes an MAE on probabilities — an improper score on a binary target (see
    ``test_the_mae_is_IMPROPER...`` in scores_test.py). No error, just a different objective."""
    torch.manual_seed(5)
    single = torch.randn(1, 2, 5, 5) * 2
    labels = (torch.rand(2, 5, 5) < 0.3).float()

    assert float(losses.crps_binary(single, labels)) == pytest.approx(
        float((torch.sigmoid(single[0]) - labels).abs().mean()), abs=1e-6
    )


@pytest.mark.parametrize('name', ['focal_bce', 'dice', 'brier'])
def test_passing_a_PROBABILITY_where_a_logit_belongs_changes_the_loss_silently(name):
    """The footgun the uniform logit contract narrows but cannot remove: a probability in [0, 1] is a perfectly valid
    logit, so the wrong input space is finite and unremarkable rather than an error. The contract's value is that there
    is now only ONE convention to get right, not that a mistake is caught."""
    torch.manual_seed(6)
    logits = torch.randn(2, 6, 6) * 2
    labels = (torch.rand(2, 6, 6) < 0.3).float()
    built = losses.build_binary_loss(BINARY_CONFIG(name)).fn

    right = float(built(logits, labels))
    wrong = float(built(torch.sigmoid(logits), labels))
    assert np.isfinite(wrong), 'the wrong space does not raise — that is the point'
    assert wrong != pytest.approx(right, abs=1e-6)


# =====================================================================================================================
# The weighting scheme
# =====================================================================================================================
def test_gamma_zero_makes_the_weights_uniform():
    """``intensity_weights(y, gamma) = (1 + y)^gamma``, so gamma = 0 means UNWEIGHTED — which is how ``weighted_mae``
    covers plain ``mae``. The search space must therefore allow 0.0, and this is why."""
    y = torch.tensor([0.0, 3.0, 24.0])
    assert torch.allclose(losses.intensity_weights(y, 0.0), torch.ones_like(y))


def test_the_weights_grow_with_the_observed_intensity():
    y = torch.tensor([0.0, 1.0, 5.0])
    weights = losses.intensity_weights(y, 2.0)
    assert float(weights[0]) == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_the_weights_come_from_the_RAW_target():
    """Not from the prediction, and not from a transformed space — there is no transformed space any more."""
    y = torch.tensor([4.0])
    assert float(losses.intensity_weights(y, 1.0)) == pytest.approx(5.0)


def test_weighted_mae_at_gamma_zero_is_a_plain_mae():
    """The identity that lets one loss name cover two behaviours."""
    prediction = torch.tensor([[1.0, 4.0, 0.0]])
    target = torch.tensor([[0.0, 2.0, 3.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    built = losses.build_regression_loss({'name': 'weighted_mae', 'intensity_weight_gamma': 0.0})
    assert float(built(prediction, target, losses.intensity_weights(target, 0.0), mask)) == pytest.approx(
        float((prediction - target).abs().mean())
    )


def test_weighted_rmse_at_gamma_zero_is_a_plain_rmse():
    """The same absorption as ``weighted_mae``, and worth pinning separately because RMSE takes the square root AFTER
    the weighted reduction — doing it per-cell first would give a different number that still looks plausible."""
    torch.manual_seed(1)
    target = torch.rand(4, 6, 7) * 24
    prediction = (target + torch.randn_like(target)).clamp(0, 24)
    mask = torch.ones_like(target)
    uniform = losses.intensity_weights(target, 0.0)

    assert float(losses.weighted_rmse(prediction, target, uniform, mask)) == pytest.approx(
        float(((prediction - target) ** 2).mean().sqrt()), abs=1e-6
    )


def test_a_positive_gamma_actually_CHANGES_the_loss():
    """The converse of the absorption identity: if weighting were silently ignored, ``gamma`` would be a search
    dimension with no effect and every trial would explore it for nothing."""
    torch.manual_seed(1)
    target = torch.rand(4, 6, 7) * 24
    prediction = (target + torch.randn_like(target)).clamp(0, 24)
    mask = torch.ones_like(target)

    unweighted = losses.weighted_mae(prediction, target, losses.intensity_weights(target, 0.0), mask)
    weighted = losses.weighted_mae(prediction, target, losses.intensity_weights(target, 2.0), mask)
    assert float(weighted) != pytest.approx(float(unweighted))


def test_wmae_psd_at_alpha_one_is_the_plain_weighted_mae():
    """The MAE-flavoured sibling of the ``wmse_psd`` identity below. Both are checked because the two composites are
    separate functions, so one can lose its spectral switch-off without the other."""
    torch.manual_seed(1)
    target = torch.rand(2, 12, 12) * 5
    prediction = torch.rand(2, 12, 12) * 5
    mask = torch.ones_like(target)
    uniform = losses.intensity_weights(target, 0.0)

    assert float(losses.wmae_psd(prediction, target, uniform, mask, alpha=1.0)) == pytest.approx(
        float(losses.weighted_mae(prediction, target, uniform, mask)), abs=1e-6
    )


# =====================================================================================================================
# psd_penalty — the spectral term the two composites share
# =====================================================================================================================
def test_the_psd_penalty_is_ZERO_for_an_identical_field():
    """It compares radial power spectra, so a field against itself must score exactly 0 — otherwise the composite
    losses carry a constant offset and ``alpha`` no longer interpolates between two comparable terms."""
    torch.manual_seed(1)
    target = torch.rand(4, 12, 12) * 24
    assert abs(float(losses.psd_penalty(target, target))) < 1e-5


def test_the_psd_penalty_PUNISHES_blur():
    """The failure mode it exists for. A blurred prediction can score well on any pointwise distance while destroying
    the small-scale structure — which on a 99.93 %-zero field is most of the signal."""
    torch.manual_seed(1)
    target = torch.rand(4, 12, 12) * 24
    blurred = torch.nn.functional.avg_pool2d(target.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    assert float(losses.psd_penalty(blurred, target)) > 0.1


def test_the_psd_penalty_stays_within_the_unit_interval():
    """It is combined as ``alpha * distance + (1 - alpha) * psd_penalty``, so an unbounded spectral term would dominate
    the sum at any alpha below 1 and the search over alpha would be meaningless."""
    torch.manual_seed(1)
    target = torch.rand(4, 12, 12) * 24
    blurred = torch.nn.functional.avg_pool2d(target.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    assert 0.0 <= float(losses.psd_penalty(blurred, target)) <= 1.0


def test_wmse_psd_at_alpha_one_is_the_plain_weighted_mse():
    """``alpha * weighted_mse + (1 - alpha) * psd_penalty``, so alpha = 1 must switch the spectral term off exactly.
    ``wmse_psd`` is implemented by neither source branch — the search spaces offered a name nobody had written."""
    prediction = torch.rand(2, 12, 12) * 5
    target = torch.rand(2, 12, 12) * 5
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = losses.intensity_weights(target, 0.5)

    combined = losses.build_regression_loss({'name': 'wmse_psd', 'intensity_weight_gamma': 0.5, 'alpha': 1.0})
    plain = losses.build_regression_loss({'name': 'weighted_mse', 'intensity_weight_gamma': 0.5})
    assert float(combined(prediction, target, weights, mask)) == pytest.approx(
        float(plain(prediction, target, weights, mask)), rel=1e-6
    )


def test_a_lower_alpha_gives_the_spectral_term_weight():
    prediction = torch.rand(2, 12, 12) * 5
    target = torch.rand(2, 12, 12) * 5
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.ones_like(target)
    at_one = losses.build_regression_loss({'name': 'wmse_psd', 'intensity_weight_gamma': 0.0, 'alpha': 1.0})
    at_half = losses.build_regression_loss({'name': 'wmse_psd', 'intensity_weight_gamma': 0.0, 'alpha': 0.5})
    assert float(at_one(prediction, target, weights, mask)) != pytest.approx(
        float(at_half(prediction, target, weights, mask))
    )


@pytest.mark.parametrize('name', ['weighted_mae', 'weighted_rmse', 'weighted_mse', 'asymmetric_huber',
                                  'wmae_psd', 'wmse_psd'])
def test_every_regression_loss_normalises_by_the_effective_WEIGHT_not_the_cell_count(name):
    """CLAUDE.md invariant: every pointwise loss reduces through ``_weighted_masked_mean``, which divides by the SUM OF
    WEIGHTS rather than by the number of cells. Inlining the reduction risks a loss on a different scale from its
    siblings, which makes two trials' numbers incomparable.

    Executable form: doubling every weight must leave a weighted MEAN unchanged. A count-normalised loss would double.
    """
    prediction = torch.rand(2, 8, 8) * 5
    target = torch.rand(2, 8, 8) * 5
    mask = torch.ones_like(target, dtype=torch.bool)
    built = losses.build_regression_loss({'name': name, 'intensity_weight_gamma': 0.0, 'alpha': 1.0,
                                         'asymmetry_tau': 0.7, 'huber_delta': 1.0})
    weights = torch.ones_like(target)
    assert float(built(prediction, target, weights, mask)) == pytest.approx(
        float(built(prediction, target, weights * 2.0, mask)), rel=1e-5
    ), f'{name} is normalised by the cell count, not the weight sum'


def test_EVERY_alias_in_EVERY_shipped_space_resolves_and_is_finite(search_spaces):
    """The sweep the gate ran per space rather than per name. The set-equality test above uses one space; this drives all
    three, because a family whose space offers a name the builder cannot resolve fails on trial 0 — after the pipeline
    has already prepared the data.

    Every key any regression loss reads is supplied, so an unresolved name is the only way this fails.
    """
    checked = 0
    for family, space in search_spaces.items():
        for name in space['loss']['name']['choices']:
            if name in losses.BINARY_LOSSES:
                continue
            built = losses.build_regression_loss({'name': name, 'intensity_weight_gamma': 0.5, 'alpha': 0.8,
                                                  'asymmetry_tau': 0.7, 'huber_delta': 1.0})
            prediction, target = torch.rand(2, 12, 12) * 5, torch.rand(2, 12, 12) * 5
            mask = torch.ones_like(target, dtype=torch.bool)
            value = float(built(prediction, target, losses.intensity_weights(target, 0.5), mask))
            assert np.isfinite(value), f'{family}: {name} -> {value}'
            checked += 1
    assert checked >= 3 * 6, f'only {checked} (family, loss) pairs exercised'


def test_the_almost_fair_crps_is_never_below_the_plain_crps():
    """``almost_fair`` gives less credit for spread than the fair estimator, so on the same samples it can only score
    higher. If the ordering inverted, the ``beta`` interpolation would be pointing the wrong way and a finetuning run
    would be rewarded for over-dispersing."""
    torch.manual_seed(7)
    samples = torch.randn(9, 3, 6, 6) * 3 + 6
    observed = torch.rand(3, 6, 6) * 12

    assert float(losses.almost_fair_crps(samples, observed)) >= float(losses.crps(samples, observed))


@pytest.mark.parametrize('name', ['weighted_mae', 'weighted_rmse', 'weighted_mse'])
def test_the_mask_excludes_cells_entirely(name):
    """A masked cell must not contribute at all, however wrong it is."""
    prediction = torch.tensor([[1.0, 1000.0]])
    target = torch.tensor([[1.0, 0.0]])
    weights = torch.ones_like(target)
    built = losses.build_regression_loss({'name': name, 'intensity_weight_gamma': 0.0})
    masked = torch.tensor([[True, False]])
    assert float(built(prediction, target, weights, masked)) == pytest.approx(0.0, abs=1e-6)


def test_asymmetric_huber_penalises_the_two_error_signs_differently():
    """``asymmetry_tau > 0.5`` makes under-prediction cost more, which is the point on a field that is 99.93 % zero:
    a symmetric loss is minimised by predicting nothing."""
    target = torch.tensor([[5.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.ones_like(target)
    built = losses.build_regression_loss({'name': 'asymmetric_huber', 'intensity_weight_gamma': 0.0,
                                         'asymmetry_tau': 0.8, 'huber_delta': 1.0})
    under = float(built(torch.tensor([[3.0]]), target, weights, mask))
    over = float(built(torch.tensor([[7.0]]), target, weights, mask))
    assert under > over, 'tau > 0.5 must punish under-prediction harder'


# =====================================================================================================================
# THE divergence this whole merge exists to avoid
# =====================================================================================================================
def test_the_training_crps_agrees_with_the_evaluation_crps():
    """Branch A and branch D each shipped a ``crps`` and a ``crps_ensemble`` with different contracts. If the trained
    objective and the reported score disagree, every ranking in the project is quietly wrong and nothing errors. So the
    two implementations are checked against each other on the same samples."""
    rng = np.random.default_rng(0)
    members = rng.gamma(1.2, 1.0, size=(8, 40))
    observation = rng.gamma(1.2, 1.0, size=40)

    trained = float(losses.crps(torch.tensor(members), torch.tensor(observation)))
    reported = scores.crps_ensemble(members, observation)
    assert trained == pytest.approx(reported, rel=1e-9)


def test_the_almost_fair_crps_also_agrees():
    rng = np.random.default_rng(1)
    members = rng.gamma(1.0, 1.0, size=(6, 30))
    observation = rng.gamma(1.0, 1.0, size=30)
    trained = float(losses.almost_fair_crps(torch.tensor(members), torch.tensor(observation)))
    assert trained == pytest.approx(scores.almost_fair_crps_ensemble(members, observation), rel=1e-9)


# =====================================================================================================================
# The two builders stay separate
# =====================================================================================================================
def test_the_ensemble_loss_is_a_separate_builder():
    """They are used TOGETHER in one expression — ``loss_reg + finetune_loss_weight * loss_crps`` — and their
    signatures differ, ``(pred, target, weights, mask)`` versus ``(samples, target)``. Folding them into one builder
    was considered and rejected for exactly that reason."""
    regression = losses.build_regression_loss({'name': 'weighted_mae', 'intensity_weight_gamma': 0.0})
    ensemble = losses.build_ensemble_loss({'enabled': True, 'loss': 'almost_fair_crps', 'loss_weight': 0.5,
                                          'beta': 0.7, 'samples': 4})
    assert regression is not ensemble and callable(regression) and callable(ensemble)


@pytest.mark.parametrize('name', ['crps', 'almost_fair_crps', 'afcrps_psd'])
def test_every_ensemble_loss_name_builds(name):
    """Including ``afcrps_psd``, which A's vendored copy could not build — the reason A needed a registry workaround at
    load time (see registry_test.py)."""
    built = losses.build_ensemble_loss({'enabled': True, 'loss': name, 'loss_weight': 1.0, 'beta': 0.7,
                                       'alpha': 0.8, 'samples': 4})
    assert callable(built)


def test_the_builder_ignores_enabled_because_the_MODULE_gates_the_phase():
    """``enabled`` is not this function's business, and asserting otherwise was tempting: the builder happily returns a
    loss for a disabled section. The gate lives in the module, where ``set_phase`` raises if the finetune phase is
    requested with ``finetuning.enabled`` false and ``self.finetune_loss`` is left None.

    Pinned here so the responsibility split stays visible from either side.
    """
    assert callable(losses.build_ensemble_loss({'enabled': False}))
    assert callable(losses.build_ensemble_loss({}))                  # and it defaults the name


def test_the_default_ensemble_loss_is_the_bias_corrected_estimator():
    """``almost_fair_crps`` rather than plain ``crps``: the standard estimator is negatively biased in the spread term
    because the same N draws estimate both expectations, and the spread is exactly what this phase is calibrating."""
    assert losses.build_ensemble_loss({}) is losses.almost_fair_crps


def test_an_unknown_loss_name_raises_rather_than_defaulting():
    with pytest.raises((ValueError, KeyError)):
        losses.build_regression_loss({'name': 'tweedie_deviance', 'intensity_weight_gamma': 0.0})
    with pytest.raises((ValueError, KeyError)):
        losses.build_binary_loss({'name': 'weighted_bce'})


# =====================================================================================================================
# The two calibration objectives (moved here in block 3c-r)
# =====================================================================================================================
def test_the_calibration_objectives_reduce_through_the_shared_helper():
    """They hand-rolled their own masked mean while living in module.py, which is exactly the duplication that made
    ``_weighted_masked_mean`` unreachable to them. Equal to an independently-computed reduction."""
    prediction = torch.tensor([[1.0, 4.0, 9.0]])
    target = torch.tensor([[0.0, 5.0, 8.0]])
    mask = torch.tensor([[True, True, False]])

    elementwise = torch.nn.functional.huber_loss(
        torch.log1p(prediction), torch.log1p(target), reduction='none', delta=1.0
    )
    expected = float((elementwise * mask).sum() / mask.sum())
    assert float(losses.log1p_huber(prediction, target, mask, 1.0)) == pytest.approx(expected, rel=1e-6)


def test_the_calibration_objectives_are_zero_on_an_empty_mask():
    prediction, target = torch.rand(1, 5) * 10, torch.rand(1, 5) * 10
    empty = torch.zeros(1, 5, dtype=torch.bool)
    assert float(losses.log1p_huber(prediction, target, empty, 1.0)) == pytest.approx(0.0)
    assert float(losses.log1p_huber_quantile(prediction, target, empty, 1.0)) == pytest.approx(0.0)


def test_the_quantile_objective_differs_from_the_pointwise_one():
    """The quantile form pairs sorted predictions against sorted observations, so it measures distribution shape rather
    than per-cell error. On a field where the two orderings differ they must not coincide."""
    prediction = torch.tensor([[1.0, 9.0, 3.0, 7.0]])
    target = torch.tensor([[8.0, 2.0, 6.0, 4.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    assert float(losses.log1p_huber(prediction, target, mask, 1.0)) != pytest.approx(
        float(losses.log1p_huber_quantile(prediction, target, mask, 1.0))
    )


@pytest.mark.source_invariant
@pytest.mark.parametrize('name', ['log1p_huber', 'log1p_huber_quantile'])
def test_the_calibration_objectives_are_DEFINED_here_and_nowhere_else(name):
    """Block 3c-r moved them out of the deterministic family's module file. They are parameter-free loss functions that
    all three families reach through ``calibration.regression.objective``, so leaving them in one family's module made
    the other two import from it — and made them bypass ``_weighted_masked_mean``, which is private to this file.

    Checked by AST: a copy left behind in a module file is the failure this pins, and a substring search would match the
    import as readily as a definition.
    """
    import ast

    from src.utils.modeling import deterministic_module, losses as losses_module, unet_module_base

    def defined(module):
        tree = ast.parse(open(module.__file__).read())
        return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    def imported(module):
        tree = ast.parse(open(module.__file__).read())
        return {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                for alias in node.names}

    assert name in defined(losses_module), f'losses.py must DEFINE {name}'
    assert name not in defined(deterministic_module), f'a copy of {name} is left behind'
    assert name not in defined(unet_module_base), f'a copy of {name} is left behind'
    assert name in imported(unet_module_base), f'the shared base must IMPORT {name} from losses'


def test_the_calibration_objectives_take_no_weights_argument():
    """Deliberately unlike every other pointwise loss: the calibrator is fitted with a plain SYMMETRIC objective so it
    is decoupled from the intensity-weighted backbone loss. A ``weights`` parameter would invite the one thing the
    design rules out."""
    import inspect

    for objective in (losses.log1p_huber, losses.log1p_huber_quantile):
        assert 'weights' not in inspect.signature(objective).parameters
