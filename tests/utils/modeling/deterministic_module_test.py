"""Tests for src/utils/modeling/deterministic_module.py — the baseline U-net, and the upstream for both other families.

The module is 47 lines: after block 3d extracted ``UnetModuleBase``, all it declares is its checkpoint marker and its
``predict_step``. So the tests here are about the two things it OWNS — the point-prediction contract and the marker —
plus the mode dispatch end to end, which is the part every other family inherits.

Everything shared lives in ``unet_module_base_test.py``.

⚠️ What was DELETED from branch A's 452-line version is a large part of what matters, and the tests that pin the
absences are as load-bearing as the ones that pin behaviour: the gamma F-transform (so training space == evaluation
space) and the whole classifier/hierarchy gate (a bounded 0-24 target needs no dry-cell mask).
"""
import numpy as np
import pytest
import torch

from src.utils.modeling.deterministic_module import CHECKPOINT_MARKER, PHASES, DeterministicUnetModule


@pytest.fixture
def make_module(unet_trial, normalization, target_stats):
    def build(mode='daily', **kwargs):
        return DeterministicUnetModule(unet_trial(**kwargs), 5, target_stats(mode=mode), normalization).eval()
    return build


# =====================================================================================================================
# The point-prediction contract
# =====================================================================================================================
def test_predict_step_returns_a_point_prediction_with_no_ensemble_keys(make_module, batch):
    """``ensemble_members`` must be ABSENT, not present with one member: a single member makes
    ``spread_skill_sums`` NaN through ``ddof=1`` and makes the report draw a 6-panel stochastic layout from a point
    forecast."""
    module = make_module()
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)

    assert set(output) == {'prediction', 'probability', 'observation'}
    assert output['prediction'].shape == (x.shape[0], x.shape[2], x.shape[3])
    assert output['observation'].shape == (x.shape[0], x.shape[2], x.shape[3])


def test_the_observation_is_returned_detached_on_the_cpu(make_module, batch):
    """The evaluation stage concatenates these across batches and hands them to numpy, so a retained graph would hold
    the whole split's activations."""
    module = make_module()
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert not output['observation'].requires_grad
    assert output['observation'].device.type == 'cpu'
    assert output['prediction'].device.type == 'cpu'


# =====================================================================================================================
# The mode dispatch, end to end
# =====================================================================================================================
def test_daily_mode_predicts_bounded_lightning_hours(make_module, batch):
    module = make_module(mode='daily')
    x, y = batch()
    with torch.no_grad():
        module.net.head.bias.fill_(100.0)                 # drive the head past the ceiling
        output = module.predict_step((x, y), 0)
    assert float(output['prediction'].min()) >= 0.0
    assert float(output['prediction'].max()) <= 24.0 + 1e-5
    assert output['probability'] is None, 'a daily run has no probabilistic head'


def test_hourly_mode_predicts_a_probability(make_module, batch):
    """The head emits a raw LOGIT and the sigmoid lives on the prediction path — which is what makes
    ``output_activation`` inherently daily-only and needs no hourly entry."""
    module = make_module(mode='hourly', loss={'name': 'brier'})
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert float(output['prediction'].min()) >= 0.0
    assert float(output['prediction'].max()) <= 1.0


def test_hourly_mode_reports_the_prediction_AS_the_probability(make_module, batch):
    """In the classification task the prediction IS the occurrence probability, so the two keys are the same array
    rather than one being derived from the other."""
    module = make_module(mode='hourly', loss={'name': 'brier'})
    x, y = batch()
    with torch.no_grad():
        output = module.predict_step((x, y), 0)
    assert output['probability'] is not None
    assert torch.equal(output['prediction'], output['probability'])


def test_hourly_mode_never_clamps_at_max_hours(make_module):
    """A probability clamped at 24 would be a no-op, but the code must not go through that path at all — the two
    task-specific final maps are exclusive."""
    module = make_module(mode='hourly', loss={'name': 'brier'})
    with torch.no_grad():
        extreme = module._to_prediction(torch.tensor([[-50.0, 50.0]]))
    assert float(extreme.min()) >= 0.0 and float(extreme.max()) <= 1.0


# =====================================================================================================================
# Phases
# =====================================================================================================================
def test_the_phase_list_shrank_from_As_five_to_three():
    """A had ``joint`` / ``classifier`` / ``regressor`` plus the two calibrations. The gate is gone, so the first three
    collapse into one ``train``."""
    assert PHASES == ('train', 'occurrence_calibration', 'regression_calibration')


def test_without_calibration_only_the_train_phase_runs(make_module):
    assert make_module().training_phases() == ('train',)


def test_daily_calibration_adds_the_regression_phase(make_module):
    module = make_module(calibration={'occurrence': 'none',
                                      'regression': {'structure': 'monotone_smooth', 'objective': 'pointwise',
                                                     'num_sigmoids': 4, 'huber_delta': 1.0}})
    assert module.training_phases() == ('train', 'regression_calibration')


def test_hourly_calibration_adds_the_occurrence_phase(make_module):
    module = make_module(mode='hourly', loss={'name': 'brier'},
                         calibration={'occurrence': 'platt',
                                      'regression': {'structure': 'none', 'objective': 'pointwise'}})
    assert module.training_phases() == ('train', 'occurrence_calibration')


def test_the_occurrence_calibration_is_inert_in_daily_mode(make_module):
    """``calibration.occurrence`` is kept in all three search spaces but calibrates the HOURLY head, so requesting it
    on a daily run must add no phase rather than fitting a Platt layer to lightning-hours."""
    module = make_module(calibration={'occurrence': 'platt',
                                      'regression': {'structure': 'none', 'objective': 'pointwise'}})
    assert 'occurrence_calibration' not in module.training_phases()


def test_the_regression_calibration_is_inert_in_hourly_mode(make_module):
    """And the converse: the monotone hour warp is daily-only."""
    module = make_module(mode='hourly', loss={'name': 'brier'},
                         calibration={'occurrence': 'none',
                                      'regression': {'structure': 'monotone_smooth', 'objective': 'pointwise',
                                                     'num_sigmoids': 4, 'huber_delta': 1.0}})
    assert 'regression_calibration' not in module.training_phases()


# =====================================================================================================================
# The checkpoint marker — every checkpoint is a valid upstream for both stochastic families
# =====================================================================================================================
def test_the_checkpoint_marker_is_the_family_name(make_module):
    checkpoint = {}
    make_module().on_save_checkpoint(checkpoint)
    assert checkpoint['module_class'] == CHECKPOINT_MARKER == 'deterministic_unet'


def test_the_saved_hyperparameters_carry_what_a_warm_start_needs(make_module, save_checkpoint):
    """``from_upstream`` reads the architecture, the channel count and the mode back out of the checkpoint, so all
    three have to be recorded — a checkpoint missing any of them cannot be warm-started from."""
    import torch as torch_module

    path = save_checkpoint(make_module(), 'upstream.ckpt')
    saved = torch_module.load(path, map_location='cpu', weights_only=False)['hyper_parameters']
    assert 'unet' in saved['trial']
    assert saved.get('in_channels') == 5
    assert (saved.get('target_stats') or {}).get('mode') == 'daily'


# =====================================================================================================================
# What was deleted — the absences are load-bearing
# =====================================================================================================================
def test_there_is_no_target_transform_anywhere(make_module):
    """The prize of the whole scope decision: training space == evaluation space, so there is no back-transform and no
    "which space is this tensor in?" question. A reintroduced transform would make every reported metric incomparable
    with the loss it was trained under."""
    module = make_module()
    for attribute in ('transform', 'transform_enabled', '_to_original_space'):
        assert not hasattr(module, attribute), attribute


def test_there_is_no_classifier_gate(make_module):
    """A's hierarchy masked dry cells with a classifier and trained the regressor on wet cells only. It existed for an
    UNBOUNDED count target; on a bounded 0-24 one it is meaningless, and it was dropped along with the report-only
    probability head."""
    module = make_module()
    for attribute in ('hierarchy_enabled', 'masking', 'recall_target', 'cls_threshold', 'compose_prediction'):
        assert not hasattr(module, attribute), attribute


def test_the_net_has_a_single_head(make_module):
    module = make_module()
    assert hasattr(module.net, 'head')
    assert not hasattr(module.net, 'classifier_head')
