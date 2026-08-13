"""Tests for src/utils/modeling/mc_dropout_eval.py — the adapter that puts MC-dropout on the shared contract.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_probabilistic_eval_compat.py``.

This wrapper is the mechanism behind the "one evaluation for all families" invariant: it is what lets MC-dropout feed
the SHARED ``scores.ensemble_partials`` rather than needing a family-specific evaluation path. So the tests are about
the contract's shape, not about the model.
"""
import inspect

import numpy as np
import pytest
import torch

from src.utils.modeling.mc_dropout_eval import MCDropoutEnsembleModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule

MEMBERS = 4
OCCURRENCE_EVENT = (0.0, True)
ENSEMBLE_KEYS = {'prediction', 'ensemble_members', 'probability', 'observation', 'ensemble_partials'}


@pytest.fixture
def adapter(mc_trial, normalization, target_stats):
    def build(mode='daily', ensemble_size=MEMBERS, **kwargs):
        module = MCDropoutModule(mc_trial(**kwargs), 5, target_stats(mode=mode), normalization).eval()
        wrapped = MCDropoutEnsembleModule(module)
        wrapped.eval_ensemble_size = ensemble_size
        wrapped.eval_occurrence_event = OCCURRENCE_EVENT
        return wrapped
    return build


def test_the_ensemble_contract_keys_and_shapes(adapter, batch):
    wrapped = adapter()
    x, y = batch()
    with torch.no_grad():
        output = wrapped.predict_step((x, y), 0)

    assert set(output) == ENSEMBLE_KEYS
    assert output['prediction'].shape == (x.shape[0], x.shape[2], x.shape[3])
    assert output['ensemble_members'].shape == (x.shape[0], MEMBERS, x.shape[2], x.shape[3])
    assert output['observation'].shape == (x.shape[0], x.shape[2], x.shape[3])
    assert isinstance(output['ensemble_partials'], dict) and output['ensemble_partials']


def test_the_prediction_is_the_ensemble_mean(adapter, batch):
    """The contract: ``prediction`` is the ensemble MEAN for a stochastic family, because that is what the pointwise,
    skill, categorical and calibration scores read."""
    wrapped = adapter()
    x, y = batch()
    with torch.no_grad():
        output = wrapped.predict_step((x, y), 0)
    assert torch.allclose(output['prediction'], output['ensemble_members'].mean(dim=1), atol=1e-6)


def test_a_single_member_request_falls_back_to_the_deterministic_dict(adapter, batch):
    """``ensemble_size <= 1`` must reproduce a deterministic family's dict exactly — no ``ensemble_members``, no
    partials. Otherwise the evaluation stage would try to finalize an ensemble suite from one member and get NaN
    spread-skill through ``ddof=1``."""
    wrapped = adapter(ensemble_size=1)
    x, y = batch()
    with torch.no_grad():
        output = wrapped.predict_step((x, y), 0)
    assert set(output) == {'prediction', 'probability', 'observation'}
    assert output['prediction'].shape == (x.shape[0], x.shape[2], x.shape[3])


def test_a_channel_mismatch_raises_rather_than_broadcasting(adapter, batch):
    """The prepared directory and the checkpoint can disagree on the channel count — most easily by preparing without
    ``upstream-model-path`` and evaluating a residual-trained checkpoint. Raising here names the cause; letting the
    first conv fail would surface as a shape error deep in the net."""
    wrapped = adapter()
    x, y = batch(channels=6)
    with pytest.raises(ValueError):
        with torch.no_grad():
            wrapped.predict_step((x, y), 0)


def test_the_adapter_exposes_the_expected_channel_count(adapter):
    assert adapter().expected_in_channels == 5


def test_the_adapter_forwards_the_wrapped_target_stats(adapter, target_stats):
    """The evaluation stage reads ``target_stats`` off whatever module the registry handed it, so the wrapper has to
    pass it through rather than shadowing it with an empty dict."""
    assert adapter().target_stats['mode'] == 'daily'


def test_hourly_mode_reports_a_probability(adapter, batch):
    wrapped = adapter(mode='hourly', loss={'name': 'brier'})
    x, y = batch()
    with torch.no_grad():
        output = wrapped.predict_step((x, y), 0)
    assert output['probability'] is not None
    assert float(output['probability'].min()) >= 0.0 and float(output['probability'].max()) <= 1.0


def test_daily_mode_reports_no_probability(adapter, batch):
    """A daily run has no probabilistic head, so ``brier_skill_score`` / ``explained_deviance`` / ``dice_occurrence``
    must be absent rather than computed from lightning-hours."""
    wrapped = adapter()
    x, y = batch()
    with torch.no_grad():
        output = wrapped.predict_step((x, y), 0)
    assert output['probability'] is None


def test_the_partials_are_what_the_shared_finalizer_consumes(adapter, batch):
    """The point of the adapter: the partials it emits must be the SAME structure the diffusion family emits, so
    ``finalize_ensemble_metrics`` treats both identically."""
    from src.utils.metrics.evaluation import finalize_ensemble_metrics, merge_ensemble_partials

    wrapped = adapter()
    x, y = batch()
    with torch.no_grad():
        output = wrapped.predict_step((x, y), 0)

    merged = merge_ensemble_partials(None, output['ensemble_partials'])
    spec = {'crps': {}, 'almost_fair_crps': {}, 'spread_skill_ratio': {}, 'rank_histogram': {}}
    flat, curves = finalize_ensemble_metrics(merged, spec, MEMBERS)
    assert any('crps' in name for name in flat), sorted(flat)
    assert 'rank_histogram' in curves


def test_the_adapter_is_a_lightning_module_so_trainer_predict_drives_it(adapter):
    """The evaluation stage calls ``trainer.predict(module, loader)``, so the wrapper has to be a LightningModule
    rather than a plain object with a ``predict_step`` method."""
    import lightning

    assert isinstance(adapter(), lightning.LightningModule)


def test_predict_step_keeps_the_lightning_signature(adapter):
    parameters = list(inspect.signature(adapter().predict_step).parameters)
    assert parameters[:2] == ['batch', 'batch_idx']
