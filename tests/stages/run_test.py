"""Tests for src/stages/run.py — the pipeline orchestrator.

Untouched template code, 397 lines, and the piece that makes every documented pipeline convention true: it substitutes
``{{$VAR}}``, classifies each parameter as an input or an output for the lazy cache, derives and exports
``PIPELINE_SEED``, and dispatches each stage as a subprocess.

A real run needs an MLflow tracking server, so the subprocess-level orchestration is Step 4's gate. What is tested here
is the plain-Python part plus the CONTRACTS the rest of the repo relies on this file for.
"""
import ast
import inspect
import os

import pytest

import run as run_stage                                          # bare name: see conftest.py


# =====================================================================================================================
# The lazy-cache contract this file implements
# =====================================================================================================================
def test_the_orchestrator_classifies_parameters_through_the_shared_helper():
    """Not its own copy. ``lazy.classify_params`` is what defines ``OUTPUT_PARAM_KEYS``, and a second implementation here
    would let the two disagree about which parameters are cache inputs."""
    source = inspect.getsource(run_stage)
    assert 'classify_params' in source


def test_the_cache_key_is_built_from_all_three_shared_primitives():
    source = inspect.getsource(run_stage)
    for primitive in ('code_state_hash', 'params_hash', 'compute_cache_key'):
        assert primitive in source, primitive


def test_the_stage_seed_is_derived_and_EXPORTED_as_pipeline_seed():
    """The other half of the automatic-seeding mechanism: this file computes the seed and puts it in the subprocess
    environment, and ``src/stages/__init__.py`` reads it back (see ``init_test.py``)."""
    source = inspect.getsource(run_stage)
    assert 'stage_seed' in source
    assert 'PIPELINE_SEED' in source


def test_the_environment_substitution_uses_the_shared_parser():
    """``parse_config`` owns the ``{{$VAR}}`` substitution, including the unset-becomes-empty-string behaviour. A second
    implementation here would be a second set of rules."""
    assert 'parse_config' in inspect.getsource(run_stage)


# =====================================================================================================================
# _coerce_bool — YAML gives strings, fire wants bools
# =====================================================================================================================
@pytest.mark.parametrize('value,expected', [
    (True, True), (False, False),
    ('true', True), ('True', True), ('TRUE', True),
    ('false', False), ('False', False),
    (1, True), (0, False),
])
def test_bool_coercion_accepts_the_forms_a_yaml_config_produces(value, expected):
    assert run_stage._coerce_bool(value, default=not expected) is expected


@pytest.mark.parametrize('default', [True, False])
def test_only_NONE_falls_back_to_the_default(default):
    """An UNSET flag takes the default; an unrecognised one does not."""
    assert run_stage._coerce_bool(None, default=default) is default


@pytest.mark.parametrize('default', [True, False])
def test_an_unrecognised_value_becomes_FALSE_rather_than_the_default(default):
    """⚠️ Worth knowing: the fallback covers `None` only. Anything else is matched against a whitelist
    (`1/true/yes/on`), so a typo like `lazy: ture` resolves to FALSE regardless of the default — silently disabling the
    lazy cache and re-running every stage, with no warning.

    Documented rather than changed: it is template code, and `False` is the safe direction for every flag it governs
    (`lazy`, `determinism`, `log_artifacts`, `log_models` — losing a cache hit or an artifact, never corrupting one).
    """
    assert run_stage._coerce_bool('maybe', default=default) is False
    assert run_stage._coerce_bool('ture', default=default) is False


# =====================================================================================================================
# Metric logging
# =====================================================================================================================
class _StubClient:
    """Records `log_metric` calls instead of talking to a tracking server."""

    def __init__(self):
        self.calls = []

    def log_metric(self, run_id, key, value):
        self.calls.append((run_id, key, value))


def test_every_scalar_in_the_metrics_file_is_logged(tmp_path):
    """`metrics-path` is one of the three `OUTPUT_PARAM_KEYS`, and this is what consumes it — a stage's scalars reach
    MLflow by being written to that file, not by the stage calling MLflow itself."""
    import json

    metrics_file = tmp_path / 'metrics.json'
    metrics_file.write_text(json.dumps({'mae': 1.5, 'rmse': 2.0}))

    client = _StubClient()
    returned = run_stage._log_metrics_to_run(client, 'run-id', str(metrics_file))

    assert returned == {'mae': 1.5, 'rmse': 2.0}
    assert sorted((key, value) for _, key, value in client.calls) == [('mae', 1.5), ('rmse', 2.0)]
    assert {run_id for run_id, _, _ in client.calls} == {'run-id'}, 'logged against the explicit run, not an ambient one'


def test_a_missing_metrics_file_RAISES_so_the_caller_must_guard(tmp_path):
    """It does not tolerate an absent file, which means `execute_stage` is responsible for only calling it when the stage
    actually declared a `metrics-path`. Pinned so the responsibility split stays visible from this side."""
    with pytest.raises(FileNotFoundError):
        run_stage._log_metrics_to_run(_StubClient(), 'run-id', str(tmp_path / 'absent.json'))


# =====================================================================================================================
# The two execution backends
# =====================================================================================================================
def test_both_backends_exist_and_take_the_same_arguments():
    """``run_sequential`` and ``run_prefect`` are alternative drivers over one stage list, so a divergence in what they
    accept means a config that works under one silently behaves differently under the other."""
    sequential = set(inspect.signature(run_stage.run_sequential).parameters)
    prefect = set(inspect.signature(run_stage.run_prefect).parameters)
    assert sequential == prefect, f'only sequential: {sequential - prefect}; only prefect: {prefect - sequential}'


def test_the_entry_point_takes_a_config_file():
    assert list(inspect.signature(run_stage.run).parameters) == ['config_file']


def test_the_orchestrator_is_wrapped_with_fire(repo_root):
    source = open(os.path.join(repo_root, 'src/stages/run.py')).read()
    assert 'Fire(' in source


# =====================================================================================================================
# Blocked until Step 4
# =====================================================================================================================
@pytest.mark.skip(reason='needs an MLflow tracking server and the Step 4 stages; covered by Step 4\'s end-to-end gate')
def test_a_full_pipeline_run_end_to_end():
    """``execute_stage`` dispatches a subprocess and logs to MLflow, so exercising it needs a live tracking server and at
    least one real stage. ``src/stages/`` holds only the template's files today."""
