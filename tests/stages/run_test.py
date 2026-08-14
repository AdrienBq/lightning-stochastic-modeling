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
# execute_stage — the per-stage core (Block 5c)
#
# It dispatches a subprocess through ``mlflow.run`` and talks to a tracking client. Both are stubbed here: what the
# tests are after is the DECISION LOGIC around the dispatch — whether the stage runs at all, what reaches its CLI, what
# gets tagged — none of which needs a server, and all of which fails silently when it is wrong.
# =====================================================================================================================
class _StubRun:
    def __init__(self, run_id='stage-run'):
        self.run_id = run_id


class _StubCachedRun:
    """The shape ``lazy.find_cached_run`` returns: ``.data.tags`` and ``.info.run_id``."""

    def __init__(self, output_fingerprint):
        from src.utils.io import lazy as lazy_module
        self.data = type('data', (), {'tags': {lazy_module.TAG_OUTPUT_FINGERPRINT: output_fingerprint}})()
        self.info = type('info', (), {'run_id': 'cached-run'})()


class _RecordingClient:
    def __init__(self):
        self.tags = []
        self.metrics = []
        self.artifacts = []

    def set_tag(self, run_id, key, value):
        self.tags.append((run_id, key, value))

    def log_metric(self, run_id, key, value):
        self.metrics.append((run_id, key, value))

    def log_artifact(self, *args, **kwargs):
        self.artifacts.append((args, kwargs))


@pytest.fixture
def stage_harness(monkeypatch, tmp_path):
    """Drive ``execute_stage`` with a stubbed subprocess dispatch and tracking client.

    Returns ``(run_stage_fn, client, dispatched)`` where ``dispatched`` collects the kwargs of every ``mlflow.run`` call
    — so "the stage was skipped" is observable as an empty list rather than inferred from a log line.
    """
    client = _RecordingClient()
    dispatched = []

    def fake_mlflow_run(**kwargs):
        dispatched.append(kwargs)
        return _StubRun()

    monkeypatch.setattr(run_stage.mlflow, 'run', fake_mlflow_run)

    context = run_stage.PipelineContext(
        tracking_client=client, orchestrator_run_id='orchestrator', experiment_id='7',
        code_hash='code-hash', project_uri='src/stages', log_artifacts=False, log_models=False,
        lazy_default=False, determinism_default=False, file_max=10 ** 9, dir_max=10 ** 9,
    )

    def call(parameters, **context_overrides):
        for key, value in context_overrides.items():
            setattr(context, key, value)
        run_stage.execute_stage('tune', parameters, 0, 1, context)

    return call, client, dispatched


def test_the_lazy_flags_are_POPPED_so_they_never_reach_the_stages_fire_cli(stage_harness):
    """``lazy:`` and ``ensure-determinism:`` are orchestrator concerns written into the same YAML block as the stage's
    own parameters. Leaving them in means fire receives ``--lazy=true`` for a stage with no such argument, and the stage
    aborts on an unrecognised flag."""
    call, _, dispatched = stage_harness
    call({'mode': 'daily', 'lazy': True, 'ensure-determinism': False})

    forwarded = dispatched[0]['parameters']
    assert 'lazy' not in forwarded and 'ensure-determinism' not in forwarded
    assert forwarded == {'mode': 'daily'}


def test_the_callers_parameter_dict_is_not_mutated(stage_harness):
    """The backends iterate the parsed config and hand each stage's block straight in. Popping in place would strip the
    flags from the config itself, so a re-read (or a second backend) would see different settings."""
    call, _, _ = stage_harness
    parameters = {'mode': 'daily', 'lazy': True}
    call(parameters)
    assert parameters == {'mode': 'daily', 'lazy': True}


def test_the_pipeline_seed_is_exported_and_matches_the_shared_derivation(stage_harness):
    """The orchestrator half of the automatic-seeding contract. The value must be exactly what ``lazy.stage_seed``
    computes from the code hash and the POST-POP parameters — a seed derived from the pre-pop dict would change whenever
    someone toggled ``lazy:``, silently making a "cached" re-run irreproducible."""
    from src.utils.io import lazy as lazy_module

    call, _, _ = stage_harness
    call({'mode': 'daily', 'lazy': False})

    expected = lazy_module.stage_seed('code-hash', lazy_module.params_hash({'mode': 'daily'}))
    assert os.environ['PIPELINE_SEED'] == str(expected)


def test_the_setup_stage_is_NEVER_cached(monkeypatch, stage_harness):
    """It creates the output directories every other stage writes into, and it is idempotent and cheap. Caching it means
    a pipeline run in a fresh checkout skips the one stage that would have made the directories exist."""
    from src.utils.io import lazy as lazy_module

    call, _, dispatched = stage_harness
    lookups = []
    monkeypatch.setattr(lazy_module, 'find_cached_run',
                        lambda *args, **kwargs: lookups.append(args) or None)

    context = run_stage.PipelineContext(
        tracking_client=_RecordingClient(), orchestrator_run_id='o', experiment_id='7', code_hash='c',
        project_uri='src/stages', log_artifacts=False, log_models=False, lazy_default=True,
        determinism_default=True, file_max=10 ** 9, dir_max=10 ** 9,
    )
    run_stage.execute_stage('setup', {'output-path': 'outputs'}, 0, 1, context)

    assert not lookups, 'setup must not even be looked up in the cache'
    assert dispatched, 'setup must always execute'


def test_a_cache_HIT_skips_the_subprocess_entirely(monkeypatch, stage_harness, tmp_path):
    """The point of the whole mechanism. The recorded output fingerprint has to match what is on disk right now —
    otherwise the cached run's outputs were edited or overwritten and reusing them would be wrong."""
    from src.utils.io import lazy as lazy_module

    outputs = tmp_path / 'prepared'
    outputs.mkdir()
    (outputs / 'target_stats.json').write_text('{}')
    parameters = {'output-path': str(outputs), 'mode': 'daily'}

    current = lazy_module.fingerprint_paths({'output-path': str(outputs)}, '/', 10 ** 9, 10 ** 9, 'output')
    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: _StubCachedRun(current))

    call, client, dispatched = stage_harness
    call(parameters, lazy_default=True)

    assert not dispatched, 'a cache hit must not dispatch the stage'
    assert any(key.startswith('lazy_skipped_') for _, key, _ in client.tags)


def test_a_cache_hit_whose_OUTPUTS_ARE_GONE_re_runs(monkeypatch, stage_harness, tmp_path):
    """``paths_present`` is the second half of a hit. A run recorded before ``outputs/`` was cleaned is a hit on key and
    a miss in reality, and skipping it would leave the next stage with no input."""
    from src.utils.io import lazy as lazy_module

    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: _StubCachedRun('whatever'))
    call, _, dispatched = stage_harness
    call({'output-path': str(tmp_path / 'never_created')}, lazy_default=True)
    assert dispatched, 'a hit whose outputs are absent must re-run'


def test_a_cache_hit_whose_OUTPUTS_CHANGED_re_runs_with_a_warning(monkeypatch, stage_harness, tmp_path, caplog):
    """Same key, different bytes on disk: someone edited the output by hand, or a previous run was interrupted mid-write.
    Reusing it would propagate whatever is there now under the authority of a recorded run."""
    import logging

    from src.utils.io import lazy as lazy_module

    outputs = tmp_path / 'prepared'
    outputs.mkdir()
    (outputs / 'target_stats.json').write_text('{}')
    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: _StubCachedRun('a-stale-fingerprint'))

    call, _, dispatched = stage_harness
    with caplog.at_level(logging.WARNING):
        call({'output-path': str(outputs)}, lazy_default=True)

    assert dispatched
    assert any('outputs differ' in record.getMessage() for record in caplog.records)


def test_the_determinism_guard_RAISES_when_it_is_switched_on(monkeypatch, stage_harness, tmp_path):
    """Identical code, parameters and inputs producing a different output is the failure ``ensure_determinism`` exists to
    surface. With the flag on it must abort the pipeline rather than record the divergence and continue."""
    from src.utils.io import lazy as lazy_module

    outputs = tmp_path / 'prepared'
    outputs.mkdir()
    (outputs / 'metrics.json').write_text('{"mae": 1.0}')
    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: _StubCachedRun('a-different-fingerprint'))

    call, _, _ = stage_harness
    with pytest.raises(RuntimeError, match='Non-deterministic stage'):
        call({'output-path': str(outputs)}, lazy_default=False, determinism_default=True)


def test_the_determinism_guard_only_WARNS_when_it_is_off(monkeypatch, stage_harness, tmp_path, caplog):
    """The default. Non-determinism is reported but does not stop a sweep — the flag is what promotes it to an error."""
    import logging

    from src.utils.io import lazy as lazy_module

    outputs = tmp_path / 'prepared'
    outputs.mkdir()
    (outputs / 'metrics.json').write_text('{"mae": 1.0}')
    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: _StubCachedRun('a-different-fingerprint'))

    call, _, _ = stage_harness
    with caplog.at_level(logging.WARNING):
        call({'output-path': str(outputs)}, lazy_default=True, determinism_default=False)
    assert any('Non-deterministic stage' in record.getMessage() for record in caplog.records)


def test_an_executed_stage_records_all_five_cache_tags(monkeypatch, stage_harness, tmp_path):
    """These tags ARE the cache: ``find_cached_run`` queries them back. A stage that executes without writing them can
    never be a future hit, so the pipeline silently loses lazy execution while appearing to work."""
    from src.utils.io import lazy as lazy_module

    outputs = tmp_path / 'prepared'
    outputs.mkdir()
    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: None)

    call, client, _ = stage_harness
    call({'output-path': str(outputs)}, lazy_default=True)

    written = {key for run_id, key, _ in client.tags if run_id == 'stage-run'}
    assert written == {lazy_module.TAG_STAGE, lazy_module.TAG_CACHE_KEY, lazy_module.TAG_CODE_STATE,
                       lazy_module.TAG_PARAMS_HASH, lazy_module.TAG_OUTPUT_FINGERPRINT}


def test_no_cache_bookkeeping_happens_when_both_flags_are_off(stage_harness, tmp_path):
    """Fingerprinting the prepared directory is not free — it is ~50 GB of samples. With lazy and determinism both off
    the stage must dispatch without paying for it."""
    outputs = tmp_path / 'prepared'
    outputs.mkdir()

    call, client, dispatched = stage_harness
    call({'output-path': str(outputs)}, lazy_default=False, determinism_default=False)

    assert dispatched
    assert not client.tags, 'no cache tags should be written when the cache is disabled'


def test_a_stages_metrics_reach_BOTH_its_own_run_and_the_orchestrator(stage_harness, tmp_path):
    """The convention every pipeline relies on for the comparison table: a stage writes ``metrics-path`` and the
    orchestrator aggregates. Logging to only the child run leaves the parent empty."""
    import json

    metrics_file = tmp_path / 'metrics.json'
    metrics_file.write_text(json.dumps({'mae': 1.5}))

    call, client, _ = stage_harness
    call({'metrics-path': str(metrics_file)})

    logged = {(run_id, key) for run_id, key, _ in client.metrics}
    assert ('stage-run', 'mae') in logged
    assert ('orchestrator', 'mae') in logged


def test_a_SKIPPED_stages_metrics_are_still_aggregated(monkeypatch, stage_harness, tmp_path):
    """Otherwise the comparison table gains a hole exactly when the cache is working. The skip path re-reads the metrics
    file the cached run left on disk."""
    import json

    from src.utils.io import lazy as lazy_module

    metrics_file = tmp_path / 'metrics.json'
    metrics_file.write_text(json.dumps({'mae': 1.5}))
    current = lazy_module.fingerprint_paths({'metrics-path': str(metrics_file)}, '/', 10 ** 9, 10 ** 9, 'output')
    monkeypatch.setattr(lazy_module, 'find_cached_run', lambda *a, **k: _StubCachedRun(current))

    call, client, dispatched = stage_harness
    call({'metrics-path': str(metrics_file)}, lazy_default=True)

    assert not dispatched, 'this test is only meaningful on the skip path'
    assert ('orchestrator', 'mae') in {(run_id, key) for run_id, key, _ in client.metrics}


def test_the_stage_is_dispatched_as_a_script_under_the_project_uri(stage_harness):
    """Stages are standalone CLI scripts, not importable modules — the entry point is a path ending in ``.py``."""
    call, _, dispatched = stage_harness
    call({'mode': 'daily'})
    assert dispatched[0]['entry_point'] == 'src/stages/tune.py'
    assert dispatched[0]['env_manager'] == 'local', 'the venv is already active; letting mlflow build one would not use it'


# =====================================================================================================================
# _make_context
# =====================================================================================================================
def test_the_context_takes_its_run_and_experiment_ids_from_the_OPEN_orchestrator_run():
    """The core addresses MLflow by explicit ``run_id`` rather than through the fluent active-run global, so this is
    where the two ids are captured — the reason ``execute_stage`` stays correct off the main thread."""
    orchestrator = type('run', (), {'info': type('info', (), {'run_id': 'r-1', 'experiment_id': 'e-9'})()})()

    context = run_stage._make_context(orchestrator, 'client', 'code', 'src/stages', True, False,
                                      True, False, 111, 222)

    assert isinstance(context, run_stage.PipelineContext)
    assert context.orchestrator_run_id == 'r-1'
    assert context.experiment_id == 'e-9'
    assert (context.log_artifacts, context.log_models) == (True, False)
    assert (context.lazy_default, context.determinism_default) == (True, False)
    assert (context.file_max, context.dir_max) == (111, 222)


def test_both_backends_build_their_context_through_the_same_helper():
    """``run_sequential`` and ``run_prefect`` must hand ``execute_stage`` identical state; a second construction site is
    how the two backends would start to differ."""
    source = inspect.getsource(run_stage)
    assert source.count('_make_context(') == 3, 'one definition plus exactly one call per backend'


# =====================================================================================================================
# Blocked until Step 4
# =====================================================================================================================
@pytest.mark.skip(reason='needs an MLflow tracking server and the Step 4 stages; covered by Step 4\'s end-to-end gate')
def test_a_full_pipeline_run_end_to_end():
    """The tests above stub ``mlflow.run`` and the tracking client, which covers the decision logic but not the
    subprocess dispatch or the real store round trip. Those need a live server and at least one real stage;
    ``src/stages/`` holds only the template's files today."""
