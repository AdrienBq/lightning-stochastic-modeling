"""Orchestrator for the analysis pipeline. All the workflow is included in a single Python program that looks at the
results of each step and decides what to submit next using custom code.

The per-stage logic lives in a single shared core, ``execute_stage`` (seeding, the lazy cache lookup/skip, the stage
subprocess dispatch and all MLflow logging/bookkeeping). Two interchangeable *orchestration backends* drive it,
selected with the optional ``orchestrator`` key in the pipeline YAML (default ``mlflow``):

* ``mlflow``  -- the historical sequential ``for`` loop (``run_sequential``); no extra dependencies.
* ``prefect`` -- the same execution wrapped in a Prefect ``@flow``/``@task`` (``run_prefect``), which adds an
  execution UI, per-stage observability and (later) remote/cluster submission. Requires ``pip install prefect``.

Both backends call ``execute_stage`` with identical arguments, so switching backends does not change results: the
lazy cache and determinism guard (src/utils/io/lazy.py, state stored as MLflow run tags) are reused unchanged --
Prefect's own task caching is deliberately NOT used (it hashes Python call arguments, not the code-state + input-file
fingerprints this template relies on). See the README section "Orchestration Backends" for the full rationale.

Lazy execution (opt-in via the ``lazy`` flag, overridable per stage): a stage is skipped when its code state,
parameters and inputs match a previously recorded successful run whose outputs are still present and byte-identical
on disk. When a stage *is* executed with unchanged code+inputs+params but yields a different output, that
non-determinism is warned about (or raised, if ``ensure_determinism`` is on). Every stage is additionally seeded
deterministically (see src/utils/seeding.py).
"""
import os
import json
import sys
import logging
from dataclasses import dataclass

import mlflow

from fire import Fire


from __init__ import root_path, console_handler
from src.utils.io.parse_config import parse_config
from src.utils.io import lazy
from src.utils.banner import make_banner

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
# surface the lazy module's warnings (skipped byte-exact checks, degraded code-state) on the console too
lazy.logger.addHandler(console_handler)


def _coerce_bool(value, default: bool) -> bool:
    """Interpret a YAML scalar (or None) as a boolean, falling back to ``default`` when unset."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _log_metrics_to_run(tracking_client, run_id: str, metrics_path: str) -> dict:
    """Read a stage's flat metrics JSON and log each scalar to the given run; return the metrics dict."""
    with open(metrics_path) as handle:
        metrics = json.load(handle)
    for key, value in metrics.items():
        tracking_client.log_metric(run_id=run_id, key=key, value=value)
    return metrics


@dataclass
class PipelineContext:
    """Shared, immutable-per-run state threaded into every ``execute_stage`` call.

    Bundling it keeps the per-stage core backend-agnostic: the MLflow loop and the Prefect task build the same
    context once and pass it down. Logging targets the orchestrator run by explicit ``run_id`` (via the client)
    rather than the fluent active-run global, so the core does not depend on an ambient ``mlflow.start_run`` and
    stays correct if stages are ever executed off the main thread.
    """
    tracking_client: object              # mlflow.tracking.MlflowClient
    orchestrator_run_id: str             # the parent ("orchestrator") MLflow run
    experiment_id: str
    code_hash: str                       # whole-repo code state, computed once per pipeline run
    project_uri: str
    log_artifacts: bool
    log_models: bool
    lazy_default: bool
    determinism_default: bool
    file_max: int
    dir_max: int


def execute_stage(stage_name: str, parameters: dict, index: int, total: int, ctx: PipelineContext) -> None:
    """Run (or lazily skip) a single pipeline stage -- the per-stage core shared by both backends.

    Args:
        stage_name: Name of the stage script under ``project_uri`` (without ``.py``).
        parameters: The stage's YAML parameter block (the ``lazy``/``ensure-determinism`` overrides are popped
            here and never reach the stage's Fire CLI).
        index: Zero-based position of the stage in the pipeline (for the progress header).
        total: Total number of stages (for the progress header).
        ctx: The shared :class:`PipelineContext`.
    """
    client = ctx.tracking_client
    parameters = dict(parameters) if parameters else {}

    # per-stage overrides for lazy/determinism, popped so they never reach the stage's Fire CLI
    stage_lazy = _coerce_bool(parameters.pop('lazy', None), ctx.lazy_default)
    stage_determinism = _coerce_bool(parameters.pop('ensure-determinism', None), ctx.determinism_default)

    print(file=sys.stderr)
    print(f'==============> Stage {index + 1}/{total}: "{stage_name}" '.ljust(103, '='), file=sys.stderr)

    # deterministic per-stage seed (from code + parameters; independent of input bytes, so cheap and active even
    # with lazy off). The stage subprocess inherits PIPELINE_SEED and seeds on import.
    # TODO(concurrency): this mutates a *process-global* env var, which is fine for the current linear execution
    #   but is NOT safe once stages run in parallel (see TODO(concurrency) in run_prefect). To parallelise, pass
    #   the seed to the stage subprocess explicitly (e.g. via a per-call env passed to mlflow.run) rather than
    #   through os.environ.
    params_hash_value = lazy.params_hash(parameters)
    os.environ['PIPELINE_SEED'] = str(lazy.stage_seed(ctx.code_hash, params_hash_value))

    # the setup stage is a cheap idempotent prerequisite (it creates the output dirs the other stages depend on):
    # never cached, but still seeded above.
    cacheable = stage_name != 'setup'
    # the fingerprint bookkeeping is only needed when lazy skipping or determinism checking is wanted
    active = cacheable and (stage_lazy or stage_determinism)

    cache_key = None
    cached = None
    inputs, outputs = ({}, {})
    if active:
        inputs, outputs = lazy.classify_params(parameters, root_path)
        inputs_fp = lazy.fingerprint_paths(inputs, root_path, ctx.file_max, ctx.dir_max, 'input')
        cache_key = lazy.compute_cache_key(ctx.code_hash, params_hash_value, inputs_fp)
        cached = lazy.find_cached_run(client, ctx.experiment_id, stage_name, cache_key)

    # ---- lazy hit: skip execution when the recorded outputs are still present and unchanged ----
    if stage_lazy and cached is not None and lazy.paths_present(outputs, root_path):
        recorded = cached.data.tags.get(lazy.TAG_OUTPUT_FINGERPRINT)
        current_fp = lazy.fingerprint_paths(outputs, root_path, ctx.file_max, ctx.dir_max, 'output')
        if recorded is not None and recorded == current_fp:
            logger.info('lazy: skipping "%s" (cache hit %s from run %s)',
                        stage_name, cache_key[:12], cached.info.run_id)
            client.set_tag(run_id=ctx.orchestrator_run_id,
                           key=f'lazy_skipped_{stage_name}', value=cache_key[:12])
            # keep the parent run's metric aggregation consistent with a freshly executed stage
            if 'metrics-path' in parameters and os.path.exists(parameters['metrics-path']):
                _log_metrics_to_run(client, ctx.orchestrator_run_id, parameters['metrics-path'])
            return
        logger.warning('lazy: "%s" cache key matched but on-disk outputs differ from the recorded '
                       'fingerprint; re-running', stage_name)

    # run stage (parameter logging is automatic)
    # TODO(concurrency): mlflow.run nests the stage's child run under the *fluent active* run (the orchestrator
    #   run opened by the backend on the main thread). That is correct for linear execution; if stages are moved
    #   onto worker threads/processes, set the parent explicitly (e.g. propagate MLFLOW_RUN_ID) instead.
    current_run = mlflow.run(
        uri='',
        entry_point=f'{ctx.project_uri}/{stage_name}.py',
        parameters=parameters,
        env_manager='local'
    )

    # log configurations as run artifacts (on the orchestrator run, addressed by explicit run_id so this does not
    # depend on the fluent active-run global)
    if 'config-path' in parameters:
        client.log_artifact(ctx.orchestrator_run_id, parameters['config-path'])

    if 'metrics-config' in parameters:
        client.log_artifact(ctx.orchestrator_run_id, parameters['metrics-config'])

    if 'model-config' in parameters:
        client.log_artifact(ctx.orchestrator_run_id, parameters['model-config'])

    # artifact logging
    if ctx.log_artifacts and 'output-path' in parameters:
        client.log_artifact(run_id=current_run.run_id, local_path=parameters['output-path'])

    # tag the run with the model that was used for ease of access
    if 'model-type' in parameters:
        client.set_tag(run_id=current_run.run_id, key='model-type', value=parameters['model-type'])
        client.set_tag(run_id=ctx.orchestrator_run_id, key='model-type', value=parameters['model-type'])

    # metrics logging (metrics are logged in every run)
    if 'metrics-path' in parameters:
        metrics = _log_metrics_to_run(client, current_run.run_id, parameters['metrics-path'])
        # also logs metrics in the orchestrator run for ease of access
        for key, value in metrics.items():
            client.log_metric(run_id=ctx.orchestrator_run_id, key=key, value=value)

    # report logging
    if 'report-path' in parameters:
        client.log_artifact(run_id=current_run.run_id, local_path=parameters['report-path'])

    # model logging (just saves it as artifact)
    # TODO: figure out how to store via log_model (temporarily stores them as artifact)
    if ctx.log_models:
        # TODO: find a better convention for the parameters
        # logging the vectorizer
        if stage_name == 'preprocess' and 'vectorizer-path' in parameters:
            client.log_artifact(run_id=current_run.run_id, local_path=parameters['vectorizer-path'])
        # logging the model
        if stage_name == 'train_evaluate' and 'output-path' in parameters:
            client.log_artifact(run_id=current_run.run_id, local_path=parameters['output-path'])

    # ---- lazy bookkeeping + determinism guard (record this run's fingerprints) ----
    if active:
        output_fp = lazy.fingerprint_paths(outputs, root_path, ctx.file_max, ctx.dir_max, 'output')
        if cached is not None:
            recorded = cached.data.tags.get(lazy.TAG_OUTPUT_FINGERPRINT)
            if recorded is not None and recorded != output_fp:
                message = (
                    f'Non-deterministic stage "{stage_name}": identical code, parameters and inputs '
                    f'produced a different output (recorded {recorded[:16]}..., now {output_fp[:16]}...).'
                )
                if stage_determinism:
                    raise RuntimeError(message)
                logger.warning(message)
        for key, value in (
            (lazy.TAG_STAGE, stage_name),
            (lazy.TAG_CACHE_KEY, cache_key),
            (lazy.TAG_CODE_STATE, ctx.code_hash),
            (lazy.TAG_PARAMS_HASH, params_hash_value),
            (lazy.TAG_OUTPUT_FINGERPRINT, output_fp),
        ):
            client.set_tag(run_id=current_run.run_id, key=key, value=value)


def _make_context(orchestrator_run, tracking_client, code_hash, project_uri, log_artifacts, log_models,
                  lazy_default, determinism_default, file_max, dir_max) -> PipelineContext:
    """Build the shared :class:`PipelineContext` from an open orchestrator run and the resolved settings."""
    return PipelineContext(
        tracking_client=tracking_client,
        orchestrator_run_id=orchestrator_run.info.run_id,
        experiment_id=orchestrator_run.info.experiment_id,
        code_hash=code_hash,
        project_uri=project_uri,
        log_artifacts=log_artifacts,
        log_models=log_models,
        lazy_default=lazy_default,
        determinism_default=determinism_default,
        file_max=file_max,
        dir_max=dir_max,
    )


# --------------------------------------------------------------------------------------------------------------
# orchestration backends
# --------------------------------------------------------------------------------------------------------------
def run_sequential(config, config_file, tags, project_uri, log_artifacts, log_models,
                   lazy_default, determinism_default, file_max, dir_max):
    """Default backend: open one orchestrator run and execute the stages strictly in order."""
    tracking_client = mlflow.tracking.MlflowClient()
    code_hash = lazy.code_state_hash(root_path)     # whole-repo code state, computed once per pipeline run

    # start the main run and all secondary runs
    with mlflow.start_run() as orchestrator:
        # logging the configuration and tags for the run
        mlflow.log_artifact(config_file)
        mlflow.set_tags(tags)

        ctx = _make_context(orchestrator, tracking_client, code_hash, project_uri, log_artifacts, log_models,
                            lazy_default, determinism_default, file_max, dir_max)

        # step by step execution
        total_steps = len(config['stages'])
        for i, stage in enumerate(config['stages']):
            for stage_name, parameters in stage.items():
                execute_stage(stage_name, parameters, i, total_steps, ctx)


def run_prefect(config, config_file, tags, project_uri, log_artifacts, log_models,
                lazy_default, determinism_default, file_max, dir_max):
    """Opt-in backend: the same execution wrapped in a Prefect flow for an execution UI and per-stage runs.

    Functionally identical to ``run_sequential`` -- each stage still calls ``execute_stage``, so the lazy cache,
    determinism guard, seeding and MLflow logging are byte-for-byte the same. Prefect only adds the orchestration
    layer (a flow run + one task run per stage, visible at ``prefect server start`` -> localhost:4200) and the
    cross-link to the MLflow run.

    Prefect is an *optional* dependency: if it is not installed this transparently falls back to
    ``run_sequential`` (logging a warning) so selecting ``orchestrator: prefect`` never breaks a pipeline.
    """
    try:
        from prefect import flow, task, get_run_logger
    except ImportError:  # Prefect is an optional dependency: degrade, never fail
        logger.warning(
            'orchestrator: prefect requested but Prefect is not installed; falling back to the mlflow backend '
            '(install it with `pip install prefect`, or see environment.yml, to use the Prefect frontend).'
        )
        return run_sequential(config, config_file, tags, project_uri, log_artifacts, log_models,
                              lazy_default, determinism_default, file_max, dir_max)

    tracking_client = mlflow.tracking.MlflowClient()
    code_hash = lazy.code_state_hash(root_path)
    stages = config['stages']
    total_steps = len(stages)

    @flow(name=os.path.basename(root_path), description=str(config.get('description', '')) or None)
    def _pipeline_flow():
        flow_logger = get_run_logger()
        flow_logger.info('Starting pipeline flow (orchestrator=prefect)')

        # the orchestrator run stays on the main thread so mlflow.run still nests stage runs under it (see the
        # TODO(concurrency) in execute_stage)
        with mlflow.start_run() as orchestrator:
            mlflow.log_artifact(config_file)
            mlflow.set_tags(tags)

            ctx = _make_context(orchestrator, tracking_client, code_hash, project_uri, log_artifacts,
                                log_models, lazy_default, determinism_default, file_max, dir_max)

            # cross-link the two stores by id: tag the MLflow run with the Prefect flow-run id, and surface the
            # MLflow run id in the Prefect logs (the two UIs hold complementary views -- execution vs experiment).
            flow_logger.info('MLflow run_id: %s', ctx.orchestrator_run_id)
            try:
                from prefect.runtime import flow_run as _prefect_flow_run
                prefect_flow_run_id = str(_prefect_flow_run.id)
                tracking_client.set_tag(ctx.orchestrator_run_id, 'prefect_flow_run_id', prefect_flow_run_id)
                flow_logger.info('Prefect flow_run_id: %s', prefect_flow_run_id)
            except Exception as link_error:  # never let a cosmetic cross-link abort the pipeline
                flow_logger.warning('could not cross-link Prefect flow-run id to MLflow (%s)', link_error)

            # one Prefect task per stage; defined here so it closes over `ctx` (avoids passing the non-picklable
            # MlflowClient as a task argument) and renamed per stage for readable task runs in the UI.
            @task
            def _stage_task(stage_name, parameters, index, total):
                execute_stage(stage_name, parameters, index, total, ctx)

            # ----------------------------------------------------------------------------------------------
            # TODO(concurrency): stages run strictly in order, exactly like run_sequential. To let independent
            #   stages run in parallel (the main reason to reach for Prefect here):
            #     1. Give the flow a task runner, e.g. `@flow(task_runner=ThreadPoolTaskRunner())` (or a
            #        Dask/Ray runner for processes/cluster -- see the README "Concurrency and HPC" section).
            #     2. Submit stages with `_stage_task.submit(...)` and thread the returned futures so Prefect can
            #        infer the DAG: `fut_b = _stage_task.submit(..., wait_for=[fut_a])`.
            #     3. Build the dependency edges. Because stages talk through files (not return values), the edges
            #        must be made explicit, two options:
            #          (a) DECLARED -- add an optional `depends_on: [stage names]` key to each stage's YAML and
            #              pop it here (like `lazy:`), mapping names -> futures to set `wait_for`.
            #          (b) INFERRED -- add an edge wherever one stage's input path equals another stage's
            #              `output-path` (reuse lazy.classify_params to get each stage's inputs/outputs).
            #   Safety: only parallelise stages with no input/output overlap; MLflow logging is already via the
            #   MlflowClient with explicit run_ids (thread-safe), but resolve the os.environ seed and mlflow.run
            #   parent-nesting TODOs in execute_stage first.
            # ----------------------------------------------------------------------------------------------
            for i, stage in enumerate(stages):
                for stage_name, parameters in stage.items():
                    _stage_task.with_options(name=stage_name)(
                        stage_name, dict(parameters) if parameters else {}, i, total_steps
                    )

        flow_logger.info('Pipeline flow complete')

    _pipeline_flow()


# --------------------------------------------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------------------------------------------
def run(config_file: str):
    # read pipeline config files
    # WARNING: experimental
    # TODO: find a way to keep track of parameters specified through environment variables
    config = parse_config(os.path.join(root_path, config_file))

    # run parameters
    project_uri = config['project_uri']
    log_artifacts = config['log_artifacts']
    log_models = config['log_models']
    tags = config['tags']

    # welcome banner: repo name (from the project directory) + version (from the pipeline tags) + optional
    # description; the name and subtitle are length-capped so the banner never overextends horizontally.
    # Disable it entirely with `banner: false` at the top of the pipeline YAML; pick a different FIGlet font
    # with `banner_font: <name>` (fonts other than the bundled `small` require pyfiglet to be installed).
    if _coerce_bool(config.get('banner'), True):
        print(make_banner(
            name=os.path.basename(root_path),
            version=str(tags.get('version', '')) if isinstance(tags, dict) else '',
            tagline=str(config.get('description', '')),
            font=str(config.get('banner_font', 'small')),
        ), file=sys.stderr)

    # lazy-execution / determinism settings (both default to off => behaviour identical to before)
    lazy_default = _coerce_bool(config.get('lazy'), False)
    determinism_default = _coerce_bool(config.get('ensure_determinism'), False)
    file_max = int(config.get('lazy_content_max_file_mb', 256)) * 1024 ** 2
    dir_max = int(config.get('lazy_content_max_dir_mb', 2048)) * 1024 ** 2

    # orchestration backend selection (default 'mlflow' => behaviour identical to before)
    backend = str(config.get('orchestrator', 'mlflow')).strip().lower()
    backend_args = (config, config_file, tags, project_uri, log_artifacts, log_models,
                    lazy_default, determinism_default, file_max, dir_max)
    if backend == 'mlflow':
        run_sequential(*backend_args)
    elif backend == 'prefect':
        run_prefect(*backend_args)
    else:
        raise ValueError(f"Unknown orchestrator '{backend}' in pipeline config (expected 'mlflow' or 'prefect')")


if __name__ == '__main__':
    Fire(run)
