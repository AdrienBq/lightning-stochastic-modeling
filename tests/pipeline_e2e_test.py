"""End-to-end pipeline test: a REAL ``run_project.py`` invocation on a synthetic ``$DATA_ROOT``.

This is the only test that exercises the whole stack in the shape a user runs it — ``run_project.py`` ->
``mlflow.projects.run`` -> the orchestrator in ``src/stages/run.py`` -> one subprocess per stage -> the library. It
mirrors no single module, which is why it sits at the ``tests/`` root beside ``completeness_test.py`` (both are in
that file's expected set, or the mirror check would flag them as orphans).

**What makes it a test rather than a smoke run.** A smoke run has exactly the right coverage and none of the
properties of a test: its assertions are a human reading a log, it needs the real dataset, and it lives in a shell
command in a plan file. Here the artifacts, the metric keys, the figure set and the tracking-store bookkeeping are all
**asserted in code**, on data any checkout can build.

**The config is DERIVED from the shipped smoke YAML, not copied.** Exactly ONE parameter differs from
``config/deterministic_unet/deterministic_unet_daily_smoke_cpu.yaml`` — the ``split-config`` path, which has to name a split
over the synthetic sample ids instead of the real dataset's ids 194-927. Every other stage parameter is the shipped
pipeline's, and ``test_the_derived_config_changes_exactly_ONE_PARAMETER_of_the_shipped_one`` pins that. A fixture copy
of the config could pass forever while the real pipeline was broken.

**What it deliberately does NOT prove** — the accepted cost of a synthetic root, and why the by-hand gates still
exist: the real 101 x 149 grid, the real ``samples/*.pt`` contents, the real year split, that 8.7 MB x 5843 is
tractable, and anything at all about GPU execution or about model quality.

⚠️ **It costs ~70 s** (five python+torch subprocesses, two one-epoch fits and a report). The pipeline run is a
SESSION-scoped fixture, so that is paid once no matter how many assertions read it.

⚠️ **Coverage does not move.** Every stage runs in a subprocess, which ``pytest-cov`` does not see into without the
``COV_CORE_*`` hook. ``tuning.py`` and ``stages/run.py`` gain confidence here without gaining measured lines — do not
read this test as a reason to raise ``--cov-fail-under``.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.conftest import build_dataset_root, write_split_config
from tests.utils.metrics.evaluation_test import EXPECTED_DAILY_KEYS

FAMILY = 'deterministic_unet'
SHIPPED_CONFIG = f'config/{FAMILY}/{FAMILY}_daily_smoke_cpu.yaml'
SHIPPED_SPLIT = 'config/split/split_smoke_cpu.yaml'
COMPARISON_CONFIG = 'config/eval/probabilistic_eval_smoke_cpu.yaml'

# 12 days -> 8 train / 2 valid / 2 test. Eight train days is what lets the shipped `feature-stats-days: 4` stand
# unchanged (so the derived config keeps its one-line diff) and gives the day-of-year climatology something to average.
N_DAYS = 12
TRAIN_FRACTION = 2 / 3

# ---------------------------------------------------------------------------------------------------------------------
# The keys of EXPECTED_DAILY_KEYS a real DAILY pipeline does not emit. Two different reasons, and keeping them apart is
# the point: the first is structural and permanent, the second is an artifact of a 2-day test split.
#
# `evaluation_test.py`'s `daily_suite` fixture calls `run_metric_suite` with a NON-None `probability`, which no daily
# pipeline produces — `deterministic_module` returns `probability=None` in daily mode because there is no occurrence
# head. That fixture is right for a unit test of the suite's full surface and wrong as a description of a daily run,
# which is why the difference is subtracted here rather than either list being "fixed".
# ---------------------------------------------------------------------------------------------------------------------
ABSENT_STRUCTURALLY = {
    'explained_deviance': 'probability-gated; daily mode has no occurrence head, so probability is None',
}
ABSENT_ON_A_TINY_SPLIT = {
    'fss_useful_scale_occurrence': 'no neighbourhood scale clears 0.5 + base_rate/2 on a 2-day untrained model',
    'mae_bin_occurrence_h3': 'the occurrence-to-h3 intensity bin is empty; the fixture target lands in the h3-h6 band',
}

# The figures config/eval/metrics_daily.yaml lists that a deterministic DAILY run cannot draw, and self-skips instead.
SELF_SKIPPED_FIGURES = {
    'reliability': 'no probability field, so no calibration curve (daily mode)',
    'rank_histogram': 'no ensemble members: the deterministic family emits none',
    'residual_bias_map': 'not a residual run', 'residual_surprise': 'not a residual run',
    'residual_histograms': 'not a residual run', 'residual_qq': 'not a residual run',
    'residual_scatters': 'not a residual run', 'residual_heteroscedasticity': 'not a residual run',
}


def _stage_blocks(config):
    """``[(stage name, parameters)]`` in pipeline order — the YAML is a list of single-key dicts."""
    return [(name, parameters) for stage in config['stages'] for name, parameters in stage.items()]


@pytest.fixture(scope='session')
def pipeline_run(tmp_path_factory, repo_root):
    """Run the shipped CPU smoke pipeline once, on a synthetic dataset, and return everything the tests read.

    Session-scoped: ~70 s, paid once. Every assertion below is a separate test over this one result, so a failure
    names the property that broke rather than "the pipeline failed".
    """
    work = tmp_path_factory.mktemp('pipeline_e2e')
    data_root = build_dataset_root(str(work / 'data'), n_days=N_DAYS, vary_activity=True)
    split_path = write_split_config(str(work / 'split.yaml'), n_days=N_DAYS, train_fraction=TRAIN_FRACTION)

    with open(os.path.join(repo_root, SHIPPED_CONFIG)) as handle:
        shipped = handle.read()
    derived_text = shipped.replace(SHIPPED_SPLIT, split_path)
    config_path = str(work / 'pipeline.yaml')
    with open(config_path, 'w') as handle:
        handle.write(derived_text)

    output_root = str(work / 'outputs')
    tracking_uri = f'file:{work / "mlruns"}'
    environment = dict(os.environ, DATA_ROOT=data_root, OUTPUT_ROOT=output_root,
                       MLFLOW_TRACKING_URI=tracking_uri)
    # ⚠️ mlflow.projects' local backend shells out to a BARE `python`, not sys.executable. Under `pytest` the venv is
    # usually not activated, so without this the stage subprocess gets the system interpreter and dies on
    # `import mlflow` — a failure that looks like a broken stage and is not one. (Recorded as a portability wart in
    # .claude/plans/step-5-portability.md: a real run depends on `python` resolving correctly too.)
    environment['PATH'] = os.path.dirname(sys.executable) + os.pathsep + environment.get('PATH', '')

    completed = subprocess.run([sys.executable, 'run_project.py', config_path, 'pipeline_e2e'],
                               cwd=repo_root, env=environment, capture_output=True, text=True)

    from src.utils.io.parse_config import parse_config

    previous = os.environ.get('OUTPUT_ROOT'), os.environ.get('DATA_ROOT')
    os.environ['OUTPUT_ROOT'], os.environ['DATA_ROOT'] = output_root, data_root
    try:
        config = parse_config(config_path)                      # the SAME substitution the orchestrator performed
    finally:
        for key, value in zip(('OUTPUT_ROOT', 'DATA_ROOT'), previous):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return {
        'completed': completed, 'config': config, 'config_path': config_path, 'shipped_text': shipped,
        'derived_text': derived_text, 'output_root': output_root, 'tracking_uri': tracking_uri,
        'data_root': data_root, 'work': str(work), 'stages': _stage_blocks(config),
    }


@pytest.fixture(scope='session')
def evaluation_metrics(pipeline_run):
    """The flat metrics JSON the ``evaluate`` stage wrote."""
    parameters = dict(pipeline_run['stages'])['evaluate']
    with open(parameters['metrics-path']) as handle:
        return json.load(handle)


# =====================================================================================================================
# 1. The run itself
# =====================================================================================================================
def test_the_pipeline_EXITS_ZERO(pipeline_run):
    """The whole point, and the assertion that must come first: every other test here would be misleading if the run
    had failed halfway."""
    completed = pipeline_run['completed']
    if completed.returncode != 0:
        tail = '\n'.join((completed.stdout + completed.stderr).splitlines()[-30:])
        pytest.fail(f'run_project.py exited {completed.returncode}:\n{tail}')


def test_the_derived_config_changes_exactly_ONE_PARAMETER_of_the_shipped_one(pipeline_run):
    """What "derived, not copied" means, made checkable: exactly one PARAMETER differs from the shipped pipeline — the
    split path, which has to name the synthetic sample ids instead of the real dataset's 194-927. Every other stage
    parameter is the shipped one.

    Comment lines are excluded from the count rather than from the rewrite: the smoke config's header documents its
    split path, and a derived config whose comment still pointed at `split_smoke_cpu.yaml` would misdescribe itself to
    anyone reading the file while debugging this test. So the comments move with the setting and the claim is about
    settings.

    If this ever fails because a SECOND parameter needed rewriting, that is the thing to know: every extra rewrite is
    one more way this test can keep passing while the real pipeline is broken.
    """
    shipped = pipeline_run['shipped_text'].splitlines()
    derived = pipeline_run['derived_text'].splitlines()
    assert len(shipped) == len(derived), 'the rewrite must not add or remove lines'

    differing = [(before, after) for before, after in zip(shipped, derived) if before != after]
    assert differing, 'nothing was rewritten, so the pipeline ran against the REAL split ids'

    parameters = [pair for pair in differing if not pair[0].lstrip().startswith('#')]
    assert len(parameters) == 1, f'expected one parameter to change, got {parameters}'
    assert parameters[0][0].strip().startswith('split-config:')
    assert all('split-config' in before for before, _ in differing), \
        f'a line unrelated to the split was rewritten: {differing}'


def test_the_shipped_smoke_tier_has_LAZY_OFF(pipeline_run):
    """A cached stage is a stage this test did not execute. The smoke tiers set `lazy: false` for exactly this reason,
    and the derived config inherits it — so a second run in one session re-runs everything."""
    assert pipeline_run['config']['lazy'] is False


def test_EVERY_stage_the_config_declares_actually_RAN(pipeline_run):
    """Five stages, in order, each as its own child run in the tracking store. Counting the runs rather than grepping
    the log is what makes this an assertion about the ORCHESTRATOR's bookkeeping rather than about its printing."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(pipeline_run['tracking_uri'])
    client = MlflowClient()
    experiment = client.get_experiment_by_name('pipeline_e2e')
    assert experiment is not None, 'the orchestrator created no experiment'

    runs = client.search_runs([experiment.experiment_id])
    children = [run for run in runs if run.data.tags.get('mlflow.parentRunId')]
    entry_points = {run.data.tags.get('mlflow.project.entryPoint') for run in children}

    declared = [name for name, _ in pipeline_run['stages']]
    assert entry_points == {f'src/stages/{name}.py' for name in declared}
    assert len(children) == len(declared), f'{len(children)} child runs for {len(declared)} stages'
    assert all(run.info.status == 'FINISHED' for run in children), \
        [(run.data.tags.get('mlflow.project.entryPoint'), run.info.status) for run in children]


def test_the_metrics_JSON_is_LOGGED_to_the_stage_run(pipeline_run, evaluation_metrics):
    """`run.py` reads `metrics-path` and logs its contents to the child run — the mechanism that makes a stage's numbers
    queryable from MLflow rather than only from a file. Nothing else tests it, because it only happens under the
    orchestrator."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(pipeline_run['tracking_uri'])
    client = MlflowClient()
    experiment = client.get_experiment_by_name('pipeline_e2e')
    evaluate_runs = [run for run in client.search_runs([experiment.experiment_id])
                     if run.data.tags.get('mlflow.project.entryPoint') == 'src/stages/evaluate.py']

    assert len(evaluate_runs) == 1
    assert set(evaluate_runs[0].data.metrics) == set(evaluation_metrics), \
        'the logged metrics and the JSON disagree'


# =====================================================================================================================
# 2. The artifacts every stage declared
# =====================================================================================================================
def test_EVERY_declared_output_EXISTS_and_is_NON_EMPTY(pipeline_run):
    """`output-path` / `metrics-path` / `report-path` are the three keys the lazy cache treats as a stage's OUTPUTS, so
    a declared one that never appears is both a broken stage and a cache that will happily record success."""
    empty, absent = [], []
    for name, parameters in pipeline_run['stages']:
        for key in ('output-path', 'metrics-path', 'report-path'):
            path = parameters.get(key)
            if path is None:
                continue
            if not os.path.exists(path):
                absent.append(f'{name}.{key} -> {path}')
            elif os.path.isdir(path):
                if not os.listdir(path):
                    empty.append(f'{name}.{key} (empty dir)')
            elif os.path.getsize(path) == 0:
                empty.append(f'{name}.{key} (0 bytes)')
    assert not absent, absent
    assert not empty, empty


def test_the_PREPARED_directory_holds_what_the_downstream_stages_READ(pipeline_run):
    """The prepare -> tune contract, checked at the file level: `tune`, `retrain_best` and `evaluate` all reach for
    these four by name, and a missing one fails three stages downstream of where it was caused."""
    prepared = dict(pipeline_run['stages'])['prepare_modeling']['output-path']
    for name in ('split_index.csv', 'target_stats.json', 'prepared_config.json', 'targets'):
        assert os.path.exists(os.path.join(prepared, name)), name
    assert os.listdir(os.path.join(prepared, 'features')), 'materialize-features: true, so features/ must be populated'


def test_the_CHECKPOINT_and_the_sweep_record_both_land(pipeline_run):
    """`retrain_best` reads `best_trial.json` out of the sweep's directory, so the two stages' outputs are a contract
    and not just files."""
    stages = dict(pipeline_run['stages'])
    for stage in ('tune', 'retrain_best'):
        directory = stages[stage]['output-path']
        assert os.path.exists(os.path.join(directory, 'best_model.ckpt')), stage
        assert os.path.exists(os.path.join(directory, 'best_trial.json')), stage
    assert os.path.exists(os.path.join(stages['tune']['output-path'], 'trials.csv'))
    assert os.path.exists(os.path.join(stages['evaluate']['output-path'], 'predictions.npz'))


# =====================================================================================================================
# 3. What the evaluation emitted
# =====================================================================================================================
def test_the_metrics_JSON_carries_the_EXPECTED_DAILY_KEYS(evaluation_metrics):
    """Imported from ``evaluation_test.py`` rather than copied, so the unit suite and the pipeline cannot drift apart
    about what the evaluation emits — minus the keys named in the two exclusion tables above.

    The assertion is an EQUALITY on the difference, deliberately: it fails when a key silently disappears, AND when an
    excluded key becomes emittable, which would mean its exclusion note is stale. Change the fixture and this test asks
    you to re-justify the exclusions, which is the intended cost.
    """
    excluded = set(ABSENT_STRUCTURALLY) | set(ABSENT_ON_A_TINY_SPLIT)
    missing = set(EXPECTED_DAILY_KEYS) - set(evaluation_metrics)
    assert missing == excluded, (
        f'unexpectedly missing: {sorted(missing - excluded)}; '
        f'no longer missing (stale exclusion): {sorted(excluded - missing)}'
    )


def test_NO_metric_is_a_non_finite_NUMBER(evaluation_metrics):
    """`json.dump` writes bare `NaN`, which is not valid JSON and which MLflow's `log_metric` rejects — so one
    undefined score would take the whole file with it. The stage drops non-finite values instead, and that is why the
    keys above are ABSENT rather than NaN."""
    import math

    bad = {key: value for key, value in evaluation_metrics.items()
           if isinstance(value, float) and not math.isfinite(value)}
    assert not bad, bad


def test_the_evaluation_emitted_a_LOT_more_than_the_pinned_minimum(evaluation_metrics):
    """Anti-vacuity for the key test: `EXPECTED_DAILY_KEYS` is a floor, not the suite. If the JSON ever shrank to
    roughly that list, the suite would have quietly stopped emitting its per-threshold families while still passing."""
    assert len(evaluation_metrics) > len(EXPECTED_DAILY_KEYS), len(evaluation_metrics)


# =====================================================================================================================
# 4. The report
# =====================================================================================================================
def test_the_report_holds_every_CONFIGURED_figure_that_a_daily_deterministic_run_CAN_draw(pipeline_run, repo_root):
    """Driven off ``config/eval/metrics_daily.yaml``'s own figure list, so adding a figure there without implementing it fails
    here. The self-skips are asserted in the OTHER direction below — together they pin the whole list."""
    import yaml

    report_path = dict(pipeline_run['stages'])['evaluate']['report-path']
    produced = set(os.listdir(report_path))
    with open(os.path.join(repo_root, 'config/eval/metrics_daily.yaml')) as handle:
        configured = yaml.safe_load(handle)['reporting']['figures']

    missing = []
    for figure in configured:
        if figure in SELF_SKIPPED_FIGURES:
            continue
        if figure == 'maps_most_extreme_days':                  # named per date, not after the figure key
            if not any(name.startswith('maps_') and name.endswith('.png') for name in produced):
                missing.append(figure)
        elif f'{figure}.png' not in produced:
            missing.append(figure)
    assert not missing, f'{missing} missing from {sorted(produced)}'


def test_the_figures_a_daily_deterministic_run_CANNOT_draw_SELF_SKIP(pipeline_run):
    """The mechanism that lets one metrics_daily.yaml serve every family: each line-or-table figure fetches its entry from
    `curves` and returns when it is absent. A `reliability.png` appearing here would mean a probability field arrived in
    daily mode; a `rank_histogram.png` would mean the deterministic family emitted ensemble members."""
    report_path = dict(pipeline_run['stages'])['evaluate']['report-path']
    produced = set(os.listdir(report_path))
    unexpected = [f'{figure} ({reason})' for figure, reason in SELF_SKIPPED_FIGURES.items()
                  if f'{figure}.png' in produced]
    assert not unexpected, unexpected


def test_the_report_CSVS_the_comparison_layer_reads_are_all_written(pipeline_run):
    """`combine_curves` reads these by name from a directory this stage fills, so the coupling is only ever checked by a
    test. `reliability_table.csv` and `rank_histogram.csv` are correctly absent here for the same reason their figures
    are."""
    report_path = dict(pipeline_run['stages'])['evaluate']['report-path']
    produced = set(os.listdir(report_path))
    for name in ('psd_curves.csv', 'fss_table.csv', 'roc_pr_curves.csv', 'roc_pr_summary.csv', 'metrics.csv'):
        assert name in produced, f'{name} missing from {sorted(produced)}'


# =====================================================================================================================
# 5. The comparison layer, under the orchestrator
#
# ⚠️ Read what this proves. It feeds ONE family's artifacts in under all three shipped labels, so it does NOT show that
# three families emit the same metric keys — that needs three trained models and stays with the by-hand cross-family
# gate. What it does prove is the plumbing no unit test can reach: that `--Deterministic-UNet <path>` survives
# run_project -> mlflow.projects' parameter list -> Fire's hyphen substitution -> the stage's `**kwargs` and comes back
# out as a row label spelled the way the config spelled it.
# =====================================================================================================================
@pytest.fixture(scope='session')
def comparison_run(pipeline_run, repo_root, tmp_path_factory):
    """Run the shipped comparison stages over the one evaluation's artifacts."""
    import yaml

    stages = dict(pipeline_run['stages'])
    metrics_path, report_path = stages['evaluate']['metrics-path'], stages['evaluate']['report-path']

    with open(os.path.join(repo_root, COMPARISON_CONFIG)) as handle:
        shipped = yaml.safe_load(handle)

    work = tmp_path_factory.mktemp('comparison_e2e')
    output_root = str(work / 'outputs')
    kept = []
    for name, parameters in _stage_blocks(shipped):
        if name not in ('tabulate_metrics', 'combine_curves'):
            continue
        source = metrics_path if name == 'tabulate_metrics' else report_path
        rewritten = {key: (os.path.join(output_root, os.path.basename(str(value))) if key == 'output-path' else source)
                     for key, value in parameters.items()}
        kept.append({name: rewritten})

    derived = {key: value for key, value in shipped.items() if key != 'stages'}
    derived['stages'] = [{'setup': {'outputs': output_root}}] + kept
    derived['lazy'] = False
    config_path = str(work / 'comparison.yaml')
    with open(config_path, 'w') as handle:
        yaml.safe_dump(derived, handle)

    environment = dict(os.environ, DATA_ROOT=pipeline_run['data_root'], OUTPUT_ROOT=output_root,
                       MLFLOW_TRACKING_URI=f'file:{work / "mlruns"}')
    environment['PATH'] = os.path.dirname(sys.executable) + os.pathsep + environment.get('PATH', '')
    completed = subprocess.run([sys.executable, 'run_project.py', config_path, 'comparison_e2e'],
                               cwd=repo_root, env=environment, capture_output=True, text=True)
    return {'completed': completed, 'stages': _stage_blocks(derived), 'labels': [
        key for key in dict(_stage_blocks(shipped))['tabulate_metrics']
        if key not in ('output-path', 'selected-metrics')
    ]}


def test_the_comparison_stages_RUN_under_the_orchestrator(comparison_run):
    completed = comparison_run['completed']
    if completed.returncode != 0:
        tail = '\n'.join((completed.stdout + completed.stderr).splitlines()[-30:])
        pytest.fail(f'the comparison pipeline exited {completed.returncode}:\n{tail}')


def test_the_family_LABELS_survive_the_whole_MLFLOW_path(comparison_run):
    """The round trip that only an end-to-end run can check: the YAML's `--Deterministic-UNet` reaches Fire as
    `Deterministic_UNet` and must come back out as `Deterministic-UNet`. Block 4d's unit tests call `tabulate` directly,
    so they cover `_display` but not the two layers of argument passing in front of it."""
    import pandas as pd

    output_path = dict(comparison_run['stages'])['tabulate_metrics']['output-path']
    table = pd.read_csv(output_path, index_col=0)
    assert sorted(table.index) == sorted(comparison_run['labels']), \
        f'row labels {sorted(table.index)} do not match the config labels {sorted(comparison_run["labels"])}'


def test_the_comparison_table_has_ONE_column_set_for_every_family(comparison_run, evaluation_metrics):
    """The property the by-hand cross-family gate reads, checked here on the plumbing: every family contributes to one
    column set. With one family's metrics fed in under three labels, the columns are its keys and no row has a NaN — an
    all-NaN row would mean a label's JSON never loaded."""
    import pandas as pd

    table = pd.read_csv(dict(comparison_run['stages'])['tabulate_metrics']['output-path'], index_col=0)
    assert set(table.columns) == set(evaluation_metrics)
    assert not table.isna().any().any(), 'no cell should be NaN when every row is the same evaluation'


def test_the_combined_figures_are_WRITTEN(comparison_run):
    """`combine_curves` under the orchestrator, reading the report directory `evaluate` actually produced. The two
    figures a daily deterministic run cannot contribute to (reliability, rank histogram) self-skip, as they do
    per-family."""
    output_path = dict(comparison_run['stages'])['combine_curves']['output-path']
    produced = set(os.listdir(output_path))
    for figure in ('combined_psd', 'combined_fss', 'combined_roc_pr'):
        assert f'{figure}.png' in produced and f'{figure}.pdf' in produced, \
            f'{figure} missing from {sorted(produced)}'
    assert not any('rank_histogram' in name for name in produced), 'a deterministic family has no rank histogram'
