"""Tests for src/stages/evaluate.py — THE single evaluation, for every family and both tasks.

Two layers, deliberately:

1. **A real in-process run**, on a directory ``prepare_modeling`` actually wrote and a checkpoint a real module
   actually saved. That is what makes the assertions mean something: the stage reads ``split_index.csv``,
   ``target_stats.json`` and ``prepared_config.json`` by their exact contracts, so a hand-built prepared directory
   would keep passing after the preparation stage changed one of them.
2. **Contract tests** for what this stage FORWARDS. Branch A's version passed five arguments to ``write_report`` that
   the 02a grammar removed, and its headline-log list had gone stale — both silent, both caught by an AST check
   against the real signatures rather than by running anything.

The bottom section is branch A's ported ``test_probabilistic_eval_hpc.py``: a subprocess run against REAL checkpoints.
It is the only test that scores two families' checkpoints and compares their metric key sets — the invariant the whole
shared-evaluation design exists for — and it needs artifacts no synthetic fixture can produce. ⚠️ A's version was gated
on ``PROB_EVAL_*`` variables that NOTHING sets, so it stayed dormant even after a real run produced exactly what it
wanted; it now DISCOVERS its artifacts from the shipped cross-family eval config and runs the moment they exist.
"""
import ast
import glob
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import evaluate as evaluate_stage                                # bare name: see conftest.py
from tests.stages.conftest import FEATURES, HEIGHT, HOURS, WIDTH


# =====================================================================================================================
# A real run: prepared by the preparation stage, scored by the evaluation stage
# =====================================================================================================================
@pytest.fixture
def scored(prepared, unet_trial, target_stats, save_checkpoint, tmp_path):
    """Prepare, build a real module, save its checkpoint, and run the stage. Returns ``(metrics, output, report)``."""
    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    def build(report=False, **overrides):
        output_path, prepared_config, _, _ = prepared(mode='daily', n_days=6)
        in_channels = len(FEATURES.split(',')) * HOURS            # hourly_stack: Vf * T
        module = DeterministicUnetModule(
            unet_trial(), in_channels, target_stats(mode='daily', hourly_threshold=2),
            {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})
        checkpoint = save_checkpoint(module, name='best_model.ckpt')

        out = str(tmp_path / 'evaluation')
        report_dir = str(tmp_path / 'report') if report else None
        metrics_path = str(tmp_path / 'evaluation' / 'test_metrics.json')
        arguments = dict(input_path=output_path, model_path=checkpoint, output_path=out,
                         metrics_path=metrics_path, report_path=report_dir, split='test',
                         accelerator='cpu', devices=1, num_workers=0, batch_size=2,
                         progress_bar=False, ensemble_size=2)
        arguments.update(overrides)
        evaluate_stage.evaluate(**arguments)

        with open(metrics_path) as handle:
            metrics = json.load(handle)
        return metrics, out, report_dir
    return build


def test_a_real_run_writes_the_metrics_json_and_the_predictions(scored):
    metrics, output, _ = scored()

    assert metrics, 'the metrics JSON is empty'
    assert os.path.exists(os.path.join(output, 'predictions.npz'))

    stored = np.load(os.path.join(output, 'predictions.npz'))
    assert {'prediction', 'observation', 'dates', 'hours'} <= set(stored)
    assert stored['prediction'].shape == stored['observation'].shape
    assert stored['prediction'].shape[-2:] == (HEIGHT, WIDTH)


def test_the_metrics_json_holds_only_JSON_VALID_numbers(scored):
    """⚠️ Not cosmetic. ``json.dump`` writes bare ``NaN``, which is not valid JSON and which MLflow's ``log_metric``
    rejects — so ONE undefined score would make the whole file unreadable and lose every other number with it. And
    NaNs are routine here: the deterministic family's ensemble scalars are all NaN by design."""
    metrics, output, _ = scored()

    for key, value in metrics.items():
        assert isinstance(value, (int, float)), f'{key} is {type(value).__name__}'
        assert np.isfinite(value), f'{key} = {value}'

    # the real proof: json.loads ACCEPTS bare NaN by default, so re-reading with that leniency switched off is what
    # a strict consumer (MLflow, jq, any other language) would do
    raw = open(os.path.join(os.path.dirname(output), 'evaluation', 'test_metrics.json')).read()
    json.loads(raw, parse_constant=lambda constant: pytest.fail(f'non-JSON constant in the metrics file: {constant}'))


def test_the_deterministic_family_reports_NO_ensemble_scalars(scored):
    """It emits no members, so CRPS / spread-skill / the rank histogram are undefined — and being undefined, they are
    dropped rather than written as NaN. That is what makes the cross-family comparison table's COLUMNS the thing to
    compare, not its values."""
    metrics, _, _ = scored()
    assert not [key for key in metrics if key.startswith(('crps', 'spread_skill', 'rank_histogram'))], metrics


def test_the_report_directory_holds_the_per_day_maps_as_PNG_AND_PDF(scored):
    """Maps get a vector copy unconditionally — they are the publication output — while the curve figures follow the
    configured ``formats``."""
    _, _, report = scored(report=True)

    pngs = sorted(glob.glob(os.path.join(report, 'maps_*.png')))
    pdfs = sorted(glob.glob(os.path.join(report, 'maps_*.pdf')))
    assert pngs, sorted(os.listdir(report))
    assert len(pngs) == len(pdfs)


def test_NO_report_path_skips_reporting_entirely(scored, tmp_path):
    metrics, _, report = scored(report=False)
    assert report is None
    assert metrics, 'the metrics are still written without a report'


def test_an_EMPTY_split_raises_naming_it(prepared, unet_trial, target_stats, save_checkpoint, tmp_path):
    """A typo in ``split:`` would otherwise evaluate zero items and report a JSON of NaNs."""
    from src.utils.modeling.deterministic_module import DeterministicUnetModule

    output_path, _, _, _ = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS
    module = DeterministicUnetModule(
        unet_trial(), in_channels, target_stats(mode='daily', hourly_threshold=2),
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})

    with pytest.raises(ValueError, match='is empty'):
        evaluate_stage.evaluate(
            input_path=output_path, model_path=save_checkpoint(module), output_path=str(tmp_path / 'out'),
            split='nonexistent', accelerator='cpu', num_workers=0, batch_size=2, progress_bar=False)


# =====================================================================================================================
# ⭐ What branch A forwarded and the 02a grammar removed
# =====================================================================================================================
def _call_to(function_name):
    """The call node for ``function_name`` inside the stage, so its keywords can be checked without running it."""
    tree = ast.parse(inspect.getsource(evaluate_stage))
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == function_name)


@pytest.mark.parametrize('dead', ['colorbar_scale', 'colorbar_integer_bins', 'quantize', 'max_val',
                                  'occurrence_event'])
def test_the_removed_map_colour_ARGUMENTS_are_not_passed_to_write_report(dead):
    """⭐ All five were on branch A's call and are gone from ``write_report``. Under the 02a grammar the scale is
    always unit bins in lightning-hours driven by ``ceil(nanmax(obs))`` per date, and the sub-1 white/grey split
    replaced the occurrence mask — there is nothing left to configure per call. Passing one is a ``TypeError`` at the
    end of a full evaluation, after the model has run over the whole split."""
    assert dead not in {keyword.arg for keyword in _call_to('write_report').keywords}
    assert dead not in inspect.signature(evaluate_stage.evaluate).parameters


@pytest.mark.parametrize('function_name,target', [
    ('write_report', 'write_report'),
    ('run_metric_suite', 'run_metric_suite'),
    ('build_baselines', 'build_baselines'),
    ('finalize_ensemble_metrics', 'finalize_ensemble_metrics'),
])
def test_every_keyword_the_stage_passes_is_ACCEPTED_by_its_callee(function_name, target):
    """The general form of the check above. This stage is the join between the model layer and the metric layer, so a
    renamed parameter anywhere in ``src/utils/metrics`` surfaces here — and only at the very end of a real run."""
    from src.utils.metrics import evaluation, reporting

    accepted = set(inspect.signature(getattr(reporting if target == 'write_report' else evaluation,
                                             target)).parameters)
    passed = {keyword.arg for keyword in _call_to(function_name).keywords}
    assert passed <= accepted, f'{target} does not accept {sorted(passed - accepted)}'


def test_the_positional_arguments_of_run_metric_suite_are_the_documented_SEVEN():
    """Its first seven are positional and order-dependent — ``prediction`` and ``observation`` are adjacent and the
    same shape, so swapping them scores the model against itself and produces a perfect, meaningless report."""
    call = _call_to('run_metric_suite')
    assert len(call.args) == 7, ast.unparse(call)
    assert [ast.unparse(argument) for argument in call.args[1:4]] == ['prediction', 'observation', 'probability']


# =====================================================================================================================
# The stale-name hazards
# =====================================================================================================================
def test_every_HEADLINE_metric_is_a_key_the_suite_actually_EMITS():
    """⭐ Branch A's headline list named ``ets_p99`` / ``sedi_p99`` / ``fss_p90_s3`` / ``rank_corr_p99`` /
    ``psd_ratio_high``, all re-keyed to absolute hour bands in Step 3. Nothing raised: the summary line just printed
    fewer entries, which reads exactly like a run where those scores were undefined."""
    from tests.utils.metrics.evaluation_test import EXPECTED_DAILY_KEYS

    ensemble_keys = {'crps', 'crps_occ', 'almost_fair_crps', 'almost_fair_crps_occ', 'spread_skill_ratio',
                     'rank_histogram_reliability'}
    emitted = set(EXPECTED_DAILY_KEYS) | ensemble_keys

    unknown = [name for name in evaluate_stage.HEADLINE_METRICS if name not in emitted]
    assert not unknown, f'headline names no score emits: {unknown}'


def test_the_headline_covers_skill_discrimination_structure_AND_the_ensemble():
    """A summary of only one kind would let a run look fine on the axis it happens to report."""
    headline = set(evaluate_stage.HEADLINE_METRICS)
    assert headline & {'ets_occurrence', 'sedi_occurrence'}, 'categorical skill'
    assert headline & {'average_precision_occurrence', 'brier_skill_score'}, 'discrimination / calibration'
    assert headline & {'psd_high_fidelity', 'fss_occurrence_s1'}, 'spatial structure'
    assert headline & {'crps', 'spread_skill_ratio'}, 'the ensemble scalars'


@pytest.mark.source_invariant
def test_the_loader_is_the_RENAMED_one():
    """``load_regression_module`` became ``load_model_module`` when the registry stopped being regression-specific."""
    from tests.conftest import executable_source

    executable = executable_source(evaluate_stage)
    assert 'load_model_module' in executable
    assert 'load_regression_module' not in executable


@pytest.mark.source_invariant
def test_the_dead_occurrence_threshold_WARNING_is_gone():
    """It warned that a checkpoint had been tuned on targets denoised at preparation time. ``resolve_occurrence_event``
    now HARD-ASSERTS that no ``occurrence_threshold`` reappears, so the branch was unreachable — a warning that can
    never fire is worse than none, because it implies the case is handled."""
    from tests.conftest import executable_source

    assert 'occurrence_threshold' not in executable_source(evaluate_stage)


def test_the_residual_diagnostics_FUNCTION_is_imported_under_an_ALIAS():
    """⚠️ The stage has a ``residual_diagnostics`` PARAMETER, which would shadow the function of the same name — and
    the shadowing is silent until the residual branch runs, where ``bool`` is not callable."""
    from src.utils.metrics import diagnostics

    assert evaluate_stage.compute_residual_diagnostics is diagnostics.residual_diagnostics
    assert 'residual_diagnostics' in inspect.signature(evaluate_stage.evaluate).parameters


# =====================================================================================================================
# The accelerator probe
# =====================================================================================================================
def test_cpu_is_returned_unchanged_and_reports_no_cuda():
    assert evaluate_stage._resolve_accelerator('cpu') == ('cpu', False)


def test_an_UNUSABLE_cuda_device_falls_back_to_cpu_with_a_warning(monkeypatch, caplog):
    """"Available" does not mean usable: another process may hold the device in exclusive mode, or it may be out of
    memory. Probing turns a crash at the first forward pass into a slower evaluation."""
    import logging

    import torch

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def busy(*args, **kwargs):
        raise RuntimeError('CUDA error: all CUDA-capable devices are busy or unavailable')

    monkeypatch.setattr(torch, 'zeros', busy)
    with caplog.at_level(logging.WARNING, logger='evaluate'):
        accelerator, use_cuda = evaluate_stage._resolve_accelerator('auto')

    assert (accelerator, use_cuda) == ('cpu', False)
    assert any('falling back to cpu' in record.getMessage().lower() for record in caplog.records)


# =====================================================================================================================
# The ensemble switch
# =====================================================================================================================
def test_a_single_member_run_WARNS_for_a_stochastic_family(prepared, mc_trial, target_stats, save_checkpoint,
                                                           tmp_path, caplog):
    """⚠️ CLAUDE.md's standing warning: ``spread_skill_sums`` uses ``ddof=1``, so one member yields a silent NaN rather
    than an error. The stage refuses to enable the suite below 2 AND says why, because the symptom otherwise is a
    metrics JSON quietly missing its ensemble half."""
    import logging

    from src.utils.modeling.mc_dropout_module import MCDropoutModule

    output_path, _, _, _ = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS
    module = MCDropoutModule(
        mc_trial(), in_channels, target_stats(mode='daily', hourly_threshold=2),
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})

    with caplog.at_level(logging.WARNING, logger='evaluate'):
        evaluate_stage.evaluate(
            input_path=output_path, model_path=save_checkpoint(module), output_path=str(tmp_path / 'single'),
            split='test', accelerator='cpu', num_workers=0, batch_size=2, progress_bar=False,
            ensemble_size=1, save_predictions=False)

    assert any('ddof=1' in record.getMessage() for record in caplog.records)


def test_an_ensemble_run_reports_the_probabilistic_suite(prepared, mc_trial, target_stats, save_checkpoint,
                                                         tmp_path):
    """The other half of the shared-evaluation claim: the SAME stage, on a stochastic checkpoint, additionally emits
    the ensemble group — without a family-specific branch anywhere."""
    from src.utils.modeling.mc_dropout_module import MCDropoutModule

    output_path, _, _, _ = prepared(mode='daily', n_days=6)
    in_channels = len(FEATURES.split(',')) * HOURS
    module = MCDropoutModule(
        mc_trial(), in_channels, target_stats(mode='daily', hourly_threshold=2),
        {'mean': [0.0] * in_channels, 'std': [1.0] * in_channels})
    metrics_path = str(tmp_path / 'ens' / 'metrics.json')

    evaluate_stage.evaluate(
        input_path=output_path, model_path=save_checkpoint(module), output_path=str(tmp_path / 'ens'),
        metrics_path=metrics_path, split='test', accelerator='cpu', num_workers=0, batch_size=2,
        progress_bar=False, ensemble_size=4, save_predictions=False)

    with open(metrics_path) as handle:
        metrics = json.load(handle)
    assert 'crps' in metrics, sorted(metrics)
    assert 'spread_skill_ratio' in metrics


# =====================================================================================================================
# Stage wiring
# =====================================================================================================================
def test_every_parameter_the_shipped_configs_pass_is_accepted(repo_root):
    from src.utils.io.parse_config import parse_config

    accepted = set(inspect.signature(evaluate_stage.evaluate).parameters)
    paths = [f'config/{family}/{family}_daily{tier}.yaml'
             for family in ('deterministic_unet', 'mc_dropout', 'diffusion')
             for tier in ('', '_smoke_cpu', '_smoke_gpu')]
    paths += ['config/eval/probabilistic_eval.yaml', 'config/eval/probabilistic_eval_smoke_cpu.yaml']

    for relative in paths:
        config = parse_config(os.path.join(repo_root, relative))
        for stage in config['stages']:
            for name, parameters in stage.items():
                if name != 'evaluate':
                    continue
                passed = {key.replace('-', '_') for key in parameters}
                assert passed <= accepted, f'{relative}: unknown {sorted(passed - accepted)}'


def test_there_is_exactly_ONE_evaluation_stage(repo_root):
    """The design invariant CLAUDE.md states as "Never add a family-specific evaluation path". Branch D had two of
    these, one per family, which is precisely what makes two families' numbers incomparable."""
    stages = {os.path.basename(path) for path in glob.glob(os.path.join(repo_root, 'src/stages/*.py'))}
    family_specific = [name for name in stages if name.startswith('evaluate') and name != 'evaluate.py']
    assert not family_specific, family_specific


def test_the_stage_is_wrapped_with_fire():
    assert 'Fire(evaluate)' in inspect.getsource(evaluate_stage)


def test_the_stage_imports_root_path_BEFORE_any_src_import(repo_root):
    lines = [line for line in open(os.path.join(repo_root, 'src/stages/evaluate.py'))
             if line.startswith('from ') or line.startswith('import ')]
    root_position = next(index for index, line in enumerate(lines) if '__init__ import root_path' in line)
    src_positions = [index for index, line in enumerate(lines) if line.startswith('from src.')]
    assert src_positions and min(src_positions) > root_position


# =====================================================================================================================
# Branch A's ported HPC test — a subprocess run against REAL checkpoints
#
# This is the only test that scores two families' real checkpoints and compares their metric KEY SETS, which is the
# invariant the shared evaluation exists for, and no synthetic fixture can produce a trained checkpoint of each family.
#
# ⚠️ A's version was gated on `PROB_EVAL_*` environment variables that NOTHING sets — not the pipeline, not conftest —
# so it stayed dormant even after a real run produced exactly the artifacts it wanted. It now DISCOVERS them from the
# shipped cross-family eval config, the same principle as every other config-driven test here, and runs the moment
# those artifacts exist. The env vars survive as an override, for a checkpoint no config names.
# =====================================================================================================================
FAMILIES = ('deterministic_unet', 'mc_dropout', 'diffusion')

# The families whose `predict_step` returns `ensemble_members`. The deterministic U-net does not, so the whole ensemble
# group is SKIPPED for it rather than written as NaN — which is why the cross-family comparison compares COLUMNS of a
# union table, not per-family key sets. See `test_families_are_scored_by_the_SAME_suite_MODULO_ensemble_capability`.
STOCHASTIC_FAMILIES = ('mc_dropout', 'diffusion')
ENSEMBLE_PREFIXES = ('crps', 'almost_fair_crps', 'spread_skill', 'rank_histogram')


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, '') else default


def _without_ensemble(keys):
    return {key for key in keys if not key.startswith(ENSEMBLE_PREFIXES)}


def _ensemble_only(keys):
    return {key for key in keys if key.startswith(ENSEMBLE_PREFIXES)}


def _configured_evaluations(repo_root):
    """``{family: (input_path, model_path)}`` as the shipped cross-family eval configs declare them.

    The family is the last segment of ``output-path`` (``$OUTPUT_ROOT/comparison/evaluation/<family>``). Both tiers are
    read, smoke FIRST: a smoke run produces these artifacts long before a full one does, and it is the run most
    likely to have happened.
    """
    from src.utils.io.parse_config import parse_config

    declared = {}
    for tier in ('_smoke_cpu', ''):
        config = parse_config(os.path.join(repo_root, f'config/eval/probabilistic_eval{tier}.yaml'))
        for stage in config['stages']:
            for name, parameters in stage.items():
                if name != 'evaluate':
                    continue
                family = os.path.basename(parameters['output-path'])
                declared.setdefault(family, (parameters['input-path'], parameters['model-path']))
    return declared


def _available_evaluations(repo_root):
    """The subset of the above whose prepared directory AND checkpoint are actually on disk."""
    available = {}
    for family, (prepared_dir, model_path) in _configured_evaluations(repo_root).items():
        if os.path.isdir(os.path.join(repo_root, prepared_dir)) \
                and os.path.exists(os.path.join(repo_root, model_path)):
            available[family] = (prepared_dir, model_path)
    return available


def _artifacts_for(repo_root, family):
    """``(input_path, model_path)`` for one family — the env override first, else the shipped config."""
    prepared_dir, checkpoint = _env('PROB_EVAL_PREPARED'), _env(f'PROB_EVAL_{family.upper()}')
    if prepared_dir and checkpoint:
        return prepared_dir, checkpoint
    return _available_evaluations(repo_root).get(family, (None, None))


def _discovery_hint():
    """The REAL reason discovery found nothing, named in the skip message.

    ⚠️ Block 4g found these tests skipping while the artifacts existed, under a message that said "run its pipeline
    first" — advice for a problem the user did not have. `OUTPUT_ROOT` was simply unset, so the discovered paths were
    ``/deterministic_and_mc_dropout_smoke_cpu/…``: ABSOLUTE, because `{{$OUTPUT_ROOT}}` substitutes to the empty
    string, and `os.path.join(repo_root, '/abs')` discards `repo_root` entirely. That is the footgun
    `parse_config_test.py` documents in its own test, and this suite was quietly a victim of it.
    """
    if not _env('OUTPUT_ROOT'):
        return ('⚠️ OUTPUT_ROOT is UNSET, so every discovered path resolved to the filesystem root and could never '
                'exist — this is a missing environment variable, NOT a missing checkpoint. Export DATA_ROOT and '
                'OUTPUT_ROOT to run these.')
    return (f'OUTPUT_ROOT={_env("OUTPUT_ROOT")} is set, so this really is a missing artifact: run the family\'s '
            f'pipeline (or its *_smoke_cpu tier) first.')


def test_the_DISCOVERY_finds_all_three_families_in_the_shipped_config(repo_root):
    """⭐ Runs ALWAYS, with no artifacts, and it is the test that keeps the ones below from going quietly dormant. If the
    eval config's ``output-path`` leaf were renamed, discovery would return nothing, they would skip forever, and
    the suite would still be green — the same failure mode the env-var gating had, just better hidden."""
    declared = _configured_evaluations(repo_root)
    assert set(declared) == set(FAMILIES), sorted(declared)
    for family, (prepared_dir, model_path) in declared.items():
        assert model_path.endswith('best_model.ckpt'), (family, model_path)
        assert 'prepared' in prepared_dir, (family, prepared_dir)


def test_the_DISCOVERY_honours_OUTPUT_ROOT(repo_root, monkeypatch):
    """⭐ The other half of the anti-dormancy guard, and the half that was missing (block 4g).

    The guard above checks the config LEAF; this checks the ENVIRONMENT, which is what actually silenced these tests.
    Every discovered path must sit under `$OUTPUT_ROOT` — because with the variable unset they become absolute at `/`,
    `os.path.isdir` says no, and the tests skip while claiming a checkpoint is missing.

    Runs always, needs no artifacts, and would have caught the real dormancy on the day it started.
    """
    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    declared = _configured_evaluations(repo_root)
    assert declared, 'discovery returned nothing at all'
    for family, (prepared_dir, model_path) in declared.items():
        assert prepared_dir.startswith('/MARKER'), (family, prepared_dir)
        assert model_path.startswith('/MARKER'), (family, model_path)


def test_an_UNSET_output_root_is_reported_as_the_ENVIRONMENT_not_a_missing_checkpoint(monkeypatch):
    """The skip message has to name the cause the user can act on. "Run the pipeline first" sent block 4g looking for
    artifacts that were already there."""
    monkeypatch.delenv('OUTPUT_ROOT', raising=False)
    assert 'OUTPUT_ROOT is UNSET' in _discovery_hint()

    monkeypatch.setenv('OUTPUT_ROOT', '/somewhere')
    assert 'missing artifact' in _discovery_hint()


def _run_stage(repo_root, prepared_dir, model_path, model_family, out_dir, report_dir, metrics_path):
    """Invoke the real evaluation stage exactly as the pipeline does, from the repo root."""
    command = [
        sys.executable, 'src/stages/evaluate.py',
        '--input_path', prepared_dir,
        '--model_path', model_path,
        '--output_path', out_dir,
        '--report_path', report_dir,
        '--metrics_config', _env('PROB_EVAL_METRICS', 'config/eval/metrics_daily.yaml'),
        '--metrics_path', metrics_path,
        '--split', _env('PROB_EVAL_SPLIT', 'valid'),
        '--accelerator', 'auto',
        '--ensemble_size', _env('PROB_EVAL_ENSEMBLE', '4'),      # must be >= 2: spread_skill_sums uses ddof=1
        '--progress_bar', 'False',
    ]
    if model_family is not None:
        command += ['--model_family', model_family]
    limit = _env('PROB_EVAL_LIMIT', '2')
    if limit is not None:
        command += ['--limit_batches', str(limit)]

    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, f'{model_path} (family={model_family})\n{result.stderr[-4000:]}'


def _check_outputs(label, metrics_path, report_dir):
    assert os.path.exists(metrics_path), f'[{label}] no metrics JSON at {metrics_path}'
    with open(metrics_path) as handle:
        metrics = json.load(handle)
    assert metrics, f'[{label}] metrics JSON is empty'

    pngs = glob.glob(os.path.join(report_dir, 'maps_*.png'))
    pdfs = glob.glob(os.path.join(report_dir, 'maps_*.pdf'))
    assert pngs, f'[{label}] no per-day maps_*.png in {report_dir}'
    assert len(pngs) == len(pdfs), f'[{label}] png/pdf mismatch ({len(pngs)} vs {len(pdfs)})'
    return set(metrics)


@pytest.mark.parametrize('family', FAMILIES)
def test_the_stage_scores_each_family_on_REAL_artifacts(family, repo_root, tmp_path):
    """Runs as soon as that family's pipeline has produced a checkpoint — no environment setup needed. Override with
    ``PROB_EVAL_PREPARED`` + ``PROB_EVAL_<FAMILY>`` to point at something the configs do not name."""
    prepared_dir, checkpoint = _artifacts_for(repo_root, family)
    if not (prepared_dir and checkpoint):
        pytest.skip(f'no {family} artifacts found. {_discovery_hint()} '
                    f'(or set PROB_EVAL_PREPARED and PROB_EVAL_{family.upper()} to override)')

    out_dir = str(tmp_path / family / 'eval')
    report_dir = str(tmp_path / family / 'report')
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.json')

    _run_stage(repo_root, prepared_dir, checkpoint, family, out_dir, report_dir, metrics_path)
    _check_outputs(family, metrics_path, report_dir)


def test_families_are_scored_by_the_SAME_suite_MODULO_ensemble_capability(repo_root, tmp_path):
    """⭐ The invariant the shared evaluation exists for — stated correctly.

    ⚠️ This test used to assert STRICT equality of metric keys across families (``keys == reference``) and it was
    WRONG: the design deliberately violates it. The deterministic family's ``predict_step`` emits no
    ``ensemble_members``, so the whole ensemble group is SKIPPED rather than written as NaN, and that is exactly what
    makes the cross-family comparison compare the COLUMNS of a union table. Two tests in this file contradicted each
    other — ``test_the_deterministic_family_reports_NO_ensemble_scalars`` requires those six keys to be absent — and
    only the dormancy below kept it invisible: run with both roots exported, the strict version failed with
    ``crps``, ``crps_occ``, ``almost_fair_crps``, ``almost_fair_crps_occ``, ``spread_skill_ratio`` and
    ``rank_histogram_reliability`` extra in the stochastic family's set. Found in block 4g.

    So the claim is split into the two things that are actually true, and together they still catch a
    family-specific evaluation path:

    1. **Outside the ensemble group, every family reports the SAME keys.** That is the shared suite.
    2. **Inside it, the keys appear exactly for the families that emit members.** That is the capability, and asserting
       it means a stochastic family silently losing its ensemble scores still fails here.

    Each family is scored on ITS OWN prepared directory, which is what the pipelines produce — diffusion's differs in
    residual mode. The claim is that the SUITE is shared, not the data.
    """
    available = {family: paths for family, paths in _available_evaluations(repo_root).items()}
    if len(available) < 2:
        pytest.skip(f'needs at least two family checkpoints; found {sorted(available)}. '
                    f'{_discovery_hint()}')

    key_sets = {}
    for family, (prepared_dir, checkpoint) in available.items():
        out_dir = str(tmp_path / family / 'eval')
        report_dir = str(tmp_path / family / 'report')
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, 'metrics.json')
        _run_stage(repo_root, prepared_dir, checkpoint, family, out_dir, report_dir, metrics_path)
        key_sets[family] = _check_outputs(family, metrics_path, report_dir)

    # 1. the shared suite: identical keys once the capability-gated group is set aside
    reference_family, reference = next(iter(key_sets.items()))
    shared_reference = _without_ensemble(reference)
    for family, keys in key_sets.items():
        shared = _without_ensemble(keys)
        assert shared == shared_reference, (
            f'{family} and {reference_family} were scored by DIFFERENT suites — '
            f'only {family}: {sorted(shared - shared_reference)}; '
            f'only {reference_family}: {sorted(shared_reference - shared)}'
        )

    # 2. the capability: the ensemble group is present exactly for the families that emit members
    for family, keys in key_sets.items():
        ensemble = _ensemble_only(keys)
        if family in STOCHASTIC_FAMILIES:
            assert ensemble, f'{family} is stochastic but reported no ensemble scores at all'
        else:
            assert not ensemble, f'{family} emits no members, so these must be absent rather than NaN: {sorted(ensemble)}'
