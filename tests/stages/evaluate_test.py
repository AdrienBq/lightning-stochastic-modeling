"""Tests for src/stages/evaluate.py — ⏳ **STEP 4 PLACEHOLDER, entirely skipped.**

The module does not exist yet: ``src/stages/`` holds only the plumber template's ``hello_world.py`` / ``run.py`` /
``setup.py``. This file exists so the mirror states Step 4's obligation rather than hiding it, and so the port of branch
A's ``tests/test_probabilistic_eval_hpc.py`` is done once rather than re-derived later.

The paths and flags below are **pre-fixed for Step 2's contract**, which is the part that would otherwise have to be
re-worked: ``config/metrics.yaml`` became ``config/eval/metrics.yaml``, the prepared leaf ``daily_lightning_hours``
became ``daily``, and the ``--colorbar_scale`` / ``--colorbar_integer_bins`` flags are gone with the 02a grammar (the
colour scale is always unit bins in lightning-hours, driven per date).

**To enable in Step 4:** delete the ``pytestmark`` line. Nothing else should need changing.

What this asserts, once the stage exists — the "one evaluation for all families" invariant, end to end on real
checkpoints: the stage completes and writes a non-empty metrics JSON, the report holds per-day ``maps_<date>.png`` AND
``.pdf``, and — the point of the whole file — when both a diffusion and an MC-dropout checkpoint are given, the two
metrics JSONs carry the SAME metric keys. Different key sets would mean the families were scored by different suites.
"""
import glob
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skip(reason='src/stages/evaluate.py is Step 4; delete this line to enable')


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, '') else default


def _run_stage(repo_root, prepared, model_path, model_family, out_dir, report_dir, metrics_path):
    """Invoke the real evaluation stage exactly as the pipeline does, from the repo root."""
    command = [
        sys.executable, 'src/stages/evaluate.py',
        '--input_path', prepared,
        '--model_path', model_path,
        '--output_path', out_dir,
        '--report_path', report_dir,
        # config/metrics.yaml -> config/eval/metrics.yaml (Step 2 grouped one directory per concern)
        '--metrics_config', _env('PROB_EVAL_METRICS', 'config/eval/metrics.yaml'),
        '--metrics_path', metrics_path,
        '--split', _env('PROB_EVAL_SPLIT', 'valid'),
        '--accelerator', 'auto',
        '--ensemble_size', _env('PROB_EVAL_ENSEMBLE', '4'),      # must be >= 2: spread_skill_sums uses ddof=1
        '--progress_bar', 'False',
        # NOTE: --colorbar_scale / --colorbar_integer_bins deliberately absent, removed with the 02a grammar
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


@pytest.mark.parametrize('family', ['diffusion', 'mc_dropout', 'deterministic_unet'])
def test_the_stage_scores_each_family(family, repo_root, tmp_path):
    """Set ``PROB_EVAL_PREPARED`` to a prepared directory (the ``daily`` leaf, not the old
    ``daily_lightning_hours``) and ``PROB_EVAL_<FAMILY>`` to a checkpoint."""
    prepared = _env('PROB_EVAL_PREPARED')
    checkpoint = _env(f'PROB_EVAL_{family.upper()}')
    if not (prepared and checkpoint):
        pytest.skip(f'set PROB_EVAL_PREPARED and PROB_EVAL_{family.upper()}')

    out_dir = str(tmp_path / family / 'eval')
    report_dir = str(tmp_path / family / 'report')
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.json')

    _run_stage(repo_root, prepared, checkpoint, family, out_dir, report_dir, metrics_path)
    _check_outputs(family, metrics_path, report_dir)


def test_all_available_families_are_scored_by_the_SAME_metric_suite(repo_root, tmp_path):
    """The invariant the whole shared-evaluation design exists for. Two families reporting different metric keys means
    ``evaluate`` grew a family-specific path."""
    prepared = _env('PROB_EVAL_PREPARED')
    available = {family: _env(f'PROB_EVAL_{family.upper()}')
                 for family in ('diffusion', 'mc_dropout', 'deterministic_unet')}
    available = {family: path for family, path in available.items() if path}
    if not prepared or len(available) < 2:
        pytest.skip('needs PROB_EVAL_PREPARED and at least two family checkpoints')

    key_sets = {}
    for family, checkpoint in available.items():
        out_dir = str(tmp_path / family / 'eval')
        report_dir = str(tmp_path / family / 'report')
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, 'metrics.json')
        _run_stage(repo_root, prepared, checkpoint, family, out_dir, report_dir, metrics_path)
        key_sets[family] = _check_outputs(family, metrics_path, report_dir)

    reference_family, reference = next(iter(key_sets.items()))
    for family, keys in key_sets.items():
        assert keys == reference, (
            f'{family} and {reference_family} were scored by DIFFERENT suites — '
            f'only {family}: {sorted(keys - reference)}; only {reference_family}: {sorted(reference - keys)}'
        )
