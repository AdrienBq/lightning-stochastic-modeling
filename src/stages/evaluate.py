"""Pipeline stage: held-out evaluation and reporting — THE single evaluation, for every family and both tasks.

This is the design invariant CLAUDE.md states as *"One evaluation for all families. Never add a family-specific
evaluation path."* Branch A had one of these plus a legacy shim; branch D had two, one per family, which is exactly
what makes two families' numbers incomparable. There is one here, and adding a second would defeat the point of the
whole merge.

It works because every family satisfies the same `predict_step` contract:

    {'observation': [B, H, W], 'prediction': [B, H, W], 'probability': [B, H, W] | None,
     'ensemble_members': [B, M, H, W], 'ensemble_partials': {...}}       # the last two: stochastic families only

`registry.load_model_module` dispatches on the checkpoint's family marker (with `_sniff_family` as the legacy
fallback) and returns MC-dropout wrapped in `MCDropoutEnsembleModule`, so its repeated forward passes arrive in the
same shape a diffusion ensemble does. Everything downstream — the baselines, the metric suite, the report — is then
identical, and the deterministic family simply contributes `None` where an ensemble would be.

`probability` is the occurrence forecast. In HOURLY mode the model's own output IS the probability, so every U-net
family returns `prediction` there and `None` in daily mode, where there is no occurrence head. That is what makes
`reliability`, `explained_deviance` and `dice_occurrence` appear on the hourly task and be absent on the daily one —
correctly, since a number computed from lightning-hours would not be a probability.

Stage outputs:

* the flat scalar metrics JSON at `--metrics-path` (auto-logged to MLflow by the orchestrator);
* the report directory at `--report-path` (figures + CSV tables, auto-logged as run artifacts);
* `predictions.npz` in `--output-path`.

⚠️ NON-FINITE METRICS ARE DROPPED from the JSON, with a count logged. That is not cosmetic: `json.dump` writes bare
`NaN`, which is not valid JSON and which MLflow's `log_metric` rejects, so one undefined score would make the whole
file unreadable and lose every other number with it. NaNs are routine here — the deterministic family's ensemble
scalars are all NaN by design.

⛔ The map-colour arguments branch A carried (`colorbar_scale`, `colorbar_integer_bins`, `quantize`, `max_val`) and
`occurrence_event` are GONE from `write_report`. Under the 02a grammar the scale is always unit bins in
lightning-hours driven by `ceil(nanmax(obs))` per date, and the sub-1 white/grey split replaced the occurrence mask.
There is nothing left to configure per call.

Usage (standalone)::

    python src/stages/evaluate.py \\
        --input_path $OUTPUT_ROOT/diffusion/prepared/daily \\
        --model_path $OUTPUT_ROOT/diffusion/best/best_model.ckpt \\
        --output_path $OUTPUT_ROOT/diffusion/evaluation \\
        --metrics_config config/eval/metrics.yaml \\
        --split test --ensemble_size 32
"""
import json
import logging
import math
import os
from typing import Optional, Union

import lightning as L
import numpy as np
import torch
from fire import Fire
from torch.utils.data import DataLoader
from yaml import safe_load

from __init__ import root_path, console_handler
from src.utils.io.data import load_prepared_artifacts
# aliased because this stage's own `residual_diagnostics` PARAMETER would otherwise shadow the function
from src.utils.metrics.diagnostics import residual_diagnostics as compute_residual_diagnostics
from src.utils.metrics.evaluation import (
    build_baselines,
    finalize_ensemble_metrics,
    merge_ensemble_partials,
    resolve_occurrence_event,
    run_metric_suite,
)
from src.utils.metrics.reporting import write_report
from src.utils.modeling.dataset import LightningMapsDataset
from src.utils.modeling.registry import load_model_module

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# The one-line summary printed to the orchestrator log when the stage finishes: the headline skill, the headline
# discrimination, the structure fidelity, and the ensemble scalars when there is an ensemble.
#
# ⚠️ Every name here must be one the suite actually emits. Branch A's list had gone stale (`ets_p99`, `fss_p90_s3`,
# `rank_corr_p99`, `psd_ratio_high` — all re-keyed to absolute bands in Step 3), and the failure was SILENT: the
# summary just printed fewer entries. `evaluate_test.py` pins this list against the suite's real key set.
HEADLINE_METRICS = (
    'ets_occurrence', 'sedi_occurrence', 'frequency_bias_occurrence',
    'mae_cond_pos', 'r2_occurrence', 'average_precision_occurrence', 'brier_skill_score',
    'psd_high_fidelity', 'fss_occurrence_s1',
    'crps', 'crps_occ', 'spread_skill_ratio', 'rank_histogram_reliability',
)

# a member stack above this is warned about: the FFT/gradient temporaries of the pooled structure scores scale with it
MEMBER_STACK_WARN_GB = 8.0


def _as_name_list(value: Union[str, list, tuple]) -> list:
    """Normalize a comma-separated string (or an already-parsed sequence) into a list of names."""
    if isinstance(value, str):
        return [name.strip() for name in value.split(',') if name.strip()]
    return list(value)


def _resolve_accelerator(accelerator: str):
    """Pick the device, falling back to CPU when CUDA is reported available but is not usable.

    "Available" does not mean usable: another process may hold the device in exclusive mode, or it may be out of
    memory. Probing is what turns a crash at the first forward pass into a slower evaluation.
    """
    use_cuda = torch.cuda.is_available() and accelerator in ('auto', 'gpu', 'cuda')
    if not use_cuda:
        return accelerator, False
    try:
        torch.zeros(1, device='cuda')
    except RuntimeError as error:
        logger.warning(
            f'CUDA is reported available but unusable ({error}); the device is likely busy (held by another '
            f'process in exclusive mode, or out of memory). Falling back to CPU for evaluation. Pass '
            f'--accelerator cpu to silence this warning, or free the GPU (check `nvidia-smi`).'
        )
        return 'cpu', False
    return accelerator, True


def evaluate(
        input_path: str,
        model_path: str,
        output_path: str,
        metrics_config: str = 'config/eval/metrics.yaml',
        metrics_path: Optional[str] = None,
        report_path: Optional[str] = None,
        split: str = 'test',
        baselines: Union[str, list] = 'zero,climatology',
        accelerator: str = 'auto',
        devices: int = 1,
        num_workers: int = 8,
        batch_size: int = 16,
        save_predictions: bool = True,
        sampling_steps: Optional[int] = None,
        ensemble_size: int = 32,
        progress_bar: bool = True,
        limit_batches: Optional[float] = None,
        model_family: Optional[str] = None,
        model_type: Optional[str] = None,
        plot_dates: Optional[Union[str, list]] = None,
        residual_diagnostics: Optional[bool] = None
) -> None:
    """Score a trained checkpoint on a held-out split and write the metrics, the report and the predictions.

    Args:
        input_path: The prepared directory the model was trained on.
        model_path: The checkpoint. Its family marker selects the module class.
        output_path: Where `predictions.npz` is written.
        metrics_config: The shared metric suite. An HOURLY run needs the hourly variant, whose categorical
            thresholds are `kind: probability` — see that file's own note for why an `occurrence` entry on a
            probability field yields POD ~ 1 and FAR ~ the base rate with no error raised.
        metrics_path: Destination of the flat metrics JSON; defaults to `<output-path>/<split>_metrics.json`.
        report_path: Destination of the report directory. `None` skips reporting entirely.
        split: `valid` | `test`.
        baselines: Which trivial baselines to score against. `persistence` is REJECTED by `build_baselines` — this
            is a diagnostic mapping, not a temporal forecast, so "yesterday's field" is not a baseline the task
            admits.
        accelerator, devices, num_workers, batch_size: Runtime knobs for the prediction pass.
        save_predictions: Write the predicted/observed stacks to `predictions.npz`.
        sampling_steps: DIFFUSION only — override the number of ODE steps at evaluation time. Ignored by families
            with no `eval_sampling_steps` attribute.
        ensemble_size: Members per item for a STOCHASTIC family. ⚠️ Must be >= 2: `scores.spread_skill_sums` uses
            `ddof=1`, so a single member yields a silent NaN rather than an error. With `> 1` the probabilistic
            group is reported, the point/skill/categorical/calibration scores use the ensemble MEAN, and the
            spatial-structure scores are POOLED over all N x M members.
        progress_bar: Lightning's per-batch bar. An ensemble run is `M x sampling_steps` integrations per batch.
        limit_batches: Lightning predict-batch limit (debugging / smoke); baselines and items are truncated to match.
        model_family: Explicit family override, authoritative over the checkpoint marker. Needed only for a
            marker-less checkpoint from a source branch.
        model_type: Free-form label tagged on the MLflow run by the orchestrator; unused here, accepted so a
            pipeline may set `model-type:` on this stage (run.py forwards every parameter).
        plot_dates: Extra `YYYY-MM-DD` dates to render as per-day maps, beyond the auto-selected extreme/median
            days. A date outside the split is warned about and skipped.
        residual_diagnostics: The residual-space discrepancy diagnostics, for a RESIDUAL-mode diffusion ensemble run
            only. `None` (default) auto-enables them there and skips them elsewhere; `True` forces them (warning and
            skipping if the run is incompatible); `False` disables them.

    Returns:
        None. Writes the metrics JSON, the report directory and (optionally) `predictions.npz`.
    """
    # TF32 matmuls + cudnn autotuning (fixed input shapes); predictions stay fp32, so metrics are unaffected
    torch.set_float32_matmul_precision('high')
    accelerator, use_cuda = _resolve_accelerator(accelerator)

    abs_input_path = os.path.join(root_path, input_path)
    abs_output_path = os.path.join(root_path, output_path)
    os.makedirs(abs_output_path, exist_ok=True)

    with open(os.path.join(root_path, metrics_config)) as handle:
        metrics_spec = safe_load(handle)

    prepared_config, split_index, target_stats = load_prepared_artifacts(abs_input_path)
    dataset = LightningMapsDataset(split_index[split_index['split'] == split], prepared_config)
    if len(dataset) == 0:
        raise ValueError(f'Split "{split}" is empty in "{input_path}".')
    logger.info(f'Evaluating on {len(dataset)} items of split "{split}" (mode "{prepared_config["mode"]}").')

    # Predictions come back in the TARGET SPACE: the feature normalization and — for the residual diffusion model —
    # the upstream add-back both live inside the checkpoint. There is no back-transform anywhere, because training
    # space == evaluation space.
    module = load_model_module(
        os.path.join(root_path, model_path), map_location='cpu', model_family=model_family
    )
    if sampling_steps is not None and hasattr(module, 'eval_sampling_steps'):
        module.eval_sampling_steps = int(sampling_steps)
        logger.info(f'Diffusion evaluation: overriding the number of ODE sampling steps to {sampling_steps}.')

    # the evaluation-side occurrence event, shared by the metric suite, the baselines and — for an ensemble family —
    # the module, where it conditions the tail CRPS and the rank histogram on the occurrence cells
    occurrence_event = resolve_occurrence_event(metrics_spec, target_stats)

    # The probabilistic suite runs only for a family exposing the ensemble interface, when metrics.yaml asks for it
    # AND ensemble_size > 1. Off, the stage is the single-sample path, which is what the deterministic family gets.
    # The section is a metric GROUP so it normally sits under `metrics:`; a top-level placement is accepted too,
    # rather than silently disabling the whole suite.
    ensemble_spec = metrics_spec.get('metrics', {}).get('ensemble') or metrics_spec.get('ensemble', {})
    ensemble_on = bool(ensemble_spec) and int(ensemble_size) > 1 and hasattr(module, 'eval_ensemble_size')
    if ensemble_on:
        module.eval_ensemble_size = int(ensemble_size)
        module.eval_occurrence_event = occurrence_event
        logger.info(
            f'Ensemble evaluation: {int(ensemble_size)} members per item for the probabilistic suite '
            f'({", ".join(sorted(ensemble_spec))}); point/skill/categorical/calibration scores use the ensemble '
            f'mean, spatial-structure scores are pooled over all members.'
        )
    elif int(ensemble_size) < 2 and hasattr(module, 'eval_ensemble_size'):
        logger.warning(
            f'ensemble-size is {ensemble_size} for a stochastic family, so the probabilistic suite is OFF. Note '
            f'that a value of 1 would not work anyway: spread_skill_sums uses ddof=1 and yields a silent NaN.'
        )

    # Residual-space diagnostics: EXCLUSIVELY a residual-mode diffusion ensemble run. Setting eval_return_residual
    # makes predict_step also return the UNCLAMPED per-member residual and the upstream — the censored
    # `clamp(P) - upstream` would hide exactly the corrections that overshoot, which are what the surprise maps show.
    is_residual_diffusion = (ensemble_on and hasattr(module, 'eval_return_residual')
                             and bool(getattr(module, 'residual_target', False)))
    diagnostics_on = is_residual_diffusion if residual_diagnostics is None else bool(residual_diagnostics)
    if diagnostics_on and not is_residual_diffusion:
        logger.warning('residual_diagnostics requested but this is not a residual-mode diffusion ensemble run; '
                       'they are skipped (they need the model\'s upstream + unclamped residual).')
        diagnostics_on = False
    if diagnostics_on:
        module.eval_return_residual = True
        logger.info('Residual diagnostics ON: the discrepancy diagnostics (bias/surprise maps, histograms, QQ, '
                    'scatters, heteroscedasticity and the resid_* scalars) are computed after the metric suite.')

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda
    )
    trainer = L.Trainer(
        accelerator=accelerator, devices=devices, benchmark=use_cuda, logger=False,
        enable_progress_bar=bool(progress_bar),
        limit_predict_batches=limit_batches
    )
    outputs = trainer.predict(module, loader)
    prediction = torch.cat([batch['prediction'] for batch in outputs]).numpy()
    observation = torch.cat([batch['observation'] for batch in outputs]).numpy()
    # the occurrence forecast: the model's own output in hourly mode, None in daily mode (no occurrence head)
    probability = torch.cat([batch['probability'] for batch in outputs]).numpy() \
        if outputs[0]['probability'] is not None else None

    # Spatial-structure stacks. For an ensemble run the structure scores are POOLED over all N x M members (every
    # item-member pair as one map, obs replicated per member): texture-faithful, where scoring the ensemble MEAN
    # would measure the smoothness of an average rather than the model's texture.
    structure_member = None                         # one member kept for predictions.npz
    ensemble_members_full = None                    # the full [N, M, H, W] stack -> the stochastic report maps
    if 'ensemble_members' in outputs[0]:
        members_full = torch.cat([batch['ensemble_members'] for batch in outputs]).numpy()   # [N, M, H, W]
        n_items, n_members = members_full.shape[0], members_full.shape[1]
        grid_shape = members_full.shape[2:]
        stack_gb = members_full.nbytes / 1024 ** 3
        if stack_gb > MEMBER_STACK_WARN_GB:
            logger.warning(
                f'Pooled-ensemble spatial metrics materialize a {stack_gb:.1f} GB member stack '
                f'({n_items} items x {n_members} members) and the FFT/gradient temporaries scale with it; this is '
                f'safe at the daily scale but heavy in hourly mode — reduce ensemble-size.'
            )
        # CAVEAT: the M members of one item share its conditioning, so the N x M pooled maps are NOT independent.
        # This does not bias the pooled point estimate of any structure metric (it inflates the effective sample
        # size, which would matter for confidence intervals — none are computed here); the estimate is just noisier
        # than the raw N x M count suggests.
        prediction_structure = members_full.reshape(n_items * n_members, *grid_shape)         # [N*M, H, W]
        observation_structure = np.repeat(observation, n_members, axis=0)                     # obs paired per member
        member_choice = np.random.default_rng(0).integers(0, n_members, size=n_items)
        structure_member = members_full[np.arange(n_items), member_choice]                   # [N, H, W]
        ensemble_members_full = members_full
    else:
        prediction_structure = prediction
        observation_structure = None                                                         # -> suite uses `observation`

    # residual-mode stacks: the per-member UNCLAMPED residual [N, M, H, W] and the upstream [N, H, W]
    residual_members_full, upstream_full = None, None
    if diagnostics_on and 'ensemble_residual_members' in outputs[0]:
        residual_members_full = torch.cat([batch['ensemble_residual_members'] for batch in outputs]).numpy()
        upstream_full = torch.cat([batch['upstream'] for batch in outputs]).numpy()

    # sum the compact per-batch ensemble partials across the split. Sums are additive; means and ratios are not,
    # which is why the division happens exactly once, in finalize_ensemble_metrics.
    ensemble_partials = None
    for batch in outputs:
        if batch.get('ensemble_partials') is not None:
            ensemble_partials = merge_ensemble_partials(ensemble_partials, batch['ensemble_partials'])

    items = dataset.items_frame().iloc[:prediction.shape[0]].reset_index(drop=True)

    baseline_stacks, occurrence_probability = build_baselines(
        _as_name_list(baselines),
        metrics_spec.get('baselines', {}),
        split_index,
        items,
        prepared_config['mode'],
        int(prepared_config['hours_per_day']),
        occurrence_event=occurrence_event
    )
    flat_metrics, curves = run_metric_suite(
        metrics_spec, prediction, observation, probability,
        baseline_stacks,
        occurrence_probability[:prediction.shape[0]] if occurrence_probability is not None else None,
        target_stats,
        prediction_structure=prediction_structure,
        observation_structure=observation_structure,
        progress=progress_bar
    )

    if ensemble_partials is not None:
        ensemble_flat, ensemble_curves = finalize_ensemble_metrics(
            ensemble_partials, ensemble_spec, int(ensemble_size)
        )
        flat_metrics.update(ensemble_flat)
        curves.update(ensemble_curves)

    if residual_members_full is not None:
        logger.info('Computing residual-space diagnostics (D_pred = unclamped r, D_true = observed - upstream)...')
        residual_mean = residual_members_full.mean(axis=1)                                    # [N, H, W] D_pred
        resid_flat, resid_curves = compute_residual_diagnostics(
            observation, prediction, upstream_full, residual_mean, residual_members_full,
            occurrence_event=occurrence_event
        )
        flat_metrics.update(resid_flat)
        curves.update(resid_curves)
        logger.info(f'Residual diagnostics: added {len(resid_flat)} resid_* scalars and the residual report block.')

    finite_metrics = {
        key: float(value) for key, value in flat_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    n_dropped = len(flat_metrics) - len(finite_metrics)
    if n_dropped:
        logger.warning(f'{n_dropped} metric(s) were undefined on this split and were dropped from the JSON.')

    metrics_file = os.path.join(root_path, metrics_path) if metrics_path is not None \
        else os.path.join(abs_output_path, f'{split}_metrics.json')
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'w') as handle:
        json.dump(finite_metrics, handle, indent=2)
    logger.info(f'{len(finite_metrics)} metrics written to "{metrics_file}".')

    if save_predictions:
        np.savez_compressed(
            os.path.join(abs_output_path, 'predictions.npz'),
            prediction=prediction.astype(np.float32),       # the ensemble mean for a stochastic family
            observation=observation.astype(np.float32),
            **({'prediction_member': structure_member.astype(np.float32)} if structure_member is not None else {}),
            **({'probability': probability.astype(np.float32)} if probability is not None else {}),
            dates=items['date'].astype(str).to_numpy(),
            hours=items['hour'].to_numpy(dtype=float)
        )

    if report_path is not None:
        # Per-day maps: a DETERMINISTIC run (no members) gets the 1 x 2 observed | predicted layout; a STOCHASTIC one
        # gets the 2 x 3 observed / ensemble-mean / ensemble-std / three-members grid. The PSD and FSS curves in
        # `curves` are already the pooled-over-members structure scores.
        write_report(
            os.path.join(root_path, report_path),
            metrics_spec.get('reporting', {}),
            flat_metrics, curves, prediction, observation, items,
            ensemble_members=ensemble_members_full,
            plot_dates=_as_name_list(plot_dates) if plot_dates else None
        )

    headline = {name: finite_metrics[name] for name in HEADLINE_METRICS if name in finite_metrics}
    logger.info(f'Done: {headline}')


if __name__ == '__main__':
    Fire(evaluate)
