"""Pipeline stage: retrain the winning configuration on (possibly re-prepared) data, WITHOUT a new sweep.

Loads the best hyperparameter configuration a previous `tune` recorded (`best_trial.json` in `--source-path`) and
fits a SINGLE model with exactly that configuration on `--input-path`, through the same per-trial fit and scoring path
the sweep used — so it writes the same `best_model.ckpt` / `best_trial.json` / metrics, on fresh data and fresh
weights. Use it to bring a tuned configuration onto re-prepared data (a corrected `hourly-threshold`, a longer split)
when a full re-sweep is not worth it.

ONE stage for all three families, exactly like `tune`, and for the same reason: only the module class differs, and
`retrain_best_config` in :mod:`src.utils.modeling.tuning` holds everything else.

⛔ NO `selection-metric` / `selection-mode` parameter. The composite is read back out of `source-path/best_trial.json`,
so a retrain cannot rank on a different score than the sweep that chose the configuration. That is why
`retrain_best_config` is called with no metric argument at all — `None` means "whatever the sweep recorded is
authoritative", and a store that records no `selection_metric` raises rather than guessing.

⚠️ A STALENESS CHECK runs before the expensive fit (`tuning._check_retrain_staleness`). A STRUCTURAL mismatch between
the tuned data and the current data — a different mode, feature set, feature aggregation or residual flag — is a hard
error, because the configuration was chosen for a different problem. Distribution drift (a re-prepared target) and
code drift only warn: those are the reasons to run this stage in the first place.

⭐ **A WARM-STARTED SWEEP IS RETRAINED WARM-STARTED.** This is not optional: if the sweep replaced phase 1 with a
deterministic upstream and ran only the finetuning phase, then refitting that configuration from scratch produces a
materially different model — two phases instead of one, from random weights instead of the upstream's — and the
hyperparameters were chosen under the first regime. `tuning.retrain_best_config` says so directly at its warm-start
branch: *"the retrained model is only meaningful with the same upstream weights, so the stage must supply them to
module_factory"*. This stage is where that happens.

The upstream is resolved in one place, in this order:

1. an explicit `--upstream-model-path`, for the case where the upstream ITSELF was retrained on the new data and the
   MC-dropout model should start from the new one;
2. otherwise **whatever the source sweep recorded in `best_trial.json`** — so the shipped pipeline needs no extra
   config key at all, and a retrain cannot silently disagree with the sweep by forgetting one.

Both go through `tune._module_factory`, the same resolution the sweep uses, so the two stages cannot drift apart in
what a warm start means. A recorded upstream that no longer exists RAISES rather than quietly falling back to a
from-scratch fit, because that fallback is exactly the silent regime change this stage exists to avoid.

Usage (standalone)::

    python src/stages/retrain_best.py \\
        --model_family deterministic_unet \\
        --source_path $OUTPUT_ROOT/deterministic_unet/tuning \\
        --input_path  $OUTPUT_ROOT/deterministic_unet/prepared/daily \\
        --output_path $OUTPUT_ROOT/deterministic_unet/best
"""
import json
import logging
import os
from typing import Optional

from fire import Fire

from __init__ import root_path, console_handler
from src.utils.modeling.tuning import retrain_best_config
from tune import MODULE_FACTORIES, _module_factory               # one dispatch AND one warm-start resolution

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def _recorded_upstream(source_path: str) -> Optional[str]:
    """The upstream checkpoint the SOURCE SWEEP warm-started from, read back out of its ``best_trial.json``.

    ``None`` when the sweep fitted from scratch, and also when the file is missing or unreadable — the harness's
    ``_load_existing_best`` raises a far better message for a broken store a moment later, and duplicating that
    diagnosis here would only make the worse one arrive first.
    """
    path = os.path.join(root_path, source_path, 'best_trial.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle).get('upstream_model_path') or None
    except (ValueError, OSError):
        return None


def retrain_best(
        model_family: str,
        source_path: str,
        input_path: str,
        output_path: str,
        metrics_config: str = 'config/eval/metrics.yaml',
        model_type: str = 'model',
        upstream_model_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        max_epochs: int = 80,
        early_stopping_patience: int = 10,
        accelerator: str = 'auto',
        devices: int = 1,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        pin_memory: Optional[bool] = None,
        feature_stats_days: Optional[int] = 256,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        compile_model: bool = True,
        progress_bar: bool = True,
        diagnostics: bool = True,
        profiler: Optional[str] = None,
        seed: int = 42,
        limit_train_batches: Optional[float] = None,
        limit_val_batches: Optional[float] = None,
        staleness_max_age_days: Optional[float] = None
) -> None:
    """Retrain the best configuration from `source_path` on `input_path`; see the module docstring.

    Args:
        model_family: `deterministic_unet` | `mc_dropout` | `diffusion`. Must match the family that produced
            `source_path` — the saved trial is that family's vocabulary, so another family's module would reject it.
        source_path: A completed `tune` output directory (holding `best_trial.json`).
        input_path: The prepared directory to retrain ON, which may differ from the one that was tuned on.
        output_path: Where the retrained `best_model.ckpt` / `best_trial.json` / metrics are written.
        metrics_config: The shared metric suite, for the same selection components the sweep used.
        model_type: Free-form label tagged on the MLflow run by the orchestrator.
        upstream_model_path: MC-DROPOUT ONLY — override the upstream checkpoint to warm-start from. Leave it unset
            (the normal case, and what the shipped pipeline does) and the stage inherits whatever the source sweep
            recorded, so a warm-started sweep is retrained warm-started automatically. Set it only when the upstream
            itself was retrained and the new one should be used.
        metrics_path: Where the metrics JSON is written (auto-logged by `run.py`).
        max_epochs, early_stopping_patience, accelerator, devices, num_workers, prefetch_factor, pin_memory,
            feature_stats_days, batch_size, precision, compile_model, progress_bar, diagnostics, profiler, seed,
            limit_train_batches, limit_val_batches: The training knobs, mirroring `tune` so the fit is identical.
        staleness_max_age_days: Optional age above which the source experiment is warned about as stale.

    Returns:
        None. Writes the retrained artifacts.
    """
    if model_family not in MODULE_FACTORIES:
        raise ValueError(
            f'Unknown model family "{model_family}" (expected one of {sorted(MODULE_FACTORIES)}).'
        )

    # inherit the sweep's warm start unless explicitly overridden (see the module docstring)
    recorded = _recorded_upstream(source_path)
    upstream = upstream_model_path or recorded
    if upstream:
        absolute = upstream if os.path.isabs(upstream) else os.path.join(root_path, upstream)
        if not os.path.exists(absolute):
            raise FileNotFoundError(
                f'The upstream checkpoint "{upstream}" does not exist. It came from '
                f'{"--upstream-model-path" if upstream_model_path else f"{source_path}/best_trial.json"}, and the '
                f'source sweep only ever fine-tuned from it — retraining without it would fit a different model '
                f'(two phases from random weights) under hyperparameters chosen for one. Restore the checkpoint, or '
                f'pass --upstream-model-path explicitly.'
            )
        origin = 'the --upstream-model-path override' if upstream_model_path \
            else f'the source sweep\'s record in "{source_path}/best_trial.json"'
        logger.info(f'Retraining WARM-STARTED from "{upstream}" ({origin}).')
        if upstream_model_path and recorded and upstream_model_path != recorded:
            logger.warning(
                f'The source sweep warm-started from "{recorded}" but "{upstream_model_path}" was requested. The '
                f'retrained model starts from DIFFERENT weights than the swept one; that is a deliberate override, '
                f'so make sure the new upstream is the retrained counterpart of the old one.'
            )

    retrain_best_config(
        root_path=root_path,
        source_path=source_path,
        input_path=input_path,
        output_path=output_path,
        metrics_config=metrics_config,
        # ONE warm-start resolution, shared with the sweep, so the two stages cannot disagree about what a warm
        # start means. `upstream` is None for every from-scratch run, which yields the plain class.
        module_factory=_module_factory(model_family, upstream),
        model_type=model_type,
        metrics_path=metrics_path,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        accelerator=accelerator,
        devices=devices,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        feature_stats_days=feature_stats_days,
        batch_size=batch_size,
        precision=precision,
        compile_model=compile_model,
        progress_bar=progress_bar,
        diagnostics=diagnostics,
        profiler=profiler,
        seed=seed,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        staleness_max_age_days=staleness_max_age_days
    )


if __name__ == '__main__':
    Fire(retrain_best)
