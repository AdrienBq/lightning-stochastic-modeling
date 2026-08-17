"""Pipeline stage: the hyperparameter sweep — ONE stage for all three model families.

Branch A had `tune_distr_regression` + `tune_diffusion` and branch D had `tune_distr_regression` + `tune_mc_dropout`,
each of D's carrying its own copy of the per-trial fit. Step 3 merged all of that into
:mod:`src.utils.modeling.tuning` — including D's two-phase train→finetune fit, which `_fit_trial` now drives by asking
the module for its own `training_phases()`. What is left here is the family dispatch and the CLI.

So this stage is thin ON PURPOSE. Everything that varies between families lives in the module class; everything that
does not lives in `run_sweep`. If a change to this file needs a per-family branch, that branch almost certainly
belongs in the module.

TWO THINGS THIS STAGE OWNS, and they are the reason it is not a bare passthrough:

1. **The module factory**, chosen by `model-family`.
2. **The MC-dropout warm start** — where one `upstream-model-path` string forks into two independent uses. See below.

⚠️ `upstream-model-path` means DIFFERENT THINGS to two families, at two different stages:

* `mc_dropout` reads it HERE, for the upstream's **weights**: `MCDropoutModule.from_upstream` loads them, takes the
  architecture from the checkpoint rather than the sampled `unet` block, and marks the module warm-started so
  `training_phases()` returns `('finetune',)` alone — phase 1 already happened, and it IS the upstream checkpoint.
* `diffusion` reads it at `prepare_modeling`, for the upstream's **predictions**, materialised as per-day maps and
  appended as the last conditioning channel. Its `tune` block carries no such key, and this stage refuses one.

`run_sweep` separately passes the same string to `search.apply_constraints`, which forces `finetuning.enabled` true —
without that, a warm-started trial that sampled `false` would run no fitting phase at all. The two uses are
independent, which is why the string is threaded to both rather than being resolved once.

Stage outputs (written to `--output-path`): `best_model.ckpt`, `best_trial.json`, `trials.csv`, the optuna journal,
and the best trial's metrics JSON at `--metrics-path`.

⛔ There is deliberately NO `selection-metric` / `selection-mode` parameter. The search space's `selection:` block is
the single source of truth: `run_sweep` reads it from `--model-config` and records it into `best_trial.json`, and
`retrain_best` reads it back from there. A stage parameter would let a retrain rank on a different composite than the
sweep that chose the configuration.

Usage (standalone)::

    python src/stages/tune.py \\
        --model_family deterministic_unet \\
        --input_path $OUTPUT_ROOT/deterministic_unet/prepared/daily \\
        --output_path $OUTPUT_ROOT/deterministic_unet/tuning \\
        --model_config config/deterministic_unet/search_space.yaml \\
        --metrics_config config/eval/metrics.yaml \\
        --n_trials 60
"""
import logging
from functools import partial
from typing import Optional

from fire import Fire

from __init__ import root_path, console_handler
from src.utils.modeling.deterministic_module import DeterministicUnetModule
from src.utils.modeling.diffusion_module import DiffusionModule
from src.utils.modeling.mc_dropout_module import MCDropoutModule
from src.utils.modeling.tuning import run_sweep

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# model-family -> the LightningModule class `run_sweep` instantiates per trial. Deliberately a local table rather
# than `registry.MODULE_REGISTRY`: that one maps CHECKPOINT MARKERS (including the legacy `distr_regression` alias)
# to classes for loading, and a legacy marker is not a family a pipeline may ask to tune.
MODULE_FACTORIES = {
    'deterministic_unet': DeterministicUnetModule,
    'mc_dropout': MCDropoutModule,
    'diffusion': DiffusionModule,
}

# only this family takes an upstream CHECKPOINT at tune time (diffusion's upstream is consumed by prepare_modeling)
WARM_START_FAMILY = 'mc_dropout'


def _module_factory(model_family: str, upstream_model_path: Optional[str]):
    """Resolve `model-family` (+ an optional upstream checkpoint) into the factory `run_sweep` calls per trial.

    Returns a plain class for every family except a warm-started MC-dropout run, which returns a `partial` of the
    alternate constructor. That constructor is the ONLY way to reach `warm_started = True`, and it loads the weights
    on the line above setting it — so a module cannot claim a warm start it did not get, and cannot skip phase 1
    while randomly initialised.
    """
    if model_family not in MODULE_FACTORIES:
        raise ValueError(
            f'Unknown model family "{model_family}" (expected one of {sorted(MODULE_FACTORIES)}).'
        )
    if not upstream_model_path:
        return MODULE_FACTORIES[model_family]

    if model_family != WARM_START_FAMILY:
        raise ValueError(
            f'upstream-model-path was given for model-family "{model_family}", which does not warm-start from a '
            f'checkpoint. Only "{WARM_START_FAMILY}" reads an upstream model at tune time; diffusion consumes its '
            f'upstream at prepare_modeling instead, as the materialised conditioning channel. Move the key to the '
            f'prepare_modeling block, or drop it.'
        )
    logger.info(
        f'WARM START: loading the upstream U-net weights from "{upstream_model_path}". The architecture comes from '
        f'the CHECKPOINT (the sampled `unet` block is ignored), and only the finetuning phase runs — phase 1 already '
        f'happened, and it is that checkpoint.'
    )
    return partial(MCDropoutModule.from_upstream, upstream_model_path)


def tune(
        model_family: str,
        input_path: str,
        output_path: str,
        model_config: str,
        metrics_config: str = 'config/eval/metrics.yaml',
        model_type: str = 'model',
        n_trials: int = 60,
        sampler: str = 'tpe',
        seed: int = 42,
        max_epochs: int = 80,
        early_stopping_patience: int = 10,
        accelerator: str = 'auto',
        devices: int = 1,
        num_workers: int = 8,
        prefetch_factor: int = 4,
        pin_memory: Optional[bool] = None,
        upstream_model_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        feature_stats_days: Optional[int] = 256,
        batch_size: Optional[int] = None,
        precision: Optional[str] = None,
        compile_model: bool = True,
        pruning: bool = True,
        pruning_startup_trials: int = 10,
        pruning_warmup_epochs: int = 5,
        progress_bar: bool = True,
        diagnostics: bool = True,
        profiler: Optional[str] = None,
        restart: bool = False,
        load_existing: bool = False,
        limit_train_batches: Optional[float] = None,
        limit_val_batches: Optional[float] = None
) -> None:
    """Sweep the search space of one model family on the prepared data.

    Args:
        model_family: `deterministic_unet` | `mc_dropout` | `diffusion`. Selects the module class, and with it the
            fitting phases, the loss dispatch and the prediction contract.
        input_path: A prepared directory from `prepare_modeling`.
        output_path: Where `best_model.ckpt` / `best_trial.json` / `trials.csv` / the optuna journal are written.
        model_config: That family's `search_space.yaml`. Also the SINGLE SOURCE OF TRUTH for the selection
            composite — its `selection:` block is read here and recorded into `best_trial.json`.
        metrics_config: The shared metric suite. Supplies the climatology baselines behind the selection
            components (`mae_cond_ss_climatology`, and the Brier denominator in hourly mode).
        model_type: Free-form label tagged on the MLflow run by the orchestrator; not used by the sweep.
        n_trials, sampler, seed: Sweep size, `random` | `tpe`, and the base seed.
        max_epochs, early_stopping_patience: Per fitting phase. ⚠️ For a WARM-STARTED MC-dropout run `max_epochs` is
            ignored — phase 1 does not run, and phase 2 takes its budget from the search space's
            `finetuning.max_epochs`.
        accelerator, devices, num_workers, prefetch_factor, pin_memory, precision, compile_model: Runtime knobs.
        upstream_model_path: MC-DROPOUT ONLY — a `deterministic_unet` checkpoint to warm-start from (see the module
            docstring). An unset `{{$UPSTREAM_MODEL}}` substitutes to the empty string, which reads as "no warm
            start". Supplying it for another family raises.
        metrics_path: Where the best trial's metrics JSON is written (auto-logged to MLflow by `run.py`).
        feature_stats_days: Cap on the train days streamed for the feature normalization.
        batch_size: Overrides the search space's sampled `batch_size` (smoke tiers use this).
        pruning, pruning_startup_trials, pruning_warmup_epochs: Optuna median pruning of unpromising trials.
        progress_bar, diagnostics, profiler: Per-trial reporting.
        restart: Discard any existing optuna study and sweep from trial 0.
        load_existing: Skip the sweep entirely and reuse the best experiment already in `output_path`.
        limit_train_batches, limit_val_batches: Lightning batch limits (debugging / smoke).

    Returns:
        None. Writes the sweep artifacts listed in the module docstring.
    """
    run_sweep(
        root_path=root_path,
        input_path=input_path,
        output_path=output_path,
        model_config=model_config,
        metrics_config=metrics_config,
        module_factory=_module_factory(model_family, upstream_model_path),
        model_type=model_type,
        # per-family study name so three families sharing an output root cannot resume each other's journal
        study_name=f'tune_{model_family}',
        n_trials=n_trials,
        sampler=sampler,
        seed=seed,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        accelerator=accelerator,
        devices=devices,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        # threaded on as well as consumed above: `apply_constraints` uses it to force finetuning.enabled, which is a
        # separate obligation from loading the weights (see the module docstring)
        upstream_model_path=upstream_model_path or None,
        metrics_path=metrics_path,
        feature_stats_days=feature_stats_days,
        batch_size=batch_size,
        precision=precision,
        compile_model=compile_model,
        pruning=pruning,
        pruning_startup_trials=pruning_startup_trials,
        pruning_warmup_epochs=pruning_warmup_epochs,
        progress_bar=progress_bar,
        diagnostics=diagnostics,
        profiler=profiler,
        restart=restart,
        load_existing=load_existing,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches
    )


if __name__ == '__main__':
    Fire(tune)
