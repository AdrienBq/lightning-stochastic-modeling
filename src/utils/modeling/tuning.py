"""Reusable hyperparameter-sweep machinery shared by ALL THREE model families.

The three families differ only in the module they instantiate (and, for residual diffusion, a couple of extra
train statistics baked into the checkpoint). Everything else — the GPU runtime setup, the feature normalization,
the optuna study with median pruning, the resume/restart bookkeeping, the per-trial fit across phases, the
best-trial selection and the metrics JSON — is identical, so it lives here and every tuning stage is a thin
wrapper around :func:`run_sweep`.

The sweep is parameterized by:
- ``module_factory(trial, in_channels, target_stats, normalization) -> LightningModule``: the model family. A
  warm-started run closes over its upstream checkpoint HERE, in the stage that builds the factory — this harness
  only needs ``upstream_model_path`` to constrain the trial and to record the provenance;
- ``augment_target_stats(prepared_config, split_index, max_days, seed) -> dict | None``: optional extra
  train-statistics merged into ``target_stats`` before the modules are built (the diffusion branch uses it to
  flag residual mode);

and by the module interface used by :func:`_fit_trial`: ``training_phases``, ``set_phase``, ``monitor_metric``,
``monitor_mode`` and (optionally) ``prepare_full_validation`` (called before the final, best-checkpoint validation
so a model whose selection metric is computed only periodically still reports it for every trial).

WHERE THE SELECTION COMPOSITE COMES FROM, and why it is not a stage flag:
- the prepared data's ``mode`` decides WHICH composite (the task determines it — see
  ``validation.selection_metric_for_mode``, which raises when the search space disagrees rather than overriding);
- the search space's ``selection`` block supplies the WEIGHTS;
- :func:`run_sweep` records both into ``best_trial.json``, and :func:`retrain_best_config` reads them back rather
  than being told. A retrain that ranked on a different composite than the sweep which chose the config would be
  comparing two different quantities and reporting the difference as a data effect.
"""
import glob
import json
import logging
import math
import os
import shutil
import sys
import time
from typing import Callable, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from yaml import safe_load

from src import console_handler
from src.utils.io import lazy
from src.utils.io.data import compute_feature_stats, compute_upstream_stats, load_prepared_artifacts
from src.utils.metrics.evaluation import climatology_conditional_mae
from src.utils.modeling.dataset import MODE_HOURLY, DayGroupedShuffleSampler, build_split_datasets
from src.utils.modeling.search import apply_constraints, flatten_trial, sample_trial, suggest_trial_optuna
from src.utils.modeling.validation import DEFAULT_SELECTION_WEIGHTS, selection_metric_for_mode

try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

SAMPLERS = ('random', 'tpe')
OPTUNA_JOURNAL_FILENAME = 'optuna_journal.log'


def _prepared_mode(abs_input_path: str) -> str:
    """The preparation ``mode`` alone, read without loading the split index or target statistics.

    Needed before those artifacts are loaded, because the selection composite is resolved up front (including on
    the ``load_existing`` path, which returns before any of them are read).
    """
    with open(os.path.join(abs_input_path, 'prepared_config.json')) as handle:
        return json.load(handle)['mode']


def _journal_storage(path: str):
    """Persistent optuna storage backed by a journal file (safe on shared/network filesystems, unlike sqlite)."""
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend     # optuna >= 4
    except ImportError:
        from optuna.storages import JournalFileStorage as JournalFileBackend
    return JournalStorage(JournalFileBackend(path))


class ThroughputDiagnostics(L.Callback):
    """Per-epoch data-starvation diagnostics: training throughput, the share of wall-clock spent waiting on the
    dataloader (a high share means the GPU is starved -> raise num_workers / materialize features), the
    first-batch latency (worker spin-up + prefetch warm-up) and the epoch's peak CUDA memory.

    Timing is wall-clock around the Lightning batch hooks; with asynchronous CUDA execution the wait/compute
    attribution is approximate, but a loader that keeps up shows ~0 wait either way.
    """

    def __init__(self, prefix: str, use_cuda: bool):
        self.prefix = prefix
        self.use_cuda = use_cuda

    def on_train_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        self._wait, self._compute, self._first_wait = 0.0, 0.0, 0.0
        self._items, self._batches = 0, 0
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()
        self._mark = time.perf_counter()

    def on_train_batch_start(self, trainer, module, batch, batch_idx, *args) -> None:
        now = time.perf_counter()
        if batch_idx == 0:
            self._first_wait = now - self._mark
        else:
            self._wait += now - self._mark
        self._mark = now

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, *args) -> None:
        now = time.perf_counter()
        self._compute += now - self._mark
        self._mark = now
        self._batches += 1
        self._items += len(batch[0]) if isinstance(batch, (tuple, list)) else len(batch)

    def on_train_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        if self._batches <= 1:
            return
        elapsed = self._first_wait + self._wait + self._compute
        wait_share = self._wait / max(self._wait + self._compute, 1e-9)
        memory = f', peak mem {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB' if self.use_cuda else ''
        logger.info(
            f'{self.prefix} epoch {trainer.current_epoch + 1}: {self._items / max(elapsed, 1e-9):.1f} items/s, '
            f'data-wait {100.0 * wait_share:.0f}% of step time, '
            f'first batch {self._first_wait:.2f}s{memory}'
        )


class TrialProgressBar(TQDMProgressBar):
    """Progress bar whose description carries the sweep context (trial index, phase, epoch). On a non-TTY stdout
    (orchestrated/batch runs) every refresh is a full printed line, so the refresh rate is heavily throttled there.
    """

    def __init__(self, prefix: str, max_epochs: int):
        super().__init__(refresh_rate=1 if sys.stdout.isatty() else 50)
        self.prefix = prefix
        self.max_epochs = max_epochs

    def on_train_epoch_start(self, trainer: L.Trainer, *args) -> None:
        super().on_train_epoch_start(trainer, *args)
        self.train_progress_bar.set_description(
            f'{self.prefix} epoch {trainer.current_epoch + 1}/{self.max_epochs}'
        )


class OptunaPruningCallback(L.Callback):
    """Reports the monitored validation score to optuna after every validation epoch and aborts the trial when
    the pruner deems it hopeless (the raised ``optuna.TrialPruned`` is handled by the sweep loop). The RAW metric
    value is reported: optuna's MedianPruner reads the study's ``direction`` (set from ``selection_mode``) and
    compares accordingly, so a minimized monitor (e.g. the diffusion flow loss, in a ``minimize`` study) and a
    maximized one (the U-net composite, in a ``maximize`` study) both prune correctly without any sign flip — the
    callback is only attached when ``monitor == selection_metric``, so the reported metric IS the study objective."""

    def __init__(self, optuna_trial, monitor: str):
        self.optuna_trial = optuna_trial
        self.monitor = monitor
        self._step = 0

    def on_validation_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.optuna_trial.report(float(value), self._step)
        self._step += 1
        if self.optuna_trial.should_prune():
            raise optuna.TrialPruned(
                f'pruned at validation epoch {self._step} ({self.monitor} = {float(value):.4f})'
            )


def _fit_trial(
        module: L.LightningModule,
        trial: dict,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        trial_dir: str,
        max_epochs: int,
        early_stopping_patience: int,
        accelerator: str,
        devices: int,
        precision: str,
        use_cuda: bool,
        limit_train_batches: Optional[float],
        limit_val_batches: Optional[float],
        optuna_trial=None,
        prune_metric: Optional[str] = None,
        progress_prefix: Optional[str] = None,
        diagnostics_prefix: Optional[str] = None,
        profiler: Optional[str] = None
) -> dict:
    """Run all training phases of one trial and return the validation metrics at the best epoch.

    The monitored metric and its direction come from the module (``monitor_metric`` / ``monitor_mode``), so the
    same loop fits the U-net (maximized composite / AP / calibration metrics) and the diffusion model (minimized
    flow-matching loss). When ``optuna_trial`` and ``prune_metric`` are given, intermediate scores are reported to
    optuna and the trial may be pruned (only in phases whose monitored metric IS the selection metric).
    """
    # The phase LIST comes from the MODULE, which is the only thing that knows its own trial schema: the
    # deterministic U-net derives it from the hierarchy/calibration blocks, MC-dropout prepends train -> finetune,
    # diffusion has a single phase. Deriving it here (as this harness used to, from U-net-specific trial keys)
    # would mean every new family had to edit the shared loop. Everything BELOW is already family-agnostic —
    # `set_phase`, the module's own `monitor_metric` / `monitor_mode`, and the best-weight restore between phases.
    phases = module.training_phases()

    trainer = None
    for phase in phases:
        module.set_phase(phase)
        monitor = module.monitor_metric
        mode = module.monitor_mode
        checkpoint = ModelCheckpoint(
            dirpath=trial_dir, filename=f'best_{phase}', monitor=monitor, mode=mode, save_top_k=1
        )
        callbacks = [checkpoint, EarlyStopping(monitor=monitor, mode=mode, patience=early_stopping_patience)]
        if optuna_trial is not None and monitor == prune_metric:
            callbacks.append(OptunaPruningCallback(optuna_trial, monitor))
        if progress_prefix is not None:
            callbacks.append(TrialProgressBar(f'{progress_prefix} [{phase}]', max_epochs))
        if diagnostics_prefix is not None:
            callbacks.append(ThroughputDiagnostics(f'{diagnostics_prefix} [{phase}]', use_cuda))
        trainer = L.Trainer(
            profiler=profiler,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            benchmark=use_cuda,                     # fixed input shapes: let cudnn autotune the conv kernels
            gradient_clip_val=float(trial['optimizer'].get('gradient_clip_val', 0.0)) or None,
            callbacks=callbacks,
            logger=CSVLogger(trial_dir, name=f'logs_{phase}'),
            enable_progress_bar=progress_prefix is not None,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=10
        )
        trainer.fit(module, train_loader, valid_loader)

        # restore the best weights of this phase before the next phase / the final validation
        if checkpoint.best_model_path:
            try:
                state = torch.load(checkpoint.best_model_path, map_location='cpu', weights_only=False)
            except TypeError:
                state = torch.load(checkpoint.best_model_path, map_location='cpu')
            module.load_state_dict(state['state_dict'])

    # let a module compute its (expensive) selection metric on the restored best state before the final
    # validation (e.g. the diffusion model samples the ODE only here, not every validation epoch)
    if hasattr(module, 'prepare_full_validation'):
        module.prepare_full_validation()

    # recompute the validation metrics at the restored best state and persist that exact state as the final ckpt
    results = trainer.validate(module, valid_loader, verbose=False)
    trainer.save_checkpoint(os.path.join(trial_dir, 'final.ckpt'))
    return dict(results[0]) if results else {}


def _write_best_metrics(
        metrics_path: Optional[str], best_metrics: dict, selection_metric: str, best_score: float, root_path: str
) -> None:
    """Write the best-trial metrics JSON (flat name -> float) at ``metrics_path`` (auto-logged by run.py)."""
    if metrics_path is None:
        return
    abs_metrics_path = os.path.join(root_path, metrics_path)
    os.makedirs(os.path.dirname(abs_metrics_path), exist_ok=True)
    payload = {
        key: float(value) for key, value in best_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    payload[selection_metric] = best_score
    with open(abs_metrics_path, 'w') as handle:
        json.dump(payload, handle, indent=2)


def _load_existing_best(abs_output_path: str, output_path: str, selection_metric: Optional[str]) -> dict:
    """Load the best experiment persisted by a previous sweep in the output path, skipping the sweep entirely.

    Raises:
        FileNotFoundError: no experiment store in the output path.
        RuntimeError: a store exists but holds no usable best experiment (missing/mismatched best_trial.json or
            missing best_model.ckpt).
    """
    best_trial_path = os.path.join(abs_output_path, 'best_trial.json')
    best_checkpoint = os.path.join(abs_output_path, 'best_model.ckpt')
    if not os.path.exists(best_trial_path) and not os.path.exists(best_checkpoint):
        raise FileNotFoundError(
            f'load_existing: no experiment store in "{output_path}" (neither best_trial.json nor '
            f'best_model.ckpt found). Run the sweep first, or disable load_existing.'
        )
    if not os.path.exists(best_trial_path):
        raise RuntimeError(
            f'load_existing: the experiment store in "{output_path}" has no best experiment '
            f'(best_trial.json is missing).'
        )
    with open(best_trial_path) as handle:
        saved = json.load(handle)
    # `selection_metric=None` means "whatever the sweep recorded is authoritative" -- the retrain path, which reads
    # the composite back out rather than being told which one to expect.
    if selection_metric is not None and saved.get('selection_metric') != selection_metric:
        raise RuntimeError(
            f'load_existing: the best experiment in "{output_path}" was selected on '
            f'"{saved.get("selection_metric")}", not the requested "{selection_metric}".'
        )
    if saved.get('selection_metric') is None:
        raise RuntimeError(
            f'The best experiment in "{output_path}" records no selection_metric, so there is no way to know '
            f'which composite it was chosen on. Re-run the sweep.'
        )
    if not os.path.exists(best_checkpoint):
        raise RuntimeError(
            f'load_existing: the experiment store in "{output_path}" has a best_trial.json but no '
            f'best_model.ckpt to go with it.'
        )
    return saved


def run_sweep(
        *,
        root_path: str,
        input_path: str,
        output_path: str,
        model_config: str,
        metrics_config: str,
        module_factory: Callable,
        model_type: str,
        study_name: str,
        n_trials: int,
        sampler: str,
        seed: int,
        max_epochs: int,
        early_stopping_patience: int,
        accelerator: str,
        devices: int,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: Optional[bool] = None,
        upstream_model_path: Optional[str] = None,
        metrics_path: Optional[str],
        feature_stats_days: Optional[int],
        batch_size: Optional[int],
        precision: Optional[str],
        compile_model: bool,
        pruning: bool,
        pruning_startup_trials: int,
        pruning_warmup_epochs: int,
        progress_bar: bool,
        diagnostics: bool,
        profiler: Optional[str],
        restart: bool,
        load_existing: bool,
        limit_train_batches: Optional[float],
        limit_val_batches: Optional[float],
        augment_target_stats: Optional[Callable] = None
) -> None:
    """Run a hyperparameter sweep for one regression model family (see the module docstring for the contract).

    The argument set mirrors the tuning-stage CLI; ``module_factory`` selects the model family and
    ``augment_target_stats`` optionally injects extra train statistics into ``target_stats`` (baked into every
    checkpoint). Writes ``best_model.ckpt``, ``best_trial.json``, ``trials.csv`` and the metrics JSON.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f'Unknown sampler "{sampler}" (expected one of {SAMPLERS}).')
    if load_existing and restart:
        raise ValueError('load_existing and restart are mutually exclusive.')

    # The selection composite is resolved HERE, before anything else, from two sources that must agree: the
    # prepared data's `mode` (authoritative -- the task determines the composite) and the search space's
    # `selection` block (which supplies the WEIGHTS and declares which composite it was written for). Resolving it
    # up front means the load-existing path below compares against the same metric name a fresh sweep would use.
    with open(os.path.join(root_path, model_config)) as handle:
        search_space = safe_load(handle)
    selection = search_space.get('selection', {})
    prepared_mode = _prepared_mode(os.path.join(root_path, input_path))
    selection_metric = selection_metric_for_mode(prepared_mode, selection.get('metric'))
    selection_mode = selection.get('mode', 'max')
    selection_weights = selection.get('components') or DEFAULT_SELECTION_WEIGHTS[selection_metric]
    logger.info(
        f'Selection: "{selection_metric}" ({selection_mode}) = '
        + ' + '.join(f'{weight:g} * {name}' for name, weight in selection_weights.items())
        + f'  [mode "{prepared_mode}"]'
    )

    # load-only path: extract the best experiment of a previous sweep without running one
    if load_existing:
        abs_output_path = os.path.join(root_path, output_path)
        saved = _load_existing_best(abs_output_path, output_path, selection_metric)
        best_score, best_metrics = saved['score'], saved.get('metrics', {})
        logger.info(
            f'load_existing: loaded the best experiment from "{output_path}" '
            f'({selection_metric} = {best_score:.4f}); skipping the sweep.'
        )
        _write_best_metrics(metrics_path, best_metrics, selection_metric, best_score, root_path)
        logger.info(f'Done: best {selection_metric} = {best_score:.4f}; model at "{output_path}/best_model.ckpt".')
        return

    # GPU runtime: TF32 matmuls (model selection is done in fp32 on CPU, so reduced precision cannot flip ranks)
    torch.set_float32_matmul_precision('high')
    use_cuda = torch.cuda.is_available() and accelerator in ('auto', 'gpu', 'cuda')
    if use_cuda:
        try:                                    # fail fast: "available" does not mean usable
            torch.zeros(1, device='cuda')
        except RuntimeError as error:
            raise RuntimeError(
                f'CUDA is reported available but unusable: {error}\n'
                'Check `nvidia-smi` on the node (stale process on an exclusive-mode GPU? missing GPU '
                'allocation in the job?) or pass --accelerator cpu to run without it.'
            ) from error
    if compile_model and use_cuda and hasattr(torch, 'compile'):
        try:                                    # probe inductor/triton once
            probe = torch.compile(lambda t: t * 2.0 + 1.0)
            probe(torch.ones(8, device='cuda'))
        except Exception as error:
            logger.warning(
                f'torch.compile probe failed on this node; disabling compilation for the sweep '
                f'(typical cause: GPU driver older than the CUDA toolkit triton targets). Error: {error}'
            )
            compile_model = False
    if compile_model and use_cuda:
        # best-effort compilation: fall back to eager for any graph inductor cannot codegen instead of failing
        # the whole trial. Some sampled architectures / the ragged last batch (a dynamic shape) trip inductor
        # bugs (e.g. slice_scatter under symbolic sizes); without this such a trial dies and is lost from the
        # sweep. Eager is the numerical reference, so trial results and rankings are unaffected, and the common
        # full-batch graph still compiles and gets the speedup — only the offending graph runs eager.
        import torch._dynamo as _dynamo                 # alias: `import torch._dynamo` would rebind the
        _dynamo.config.suppress_errors = True            # module-global `torch` as a function-local (UnboundLocalError)
    if precision is None:
        precision = 'bf16-mixed' if use_cuda else '32-true'
    logger.info(f'Runtime: cuda={use_cuda}, precision={precision}, compile={compile_model and use_cuda}'
                f'{" (best-effort: eager fallback on codegen errors)" if compile_model and use_cuda else ""}.')

    abs_input_path = os.path.join(root_path, input_path)
    abs_output_path = os.path.join(root_path, output_path)
    os.makedirs(abs_output_path, exist_ok=True)

    if restart:
        logger.info(f'Restart requested: discarding the saved sweep state in "{output_path}".')
        shutil.rmtree(os.path.join(abs_output_path, 'trials'), ignore_errors=True)
        for path in [
            os.path.join(abs_output_path, name) for name in ('trials.csv', 'best_trial.json', 'best_model.ckpt')
        ] + glob.glob(os.path.join(abs_output_path, f'{OPTUNA_JOURNAL_FILENAME}*')):
            if os.path.exists(path):
                os.remove(path)

    # resume: keep completed/pruned trials of an interrupted sweep, retry the failed ones
    trials_csv_path = os.path.join(abs_output_path, 'trials.csv')
    trial_rows, done_trials = [], set()
    if os.path.exists(trials_csv_path):
        previous = pd.read_csv(trials_csv_path)
        if 'status' not in previous.columns:
            previous['status'] = 'completed'
        kept = previous[previous['status'] != 'failed']
        trial_rows = kept.to_dict('records')
        done_trials = set(kept['trial'].astype(int).tolist())
        logger.info(
            f'Resuming the sweep: keeping {len(done_trials)} recorded trial(s), retrying '
            f'{len(previous) - len(kept)} failed one(s); pass restart=true to start from scratch.'
        )

    prepared_config, split_index, target_stats = load_prepared_artifacts(abs_input_path)
    # record the (ordered) feature provenance in target_stats so it is baked into every checkpoint; a downstream
    # residual-diffusion preparation can then verify the conditioning channel set/order matches when it reuses
    # this model upstream (the per-channel normalization is positional, so a feature reorder must be caught)
    target_stats = {
        **target_stats,
        'features': list(prepared_config['features']),
        'feature_aggregation': prepared_config.get('feature_aggregation')
    }
    datasets = build_split_datasets(split_index, prepared_config, splits=['train', 'valid'])
    in_channels = datasets['train'].in_channels
    logger.info(
        f'Tuning "{model_type}" on {len(datasets["train"])} train / {len(datasets["valid"])} valid items '
        f'({in_channels} input channels, mode "{prepared_config["mode"]}").'
    )

    # feature normalization on the train split, expanded to per-channel buffers and baked into every checkpoint
    train_rows = split_index[split_index['split'] == 'train']
    feature_stats = compute_feature_stats(
        train_rows,
        prepared_config['variable_names'],
        prepared_config['features'],
        max_days=feature_stats_days,
        seed=seed,
        feature_layout=prepared_config.get('feature_layout') or 'time_major'
    )
    channel_names = datasets['train'].channel_variable_names
    if 'upstream' in channel_names:             # residual mode: standardize the appended upstream channel too
        upstream_stats = compute_upstream_stats(train_rows, max_days=feature_stats_days, seed=seed)
        feature_stats['mean']['upstream'] = upstream_stats['mean']
        feature_stats['std']['upstream'] = upstream_stats['std']
    with open(os.path.join(abs_output_path, 'feature_stats.json'), 'w') as handle:
        json.dump(feature_stats, handle, indent=2)
    normalization = {
        'mean': [feature_stats['mean'][name] for name in channel_names],
        'std': [feature_stats['std'][name] for name in channel_names]
    }
    logger.info(
        f'Feature normalization fitted on the train split: '
        f'{ {name: round(feature_stats["mean"][name], 3) for name in prepared_config["features"]} } (means).'
    )

    # model-independent denominator of the mae_cond_ss_climatology selection component (the climatology
    # baseline's conditional MAE on the valid occurrence cells): computed once and injected into every trial's
    # module, so the composite valid_tail_score's skill term is normalized identically across trials
    with open(os.path.join(root_path, metrics_config)) as handle:
        metrics_spec = safe_load(handle)
    selection_climatology_cond_mae, selection_occurrence_event = climatology_conditional_mae(
        split_index, datasets['valid'].items_frame(), prepared_config, metrics_spec, target_stats
    )
    logger.info(
        f'Selection denominator: climatology conditional-MAE = {selection_climatology_cond_mae:.4f} on the valid '
        f'split (occurrence value={selection_occurrence_event[0]:g}, strict={selection_occurrence_event[1]}); '
        f'valid_tail_score includes mae_cond_ss_climatology = 1 - mae_cond(model) / this.'
    )

    # optional model-family-specific train statistics merged into target_stats (and persisted for provenance)
    if augment_target_stats is not None:
        extra_stats = augment_target_stats(prepared_config, split_index, feature_stats_days, seed)
        if extra_stats:
            target_stats = {**target_stats, **extra_stats}
            with open(os.path.join(abs_output_path, 'augmented_target_stats.json'), 'w') as handle:
                json.dump(extra_stats, handle, indent=2)
            logger.info(f'Augmented target statistics for this model family: {extra_stats}.')

    # optional TPE sampling (and trial pruning) through optuna
    study = None
    if sampler == 'tpe':
        if optuna is None:
            logger.warning('optuna is not installed; falling back to random sampling (and no pruning).')
            sampler = 'random'
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=pruning_startup_trials, n_warmup_steps=pruning_warmup_epochs
            ) if pruning else optuna.pruners.NopPruner()
            study = optuna.create_study(
                study_name=study_name,
                storage=_journal_storage(os.path.join(abs_output_path, OPTUNA_JOURNAL_FILENAME)),
                load_if_exists=True,
                direction='maximize' if selection_mode == 'max' else 'minimize',
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=pruner
            )
            for stale in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
                study.tell(stale.number, state=optuna.trial.TrialState.FAIL)

    # without materialized feature files, shuffled hourly-mode training reloads one full day per item;
    # day-grouped shuffling keeps the per-worker cache effective
    train_dataset = datasets['train']
    use_day_grouped_sampler = train_dataset.mode == MODE_HOURLY and not train_dataset.uses_materialized_features
    if use_day_grouped_sampler:
        logger.warning(
            'No materialized feature files in the prepared data: falling back to day-grouped shuffling. '
            'Re-run the preparation stage (materialize_features=true) for the fast memory-mapped path.'
        )

    sign = 1.0 if selection_mode == 'max' else -1.0
    best_score, best_checkpoint, best_trial, best_metrics = -math.inf, None, None, {}
    best_trial_path = os.path.join(abs_output_path, 'best_trial.json')
    if os.path.exists(best_trial_path):                            # restore the resumed sweep's best state
        with open(best_trial_path) as handle:
            saved = json.load(handle)
        if saved.get('selection_metric') == selection_metric:
            best_score, best_trial = saved['score'], saved['trial']
            best_metrics = saved.get('metrics', {})
            saved_checkpoint = os.path.join(abs_output_path, 'best_model.ckpt')
            best_checkpoint = saved_checkpoint if os.path.exists(saved_checkpoint) else None
            logger.info(f'Restored the saved best trial: {selection_metric} = {best_score:.4f}.')
        else:
            logger.warning('The saved best_trial.json uses a different selection metric; ignoring it.')

    try:
        for trial_index in range(int(n_trials)):
            if trial_index in done_trials:
                continue
            rng = np.random.default_rng(seed + trial_index)
            L.seed_everything(seed + trial_index, workers=True)

            if study is not None:
                optuna_trial = study.ask()
                trial = suggest_trial_optuna(search_space, optuna_trial)
            else:
                optuna_trial = None
                trial = sample_trial(search_space, rng)
            trial = apply_constraints(trial, upstream_model_path=upstream_model_path)

            trial_batch_size = int(batch_size or trial['optimizer']['batch_size'])
            loader_kwargs = dict(
                num_workers=num_workers, persistent_workers=num_workers > 0,
                pin_memory=use_cuda if pin_memory is None else pin_memory
            )
            if num_workers > 0:                                      # a deeper prefetch buffer bridges the periodic
                loader_kwargs['prefetch_factor'] = prefetch_factor   # GPU-idle from synchronized worker delivery:
                # uniform-cost items make all workers finish in lock-step, so batches arrive in bursts of num_workers
                # and the GPU drains each burst before the next lands; the deeper buffer hides that inter-burst gap
            sampling = {'sampler': DayGroupedShuffleSampler(train_dataset)} if use_day_grouped_sampler \
                else {'shuffle': True}
            train_loader = DataLoader(train_dataset, batch_size=trial_batch_size, **sampling, **loader_kwargs)
            valid_loader = DataLoader(
                datasets['valid'], batch_size=2 * trial_batch_size, shuffle=False, **loader_kwargs
            )

            trial_dir = os.path.join(abs_output_path, 'trials', f'trial_{trial_index:03d}')
            os.makedirs(trial_dir, exist_ok=True)
            logger.info(f'--- Trial {trial_index + 1}/{n_trials}')

            status = 'completed'
            try:
                module = module_factory(trial, in_channels, target_stats, normalization)
                module.valid_climatology_cond_mae = selection_climatology_cond_mae
                module.selection_occurrence_event = selection_occurrence_event
                if compile_model and use_cuda:
                    if hasattr(module.net, 'compile'):
                        module.net.compile()            # in place: state-dict keys (and checkpoints) are unchanged
                    else:
                        logger.warning('torch.nn.Module.compile is unavailable in this torch build; skipping.')
                metrics = _fit_trial(
                    module, trial, train_loader, valid_loader, trial_dir,
                    max_epochs, early_stopping_patience, accelerator, devices, precision, use_cuda,
                    limit_train_batches, limit_val_batches,
                    optuna_trial=optuna_trial if pruning else None, prune_metric=selection_metric,
                    progress_prefix=f'trial {trial_index + 1}/{n_trials}' if progress_bar else None,
                    diagnostics_prefix=f'trial {trial_index + 1}/{n_trials}' if diagnostics else None,
                    profiler=profiler
                )
                score = float(metrics.get(selection_metric, float('nan')))
            except Exception as error:                  # a single broken trial must not lose the whole sweep
                if optuna is not None and isinstance(error, optuna.TrialPruned):
                    status = 'pruned'
                    logger.info(f'Trial {trial_index} {error}.')
                else:
                    status = 'failed'
                    logger.warning(f'Trial {trial_index} failed: {error}')
                metrics, score = {}, float('nan')

            if study is not None:
                if status == 'pruned':
                    study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
                else:
                    study.tell(optuna_trial, score if math.isfinite(score) else -math.inf * sign)

            trial_rows.append({
                'trial': trial_index, 'status': status, selection_metric: score,
                **{key: value for key, value in metrics.items() if key != selection_metric},
                **flatten_trial(trial)
            })
            pd.DataFrame(trial_rows).to_csv(os.path.join(abs_output_path, 'trials.csv'), index=False)

            is_better = math.isfinite(score) and (
                not math.isfinite(best_score) or sign * score > sign * best_score
            )
            if is_better:
                best_score, best_trial, best_metrics = score, trial, metrics
                best_checkpoint = os.path.join(abs_output_path, 'best_model.ckpt')
                shutil.copyfile(os.path.join(trial_dir, 'final.ckpt'), best_checkpoint)
                with open(best_trial_path, 'w') as handle:
                    json.dump({
                        'score': best_score, 'selection_metric': selection_metric,
                        'selection_mode': selection_mode, 'selection_weights': selection_weights,
                        'upstream_model_path': upstream_model_path,
                        'trial': best_trial, 'metrics': best_metrics,
                        # deterministic provenance: the repo code state when this best config was tuned, so a later
                        # retrain_best_config can warn when the model/training code has changed since (staleness)
                        'code_state': lazy.code_state_hash(root_path)
                    }, handle, indent=2)
                logger.info(f'New best trial: {selection_metric} = {score:.4f}.')
    except KeyboardInterrupt:                        # Ctrl+C: stop the sweep early, but keep the progress so far
        if best_trial is None:                       # nothing salvageable yet -> let the shutdown abort the pipeline
            logger.warning('Interrupted before any trial completed; propagating the shutdown signal.')
            raise
        logger.warning(
            f'Interrupted; stopping the sweep early and proceeding with the best trial so far '
            f'({selection_metric} = {best_score:.4f}).'
        )

    if best_trial is None:
        raise RuntimeError('All trials failed; no model was selected.')

    _write_best_metrics(metrics_path, best_metrics, selection_metric, best_score, root_path)
    logger.info(f'Done: best {selection_metric} = {best_score:.4f}; model at "{output_path}/best_model.ckpt".')


# =====================================================================================================================
# Retrain a tuned best config on (possibly corrected) data WITHOUT a sweep
# =====================================================================================================================
# Structural target_stats fields: if any differ between the tuned data and the current data the saved trial is
# INCOMPATIBLE (the network / conditioning channels / target space would not match) -> hard error, not a warning.
_STRUCTURAL_KEYS = ('mode', 'residual_target', 'features', 'feature_aggregation')
# Distribution fields: a difference is expected when retraining on corrected data and only WARNS (the tuned
# hyperparameters may no longer be optimal, but the config is still trainable).
_DISTRIBUTION_KEYS = ('hourly_threshold', 'zero_proportion', 'positive_mean')


def _load_checkpoint_hparams(checkpoint_path: str) -> dict:
    """Read a Lightning checkpoint's ``hyper_parameters`` (trial / in_channels / target_stats / normalization)."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return dict(checkpoint.get('hyper_parameters', {})) if isinstance(checkpoint, dict) else {}


def _check_retrain_staleness(source_path: str, abs_source_path: str, saved_best: dict,
                             current_target_stats: dict, current_in_channels: int, current_code_state: str,
                             staleness_max_age_days: Optional[float]) -> None:
    """Compare the source tuning experiment to the CURRENT data + code and report staleness.

    Metadata exploited (all available from the source tuning directory):
      * the source ``best_model.ckpt`` baked ``target_stats`` + ``in_channels`` -> data/structure drift (this is
        the signal that catches a re-prepared target, e.g. a changed ``hourly_threshold``);
      * the ``code_state`` recorded in ``best_trial.json`` -> model/training code drift since tuning (absent on
        experiments tuned before this field existed -> reported as "unknown");
      * the ``best_model.ckpt`` file mtime -> the experiment's age.

    Raises ``ValueError`` on a STRUCTURAL incompatibility (the saved config cannot be retrained on this data);
    otherwise only logs warnings/info so the retrain proceeds.
    """
    source_checkpoint = os.path.join(abs_source_path, 'best_model.ckpt')
    source_stats, source_in_channels = {}, None
    if os.path.exists(source_checkpoint):
        hparams = _load_checkpoint_hparams(source_checkpoint)
        source_stats = hparams.get('target_stats', {}) or {}
        source_in_channels = hparams.get('in_channels')

    # --- structural compatibility (hard errors): the saved config would not match this data ---
    structural_mismatch = []
    if source_in_channels is not None and int(source_in_channels) != int(current_in_channels):
        structural_mismatch.append(f'in_channels {source_in_channels} -> {current_in_channels}')
    for key in _STRUCTURAL_KEYS:
        if key in source_stats and source_stats.get(key) != current_target_stats.get(key):
            structural_mismatch.append(f'{key} {source_stats.get(key)!r} -> {current_target_stats.get(key)!r}')
    if structural_mismatch:
        raise ValueError(
            f'The best config in "{source_path}" was tuned on STRUCTURALLY different data and cannot be retrained '
            f'on the current input ({"; ".join(structural_mismatch)}). Re-run the full sweep on the current data, '
            f'or point at a compatible tuning directory.'
        )

    # --- distribution drift (warnings): the config is trainable but its optimum may have shifted ---
    drift = []
    for key in _DISTRIBUTION_KEYS:
        if key in source_stats and key in current_target_stats and source_stats[key] != current_target_stats[key]:
            drift.append(f'{key}: {source_stats[key]} -> {current_target_stats[key]}')
    # positive-target quantiles (the exceedance thresholds) shifting is the clearest data-drift fingerprint
    src_q, cur_q = source_stats.get('positive_quantiles'), current_target_stats.get('positive_quantiles')
    if isinstance(src_q, dict) and isinstance(cur_q, dict) and src_q != cur_q:
        drift.append(f'positive_quantiles: {src_q} -> {cur_q}')
    if drift:
        logger.warning(
            '=' * 100 + '\n'
            f'!!! STALE TRIALS (data drift): the best config from "{source_path}" was tuned on data whose target '
            f'statistics differ from the current input:\n    ' + '\n    '.join(drift) + '\n'
            f'Retraining it here is exactly the intended use (e.g. a corrected hourly_threshold), but the tuned '
            f'hyperparameters were optimal for the OLD target distribution and may be sub-optimal now — run a fresh '
            f'sweep if you have the time.\n' + '=' * 100
        )

    # --- code drift (warning): the model/training code changed since tuning ---
    source_code_state = saved_best.get('code_state')
    if source_code_state is None:
        logger.warning(
            'STALE-CHECK: the source experiment predates code-state tracking (no `code_state` in best_trial.json), '
            'so it cannot be verified that the model/training code is unchanged since tuning. If the model or '
            'search-space code has changed, the saved hyperparameters may be invalid or no longer optimal.'
        )
    elif source_code_state != current_code_state:
        logger.warning(
            '=' * 100 + '\n'
            f'!!! STALE TRIALS (code drift): the repo code state changed since this config was tuned '
            f'(tuned at {source_code_state[:12]}..., now {current_code_state[:12]}...). The model architecture, '
            f'training loop or search space may have changed — the saved hyperparameters may be invalid or '
            f'sub-optimal. Verify the config still trains as intended.\n' + '=' * 100
        )

    # --- age (info, optional hard cap) ---
    if os.path.exists(source_checkpoint):
        age_days = max(0.0, (time.time() - os.path.getmtime(source_checkpoint)) / 86400.0)
        logger.info(f'Source experiment age: {age_days:.1f} day(s) (best_model.ckpt mtime).')
        if staleness_max_age_days is not None and age_days > float(staleness_max_age_days):
            logger.warning(
                f'STALE TRIALS (age): the source experiment is {age_days:.1f} days old (> '
                f'{float(staleness_max_age_days):.1f} day cap).'
            )


def retrain_best_config(
        *,
        root_path: str,
        source_path: str,
        input_path: str,
        output_path: str,
        metrics_config: str,
        module_factory: Callable,
        model_type: str,
        augment_target_stats: Optional[Callable] = None,
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
    """Retrain the BEST tuned config of a previous sweep on the current (possibly corrected) data, WITHOUT a sweep.

    Loads ``best_trial.json`` from ``source_path`` (the tuned config + its selection metric) and trains a single
    model with exactly that configuration on ``input_path``, through the same per-trial fit + scoring path the
    sweep uses (:func:`_fit_trial`), so the reported metrics, the saved ``best_model.ckpt`` and the new
    ``best_trial.json`` are produced identically — just on fresh data and weights. Before training it runs a
    staleness check (see :func:`_check_retrain_staleness`): a STRUCTURAL mismatch (different mode / target / feature
    set / residual mode / channel count) is a hard error; distribution drift (e.g. a re-prepared ``hourly_threshold``)
    and code drift only warn. This is the quick "retrain my tuned best config on the corrected data" path when a
    full re-sweep is too expensive.

    Args mirror the tuning stage's training knobs; ``module_factory`` / ``augment_target_stats`` select the model
    family exactly as in :func:`run_sweep`. ``source_path`` may equal ``output_path`` (retrain in place).
    """
    abs_source_path = os.path.join(root_path, source_path)
    abs_input_path = os.path.join(root_path, input_path)
    abs_output_path = os.path.join(root_path, output_path)
    os.makedirs(abs_output_path, exist_ok=True)

    # 1) load the saved best config (the hyperparameters to retrain). The SELECTION comes from the same file: the
    # sweep recorded which composite it ranked on, so a retrain cannot silently score the chosen config against a
    # different one. Taking it from a CLI flag or re-reading the search space would reintroduce exactly that gap.
    saved_best = _load_existing_best(abs_source_path, source_path, selection_metric=None)
    trial = saved_best['trial']
    selection_metric = saved_best['selection_metric']
    selection_mode = saved_best.get('selection_mode', 'max')
    logger.info(
        f'Retraining the best config from "{source_path}" (tuned {selection_metric} = {saved_best["score"]:.4f}) '
        f'on the data in "{input_path}".'
    )
    # a warm-started sweep produced weights that were only ever fine-tuned, so a retrain must warm-start too
    if saved_best.get('upstream_model_path'):
        logger.info(
            f'The source sweep warm-started from "{saved_best["upstream_model_path"]}"; the retrained model is '
            f'only meaningful with the same upstream weights, so the stage must supply them to module_factory.'
        )

    # GPU runtime (mirrors run_sweep): TF32 matmuls; selection metrics stay fp32 on CPU
    torch.set_float32_matmul_precision('high')
    use_cuda = torch.cuda.is_available() and accelerator in ('auto', 'gpu', 'cuda')
    if use_cuda:
        try:
            torch.zeros(1, device='cuda')
        except RuntimeError as error:
            raise RuntimeError(f'CUDA reported available but unusable: {error}; pass --accelerator cpu.') from error
    # torch.compile robustness, mirroring run_sweep: probe inductor/triton once and disable compilation if it
    # cannot codegen on this node (driver older than the CUDA toolkit), and suppress codegen errors so any graph
    # inductor cannot lower (the sampled arch / the ragged last batch) falls back to eager instead of crashing the
    # one expensive retrain. Eager is the numerical reference, so results are unaffected when it does compile.
    if compile_model and use_cuda and hasattr(torch, 'compile'):
        try:
            probe = torch.compile(lambda t: t * 2.0 + 1.0)
            probe(torch.ones(8, device='cuda'))
        except Exception as error:
            logger.warning(f'torch.compile probe failed on this node; disabling compilation for the retrain. {error}')
            compile_model = False
    if compile_model and use_cuda:
        # best-effort compilation: fall back to eager for any graph inductor cannot codegen instead of failing
        # the whole trial. Some sampled architectures / the ragged last batch (a dynamic shape) trip inductor
        # bugs (e.g. slice_scatter under symbolic sizes); without this such a trial dies and is lost from the
        # sweep. Eager is the numerical reference, so trial results and rankings are unaffected, and the common
        # full-batch graph still compiles and gets the speedup — only the offending graph runs eager.
        import torch._dynamo as _dynamo                 # alias: `import torch._dynamo` would rebind the
        _dynamo.config.suppress_errors = True            # module-global `torch` as a function-local (UnboundLocalError)
    if precision is None:
        precision = 'bf16-mixed' if use_cuda else '32-true'

    # 2) build the training context on the CURRENT data — identical to run_sweep's setup (shared helpers)
    prepared_config, split_index, target_stats = load_prepared_artifacts(abs_input_path)
    target_stats = {
        **target_stats,
        'features': list(prepared_config['features']),
        'feature_aggregation': prepared_config.get('feature_aggregation'),
        # residual_target lives in prepared_config, NOT target_stats.json; carry it here so the staleness check's
        # structural comparison matches the source checkpoint (which bakes it) and the diffusion augment below
        'residual_target': bool(prepared_config.get('residual_target', False))
    }
    datasets = build_split_datasets(split_index, prepared_config, splits=['train', 'valid'])
    in_channels = datasets['train'].in_channels

    # 3) staleness check (against the source experiment) BEFORE the expensive training
    _check_retrain_staleness(
        source_path, abs_source_path, saved_best, target_stats, in_channels,
        lazy.code_state_hash(root_path), staleness_max_age_days
    )

    # feature normalization on the train split (+ the appended upstream channel in residual mode), baked into the ckpt
    train_rows = split_index[split_index['split'] == 'train']
    feature_stats = compute_feature_stats(
        train_rows, prepared_config['variable_names'], prepared_config['features'],
        max_days=feature_stats_days, seed=seed,
        feature_layout=prepared_config.get('feature_layout') or 'time_major'
    )
    channel_names = datasets['train'].channel_variable_names
    if 'upstream' in channel_names:
        upstream_stats = compute_upstream_stats(train_rows, max_days=feature_stats_days, seed=seed)
        feature_stats['mean']['upstream'] = upstream_stats['mean']
        feature_stats['std']['upstream'] = upstream_stats['std']
    with open(os.path.join(abs_output_path, 'feature_stats.json'), 'w') as handle:
        json.dump(feature_stats, handle, indent=2)
    normalization = {
        'mean': [feature_stats['mean'][name] for name in channel_names],
        'std': [feature_stats['std'][name] for name in channel_names]
    }

    # selection-score denominator (climatology conditional MAE on the valid occurrence cells), as in the sweep
    with open(os.path.join(root_path, metrics_config)) as handle:
        metrics_spec = safe_load(handle)
    selection_climatology_cond_mae, selection_occurrence_event = climatology_conditional_mae(
        split_index, datasets['valid'].items_frame(), prepared_config, metrics_spec, target_stats
    )

    # model-family-specific train statistics (e.g. the diffusion generation-space transform on the current data)
    if augment_target_stats is not None:
        extra_stats = augment_target_stats(prepared_config, split_index, feature_stats_days, seed)
        if extra_stats:
            target_stats = {**target_stats, **extra_stats}
            with open(os.path.join(abs_output_path, 'augmented_target_stats.json'), 'w') as handle:
                json.dump(extra_stats, handle, indent=2)

    # 4) loaders + single fixed-config fit, exactly like one sweep trial
    trial_batch_size = int(batch_size or trial['optimizer']['batch_size'])
    loader_kwargs = dict(num_workers=num_workers, persistent_workers=num_workers > 0,
                         pin_memory=use_cuda if pin_memory is None else pin_memory)
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    train_dataset = datasets['train']
    use_day_grouped_sampler = train_dataset.mode == MODE_HOURLY and not train_dataset.uses_materialized_features
    sampling = {'sampler': DayGroupedShuffleSampler(train_dataset)} if use_day_grouped_sampler else {'shuffle': True}
    train_loader = DataLoader(train_dataset, batch_size=trial_batch_size, **sampling, **loader_kwargs)
    valid_loader = DataLoader(datasets['valid'], batch_size=2 * trial_batch_size, shuffle=False, **loader_kwargs)

    trial_dir = os.path.join(abs_output_path, 'trials', 'retrain_best')
    os.makedirs(trial_dir, exist_ok=True)
    L.seed_everything(seed, workers=True)
    module = module_factory(trial, in_channels, target_stats, normalization)
    module.valid_climatology_cond_mae = selection_climatology_cond_mae
    module.selection_occurrence_event = selection_occurrence_event
    if compile_model and use_cuda and hasattr(module.net, 'compile'):
        module.net.compile()
    metrics = _fit_trial(
        module, trial, train_loader, valid_loader, trial_dir,
        max_epochs, early_stopping_patience, accelerator, devices, precision, use_cuda,
        limit_train_batches, limit_val_batches,
        optuna_trial=None, prune_metric=None,
        progress_prefix='retrain-best' if progress_bar else None,
        diagnostics_prefix='retrain-best' if diagnostics else None,
        profiler=profiler
    )
    score = float(metrics.get(selection_metric, float('nan')))

    # 5) persist the retrained artifacts in the same layout the sweep produces (best_model.ckpt + best_trial.json)
    best_checkpoint = os.path.join(abs_output_path, 'best_model.ckpt')
    shutil.copyfile(os.path.join(trial_dir, 'final.ckpt'), best_checkpoint)
    with open(os.path.join(abs_output_path, 'best_trial.json'), 'w') as handle:
        json.dump({
            'score': score, 'selection_metric': selection_metric, 'trial': trial, 'metrics': metrics,
            'code_state': lazy.code_state_hash(root_path),
            'retrained_from': source_path, 'retrained_on': input_path
        }, handle, indent=2)
    _write_best_metrics(metrics_path, metrics, selection_metric, score, root_path)
    logger.info(
        f'Retrain done: {selection_metric} = {score:.4f} (was {saved_best["score"]:.4f} on the source data); '
        f'model at "{output_path}/best_model.ckpt".'
    )
