"""Pipeline stage: prepare the batta_torch ERA5 + ATDnet data for every model family and both tasks.

ONE preparation stage, shared by all three families — the prepared directory is family-agnostic, and `mode` is the
only key that selects the task:

* ``mode: daily``  — one item per day; the target is the number of hours with qualifying lightning, per cell,
  **bounded 0-24** (the informal name for this is *daily lightning hours*, and it is not a config value). Features
  are the hourly-stacked or daily-averaged ERA5 maps.
* ``mode: hourly`` — one item per (day, hour); the target is **0/1 occurrence** per cell per hour. Features are that
  hour's ERA5 maps, so ``feature-aggregation`` is not read at all.

``hourly-threshold`` applies to **both**, and identically: it is the minimum hourly stroke count for an hour to
count at all (``2`` drops single-stroke observational noise). Daily mode then counts the qualifying hours; hourly
mode emits whether the hour qualified. Sharing one cutoff is what keeps the two tasks' denoising consistent — the
hourly binary target is exactly the per-hour indicator that the daily target sums.

Everything except the target derivation is shared between the modes: sample indexing, split assignment, feature
materialisation, the streamed train statistics, and the upstream-prediction pass. Only ``_derive_target`` branches.

RESIDUAL MODE (diffusion only). With an ``upstream_model_path`` the stage ADDITIONALLY runs that checkpoint over
every prepared item and stores its target-space prediction as per-day ``upstream/<date>.npy`` maps, records the
``upstream_filename`` column and the ``residual_target`` flag, and leaves ``targets/`` untouched. The diffusion model
then conditions on the upstream prediction (appended as the LAST input channel by the dataset) and learns the
discrepancy ``observed - upstream``, reconstructing ``clamp(upstream + residual)`` at inference. Because the raw
target is preserved, the baselines, the target statistics and the whole metric suite stay in target space and the
evaluation stage is shared with the other two families.

⚠️ MC-dropout also reads ``UPSTREAM_MODEL``, but at a DIFFERENT stage and for a different thing: it wants the
upstream's *weights* (a warm start, read by ``tune``), where diffusion wants its *predictions* (an extra conditioning
channel, materialised here).

Stage outputs (written to ``--output_path``):

* ``targets/<date>.npy``    — ``[H, W]`` float32 0-24 (daily) or ``[T, H, W]`` uint8 0/1 (hourly);
* ``features/<date>.npy``   — the predictor maps, memory-mappable, in the mode's access layout;
* ``split_index.csv``       — ``date, sample_id, file, target_filename, split`` (+ ``feature_filename``,
  + ``upstream_filename`` in residual mode);
* ``target_stats.json``     — the train target marginal: ``mode``, ``hourly_threshold``, the zero-proportion
  diagnostic, ``positive_quantiles``, ``positive_mean``. There is **no** ``gamma_shape`` / ``gamma_scale`` and no
  ``target_variable``: the F-transform is gone, so training space == evaluation space;
* ``prepared_config.json``  — how the directory was built, read back by the dataset and every downstream stage.

Usage (standalone)::

    # full-target
    python src/stages/prepare_modeling.py \\
        --data_path $DATA_ROOT \\
        --output_path outputs/deterministic_unet/prepared/daily \\
        --mode daily

    # residual (diffusion on the discrepancy of an upstream deterministic U-net)
    python src/stages/prepare_modeling.py \\
        --data_path $DATA_ROOT \\
        --output_path outputs/diffusion/prepared/daily \\
        --mode daily \\
        --upstream_model_path outputs/deterministic_unet/best/best_model.ckpt
"""
import json
import logging
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from fire import Fire

from __init__ import root_path, console_handler
from src.utils.io.data import (
    MODE_DAILY,
    MODE_HOURLY,
    MODES,
    TARGET_VARIABLE_NAME,
    assign_splits_from_config,
    index_samples,
    load_dataset_metadata,
    load_sample_tensor,
    load_split_config,
    metadata_variable_names,
    normalize_mode
)

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

QUANTILE_LEVELS = (0.5, 0.9, 0.95, 0.99, 0.999)
MAX_POSITIVES_PER_DAY = 20_000           # subsampling cap for the train positive-target reservoir
FEATURE_DTYPES = ('float32', 'float16')  # float16 halves the features/ footprint (~3 decimal digits kept)


def _as_name_list(value: Union[str, list, tuple]) -> list:
    """Normalize a comma-separated string (or an already-parsed sequence) into a list of names."""
    if isinstance(value, str):
        return [name.strip() for name in value.split(',') if name.strip()]
    return list(value)


def _feature_layout_for_mode(mode: str) -> str:
    """Storage layout of the materialized feature files, chosen by access pattern:
    - hourly mode reads one hour at a time -> ``time_major`` ``[T, Vf, H, W]`` (a slice is one contiguous read);
    - daily mode reads the whole day and stacks channels variable-major -> ``variable_major`` ``[Vf, T, H, W]``
      (the ``[Vf * T, H, W]`` channel stack is then a free reshape instead of a strided transpose-copy per item).
    """
    return 'time_major' if mode == MODE_HOURLY else 'variable_major'


def _write_feature_file(
        features_dir: str,
        filename: str,
        sample: np.ndarray,
        feature_positions: List[int],
        dtype: np.dtype,
        layout: str
) -> None:
    """Write one day's predictor maps C-contiguously in the requested layout (see _feature_layout_for_mode)."""
    features = sample[feature_positions]                                     # [Vf, T, H, W]
    if layout == 'time_major':
        features = features.transpose(1, 0, 2, 3)                            # [T, Vf, H, W]
    np.save(os.path.join(features_dir, filename), np.ascontiguousarray(features, dtype=dtype))


def _backfill_features(abs_output_path: str, feature_dtype: str) -> None:
    """Materialize the per-day feature files into an already-prepared directory (older runs), leaving targets,
    split assignment and statistics untouched. Also re-materializes features whose stored layout or dtype no
    longer matches the request: a layout that mismatches the mode's access pattern (e.g. time-major files in a
    daily-mode directory, written before layouts were mode-specific), or a stored dtype that differs from the
    requested ``feature_dtype`` (so flipping feature-dtype, e.g. float32 -> float16, takes effect even under
    overwrite=false). A no-op when up-to-date features are already there."""
    with open(os.path.join(abs_output_path, 'prepared_config.json')) as handle:
        prepared_config = json.load(handle)
    layout = _feature_layout_for_mode(normalize_mode(prepared_config['mode']))
    has_features = bool(prepared_config.get('feature_dtype'))
    relayout = has_features and prepared_config.get('feature_layout', 'time_major') != layout
    redtype = has_features and np.dtype(prepared_config['feature_dtype']) != np.dtype(feature_dtype)
    rewrite = relayout or redtype                                    # a layout/dtype change must touch every file
    if has_features and not rewrite:
        logger.info('Prepared artifacts (including materialized features) already present; nothing to do.')
        return

    split_index = pd.read_csv(os.path.join(abs_output_path, 'split_index.csv'))
    features_dir = os.path.join(abs_output_path, 'features')
    os.makedirs(features_dir, exist_ok=True)
    dtype = np.dtype(feature_dtype)                                  # rewrites and backfills honor the request
    feature_positions = [prepared_config['variable_names'].index(name) for name in prepared_config['features']]

    reasons = [name for name, active in (('layout', relayout), ('dtype', redtype)) if active]
    action = f'Rewriting ({"/".join(reasons)})' if rewrite else 'Backfilling'
    logger.info(f'{action} {len(split_index)} per-day feature files ({dtype.name}) into "{features_dir}".')
    feature_filenames = []
    for position, row in split_index.iterrows():
        feature_filename = row['target_filename']                           # same <date>.npy naming as targets/
        feature_filenames.append(feature_filename)
        # backfills are resumable across interruptions; layout/dtype rewrites must touch every file
        if not rewrite and os.path.exists(os.path.join(features_dir, feature_filename)):
            continue
        sample = load_sample_tensor(row['file'], prepared_config['variable_names'])
        _write_feature_file(features_dir, feature_filename, sample, feature_positions, dtype, layout)
        if (position + 1) % 500 == 0:
            logger.info(f'Processed {position + 1}/{len(split_index)} feature files.')

    split_index['feature_filename'] = feature_filenames
    split_index.to_csv(os.path.join(abs_output_path, 'split_index.csv'), index=False)
    prepared_config['feature_dtype'] = dtype.name
    prepared_config['feature_layout'] = layout
    with open(os.path.join(abs_output_path, 'prepared_config.json'), 'w') as handle:
        json.dump(prepared_config, handle, indent=2)
    logger.info('Feature materialization complete.')


# =====================================================================================================================
# The target — the ONE thing that branches on the mode
# =====================================================================================================================
def _qualifying_hours(lightning: np.ndarray, hourly_threshold: int) -> np.ndarray:
    """Boolean ``[T, H, W]``: the hours whose stroke count reaches ``hourly_threshold``.

    The single definition of "an hour with lightning", shared by both tasks. Daily mode SUMS this over the day and
    hourly mode emits it directly, which is what makes the two targets consistent rather than merely similar: the
    daily 0-24 count is exactly the per-cell sum of the hourly 0/1 field prepared from the same threshold.
    """
    return lightning >= hourly_threshold


def _daily_aggregation(lightning: np.ndarray, hourly_threshold: int) -> np.ndarray:
    """Reduce the hourly lightning grids ``[T, H, W]`` to the daily target map ``[H, W]`` float32: the number of
    hours with qualifying lightning, so bounded ``0-T`` (24 on this dataset).

    ⚠️ Branch A also offered ``lightning_counts`` (the daily stroke sum) and ``lightning_peak`` (the max hourly
    count) through a ``target_variable`` parameter. Both are UNBOUNDED heavy-tailed counts, which is the regime the
    classification-first scope removed along with the gamma F-transform, so the parameter is gone and this is the
    only aggregation. See CLAUDE.md, "Current scope".
    """
    return _qualifying_hours(lightning, hourly_threshold).sum(axis=0).astype(np.float32)


def _derive_target(lightning: np.ndarray, mode: str, hourly_threshold: int) -> np.ndarray:
    """The per-item target from one day's hourly lightning grids ``[T, H, W]``.

    Hourly mode is stored as ``uint8`` rather than float32: the values are exactly 0 and 1, so uint8 is lossless and
    four times smaller — and an hourly prepared directory is 24x the items of a daily one, so the factor is worth
    having. Every reader casts to float32 on load (``evaluation._load_target``, ``LightningMapsDataset.__getitem__``).
    """
    if mode == MODE_HOURLY:
        return _qualifying_hours(lightning, hourly_threshold).astype(np.uint8)   # [T, H, W], 0/1 occurrence
    return _daily_aggregation(lightning, hourly_threshold)                       # [H, W], 0-24 lightning-hours


def _new_accumulator() -> dict:
    """A reservoir for the train target marginal: subsampled positives plus zero/total/max running counts."""
    return {'positives': [], 'n_zero': 0, 'n_total': 0, 'max': 0.0}


def _accumulate(acc: dict, target_map: np.ndarray, rng: np.random.Generator) -> None:
    """Fold one day's target map into the marginal accumulator (positives capped per day at MAX_POSITIVES_PER_DAY)."""
    flat = target_map.astype(np.float64).ravel()
    positives = flat[flat > 0]
    acc['n_zero'] += int(flat.size - positives.size)
    acc['n_total'] += int(flat.size)
    if flat.size:
        acc['max'] = max(acc['max'], float(flat.max()))
    if positives.size > MAX_POSITIVES_PER_DAY:
        positives = rng.choice(positives, size=MAX_POSITIVES_PER_DAY, replace=False)
    acc['positives'].append(positives)


def _zero_proportion_report(target_zero_proportion: float, raw_hourly_zero_cells: int,
                            raw_hourly_total_cells: int) -> dict:
    """The sparsity diagnostic over the train split — the number every design choice in this project is downstream of
    (~99.93 % of daily cells are zero).

    Reports the written target's zero proportion beside the RAW hourly grid's. ``raw_hourly`` counts ``lightning == 0``
    on the untouched field, so it is unaffected by ``hourly_threshold``; the target is affected by BOTH the threshold
    and (in daily mode) the aggregation, and those pull in opposite directions:

    * hourly — ``target >= raw_hourly``, always. Only the threshold acts, and it can only zero more cells.
    * daily — **the sign is data-dependent.** Aggregating T hours into one count pushes the zero proportion DOWN (a
      cell is zero only if none of its hours qualified), while the threshold pushes it UP. Which wins depends on how
      much of the activity is single-stroke: on the real dataset the aggregation dominates, but on a sparse subset
      where most active hours carry one stroke the threshold can.

    Both raw numbers are reported rather than one signed difference precisely because of that: a single number would
    have to be interpreted differently per mode, and its sign is not a property to rely on.
    """
    raw_hourly = raw_hourly_zero_cells / max(raw_hourly_total_cells, 1)
    return {
        'target': target_zero_proportion,
        'raw_hourly': raw_hourly,
        'target_minus_raw_hourly': target_zero_proportion - raw_hourly,
    }


def _log_zero_proportion(mode: str, report: dict) -> None:
    """Emit the sparsity diagnostic to the stage log.

    The direction is read off the measured difference rather than assumed from the mode — in daily mode the
    aggregation and the threshold pull opposite ways and either can win (see :func:`_zero_proportion_report`).
    """
    difference = report['target_minus_raw_hourly']
    direction = 'less sparse' if difference < 0 else 'sparser' if difference > 0 else 'unchanged'
    logger.info(
        f'Zero proportion (train, mode "{mode}"): raw hourly {report["raw_hourly"]:.5f} -> target '
        f'{report["target"]:.5f} — the target is {direction} by {abs(difference):.5f}.'
    )


def _prepare_base(
        data_path: str,
        output_path: str,
        mode: str,
        feature_aggregation: str,
        features: Union[str, list],
        split_config: str,
        overwrite: bool,
        max_samples: Optional[int],
        materialize_features: bool,
        feature_dtype: str,
        hourly_threshold: int
) -> None:
    """Targets, split assignment and train target statistics — everything except the upstream pass."""
    mode = normalize_mode(mode)
    if mode not in MODES:
        raise ValueError(f'Unknown mode "{mode}" (expected one of {MODES}).')
    hourly_threshold = int(hourly_threshold)
    if hourly_threshold < 1:
        raise ValueError(
            f'hourly_threshold is the MINIMUM hourly stroke count for an hour to count, so it must be >= 1 '
            f'(1 = every hour with any stroke; 2 = drop single-stroke hours, the configured default); got '
            f'{hourly_threshold}. A value of 0 would count empty hours.'
        )
    if mode == MODE_DAILY and feature_aggregation not in ('hourly_stack', 'daily_mean'):
        raise ValueError(f'Unknown feature aggregation "{feature_aggregation}".')
    if feature_dtype not in FEATURE_DTYPES:
        raise ValueError(f'Unknown feature dtype "{feature_dtype}" (expected one of {FEATURE_DTYPES}).')

    abs_data_path = os.path.join(root_path, data_path)
    abs_output_path = os.path.join(root_path, output_path)
    config_file = os.path.join(abs_output_path, 'prepared_config.json')
    if os.path.exists(config_file) and not overwrite:
        # the on-disk TARGETS depend on mode and hourly_threshold. The overwrite=False fast path only backfills
        # features and never recomputes targets, and hourly_threshold is NOT encoded in the output path — so
        # flipping it (2 -> 3) on an existing dir would otherwise SILENTLY reuse the stale targets and train on the
        # wrong one. Refuse loudly instead. A dir predating the key reads as 1, its effective legacy value.
        with open(config_file) as handle:
            existing_prepared = json.load(handle)
        for key, requested in (('mode', mode), ('hourly_threshold', hourly_threshold)):
            existing_value = existing_prepared.get(key, 1 if key == 'hourly_threshold' else None)
            if existing_value != requested:
                raise ValueError(
                    f'Prepared directory "{output_path}" was built with {key}={existing_value!r} but '
                    f'{key}={requested!r} is requested: its targets are STALE for this run (the target depends on '
                    f'this parameter). Re-prepare with overwrite=true, or use a fresh output directory — refusing '
                    f'to skip and train on the existing (mismatched) target.'
                )
        if materialize_features:
            _backfill_features(abs_output_path, feature_dtype)
        else:
            logger.info(f'Prepared artifacts already present in "{output_path}" and overwrite=False; nothing to do.')
        return

    # train/valid/test split specification (single source of truth, shared across stages and families)
    split_spec = load_split_config(os.path.join(root_path, split_config))
    logger.info(f'Using split "{split_config}" (method "{split_spec["method"]}").')

    # dataset description and variable bookkeeping
    metadata = load_dataset_metadata(abs_data_path)
    variable_names = metadata_variable_names(metadata)
    feature_names = _as_name_list(features)
    unknown = [name for name in feature_names if name not in variable_names]
    if unknown:
        raise ValueError(f'Unknown feature variables {unknown}; dataset variables are {variable_names}.')
    if TARGET_VARIABLE_NAME not in variable_names:
        raise ValueError(f'Target variable "{TARGET_VARIABLE_NAME}" not declared in the dataset metadata.')
    lightning_position = variable_names.index(TARGET_VARIABLE_NAME)

    # sample index and split assignment from the shared split config
    index = index_samples(abs_data_path)
    index['split'] = assign_splits_from_config(index, split_spec)
    n_dropped = int(index['split'].isna().sum())
    if n_dropped:
        logger.info(f'Dropping {n_dropped} samples not assigned to any split by "{split_config}".')
        index = index.dropna(subset=['split']).reset_index(drop=True)
    if max_samples is not None:
        index = index.head(int(max_samples))
    for split_name in ('train', 'valid', 'test'):
        if not (index['split'] == split_name).any():
            raise ValueError(f'The "{split_name}" split is empty; check "{split_config}" against the dataset.')
    logger.info(
        f'Preparing {len(index)} samples '
        f'({(index["split"] == "train").sum()} train / {(index["split"] == "valid").sum()} valid / '
        f'{(index["split"] == "test").sum()} test) in mode "{mode}" '
        f'(hourly_threshold {hourly_threshold}).'
    )

    targets_dir = os.path.join(abs_output_path, 'targets')
    os.makedirs(targets_dir, exist_ok=True)
    features_dir = os.path.join(abs_output_path, 'features')
    if materialize_features:
        os.makedirs(features_dir, exist_ok=True)
    feature_positions = [variable_names.index(name) for name in feature_names]
    feature_np_dtype = np.dtype(feature_dtype)
    feature_layout = _feature_layout_for_mode(mode)

    # one pass over all samples: write per-day targets (and features) and accumulate train target statistics.
    rng = np.random.default_rng(0)
    main_acc = _new_accumulator()
    raw_hourly_zero_cells, raw_hourly_total_cells = 0, 0
    hours_per_day, grid_shape = None, None
    target_filenames = []

    for position, row in index.iterrows():
        sample = load_sample_tensor(row['file'], variable_names)            # [V, T, H, W]
        lightning = sample[lightning_position]                              # [T, H, W]
        if hours_per_day is None:
            hours_per_day, grid_shape = int(lightning.shape[0]), tuple(lightning.shape[1:])

        target = _derive_target(lightning, mode, hourly_threshold)
        target_filename = f'{row["date"].date()}.npy'
        np.save(os.path.join(targets_dir, target_filename), target)
        target_filenames.append(target_filename)
        if materialize_features:
            _write_feature_file(
                features_dir, target_filename, sample, feature_positions, feature_np_dtype, feature_layout
            )

        if row['split'] == 'train':
            _accumulate(main_acc, target, rng)
            raw_hourly_zero_cells += int((lightning == 0).sum())
            raw_hourly_total_cells += int(lightning.size)

        if (position + 1) % 500 == 0:
            logger.info(f'Processed {position + 1}/{len(index)} samples.')

    index['target_filename'] = target_filenames

    # train target marginal statistics. NO gamma fit: the gamma F-transform is removed, so there is no distribution
    # to condition and training space == evaluation space (CLAUDE.md, "Current scope"). `positive_quantiles` stays
    # because `evaluation.resolve_threshold`'s generic `train_positive_quantile` kind still reads it.
    positives = np.concatenate(main_acc['positives']) if main_acc['positives'] else np.array([])
    if positives.size == 0:
        raise ValueError('No positive targets found in the train split; cannot fit the target statistics.')
    zero_proportion = main_acc['n_zero'] / max(main_acc['n_total'], 1)
    target_stats = {
        'mode': mode,
        'hourly_threshold': hourly_threshold,        # the denoising baked into the target
        'zero_proportion': _zero_proportion_report(
            zero_proportion, raw_hourly_zero_cells, raw_hourly_total_cells
        ),
        'n_train_cells': main_acc['n_total'],
        'n_train_positive_subsampled': int(positives.size),
        'positive_quantiles': {str(level): float(np.quantile(positives, level)) for level in QUANTILE_LEVELS},
        'positive_mean': float(positives.mean()),
        'max': main_acc['max'],
    }
    with open(os.path.join(abs_output_path, 'target_stats.json'), 'w') as handle:
        json.dump(target_stats, handle, indent=2)
    logger.info(
        f'Target marginal: zero proportion {zero_proportion:.5f}, positive mean '
        f'{target_stats["positive_mean"]:.2f}, max {target_stats["max"]:.0f}.'
    )
    _log_zero_proportion(mode, target_stats['zero_proportion'])

    # split index and preparation config (the features/ files mirror the targets/ naming)
    index_columns = ['date', 'sample_id', 'file', 'target_filename', 'split']
    if materialize_features:
        index['feature_filename'] = target_filenames
        index_columns.append('feature_filename')
    index[index_columns].to_csv(os.path.join(abs_output_path, 'split_index.csv'), index=False)
    prepared_config = {
        'mode': mode,
        'feature_aggregation': feature_aggregation,
        'hourly_threshold': hourly_threshold,
        'features': feature_names,
        'variable_names': variable_names,
        'hours_per_day': hours_per_day,
        'grid_shape': list(grid_shape),
        'feature_dtype': feature_dtype if materialize_features else None,
        'feature_layout': feature_layout if materialize_features else None,
        'split_config': split_config,
        'split': split_spec,
        'split_counts': {name: int((index['split'] == name).sum()) for name in ('train', 'valid', 'test')},
        'data_path': data_path
    }
    with open(config_file, 'w') as handle:
        json.dump(prepared_config, handle, indent=2)

    logger.info(
        f'Done: {len(index)} samples prepared, grid {grid_shape}, {hours_per_day} hourly steps per day; '
        f'artifacts in "{output_path}".'
    )


def _materialize_upstream(
        output_path: str,
        upstream_model_path: str,
        overwrite: bool,
        accelerator: str,
        devices: int,
        num_workers: int,
        batch_size: int
) -> None:
    """Run the upstream model over every prepared item and store its target-space predictions as per-day
    ``upstream/<date>.npy`` maps, then flag the prepared directory as residual mode.

    The raw ``targets/`` are left untouched; the diffusion model forms the residual ``target - upstream`` on the
    fly and reconstructs the target-space prediction by adding the upstream prediction back. Idempotent: skips
    when predictions for the same upstream model are already present (unless ``overwrite``).
    """
    import torch
    from torch.utils.data import DataLoader

    from src.utils.io.data import load_prepared_artifacts
    from src.utils.modeling.dataset import LightningMapsDataset
    from src.utils.modeling.registry import load_model_module

    abs_output_path = os.path.join(root_path, output_path)
    abs_model_path = upstream_model_path if os.path.isabs(upstream_model_path) \
        else os.path.join(root_path, upstream_model_path)
    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f'Upstream model checkpoint "{upstream_model_path}" not found.')

    # identity of the upstream checkpoint for idempotency: resolved path + size + mtime, so an equivalent path
    # spelling reuses the materialized predictions while an in-place RETUNE (same path, new content) invalidates
    # them (size/mtime change) instead of silently reusing stale predictions
    model_stat = os.stat(abs_model_path)
    model_signature = f'{os.path.abspath(abs_model_path)}:{model_stat.st_size}:{model_stat.st_mtime_ns}'

    prepared_config, split_index, _ = load_prepared_artifacts(abs_output_path)
    upstream_dir = os.path.join(abs_output_path, 'upstream')

    already = (
        bool(prepared_config.get('residual_target'))
        and prepared_config.get('upstream_model_signature') == model_signature
        and 'upstream_filename' in split_index.columns
        and split_index['upstream_filename'].notna().all()
        and all(os.path.exists(os.path.join(upstream_dir, name)) for name in split_index['upstream_filename'])
    )
    if already and not overwrite:
        logger.info(
            f'Upstream predictions for "{upstream_model_path}" already materialized in "{output_path}"; '
            f'nothing to do.'
        )
        return

    # dataset over the BASE features of every split (no upstream channel yet), in index order
    base_config = dict(prepared_config)
    base_config['residual_target'] = False
    dataset = LightningMapsDataset(split_index, base_config)
    if len(dataset) == 0:
        raise ValueError(f'No prepared items in "{output_path}"; run the base preparation first.')

    module = load_model_module(abs_model_path, map_location='cpu')
    upstream_in_channels = int(module.feature_mean.shape[0])
    if upstream_in_channels != dataset.in_channels:
        raise ValueError(
            f'Upstream model expects {upstream_in_channels} input channels but the prepared base features have '
            f'{dataset.in_channels}; the upstream model must be trained on the same feature/mode configuration.'
        )
    upstream_stats = getattr(module, 'target_stats', {}) or {}
    # mode AND hourly_threshold must match: the residual is (prepared target - upstream prediction), so the upstream
    # must live in the SAME target space, including the same hourly-count denoising. An upstream trained with a
    # different hourly_threshold would silently mix target spaces.
    for key in ('mode', 'hourly_threshold'):
        if key in upstream_stats and upstream_stats[key] != prepared_config.get(key):
            raise ValueError(
                f'Upstream model {key}="{upstream_stats[key]}" does not match the prepared data '
                f'{key}="{prepared_config.get(key)}"; the residual would mix incompatible targets.'
            )
    # ordered feature provenance: the upstream model standardizes its conditioning channels POSITIONALLY, so the
    # feature set AND order must match the base features this directory feeds it (a reorder, or a subset with the
    # same channel count, would otherwise silently mis-standardize every channel). `tuning.run_sweep` records both
    # keys into target_stats at load time precisely so this check can be made; a checkpoint predating that carries
    # no provenance, and the best we can do there is warn.
    upstream_features = upstream_stats.get('features')
    if upstream_features is not None:
        if list(upstream_features) != list(prepared_config['features']):
            raise ValueError(
                f'Upstream model was trained on features {list(upstream_features)} but the prepared data uses '
                f'{list(prepared_config["features"])} (set/order mismatch): the conditioning channels would be '
                f'mis-standardized. Re-prepare (or re-tune the upstream) with matching features in the same order.'
            )
        upstream_aggregation = upstream_stats.get('feature_aggregation')
        if upstream_aggregation and upstream_aggregation != prepared_config.get('feature_aggregation'):
            raise ValueError(
                f'Upstream model feature_aggregation="{upstream_aggregation}" does not match the prepared '
                f'"{prepared_config.get("feature_aggregation")}".'
            )
    else:
        logger.warning(
            'The upstream checkpoint carries no feature provenance: cannot verify its feature set/order matches %s '
            '(aggregation %s). Ensure the upstream model was trained on exactly these features, in this order, or '
            'its predictions will be silently mis-standardized.',
            list(prepared_config['features']), prepared_config.get('feature_aggregation')
        )

    use_cuda = torch.cuda.is_available() and accelerator in ('auto', 'gpu', 'cuda')
    if use_cuda:
        torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if use_cuda else 'cpu')
    module = module.to(device).eval()
    logger.info(
        f'Materializing upstream predictions for {len(dataset)} items (mode "{prepared_config["mode"]}", '
        f'device "{device.type}") into "{upstream_dir}".'
    )

    mode = prepared_config['mode']
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    items = dataset.items_frame().reset_index(drop=True)               # date, hour per item, in iteration order
    os.makedirs(upstream_dir, exist_ok=True)

    def flush(date: pd.Timestamp, buffer: dict) -> None:
        filename = f'{pd.Timestamp(date).date()}.npy'
        if mode == MODE_HOURLY:
            array = np.stack([buffer[hour] for hour in sorted(buffer)], axis=0)        # [T, H, W]
        else:
            array = next(iter(buffer.values()))                                        # [H, W]
        np.save(os.path.join(upstream_dir, filename), np.ascontiguousarray(array, dtype=np.float32))

    current_date, day_buffer, global_index, n_days = None, {}, 0, 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            x, y = batch[0].to(device), batch[1]
            prediction = module.predict_step((x, y), batch_index)['prediction'].float().numpy()   # [b, H, W]
            for local in range(prediction.shape[0]):
                item = items.iloc[global_index]
                item_date = pd.Timestamp(item['date'])
                if current_date is not None and item_date != current_date:
                    flush(current_date, day_buffer)
                    n_days += 1
                    day_buffer = {}
                current_date = item_date
                hour_key = 0 if mode == MODE_DAILY else int(item['hour'])
                day_buffer[hour_key] = prediction[local]
                global_index += 1
    if day_buffer:
        flush(current_date, day_buffer)
        n_days += 1
    logger.info(f'Wrote {n_days} per-day upstream prediction files.')

    # upstream files mirror the targets/ naming, so the column equals target_filename; flag the directory and
    # persist (dropping the computed absolute-path columns that load_prepared_artifacts adds in memory)
    split_index['upstream_filename'] = split_index['target_filename']
    persistent_columns = [
        column for column in split_index.columns
        if column not in ('target_file', 'feature_file', 'upstream_file')
    ]
    split_index[persistent_columns].to_csv(os.path.join(abs_output_path, 'split_index.csv'), index=False)

    prepared_config['residual_target'] = True
    prepared_config['upstream_model_path'] = upstream_model_path
    prepared_config['upstream_model_signature'] = model_signature
    with open(os.path.join(abs_output_path, 'prepared_config.json'), 'w') as handle:
        json.dump(prepared_config, handle, indent=2)
    logger.info(
        f'Residual mode enabled in "{output_path}": the diffusion model will learn '
        f'(observed target - upstream prediction).'
    )


def prepare_modeling(
        data_path: str,
        output_path: str,
        mode: str = MODE_DAILY,
        feature_aggregation: str = 'hourly_stack',
        features: Union[str, list] = 'MU_LI,MU_MIXR,RH_500850,cp,lsm',
        split_config: str = 'config/split/split.yaml',
        overwrite: bool = False,
        max_samples: Optional[int] = None,
        materialize_features: bool = True,
        feature_dtype: str = 'float16',
        hourly_threshold: int = 2,
        upstream_model_path: Optional[str] = None,
        upstream_accelerator: str = 'auto',
        upstream_devices: int = 1,
        upstream_num_workers: int = 8,
        upstream_batch_size: int = 16
) -> None:
    """Prepare targets, split assignment and target statistics, optionally augmented for the diffusion residual.

    Args:
        data_path: Path to the batta_torch dataset directory (with metadata.json, metadata.csv and samples/).
            Reach it through ``$DATA_ROOT``; never hardcode a data path.
        output_path: Directory (relative to root_path unless absolute) where the artifacts are written.
        mode: The task. ``daily`` — the 0-24 lightning-hours regression; ``hourly`` — the 0/1 occurrence
            classification. This is the ONLY key that selects between them. Legacy names are accepted with a
            warning (``daily_lightning_hours`` -> ``daily``), so directories prepared under the old name still load.
        feature_aggregation: ``hourly_stack`` or ``daily_mean``. DAILY MODE ONLY — in hourly mode an item is already
            one hour, so there is nothing to stack and the key is not read.
        features: Comma-separated ERA5 predictor names (must match the metadata variable names). The ORDER matters:
            the model standardizes its channels positionally, and the residual check below verifies it.
        split_config: Path to the train/valid/test split YAML (the shared year-based split).
        overwrite: If False and the output directory already holds prepared artifacts, the base stage only
            backfills missing feature files; the upstream materialization likewise skips when already present.
            A ``mode`` or ``hourly_threshold`` that disagrees with the existing directory RAISES rather than
            skipping, because the targets on disk would be stale.
        max_samples: Optional cap on the number of days processed (debugging only — it takes the EARLIEST days, so
            on the year split it would keep test-only days; the smoke tiers use a by_sample_id split instead).
        materialize_features: Also write the per-day predictor maps as memory-mappable ``features/<date>.npy``.
        feature_dtype: Storage dtype of the materialized features, ``float16`` (default, halves the footprint) or
            ``float32``. Statistics are always accumulated in float64 regardless, so storage precision never
            affects normalization.
        hourly_threshold: The MINIMUM hourly stroke count for an hour to count (the value IS the cutoff), applied
            in BOTH modes: ``2`` (default) drops single-stroke observational noise. Daily mode then counts the
            qualifying hours; hourly mode emits whether the hour qualified. It is applied to the raw hourly counts
            only, so it is baked into the target and inherited by the residual (never re-applied to it). Changing it
            changes the target — re-prepare and re-train.
        upstream_model_path: Optional path to a checkpoint of any family. When set, the stage ADDITIONALLY runs that
            model over every item and stores its target-space prediction as per-day ``upstream/<date>.npy`` maps,
            flags the directory as residual mode (``residual_target``) and records the ``upstream_filename``
            column — the diffusion model then learns the discrepancy. Unset (or an empty string, which is what an
            unset ``{{$UPSTREAM_MODEL}}`` substitutes to) leaves the plain full-target preparation.
        upstream_accelerator: Accelerator for the upstream prediction pass (``auto``/``gpu``/``cpu``).
        upstream_devices: Number of devices for the upstream prediction pass (prediction order assumes 1).
        upstream_num_workers: DataLoader workers for the upstream prediction pass.
        upstream_batch_size: Batch size for the upstream prediction pass.

    Returns:
        None. Writes the base artifacts (and, in residual mode, the ``upstream/`` maps + residual flags).
    """
    _prepare_base(
        data_path=data_path,
        output_path=output_path,
        mode=mode,
        feature_aggregation=feature_aggregation,
        features=features,
        split_config=split_config,
        overwrite=overwrite,
        max_samples=max_samples,
        materialize_features=materialize_features,
        feature_dtype=feature_dtype,
        hourly_threshold=hourly_threshold
    )

    # an unset {{$UPSTREAM_MODEL}} placeholder substitutes to an empty string, not None: treat both as "no upstream"
    if not upstream_model_path:
        logger.info(
            'No upstream model provided: prepared standard (full-target) artifacts. A diffusion model trained on '
            'these learns the ENTIRE target rather than a correction.'
        )
        return

    _materialize_upstream(
        output_path=output_path,
        upstream_model_path=upstream_model_path,
        overwrite=overwrite,
        accelerator=upstream_accelerator,
        devices=upstream_devices,
        num_workers=upstream_num_workers,
        batch_size=upstream_batch_size
    )


if __name__ == '__main__':
    Fire(prepare_modeling)
