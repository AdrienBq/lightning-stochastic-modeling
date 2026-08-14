"""I/O helpers for the batta_torch ERA5 + ATDnet dataset.

The dataset directory (see CLAUDE.md) is expected to contain:
- ``metadata.json``: dataset description; ``variable_1``..``variable_N`` name the channels stored in each sample,
  the last one (``lightnings``) being the gridded ATDnet target;
- ``metadata.csv``: one record per sample with columns ``date,id,num_lightnings,pixels_with_lightning``;
- ``samples/``: torch checkpoints with progressive names (e.g. ``sample_123456.pt``), each containing all variables.

The exact on-disk layout of a sample checkpoint is not fully standardized across the project's branches, so
:func:`load_sample_tensor` normalizes the common cases (dict of named tensors, dict of ``variable_i`` keys, or a
single stacked tensor) into a ``[V, T, H, W]`` float32 array, with ``T`` the number of hourly steps (1 for purely
daily fields).

This module has NO internal dependencies: it imports nothing from ``src``. That is deliberate, and it is what the
removal of the target transform bought — the only internal import here used to be a deferred
``from src.utils.modeling.transforms import ...`` inside the diffusion generation-space statistics.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)

METADATA_JSON_FILENAME = 'metadata.json'
METADATA_CSV_FILENAME = 'metadata.csv'
SAMPLES_DIRNAME = 'samples'
TARGET_VARIABLE_NAME = 'lightnings'

# ONE AXIS: the mode fully determines both the target and the task. There is no `target_variable` parameter --
# the other quantities that used to need one (`lightning_counts`, `lightning_peak`) are unbounded counts and are
# out of scope, which leaves exactly one legal target per mode:
#
#   MODE_DAILY  -> hours in the day with lightning, per cell, a bounded 0-24 integer  -> REGRESSION
#   MODE_HOURLY -> 0 or 1 per cell per hour                                           -> CLASSIFICATION
MODE_DAILY = 'daily'
MODE_HOURLY = 'hourly'
MODES = (MODE_DAILY, MODE_HOURLY)

# `daily_lightning_hours` was the old name for the daily mode and still resolves to the same target it always did,
# so it is accepted (with a deprecation warning) and artifacts prepared under it keep loading.
#
# `hourly_counts` is deliberately NOT an alias. It named the unbounded hourly stroke count, which the hourly mode
# no longer produces -- accepting it would silently answer a request for counts with a binary target, i.e. change
# the meaning of the request while looking like a rename. It now fails in `normalize_mode` instead.
LEGACY_MODE_ALIASES = {'daily_lightning_hours': MODE_DAILY}

# Single-stroke observational-noise denoising (ATDnet location/detection errors) is applied at PREPARATION time via
# the `hourly_threshold` parameter (an hour counts only with `>= hourly_threshold` strokes -- 2 drops single-stroke
# hours, 1 keeps every hour with any stroke). Both modes share the cutoff, so the two tasks denoise identically:
# the daily mode counts the hours that qualify, the hourly mode emits 0/1 per hour. The filter is applied BEFORE
# the target is written, so it is baked into `targets/<date>.npy` permanently.
#
# The evaluation-side occurrence event is therefore simply `target > 0` (see `resolve_occurrence_event`): any
# positive stored value has already survived the filter. There is no eval-side `occurrence_threshold` knob -- it
# would be a second, competing definition of "this cell had lightning". Changing `hourly_threshold` changes the
# target itself and requires re-preparing AND re-training; it cannot be adjusted at evaluation time.


def normalize_mode(mode: str) -> str:
    """Normalize a preparation mode name, mapping the deprecated name to its canonical one.

    Args:
        mode: Mode name (``daily`` / ``hourly``, or the legacy ``daily_lightning_hours``).

    Returns:
        The canonical mode name (a deprecation warning is logged when the legacy name was mapped).

    Raises:
        ValueError: If the name is neither canonical nor the accepted legacy alias. In particular
            ``hourly_counts`` lands here: it named an unbounded count target that no longer exists, so mapping it
            onto the binary hourly target would silently change what was asked for.
    """
    if mode in MODES:
        return mode
    if mode in LEGACY_MODE_ALIASES:
        canonical = LEGACY_MODE_ALIASES[mode]
        logger.warning(f'Mode name "{mode}" is deprecated; use "{canonical}" instead.')
        return canonical
    raise ValueError(
        f'Unknown mode "{mode}" (expected one of {MODES}, or the legacy alias '
        f'{sorted(LEGACY_MODE_ALIASES)}). Note that "hourly_counts" is no longer accepted: the hourly mode now '
        f'produces a 0/1 occurrence target, not an unbounded stroke count.'
    )


def load_dataset_metadata(data_path: str) -> dict:
    """Load the dataset description (``metadata.json``) of a batta_torch-style dataset.

    Args:
        data_path: Path to the dataset directory.

    Returns:
        The parsed metadata dictionary.
    """
    with open(os.path.join(data_path, METADATA_JSON_FILENAME)) as handle:
        return json.load(handle)


def metadata_variable_names(metadata: dict) -> List[str]:
    """Return the ordered list of variable names declared in the dataset metadata.

    Args:
        metadata: Parsed contents of ``metadata.json``.

    Returns:
        Variable names, ordered as ``variable_1``..``variable_N``.
    """
    num_variables = int(metadata['num_variables'])
    return [metadata[f'variable_{i}'] for i in range(1, num_variables + 1)]


def high_lightning_days(data_path: str, quantile: float = 0.95) -> pd.DataFrame:
    """The days whose domain-wide stroke count reaches a quantile of the dataset's own marginal — "the extremes".

    Reads ``num_lightnings`` from ``metadata.csv`` (the total raw ATDnet stroke count per day over the whole domain)
    and returns the days at or above its ``quantile``, most active first. Analysis helper: it answers a question about
    the DATA, not about a model, and no pipeline stage calls it — which is why it lives here beside the other
    ``metadata.csv`` readers rather than being a stage of its own.

    ⚠️ The quantile is taken over EVERY day in ``metadata.csv``, train/valid/test alike. That is deliberate for
    describing the dataset, and wrong for anything that feeds a model: selecting on a threshold computed with the test
    years in it leaks. Filter to the train split first if a result is going to inform a modelling choice.

    Args:
        data_path: Path to the dataset directory (holding ``metadata.csv``).
        quantile: Quantile of the daily domain-wide stroke count, in ``[0, 1]``.

    Returns:
        DataFrame with ``date``, ``num_lightnings``, ``pixels_with_lightning`` and ``quantile_threshold`` (the
        resolved stroke count), sorted by ``num_lightnings`` descending. The threshold column is constant and is
        carried so a written-out CSV records what produced it.
    """
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError(f'quantile must lie in [0, 1]; got {quantile}.')
    metadata = pd.read_csv(os.path.join(data_path, METADATA_CSV_FILENAME), parse_dates=['date'])

    threshold = int(round(float(metadata['num_lightnings'].quantile(quantile))))
    logger.info(
        f'{quantile:.0%} quantile of num_lightnings: {threshold} strokes (over {len(metadata)} days).'
    )

    columns = ['date', 'num_lightnings', 'pixels_with_lightning']
    days = metadata[metadata['num_lightnings'] >= threshold][columns].copy()
    for column in ('num_lightnings', 'pixels_with_lightning'):
        days[column] = days[column].round().astype(int)
    days['quantile_threshold'] = threshold
    return days.sort_values('num_lightnings', ascending=False).reset_index(drop=True)


def index_samples(data_path: str) -> pd.DataFrame:
    """Build an index of available samples by joining ``metadata.csv`` with the files in ``samples/``.

    Sample files have progressive names embedding an integer (e.g. ``sample_000123.pt``); the integer is matched
    against the ``id`` column of ``metadata.csv``. Records without a matching file (or files without a record) are
    dropped with a warning.

    Args:
        data_path: Path to the dataset directory.

    Returns:
        DataFrame with columns ``date`` (datetime64), ``sample_id`` (int) and ``file`` (absolute path), sorted by
        date.

    Raises:
        ValueError: If no sample file can be matched against the metadata records.
    """
    records = pd.read_csv(os.path.join(data_path, METADATA_CSV_FILENAME), parse_dates=['date'])

    # map the integer embedded in each filename to its absolute path
    samples_dir = os.path.join(data_path, SAMPLES_DIRNAME)
    file_by_id: Dict[int, str] = {}
    for filename in os.listdir(samples_dir):
        if not filename.endswith('.pt'):
            continue
        match = re.search(r'(\d+)', filename)
        if match is None:
            continue
        file_by_id[int(match.group(1))] = os.path.join(samples_dir, filename)

    index = records.rename(columns={'id': 'sample_id'})[['date', 'sample_id']].copy()
    index['file'] = index['sample_id'].map(file_by_id)

    n_missing = int(index['file'].isna().sum())
    if n_missing:
        logger.warning(f'{n_missing} metadata records have no matching sample file and were dropped.')
        index = index.dropna(subset=['file'])
    if index.empty:
        raise ValueError(f'No sample file in "{samples_dir}" could be matched against "{METADATA_CSV_FILENAME}".')

    return index.sort_values('date').reset_index(drop=True)


def load_sample_tensor(file_path: str, variable_names: List[str]) -> np.ndarray:
    """Load one sample checkpoint and normalize it to a ``[V, T, H, W]`` float32 array.

    Supported layouts:
    - dict keyed by variable names (or by ``variable_1``..``variable_N``), each entry ``[T, H, W]`` or ``[H, W]``;
    - a single stacked tensor ``[V, T, H, W]``, ``[T, V, H, W]`` or ``[V, H, W]`` (``V`` matched against
      ``len(variable_names)``).

    Args:
        file_path: Path to the ``.pt`` checkpoint.
        variable_names: Ordered variable names from the dataset metadata.

    Returns:
        Array of shape ``[V, T, H, W]`` (``T = 1`` for daily fields).

    Raises:
        ValueError: If the checkpoint layout cannot be interpreted.
    """
    payload = torch.load(file_path, map_location='cpu')
    num_variables = len(variable_names)

    if isinstance(payload, dict):
        # resolve each variable either by its name or by its positional variable_i key
        per_variable = []
        for i, name in enumerate(variable_names):
            if name in payload:
                entry = payload[name]
            elif f'variable_{i + 1}' in payload:
                entry = payload[f'variable_{i + 1}']
            else:
                raise ValueError(f'Variable "{name}" not found in sample "{file_path}" (keys: {list(payload)}).')
            entry = torch.as_tensor(entry).float()
            if entry.ndim == 2:                                 # [H, W] -> [1, H, W]
                entry = entry.unsqueeze(0)
            if entry.ndim != 3:
                raise ValueError(f'Variable "{name}" in "{file_path}" has unsupported shape {tuple(entry.shape)}.')
            per_variable.append(entry)

        # broadcast static fields (T = 1) to the common number of time steps
        n_steps = max(entry.shape[0] for entry in per_variable)
        per_variable = [
            entry if entry.shape[0] == n_steps else entry.expand(n_steps, -1, -1)
            for entry in per_variable
        ]
        return torch.stack(per_variable, dim=0).numpy().astype(np.float32)

    tensor = torch.as_tensor(payload).float()
    if tensor.ndim == 4:
        if tensor.shape[0] == num_variables:                    # [V, T, H, W]
            return tensor.numpy().astype(np.float32)
        if tensor.shape[1] == num_variables:                    # [T, V, H, W]
            return tensor.permute(1, 0, 2, 3).numpy().astype(np.float32)
    if tensor.ndim == 3 and tensor.shape[0] == num_variables:   # [V, H, W] -> [V, 1, H, W]
        return tensor.unsqueeze(1).numpy().astype(np.float32)

    raise ValueError(
        f'Cannot interpret sample "{file_path}" with shape {tuple(tensor.shape)} '
        f'for {num_variables} variables.'
    )


SPLIT_NAMES = ('train', 'valid', 'test')


def load_split_config(split_config_path: str) -> dict:
    """Load and validate the train/valid/test split specification (see config/split/split.yaml).

    Args:
        split_config_path: Path to the split YAML.

    Returns:
        The parsed split config dict.

    Raises:
        ValueError: If the ``method`` is unknown or the spec for the selected method is missing/malformed.
    """
    with open(split_config_path) as handle:
        config = yaml.safe_load(handle)

    method = config.get('method')
    if method not in ('by_year', 'by_sample_id'):
        raise ValueError(f'Split config "method" must be "by_year" or "by_sample_id", got "{method}".')
    if method not in config:
        raise ValueError(f'Split config selects method "{method}" but has no "{method}" section.')

    spec = config[method]
    missing = [name for name in SPLIT_NAMES if name not in spec]
    if missing:
        raise ValueError(f'Split config "{method}" section is missing splits {missing}.')
    return config


def assign_splits_by_year(dates: pd.Series, by_year: Dict[str, List[int]]) -> pd.Series:
    """Assign each sample to a split by the calendar year of its date.

    Args:
        dates: Series of datetime64 sample dates.
        by_year: Mapping split name -> list of calendar years.

    Returns:
        Series of strings in {'train', 'valid', 'test'} with NaN for years not listed in any split.

    Raises:
        ValueError: If a year appears in more than one split.
    """
    year_to_split: Dict[int, str] = {}
    for split_name in SPLIT_NAMES:
        for year in by_year[split_name]:
            if year in year_to_split:
                raise ValueError(f'Year {year} is assigned to both "{year_to_split[year]}" and "{split_name}".')
            year_to_split[int(year)] = split_name
    return dates.dt.year.map(year_to_split).astype(object)


def assign_splits_by_sample_id(sample_ids: pd.Series, by_sample_id: Dict[str, List[List[int]]]) -> pd.Series:
    """Assign each sample to a split by inclusive [first, last] ranges of its integer sample id.

    Samples whose id falls outside every range are left unassigned (NaN) and dropped by the caller rather than
    raising: that is what lets the smoke tiers name a handful of specific days
    (config/split/split_smoke_{cpu,gpu}.yaml) and ignore the other ~5800.

    Args:
        sample_ids: Series of integer sample ids (the number embedded in each .pt filename).
        by_sample_id: Mapping split name -> list of inclusive [first, last] id ranges.

    Returns:
        Series of strings in {'train', 'valid', 'test'} with NaN for ids outside every range.

    Raises:
        ValueError: If a sample id falls in ranges of more than one split.
    """
    splits = pd.Series(np.nan, index=sample_ids.index, dtype=object)
    for split_name in SPLIT_NAMES:
        for first, last in by_sample_id[split_name]:
            in_range = (sample_ids >= first) & (sample_ids <= last)
            clash = in_range & splits.notna() & (splits != split_name)
            if clash.any():
                raise ValueError(f'Sample-id range [{first}, {last}] of "{split_name}" overlaps another split.')
            splits[in_range] = split_name
    return splits


def assign_splits_from_config(index: pd.DataFrame, split_config: dict) -> pd.Series:
    """Assign the train/valid/test split to a sample index according to a loaded split config.

    Applies the selected ``method``; when ``cross_check`` is true and both specifications are present, the
    two assignments are required to agree on every sample (guarding against dataset drift / re-numbering).
    The smoke tiers set ``cross_check: false`` precisely because they name sample-id ranges that are a strict
    subset of one year, which no ``by_year`` specification can reproduce.

    Args:
        index: Sample index with columns ``date`` (datetime64) and ``sample_id`` (int).
        split_config: Parsed split config (see :func:`load_split_config`).

    Returns:
        Series of strings in {'train', 'valid', 'test'} aligned with ``index`` (NaN for unassigned samples).

    Raises:
        ValueError: If the cross-check is requested and the two specifications disagree.
    """
    method = split_config['method']
    if method == 'by_year':
        splits = assign_splits_by_year(index['date'], split_config['by_year'])
    else:
        splits = assign_splits_by_sample_id(index['sample_id'], split_config['by_sample_id'])

    # cross-check the two specifications agree, when both are available and requested
    if split_config.get('cross_check', False) and 'by_year' in split_config and 'by_sample_id' in split_config:
        other = (
            assign_splits_by_sample_id(index['sample_id'], split_config['by_sample_id']) if method == 'by_year'
            else assign_splits_by_year(index['date'], split_config['by_year'])
        )
        both_assigned = splits.notna() & other.notna()
        disagreements = both_assigned & (splits != other)
        if disagreements.any():
            n = int(disagreements.sum())
            example = index.loc[disagreements, ['sample_id', 'date']].iloc[0].to_dict()
            raise ValueError(
                f'Split cross-check failed: by_year and by_sample_id disagree on {n} sample(s), '
                f'e.g. {example} ({splits[disagreements].iloc[0]} vs {other[disagreements].iloc[0]}). '
                f'Set cross_check: false in the split config to bypass.'
            )
    return splits


def compute_feature_stats(
        index: pd.DataFrame,
        variable_names: List[str],
        feature_names: List[str],
        max_days: Optional[int] = None,
        seed: int = 0,
        eps: float = 1e-6,
        feature_layout: str = 'time_major'
) -> Dict[str, Dict[str, float]]:
    """Compute per-variable feature mean/std by streaming (a subsample of) the train sample checkpoints.

    Feature scaling is owned by the tuning stage (the resulting statistics are baked into the model checkpoint
    as normalization buffers), NOT loaded from the dataset directory: the dataset-side scalers (e.g.
    ``scaler_full``) are deprecated.

    When the index carries materialized per-day feature files (``feature_file`` column written by the
    preparation stage), those are streamed instead of the full sample checkpoints (much less I/O: only the
    predictor channels are read).

    Accumulation is in float64 whatever the stored precision, so ``feature-dtype: float16`` (which halves the
    ``features/`` footprint and the per-item read) cannot affect the normalization that gets baked into every
    checkpoint.

    Args:
        index: Train-split rows of the sample index (columns ``date`` and ``file``, optionally ``feature_file``).
        variable_names: Ordered variable names from the dataset metadata.
        feature_names: Subset of variables used as predictors.
        max_days: Optional cap on the number of days streamed (uniform random subsample); mean/std converge
            quickly, so a few hundred days are plenty.
        seed: Seed of the day subsample.
        eps: Lower floor of the standard deviations.
        feature_layout: Storage layout of the materialized feature files (``feature_layout`` of
            prepared_config.json): ``time_major`` ``[T, Vf, H, W]`` or ``variable_major`` ``[Vf, T, H, W]``.

    Returns:
        Dict ``{'mean': {name: float}, 'std': {name: float}}`` over the requested features (NaN-aware).
    """
    rows = index
    if max_days is not None and len(rows) > max_days:
        chosen = np.random.default_rng(seed).choice(len(rows), size=max_days, replace=False)
        rows = rows.iloc[np.sort(chosen)]
        logger.info(f'Computing feature statistics on a {max_days}-day subsample of the train split.')

    use_feature_files = 'feature_file' in rows.columns and rows['feature_file'].notna().all()
    positions = {name: variable_names.index(name) for name in feature_names}
    sums = {name: 0.0 for name in feature_names}
    square_sums = {name: 0.0 for name in feature_names}
    counts = {name: 0 for name in feature_names}
    for _, row in rows.iterrows():
        if use_feature_files:
            # materialized features hold the feature_names channels (possibly reduced precision)
            day_features = np.load(row['feature_file'])
            if feature_layout == 'time_major':                              # [T, Vf, H, W] -> [Vf, T, H, W]
                day_features = day_features.transpose(1, 0, 2, 3)
        else:
            sample = load_sample_tensor(row['file'], variable_names)
            day_features = None
        for feature_position, name in enumerate(feature_names):
            raw = day_features[feature_position] if use_feature_files else sample[positions[name]]
            # float64 for BOTH paths: the feature files may be float16 and the sample checkpoints are float32,
            # and a float32 running sum over ~360k cells/day is not a precision the scalers should inherit
            values = np.asarray(raw, dtype=np.float64)
            finite = np.isfinite(values)
            sums[name] += float(values[finite].sum())
            square_sums[name] += float((values[finite] ** 2).sum())
            counts[name] += int(finite.sum())

    means = {name: sums[name] / max(counts[name], 1) for name in feature_names}
    stds = {
        name: float(max(np.sqrt(max(square_sums[name] / max(counts[name], 1) - means[name] ** 2, 0.0)), eps))
        for name in feature_names
    }
    return {'mean': means, 'std': stds}


def load_prepared_artifacts(prepared_path: str) -> Tuple[dict, pd.DataFrame, dict]:
    """Load the artifacts written by the preparation stage (``prepare_modeling`` / ``prepare_classification``).

    Args:
        prepared_path: Output directory of the preparation stage.

    Returns:
        Tuple (prepared_config, split_index, target_stats); split_index has columns ``date`` (datetime64),
        ``sample_id``, ``file``, ``target_file`` (absolute) and ``split``, plus ``feature_file`` (absolute)
        when the preparation stage materialized per-day feature files and ``upstream_file`` in residual mode.
        Legacy artifacts are normalized in place (the deprecated mode name is mapped to the canonical one).
    """
    with open(os.path.join(prepared_path, 'prepared_config.json')) as handle:
        prepared_config = json.load(handle)
    with open(os.path.join(prepared_path, 'target_stats.json')) as handle:
        target_stats = json.load(handle)

    # normalize artifacts written before the mode rename
    for payload in (prepared_config, target_stats):
        if payload.get('mode') in LEGACY_MODE_ALIASES:
            payload['mode'] = LEGACY_MODE_ALIASES[payload['mode']]

    split_index = pd.read_csv(os.path.join(prepared_path, 'split_index.csv'), parse_dates=['date'])
    split_index['target_file'] = split_index['target_filename'].map(
        lambda name: os.path.join(prepared_path, 'targets', name)
    )
    if 'feature_filename' in split_index.columns:
        split_index['feature_file'] = split_index['feature_filename'].map(
            lambda name: os.path.join(prepared_path, 'features', name) if isinstance(name, str) else None
        )
    # residual (diffusion) mode: per-day upstream-prediction maps written by the preparation stage
    if 'upstream_filename' in split_index.columns:
        split_index['upstream_file'] = split_index['upstream_filename'].map(
            lambda name: os.path.join(prepared_path, 'upstream', name) if isinstance(name, str) else None
        )
    return prepared_config, split_index, target_stats


def compute_upstream_stats(
        index: pd.DataFrame,
        max_days: Optional[int] = None,
        seed: int = 0,
        eps: float = 1e-6
) -> Dict[str, float]:
    """Mean/std of the train upstream-prediction maps (the appended ``upstream`` conditioning channel of the
    diffusion residual model), streamed from the per-day ``upstream/<date>.npy`` files; NaN-aware, accumulated
    in float64. Used by the tuning stage to standardize the upstream channel like any other feature channel.

    Args:
        index: Train-split rows with a ``upstream_file`` column.
        max_days: Optional cap on the number of train days streamed (uniform random subsample).
        seed: Seed of the day subsample.
        eps: Lower floor of the standard deviation.

    Returns:
        Dict ``{'mean': float, 'std': float}``.
    """
    rows = index
    if max_days is not None and len(rows) > max_days:
        chosen = np.random.default_rng(seed).choice(len(rows), size=max_days, replace=False)
        rows = rows.iloc[np.sort(chosen)]
    if 'upstream_file' not in rows.columns or not rows['upstream_file'].notna().all():
        raise ValueError('Upstream-channel statistics requested but the train index has no usable upstream_file.')

    total, total_square, count = 0.0, 0.0, 0
    for _, row in rows.iterrows():
        values = np.load(row['upstream_file']).astype(np.float64)
        finite = np.isfinite(values)
        total += float(values[finite].sum())
        total_square += float((values[finite] ** 2).sum())
        count += int(finite.sum())

    mean = total / max(count, 1)
    std = float(max(np.sqrt(max(total_square / max(count, 1) - mean ** 2, 0.0)), eps))
    return {'mean': float(mean), 'std': std}
