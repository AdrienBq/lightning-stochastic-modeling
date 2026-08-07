"""Torch dataset assembling (ERA5 features, lightning target) map pairs from the prepared artifacts.

Features come from one of two sources:
- Materialized per-day feature ``.npy`` files written by the preparation stage (``feature_file`` column of the
  split index): memory-mapped, time-major ``[T, Vf, H, W]``, so an hourly item is a single small contiguous read.
  This is the fast path — in ``hourly`` mode it avoids deserializing a full multi-variable day checkpoint
  per item (a ~25x I/O amplification under shuffling).
- Fallback: lazy reads of the original batta_torch ``.pt`` sample checkpoints (one per day), with a small
  per-worker file cache amortizing repeated reads of the same day's checkpoint. Under random shuffling the cache
  is nearly useless, so hourly-mode training should pair this path with :class:`DayGroupedShuffleSampler`.

Targets are always read from the lightweight per-day target ``.npy`` files of the preparation stage, verbatim: the
mode decides what they CONTAIN, and the dataset only decides how they are sliced.
- ``daily`` — one item per day, target ``[H, W]`` of lightning-hours in 0-24. The REGRESSION task.
- ``hourly`` — each day expands into ``hours_per_day`` items, target ``[H, W]`` of 0/1 occurrence. The
  CLASSIFICATION task.
Both are written by the preparation stage under one denoising cutoff (``hourly_threshold``), so neither is derived
here and there is nothing to threshold at read time.

Features are returned RAW (unscaled, NaNs preserved): standardization and NaN imputation are owned by the model
(per-channel normalization buffers fitted at tuning time and persisted in the checkpoint), since the dataset-side
scalers (e.g. ``scaler_full``) are deprecated.

HOW THE DAILY INPUT CHANNELS ARE BUILT — ``feature_aggregation``, which is DAILY-MODE ONLY (an hourly item is
already a single hour, so the key is never read there):
- ``hourly_stack`` — the option that does NOT aggregate. The day's T hourly maps are concatenated onto the CHANNEL
  axis, giving ``Vf * T`` channels (5 variables x 24 hours = 120 here), so the network sees the whole diurnal cycle
  and learns its own weighting over hours rather than being handed a summary. A static field such as ``lsm`` does
  contribute T identical channels, but it was already expanded upstream — per variable by ``load_sample_tensor``,
  or at prepare time for materialized features — so every variable shares one T axis by the time this module reads
  it. The ``shape[T] == 1`` broadcast below is the remaining WHOLE-DAY case, where the stored array has no usable
  time axis at all.
- ``daily_mean`` — the aggregating option: mean over the day, ``Vf`` channels.
Both names read backwards (the key says "aggregation" yet its default value aggregates nothing, and the value that
does not aggregate is the one named "hourly"), but they are the on-disk vocabulary in ``prepared_config.json`` and
are kept as-is. Read ``hourly_stack`` as "keep the hours, as channels".

WRITTEN BY THE PREPARATION STAGE, NOT CONFIGURED — two fields of ``prepared_config.json`` that look like tunables
and appear nowhere in ``config/``:
- ``hours_per_day`` — DISCOVERED FROM THE DATA: the first axis of a day's lightning tensor (24 for this dataset).
- ``feature_layout`` — DERIVED FROM THE MODE, purely an I/O optimization: ``time_major [T, Vf, H, W]`` for hourly
  (one hour is then a contiguous read) and ``variable_major [Vf, T, H, W]`` for daily (the channel reshape above is
  then free). Either layout yields the SAME channel order; only the copy cost differs. It is stored as literal
  ``None`` when features were not materialized, which is why it is read with ``or 'time_major'`` below rather than
  through a dict default.

Residual (diffusion) mode: when the preparation stage was given an upstream model (``residual_target`` in the
prepared config, with per-day ``upstream/<date>.npy`` prediction maps), the upstream prediction is appended as an
extra conditioning channel of ``x`` AND returned as a third item, so the diffusion module can form the residual
target (``observed - upstream``) and add the upstream prediction back at inference. ``__getitem__`` then yields
``(x, y, upstream)`` instead of ``(x, y)``.
"""
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.utils.io.data import MODE_DAILY, MODE_HOURLY, load_sample_tensor, normalize_mode


class LightningMapsDataset(Dataset):
    """Map-to-map dataset for one split of the prepared data.

    Args:
        index: Split-filtered rows of ``split_index.csv`` with columns ``date``, ``file`` (sample checkpoint),
            ``target_file`` (per-day target ``.npy``) and, when the preparation stage materialized features,
            ``feature_file`` (per-day feature ``.npy``, memory-mapped at read time).
        prepared_config: Contents of ``prepared_config.json`` from the preparation stage.
        cache_size: Number of day checkpoints kept in the in-worker cache (fallback ``.pt`` path only).
    """

    def __init__(
            self,
            index: pd.DataFrame,
            prepared_config: dict,
            cache_size: int = 2
    ):
        self.index = index.reset_index(drop=True)
        self.mode = normalize_mode(prepared_config['mode'])     # raises on an unknown name, so no re-check here
        self.feature_aggregation = prepared_config.get('feature_aggregation', 'hourly_stack')
        if self.mode == MODE_DAILY and self.feature_aggregation not in ('hourly_stack', 'daily_mean'):
            raise ValueError(f'Unknown feature aggregation "{self.feature_aggregation}".')
        self.feature_names: List[str] = prepared_config['features']
        self.variable_names: List[str] = prepared_config['variable_names']
        self.feature_positions = [self.variable_names.index(name) for name in self.feature_names]
        self.hours_per_day = int(prepared_config['hours_per_day'])
        self.cache_size = cache_size
        self._cache: 'OrderedDict[str, np.ndarray]' = OrderedDict()
        self.uses_materialized_features = bool(
            len(self.index) > 0
            and 'feature_file' in self.index.columns
            and self.index['feature_file'].notna().all()
        )
        # storage layout of the materialized files: 'time_major' [T, Vf, H, W] (hourly prepares; an hourly
        # slice is one contiguous read) or 'variable_major' [Vf, T, H, W] (daily prepares; the stacked channel
        # view is a free reshape). Directories materialized before layouts existed are time-major.
        self.feature_layout = prepared_config.get('feature_layout') or 'time_major'

        # residual (diffusion) mode: an upstream model's per-item prediction is appended as an extra conditioning
        # channel and returned separately so the module can build the residual target / add it back at inference
        self.residual_target = bool(prepared_config.get('residual_target', False))
        upstream_present = 'upstream_file' in self.index.columns and (
            not len(self.index) or self.index['upstream_file'].notna().all()
        )
        if self.residual_target and not upstream_present:
            raise ValueError(
                'prepared_config marks residual_target=true but the split index has no usable "upstream_file" '
                'column; re-run the preparation stage with an upstream model.'
            )
        self.uses_upstream = self.residual_target and len(self.index) > 0

        # one item per day (daily mode) or per (day, hour) pair (hourly mode)
        if self.mode == MODE_HOURLY:
            self.items = [
                (row_position, hour)
                for row_position in range(len(self.index))
                for hour in range(self.hours_per_day)
            ]
        else:
            self.items = [(row_position, None) for row_position in range(len(self.index))]

    @property
    def in_channels(self) -> int:
        """Number of input channels seen by the network."""
        return len(self.channel_variable_names)

    @property
    def channel_variable_names(self) -> List[str]:
        """Variable name backing each input channel, in channel order (variable-major when hourly-stacked);
        used by the tuning stage to expand per-variable normalization statistics to per-channel buffers. In
        residual mode the appended upstream-prediction channel is named ``upstream`` and comes last."""
        if self.mode == MODE_DAILY and self.feature_aggregation == 'hourly_stack':
            names = [name for name in self.feature_names for _ in range(self.hours_per_day)]
        else:
            names = list(self.feature_names)
        if self.residual_target:
            names = names + ['upstream']
        return names

    def items_frame(self) -> pd.DataFrame:
        """One row per item (in iteration order) with columns ``date`` and ``hour`` (NaN in daily mode)."""
        rows = self.index.iloc[[row for row, _ in self.items]].reset_index(drop=True)
        rows['hour'] = [hour for _, hour in self.items]
        return rows

    def _load_features(self, file_path: str) -> np.ndarray:
        if file_path in self._cache:
            self._cache.move_to_end(file_path)
            return self._cache[file_path]
        sample = load_sample_tensor(file_path, self.variable_names)[self.feature_positions]   # [Vf, T, H, W]
        self._cache[file_path] = sample
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sample

    def _item_features_materialized(self, row: pd.Series, hour: Optional[int]) -> np.ndarray:
        """Read the item's feature map from the materialized per-day ``.npy`` (memory-mapped); hourly items
        copy a single time slice, daily items read the whole day. The output channel order is ALWAYS
        variable-major (matching the fallback path and channel_variable_names) regardless of the storage
        layout; with the mode-matched layout no per-item transpose-copy is needed."""
        day = np.load(row['feature_file'], mmap_mode='r')
        time_axis = 0 if self.feature_layout == 'time_major' else 1
        if self.mode == MODE_HOURLY:                                        # np.array copies out of the read-only
            time_position = min(hour, day.shape[time_axis] - 1)             # map (static daily fields have T = 1)
            hour_slice = day[time_position] if time_axis == 0 else day[:, time_position]
            return np.array(hour_slice)                                     # [Vf, H, W] (native storage dtype)
        full = np.array(day)                                               # native storage dtype (e.g. float16)
        if self.feature_aggregation == 'daily_mean':
            return full.mean(axis=time_axis)                                # [Vf, H, W]
        if full.shape[time_axis] == 1:
            shape = list(full.shape)
            shape[time_axis] = self.hours_per_day
            full = np.broadcast_to(full, shape)
        if time_axis == 0:                                                  # time-major storage: reorder to
            full = full.transpose(1, 0, 2, 3)                               # variable-major (strided copy below)
        return full.reshape(-1, *full.shape[2:])                            # [Vf * T, H, W]

    def _item_upstream(self, row: pd.Series, hour: Optional[int]) -> np.ndarray:
        """Read the item's upstream-prediction map (residual mode) from the per-day ``upstream/<date>.npy``
        (memory-mapped); hourly items copy a single time slice, daily items read the whole ``[H, W]`` map."""
        upstream = np.load(row['upstream_file'], mmap_mode='r')             # [H, W] daily | [T, H, W] hourly
        if self.mode == MODE_HOURLY:
            time_position = min(hour, upstream.shape[0] - 1)
            return np.array(upstream[time_position], dtype=np.float32)      # [H, W]
        return np.array(upstream, dtype=np.float32)                        # [H, W]

    def _item_features_checkpoint(self, row: pd.Series, hour: Optional[int]) -> np.ndarray:
        """Fallback: slice the item's feature map out of the (cached) full day checkpoint."""
        features = self._load_features(row['file'])                         # [Vf, T, H, W]
        if self.mode == MODE_HOURLY:
            time_position = min(hour, features.shape[1] - 1)                # static daily fields have T = 1
            return features[:, time_position]                               # [Vf, H, W]
        if self.feature_aggregation == 'daily_mean':
            return features.mean(axis=1)                                    # [Vf, H, W]
        stacked = np.broadcast_to(
            features, (features.shape[0], self.hours_per_day) + features.shape[2:]
        ) if features.shape[1] == 1 else features
        return stacked.reshape(-1, *features.shape[2:])                     # [Vf * T, H, W]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int):
        row_position, hour = self.items[item]
        row = self.index.iloc[row_position]

        x = self._item_features_materialized(row, hour) if self.uses_materialized_features \
            else self._item_features_checkpoint(row, hour)

        # residual mode: append the upstream prediction as the last conditioning channel (kept raw, like the
        # features — the model standardizes it through its checkpoint buffers) and also return it separately
        upstream = None
        if self.uses_upstream:
            upstream = self._item_upstream(row, hour)                       # [H, W]
            # match x's storage dtype so appending the conditioning channel does not upcast the whole stack
            x = np.concatenate([x, upstream[None].astype(x.dtype, copy=False)], axis=0)   # [..., +1, H, W]

        # features stay raw (NaNs included) AND in their on-disk storage dtype (e.g. float16): keeping them
        # narrow halves the per-batch collate / pin-memory / host->device copy, the loader's dominant cost. The
        # model casts to float32 inside its normalization step (the per-channel buffers are float32).
        x = np.ascontiguousarray(x)

        target = np.load(row['target_file']).astype(np.float32)
        if self.mode == MODE_HOURLY:
            target = target[hour]

        if self.uses_upstream:
            return (
                torch.from_numpy(x),
                torch.from_numpy(target),
                torch.from_numpy(np.ascontiguousarray(upstream, dtype=np.float32))
            )
        return torch.from_numpy(x), torch.from_numpy(target)


class DayGroupedShuffleSampler(Sampler[int]):
    """Shuffled sampler for hourly-mode training off the fallback ``.pt`` path: days are visited in random order
    (hours shuffled within each day) but each day's items stay contiguous, so the per-worker day-checkpoint cache
    keeps hitting instead of reloading a full multi-variable day per item. Slightly less mixed batches than fully
    uniform shuffling; unnecessary (and unused) when features are materialized.

    ⚠️ NOT DEAD CODE, though it currently never runs. ``tuning.py`` constructs it as the ``sampler=`` argument of the
    TRAIN ``DataLoader`` at both fit sites (the sweep trial and the best-config retrain), under the guard
    ``mode == MODE_HOURLY and not uses_materialized_features``. Every pipeline config sets
    ``materialize-features: true``, so that guard is false today — it becomes live the first time an hourly run
    skips materialization, which is a plausible choice given the cost of materializing hourly features for 5843
    days.
    """

    def __init__(self, dataset: LightningMapsDataset):
        super().__init__()
        if dataset.mode != MODE_HOURLY:
            raise ValueError('DayGroupedShuffleSampler only applies to hourly-mode datasets.')
        self.n_days = len(dataset.index)
        self.hours_per_day = dataset.hours_per_day

    def __len__(self) -> int:
        return self.n_days * self.hours_per_day

    def __iter__(self) -> Iterator[int]:
        for day in torch.randperm(self.n_days).tolist():
            base = day * self.hours_per_day
            for hour in torch.randperm(self.hours_per_day).tolist():
                yield base + hour


def build_split_datasets(
        split_index: pd.DataFrame,
        prepared_config: dict,
        splits: Optional[List[str]] = None
) -> Dict[str, LightningMapsDataset]:
    """Build one dataset per requested split from the full split index.

    Datasets only — :class:`DayGroupedShuffleSampler` is deliberately NOT built here. A sampler is a ``DataLoader``
    argument, and it applies to the train split alone, so it belongs with the loaders in ``tuning.py`` (see that
    class's docstring for the guard).
    """
    splits = splits or ['train', 'valid', 'test']
    return {
        split: LightningMapsDataset(split_index[split_index['split'] == split], prepared_config)
        for split in splits
    }
