"""Measure the EXACT sparsity of the ATDnet lightning channel over the whole dataset, by season and by year.

This script is the provenance of the sparsity table in [README.md](../README.md). It exists because the repo carried
a wrong figure — "~99.93 % of cells are zero" — in about forty places, asserted rather than measured, and the error was
a factor of 6 on the hourly base rate and a factor of 67 on the daily one. Anything quoting a sparsity number should
quote this script's output, and should say WHICH of the six quantities it means.

    export DATA_ROOT=/path/to/era5_postprocess
    python scripts/sparsity.py                       # table to stdout
    python scripts/sparsity.py sparsity.json         # table to stdout, counts to JSON

⚠️ It reads every sample: 5843 files x 8.7 MB is ~51 GB of I/O, about 11 minutes on a warm local disk. There is no
sampling mode on purpose — the point of this script is to be exact, and a stratified estimate is one line of edit away
(`files = files[::12]`) if you want a 1-minute approximation. Every 12th day agrees with the exact figures to within
0.03 percentage points, so the approximation is fine for a sanity check and not for a documented table.

SIX QUANTITIES, AND THEY ARE NOT INTERCHANGEABLE. Conflating them is what produced the original error:

    raw == 0          cells with no stroke at all
    hourly == 1       exactly one stroke -- the observational noise `hourly-threshold: 2` exists to drop
    hourly >= 2       the HOURLY OCCURRENCE event: the positive class of `mode: hourly`
    hourly target 0   its complement, 1 - P(hourly >= 2)
    daily target 0    cells whose DAILY target (the count of qualifying hours, 0-24) is zero
    daily positive    its complement -- the daily occurrence base rate

The daily figures are far less sparse than the hourly ones, because a cell needs only ONE qualifying hour in 24 to be a
positive day. That is the whole reason the two tasks want different metric suites.
"""
import csv
import json
import os
import sys

import torch

# The dataset root, reached through the environment exactly as every stage and config does -- NEVER hardcoded.
# (`job_scripts/` may hardcode a path because it is gitignored machine-specific launch material; this is not.)
DATA_ROOT = os.environ.get('DATA_ROOT')

# The denoising cutoff, matching `hourly-threshold: 2` in every shipped pipeline: an hour counts as a lightning-hour
# only when it carries at least this many strokes. Changing it changes the target, so it is not a free parameter.
THRESHOLD = 2

SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
          6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
KEYS = ('cell_hours', 'raw_zero', 'exactly_one', 'ge_threshold', 'daily_cells', 'daily_zero')


def load_dates(data_root):
    """``{sample id: 'YYYY-MM-DD'}`` from ``metadata.csv`` — the only way to get a sample's season."""
    dates = {}
    with open(os.path.join(data_root, 'metadata.csv')) as handle:
        for row in csv.DictReader(handle):
            dates[int(row['id'])] = row['date']
    return dates


def sample_counts(path):
    """The six raw counts for one day's sample.

    A sample is a single tensor ``[24 hours, 6 variables, 101 lat, 149 lon]`` and channel 5 is ``lightnings``, the
    stroke count per cell-hour. The variable ORDER comes from ``metadata.json`` (``variable_6: lightnings``); it is
    read positionally here because the whole point is to touch the raw file rather than the prepared artifacts.
    """
    lightning = torch.load(path, weights_only=False)[:, 5].numpy()
    qualifying = lightning >= THRESHOLD
    daily = qualifying.sum(axis=0)                       # [lat, lon] lightning-hours, 0-24: the DAILY target
    return {
        'cell_hours': lightning.size,
        'raw_zero': int((lightning == 0).sum()),
        'exactly_one': int((lightning == 1).sum()),
        'ge_threshold': int(qualifying.sum()),
        'daily_cells': daily.size,
        'daily_zero': int((daily == 0).sum()),
    }


def measure(data_root, progress_every=500):
    """Accumulate the counts over every sample, bucketed by ``all`` / season / year. Returns ``(buckets, n_days)``."""
    dates = load_dates(data_root)
    files = sorted(os.listdir(os.path.join(data_root, 'samples')))
    years = sorted({date[:4] for date in dates.values()})

    buckets = {name: dict.fromkeys(KEYS, 0) for name in ('all', 'DJF', 'MAM', 'JJA', 'SON', *years)}
    n_days = dict.fromkeys(buckets, 0)

    for index, name in enumerate(files):
        counts = sample_counts(os.path.join(data_root, 'samples', name))
        date = dates[int(name.split('_')[1].split('.')[0])]
        for label in ('all', SEASON[int(date[5:7])], date[:4]):
            for key, value in counts.items():
                buckets[label][key] += value
            n_days[label] += 1
        if progress_every and index % progress_every == 0:
            print(f'  {index}/{len(files)}', file=sys.stderr, flush=True)
    return buckets, n_days


def format_table(buckets, n_days):
    """The markdown table as it appears in README.md, so the two cannot drift by hand-transcription."""
    lines = ['| group | days | cell-hours | raw == 0 | hourly == 1 | hourly >= 2 | hourly target 0 | daily target 0 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    years = sorted(label for label in buckets if label.isdigit())
    for label in ('all', 'DJF', 'MAM', 'JJA', 'SON', *years):
        bucket = buckets[label]
        total, daily_cells = bucket['cell_hours'], bucket['daily_cells']
        if not total:
            continue
        name = f'**{label}**' if label == 'all' else label
        lines.append(
            f'| {name} | {n_days[label]:,} | {total:,} '
            f'| {bucket["raw_zero"] / total * 100:.4f} % '
            f'| {bucket["exactly_one"] / total * 100:.4f} % '
            f'| {bucket["ge_threshold"] / total * 100:.4f} % '
            f'| {(total - bucket["ge_threshold"]) / total * 100:.4f} % '
            f'| {bucket["daily_zero"] / daily_cells * 100:.4f} % |'
        )
    return '\n'.join(lines)


def main():
    if not DATA_ROOT:
        raise SystemExit('DATA_ROOT is not set. export it to the dataset root (metadata.json, metadata.csv, samples/).')
    if not os.path.isdir(os.path.join(DATA_ROOT, 'samples')):
        raise SystemExit(f'DATA_ROOT="{DATA_ROOT}" has no samples/ directory: not a dataset root.')

    buckets, n_days = measure(DATA_ROOT)
    print(format_table(buckets, n_days))
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as handle:
            json.dump({'threshold': THRESHOLD, 'buckets': buckets, 'n_days': n_days}, handle, indent=2)
        print(f'\ncounts -> {sys.argv[1]}', file=sys.stderr)


if __name__ == '__main__':
    main()
