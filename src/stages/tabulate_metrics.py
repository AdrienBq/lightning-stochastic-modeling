"""Pipeline stage: assemble the families' evaluation metrics into ONE comparison table (CSV).

Reads the flat metrics JSONs the `evaluate` stage wrote — one per family — and puts them in a single table, families
as rows and metric keys as columns. Driven by `**kwargs`: each extra `--<Family-Label> <path-to-metrics.json>` flag
adds one row, so the stage needs no knowledge of which families exist.

**This stage is where the merge is checked.** `config/eval/probabilistic_eval.yaml` runs the same `evaluate` with the
same `metrics-config` three times and then tabulates the result; if the three JSONs did not agree on their metric
keys, the pipelines would not actually be merged. The columns here are identical across families BY CONSTRUCTION
(one DataFrame has one column set), so the CSV alone cannot show a disagreement — what shows it is the NaN pattern,
which is why the per-family missing-key list is LOGGED rather than left for a human to spot in the table.

Expected NaNs, which are correct: the deterministic U-net has no `crps` / `spread_skill_ratio` /
`rank_histogram_reliability` because it emits no ensemble, and `dice_*` / `explained_deviance` are absent from every
daily-mode run because there is no occurrence head to produce a probability. A NaN in a *point* metric is not
expected and is what the log is for.

⚠️ **The module is named `tabulate_metrics`, not `tabulate`, on purpose.** Stage scripts run as
`python src/stages/<name>.py`, which puts `src/stages` on `sys.path`, so a module named `tabulate.py` here would
SHADOW the PyPI `tabulate` package that torch imports internally (`torch._dynamo.utils`) — a stage that fails on an
unrelated import. The Fire-wrapped function is still `tabulate`; only the file name differs.

**Labels come from the flag names**, with Fire's hyphen substitution undone: Fire maps `--MC-Dropout` to the kwargs
key `MC_Dropout`, and `_display` turns it back into `MC-Dropout`. So the row index reads exactly as the flag was
written in the YAML, and a new family needs no code change here.

Usage (standalone)::

    python src/stages/tabulate_metrics.py \\
        --output-path $OUTPUT_ROOT/comparison/metrics_comparison_test.csv \\
        --Deterministic-UNet $OUTPUT_ROOT/comparison/evaluation/deterministic_unet/test_metrics.json \\
        --MC-Dropout         $OUTPUT_ROOT/comparison/evaluation/mc_dropout/test_metrics.json \\
        --Diffusion          $OUTPUT_ROOT/comparison/evaluation/diffusion/test_metrics.json \\
        [--selected-metrics average_precision_occurrence,mae_cond_pos,crps]
"""
import json
import logging
import os
from typing import List, Optional, Union

from fire import Fire

from __init__ import root_path, console_handler

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


def _display(key: str) -> str:
    """Undo Fire's hyphen substitution: the label IS the flag name, so `--MC-Dropout` (kwargs key `MC_Dropout`)
    labels its row `MC-Dropout`. Branch A carried a hardcoded map instead, which covered `U_net` / `Diffusion_Model`
    — neither of them a label any shipped config uses — so every family in `probabilistic_eval.yaml` would have been
    mislabelled. Restoring the hyphen needs no map and cannot go stale when a family is added."""
    return key.replace('_', '-')


def _as_name_list(value: Union[str, list, tuple]) -> list:
    """Normalize a comma-separated string (or an already-parsed sequence) into a list of names — mirrors
    `evaluate._as_name_list`, so `--selected-metrics a,b` is accepted as the string YAML supplies or as the tuple
    Fire may have parsed it into."""
    if isinstance(value, str):
        return [name.strip() for name in value.split(',') if name.strip()]
    return [str(name).strip() for name in value]


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root_path, path)


def tabulate(output_path: str, selected_metrics: Optional[List[str]] = None, **kwargs) -> None:
    """Build a families x metrics comparison table from per-family metrics JSONs and write it to `output_path`.

    Args:
        output_path: CSV destination (relative to `root_path` unless absolute); parent directories are created.
        selected_metrics: Optional subset of metric names (a list, or a comma-separated string) to keep as columns.
            Names absent from EVERY family are warned about and dropped rather than fatal, which is what lets one
            selection list be reused across configs. `None` keeps the union of all metrics.
        **kwargs: family label -> metrics-JSON path. Fire maps `--MC-Dropout=p` to the key `MC_Dropout`;
            :func:`_display` restores the label. A family whose JSON is missing is warned about and skipped, so a
            partial comparison still produces a table.

    Returns:
        None. Writes the comparison CSV.
    """
    import pandas as pd

    if not kwargs:
        raise ValueError('tabulate needs at least one `--<Family-Label> <metrics.json>` argument.')

    rows = {}
    for key, path in kwargs.items():
        label = _display(key)
        absolute = _resolve(str(path))
        if not os.path.exists(absolute):
            logger.warning(f'Metrics JSON for "{label}" not found at "{path}"; skipping this family.')
            continue
        with open(absolute) as handle:
            metrics = json.load(handle)
        rows[label] = metrics
        logger.info(f'Loaded {len(metrics)} metrics for "{label}" from "{path}".')

    if not rows:
        # an empty table would be logged as a successful comparison of nothing, which is worse than failing here
        raise FileNotFoundError('tabulate found none of the provided metrics JSON paths.')

    table = pd.DataFrame.from_dict(rows, orient='index')        # rows = families, columns = union of metric keys
    table.index.name = 'model'
    table = table.reindex(sorted(table.columns), axis=1)        # stable column order for clean diffs

    if selected_metrics is not None:
        requested = _as_name_list(selected_metrics)
        present = [name for name in requested if name in table.columns]
        missing = [name for name in requested if name not in table.columns]
        if missing:
            logger.warning(f'{len(missing)} selected metric(s) absent from every family (ignored): '
                           f'{", ".join(missing)}.')
        table = table[present]

    # The asymmetries, named. The columns match by construction, so this log is the only place the table's NaN
    # pattern is stated in words -- and the only place a family missing a POINT metric (a real merge failure, unlike
    # a deterministic family missing crps) becomes visible without reading the CSV cell by cell.
    for label in table.index:
        absent = [name for name in table.columns if table.loc[label].isna()[name]]
        if absent:
            logger.info(f'"{label}" has no value for {len(absent)} of {table.shape[1]} metrics: '
                        f'{", ".join(sorted(absent))}.')

    absolute_output = _resolve(output_path)
    os.makedirs(os.path.dirname(absolute_output) or '.', exist_ok=True)
    table.to_csv(absolute_output)
    logger.info(f'Wrote {table.shape[0]} families x {table.shape[1]} metrics to "{output_path}".')


if __name__ == '__main__':
    Fire(tabulate)
