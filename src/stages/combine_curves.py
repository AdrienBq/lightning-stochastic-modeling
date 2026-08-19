"""Pipeline stage: overlay the families' evaluation curves into shared comparison figures (png + pdf).

The visual half of the cross-family comparison, and the counterpart to `tabulate_metrics`: that stage puts the
scalars in one table, this one puts the curves on one set of axes. It reads the per-figure CSVs each `evaluate` run
already wrote into its report directory — nothing is recomputed here, so a comparison costs seconds and can be
redrawn without re-running an evaluation.

Same `**kwargs` convention as `tabulate_metrics`, except each `--<Family-Label> <report-dir>` points at the family's
REPORT directory rather than at a metrics JSON. One colour per family, assigned over the sorted labels and reused
across every figure, so a line's colour means the same thing in all four.

Produced under `output_path`, each as png AND pdf:

| Figure | Read from | Note |
|---|---|---|
| `combined_psd` | `psd_curves.csv` | wavelength in km, x inverted; ±1σ band for an ensemble family |
| `combined_fss` | `fss_table.csv` | colour = family, linestyle = exceedance threshold |
| `combined_reliability` | `reliability_table.csv` | calibration + the bin populations beside it |
| `combined_roc_pr` | `roc_pr_curves.csv` (+ `roc_pr_summary.csv`) | ROC + PR on the headline event |
| `combined_rank_histogram` | `rank_histogram.csv` | ensemble families only |

**Every read is guarded and every figure self-skips.** A family missing a CSV is dropped from that figure; a figure
with no contributing family is skipped with a log line rather than emitting blank axes. This is not defensiveness for
its own sake — the asymmetries are expected and correct: the deterministic U-net produces no rank histogram, and a
daily-mode run produces neither reliability nor ROC/PR from a probability, because it has no occurrence head.

⛔ **There is no `combined_qq`.** Branch A drew one from `qq_table.csv`, a file this repo never writes: the
target-space QQ figure went with the 02a grammar in Step 2. Ported unchanged it would have logged *"no model had a
usable qq_table.csv; skipped"* on every run forever — a permanent warning about a file nobody is meant to produce.

Usage (standalone)::

    python src/stages/combine_curves.py \\
        --output-path $OUTPUT_ROOT/comparison/curves \\
        --Deterministic-UNet $OUTPUT_ROOT/comparison/reports/deterministic_unet \\
        --MC-Dropout         $OUTPUT_ROOT/comparison/reports/mc_dropout \\
        --Diffusion          $OUTPUT_ROOT/comparison/reports/diffusion
"""
import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from matplotlib.lines import Line2D

from __init__ import root_path, console_handler
# importing the palettes module installs the IBM/Tol axes.prop_cycle the per-family colours are drawn from
from src.utils.plotting import palettes  # noqa: F401

logger = logging.getLogger(__name__)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

_LINESTYLES = ['-', '--', ':', '-.']                    # one linestyle per exceedance threshold, within a family


def _display(key: str) -> str:
    """Undo Fire's hyphen substitution, exactly as `tabulate_metrics._display` does — the label is the flag name, so
    the legends here and the row index there read identically for the same family."""
    return key.replace('_', '-')


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root_path, path)


def _model_colors(model_names) -> Dict[str, object]:
    """One colour per family from the installed IBM/Tol prop_cycle, assigned over SORTED labels so the same family
    keeps the same colour across every figure and across re-runs (kwargs order is not stable)."""
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    return {name: (cycle[i % len(cycle)] if cycle else None) for i, name in enumerate(sorted(model_names))}


def _read_csv(report_dir: str, filename: str) -> Optional[pd.DataFrame]:
    """Read one per-figure CSV, returning None rather than raising when it is absent, empty or unreadable — so one
    family's missing or truncated curve never costs the whole comparison."""
    path = os.path.join(report_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:                          # noqa: BLE001 — a broken CSV must not abort the run
        logger.warning(f'Could not read "{path}": {error}.')
        return None


def _numeric(table: pd.DataFrame, *columns) -> Optional[tuple]:
    """Coerce columns to float and return them masked to the rows finite in ALL of them, or None if none survive.

    Two reasons this is not just `.to_numpy(dtype=float)`. A hand-edited or truncated CSV can leave object dtype that
    would make `np.isfinite` raise and abort the stage; and `reliability_table.csv` legitimately carries NaN rows for
    the probability bins no cell fell into, which must be dropped rather than plotted as gaps.
    """
    if table is None or table.empty or not set(columns).issubset(table.columns):
        return None
    arrays = [pd.to_numeric(table[name], errors='coerce').to_numpy(dtype=float) for name in columns]
    finite = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if not finite.any():
        return None
    return tuple(array[finite] for array in arrays)


def _save(figure, output_dir: str, name: str) -> None:
    """Persist a comparison figure as BOTH png (raster preview) and pdf (vector, for publication). The per-family
    reports save line figures as png only; these are the ones that go in a paper."""
    figure.savefig(os.path.join(output_dir, f'{name}.png'), dpi=150, bbox_inches='tight')
    figure.savefig(os.path.join(output_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close(figure)


def _combined_psd(report_dirs, colors, output_dir) -> None:
    """Radially-averaged power spectra, all families on one loglog axis with the observed reference drawn once.

    Reads `wavelength_km`, not `wavelength_px`, and inverts the x-axis — the same convention as the per-family
    figure, so a reader moving between the two is not silently comparing pixels with kilometres.
    """
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    drawn, observation_drawn = False, False
    for name in sorted(report_dirs):
        table = _read_csv(report_dirs[name], 'psd_curves.csv')
        pair = _numeric(table, 'wavelength_km', 'model')
        if pair is None:
            continue
        wavelengths, model = pair
        axis.loglog(wavelengths, model, color=colors[name], linewidth=2, label=name)
        band = _numeric(table, 'wavelength_km', 'model', 'model_std')    # stochastic families only
        if band is not None:
            band_wavelengths, band_model, half_width = band
            lower = np.clip(band_model - half_width, a_min=np.finfo(float).tiny, a_max=None)   # keep positive (log y)
            axis.fill_between(band_wavelengths, lower, band_model + half_width, color=colors[name], alpha=0.18,
                              linewidth=0)
        if not observation_drawn:
            observed = _numeric(table, 'wavelength_km', 'obs')
            if observed is not None:
                axis.loglog(*observed, color='black', linestyle='--', linewidth=1.5, label='observed')
                observation_drawn = True
        drawn = True
    if not drawn:
        plt.close(figure)
        logger.warning('combined_psd: no family had a usable psd_curves.csv; skipped.')
        return
    axis.set_xlabel('Wavelength [km]')
    axis.set_ylabel('Radially-averaged power')
    axis.set_title('Power spectral density — all families (high frequencies to the right)')
    axis.invert_xaxis()
    axis.legend()
    axis.grid(True, which='both', alpha=0.3)
    _save(figure, output_dir, 'combined_psd')


def _combined_fss(report_dirs, colors, output_dir) -> None:
    """FSS against neighbourhood scale, with TWO legends: colour is the family, linestyle the exceedance threshold.

    A single legend cannot carry a families x thresholds grid, and collapsing to one threshold would hide the thing
    the figure is for — a family can win at the occurrence threshold and lose at h6.
    """
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    threshold_style: Dict[str, str] = {}
    drawn_models = []
    for name in sorted(report_dirs):
        table = _read_csv(report_dirs[name], 'fss_table.csv')
        if table is None or table.empty or not {'threshold', 'scale', 'fss'}.issubset(table.columns):
            continue
        contributed = False
        for threshold, block in table.groupby('threshold'):
            threshold = str(threshold)
            if threshold not in threshold_style:
                threshold_style[threshold] = _LINESTYLES[len(threshold_style) % len(_LINESTYLES)]
            pair = _numeric(block.sort_values('scale'), 'scale', 'fss')
            if pair is None:
                continue
            axis.plot(*pair, color=colors[name], linestyle=threshold_style[threshold], marker='o', markersize=3)
            contributed = True
        if contributed:
            drawn_models.append(name)
    if not drawn_models:
        plt.close(figure)
        logger.warning('combined_fss: no family had a usable fss_table.csv; skipped.')
        return
    axis.axhline(0.5, color='grey', linestyle='--', linewidth=1)
    axis.set_xlabel('neighbourhood scale [pixels]')
    axis.set_ylabel('FSS')
    axis.set_ylim(0, 1)
    axis.set_title('Fractions skill score vs scale — all families')
    model_handles = [Line2D([0], [0], color=colors[name], lw=2, label=name) for name in drawn_models]
    style_handles = [Line2D([0], [0], color='black', linestyle=style, label=threshold)
                     for threshold, style in threshold_style.items()]
    first = axis.legend(handles=model_handles, title='family', loc='upper left', fontsize=8)
    axis.add_artist(first)                              # two legends on one axis: the first must be re-added by hand
    axis.legend(handles=style_handles, title='threshold', loc='lower right', fontsize=8)
    axis.grid(alpha=0.3)
    _save(figure, output_dir, 'combined_fss')


def _combined_reliability(report_dirs, colors, output_dir) -> None:
    """Occurrence reliability for every family, with the forecast-count histogram beside it.

    Both panels, for the reason the per-family figure gives: a reliability curve carried by one populated bin looks
    identical to a genuinely calibrated one, and only the bin populations tell them apart. Absent for every family in
    daily mode — there is no occurrence head, so no probability to be calibrated about.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    drawn = False
    for name in sorted(report_dirs):
        table = _read_csv(report_dirs[name], 'reliability_table.csv')
        pair = _numeric(table, 'mean_probability', 'observed_frequency')
        if pair is None:
            continue
        axes[0].plot(*pair, marker='o', color=colors[name], linewidth=1.6, label=name)
        counts = _numeric(table, 'mean_probability', 'counts')
        if counts is not None:
            axes[1].step(*counts, where='mid', color=colors[name], linewidth=1.6, label=name)
        drawn = True
    if not drawn:
        plt.close(figure)
        logger.info('combined_reliability: no family had a usable reliability_table.csv; skipped (expected in daily '
                    'mode, where no family has an occurrence head).')
        return
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='perfect')
    axes[0].set_xlabel('forecast probability')
    axes[0].set_ylabel('observed frequency')
    axes[0].set_title('Occurrence reliability — all families')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('forecast probability')
    axes[1].set_ylabel('cell count (log)')
    axes[1].set_title('Forecast sharpness (bin populations)')
    axes[1].grid(alpha=0.3, axis='y')
    figure.tight_layout()
    _save(figure, output_dir, 'combined_reliability')


def _headline_threshold(table: pd.DataFrame) -> Optional[str]:
    """The event to compare the families on: `occurrence` when the report has it, otherwise the FIRST threshold the
    table lists.

    One event, not all of them. Four thresholds x three families is twelve lines per panel, and a PR panel on a log
    axis is unreadable at that density — the per-family reports carry every threshold. `occurrence` is the headline
    because `average_precision_occurrence` is the discrimination term of the selection score (config/eval/metrics_daily.yaml).
    The fallback is first-listed rather than alphabetical because the rows keep the config's declaration order, which
    is what makes it meaningful for the hourly task, whose thresholds are probability cuts with no `occurrence` among
    them.
    """
    if 'threshold' not in table.columns:
        return None
    names = table['threshold'].astype(str)
    if (names == 'occurrence').any():
        return 'occurrence'
    return names.iloc[0] if len(names) else None


def _combined_roc_pr(report_dirs, colors, output_dir) -> None:
    """ROC and precision-recall for every family on the headline event, drawn from the curve POINTS.

    Both panels for the reason the per-family figure documents: at a ~0.43 % base rate the ROC curve is flattered by
    the correct-negative mass while the PR curve exposes the real trade-off, and a family strong on the left and weak
    on the right is exploiting the imbalance. AUC and AP annotate the legend from `roc_pr_summary.csv`; that file also
    carries the base rate, which sets the PR panel's no-skill floor (drawn once — the event is the same for all).
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    drawn, base_rate_drawn = False, False
    for name in sorted(report_dirs):
        table = _read_csv(report_dirs[name], 'roc_pr_curves.csv')
        if table is None or table.empty:
            continue
        threshold = _headline_threshold(table)
        if threshold is None:
            continue
        block = table[table['threshold'].astype(str) == threshold]
        summary = _read_csv(report_dirs[name], 'roc_pr_summary.csv')
        scalars = {}
        if summary is not None and 'threshold' in summary.columns:
            match = summary[summary['threshold'].astype(str) == threshold]
            scalars = match.iloc[0].to_dict() if len(match) else {}

        roc = _numeric(block, 'fpr', 'tpr')
        if roc is not None:
            axes[0].plot(*roc, color=colors[name], linewidth=1.8,
                         label=f'{name}  (AUC {scalars.get("roc_auc", float("nan")):.3f})')
            drawn = True
        precision_recall = _numeric(block, 'recall', 'precision')
        if precision_recall is not None:
            axes[1].plot(*precision_recall, color=colors[name], linewidth=1.8,
                         label=f'{name}  (AP {scalars.get("average_precision", float("nan")):.3f})')
            drawn = True
        base_rate = float(scalars.get('base_rate', float('nan')))
        if not base_rate_drawn and np.isfinite(base_rate) and base_rate > 0:
            axes[1].axhline(base_rate, color='black', linestyle=':', linewidth=1,
                            label=f'no skill ({base_rate:.2e})')
            base_rate_drawn = True
    if not drawn:
        plt.close(figure)
        logger.info('combined_roc_pr: no family had a usable roc_pr_curves.csv; skipped.')
        return
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='no skill')
    axes[0].set_xlabel('false positive rate')
    axes[0].set_ylabel('true positive rate')
    axes[0].set_title('ROC — optimistic when negatives dominate')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('recall')
    axes[1].set_ylabel('precision')
    axes[1].set_yscale('log')
    axes[1].set_title('Precision-recall — the honest view at this base rate')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which='both')
    figure.tight_layout()
    _save(figure, output_dir, 'combined_roc_pr')


def _combined_rank_histogram(report_dirs, colors, output_dir) -> None:
    """Talagrand rank histograms as step lines, one per ensemble family, against the uniform reference.

    Step lines rather than the bars the per-family figure uses: overlapping bars from three families hide each other,
    and the shape (flat / U / domed / sloped) is what is being compared.
    """
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    drawn, n_bins = False, 0
    for name in sorted(report_dirs):
        table = _read_csv(report_dirs[name], 'rank_histogram.csv')
        pair = _numeric(table, 'rank', 'frequency')
        if pair is None:
            continue
        rank, frequency = pair
        axis.step(rank, frequency, where='mid', color=colors[name], linewidth=1.6, label=name)
        n_bins = max(n_bins, int(rank.size))
        drawn = True
    if not drawn:
        plt.close(figure)
        logger.info('combined_rank_histogram: no family had a rank_histogram.csv; skipped (expected for a '
                    'deterministic-only comparison).')
        return
    if n_bins:
        # the reference is 1/(M+1) for the LARGEST ensemble present; families with fewer members have their own,
        # which is why the label names the bin count rather than claiming one uniform line fits every curve
        axis.axhline(1.0 / n_bins, color='black', linestyle='--', linewidth=1, label=f'uniform (1/{n_bins})')
    axis.set_xlabel('observation rank among the ensemble members')
    axis.set_ylabel('frequency')
    axis.set_title('Rank (Talagrand) histogram — ensemble families')
    axis.legend()
    axis.grid(alpha=0.3, axis='y')
    _save(figure, output_dir, 'combined_rank_histogram')


def combine_curves(output_path: str, **kwargs) -> None:
    """Overlay the families' curve CSVs into shared comparison figures (png + pdf) under `output_path`.

    Args:
        output_path: Output DIRECTORY (relative to `root_path` unless absolute); created if missing.
        **kwargs: family label -> that family's REPORT directory, holding the CSVs `evaluate` wrote. Fire maps
            `--MC-Dropout=dir` to `MC_Dropout`; :func:`_display` restores the label. A family whose directory is
            missing is warned about and skipped.

    Returns:
        None. Writes `combined_psd` / `combined_fss` / `combined_reliability` / `combined_roc_pr` /
        `combined_rank_histogram`, each as png AND pdf. A figure no family could contribute to is skipped.
    """
    if not kwargs:
        raise ValueError('combine_curves needs at least one `--<Family-Label> <report-dir>` argument.')

    report_dirs = {}
    for key, path in kwargs.items():
        label = _display(key)
        absolute = _resolve(str(path))
        if not os.path.isdir(absolute):
            logger.warning(f'Report directory for "{label}" not found at "{path}"; skipping this family.')
            continue
        report_dirs[label] = absolute

    if not report_dirs:
        raise FileNotFoundError('combine_curves found none of the provided report directories.')

    output_dir = _resolve(output_path)
    os.makedirs(output_dir, exist_ok=True)
    colors = _model_colors(report_dirs.keys())

    _combined_psd(report_dirs, colors, output_dir)
    _combined_fss(report_dirs, colors, output_dir)
    _combined_reliability(report_dirs, colors, output_dir)
    _combined_roc_pr(report_dirs, colors, output_dir)
    _combined_rank_histogram(report_dirs, colors, output_dir)
    logger.info(f'Combined comparison figures for {len(report_dirs)} families written to "{output_path}".')


if __name__ == '__main__':
    Fire(combine_curves)
