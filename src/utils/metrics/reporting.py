"""Report generation for the evaluation stage: figures (png) and tables (csv) from the metric-suite curves.

The figure list is driven by the ``reporting`` section of the active metrics config (``metrics_daily.yaml`` or
``metrics_hourly.yaml``); every figure is also exported as a
CSV table when ``csv`` is among the requested formats, so results can be compared numerically across families and
across team members' experiments.

**How one config serves every family without branching.** ``write_report`` holds a dict mapping each figure name to
a lambda, and each line-or-table figure fetches its entry from ``curves`` and returns immediately when it is
missing. A deterministic run never populates ``curves['rank_histogram']`` and a non-residual run never populates
``curves['residual']``, so those figures skip themselves and a metrics config can list all fourteen unconditionally.

Two things to know before adding a figure:

* ``maps_most_extreme_days`` does NOT consult ``curves`` — it takes ``prediction`` / ``observation`` / ``items`` /
  ``ensemble_members`` directly and picks its layout from whether ``ensemble_members`` is None. The self-skip
  mechanism covers the line and table figures only.
* every figure call is wrapped in ``try/except Exception`` that only logs a warning, on the reasoning that a broken
  figure must not lose a whole evaluation run. A genuine bug is therefore swallowed exactly like a deliberately
  absent curve: the run still reports success and the only trace is a line in the log.

The map styling follows ``.claude/plans/inventory-figures.md`` §1 (the 02a grammar), implemented in
``src/utils/plotting/maps.py``: cartopy ``EuroPP`` axes over a ``PlateCarree`` data transform, ``origin='upper'``,
unit bins in lightning-hours, and ONE warm palette shared by every panel of a figure under a single colorbar. (The
warm/cool over/under diff encoding was dropped in Step 4 block 4e — see that module's docstring.)
"""
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, Normalize

# importing the plotting package applies the Inter font and the IBM/Tol colour cycle globally (line figures pick
# the palette up automatically through the rcParams prop_cycle)
from src.utils.plotting.palettes import (
    get_color, ibm_diverging_palette_factory, ibm_linear_palette_factory,
)
from src.utils.plotting.maps import (
    DISPLAY_EXTENT, GRID_EXTENT, KM_PER_PIXEL, add_lightning_colorbar, add_map_axis, draw_map,
    geographic_context, make_lightning_cmap,
)

# diverging map for the residual diagnostics (signed bias / log-surprise): orange (negative) -> white (0) ->
# purple (positive). The two saturated "special" colours flag the +/-inf surprise cells (a correction where the
# truth needs none = overcorrected; no correction where the truth needs one = failed).
RESID_DIVERGING_CMAP = ibm_diverging_palette_factory('orange', 'purple')
SURPRISE_OVER_COLOR = get_color('hot_pink')          # +inf: overcorrected
SURPRISE_UNDER_COLOR = get_color('aqua')             # -inf: failed to correct

# 02a's PSD figure colours
PSD_OBS_COLOR = 'steelblue'
PSD_MODEL_COLOR = 'darkorange'

logger = logging.getLogger(__name__)


def _save_figure(figure, report_path: str, name: str, formats) -> None:
    if 'png' in formats:
        figure.savefig(os.path.join(report_path, f'{name}.png'), dpi=150, bbox_inches='tight')
    plt.close(figure)


def _save_map_figure(figure, report_path: str, name: str) -> None:
    """Persist a map figure as BOTH png (raster preview) and pdf (vector, for publication), regardless of the
    ``formats`` requested for the line/table figures.

    🐛 **No ``bbox_inches='tight'`` here, unlike the line figures.** A cartopy ``GeoAxes`` returns a NON-FINITE tight
    bbox, and matplotlib's tight-bbox machinery silently DISCARDS non-finite bboxes rather than failing — so the saved
    box was the union of the only finite artists, the two colorbars and the suptitle. Every map figure came out cropped
    to roughly its right half: the "Observed" panel was absent from the file entirely and the title sat at the left
    edge, which is how it was spotted. An 11 x 5.5 in figure saved 895 x 771 px instead of 1650 x 825.

    Nothing is lost by dropping it: these figures are laid out EXPLICITLY (``add_gridspec(left=…, right=…)`` plus
    hand-placed colorbar axes), so there is no stray whitespace for a tight box to trim. The line figures keep it —
    plain Axes report finite bboxes and are not laid out by hand.
    """
    figure.savefig(os.path.join(report_path, f'{name}.png'), dpi=150)
    figure.savefig(os.path.join(report_path, f'{name}.pdf'))
    plt.close(figure)


# =====================================================================================================================
# Per-day maps — the 02a grammar (see src/utils/plotting/maps.py)
# =====================================================================================================================
def _category_selection(totals: np.ndarray, errors: np.ndarray, n_samples: int) -> Dict[str, list]:
    """Top-``n_samples`` item indices for each category: highest activity, closest to the median activity, and
    largest total error."""
    by_activity = np.argsort(totals)                                        # ascending
    midpoint = len(by_activity) // 2
    lo = max(0, midpoint - n_samples // 2)
    return {
        'most_active': list(by_activity[::-1][:n_samples]),
        'median_activity': list(by_activity[lo:lo + n_samples]),
        'worst_error': list(np.argsort(errors)[::-1][:n_samples]),
    }


def _select_plot_indices(observation, prediction, items, plot_dates) -> List[Tuple[int, str]]:
    """Item indices to render, in order, deduplicated: any requested ``plot_dates`` first, then the auto-selected
    most-active / median-activity / worst-error days. Each is tagged with why it was chosen (the tag goes into the
    file name, not the title). Extremeness is defined by the total observed activity over the domain."""
    totals = observation.sum(axis=(-2, -1))
    errors = np.abs(prediction - observation).sum(axis=(-2, -1))        # vs the point estimate (ensemble mean)
    n_samples = min(3, len(totals))

    selected: List[Tuple[int, str]] = []
    seen = set()

    def add(index, tag):
        index = int(index)
        if index not in seen:
            seen.add(index)
            selected.append((index, tag))

    if plot_dates:
        item_dates = pd.to_datetime(items['date']).dt.date.astype(str).to_numpy()
        for requested in plot_dates:
            key = str(pd.Timestamp(requested).date())
            matches = np.where(item_dates == key)[0]
            if matches.size == 0:
                logger.warning(f'Requested plot date "{requested}" is not in the evaluated split; skipped.')
            for index in matches:
                add(index, 'requested')

    for tag, indices in _category_selection(totals, errors, n_samples).items():
        for index in indices:
            add(index, tag)
    return selected


def _deterministic_day_figure(observation, prediction, title, projection, data_crs):
    """1 x 2: observations | predictions, both on the one shared palette, with a single colorbar on the right."""
    cmap, norm = make_lightning_cmap(np.nanmax(observation))
    figure = plt.figure(figsize=(11, 5.5))
    grid = figure.add_gridspec(1, 2, left=0.05, right=0.80, top=0.88, bottom=0.10, wspace=0.05)
    figure.suptitle(title, fontsize=14, fontweight='bold')
    for column, (field, panel_title) in enumerate(
            ((observation, 'observations'), (prediction, 'predictions'))):
        draw_map(add_map_axis(figure, grid[0, column], projection), field, panel_title, data_crs,
                 cmap, norm, left_labels=(column == 0))
    add_lightning_colorbar(figure, cmap, norm, 0.10, 0.78)
    return figure


def _stochastic_day_figure(observation, mean, std, members, title, projection, data_crs, rng):
    """2 x 3 — row 0: observations | ensemble mean | ensemble std; row 1: up to three randomly chosen members.

    Every panel but the std shares the one lightning palette and its single colorbar, so the observation, the mean and
    the members are directly comparable by eye. The STD is a spread, not a count of hours, so it keeps its own
    continuous ``viridis`` scale and a detached colorbar spanning row 0 only.

    The members are labelled ``member 1..3`` in the order drawn, not by their index in the ensemble: MC-dropout and
    diffusion members are exchangeable, so a member's position in the stack carries no meaning worth putting in a title.
    """
    cmap, norm = make_lightning_cmap(np.nanmax(observation))
    n_members = int(members.shape[0])
    k = min(3, n_members)
    chosen_index = rng.choice(n_members, size=k, replace=False)
    chosen = [members[index] for index in chosen_index]

    n_cols, n_rows = 3, 2
    grid_top, grid_bottom, hspace = 0.90, 0.08, 0.17
    cell_height = (grid_top - grid_bottom) / (n_rows + hspace)
    row0_bottom = grid_top - cell_height

    figure = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    grid = figure.add_gridspec(n_rows, n_cols, left=0.05, right=0.78, top=grid_top, bottom=grid_bottom,
                               hspace=hspace, wspace=0.05)
    figure.suptitle(title, fontsize=14, fontweight='bold')

    draw_map(add_map_axis(figure, grid[0, 0], projection), observation, 'observations', data_crs,
             cmap, norm, left_labels=True)
    draw_map(add_map_axis(figure, grid[0, 1], projection), mean, 'ensemble mean', data_crs, cmap, norm)
    std_image = draw_map(add_map_axis(figure, grid[0, 2], projection), std, 'ensemble std',
                         data_crs, 'viridis', norm=None, vmin=0)
    for column in range(n_cols):
        ax = add_map_axis(figure, grid[1, column], projection)
        if column < k:
            draw_map(ax, chosen[column], f'member {column + 1}', data_crs, cmap, norm,
                     left_labels=(column == 0))
        else:                                                            # fewer than 3 members: blank the slot
            ax.set_axis_off()

    # the std colorbar is detached, right of the std panel, spanning row 0 only
    figure.colorbar(std_image, cax=figure.add_axes([0.80, row0_bottom, 0.016, cell_height]), label='h / day')
    add_lightning_colorbar(figure, cmap, norm, grid_bottom, grid_top - grid_bottom)
    return figure


def maps_most_extreme_days(prediction, observation, items, report_path, ensemble_members=None, plot_dates=None,
                           model_family=None):
    """One map figure PER DAY (``maps_<date>.png`` + ``.pdf``) for the most extreme and median observed days, plus
    any requested ``plot_dates``.

    DETERMINISTIC models (``ensemble_members`` None) get the 1 x 2 Observed | Predicted layout. STOCHASTIC models (an
    MC-dropout or diffusion ensemble run, ``ensemble_members`` a ``[N, M, H, W]`` stack) get the 2 x 3 observed /
    ensemble-mean / ensemble-std / three-members grid.

    The colour scale is observation-driven and PER DATE (``ceil(nanmax(obs))``), so every panel of one figure shares
    a scale while different days may not — the accepted trade-off of the 02a grammar, which keeps each day's own
    dynamic range legible.

    Each figure's title is ``<date>[ hHH] event, <family> model``; the selection category (most active / median /
    worst error) is encoded in the FILE NAME instead, so the title says what was plotted rather than why it was
    picked. ``model_family`` is the family the CHECKPOINT resolved to, so a figure cannot be mislabelled by a stale
    CLI argument; it is omitted from the title when unknown rather than guessed at.
    """
    projection, data_crs = geographic_context()
    stochastic = ensemble_members is not None
    rng = np.random.default_rng(0)                                       # reproducible member sampling
    used_names = set()                                                   # guard against a same-(date,hour) overwrite

    selected = _select_plot_indices(observation, prediction, items, plot_dates)
    tag_totals = {}                                                      # how many figures each category contributes
    for _, tag in selected:
        tag_totals[tag] = tag_totals.get(tag, 0) + 1
    tag_seen = {}

    for index, tag in selected:
        date = pd.Timestamp(items.iloc[index]['date']).date()
        hour = items.iloc[index].get('hour')
        suffix = f'_h{int(hour):02d}' if pd.notna(hour) else ''
        stamp = f'{date}' + (f' h{int(hour):02d}' if pd.notna(hour) else '')
        title = f'{stamp} event' + (f', {model_family} model' if model_family else '')
        # FILE NAME: maps_<date>[_hHH]_<category>[_<n>]. The per-category ordinal is added only when that category
        # contributes more than one day; a leftover collision falls back to the item index rather than overwriting.
        ordinal = tag_seen.get(tag, 0)
        tag_seen[tag] = ordinal + 1
        tag_suffix = f'_{tag}' + (f'_{ordinal}' if tag_totals[tag] > 1 else '')
        name = f'maps_{date}{suffix}{tag_suffix}'
        if name in used_names:
            name = f'{name}_{index}'
        used_names.add(name)

        if stochastic:
            members = np.asarray(ensemble_members[index])                # [M, H, W]
            figure = _stochastic_day_figure(
                observation[index], prediction[index], members.std(axis=0), members, title,
                projection, data_crs, rng
            )
        else:
            figure = _deterministic_day_figure(
                observation[index], prediction[index], title, projection, data_crs
            )
        _save_map_figure(figure, report_path, name)


# =====================================================================================================================
# Curve and table figures
# =====================================================================================================================
def _psd_curves(curves, report_path, formats):
    """Radially-averaged PSD of model / observations / baselines, with the wavelength axis in KILOMETRES.

    Styled per the 02a PSD figure: loglog, x inverted so large scales sit on the left, observations in steelblue and
    the model in darkorange. Where the notebook overlays one date's individual members, the report has the
    split-aggregate spectrum plus the +/-1 sigma ensemble band from ``curves['psd']['model_std']`` — the same
    information at split scale.
    """
    psd = curves.get('psd')
    if not psd:
        return
    wavelengths_px = np.asarray(psd['wavelengths'])
    wavelengths_km = wavelengths_px * KM_PER_PIXEL
    model_std = psd.get('model_std')                # ensemble spread of the model PSD (stochastic families only)

    figure, axis = plt.subplots(figsize=(8, 5))
    fixed_colors = {'obs': PSD_OBS_COLOR, 'model': PSD_MODEL_COLOR}
    for name, power in psd.items():
        if name in ('wavelengths', 'model_std'):    # model_std is the band half-width, not a curve of its own
            continue
        axis.loglog(wavelengths_km, power, label=name, color=fixed_colors.get(name),
                    linewidth=2 if name in ('model', 'obs') else 1)
    if model_std is not None:
        model_power = np.asarray(psd['model'])
        half_width = np.asarray(model_std)
        lower = np.clip(model_power - half_width, a_min=np.finfo(float).tiny, a_max=None)   # keep positive (log y)
        axis.fill_between(wavelengths_km, lower, model_power + half_width, color=PSD_MODEL_COLOR, alpha=0.2,
                          linewidth=0, label='model ±1σ (ensemble)')
    axis.set_xlabel(f'Wavelength [km]   ({KM_PER_PIXEL:g} km / pixel)')
    axis.set_ylabel('Radially-averaged power')
    axis.set_title('Power spectral density (high frequencies to the right)')
    axis.invert_xaxis()
    axis.legend()
    axis.grid(True, which='both', alpha=0.3)
    _save_figure(figure, report_path, 'psd_curves', formats)

    if 'csv' in formats:
        table = pd.DataFrame({name: power for name, power in psd.items() if name != 'wavelengths'})
        table.insert(0, 'wavelength_px', wavelengths_px)
        table.insert(1, 'wavelength_km', wavelengths_km)
        table.to_csv(os.path.join(report_path, 'psd_curves.csv'), index=False)


def _fss_vs_scale(curves, report_path, formats):
    fss = curves.get('fss', {})
    if not fss:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    for threshold_name, by_scale in fss.items():
        scales = sorted(by_scale)
        axis.plot(scales, [by_scale[scale] for scale in scales], marker='o', label=threshold_name)
    axis.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='FSS = 0.5')
    axis.set_xlabel('neighborhood scale [pixels]')
    axis.set_ylabel('FSS')
    axis.set_ylim(0, 1)
    axis.set_title('Fractions skill score vs neighborhood scale')
    axis.legend()
    axis.grid(alpha=0.3)
    _save_figure(figure, report_path, 'fss_vs_scale', formats)

    if 'csv' in formats:
        rows = [
            {'threshold': threshold_name, 'scale': scale, 'fss': value}
            for threshold_name, by_scale in fss.items()
            for scale, value in sorted(by_scale.items())
        ]
        pd.DataFrame(rows).to_csv(os.path.join(report_path, 'fss_table.csv'), index=False)


def _reliability(curves, report_path, formats):
    """Reliability diagram of the occurrence forecast, plus the forecast-count histogram beside it.

    Promoted to a headline calibration diagnostic under the classification-first scope. Self-skips without a
    probabilistic occurrence forecast: a classifier-less regressor would only produce a degenerate two-point curve.
    The count histogram is what distinguishes a genuinely well-calibrated forecast from one whose reliability curve
    is carried by a single populated bin.
    """
    reliability = curves.get('reliability')
    if reliability is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='perfect')
    axes[0].plot(reliability['mean_probability'], reliability['observed_frequency'], marker='o', label='model')
    axes[0].set_xlabel('forecast probability')
    axes[0].set_ylabel('observed frequency')
    axes[0].set_title('Occurrence reliability')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    counts = np.asarray(reliability['counts'], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, counts.size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axes[1].bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, alpha=0.7)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('forecast probability')
    axes[1].set_ylabel('cell count (log)')
    axes[1].set_title('Forecast sharpness (bin populations)')
    axes[1].grid(alpha=0.3, axis='y')

    figure.tight_layout()
    _save_figure(figure, report_path, 'reliability', formats)

    if 'csv' in formats:
        pd.DataFrame(reliability).to_csv(os.path.join(report_path, 'reliability_table.csv'), index=False)


def _roc_pr_curves(curves, report_path, formats):
    """ROC and precision-recall curves side by side, one line per event threshold.

    Drawing BOTH is the point. At a ~0.07 % base rate the ROC curve is flattered by the enormous correct-negative
    mass, while the PR curve exposes the real precision/recall trade-off. A model that looks strong on the left
    panel and weak on the right is exploiting the imbalance, and the two panels together make that visible where
    either alone would not.

    Each PR panel carries its own no-skill line at the event's base rate (a random forecast's precision), which is
    why the two panels' reference lines differ: the ROC diagonal is universal, the PR floor is not.

    Exports TWO tables: `roc_pr_summary.csv` (per-threshold roc_auc / average_precision / base_rate) and
    `roc_pr_curves.csv` (the curve points, long-format). `combine_curves` needs both — the points to overlay the
    families' curves, the summary for the no-skill line and the legend annotations.
    """
    roc_pr = curves.get('roc_pr')
    if not roc_pr:
        return
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    drawn = False
    for threshold_name, block in roc_pr.items():
        false_positive_rate = np.asarray(block.get('fpr', []))
        true_positive_rate = np.asarray(block.get('tpr', []))
        recall = np.asarray(block.get('recall', []))
        precision = np.asarray(block.get('precision', []))
        if false_positive_rate.size:
            axes[0].plot(false_positive_rate, true_positive_rate,
                         label=f'{threshold_name}  (AUC {block.get("roc_auc", float("nan")):.3f})')
            drawn = True
        if recall.size:
            axes[1].plot(recall, precision,
                         label=f'{threshold_name}  (AP {block.get("average_precision", float("nan")):.3f})')
            base_rate = float(block.get('base_rate', float('nan')))
            if np.isfinite(base_rate) and base_rate > 0:
                axes[1].axhline(base_rate, linestyle=':', linewidth=1,
                                label=f'{threshold_name} no-skill ({base_rate:.2e})')
            drawn = True
    if not drawn:
        plt.close(figure)
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
    _save_figure(figure, report_path, 'roc_pr_curves', formats)

    if 'csv' in formats:
        rows = []
        for threshold_name, block in roc_pr.items():
            rows.append({
                'threshold': threshold_name,
                'roc_auc': block.get('roc_auc', float('nan')),
                'average_precision': block.get('average_precision', float('nan')),
                'base_rate': block.get('base_rate', float('nan')),
                'from_probability': block.get('from_probability', False),
            })
        pd.DataFrame(rows).to_csv(os.path.join(report_path, 'roc_pr_summary.csv'), index=False)

        # The curve POINTS, long-format, one row per decision cut. Written alongside the scalar summary because a
        # cross-family ROC/PR overlay cannot be rebuilt from AUC and AP -- combine_curves needs the points, and every
        # other curve figure already exports its own (psd_curves.csv, fss_table.csv, reliability_table.csv). All four
        # arrays share one length by construction (`finalize_ranking_metrics` derives them from one set of bin edges,
        # and `tpr` IS `recall`; both columns are kept so a reader can follow which panel uses which).
        points = [
            pd.DataFrame({'threshold': threshold_name, 'fpr': block['fpr'], 'tpr': block['tpr'],
                          'recall': block['recall'], 'precision': block['precision']})
            for threshold_name, block in roc_pr.items() if np.asarray(block.get('fpr', [])).size
        ]
        if points:
            pd.concat(points, ignore_index=True).to_csv(
                os.path.join(report_path, 'roc_pr_curves.csv'), index=False
            )


def _confusion_matrix(curves, report_path, formats):
    """2 x 2 contingency counts per event threshold — the raw hits / misses / false alarms / correct negatives.

    These are the numbers pod / far / csi / ets are ratios OF, and on this target the ratios alone hide the scale:
    a CSI of 0.3 means something very different on 50 observed events than on 50 000. The colour is log-scaled
    because the correct-negative cell is three to four orders of magnitude larger than the rest, which would
    otherwise flatten the whole matrix to one shade.
    """
    confusion = curves.get('confusion')
    if not confusion:
        return
    names = list(confusion)
    figure, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.8), squeeze=False)
    cmap = ibm_linear_palette_factory('purple')

    for column, threshold_name in enumerate(names):
        block = confusion[threshold_name]
        table = np.array([
            [block['hits'], block['misses']],
            [block['false_alarms'], block['correct_negatives']],
        ], dtype=np.float64)
        axis = axes[0][column]
        positive = table[table > 0]
        norm = LogNorm(vmin=max(positive.min(), 1.0), vmax=max(table.max(), 1.0)) if positive.size else None
        axis.imshow(np.where(table > 0, table, np.nan), cmap=cmap, norm=norm)
        for row in range(2):
            for col in range(2):
                axis.text(col, row, f'{table[row, col]:,.0f}', ha='center', va='center', fontsize=9)
        # 🐛 The axes follow the LAYOUT above, and used to be the other way round. `misses` is `~pred & obs`, so
        # row 0 = [hits, misses] is "observed yes" and column 0 = [hits, false_alarms] is "predicted yes": rows are the
        # OBSERVATION, columns the PREDICTION. Labelled the other way, the "obs yes" column showed
        # hits + false_alarms — every cell in the domain for a model that over-forecasts — so the figure read as
        # "lightning was observed at every pixel", which is impossible on a 99.93 %-zero target. The CSV was never
        # affected: it is written from the named keys. Found by reading the block 4e gate's report.
        axis.set_xticks([0, 1], ['pred yes', 'pred no'])
        axis.set_yticks([0, 1], ['obs yes', 'obs no'])
        axis.set_title(threshold_name, fontsize=10)

    figure.suptitle('Contingency counts per event threshold (log colour)', fontsize=12)
    figure.tight_layout()
    _save_figure(figure, report_path, 'confusion_matrix', formats)

    if 'csv' in formats:
        pd.DataFrame([{'threshold': name, **confusion[name]} for name in names]).to_csv(
            os.path.join(report_path, 'confusion_matrix.csv'), index=False
        )


def _error_by_intensity_bin(curves, report_path, formats):
    error_by_bin = curves.get('error_by_bin')
    if not error_by_bin:
        return
    table = pd.DataFrame(error_by_bin)
    figure, axis = plt.subplots(figsize=(8, 5))
    table.plot.bar(ax=axis)
    axis.set_ylabel('MAE')
    axis.set_title('MAE by observed-intensity bin (model vs baselines)')
    axis.grid(alpha=0.3, axis='y')
    plt.setp(axis.get_xticklabels(), rotation=30, ha='right')
    _save_figure(figure, report_path, 'error_by_intensity_bin', formats)

    if 'csv' in formats:
        table.to_csv(os.path.join(report_path, 'error_by_intensity_bin.csv'))


def _rank_histogram(curves, report_path, formats):
    """Talagrand rank histogram of the ensemble (occurrence cells): bar of the observation-rank frequencies with
    the uniform reference. Flat = calibrated; U-shaped = under-dispersed; domed = over-dispersed; sloped = biased.
    Only drawn for an ensemble run (the curve is absent otherwise)."""
    rank = curves.get('rank_histogram')
    if not rank:
        return
    counts = np.asarray(rank['counts'], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return
    n_members = int(rank.get('n_members', counts.size - 1))
    frequencies = counts / total

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(np.arange(counts.size), frequencies, width=0.9, alpha=0.7, label='observed')
    axis.axhline(1.0 / counts.size, color='black', linestyle='--', linewidth=1,
                 label=f'uniform (1/{counts.size})')
    axis.set_xlabel(f'observation rank among the {n_members} ensemble members')
    axis.set_ylabel('frequency')
    axis.set_title('Rank (Talagrand) histogram at occurrence cells')
    axis.legend()
    axis.grid(alpha=0.3, axis='y')
    _save_figure(figure, report_path, 'rank_histogram', formats)

    if 'csv' in formats:
        pd.DataFrame({
            'rank': np.arange(counts.size, dtype=int),
            'count': counts.astype(np.int64),
            'frequency': frequencies
        }).to_csv(os.path.join(report_path, 'rank_histogram.csv'), index=False)


# =====================================================================================================================
# Residual-mode diffusion diagnostics figures — driven by curves['residual'] (see src/utils/metrics/diagnostics.py).
# Every figure returns early when the residual block (a residual diffusion ensemble run) is absent. Saved png+pdf.
# =====================================================================================================================
def _diverging_norm(values: np.ndarray, robust_percentile: float = 99.0) -> Normalize:
    """Symmetric Normalize centred at 0 for a signed field; the extent is the robust max ``|value|`` (a few
    outliers do not wash out the map)."""
    finite = np.asarray(values)[np.isfinite(values)]
    extent = float(np.percentile(np.abs(finite), robust_percentile)) if finite.size else 1.0
    return Normalize(vmin=-max(extent, 1e-6), vmax=max(extent, 1e-6))


def _solid_cmap(color):
    cmap = ListedColormap([color]).copy()
    cmap.set_bad(alpha=0.0)
    return cmap


def _residual_imshow(ax, data, cmap, norm, data_crs):
    """imshow a signed ``[H, W]`` residual field on the same geographic footing as the lightning maps."""
    return ax.imshow(data, cmap=cmap, norm=norm, origin='upper', transform=data_crs, extent=GRID_EXTENT)


def _residual_map_panel(figure, spec, field, norm, projection, data_crs, title, specials=()):
    """One diverging residual map panel (white background). ``specials`` overlays solid-colour boolean masks (the
    +/-inf surprise categories) on top of the diverging finite field."""
    ax = add_map_axis(figure, spec, projection)
    ax.set_facecolor('white')
    _residual_imshow(ax, field, RESID_DIVERGING_CMAP, norm, data_crs)
    for mask, color in specials:
        if mask.any():
            overlay = np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float))
            _residual_imshow(ax, overlay, _solid_cmap(color), Normalize(0.0, 1.0), data_crs)
    ax.set_extent(DISPLAY_EXTENT, crs=data_crs)
    ax.coastlines(linewidth=0.8)
    ax.set_title(title, fontsize=9)
    return ax


def _diverging_colorbar(figure, cax, norm, label):
    mappable = ScalarMappable(norm=norm, cmap=RESID_DIVERGING_CMAP)
    mappable.set_array([])
    figure.colorbar(mappable, cax=cax, orientation='horizontal')
    cax.set_xlabel(label, fontsize=7)
    cax.tick_params(labelsize=6)


def _residual_bias_map(curves, report_path, formats):
    residual = curves.get('residual')
    if not residual or 'bias_map' not in residual:
        return
    projection, data_crs = geographic_context()
    bias = np.asarray(residual['bias_map'])
    norm = _diverging_norm(bias)
    figure = plt.figure(figsize=(5.6, 4.8))
    grid = figure.add_gridspec(2, 1, height_ratios=[12, 1], hspace=0.3)
    _residual_map_panel(figure, grid[0], bias, norm, projection, data_crs, 'mean predicted discrepancy  D_pred')
    cax = figure.add_subplot(grid[1].subgridspec(1, 3, width_ratios=[1, 3, 1])[0, 1])
    _diverging_colorbar(figure, cax, norm, 'mean residual  (− under-prediction corrected / + over)')
    figure.suptitle('Discrepancy bias', fontsize=11)
    _save_map_figure(figure, report_path, 'residual_bias_map')


def _residual_surprise(curves, report_path, formats):
    residual = curves.get('residual')
    if not residual or 'surprise_magnitude' not in residual:
        return
    from matplotlib.patches import Patch
    projection, data_crs = geographic_context()
    figure = plt.figure(figsize=(10.5, 5.4))
    grid = figure.add_gridspec(2, 2, height_ratios=[12, 1], hspace=0.34, wspace=0.06)
    panes = [('surprise_magnitude', 'magnitude:  log( mean|D_pred| / mean|D_true| )', 'log-ratio  (0 = matched)'),
             ('surprise_direction', 'direction:  mean sign(D_pred) / mean sign(D_true)', 'ratio  (1 = matched)')]
    for col, (key, title, cbar_label) in enumerate(panes):
        pane = residual[key]
        value, category = np.asarray(pane['value']), np.asarray(pane['category'])
        finite_value = np.ma.masked_where(category != 0, value)        # only finite cells get the diverging colour
        norm = _diverging_norm(value[category == 0]) if (category == 0).any() else Normalize(-1.0, 1.0)
        specials = [(category == 1, SURPRISE_OVER_COLOR), (category == -1, SURPRISE_UNDER_COLOR)]
        _residual_map_panel(figure, grid[0, col], finite_value, norm, projection, data_crs, title, specials)
        cax = figure.add_subplot(grid[1, col].subgridspec(1, 3, width_ratios=[1, 3, 1])[0, 1])
        _diverging_colorbar(figure, cax, norm, cbar_label)
    figure.legend(
        handles=[Patch(color=SURPRISE_OVER_COLOR, label='+∞ overcorrected  (D_pred ≠ 0, D_true = 0)'),
                 Patch(color=SURPRISE_UNDER_COLOR, label='−∞ failed to correct  (D_pred = 0, D_true ≠ 0)')],
        loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle('Discrepancy surprise   (white = matched / nothing to correct)', fontsize=11)
    _save_map_figure(figure, report_path, 'residual_surprise')


def _residual_histograms(curves, report_path, formats):
    residual = curves.get('residual')
    if not residual or 'hist_pixel' not in residual:
        return
    pixel, image = residual['hist_pixel'], residual.get('hist_image', {})
    pred_color, true_color = get_color('purple'), get_color('orange')
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    if pixel:
        centers = 0.5 * (pixel['edges'][:-1] + pixel['edges'][1:])
        widths = np.diff(pixel['edges'])
        axes[0].bar(centers, pixel['pred_density'], width=widths * 0.95, alpha=0.45, color=pred_color,
                    label='D_pred (members)')
        axes[0].bar(centers, pixel['true_density'], width=widths * 0.95, alpha=0.45, color=true_color,
                    label='D_true')
        if 'kde_grid' in pixel:
            axes[0].plot(pixel['kde_grid'], pixel['pred_kde'], color=pred_color, linewidth=2)
            axes[0].plot(pixel['kde_grid'], pixel['true_kde'], color=true_color, linewidth=2)
        axes[0].set_title('pixel-wise residual')
        axes[0].set_xlabel('residual value')
        axes[0].set_ylabel('density')
        axes[0].legend()
    for ax, key, title in ((axes[1], 'mae', 'per-image MAE  mean|·|'), (axes[2], 'rms', 'per-image RMS')):
        if image:
            pred, true = np.asarray(image[f'pred_{key}']), np.asarray(image[f'true_{key}'])
            bins = np.linspace(0.0, max(float(pred.max()), float(true.max()), 1e-6), 20)
            ax.hist(pred, bins=bins, alpha=0.5, color=pred_color, label='D_pred')
            ax.hist(true, bins=bins, alpha=0.5, color=true_color, label='D_true')
            ax.set_title(title)
            ax.set_xlabel('per-image magnitude')
            ax.set_ylabel('image count')
            ax.legend()
    figure.suptitle('Predicted vs true discrepancy — distributions', fontsize=12)
    figure.tight_layout()
    _save_map_figure(figure, report_path, 'residual_histograms')
    if 'csv' in formats and image:
        pd.DataFrame(image).to_csv(os.path.join(report_path, 'residual_hist_image.csv'), index=False)


def _residual_qq(curves, report_path, formats):
    residual = curves.get('residual')
    if not residual or 'qq' not in residual:
        return
    qq = residual['qq']
    pred, true = np.asarray(qq['pred']), np.asarray(qq['true'])
    figure, axis = plt.subplots(figsize=(5.6, 5.6))
    lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    axis.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='perfect')
    axis.plot(true, pred, marker='.', linewidth=1, color=get_color('purple'), label='residual QQ')
    axis.set_xlabel('true discrepancy quantile')
    axis.set_ylabel('predicted discrepancy quantile')
    axis.set_title('Residual QQ  (D_pred vs D_true)')
    axis.legend()
    axis.grid(alpha=0.3)
    _save_map_figure(figure, report_path, 'residual_qq')
    if 'csv' in formats:
        pd.DataFrame(qq).to_csv(os.path.join(report_path, 'residual_qq.csv'), index=False)


def _residual_scatters(curves, report_path, formats):
    residual = curves.get('residual')
    if not residual or 'scatter' not in residual:
        return
    scatter = residual['scatter']
    rows = [('obs', 'true target O'), ('discrepancy', 'true discrepancy D_true'), ('upstream', 'upstream U')]
    figure, axes = plt.subplots(len(rows), 2, figsize=(10, 4.0 * len(rows)), squeeze=False)
    hexmap = ibm_linear_palette_factory('purple')
    for row, (name, label) in enumerate(rows):
        blocks = scatter[name]
        pixel = blocks['pixel']
        axes[row][0].hexbin(pixel['x'], pixel['y'], gridsize=40, mincnt=1, cmap=hexmap, bins='log')
        axes[row][0].set_xlabel(f'{label}  (pixel)')
        axes[row][0].set_ylabel('D_pred  (pixel)')
        axes[row][0].set_title(f'D_pred vs {label} — pixel')
        axes[row][0].grid(alpha=0.2)
        image = blocks['image']
        axes[row][1].scatter(image['x'], image['y'], s=14, alpha=0.6, color=get_color('purple'))
        axes[row][1].set_xlabel(f'{label}  (per-image)')
        axes[row][1].set_ylabel('|D_pred|  (per-image)')
        axes[row][1].set_title('per-image')
        axes[row][1].grid(alpha=0.2)
    figure.suptitle('Predicted discrepancy vs target / discrepancy / upstream  (correlations in metrics JSON)',
                    fontsize=12)
    figure.tight_layout()
    _save_map_figure(figure, report_path, 'residual_scatters')


def _residual_heteroscedasticity(curves, report_path, formats):
    residual = curves.get('residual')
    if not residual or 'heteroscedasticity' not in residual:
        return
    het = residual['heteroscedasticity']
    figure, axis = plt.subplots(figsize=(7, 5))
    drawn = False
    for name, color in (('upstream', get_color('purple')), ('obs', get_color('orange'))):
        block = het.get(name, {})
        centers = np.asarray(block.get('bin_center', []))
        if centers.size:
            axis.plot(centers, np.asarray(block['rms_error']), marker='o', color=color, label=f'vs {name}')
            drawn = True
    if not drawn:
        plt.close(figure)
        return
    axis.set_xlabel('conditioning value  (decile median)')
    axis.set_ylabel('RMS( D_pred − D_true )')
    axis.set_title('Correction-error heteroscedasticity')
    axis.legend()
    axis.grid(alpha=0.3)
    _save_map_figure(figure, report_path, 'residual_heteroscedasticity')


def write_report(
        report_path: str,
        reporting_config: dict,
        metrics_flat: Dict[str, float],
        curves: dict,
        prediction: np.ndarray,
        observation: np.ndarray,
        items: pd.DataFrame,
        ensemble_members: Optional[np.ndarray] = None,
        plot_dates: Optional[Sequence[str]] = None,
        model_family: Optional[str] = None
) -> None:
    """Write all requested figures and tables to the report directory.

    Args:
        report_path: Output directory (created if missing).
        reporting_config: The ``reporting`` section of the metrics config (``figures`` and ``formats`` lists).
        metrics_flat: Flat scalar metrics (always exported as metrics.csv when csv is requested).
        curves: Curves payload returned by ``run_metric_suite`` (plus ``residual`` from ``residual_diagnostics``
            and ``rank_histogram`` from ``finalize_ensemble_metrics`` when those ran).
        prediction: Model predictions in the target space, ``[N, H, W]`` — the ensemble MEAN for a stochastic
            family, the single prediction otherwise.
        observation: Observed targets, ``[N, H, W]``.
        items: One row per item, in prediction order (columns ``date``, ``hour``).
        ensemble_members: Full ensemble stack ``[N, M, H, W]`` for a stochastic family — switches the per-day maps
            to the 2 x 3 observed / ensemble-mean / ensemble-std / members layout. ``None`` (a deterministic model)
            keeps the 1 x 2 observed | predicted layout.
        plot_dates: Optional list of ``YYYY-MM-DD`` dates within the split to render in addition to the
            auto-selected extreme/median days.

    The map colour controls A carried (``colorbar_scale``, ``colorbar_integer_bins``, ``quantize``, ``max_val``,
    ``occurrence_event``) are GONE: under the 02a grammar the scale is always unit bins in lightning-hours, driven
    by ``ceil(nanmax(obs))`` per date, and the sub-1 white/grey split replaces the occurrence mask. There is nothing
    left to configure per call.
    """
    os.makedirs(report_path, exist_ok=True)
    figures = reporting_config.get('figures', [])
    formats = reporting_config.get('formats', ['png', 'csv'])

    if 'csv' in formats:
        pd.Series(metrics_flat, name='value').rename_axis('metric').to_csv(
            os.path.join(report_path, 'metrics.csv')
        )

    builders = {
        'maps_most_extreme_days': lambda: maps_most_extreme_days(
            prediction, observation, items, report_path,
            ensemble_members=ensemble_members, plot_dates=plot_dates, model_family=model_family
        ),
        'roc_pr_curves': lambda: _roc_pr_curves(curves, report_path, formats),
        'confusion_matrix': lambda: _confusion_matrix(curves, report_path, formats),
        'reliability': lambda: _reliability(curves, report_path, formats),
        'psd_curves': lambda: _psd_curves(curves, report_path, formats),
        'fss_vs_scale': lambda: _fss_vs_scale(curves, report_path, formats),
        'error_by_intensity_bin': lambda: _error_by_intensity_bin(curves, report_path, formats),
        'rank_histogram': lambda: _rank_histogram(curves, report_path, formats),
        # residual-mode diffusion diagnostics (skipped automatically when curves['residual'] is absent)
        'residual_bias_map': lambda: _residual_bias_map(curves, report_path, formats),
        'residual_surprise': lambda: _residual_surprise(curves, report_path, formats),
        'residual_histograms': lambda: _residual_histograms(curves, report_path, formats),
        'residual_qq': lambda: _residual_qq(curves, report_path, formats),
        'residual_scatters': lambda: _residual_scatters(curves, report_path, formats),
        'residual_heteroscedasticity': lambda: _residual_heteroscedasticity(curves, report_path, formats),
    }
    for figure_name in figures:
        builder = builders.get(figure_name)
        if builder is None:
            logger.warning(f'Unknown report figure "{figure_name}" (skipped).')
            continue
        try:
            builder()
        except Exception as error:                                          # a broken figure must not lose the run
            logger.warning(f'Failed to build report figure "{figure_name}": {error}')
    logger.info(f'Report written to "{report_path}".')
