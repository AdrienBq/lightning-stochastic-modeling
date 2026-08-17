"""Tests for src/stages/combine_curves.py — the cross-family curve overlays.

Ported from the combine half of branch A's ``tests/test_tabulate_and_combine.py``, and this is where the port is not
mechanical. A wrote four figures, one of them ``combined_qq`` from ``qq_table.csv`` — a file this repo never produces
(the target-space QQ went with the 02a grammar in Step 2). Its four-pair test is replaced here by the five-figure
equivalent plus ``test_NO_qq_figure_is_produced``, which pins the removal.

⚠️ **The fixture does not hand-write the CSVs; ``reporting`` writes them.** A hand-written fixture is the one thing that
cannot catch the most likely bug in this stage — a column name drifting apart from the module that produces it. A's
fixture wrote ``x,y`` for every curve, which matches no real schema, so every one of its figures would have self-skipped
and every test would have passed while drawing nothing. Building the report directories by calling the real
``reporting._psd_curves`` / ``_fss_vs_scale`` / ``_reliability`` / ``_roc_pr_curves`` / ``_rank_histogram`` means a
renamed column fails here.
"""
import logging
import os

import numpy as np
import pytest

import combine_curves as combine_stage
from src.utils.metrics import reporting
from src.utils.plotting.maps import KM_PER_PIXEL

# CSV -> the figure it feeds. `qq` is deliberately absent (see the module docstring).
FIGURE_FOR_CSV = {
    'psd_curves.csv': 'combined_psd',
    'fss_table.csv': 'combined_fss',
    'reliability_table.csv': 'combined_reliability',
    'roc_pr_curves.csv': 'combined_roc_pr',
    'rank_histogram.csv': 'combined_rank_histogram',
}
WAVELENGTHS_PX = np.array([2.0, 4.0, 8.0, 16.0])
MEMBERS = 8


def _curves(ensemble: bool):
    """One family's metric-suite curve blocks. ``ensemble`` adds the two things only a stochastic family produces: the
    PSD spread band and the rank histogram."""
    curves = {
        'psd': {'wavelengths': WAVELENGTHS_PX,
                'obs': np.array([1e3, 5e2, 2e2, 1e2]),
                'model': np.array([9e2, 4e2, 1.8e2, 0.9e2])},
        'fss': {'occurrence': {1: 0.42, 3: 0.61, 5: 0.68}, 'h6': {1: 0.20, 3: 0.35, 5: 0.44}},
        # the third bin is NaN: a probability bin no cell fell into, which `reliability_curve` emits and which the
        # overlay must drop rather than draw as a gap
        'reliability': {'mean_probability': [0.02, 0.30, float('nan'), 0.85],
                        'observed_frequency': [0.01, 0.28, float('nan'), 0.90],
                        'counts': [90000, 800, 0, 60]},
        # `h6` deliberately FIRST: the occurrence preference is only exercised when it is not also the first-listed
        # threshold, and the fixture's two blocks carry different AUCs so picking the wrong one is detectable
        'roc_pr': {
            'h6': {'fpr': [0.0, 0.20, 0.60, 1.0], 'tpr': [0.0, 0.40, 0.70, 1.0],
                   'recall': [0.0, 0.40, 0.70, 1.0], 'precision': [1.0, 0.15, 0.04, 0.005],
                   'roc_auc': 0.74, 'average_precision': 0.09, 'base_rate': 0.002,
                   'from_probability': True},
            'occurrence': {'fpr': [0.0, 0.10, 0.45, 1.0], 'tpr': [0.0, 0.55, 0.85, 1.0],
                           'recall': [0.0, 0.55, 0.85, 1.0], 'precision': [1.0, 0.30, 0.08, 0.01],
                           'roc_auc': 0.88, 'average_precision': 0.21, 'base_rate': 0.01,
                           'from_probability': True},
        },
    }
    if ensemble:
        curves['psd']['model_std'] = np.array([1e2, 5e1, 2e1, 1e1])
        curves['rank_histogram'] = {'counts': [12, 9, 11, 10, 13, 8, 12, 10, 15], 'n_members': MEMBERS}
    return curves


def _write_report_csvs(directory, curves, include=tuple(FIGURE_FOR_CSV)):
    """Produce the per-figure CSVs through ``reporting`` itself, for only the requested files."""
    writers = {
        'psd_curves.csv': reporting._psd_curves,
        'fss_table.csv': reporting._fss_vs_scale,
        'reliability_table.csv': reporting._reliability,
        'roc_pr_curves.csv': reporting._roc_pr_curves,
        'rank_histogram.csv': reporting._rank_histogram,
    }
    for name in include:
        writers[name](curves, str(directory), ['csv'])


@pytest.fixture
def report_dirs(tmp_path):
    """Two families' report directories: a deterministic one and an ensemble one, exactly as ``evaluate`` leaves them."""
    def build(include=tuple(FIGURE_FOR_CSV)):
        dirs = {}
        for name, ensemble in (('Deterministic_UNet', False), ('MC_Dropout', True)):
            directory = tmp_path / name
            directory.mkdir(exist_ok=True)
            _write_report_csvs(directory, _curves(ensemble), include)
            dirs[name] = str(directory)
        return dirs
    return build


@pytest.fixture
def captured_axes(monkeypatch):
    """``{figure name: [axes]}`` for every figure the stage saves — the figures are inspected as FIGURES, not as files.

    Whether a png exists says nothing about what was drawn on it; A's suite checked only the files, which is how a
    fixture with the wrong column names passed while producing empty axes.
    """
    captured = {}
    original = combine_stage._save

    def spy(figure, output_dir, name):
        captured[name] = list(figure.axes)
        return original(figure, output_dir, name)

    monkeypatch.setattr(combine_stage, '_save', spy)
    return captured


# =====================================================================================================================
# The figure set
# =====================================================================================================================
def test_ONE_figure_is_written_per_curve_type_in_BOTH_formats(report_dirs, tmp_path):
    """png for the preview, pdf for publication — these are the comparison figures that go in a paper, which is why they
    get the vector format the per-family line figures do not."""
    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **report_dirs())

    produced = set(os.listdir(out))
    for figure in FIGURE_FOR_CSV.values():
        assert f'{figure}.png' in produced, f'{figure}.png missing from {sorted(produced)}'
        assert f'{figure}.pdf' in produced, f'{figure}.pdf missing from {sorted(produced)}'


def test_every_csv_the_stage_READS_is_one_REPORTING_writes(report_dirs):
    """The coupling, asserted directly. The stage reads by file name from a directory another module fills, so nothing
    but a test connects the two — a rename on either side is otherwise a figure that silently stops being drawn."""
    directory = report_dirs()['MC_Dropout']
    produced = set(os.listdir(directory))
    missing = [name for name in FIGURE_FOR_CSV if name not in produced]
    assert not missing, f'reporting did not write {missing}; the stage would skip those figures'

    summary = os.path.join(directory, 'roc_pr_summary.csv')
    assert os.path.exists(summary), 'the ROC/PR legend annotations and the no-skill line come from the summary CSV'


def test_NO_qq_figure_is_produced(report_dirs, tmp_path):
    """The removal, pinned. Left in, ``_combined_qq`` would log "no model had a usable qq_table.csv; skipped" on every
    run forever — a permanent warning about a file nothing is meant to write."""
    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **report_dirs())
    assert not any('qq' in name for name in os.listdir(out))


# =====================================================================================================================
# What is actually drawn
# =====================================================================================================================
def test_the_PSD_axis_is_in_KILOMETRES_and_INVERTED(report_dirs, tmp_path, captured_axes):
    """Both halves matter and both are silent when wrong. ``psd_curves.csv`` carries ``wavelength_px`` AND
    ``wavelength_km``: reading the wrong column is a 27.75x error that still produces a plausible-looking loglog figure,
    and forgetting the inversion puts the high frequencies on the wrong side of a figure whose title says otherwise."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    axis = captured_axes['combined_psd'][0]
    assert 'km' in axis.get_xlabel()
    lower, upper = axis.get_xlim()
    assert lower > upper, 'the wavelength axis must be inverted (large scales on the left)'

    expected = WAVELENGTHS_PX * KM_PER_PIXEL
    model_lines = [line for line in axis.lines if line.get_color() != 'black']
    assert model_lines, 'no family curve was drawn'
    for line in model_lines:
        assert np.allclose(line.get_xdata(), expected), 'the PSD x-data is not the kilometre column'


def test_the_OBSERVED_reference_is_drawn_exactly_ONCE(report_dirs, tmp_path, captured_axes):
    """Every family's CSV carries the same ``obs`` column, so drawing it per family would stack identical black dashes
    and put one legend entry per family for a single curve."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    axis = captured_axes['combined_psd'][0]
    observed = [line for line in axis.lines if line.get_label() == 'observed']
    assert len(observed) == 1


def test_the_ENSEMBLE_spread_band_is_drawn_only_for_the_family_that_has_one(report_dirs, tmp_path, captured_axes):
    """``model_std`` is present only in a stochastic family's PSD CSV. A band for the deterministic family would claim a
    spread it does not have."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())
    assert len(captured_axes['combined_psd'][0].collections) == 1


def test_the_FSS_figure_carries_TWO_legends_one_per_dimension(report_dirs, tmp_path, captured_axes):
    """Colour is the family and linestyle the threshold, so a single legend cannot describe the figure — and collapsing
    to one threshold would hide the thing it exists for: a family can win at ``occurrence`` and lose at ``h6``."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    axis = captured_axes['combined_fss'][0]
    legend_titles = {legend.get_title().get_text() for legend in
                     [artist for artist in axis.get_children() if hasattr(artist, 'get_title')
                      and artist.__class__.__name__ == 'Legend']}
    assert {'family', 'threshold'} <= legend_titles, f'found legends {legend_titles}'

    styles = {line.get_linestyle() for line in axis.lines if line.get_marker() == 'o'}
    assert len(styles) == 2, f'two thresholds must draw two linestyles, found {styles}'


def test_the_RELIABILITY_figure_has_both_the_curve_and_the_BIN_POPULATIONS(report_dirs, tmp_path, captured_axes):
    """A reliability curve carried by one populated bin looks identical to a genuinely calibrated one; only the
    populations tell them apart, which is why the count panel is not optional."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    reliability_axis, counts_axis = captured_axes['combined_reliability']
    assert any(list(line.get_xdata()) == [0, 1] and list(line.get_ydata()) == [0, 1]
               for line in reliability_axis.lines), 'the perfect-calibration diagonal is missing'
    assert counts_axis.get_yscale() == 'log'
    assert len(counts_axis.lines) == 2, 'one population curve per family'


def test_the_EMPTY_reliability_bins_are_DROPPED(report_dirs, tmp_path, captured_axes):
    """``reliability_curve`` writes NaN for a probability bin no cell fell into. Plotted, matplotlib breaks the line
    there and the figure claims a gap in the calibration rather than a bin nobody forecast."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    curves = [line for line in captured_axes['combined_reliability'][0].lines if line.get_marker() == 'o']
    assert curves, 'no reliability curve was drawn'
    for line in curves:
        assert len(line.get_xdata()) == 3, 'the NaN bin is still in the curve'
        assert np.isfinite(line.get_xdata()).all()


def test_the_ROC_PR_panels_compare_ONE_threshold_across_the_families(report_dirs, tmp_path, captured_axes):
    """Four thresholds x three families is twelve lines per panel, and a PR panel on a log axis is unreadable at that
    density. The per-family reports keep every threshold; the comparison keeps the headline event."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    roc_axis, pr_axis = captured_axes['combined_roc_pr']
    families = [line for line in roc_axis.lines if line.get_label() != 'no skill']
    assert len(families) == 2, f'expected one ROC curve per family, found {len(families)}'
    assert all('AUC' in line.get_label() for line in families), 'the legend must carry the summary AUC'
    assert any('AP' in line.get_label() for line in pr_axis.lines)


def test_the_headline_event_is_OCCURRENCE_when_the_report_has_it(report_dirs, tmp_path, captured_axes):
    """``average_precision_occurrence`` is the discrimination term of the selection score, so the occurrence event is
    what the families are ranked on. Anti-vacuity: the fixture's ``h6`` block has a different AUC, so picking the wrong
    threshold changes the number in the legend."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    labels = [line.get_label() for line in captured_axes['combined_roc_pr'][0].lines]
    assert any('0.880' in label for label in labels), f'the occurrence AUC (0.88) is absent from {labels}'
    assert not any('0.740' in label for label in labels), 'h6 (AUC 0.74) was drawn instead of occurrence'


def test_the_FIRST_listed_threshold_is_used_when_there_is_NO_occurrence(tmp_path, captured_axes):
    """The hourly task's thresholds are probability cuts (``p50``, ...) with no ``occurrence`` among them, so the
    fallback is not hypothetical. First-listed rather than alphabetical because the CSV keeps the config's declaration
    order, which is a statement of priority; alphabetical would silently pick ``p10`` over ``p50``."""
    curves = {'roc_pr': {
        'p50': {'fpr': [0.0, 0.3, 1.0], 'tpr': [0.0, 0.7, 1.0], 'recall': [0.0, 0.7, 1.0],
                'precision': [1.0, 0.2, 0.02], 'roc_auc': 0.81, 'average_precision': 0.15, 'base_rate': 0.02},
        'p10': {'fpr': [0.0, 0.5, 1.0], 'tpr': [0.0, 0.6, 1.0], 'recall': [0.0, 0.6, 1.0],
                'precision': [1.0, 0.1, 0.01], 'roc_auc': 0.62, 'average_precision': 0.07, 'base_rate': 0.05},
    }}
    directory = tmp_path / 'Hourly_UNet'
    directory.mkdir()
    _write_report_csvs(directory, curves, include=('roc_pr_curves.csv',))

    combine_stage.combine_curves(str(tmp_path / 'combined'), Hourly_UNet=str(directory))

    labels = [line.get_label() for line in captured_axes['combined_roc_pr'][0].lines]
    assert any('0.810' in label for label in labels), f'p50 (first listed) should have been drawn: {labels}'


def test_the_PR_no_skill_line_is_drawn_ONCE_at_the_base_rate(report_dirs, tmp_path, captured_axes):
    """The documented reason both panels exist: the ROC diagonal is universal, the PR floor is not. The event is the same
    for every family, so one line — one per family would be three identical lines and three legend entries."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    pr_axis = captured_axes['combined_roc_pr'][1]
    no_skill = [line for line in pr_axis.lines if line.get_linestyle() == ':']
    assert len(no_skill) == 1
    assert abs(no_skill[0].get_ydata()[0] - 0.01) < 1e-9
    assert pr_axis.get_yscale() == 'log'


def test_the_rank_histogram_UNIFORM_reference_matches_the_bin_count(report_dirs, tmp_path, captured_axes):
    """M+1 bins for M members, so the reference is 1/9 for the fixture's 8-member ensemble. A reference at the wrong
    height turns a calibrated histogram into an apparently biased one."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    axis = captured_axes['combined_rank_histogram'][0]
    reference = [line for line in axis.lines if line.get_linestyle() == '--']
    assert reference and abs(reference[0].get_ydata()[0] - 1.0 / (MEMBERS + 1)) < 1e-9


def test_each_family_keeps_ONE_colour_across_EVERY_figure(report_dirs, tmp_path, captured_axes):
    """A line's colour must mean the same family in all five figures, or reading them side by side is worse than reading
    them separately. The assignment is over sorted labels because kwargs order is not stable."""
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    colors = combine_stage._model_colors(['MC-Dropout', 'Deterministic-UNet'])
    assert colors == combine_stage._model_colors(['Deterministic-UNet', 'MC-Dropout'])
    assert len(set(colors.values())) == 2, 'two families must get two colours'

    deterministic = colors['Deterministic-UNet']
    for figure in ('combined_psd', 'combined_fss', 'combined_reliability', 'combined_roc_pr'):
        drawn = {line.get_color() for axis in captured_axes[figure] for line in axis.lines}
        assert deterministic in drawn, f'{figure} did not draw the deterministic family in its colour'


# =====================================================================================================================
# The self-skips — the asymmetries between families are expected, not errors
# =====================================================================================================================
def test_a_family_missing_a_csv_is_dropped_from_THAT_FIGURE_ONLY(report_dirs, tmp_path, captured_axes):
    """The deterministic family has no rank histogram, which must not cost it its PSD curve."""
    dirs = report_dirs()
    os.remove(os.path.join(dirs['Deterministic_UNet'], 'psd_curves.csv'))

    combine_stage.combine_curves(str(tmp_path / 'combined'), **dirs)

    psd_families = [line for line in captured_axes['combined_psd'][0].lines if line.get_label() != 'observed']
    assert len(psd_families) == 1, 'the family whose CSV was removed is still in the PSD figure'
    assert len(captured_axes['combined_fss'][0].lines) > 1, 'the other figures must be unaffected'


def test_a_figure_NO_family_can_contribute_to_is_SKIPPED(tmp_path, caplog):
    """Blank axes published as a figure are worse than a missing file: they look like a result. The deterministic-only
    comparison is the routine case — no family has a rank histogram."""
    dirs = {}
    for name in ('Deterministic_UNet', 'Second_UNet'):
        directory = tmp_path / name
        directory.mkdir()
        _write_report_csvs(directory, _curves(ensemble=False), include=('psd_curves.csv',))
        dirs[name] = str(directory)

    out = str(tmp_path / 'combined')
    with caplog.at_level(logging.INFO, logger=combine_stage.logger.name):
        combine_stage.combine_curves(out, **dirs)

    produced = os.listdir(out)
    assert any(name.startswith('combined_psd') for name in produced)
    assert not any('rank_histogram' in name for name in produced)
    assert any('rank_histogram' in record.message for record in caplog.records), \
        'a skipped figure must say so; a silently absent file is indistinguishable from a bug'


def test_an_EMPTY_or_MALFORMED_csv_does_not_abort_the_comparison(report_dirs, tmp_path):
    """A truncated CSV from an interrupted run must not lose the whole cross-family figure set."""
    dirs = report_dirs()
    open(os.path.join(dirs['Deterministic_UNet'], 'psd_curves.csv'), 'w').close()          # empty
    with open(os.path.join(dirs['MC_Dropout'], 'fss_table.csv'), 'w') as handle:
        handle.write('not,a,valid\ncurve\n')                                               # ragged

    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **dirs)                                              # must not raise

    produced = os.listdir(out)
    assert any(name.startswith('combined_psd') for name in produced), 'the surviving family still has a PSD curve'
    assert any(name.startswith('combined_reliability') for name in produced)


def test_NON_NUMERIC_curve_values_are_coerced_rather_than_raising(report_dirs, tmp_path):
    """A hand-edited CSV leaves object dtype, on which ``np.isfinite`` raises — and one edited file would then abort a
    comparison that had nothing to do with it."""
    dirs = report_dirs()
    path = os.path.join(dirs['MC_Dropout'], 'rank_histogram.csv')
    with open(path, 'w') as handle:
        handle.write('rank,count,frequency\n0,10,0.11\n1,12,n/a\n2,9,0.10\n')

    combine_stage.combine_curves(str(tmp_path / 'combined'), **dirs)
    assert os.path.exists(os.path.join(str(tmp_path / 'combined'), 'combined_rank_histogram.png'))


# =====================================================================================================================
# The helpers, directly — the pure ones, where the edge cases live
# =====================================================================================================================
def test_read_csv_returns_NONE_for_absent_empty_and_UNPARSEABLE_files(tmp_path):
    """The never-raise contract, in one place: a figure treats None as "this family has no such curve", which is also
    the right handling for an interrupted run's unreadable file."""
    assert combine_stage._read_csv(str(tmp_path), 'not_written.csv') is None

    open(os.path.join(str(tmp_path), 'empty.csv'), 'w').close()
    assert combine_stage._read_csv(str(tmp_path), 'empty.csv') is None, 'pandas raises EmptyDataError here'

    with open(os.path.join(str(tmp_path), 'truncated.csv'), 'w') as handle:
        handle.write('threshold,fpr\n"occurrence,0.0\n')       # a quote the write never closed -> ParserError
    assert combine_stage._read_csv(str(tmp_path), 'truncated.csv') is None


def test_a_RAGGED_csv_is_parsed_and_skipped_by_the_FIGURE_not_by_the_read(tmp_path):
    """Where the division of labour actually falls, which is not where it looks. `pd.read_csv` does NOT reject extra
    fields — it absorbs `a,b` + four values per row without complaint — so `_read_csv` cannot be the thing that
    protects against a wrong-shaped table. The column checks and `_numeric` are, and a test asserting None here would
    have been asserting something false about pandas rather than about this stage."""
    with open(os.path.join(str(tmp_path), 'psd_curves.csv'), 'w') as handle:
        handle.write('not,a,valid\ncurve,1,2\n')

    assert combine_stage._read_csv(str(tmp_path), 'psd_curves.csv') is not None
    assert combine_stage._numeric(combine_stage._read_csv(str(tmp_path), 'psd_curves.csv'),
                                  'wavelength_km', 'model') is None


def test_numeric_returns_NONE_rather_than_an_empty_array(tmp_path):
    """None and an empty array are not interchangeable: a caller that checks `is None` would go on to plot the empty
    one, adding a legend entry for a family that contributed nothing."""
    import pandas as pd

    assert combine_stage._numeric(None, 'x') is None
    assert combine_stage._numeric(pd.DataFrame(), 'x') is None
    assert combine_stage._numeric(pd.DataFrame({'x': [1.0]}), 'x', 'missing') is None
    assert combine_stage._numeric(pd.DataFrame({'x': [float('nan')]}), 'x') is None


def test_numeric_drops_a_row_NON_FINITE_in_ANY_column(tmp_path):
    """The columns are plotted against each other, so a row usable in one and not the other would misalign the pair —
    x[2] against y[3] — and draw a curve nobody can see is wrong."""
    import pandas as pd

    table = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0], 'y': [1.0, float('nan'), 3.0, float('inf')]})
    x, y = combine_stage._numeric(table, 'x', 'y')
    assert list(x) == [1.0, 3.0] and list(y) == [1.0, 3.0]


def test_numeric_coerces_OBJECT_dtype_instead_of_raising(tmp_path):
    """`np.isfinite` raises on object dtype, which a hand-edited CSV leaves behind — and one edited file would abort a
    comparison that had nothing to do with it."""
    import pandas as pd

    table = pd.DataFrame({'x': ['1.0', 'n/a', '3.0']})
    (x,) = combine_stage._numeric(table, 'x')
    assert list(x) == [1.0, 3.0]


def test_headline_threshold_prefers_OCCURRENCE_over_the_first_row(tmp_path):
    import pandas as pd

    table = pd.DataFrame({'threshold': ['h6', 'h6', 'occurrence', 'occurrence']})
    assert combine_stage._headline_threshold(table) == 'occurrence'
    assert combine_stage._headline_threshold(pd.DataFrame({'threshold': ['p50', 'p10']})) == 'p50'
    assert combine_stage._headline_threshold(pd.DataFrame({'fpr': [0.0]})) is None


def test_ONE_unusable_fss_threshold_does_not_cost_the_family_its_OTHER_thresholds(report_dirs, tmp_path,
                                                                                  captured_axes):
    """The skip is per THRESHOLD GROUP, not per family. A partially-written `fss_table.csv` — one threshold's rows
    truncated mid-write — must still contribute the thresholds that are intact."""
    dirs = report_dirs()
    path = os.path.join(dirs['MC_Dropout'], 'fss_table.csv')
    with open(path, 'w') as handle:
        handle.write('threshold,scale,fss\noccurrence,1,0.4\noccurrence,3,0.6\nh6,1,\nh6,3,\n')

    combine_stage.combine_curves(str(tmp_path / 'combined'), **dirs)

    colors = combine_stage._model_colors(['Deterministic-UNet', 'MC-Dropout'])
    drawn = [line for line in captured_axes['combined_fss'][0].lines
             if line.get_marker() == 'o' and line.get_color() == colors['MC-Dropout']]
    assert len(drawn) == 1, 'the intact threshold should still be drawn, and only it'


def test_a_ranking_csv_with_NO_threshold_column_skips_that_family(tmp_path, captured_axes):
    """`_headline_threshold` cannot name an event without the column that identifies it, and a curve drawn under an
    unknown event would be compared against the other families' occurrence curves as if it were the same thing."""
    directory = tmp_path / 'Broken_UNet'
    directory.mkdir()
    with open(os.path.join(str(directory), 'roc_pr_curves.csv'), 'w') as handle:
        handle.write('fpr,tpr,recall,precision\n0.0,0.0,0.0,1.0\n1.0,1.0,1.0,0.01\n')

    combine_stage.combine_curves(str(tmp_path / 'combined'), Broken_UNet=str(directory))
    assert 'combined_roc_pr' not in captured_axes, 'the figure had no contributing family and must self-skip'


def test_EVERY_figure_builder_is_WIRED_into_the_stage(report_dirs, tmp_path, monkeypatch):
    """Two failure modes at once, neither of which any other test sees. A `_combined_*` function added and not called
    is dead code that looks like a feature; and `_combined_qq` coming back would be caught here as well as by the
    file-level test, since the set is compared against what the module DEFINES."""
    builders = ('_combined_psd', '_combined_fss', '_combined_reliability', '_combined_roc_pr',
                '_combined_rank_histogram')

    defined = {name for name in dir(combine_stage) if name.startswith('_combined_')}
    assert defined == set(builders), f'a figure builder is defined but not listed here: {defined ^ set(builders)}'

    called = []
    for name in builders:
        monkeypatch.setattr(combine_stage, name, lambda *a, _name=name, **k: called.append(_name))
    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())
    assert called == list(builders), 'every builder runs, in a fixed order'


# =====================================================================================================================
# Arguments
# =====================================================================================================================
def test_no_family_argument_at_all_RAISES(tmp_path):
    with pytest.raises(ValueError, match='at least one'):
        combine_stage.combine_curves(str(tmp_path / 'combined'))


def test_EVERY_report_directory_missing_RAISES(tmp_path):
    """One absent directory is a family still training; all of them absent means the comparison was run before the
    evaluations, and an empty output directory would look like a comparison that found nothing to say."""
    with pytest.raises(FileNotFoundError):
        combine_stage.combine_curves(str(tmp_path / 'combined'), MC_Dropout=str(tmp_path / 'absent'))


def test_the_LEGEND_labels_match_the_labels_tabulate_metrics_uses(report_dirs, tmp_path, captured_axes):
    """One family, one name, in the table and in the figures. Two spellings of the same family across the two halves of
    one comparison is the kind of thing nobody notices until a figure and a table disagree."""
    import tabulate_metrics as tabulate_stage

    combine_stage.combine_curves(str(tmp_path / 'combined'), **report_dirs())

    labels = {line.get_label() for line in captured_axes['combined_psd'][0].lines}
    assert tabulate_stage._display('MC_Dropout') in labels
    assert combine_stage._display('MC_Dropout') == tabulate_stage._display('MC_Dropout')
