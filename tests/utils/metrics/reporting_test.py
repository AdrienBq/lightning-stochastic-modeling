"""Tests for src/utils/metrics/reporting.py — the figures and tables the evaluation stage writes.

Ported from branch ``aru-probabilistic-eval``'s ``tests/test_residual_diagnostics.py`` (the six residual figures and
their self-skip) and ``tests/test_probabilistic_eval_compat.py`` (the per-day maps, both layouts, the ``plot_dates``
argument, the title/filename split).

**Four of A's tests are deliberately NOT ported**, because they test configuration that no longer exists:
``test_resolve_map_norm_quantize_and_cap``, ``test_report_deterministic_quantize``, ``test_report_quantize_max_val``
and the three-way colorbar-mode loop. Under the 02a grammar the colour scale is always unit bins in lightning-hours
driven by ``ceil(nanmax(obs))`` per date, so ``_resolve_map_norm``, ``quantize``, ``max_val``, ``colorbar_scale`` and
``colorbar_integer_bins`` are all gone and there is nothing left to configure per call. The colour-axis coverage they
provided now lives in ``tests/utils/plotting/maps_test.py`` against ``make_lightning_cmap``.

``write_report`` also no longer takes ``occurrence_event`` — the sub-1 white/grey split replaced the occurrence mask.

⚠️ Every figure call inside ``write_report`` is wrapped in ``try/except Exception`` that only logs a warning, so a
broken figure is swallowed exactly like a deliberately absent curve. That is why these tests assert on the FILES
produced rather than on the call returning: a figure that raises leaves no file, and nothing else would reveal it.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import diagnostics, reporting
from src.utils.plotting import maps

H = W = 16
MEMBERS = 4
RESIDUAL_FIGURES = ['residual_bias_map', 'residual_surprise', 'residual_histograms', 'residual_qq',
                    'residual_scatters', 'residual_heteroscedasticity']


CATEGORY_TAGS = ('most_active', 'median_activity', 'worst_error', 'requested')


def _report_arrays(n_items=4, seed=0):
    """Sparse bounded lightning-hour maps with an active blob per item, plus an ensemble and the item table."""
    rng = np.random.default_rng(seed)
    observation = rng.poisson(0.4, size=(n_items, H, W)).astype(np.float32)
    observation[:, 4:8, 4:8] += rng.poisson(6.0, size=(n_items, 4, 4))
    observation = np.clip(observation, 0, 24)
    prediction = np.clip(observation + rng.normal(0, 1.5, observation.shape), 0, 24).astype(np.float32)
    members = np.clip(
        prediction[:, None, :, :] + rng.normal(0, 1.0, (n_items, MEMBERS, H, W)), 0, 24
    ).astype(np.float32)
    dates = pd.date_range('2015-07-20', periods=n_items, freq='D').strftime('%Y-%m-%d').tolist()
    items = pd.DataFrame({'date': dates, 'hour': [np.nan] * n_items})
    return prediction, observation, members, items, dates


@pytest.fixture
def report_arrays():
    return _report_arrays


# =====================================================================================================================
# Per-day maps: one png + pdf per day, both layouts  (ported)
#
# ⚠️ Rendering a cartopy figure is EXPENSIVE — a 6-panel stochastic day is several seconds, and these tests were
# ~15 s each when every one rendered its own report. The four assertions about the deterministic report are about
# DIFFERENT properties of the SAME output, so the render is module-scoped and shared. Keep it that way: re-rendering
# per test cost 46 s for three redundant copies of one figure set.
# =====================================================================================================================
@pytest.fixture(scope='module')
def deterministic_report(tmp_path_factory):
    """Render the deterministic per-day report ONCE. Returns ``(produced file names, titles handed to the builder)``."""
    directory = tmp_path_factory.mktemp('deterministic')
    prediction, observation, _, items, _ = _report_arrays(n_items=6, seed=3)

    captured = []
    original = reporting._deterministic_day_figure

    def spy(observation_map, prediction_map, title, *args, **kwargs):
        captured.append(title)
        return original(observation_map, prediction_map, title, *args, **kwargs)

    reporting._deterministic_day_figure = spy
    try:
        reporting.maps_most_extreme_days(prediction, observation, items, str(directory))
    finally:
        reporting._deterministic_day_figure = original
    return set(os.listdir(str(directory))), captured


@pytest.fixture(scope='module')
def stochastic_report(tmp_path_factory):
    """Render the 2x3 stochastic per-day report ONCE, with a requested plot date included."""
    directory = tmp_path_factory.mktemp('stochastic')
    prediction, observation, members, items, dates = _report_arrays(n_items=3, seed=0)
    reporting.maps_most_extreme_days(prediction, observation, items, str(directory),
                                     ensemble_members=members, plot_dates=[dates[1]])
    return set(os.listdir(str(directory))), dates


def test_deterministic_layout_writes_a_png_and_a_pdf_per_day(deterministic_report):
    produced, _ = deterministic_report
    pngs = {name for name in produced if name.endswith('.png')}
    assert pngs, f'no per-day maps written: {sorted(produced)}'
    for name in pngs:
        assert name.startswith('maps_')
        assert name.replace('.png', '.pdf') in produced, 'every map must be saved as png AND pdf'


def test_stochastic_layout_writes_a_png_and_a_pdf_per_day(stochastic_report):
    produced, _ = stochastic_report
    pngs = {name for name in produced if name.endswith('.png')}
    assert pngs
    for name in pngs:
        assert name.replace('.png', '.pdf') in produced


def test_a_requested_plot_date_is_always_rendered(stochastic_report):
    """``plot_dates`` is the only way to force a specific day into the report, and the only argument of
    ``maps_most_extreme_days`` a config can still drive."""
    produced, dates = stochastic_report
    assert f'maps_{dates[1]}_requested.png' in produced, sorted(produced)
    assert f'maps_{dates[1]}_requested.pdf' in produced


def test_an_unmatched_plot_date_is_skipped_rather_than_fatal(tmp_path):
    """A date outside the evaluated split must not abort the report — the auto-selected days still render. Only two
    items here, so this is the cheapest possible render."""
    prediction, observation, _, items, _ = _report_arrays(n_items=2)
    reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path), plot_dates=['1999-01-01'])

    produced = os.listdir(str(tmp_path))
    assert any(name.endswith('.png') for name in produced)
    assert not any('1999-01-01' in name for name in produced)


@pytest.fixture(scope='module')
def both_layout_sizes(tmp_path_factory):
    """Render the SAME three days under both layouts once, and return ``{name: bytes}`` for each.

    Module-scoped for the usual reason (see the note above): this is two full cartopy reports, and the two tests below
    read different properties of the same output.
    """
    directory = tmp_path_factory.mktemp('layout_sizes')
    prediction, observation, members, items, _ = _report_arrays(n_items=3, seed=3)

    def render(name, **kwargs):
        target = directory / name
        target.mkdir()
        reporting.maps_most_extreme_days(prediction, observation, items, str(target), **kwargs)
        return {entry: os.path.getsize(os.path.join(str(target), entry))
                for entry in os.listdir(str(target)) if entry.endswith('.png')}

    return render('deterministic'), render('stochastic', ensemble_members=members)


def test_the_rendered_maps_are_not_BLANK_canvases(both_layout_sizes):
    """Every figure call in ``write_report`` is wrapped in a warn-only ``try/except``, so a figure that fails halfway
    can still leave a file behind. A blank 16x16 cartopy canvas is a few kB; a real map with coastlines and a colorbar is
    tens of kB — the file SIZE is the cheapest available proof that something was actually drawn."""
    plain, _ = both_layout_sizes
    assert plain, 'nothing rendered'
    assert min(plain.values()) > 20000, plain


def test_the_stochastic_layout_renders_LARGER_files_for_the_same_days(both_layout_sizes):
    """Six panels against two, on the same selected days — so the stochastic layout must select the same set (selection
    is by observed activity and error, neither of which the ensemble changes) and render every one of them bigger. A
    stochastic run that quietly fell back to the deterministic figure would pass every file-name test."""
    plain, ensemble = both_layout_sizes

    assert set(plain) == set(ensemble), (sorted(plain), sorted(ensemble))
    bigger = [name for name in plain if ensemble[name] > plain[name]]
    assert len(bigger) == len(plain), {name: (plain[name], ensemble[name]) for name in plain}


def test_the_title_is_the_date_only(deterministic_report):
    """A plain date title plus a fixed per-figure colour scale is what makes days comparable; the selection category
    ("why was this day chosen") belongs in the file name, not stamped across the figure."""
    _, titles = deterministic_report
    assert titles, 'the deterministic day figure was never built'
    for title in titles:
        assert not any(tag in title for tag in CATEGORY_TAGS)
        pd.Timestamp(title.split(' ')[0])                       # the leading token parses as a date


def test_each_file_name_carries_exactly_one_category_tag(deterministic_report):
    produced, _ = deterministic_report
    for name in (name for name in produced if name.endswith('.png')):
        tags = [tag for tag in CATEGORY_TAGS if f'_{tag}' in name]
        assert len(tags) == 1, f'{name} must carry exactly one category tag, got {tags}'


def test_a_category_contributing_several_days_gets_an_ordinal(deterministic_report):
    """Without the ordinal, two days chosen by the same category would collide on one file name and the second would
    silently overwrite the first."""
    import re

    produced, _ = deterministic_report
    by_tag = {}
    for name in (name for name in produced if name.endswith('.png')):
        for tag in CATEGORY_TAGS:
            if f'_{tag}' in name:
                by_tag.setdefault(tag, []).append(name)
    multi = {tag: files for tag, files in by_tag.items() if len(files) > 1}
    assert multi, f'expected a multi-day category to exercise the ordinal: {sorted(produced)}'
    for tag, files in multi.items():
        assert all(re.search(rf'_{tag}_\d+\.png$', name) for name in files), files


# =====================================================================================================================
# An HOURLY run's maps are the DAILY TOTALS  (Step 4 block 4f-r)
#
# Summing the hours is a change of UNITS, not a plotting convenience, and that is the whole reason it is right: the 0/1
# hourly observation summed over a date IS the 0-24 lightning-hours field `mode: daily` prepares (both come from one
# `hourly-threshold`), and the predicted PROBABILITIES summed over a date are the EXPECTED number of lightning-hours —
# the quantity a daily model predicts directly. So the two tasks' figures are comparable panel for panel.
#
# What it replaced: a 0/1 observation pins `max_val = ceil(nanmax(obs))` at 1, so the warm palette collapsed to
# white-below-0.5 / grey-above for every hourly figure, under an `h / day` label naming a unit that was not on the axis.
# =====================================================================================================================
def _hourly_arrays(n_days=3, hours=24, seed=0):
    """An hourly stack in the shape the evaluation stage hands over: ``[n_days * hours, H, W]`` of 0/1 observations and
    probabilities, with the item table ordered by (date, hour) and its ``hour`` column POPULATED."""
    rng = np.random.default_rng(seed)
    n_items = n_days * hours
    observation = (rng.random((n_items, H, W)) < 0.04).astype(np.float32)
    observation[:, 4:8, 4:8] = (rng.random((n_items, 4, 4)) < 0.5).astype(np.float32)   # an active blob
    prediction = np.clip(0.5 * observation + 0.05 * rng.random(observation.shape), 0, 1).astype(np.float32)
    members = np.clip(
        prediction[:, None, :, :] + rng.normal(0, 0.02, (n_items, MEMBERS, H, W)), 0, 1
    ).astype(np.float32)
    dates = pd.date_range('2015-07-20', periods=n_days, freq='D').strftime('%Y-%m-%d').tolist()
    items = pd.DataFrame({'date': np.repeat(dates, hours), 'hour': list(range(hours)) * n_days})
    return prediction, observation, members, items, dates


def test_the_hourly_stack_is_summed_into_days():
    """⭐ The core of it, asserted on the arrays rather than on a rendered figure: D rows out of D*24, and each row the
    exact sum of its date's hours. Checked against an independent per-date mask, not against another reduction."""
    prediction, observation, members, items, dates = _hourly_arrays(n_days=3, hours=24)

    daily_prediction, daily_observation, daily_items, daily_members = reporting._sum_hours_into_days(
        prediction, observation, items, members
    )

    assert daily_observation.shape == (3, H, W) and daily_prediction.shape == (3, H, W)
    assert daily_members.shape == (3, MEMBERS, H, W)
    assert list(daily_items['date']) == dates
    for position, date in enumerate(dates):
        mask = (items['date'] == date).to_numpy()
        assert np.allclose(daily_observation[position], observation[mask].sum(axis=0))
        assert np.allclose(daily_prediction[position], prediction[mask].sum(axis=0))
        assert np.allclose(daily_members[position], members[mask].sum(axis=0))


def test_the_summed_observation_IS_a_count_of_lightning_hours():
    """The claim that makes this the daily target rather than an arbitrary aggregate: summing a 0/1 field over 24 hours
    gives whole numbers in 0-24 — exactly what ``_daily_aggregation`` writes in daily mode."""
    _, observation, _, items, _ = _hourly_arrays(n_days=2, hours=24)
    _, daily_observation, _, _ = reporting._sum_hours_into_days(observation, observation, items, None)

    assert np.all(daily_observation == np.round(daily_observation)), 'not integral -> not a count of hours'
    assert daily_observation.min() >= 0 and daily_observation.max() <= 24
    assert daily_observation.max() > 1, 'the whole point: a 0/1 field would pin the colour axis at max_val = 1'


def test_the_summed_prediction_is_the_EXPECTED_lightning_hours():
    """``sum_h P(lightning at hour h)`` — bounded by the same 0-24, and continuous rather than integral, which is what
    a daily regression's prediction also is."""
    prediction, observation, _, items, _ = _hourly_arrays(n_days=2, hours=24)
    daily_prediction, _, _, _ = reporting._sum_hours_into_days(prediction, observation, items, None)

    assert daily_prediction.min() >= 0 and daily_prediction.max() <= 24
    assert not np.all(daily_prediction == np.round(daily_prediction)), 'an expectation need not be integral'


def test_the_summed_items_carry_NO_hour_column():
    """It is the absence of ``hour`` that routes the caller onto its daily naming and title branch, so this is not
    tidiness — it is the mechanism."""
    prediction, observation, _, items, _ = _hourly_arrays(n_days=2, hours=6)
    _, _, daily_items, _ = reporting._sum_hours_into_days(prediction, observation, items, None)
    assert 'hour' not in daily_items.columns


def test_a_DAILY_run_is_not_aggregated(report_arrays):
    """The other direction. A daily item table carries an all-``NaN`` ``hour`` column, which must NOT trip the sum —
    doing so would collapse a multi-day split into one figure per unique date with the values doubled."""
    prediction, observation, _, items, _ = report_arrays(n_items=4)
    assert items['hour'].isna().all()
    assert not ('hour' in items and items['hour'].notna().any()), 'the detector must read False on a daily table'


def test_hours_split_across_NON_CONTIGUOUS_runs_RAISE():
    """``np.add.reduceat`` sums contiguous runs. If a date's items were not adjacent it would silently produce one row
    per RUN instead of one per date — every figure then drawn from a partial day, with nothing to say so. The prepared
    index and an unshuffled loader guarantee adjacency; this pins that the guarantee is checked."""
    prediction, observation, _, items, _ = _hourly_arrays(n_days=2, hours=4)
    shuffled = items.iloc[[0, 4, 1, 5, 2, 6, 3, 7]].reset_index(drop=True)      # interleave the two days
    with pytest.raises(ValueError, match='not grouped by date'):
        reporting._sum_hours_into_days(prediction, observation, shuffled, None)


def test_an_hourly_report_writes_ONE_figure_PER_DAY_named_by_the_DATE_ALONE(tmp_path):
    """End to end through the public function: 2 days x 6 hours in, and exactly ``maps_<date>.png`` out — no ``_hHH``
    segment (the hours are summed) and no ``_most_active_0`` segment (one category, so the tag would be the same word
    on every file). Six hours rather than 24 to keep the cartopy renders down."""
    prediction, observation, _, items, dates = _hourly_arrays(n_days=2, hours=6)
    reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path))

    pngs = sorted(name for name in os.listdir(str(tmp_path)) if name.endswith('.png'))
    assert pngs, 'no per-day maps written'
    assert pngs == [f'maps_{date}.png' for date in dates if f'maps_{date}.png' in pngs], pngs
    assert not [name for name in pngs if any(tag in name for tag in CATEGORY_TAGS)], pngs
    assert len(pngs) <= 3, f'at most the 3 most-active days: {pngs}'


def test_an_hourly_report_plots_the_MOST_ACTIVE_days_ONLY(tmp_path):
    """⭐ The category narrowing, asserted where it is decided rather than by counting files. ``worst_error`` ranks on
    the error in the DAILY TOTAL once the hours are summed, so it selects on a quantity these maps cannot show;
    ``median_activity`` exists for a contrast that is the daily task's product, not this one's."""
    prediction, observation, _, items, _ = _hourly_arrays(n_days=5, hours=4)

    captured = []
    original = reporting._select_plot_indices

    def spy(observation_arg, prediction_arg, items_arg, plot_dates_arg, categories=None):
        captured.append(categories)
        return original(observation_arg, prediction_arg, items_arg, plot_dates_arg, categories=categories)

    reporting._select_plot_indices = spy
    try:
        reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path))
    finally:
        reporting._select_plot_indices = original

    assert captured == [reporting.HOURLY_PLOT_CATEGORIES]
    assert reporting.HOURLY_PLOT_CATEGORIES == ('most_active',)


def test_a_DAILY_report_still_gets_ALL_THREE_categories(report_arrays, tmp_path):
    """The narrowing is HOURLY-ONLY. The daily maps are the primary product and the worst-error day is a real
    diagnostic there — nothing cancels it, because a daily item is already the whole day."""
    prediction, observation, _, items, _ = report_arrays(n_items=5)

    captured = []
    original = reporting._select_plot_indices

    def spy(*args, categories=None, **kwargs):
        captured.append(categories)
        return original(*args, categories=categories, **kwargs)

    reporting._select_plot_indices = spy
    try:
        reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path))
    finally:
        reporting._select_plot_indices = original

    assert captured == [None], 'a daily run must not be narrowed'
    produced = [name for name in os.listdir(str(tmp_path)) if name.endswith('.png')]
    assert {tag for tag in CATEGORY_TAGS if any(f'_{tag}' in name for name in produced)} >= \
        {'most_active', 'median_activity', 'worst_error'}, produced


def test_category_selection_REJECTS_an_unknown_category():
    """A typo would otherwise silently narrow the selection to nothing and write no maps at all — a report that
    quietly lost a figure set, which is the failure mode this repo keeps closing."""
    totals, errors = np.arange(6.0), np.arange(6.0)[::-1].copy()
    with pytest.raises(ValueError, match='Unknown plot categories'):
        reporting._category_selection(totals, errors, n_samples=2, categories=('most_actve',))


def test_category_selection_narrows_to_the_requested_subset():
    totals, errors = np.array([1.0, 5.0, 3.0, 2.0]), np.array([4.0, 1.0, 2.0, 3.0])
    full = reporting._category_selection(totals, errors, n_samples=2)
    narrowed = reporting._category_selection(totals, errors, n_samples=2, categories=('most_active',))
    assert set(narrowed) == {'most_active'}
    assert narrowed['most_active'] == full['most_active'], 'narrowing must not change the ranking'


def test_the_hourly_maps_reach_a_MULTI_BIN_colour_axis(tmp_path, monkeypatch):
    """⭐ The user-visible symptom this change fixes, pinned at the point the palette is built: ``make_lightning_cmap``
    must be called with a ``max_val`` above 1, or the figure is two colours whatever else is right.

    Asserted on the ARGUMENT rather than on the rendered pixels deliberately — block 4e's lesson was the opposite
    (assertions on the figure object passed while the output was wrong), and this is the one place the two coincide:
    ``max_val`` IS the colour axis, and a wrong one cannot be compensated downstream.
    """
    from src.utils.plotting import maps as maps_module

    seen = []
    original = maps_module.make_lightning_cmap

    def spy(max_val, *args, **kwargs):
        seen.append(float(max_val))
        return original(max_val, *args, **kwargs)

    monkeypatch.setattr(reporting, 'make_lightning_cmap', spy)
    prediction, observation, _, items, _ = _hourly_arrays(n_days=1, hours=6)
    reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path))

    assert seen, 'the palette was never built'
    assert min(seen) > 1.0, f'colour axis still collapsed to the 0/1 range: max_val values {seen}'


# =====================================================================================================================
# The curve and table figures
# =====================================================================================================================
def test_psd_curves_writes_the_figure_and_both_wavelength_axes(tmp_path):
    wavelengths = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
    curves = {'psd': {'wavelengths': wavelengths,
                      'obs': np.array([1e3, 5e2, 2e2, 8e1, 3e1]),
                      'model': np.array([9e2, 4e2, 1e2, 5e1, 2e1]),
                      'model_std': np.array([1e2, 5e1, 2e1, 8.0, 3.0])}}
    reporting._psd_curves(curves, str(tmp_path), ['png', 'csv'])

    assert os.path.exists(os.path.join(str(tmp_path), 'psd_curves.png'))
    table = pd.read_csv(os.path.join(str(tmp_path), 'psd_curves.csv'))
    assert {'wavelength_px', 'wavelength_km'} <= set(table.columns)
    assert 'model_std' in table.columns, 'the ensemble band must reach the table too'


@pytest.mark.parametrize('figure,curves', [
    ('fss_vs_scale', {'fss': {'occurrence': {1: 0.4, 3: 0.6, 5: 0.7}}}),
    ('reliability', {'reliability': {'mean_probability': [0.05, 0.3, 0.7],
                                     'observed_frequency': [0.04, 0.35, 0.6],
                                     'counts': [1000, 100, 10]}}),
    ('roc_pr_curves', {'roc_pr': {'occurrence': {'fpr': [0.0, 0.5, 1.0], 'tpr': [0.0, 0.8, 1.0],
                                                 'recall': [1.0, 0.5, 0.0], 'precision': [0.05, 0.3, 1.0],
                                                 'roc_auc': 0.9, 'average_precision': 0.3, 'base_rate': 0.02}}}),
    ('confusion_matrix', {'confusion': {'h6': {'hits': 40, 'misses': 12, 'false_alarms': 8,
                                               'correct_negatives': 100000}}}),
    ('error_by_intensity_bin', {'error_by_bin': {'model': {'h3': 1.0, 'h6': 2.0}, 'zero': {'h3': 3.0, 'h6': 6.0}}}),
    ('rank_histogram', {'rank_histogram': {'counts': [10, 12, 9, 11, 13], 'n_members': 4}}),
])
def test_each_curve_figure_renders(figure, curves, tmp_path):
    builder = getattr(reporting, f'_{figure}')
    builder(curves, str(tmp_path), ['png', 'csv'])
    assert os.path.exists(os.path.join(str(tmp_path), f'{figure}.png')), sorted(os.listdir(str(tmp_path)))


@pytest.mark.parametrize('figure', ['psd_curves', 'fss_vs_scale', 'reliability', 'roc_pr_curves', 'confusion_matrix',
                                    'error_by_intensity_bin', 'rank_histogram'])
def test_each_curve_figure_self_skips_when_its_curve_is_absent(figure, tmp_path):
    """The self-skip is what lets metrics_daily.yaml list all fourteen figures unconditionally: a deterministic run never
    populates ``rank_histogram`` and a non-residual run never populates ``residual``."""
    getattr(reporting, f'_{figure}')({}, str(tmp_path), ['png', 'csv'])
    assert not os.listdir(str(tmp_path))


# =====================================================================================================================
# The six residual figures  (ported)
# =====================================================================================================================
@pytest.fixture(scope='module')
def residual_curves():
    """Module-scoped: ``residual_diagnostics`` over a 10-item ensemble is seconds, and eight tests read the same
    curves without mutating them."""
    rng = np.random.default_rng(0)
    n = 10
    upstream = np.clip(np.abs(rng.normal(6, 3, (n, H, W))), 0, 24)
    true_residual = rng.normal(0, 3.0, (n, H, W))
    observation = np.clip(upstream + true_residual, 0, 24)
    members = true_residual[:, None] + rng.normal(0, 1.2, (n, MEMBERS, H, W))
    residual_mean = members.mean(axis=1)
    prediction = np.clip(upstream + residual_mean, 0, 24)
    _, curves = diagnostics.residual_diagnostics(observation, prediction, upstream, residual_mean, members,
                                                 occurrence_event=(0.0, True))
    items = pd.DataFrame({'date': ['2015-07-20'] * n, 'hour': [np.nan] * n})
    return curves, prediction, observation, items


# ⚠️ A's ``test_residual_figures_render_png_and_pdf`` rendered all six through ``write_report``. Superseded by
# ``test_each_residual_builder_renders_png_and_pdf_when_called_DIRECTLY`` below, which asserts the same files and is
# strictly stronger: ``write_report`` catches every exception and only warns, so its version reported a broken builder
# as a missing file with no traceback. Dropping it also saves one full set of cartopy renders.


def test_residual_figures_self_skip_without_a_residual_block(tmp_path):
    items = pd.DataFrame({'date': ['2015-07-20'], 'hour': [np.nan]})
    observation = np.zeros((1, H, W))
    reporting.write_report(str(tmp_path), {'figures': RESIDUAL_FIGURES, 'formats': ['png']}, {}, {},
                           observation, observation, items)
    assert not [name for name in os.listdir(str(tmp_path)) if name.startswith('residual_')]


# =====================================================================================================================
# write_report: the config-driven dispatch
# =====================================================================================================================
def test_write_report_renders_every_configured_figure(report_arrays, metrics_config, tmp_path):
    """The whole shipped figure list against a stochastic run, so the non-residual figures must all appear. This is
    the test that catches a figure raising inside the ``try/except`` that only warns.

    Two items only: the point is the figure SET, and each extra item adds a multi-second cartopy render."""
    prediction, observation, members, items, _ = report_arrays(n_items=2)
    curves = {
        'psd': {'wavelengths': np.array([2.0, 4.0, 8.0]), 'obs': np.array([1e3, 5e2, 2e2]),
                'model': np.array([9e2, 4e2, 1e2])},
        'fss': {'occurrence': {1: 0.4, 3: 0.6}},
        'reliability': {'mean_probability': [0.05, 0.5], 'observed_frequency': [0.04, 0.45], 'counts': [900, 90]},
        'roc_pr': {'occurrence': {'fpr': [0.0, 1.0], 'tpr': [0.0, 1.0], 'recall': [1.0, 0.0],
                                  'precision': [0.05, 1.0], 'roc_auc': 0.8, 'average_precision': 0.2,
                                  'base_rate': 0.05}},
        'confusion': {'h6': {'hits': 5, 'misses': 2, 'false_alarms': 3, 'correct_negatives': 900}},
        'error_by_bin': {'model': {'h3': 1.0}, 'zero': {'h3': 2.0}},
        'rank_histogram': {'counts': [10, 12, 9, 11, 13], 'n_members': MEMBERS},
    }
    reporting.write_report(str(tmp_path), metrics_config['reporting'], {'mae': 1.0}, curves,
                           prediction, observation, items, ensemble_members=members)

    produced = set(os.listdir(str(tmp_path)))
    expected = {'psd_curves', 'fss_vs_scale', 'reliability', 'roc_pr_curves', 'confusion_matrix',
                'error_by_intensity_bin', 'rank_histogram'}
    missing = [name for name in expected if f'{name}.png' not in produced]
    assert not missing, f'missing {missing} of {sorted(produced)}'
    assert any(name.startswith('maps_') and name.endswith('.png') for name in produced)
    assert 'metrics.csv' in produced, 'the flat metrics table is written whenever csv is requested'


# =====================================================================================================================
# The two panel layouts, inspected as FIGURES rather than as files
#
# The file-level tests above prove a day rendered; these prove WHAT was rendered. Both fixtures build one figure and
# are module-scoped, because a 6-panel cartopy figure is seconds.
# =====================================================================================================================
def _map_and_colorbar_axes(figure):
    """Split a report figure's axes into map panels (cartopy GeoAxes carry ``coastlines``) and colorbars."""
    maps = [axis for axis in figure.axes if hasattr(axis, 'coastlines')]
    return maps, [axis for axis in figure.axes if not hasattr(axis, 'coastlines')]


@pytest.fixture(scope='module')
def deterministic_figure():
    import matplotlib.pyplot as plt

    projection, data_crs = maps.geographic_context()
    prediction, observation, _, _, _ = _report_arrays(n_items=1, seed=3)
    figure = reporting._deterministic_day_figure(observation[0], prediction[0], '2015-07-14',
                                                 projection, data_crs)
    yield figure
    plt.close(figure)


@pytest.fixture(scope='module')
def stochastic_figure():
    import matplotlib.pyplot as plt

    projection, data_crs = maps.geographic_context()
    _, observation, members, _, _ = _report_arrays(n_items=1, seed=0)
    day = members[0]
    figure = reporting._stochastic_day_figure(observation[0], day.mean(axis=0), day.std(axis=0), day,
                                              '2015-07-14', projection, data_crs,
                                              np.random.default_rng(0))
    yield figure, day
    plt.close(figure)


def test_the_deterministic_layout_is_two_map_panels_and_ONE_colorbar(deterministic_figure):
    """Observations and predictions side by side, with a SINGLE colorbar serving both — which is what makes the two
    panels comparable by eye. A per-panel colour axis would rescale each independently and make a quiet day look like
    an active one. One bar, not the two the diff encoding needed (block 4e)."""
    panels, colorbars = _map_and_colorbar_axes(deterministic_figure)
    assert len(panels) == 2
    assert len(colorbars) == 1
    assert tuple(deterministic_figure.get_size_inches()) == (11.0, 5.5)


@pytest.mark.parametrize('layout', ['deterministic', 'stochastic'])
def test_a_SAVED_map_figure_keeps_its_FULL_canvas(tmp_path, layout):
    """🐛 The crop found by reading the block 4e gate's report, and the gap it exposed in this file.

    A cartopy ``GeoAxes`` returns a NON-FINITE tight bbox, and matplotlib's tight-bbox machinery silently DISCARDS
    non-finite bboxes instead of failing. With ``bbox_inches='tight'`` the saved box was therefore the union of the only
    finite artists — the two colorbars and the suptitle — so an 11 x 5.5 in figure was written as 895 x 771 px instead
    of 1650 x 825, the "Observed" panel was absent from the FILE, and the title sat at the left edge.

    ⚠️ Every test above passes on the figure OBJECT: two panels, two colorbars, the right figsize. All of them were
    green throughout, because nothing here had ever looked at what ``savefig`` actually wrote. That is the lesson worth
    keeping — a figure test that never opens the file cannot see half the failure modes of saving one.
    """
    import matplotlib.image as mpimg

    projection, data_crs = maps.geographic_context()
    prediction, observation, members, _, _ = _report_arrays(n_items=1, seed=1)
    if layout == 'deterministic':
        figure = reporting._deterministic_day_figure(observation[0], prediction[0], '2015-07-14',
                                                     projection, data_crs)
    else:
        day = members[0]
        figure = reporting._stochastic_day_figure(observation[0], day.mean(axis=0), day.std(axis=0), day,
                                                  '2015-07-14', projection, data_crs, np.random.default_rng(0))
    expected = tuple(int(round(inches * 150)) for inches in figure.get_size_inches())

    reporting._save_map_figure(figure, str(tmp_path), 'maps_canvas_probe')

    height, width = mpimg.imread(os.path.join(str(tmp_path), 'maps_canvas_probe.png')).shape[:2]
    assert (width, height) == expected, f'{layout}: saved {width}x{height}, figure declares {expected}'
    assert os.path.exists(os.path.join(str(tmp_path), 'maps_canvas_probe.pdf')), 'the vector copy is saved too'


def test_BOTH_panels_are_a_SINGLE_layer_on_the_SAME_scale(deterministic_figure):
    """What replaced the diff encoding in block 4e: the prediction is one plain layer, like the observation, and both
    read off the one colorbar. Two layers on the prediction panel would mean the masked over/under drawing came back —
    and with a single colorbar its cool half would have no scale to be read against."""
    panels, _ = _map_and_colorbar_axes(deterministic_figure)
    observed_image, predicted_image = panels[0].get_images(), panels[1].get_images()

    assert len(observed_image) == 1 and len(predicted_image) == 1
    assert observed_image[0].get_cmap().name == predicted_image[0].get_cmap().name
    assert list(observed_image[0].norm.boundaries) == list(predicted_image[0].norm.boundaries)


def test_the_stochastic_layout_is_a_two_by_three_grid_with_TWO_colorbars(stochastic_figure):
    """Six panels — observations, ensemble mean, ensemble std, and three members — and TWO colorbars: the shared
    lightning-hours bar, plus the std panel's own, which is on a different quantity and cannot share it."""
    figure, _ = stochastic_figure
    panels, colorbars = _map_and_colorbar_axes(figure)
    assert len(panels) == 6
    assert len(colorbars) == 2
    assert tuple(figure.get_size_inches()) == (15.0, 10.0)


def test_the_spread_panel_is_a_CONTINUOUS_viridis_layer_starting_at_zero(stochastic_figure):
    """Ensemble spread is not a signed error, so it gets neither the diff encoding nor the lightning-hours bins. Starting
    the colour axis at 0 is what makes "no spread" read as no spread rather than as the bottom of a rescaled range."""
    figure, _ = stochastic_figure
    panels, _ = _map_and_colorbar_axes(figure)
    spread_axis = panels[2]

    assert len(spread_axis.get_images()) == 1
    image = spread_axis.get_images()[0]
    assert image.get_cmap().name == 'viridis'
    assert image.get_clim()[0] == 0.0


def test_the_spread_colour_axis_is_derived_from_THE_DAY(stochastic_figure):
    """It was a hardcoded 8 (inventory issue #2), which saturates every active day to one colour and makes every quiet
    day look uniformly calm. Now it is the day's own maximum spread."""
    figure, members = stochastic_figure
    panels, _ = _map_and_colorbar_axes(figure)
    image = panels[2].get_images()[0]
    assert image.get_clim()[1] == pytest.approx(float(members.std(axis=0).max()), abs=1e-9)


def test_the_three_member_panels_share_the_OBSERVATION_scale(stochastic_figure):
    """Each member is a single plain layer on the figure's one palette, like every other lightning panel. Sharing the
    observation-driven scale is what lets a reader see that the members disagree about WHERE, not just by how much — if
    each member rescaled to its own range, three different-looking panels could hold the same field."""
    figure, _ = stochastic_figure
    panels, _ = _map_and_colorbar_axes(figure)
    reference = panels[0].get_images()[0]

    for axis in panels[3:6]:
        images = axis.get_images()
        assert len(images) == 1
        assert list(images[0].norm.boundaries) == list(reference.norm.boundaries)


def test_a_SHORT_ensemble_blanks_the_spare_panel_without_erroring():
    """``ensemble-size`` is a config value and the layout is fixed at three member panels, so M = 2 must leave the third
    empty rather than raise — an evaluation run should not die on a legal smoke-tier setting."""
    import matplotlib.pyplot as plt

    projection, data_crs = maps.geographic_context()
    _, observation, members, _, _ = _report_arrays(n_items=1, seed=0)
    day = members[0][:2]
    figure = reporting._stochastic_day_figure(observation[0], day.mean(axis=0), day.std(axis=0), day,
                                              '2015-07-14', projection, data_crs,
                                              np.random.default_rng(0))
    panels, _ = _map_and_colorbar_axes(figure)

    member_axes = panels[3:6]
    assert sum(len(axis.get_images()) == 1 for axis in member_axes) == 2
    assert sum(len(axis.get_images()) == 0 for axis in member_axes) == 1
    plt.close(figure)


# =====================================================================================================================
# The PSD figure: kilometres, and large scales on the LEFT
# =====================================================================================================================
def test_the_psd_x_axis_is_DESCENDING_so_large_scales_sit_on_the_left(tmp_path):
    """Read left to right as "synoptic to convective", which is how a meteorologist reads a spectrum. Ascending
    wavelength would put the 27 km pixel scale on the left and invert the whole story."""
    curves = {'psd': {'wavelengths': np.array([32.0, 16.0, 8.0, 4.0, 2.0]),
                      'obs': np.array([1e3, 5e2, 2e2, 8e1, 3e1]),
                      'model': np.array([9e2, 4e2, 1e2, 5e1, 2e1])}}

    # `_save_figure` closes the figure, so capture it on the way past. Rebuilding an axis here and calling
    # invert_xaxis on it would test matplotlib rather than `_psd_curves` — which is what the gate this replaces did.
    captured = []
    original = reporting._save_figure

    def spy(figure, *args, **kwargs):
        captured.append((figure.axes[0].get_xlim(), figure.axes[0].get_xscale()))
        return original(figure, *args, **kwargs)

    reporting._save_figure = spy
    try:
        reporting._psd_curves(curves, str(tmp_path), ['png'])
    finally:
        reporting._save_figure = original

    assert captured, '_psd_curves never saved a figure'
    (left, right), scale = captured[0]
    assert left > right, (left, right)
    assert scale == 'log'


@pytest.mark.source_invariant
def test_the_psd_axis_is_LABELLED_in_kilometres():
    """The pixel axis is what the FFT produces and the kilometre axis is what a reader can interpret. Both are in the
    table; only kilometres are on the figure."""
    import inspect

    source = inspect.getsource(reporting._psd_curves)
    assert 'invert_xaxis' in source
    assert 'Wavelength [km]' in source
    assert 'Wavelength [pixels]' not in source


def test_the_psd_curves_use_the_02a_colours():
    """Fixed rather than taken from the matplotlib cycle, so the observed and model curves mean the same thing in every
    figure of every report."""
    assert reporting.PSD_OBS_COLOR == 'steelblue'
    assert reporting.PSD_MODEL_COLOR == 'darkorange'


# =====================================================================================================================
# The figure set is exactly the inventory's fourteen
# =====================================================================================================================
def test_the_configured_figure_list_has_no_DUPLICATES(metrics_config):
    """A repeated name renders the same figure twice and, for the per-day maps, would overwrite its own files."""
    configured = metrics_config['reporting']['figures']
    assert len(configured) == len(set(configured)), configured


def test_all_FOURTEEN_inventory_figures_are_wired(metrics_config):
    """The count is the check: a figure the inventory decided to keep but nobody wired appears nowhere and raises
    nothing, because ``write_report`` warns and continues on an unknown name."""
    assert len(metrics_config['reporting']['figures']) == 14, metrics_config['reporting']['figures']


def test_the_never_implemented_qq_plot_is_gone_from_BOTH_the_config_and_the_code(metrics_config):
    """It was declared in the config and never implemented, so it self-skipped on every run — a figure that looked
    configured and produced nothing. Removed from both sides in §4."""
    import inspect

    assert 'qq_plot' not in metrics_config['reporting']['figures']
    assert 'qq_plot' not in inspect.getsource(reporting)


def test_the_map_figure_is_named_maps_most_extreme_days(metrics_config):
    """Renamed from ``maps_worst_best_days``: the selection is by four categories, not two, so the old name described a
    figure that no longer existed."""
    configured = metrics_config['reporting']['figures']
    assert 'maps_most_extreme_days' in configured
    assert 'maps_worst_best_days' not in configured


def test_the_confusion_matrix_AXES_match_the_way_its_CELLS_are_laid_out(tmp_path):
    """🐛 The transposition found by reading the Step 4 block 4e gate's report, pinned.

    ``misses`` is ``~pred & obs``, so ``[[hits, misses], [false_alarms, correct_negatives]]`` puts the OBSERVATION on
    the rows and the PREDICTION on the columns. The labels used to say the opposite, which made the "obs yes" column
    contain ``hits + false_alarms`` — for an over-forecasting model, every cell in the domain. The figure then read as
    "lightning was observed at every pixel", which is impossible on a target that is 95.3 % zero, and it is the sort of
    error that discredits a whole report rather than one panel.

    The four counts are deliberately DISTINCT so any transposition or rotation moves a value into a cell this test
    checks, and the labels are asserted together with the cell positions — either alone would still permit the bug.
    """
    captured = {}
    original = reporting._save_figure

    def spy(figure, path, name, formats):
        captured['axes'] = list(figure.axes)
        return original(figure, path, name, formats)

    curves = {'confusion': {'occurrence': {'hits': 11, 'misses': 22, 'false_alarms': 33,
                                           'correct_negatives': 44}}}
    reporting._save_figure = spy
    try:
        reporting._confusion_matrix(curves, str(tmp_path), ['png'])
    finally:
        reporting._save_figure = original

    axis = captured['axes'][0]
    assert [label.get_text() for label in axis.get_xticklabels()] == ['pred yes', 'pred no']
    assert [label.get_text() for label in axis.get_yticklabels()] == ['obs yes', 'obs no']

    cells = {(round(text.get_position()[0]), round(text.get_position()[1])): text.get_text()
             for text in axis.texts}
    assert cells[(0, 0)] == '11', 'pred yes / obs yes must be the HITS'
    assert cells[(1, 0)] == '22', 'pred NO / obs yes must be the MISSES'
    assert cells[(0, 1)] == '33', 'pred yes / obs NO must be the FALSE ALARMS'
    assert cells[(1, 1)] == '44', 'pred no / obs no must be the CORRECT NEGATIVES'


def test_the_confusion_CSV_is_written_from_the_NAMED_keys(tmp_path):
    """Why the transposition never reached the numbers: the CSV carries one named column per count, so it was right
    while the figure was wrong. A reader comparing the two would have found the figure at fault — which is exactly what
    happened."""
    curves = {'confusion': {'occurrence': {'hits': 11, 'misses': 22, 'false_alarms': 33,
                                           'correct_negatives': 44}}}
    reporting._confusion_matrix(curves, str(tmp_path), ['png', 'csv'])

    table = pd.read_csv(os.path.join(str(tmp_path), 'confusion_matrix.csv'))
    row = table.iloc[0]
    assert (row['hits'], row['misses'], row['false_alarms'], row['correct_negatives']) == (11, 22, 33, 44)


@pytest.mark.source_invariant
def test_LogNorm_survives_ONLY_for_the_confusion_matrix():
    """``colorbar_scale: log`` was removed from the map path with the 02a grammar: on a field that is 95.3 % zero a log
    colour axis makes an almost-empty map look populated. ``LogNorm`` is still imported, for the confusion matrix, whose
    counts genuinely span four orders of magnitude — so the import must remain while no map function uses it."""
    import ast

    from src.utils.metrics import reporting as reporting_module

    tree = ast.parse(open(reporting_module.__file__).read())
    users = [node.name for node in tree.body
             if isinstance(node, ast.FunctionDef)
             and 'LogNorm' in {inner.id for inner in ast.walk(node) if isinstance(inner, ast.Name)}]
    assert users == ['_confusion_matrix'], users


def test_every_configured_figure_has_a_dispatch_entry(metrics_config):
    """config <-> code parity. A configured name with no builder only produces a log warning, so the report would come
    out quietly short of a figure."""
    import ast
    import inspect

    source = inspect.getsource(reporting.write_report)
    tree = ast.parse(source.lstrip())
    builders = next(node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                    and getattr(node.targets[0], 'id', None) == 'builders')
    dispatched = {key.value for key in builders.keys}

    configured = set(metrics_config['reporting']['figures'])
    assert configured <= dispatched, f'configured but not dispatched: {sorted(configured - dispatched)}'
    assert dispatched <= configured, f'dispatched but never configured: {sorted(dispatched - configured)}'


def test_an_unknown_figure_name_is_skipped_rather_than_fatal(report_arrays, tmp_path, caplog):
    prediction, observation, _, items, _ = report_arrays()
    reporting.write_report(str(tmp_path), {'figures': ['not_a_figure'], 'formats': ['png']}, {}, {},
                           prediction, observation, items)
    assert not [name for name in os.listdir(str(tmp_path)) if name.endswith('.png')]


# =====================================================================================================================
# Block 5c — the private builders, driven DIRECTLY
#
# Everything above reaches these through ``write_report``, which wraps every figure in a ``try/except Exception`` that
# only warns. That is the right behaviour for a pipeline and the wrong one for a test: a builder that raises is
# indistinguishable from a builder that self-skipped, and the traceback is thrown away. Calling them directly is what
# makes a failure legible.
# =====================================================================================================================
def test_the_three_selection_categories_pick_the_right_ends_of_each_ordering():
    """``_category_selection`` is where "the most extreme days" is defined. Activity and error are ranked
    independently, because the worst-error day is often NOT the most active one — that divergence is the reason both
    categories exist rather than one."""
    totals = np.array([0.0, 50.0, 10.0, 30.0, 20.0])            # ascending order: 0, 2, 4, 3, 1
    errors = np.array([9.0, 1.0, 2.0, 3.0, 4.0])                # item 0 is the worst error and the LEAST active

    selection = reporting._category_selection(totals, errors, n_samples=2)

    assert selection['most_active'] == [1, 3], 'descending by observed total'
    assert selection['worst_error'] == [0, 4], 'descending by total absolute error'
    assert set(selection['median_activity']) <= {2, 4, 3}, 'drawn from the middle of the activity ordering'
    assert len(selection['median_activity']) == 2


def test_the_selection_asks_for_no_more_items_than_exist():
    """A smoke split is 2 days. Requesting 3 of each category must not index past the end."""
    selection = reporting._category_selection(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_samples=3)
    for indices in selection.values():
        assert all(0 <= int(index) < 2 for index in indices), selection


def test_extremeness_is_defined_by_the_OBSERVED_activity_not_the_predicted():
    """Otherwise a model that hallucinates a storm chooses which days get plotted, and the report's "most active days"
    become a picture of the model rather than of the split."""
    observation = np.zeros((3, 4, 4))
    observation[1] = 5.0                                        # item 1 is the only real activity
    prediction = np.zeros((3, 4, 4))
    prediction[2] = 99.0                                        # item 2 is a hallucination
    items = pd.DataFrame({'date': ['2015-07-20', '2015-07-21', '2015-07-22'], 'hour': [np.nan] * 3})

    selected = reporting._select_plot_indices(observation, prediction, items, plot_dates=None)
    most_active = next(index for index, tag in selected if tag == 'most_active')
    assert most_active == 1


def test_a_requested_date_comes_FIRST_and_is_not_repeated_by_a_category():
    """The dedup is by item index across all four categories, so a day that is both requested and the worst error is
    rendered once — a second render would overwrite the first file with a different category tag."""
    prediction, observation, _, items, dates = _report_arrays(n_items=4, seed=1)

    selected = reporting._select_plot_indices(observation, prediction, items, plot_dates=[dates[2]])
    indices = [index for index, _ in selected]

    assert selected[0] == (2, 'requested')
    assert len(indices) == len(set(indices)), f'an item was selected twice: {selected}'


def test_an_unknown_requested_date_WARNS_and_is_skipped(caplog):
    """``plot_dates`` is a user-supplied list against whatever split is being evaluated, so a date outside it is a
    typo or a wrong split — worth a warning, not worth losing the report over."""
    import logging

    prediction, observation, _, items, _ = _report_arrays(n_items=2)
    with caplog.at_level(logging.WARNING):
        selected = reporting._select_plot_indices(observation, prediction, items, plot_dates=['1999-01-01'])

    assert not any(tag == 'requested' for _, tag in selected)
    assert any('1999-01-01' in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------------------------------------------------
# Saving: the map figures ignore `formats`, the curve figures obey it
# ---------------------------------------------------------------------------------------------------------------------
def test_a_map_figure_is_saved_as_BOTH_png_and_pdf_whatever_formats_asks(tmp_path):
    """Deliberate asymmetry, and the reason there are two save helpers: maps are the publication output and get a
    vector copy unconditionally, while the curve/table figures follow the configured ``formats``."""
    import matplotlib.pyplot as plt

    figure = plt.figure()
    reporting._save_map_figure(figure, str(tmp_path), 'a_map')

    assert os.path.exists(os.path.join(str(tmp_path), 'a_map.png'))
    assert os.path.exists(os.path.join(str(tmp_path), 'a_map.pdf'))
    assert not plt.fignum_exists(figure.number), 'a report renders dozens of figures; leaking them exhausts memory'


def test_a_curve_figure_writes_no_png_when_png_was_not_requested(tmp_path):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    reporting._save_figure(figure, str(tmp_path), 'a_curve', formats=['csv'])

    assert not os.path.exists(os.path.join(str(tmp_path), 'a_curve.png'))
    assert not plt.fignum_exists(figure.number), 'the figure is closed even when nothing was written'


# ---------------------------------------------------------------------------------------------------------------------
# The residual map primitives
# ---------------------------------------------------------------------------------------------------------------------
def test_the_diverging_norm_is_SYMMETRIC_about_zero():
    """A signed residual read through an asymmetric colour axis puts white somewhere other than zero, so a map with no
    bias looks biased. Symmetry is the whole property."""
    values = np.array([-1.0, -0.5, 0.2, 3.0])
    norm = reporting._diverging_norm(values)
    assert norm.vmin == -norm.vmax
    assert abs(norm(0.0) - 0.5) < 1e-12, 'zero must land at the centre of the colormap'


def test_the_diverging_norm_uses_a_ROBUST_extent(): 
    """One extreme cell would otherwise set the axis and flatten everything else to white. The 99th percentile of
    ``|value|`` is what keeps the bulk of the field legible."""
    values = np.concatenate([np.full(999, 1.0), np.array([1000.0])])
    assert reporting._diverging_norm(values).vmax < 100.0


@pytest.mark.parametrize('values', [np.array([]), np.array([np.nan, np.inf]), np.zeros(10)])
def test_the_diverging_norm_never_collapses_to_a_zero_width_axis(values):
    """An all-zero residual map (a model that corrected nothing) or an all-NaN one would otherwise produce
    ``Normalize(0, 0)``, which matplotlib renders as a single flat colour with a division warning."""
    norm = reporting._diverging_norm(values)
    assert norm.vmax > norm.vmin


def test_a_solid_colormap_is_one_opaque_colour_with_TRANSPARENT_masked_cells():
    """The +/-inf surprise categories are drawn as overlays on top of the diverging field, so every cell outside the
    category has to be see-through — an opaque bad-colour would paint the whole panel."""
    cmap = reporting._solid_cmap(reporting.SURPRISE_OVER_COLOR)
    assert cmap.N == 1
    assert cmap.get_bad()[3] == 0.0, 'masked cells must be fully transparent'


def test_a_residual_field_is_drawn_with_the_SAME_geographic_footing_as_the_lightning_maps():
    """``origin='upper'`` with ``extent=GRID_EXTENT`` is what puts array row 0 at the NORTH edge. The residual maps are
    a separate code path from ``maps.draw_map``, so this invariant has to be pinned on both or one can drift and mirror
    its field about the domain's mid-latitude — a change no metric would notice."""
    class _Recorder:
        def __init__(self):
            self.kwargs = None

        def imshow(self, data, **kwargs):
            self.kwargs = kwargs
            return 'image'

    recorder = _Recorder()
    reporting._residual_imshow(recorder, np.zeros((4, 4)), 'viridis', None, 'a-crs')

    assert recorder.kwargs['origin'] == 'upper'
    assert recorder.kwargs['extent'] == maps.GRID_EXTENT
    assert recorder.kwargs['transform'] == 'a-crs'


def test_a_residual_panel_draws_one_layer_per_NON_EMPTY_special_mask():
    """The overlays are the ``+/-inf`` surprise categories. An empty category must add no layer at all — a fully-masked
    imshow still consumes a draw and, more to the point, would make "there are 3 layers" stop meaning "both categories
    occurred"."""
    import matplotlib.pyplot as plt

    projection, data_crs = maps.geographic_context()
    field = np.zeros((8, 8))
    norm = reporting._diverging_norm(field)
    empty, populated = np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=bool)
    populated[0, 0] = True

    figure = plt.figure(figsize=(4, 4))
    grid = figure.add_gridspec(1, 2)
    only_base = reporting._residual_map_panel(figure, grid[0, 0], field, norm, projection, data_crs, 'base',
                                              specials=[(empty, reporting.SURPRISE_OVER_COLOR)])
    with_overlay = reporting._residual_map_panel(figure, grid[0, 1], field, norm, projection, data_crs, 'overlay',
                                                 specials=[(populated, reporting.SURPRISE_OVER_COLOR),
                                                           (empty, reporting.SURPRISE_UNDER_COLOR)])

    assert len(only_base.get_images()) == 1
    assert len(with_overlay.get_images()) == 2
    assert only_base.get_title() == 'base'
    plt.close(figure)


def test_the_residual_colorbar_is_labelled_and_carries_the_diverging_map():
    """It is detached from any axes, so it takes its colormap explicitly — a mismatch with the panel it describes would
    be invisible in the figure and wrong in every reading of it."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(4, 2))
    cax = figure.add_subplot(1, 1, 1)
    reporting._diverging_colorbar(figure, cax, reporting._diverging_norm(np.array([-1.0, 1.0])), 'a label')

    assert cax.get_xlabel() == 'a label'
    assert cax.images or cax.collections or cax.patches, 'the colorbar must actually be drawn into the axes'
    plt.close(figure)


# ---------------------------------------------------------------------------------------------------------------------
# The curve figures, per-figure properties
# ---------------------------------------------------------------------------------------------------------------------
def test_the_fss_axis_is_pinned_to_zero_ONE_with_the_half_reference(tmp_path):
    """FSS is a fraction and the useful-scale criterion sits just above 0.5. An autoscaled axis would make an FSS of
    0.55 fill the panel and read as a strong result."""
    captured = {}
    original = reporting._save_figure

    def spy(fig, path, name, formats):
        captured['axis'] = fig.axes[0]
        return original(fig, path, name, formats)

    curves = {'fss': {'h6': {1: 0.30, 3: 0.42, 5: 0.55}}}
    reporting._save_figure = spy
    try:
        reporting._fss_vs_scale(curves, str(tmp_path), ['png'])
    finally:
        reporting._save_figure = original

    assert os.path.exists(os.path.join(str(tmp_path), 'fss_vs_scale.png'))
    assert captured['axis'].get_ylim() == (0.0, 1.0)
    assert any(abs(line.get_ydata()[0] - 0.5) < 1e-9 for line in captured['axis'].lines
               if len(line.get_ydata()) and line.get_linestyle() == '--')


def test_the_reliability_figure_carries_the_perfect_diagonal_and_a_LOG_count_panel(tmp_path):
    """Two panels, and the second is the one that makes the first readable: at a 0.43 % base rate almost every cell
    lands in the lowest probability bin, so a linear count axis shows one bar and nothing else. The diagonal is the
    reference a reliability curve is read against."""
    captured = {}
    original = reporting._save_figure

    def spy(fig, path, name, formats):
        captured['axes'] = list(fig.axes)
        return original(fig, path, name, formats)

    curves = {'reliability': {'mean_probability': [0.02, 0.3, 0.8], 'observed_frequency': [0.01, 0.35, 0.7],
                              'counts': [100000, 500, 20]}}
    reporting._save_figure = spy
    try:
        reporting._reliability(curves, str(tmp_path), ['png', 'csv'])
    finally:
        reporting._save_figure = original

    reliability_axis, counts_axis = captured['axes']
    assert any(list(line.get_xdata()) == [0, 1] and list(line.get_ydata()) == [0, 1]
               for line in reliability_axis.lines), 'the perfect-calibration diagonal is missing'
    assert counts_axis.get_yscale() == 'log'
    assert os.path.exists(os.path.join(str(tmp_path), 'reliability_table.csv'))


def test_the_pr_panel_carries_a_no_skill_line_at_the_BASE_RATE(tmp_path):
    """The documented reason both panels are drawn: the ROC diagonal is universal, the PR floor is not. At this base
    rate a precision of 0.05 is five times no-skill, and only the second panel's own reference line says so."""
    captured = {}
    original = reporting._save_figure

    def spy(fig, path, name, formats):
        captured['axes'] = list(fig.axes)
        return original(fig, path, name, formats)

    curves = {'roc_pr': {'occurrence': {'fpr': [0.0, 0.4, 1.0], 'tpr': [0.0, 0.9, 1.0],
                                        'recall': [1.0, 0.5, 0.0], 'precision': [0.01, 0.2, 1.0],
                                        'roc_auc': 0.93, 'average_precision': 0.18, 'base_rate': 0.01}}}
    reporting._save_figure = spy
    try:
        reporting._roc_pr_curves(curves, str(tmp_path), ['png', 'csv'])
    finally:
        reporting._save_figure = original

    roc_axis, pr_axis = captured['axes']
    assert any(list(line.get_ydata()) == [0, 1] for line in roc_axis.lines), 'the ROC diagonal'
    assert any(abs(line.get_ydata()[0] - 0.01) < 1e-9 for line in pr_axis.lines
               if line.get_linestyle() == ':'), 'the PR no-skill line at the base rate'
    assert pr_axis.get_yscale() == 'log'


def test_a_ranking_block_with_EMPTY_curves_writes_nothing(tmp_path):
    """``roc_pr`` is present but carries no points — the shape ``finalize_ranking_metrics`` returns when the split held
    no positive cells. An empty pair of axes would be published as a figure."""
    curves = {'roc_pr': {'occurrence': {'fpr': [], 'tpr': [], 'recall': [], 'precision': []}}}
    reporting._roc_pr_curves(curves, str(tmp_path), ['png'])
    assert not [name for name in os.listdir(str(tmp_path)) if name.endswith('.png')]


def test_the_ranking_curves_export_their_POINTS_as_well_as_the_summary(tmp_path):
    """Two CSVs, because AUC and AP cannot be turned back into a curve. ``combine_curves`` overlays the families' ROC and
    PR curves, and the points are the only thing that makes that possible — every other curve figure already exports its
    own, so the summary-only export was the odd one out."""
    curves = {'roc_pr': {
        'occurrence': {'fpr': [0.0, 0.4, 1.0], 'tpr': [0.0, 0.9, 1.0], 'recall': [0.0, 0.9, 1.0],
                       'precision': [1.0, 0.2, 0.01], 'roc_auc': 0.93, 'average_precision': 0.18,
                       'base_rate': 0.01},
        'h6': {'fpr': [0.0, 0.6, 1.0], 'tpr': [0.0, 0.7, 1.0], 'recall': [0.0, 0.7, 1.0],
               'precision': [1.0, 0.1, 0.004], 'roc_auc': 0.71, 'average_precision': 0.06, 'base_rate': 0.002},
    }}
    reporting._roc_pr_curves(curves, str(tmp_path), ['csv'])

    points = pd.read_csv(os.path.join(str(tmp_path), 'roc_pr_curves.csv'))
    assert list(points.columns) == ['threshold', 'fpr', 'tpr', 'recall', 'precision']
    assert len(points) == 6, 'both thresholds, all three cuts each'
    assert set(points['threshold']) == {'occurrence', 'h6'}
    # the declaration order is preserved, which is what combine_curves' first-listed fallback relies on
    assert points['threshold'].iloc[0] == 'occurrence'

    summary = pd.read_csv(os.path.join(str(tmp_path), 'roc_pr_summary.csv'))
    assert {'roc_auc', 'average_precision', 'base_rate'} <= set(summary.columns), \
        'the summary still carries the scalars the legend and the no-skill line need'


def test_a_ranking_block_with_EMPTY_curves_writes_NEITHER_csv(tmp_path):
    """The ``not drawn`` return sits BEFORE the csv block, so a split with no positive cells produces no ranking tables at
    all — not a header-only points file and not an all-NaN summary. The scalars are not lost: ``roc_auc_*`` and
    ``average_precision_*`` are in the flat metrics JSON, NaN like every other undefined score."""
    curves = {'roc_pr': {'occurrence': {'fpr': [], 'tpr': [], 'recall': [], 'precision': [],
                                        'roc_auc': float('nan'), 'average_precision': float('nan')}}}
    reporting._roc_pr_curves(curves, str(tmp_path), ['csv'])

    assert not [name for name in os.listdir(str(tmp_path)) if name.startswith('roc_pr')]


def test_the_intensity_bin_table_keeps_the_BINS_as_its_row_labels(tmp_path):
    """The one CSV in this module written with its index — the bins are the row identity, and dropping them leaves a
    table of unlabelled numbers."""
    curves = {'error_by_bin': {'model': {'0-1h': 0.4, '1-6h': 2.0, '6h+': 5.0},
                               'climatology': {'0-1h': 0.9, '1-6h': 3.0, '6h+': 7.0}}}
    reporting._error_by_intensity_bin(curves, str(tmp_path), ['png', 'csv'])

    table = pd.read_csv(os.path.join(str(tmp_path), 'error_by_intensity_bin.csv'), index_col=0)
    assert list(table.index) == ['0-1h', '1-6h', '6h+']
    assert {'model', 'climatology'} <= set(table.columns)


def test_the_rank_histogram_normalises_to_FREQUENCIES_against_a_uniform_reference(tmp_path):
    """Reading it is a comparison against uniform, so the bars must be frequencies and the reference must be
    ``1 / (M + 1)``. Raw counts would make the reference height depend on the split size."""
    captured = {}
    original = reporting._save_figure

    def spy(fig, path, name, formats):
        captured['axis'] = fig.axes[0]
        return original(fig, path, name, formats)

    curves = {'rank_histogram': {'counts': [30, 10, 10, 10, 40], 'n_members': 4}}
    reporting._save_figure = spy
    try:
        reporting._rank_histogram(curves, str(tmp_path), ['png', 'csv'])
    finally:
        reporting._save_figure = original

    heights = [patch.get_height() for patch in captured['axis'].patches]
    assert abs(sum(heights) - 1.0) < 1e-9, heights
    assert any(abs(line.get_ydata()[0] - 0.2) < 1e-9 for line in captured['axis'].lines)

    table = pd.read_csv(os.path.join(str(tmp_path), 'rank_histogram.csv'))
    assert list(table['count']) == [30, 10, 10, 10, 40]


def test_an_EMPTY_rank_histogram_writes_nothing_rather_than_dividing_by_zero(tmp_path):
    """All-zero counts is what a deterministic family produces if the curve is populated at all, and normalising it
    would be ``0 / 0``."""
    reporting._rank_histogram({'rank_histogram': {'counts': [0, 0, 0], 'n_members': 2}}, str(tmp_path), ['png'])
    assert not os.listdir(str(tmp_path))


# ---------------------------------------------------------------------------------------------------------------------
# The six residual builders, called directly
# ---------------------------------------------------------------------------------------------------------------------
RESIDUAL_BUILDERS = [
    reporting._residual_bias_map, reporting._residual_surprise, reporting._residual_histograms,
    reporting._residual_qq, reporting._residual_scatters, reporting._residual_heteroscedasticity,
]


@pytest.mark.parametrize('builder', RESIDUAL_BUILDERS, ids=lambda builder: builder.__name__)
def test_each_residual_builder_renders_png_and_pdf_when_called_DIRECTLY(builder, residual_curves, tmp_path):
    """The same six figures the ported test renders through ``write_report`` — but that path swallows exceptions and
    reports only a missing file. Here a broken builder surfaces its traceback."""
    curves, _, _, _ = residual_curves
    builder(curves, str(tmp_path), ['png', 'csv'])

    name = builder.__name__.lstrip('_')
    assert os.path.exists(os.path.join(str(tmp_path), f'{name}.png')), sorted(os.listdir(str(tmp_path)))
    assert os.path.exists(os.path.join(str(tmp_path), f'{name}.pdf'))


@pytest.mark.parametrize('builder', RESIDUAL_BUILDERS, ids=lambda builder: builder.__name__)
def test_each_residual_builder_self_skips_without_its_block(builder, tmp_path):
    """Two skips, not one: no ``residual`` key at all (a deterministic or full-target run) and a residual block missing
    this figure's own key (a diagnostics version that computed less)."""
    builder({}, str(tmp_path), ['png', 'csv'])
    builder({'residual': {}}, str(tmp_path), ['png', 'csv'])
    assert not os.listdir(str(tmp_path))


def test_the_heteroscedasticity_figure_skips_when_every_decile_is_EMPTY(tmp_path):
    """The block exists but holds no bins — what ``_decile_heteroscedasticity`` returns when the conditioning field has
    no spread. The builder opens a figure before it knows that, so the guard has to close it as well as return."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    reporting._residual_heteroscedasticity(
        {'residual': {'heteroscedasticity': {'upstream': {'bin_center': []}, 'obs': {'bin_center': []}}}},
        str(tmp_path), ['png'])

    assert not os.listdir(str(tmp_path))
    assert len(plt.get_fignums()) == before, 'the opened figure must be closed on the skip path'


# =====================================================================================================================
# Titles: the global one names the event AND the family, every panel names its own field
# =====================================================================================================================
def test_the_GLOBAL_title_names_the_event_and_the_family(tmp_path):
    """`<date> event, <family> model`. The family matters on a figure that will sit beside two others in a comparison:
    a map captioned only by date is unattributable the moment it leaves its own report directory."""
    prediction, observation, _, items, dates = _report_arrays(n_items=2, seed=3)

    captured = []
    original = reporting._deterministic_day_figure

    def spy(observation_map, prediction_map, title, *args, **kwargs):
        captured.append(title)
        return original(observation_map, prediction_map, title, *args, **kwargs)

    reporting._deterministic_day_figure = spy
    try:
        reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path),
                                         model_family='mc_dropout')
    finally:
        reporting._deterministic_day_figure = original

    assert captured, 'no figure was built'
    for title in captured:
        assert title.endswith(' event, mc_dropout model'), title
        assert any(title.startswith(date) for date in dates), title


def test_an_UNKNOWN_family_leaves_the_title_at_the_event_alone(tmp_path):
    """`model_family` is optional, and a figure built without one must not say "None model". Omitted rather than
    guessed: `maps_most_extreme_days` is reachable outside the evaluate stage, where no checkpoint resolved a family."""
    prediction, observation, _, items, _ = _report_arrays(n_items=2, seed=3)

    captured = []
    original = reporting._deterministic_day_figure

    def spy(observation_map, prediction_map, title, *args, **kwargs):
        captured.append(title)
        return original(observation_map, prediction_map, title, *args, **kwargs)

    reporting._deterministic_day_figure = spy
    try:
        reporting.maps_most_extreme_days(prediction, observation, items, str(tmp_path))
    finally:
        reporting._deterministic_day_figure = original

    for title in captured:
        assert title.endswith(' event'), title
        assert 'None' not in title and 'model' not in title


def test_the_DETERMINISTIC_panels_are_titled_observations_and_predictions(deterministic_figure):
    panels, _ = _map_and_colorbar_axes(deterministic_figure)
    assert [axis.get_title() for axis in panels] == ['observations', 'predictions']


def test_the_STOCHASTIC_panels_name_the_mean_the_std_and_each_MEMBER(stochastic_figure):
    """Six titles, in layout order. The members are numbered by DRAW order rather than by their index in the stack:
    MC-dropout and diffusion members are exchangeable, so a stack position is not a fact about the member."""
    figure, _ = stochastic_figure
    panels, _ = _map_and_colorbar_axes(figure)

    assert [axis.get_title() for axis in panels] == [
        'observations', 'ensemble mean', 'ensemble std', 'member 1', 'member 2', 'member 3',
    ]


def test_a_SHORT_ensemble_titles_only_the_members_it_HAS():
    """M = 2 is a legal smoke-tier setting, and the blanked third slot must carry no title — a "member 3" label over an
    empty panel would read as a member that failed to render."""
    import matplotlib.pyplot as plt

    projection, data_crs = maps.geographic_context()
    _, observation, members, _, _ = _report_arrays(n_items=1, seed=0)
    day = members[0][:2]
    figure = reporting._stochastic_day_figure(observation[0], day.mean(axis=0), day.std(axis=0), day,
                                              '2015-07-14 event, mc_dropout model', projection, data_crs,
                                              np.random.default_rng(0))
    panels, _ = _map_and_colorbar_axes(figure)

    titles = [axis.get_title() for axis in panels]
    assert titles[3:] == ['member 1', 'member 2', '']
    plt.close(figure)
