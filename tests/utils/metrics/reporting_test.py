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
    """The self-skip is what lets metrics.yaml list all fourteen figures unconditionally: a deterministic run never
    populates ``rank_histogram`` and a non-residual run never populates ``residual``."""
    getattr(reporting, f'_{figure}')({}, str(tmp_path), ['png', 'csv'])
    assert not os.listdir(str(tmp_path))


# =====================================================================================================================
# The six residual figures  (ported)
# =====================================================================================================================
@pytest.fixture
def residual_curves():
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


def test_residual_figures_render_png_and_pdf(residual_curves, tmp_path):
    curves, prediction, observation, items = residual_curves
    reporting.write_report(str(tmp_path), {'figures': RESIDUAL_FIGURES, 'formats': ['png', 'csv']}, {}, curves,
                           prediction, observation, items)
    for figure in RESIDUAL_FIGURES:
        assert os.path.exists(os.path.join(str(tmp_path), f'{figure}.png')), f'{figure}.png'
        assert os.path.exists(os.path.join(str(tmp_path), f'{figure}.pdf')), f'{figure}.pdf'


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
