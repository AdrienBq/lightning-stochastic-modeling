"""Tests for src/utils/plotting/__init__.py — ``show_plot_and_save``, and the font side effect.

Untouched plumber template code. It gets a test file because the package ``__init__`` carries real behaviour rather than
being a marker, and because ``show_plot_and_save`` has one property the whole suite depends on without saying so: it
CLOSES the figure. A report run renders a few hundred figures, and matplotlib keeps every unclosed one alive.

Note the report figures in ``metrics/reporting.py`` do NOT go through this helper — they call ``figure.savefig`` directly
via ``_save_figure`` / ``_save_map_figure``, because they need png+pdf and a fixed dpi rather than the ``show`` behaviour.
So this is the interactive/notebook-facing path, kept because the template ships it.
"""
import os

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                              # noqa: E402

from src.utils.plotting import linestyles, markers, show_plot_and_save        # noqa: E402


@pytest.fixture
def figure():
    made = plt.figure()
    made.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
    yield made
    plt.close(made)


def test_it_writes_the_file_and_returns_the_path(figure, tmp_path):
    path = show_plot_and_save(fig=figure, output_dir=str(tmp_path),
                              output_filename_pattern='plot_{}.png', show=False)
    assert os.path.exists(path)
    assert os.path.basename(path) == 'plot_.png'
    assert os.path.getsize(path) > 0


def test_the_args_are_joined_with_underscores(figure, tmp_path):
    path = show_plot_and_save('mae', 'valid', fig=figure, output_dir=str(tmp_path),
                              output_filename_pattern='plot_{}.png', show=False)
    assert os.path.basename(path) == 'plot_mae_valid.png'


def test_none_args_are_dropped_rather_than_stringified(figure, tmp_path):
    """Documented behaviour, and the reason a caller can pass an optional label straight through without branching. A
    naive join would produce ``mae_None_valid``."""
    path = show_plot_and_save('mae', None, 'valid', fig=figure, output_dir=str(tmp_path),
                              output_filename_pattern='plot_{}.png', show=False)
    assert os.path.basename(path) == 'plot_mae_valid.png'
    assert 'None' not in path


def test_the_figure_is_CLOSED_after_saving(figure, tmp_path):
    """The property the whole suite leans on. matplotlib holds every unclosed figure in its global registry, so a report
    run that renders a few hundred would accumulate all of them — and the warning about it only appears after 20."""
    show_plot_and_save(fig=figure, output_dir=str(tmp_path), output_filename_pattern='plot_{}.png', show=False)
    assert figure.number not in plt.get_fignums()


def test_it_saves_at_the_requested_dpi(figure, tmp_path):
    low = show_plot_and_save('low', fig=plt.figure(), output_dir=str(tmp_path),
                             output_filename_pattern='{}.png', show=False, dpi=50)
    high_figure = plt.figure()
    high_figure.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
    high = show_plot_and_save('high', fig=high_figure, output_dir=str(tmp_path),
                              output_filename_pattern='{}.png', show=False, dpi=200)
    assert os.path.getsize(high) > os.path.getsize(low)


def test_a_pattern_WITHOUT_a_placeholder_silently_ignores_the_args(figure, tmp_path):
    """A documented footgun rather than an error: `str.format` discards extra positional arguments, so a pattern with no
    `{}` writes EVERY figure to the same name and each one overwrites the last. No exception, no warning.

    Pinned rather than fixed — it is template code, and the report path does not use this helper. But it is exactly the
    collision `maps_most_extreme_days` goes to some trouble to avoid with its per-category ordinal, so it is worth having
    written down.
    """
    first = show_plot_and_save('a', fig=figure, output_dir=str(tmp_path),
                               output_filename_pattern='no_placeholder.png', show=False)
    second_figure = plt.figure()
    second_figure.add_subplot(1, 1, 1).plot([0, 1], [1, 0])
    second = show_plot_and_save('b', fig=second_figure, output_dir=str(tmp_path),
                                output_filename_pattern='no_placeholder.png', show=False)
    assert first == second, 'both figures went to one file'
    assert len([name for name in os.listdir(str(tmp_path)) if name.endswith('.png')]) == 1


def test_the_marker_and_linestyle_cycles_are_usable():
    """Offered alongside the colour prop cycle so a figure with more series than colours can still distinguish them —
    and so a print-in-greyscale figure remains readable."""
    assert len(markers) >= 4 and len(set(markers)) == len(markers)
    assert len(linestyles) >= 4 and len(set(linestyles)) == len(linestyles)
    for style in linestyles:
        assert style in ('solid', 'dashed', 'dashdot', 'dotted')


def test_importing_the_package_registers_the_bundled_font():
    """The template bundles Inter and applies it globally at import. Asserting the rcParam rather than the glyph
    rendering: a missing font falls back silently, which is cosmetic, but a changed rcParam means the side effect was
    dropped."""
    assert plt.rcParams['font.family'] == ['Inter']
