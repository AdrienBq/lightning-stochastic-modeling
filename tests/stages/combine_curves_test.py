"""Tests for src/stages/combine_curves.py — ⏳ **STEP 4 PLACEHOLDER, entirely skipped.**

Ported from the combine half of branch A's ``tests/test_tabulate_and_combine.py``. The module does not exist yet and no
shipped config references it: it overlays several runs' curve CSVs onto one figure per curve type, for cross-family
comparison.

**Pre-fixed for Step 2's contract, and this is where the port is not mechanical.** A's version wrote FOUR curve pairs,
one of which was ``qq`` — the quantile-quantile curve, which was declared in ``metrics.yaml`` but never implemented, and
which Step 2 removed outright along with ``scores.quantile_quantile``. So A's
``test_combine_curves_writes_all_four_pairs`` and ``test_combine_curves_survives_empty_or_junk_qq_csv`` are replaced
below by their three-pair equivalents. Porting the four-pair version unchanged would have demanded a curve type this
repo deliberately does not produce.

**To enable in Step 4:** delete the ``pytestmark`` line.
"""
import os

import pytest

pytestmark = pytest.mark.skip(reason='src/stages/combine_curves.py is Step 4; delete this line to enable')

# `qq` is deliberately absent — declared but never implemented on branch A, removed in Step 2 along with
# scores.quantile_quantile, because a staircase over ~24 integers says nothing about a bounded target.
CURVE_TYPES = ('psd_curves', 'fss_table', 'reliability_table')


@pytest.fixture
def report_dirs(tmp_path):
    """Two runs' report directories, each holding the curve CSVs that `write_report` emits."""
    def build(include=CURVE_TYPES):
        dirs = {}
        for name in ('U_net', 'MC_Dropout'):
            directory = tmp_path / name
            directory.mkdir()
            for curve in include:
                (directory / f'{curve}.csv').write_text('x,y\n1,0.5\n2,0.6\n')
            dirs[name] = str(directory)
        return dirs
    return build


def test_one_figure_is_written_per_curve_type(report_dirs, tmp_path):
    import combine_curves as combine_stage

    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **report_dirs())
    produced = set(os.listdir(out))
    for curve in CURVE_TYPES:
        assert any(name.startswith(curve) for name in produced), f'{curve} missing from {sorted(produced)}'


def test_a_missing_curve_csv_is_tolerated(report_dirs, tmp_path):
    """A deterministic run has no rank histogram and a non-residual run no residual curves, so a run legitimately lacks
    curve types the others have — the same self-skip logic `write_report` uses."""
    import combine_curves as combine_stage

    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **report_dirs(include=('psd_curves',)))
    assert os.listdir(out)


def test_an_empty_or_malformed_csv_does_not_abort_the_comparison(report_dirs, tmp_path):
    """A truncated CSV from an interrupted run must not lose the whole cross-family figure set."""
    import combine_curves as combine_stage

    dirs = report_dirs()
    open(os.path.join(dirs['U_net'], 'psd_curves.csv'), 'w').close()             # empty
    with open(os.path.join(dirs['MC_Dropout'], 'fss_table.csv'), 'w') as handle:
        handle.write('not,a,valid\ncurve\n')                                     # ragged

    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **dirs)                                    # must not raise
    assert os.path.isdir(out)


def test_no_qq_figure_is_produced(report_dirs, tmp_path):
    """The removal, pinned. `qq_plot` was declared in metrics.yaml but never implemented, and dropping it also closed
    that inconsistency — so a `qq` curve reappearing here would mean it came back."""
    import combine_curves as combine_stage

    out = str(tmp_path / 'combined')
    combine_stage.combine_curves(out, **report_dirs())
    assert not any('qq' in name for name in os.listdir(out))
