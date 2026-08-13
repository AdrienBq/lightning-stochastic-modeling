"""Tests for src/stages/tabulate_metrics.py — ⏳ **STEP 4 PLACEHOLDER, entirely skipped.**

Ported from the tabulate half of branch A's ``tests/test_tabulate_and_combine.py``. The module does not exist yet, and no
shipped config references it — it is a cross-family comparison stage that collects several runs' ``metrics.json`` into one
table.

**Pre-fixed for Step 2's contract:** ``ets_p99`` was re-keyed to the absolute band ``ets_h6``, so the
selected-metrics tests below name the new key. A mechanical port would silently select a metric that no longer exists and
still pass, because the stage SKIPS unknown names by design.

**To enable in Step 4:** delete the ``pytestmark`` line and import the stage by its bare name (see ``conftest.py`` —
stage modules are not importable as ``src.stages.X``).
"""
import json
import os

import pytest

pytestmark = pytest.mark.skip(reason='src/stages/tabulate_metrics.py is Step 4; delete this line to enable')


@pytest.fixture
def metrics_files(tmp_path):
    """Three families' metrics JSONs with deliberately OVERLAPPING but unequal key sets."""
    def build():
        paths = {}
        for name, metrics in {
            'U_net': {'mae': 1.0, 'ets_h6': 0.3, 'psd_full_fidelity': 0.7},
            'MC_Dropout': {'mae': 0.9, 'ets_h6': 0.35, 'crps': 0.4, 'spread_skill_ratio': 0.8},
            'Diffusion_Model': {'mae': 1.1, 'ets_h6': 0.31, 'crps': 0.38},
        }.items():
            path = tmp_path / f'{name}.json'
            path.write_text(json.dumps(metrics))
            paths[name] = str(path)
        return paths
    return build


def test_the_table_is_the_UNION_of_every_runs_metrics(metrics_files, tmp_path):
    """A deterministic run has no ``crps`` and a stochastic one has no reason to lack ``mae``, so an intersection would
    drop exactly the metrics that distinguish the families."""
    import tabulate_metrics as tabulate_stage

    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())

    import pandas as pd
    table = pd.read_csv(out)
    assert {'mae', 'ets_h6', 'crps', 'psd_full_fidelity', 'spread_skill_ratio'} <= set(table.columns) | set(
        table.iloc[:, 0].astype(str))


def test_the_display_names_come_from_the_keyword_arguments(metrics_files, tmp_path):
    """``U_net=...`` becomes the column label, so a table can be built from arbitrary run directories without a naming
    convention."""
    import tabulate_metrics as tabulate_stage

    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())
    contents = open(out).read()
    for label in ('U_net', 'MC_Dropout', 'Diffusion_Model'):
        assert label in contents


def test_selected_metrics_restricts_the_table_and_ignores_unknown_names(metrics_files, tmp_path):
    """Unknown names are skipped rather than fatal, which is what makes one selection list reusable across families —
    and also why the ``ets_p99`` -> ``ets_h6`` re-key had to be applied here rather than left to fail loudly."""
    import tabulate_metrics as tabulate_stage

    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, selected_metrics=['crps', 'ets_h6', 'not_a_metric'],
                            MC_Dropout=metrics_files()['MC_Dropout'])
    contents = open(out).read()
    assert 'ets_h6' in contents and 'crps' in contents
    assert 'not_a_metric' not in contents
    assert 'mae' not in contents


def test_selected_metrics_accepts_a_comma_separated_string(metrics_files, tmp_path):
    """Because it arrives from a YAML scalar via fire, not as a Python list."""
    import tabulate_metrics as tabulate_stage

    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, selected_metrics='ets_h6,crps',
                            MC_Dropout=metrics_files()['MC_Dropout'])
    contents = open(out).read()
    assert 'ets_h6' in contents and 'crps' in contents


def test_a_missing_run_is_skipped_but_ALL_missing_raises(metrics_files, tmp_path):
    """Skipping one absent run lets a partial comparison proceed; producing an empty table silently would look like a
    successful comparison of nothing."""
    import tabulate_metrics as tabulate_stage

    paths = metrics_files()
    absent = str(tmp_path / 'absent.json')

    tabulate_stage.tabulate(str(tmp_path / 'partial.csv'), MC_Dropout=paths['MC_Dropout'], U_net=absent)

    with pytest.raises(Exception):
        tabulate_stage.tabulate(str(tmp_path / 'empty.csv'), U_net=absent)
