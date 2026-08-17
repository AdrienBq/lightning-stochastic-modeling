"""Tests for src/stages/tabulate_metrics.py — the cross-family comparison table.

Ported from the tabulate half of branch A's ``tests/test_tabulate_and_combine.py``, with two corrections the mechanical
port would have missed:

* **the metric keys.** ``ets_p99`` was re-keyed to the absolute band ``ets_h6`` in Step 2, and the stage SKIPS unknown
  selected metrics by design — so a fixture naming the old key would still pass while selecting nothing;
* **the row labels.** A carried a hardcoded ``_DISPLAY_NAMES`` map covering ``U_net`` / ``Diffusion_Model``, neither of
  which is a label any shipped config uses. Its tests asserted those same invented labels, so the mislabelling was
  invisible. The labels are now derived (Fire's hyphen substitution undone) and pinned against the SHIPPED config.
"""
import json
import os

import pytest

import tabulate_metrics as tabulate_stage

# The labels config/eval/probabilistic_eval.yaml actually uses, as Fire delivers them (hyphen -> underscore).
FIRE_KEYS = ('Deterministic_UNet', 'MC_Dropout', 'Diffusion')


@pytest.fixture
def metrics_files(tmp_path):
    """Three families' metrics JSONs with deliberately OVERLAPPING but unequal key sets — the real asymmetry: only the
    stochastic families have ``crps`` / ``spread_skill_ratio``, and only the deterministic one was given a key of its
    own, so an intersection would be detectably narrower than the union in both directions."""
    def build():
        paths = {}
        for name, metrics in {
            'Deterministic_UNet': {'mae': 1.0, 'ets_h6': 0.30, 'psd_full_fidelity': 0.7},
            'MC_Dropout': {'mae': 0.9, 'ets_h6': 0.35, 'crps': 0.40, 'spread_skill_ratio': 0.8},
            'Diffusion': {'mae': 1.1, 'ets_h6': 0.31, 'crps': 0.38},
        }.items():
            path = tmp_path / f'{name}.json'
            path.write_text(json.dumps(metrics))
            paths[name] = str(path)
        return paths
    return build


def _read(path):
    import pandas as pd
    return pd.read_csv(path, index_col=0)


# =====================================================================================================================
# The table
# =====================================================================================================================
def test_the_table_is_the_UNION_of_every_families_metrics(metrics_files, tmp_path):
    """A deterministic family has no ``crps`` and no reason to lack ``mae``, so an intersection would drop exactly the
    columns that distinguish the families — and the comparison would silently become a comparison of what they share."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())

    table = _read(out)
    assert set(table.columns) == {'mae', 'ets_h6', 'crps', 'psd_full_fidelity', 'spread_skill_ratio'}
    assert table.shape[0] == 3


def test_a_metric_a_family_LACKS_is_NaN_rather_than_absent(metrics_files, tmp_path):
    """What makes the columns identical across families: the deterministic row carries NaN for the ensemble scalars
    instead of the table growing a ragged shape. That NaN pattern is the only place the asymmetry shows."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())

    table = _read(out)
    assert table.loc['Deterministic-UNet'].isna()['crps']
    assert not table.loc['MC-Dropout'].isna()['crps']
    assert not table.loc['Deterministic-UNet'].isna()['mae'], 'a point metric must be present for every family'


def test_the_columns_are_SORTED(metrics_files, tmp_path):
    """Dict iteration order would otherwise make the column order depend on which family happened to be read first,
    and every re-run would produce a diff."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())
    assert list(_read(out).columns) == sorted(_read(out).columns)


def test_the_parent_directory_is_created(metrics_files, tmp_path):
    out = str(tmp_path / 'nested' / 'deeper' / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())
    assert os.path.exists(out)


# =====================================================================================================================
# The row labels — the correction to branch A
# =====================================================================================================================
def test_the_row_labels_restore_the_HYPHENS_fire_removed(metrics_files, tmp_path):
    """Fire turns ``--MC-Dropout`` into the kwargs key ``MC_Dropout``; the label must read as it was written in the
    YAML. Branch A's underscore-to-SPACE fallback would give "Deterministic UNet", which matches neither the config nor
    the label ``combine_curves`` puts in its legends."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, **metrics_files())

    labels = list(_read(out).index)
    assert sorted(labels) == ['Deterministic-UNet', 'Diffusion', 'MC-Dropout']
    assert not any('_' in label or ' ' in label for label in labels)


def test_the_row_labels_ROUND_TRIP_the_labels_the_SHIPPED_config_uses(repo_root):
    """Derived from ``config/eval/probabilistic_eval.yaml`` rather than from a fixture, so it cannot drift from the
    pipeline it describes: every family flag in the shipped config must survive Fire's substitution and come back
    IDENTICAL. This is what fails if a label is ever written with an underscore — ``Deterministic_UNet`` would come back
    hyphenated, and the table would disagree with the config that produced it.
    """
    from src.utils.io.parse_config import parse_config

    reserved = {'output-path', 'selected-metrics'}
    checked = 0
    for tier in ('', '_smoke_cpu'):
        config = parse_config(os.path.join(repo_root, f'config/eval/probabilistic_eval{tier}.yaml'))
        labels_per_stage = {}
        for stage in config['stages']:
            for name, parameters in stage.items():
                if name not in ('tabulate_metrics', 'combine_curves'):
                    continue
                labels = {key for key in parameters if key not in reserved}
                labels_per_stage[name] = labels
                for label in labels:
                    assert tabulate_stage._display(label.replace('-', '_')) == label, \
                        f'label "{label}" in {name} does not survive Fire\'s hyphen substitution'
                    checked += 1
        # both halves of one comparison must be ABOUT the same families: a family in the table but not in the figures
        # (or the reverse) is a comparison that quietly answers two different questions
        assert len(labels_per_stage) == 2, f'probabilistic_eval{tier}.yaml lost a comparison stage'
        assert labels_per_stage['tabulate_metrics'] == labels_per_stage['combine_curves'], \
            f'the table and the figures compare different families: {labels_per_stage}'
    assert checked >= 6, f'the shipped configs declared only {checked} family labels; the sweep found too little'


# =====================================================================================================================
# selected-metrics
# =====================================================================================================================
def test_selected_metrics_restricts_the_table_and_ignores_unknown_names(metrics_files, tmp_path):
    """Unknown names are skipped rather than fatal, which is what makes one selection list reusable across configs —
    and also why the ``ets_p99`` -> ``ets_h6`` re-key had to be applied to these fixtures rather than left to fail
    loudly: a stale name selects nothing and says nothing."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, selected_metrics=['crps', 'ets_h6', 'not_a_metric'],
                            MC_Dropout=metrics_files()['MC_Dropout'])

    table = _read(out)
    assert list(table.columns) == ['crps', 'ets_h6']
    assert 'mae' not in table.columns


def test_selected_metrics_accepts_a_comma_separated_string(metrics_files, tmp_path):
    """Because it arrives from a YAML scalar through Fire, not as a Python list."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, selected_metrics='ets_h6,crps', MC_Dropout=metrics_files()['MC_Dropout'])
    assert set(_read(out).columns) == {'crps', 'ets_h6'}


def test_selected_metrics_preserves_the_REQUESTED_order(metrics_files, tmp_path):
    """The one place the sorted-column rule yields: an explicit selection is a statement about priority, so the columns
    follow the request rather than the alphabet."""
    out = str(tmp_path / 'table.csv')
    tabulate_stage.tabulate(out, selected_metrics='mae,crps,ets_h6', MC_Dropout=metrics_files()['MC_Dropout'])
    assert list(_read(out).columns) == ['mae', 'crps', 'ets_h6']


# =====================================================================================================================
# Failure modes and the log
# =====================================================================================================================
def test_a_missing_family_is_skipped_but_ALL_missing_RAISES(metrics_files, tmp_path):
    """Skipping one absent family lets a partial comparison proceed while a family is still training; writing an empty
    table would be logged as a successful comparison of nothing."""
    paths = metrics_files()
    absent = str(tmp_path / 'absent.json')

    tabulate_stage.tabulate(str(tmp_path / 'partial.csv'), MC_Dropout=paths['MC_Dropout'], Diffusion=absent)
    assert list(_read(str(tmp_path / 'partial.csv')).index) == ['MC-Dropout']

    with pytest.raises(FileNotFoundError):
        tabulate_stage.tabulate(str(tmp_path / 'empty.csv'), MC_Dropout=absent)


def test_no_family_argument_at_all_RAISES(tmp_path):
    with pytest.raises(ValueError, match='at least one'):
        tabulate_stage.tabulate(str(tmp_path / 'table.csv'))


def test_the_metrics_a_family_LACKS_are_named_in_the_LOG(metrics_files, tmp_path, caplog):
    """The columns match by construction, so the CSV alone cannot show a disagreement. This log is the only place the
    asymmetry is stated in words — and the only way a family missing a POINT metric (a real merge failure, unlike a
    deterministic family missing ``crps``) is noticed without reading the table cell by cell."""
    import logging

    with caplog.at_level(logging.INFO, logger=tabulate_stage.logger.name):
        tabulate_stage.tabulate(str(tmp_path / 'table.csv'), **metrics_files())

    deterministic = [record.message for record in caplog.records if 'Deterministic-UNet' in record.message
                     and 'has no value' in record.message]
    assert deterministic, 'the deterministic family\'s missing ensemble scalars were not reported'
    assert 'crps' in deterministic[0] and 'spread_skill_ratio' in deterministic[0]


def test_a_family_lacking_NOTHING_is_not_reported(metrics_files, tmp_path, caplog):
    """Anti-vacuity for the test above: with one family there is nothing absent, so a log line would mean the stage is
    reporting the union of a single row against itself."""
    import logging

    with caplog.at_level(logging.INFO, logger=tabulate_stage.logger.name):
        tabulate_stage.tabulate(str(tmp_path / 'table.csv'), MC_Dropout=metrics_files()['MC_Dropout'])

    assert not [record for record in caplog.records if 'has no value' in record.message]
