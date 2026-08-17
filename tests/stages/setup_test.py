"""Tests for src/stages/setup.py — the outputs-directory setup stage.

Untouched template code, and the one stage that ships with the repo. Thin, but not vacuous: ``make_dirs`` takes a
``hard_clean`` flag that calls ``shutil.rmtree``, and a destructive default would be a genuine hazard in a pipeline that
runs it before every experiment.
"""
import os

import pytest

# ⚠️ NOT `as setup_module`: pytest treats a module-level `setup_module` as its xunit-style module setup
# HOOK and CALLS it, which fails with `module 'setup' has no attribute '__code__'`.
import setup as setup_stage                                  # bare name: see conftest.py
from setup import make_dirs, setup


def test_it_creates_a_missing_directory(tmp_path):
    target = str(tmp_path / 'outputs' / 'nested')
    make_dirs(target)
    assert os.path.isdir(target)


def test_it_is_idempotent_on_an_existing_directory(tmp_path):
    target = str(tmp_path / 'outputs')
    make_dirs(target)
    marker = os.path.join(target, 'keep.txt')
    open(marker, 'w').close()

    make_dirs(target)
    assert os.path.exists(marker), 'the default must not touch existing contents'


def test_hard_clean_defaults_to_FALSE(tmp_path):
    """The important one. This stage runs at the start of a pipeline, so a destructive default would wipe a previous
    experiment's outputs — and the lazy cache would then re-run every stage without saying why."""
    import inspect

    assert inspect.signature(make_dirs).parameters['hard_clean'].default is False
    assert inspect.signature(setup).parameters['hard_clean'].default is False


def test_hard_clean_wipes_the_directory_when_asked(tmp_path):
    target = str(tmp_path / 'outputs')
    make_dirs(target)
    marker = os.path.join(target, 'gone.txt')
    open(marker, 'w').close()

    make_dirs(target, hard_clean=True)
    assert os.path.isdir(target)
    assert not os.path.exists(marker)


def test_setup_creates_every_path_it_is_given(tmp_path, monkeypatch):
    """Paths are joined onto ``root_path``, so the stage is pointed at a temporary root for the test."""
    import setup as setup_module

    monkeypatch.setattr(setup_module, 'root_path', str(tmp_path))
    setup_stage.setup(prepared='outputs/prepared', reports='outputs/reports')

    assert os.path.isdir(tmp_path / 'outputs' / 'prepared')
    assert os.path.isdir(tmp_path / 'outputs' / 'reports')


def test_setup_with_no_paths_is_a_no_op(tmp_path, monkeypatch):
    import setup as setup_module

    monkeypatch.setattr(setup_module, 'root_path', str(tmp_path))
    setup_stage.setup()
    assert not list(tmp_path.iterdir())


# =====================================================================================================================
# ⚠️ The unset-OUTPUT_ROOT guard  (Step 4 block 4c-r)
#
# The outputs moved off the source tree behind `{{$OUTPUT_ROOT}}`, and `parse_config` substitutes an UNSET variable to
# the EMPTY STRING rather than erroring. So `'{{$OUTPUT_ROOT}}/mc_dropout/prepared'` becomes `/mc_dropout/prepared` —
# absolute, at the filesystem root — and `os.path.join(root_path, ...)` discards the repo root entirely.
#
# This stage is where that is caught: it runs FIRST in every pipeline and already receives every directory. The check
# cannot live in `parse_config`, because `UPSTREAM_MODEL` RELIES on the empty-string behaviour (unset = no warm start,
# read that way by both `tune` and `prepare_modeling`), so a blanket raise on unset variables would break both
# stochastic families.
# =====================================================================================================================
@pytest.mark.parametrize('path', [
    '/mc_dropout/prepared',                                      # {{$OUTPUT_ROOT}} unset
    '/deterministic_unet/tuning',
    '/',                                                         # the variable was the WHOLE path
])
def test_a_path_at_the_FILESYSTEM_ROOT_is_recognised_as_an_unset_variable(path):
    assert setup_stage.looks_like_an_unset_root(path)


@pytest.mark.parametrize('path', [
    'outputs/prepared',                                          # relative: the pre-relocation form, still valid
    'outputs/mc_dropout/prepared/daily',
    '/tmp/lightning-outputs',                                    # absolute under an EXISTING mount: deliberate
])
def test_a_RELATIVE_or_genuinely_absolute_path_is_accepted(path):
    """The discriminator is the TOP-LEVEL segment, not absoluteness: a real absolute path sits under an existing mount
    (`/tmp` here, `/scratch` on a cluster), while an unset variable leaves a first segment that is a project or family
    name and does not exist. Rejecting all absolute paths would forbid the very thing the relocation is for."""
    assert not setup_stage.looks_like_an_unset_root(path)


def test_setup_REFUSES_to_create_a_directory_at_the_filesystem_root(tmp_path):
    """Without the guard the symptom is a bare ``PermissionError`` naming a directory nobody asked for — or, running
    privileged, a SUCCESSFUL ``makedirs`` at ``/`` that scatters the output tree across the root filesystem."""
    with pytest.raises(ValueError, match='EMPTY STRING'):
        setup_stage.setup(prepared='/mc_dropout/prepared', tuning='/mc_dropout/tuning')

    assert not os.path.isdir('/mc_dropout'), 'the guard must refuse BEFORE creating anything'


def test_the_error_names_EVERY_offending_key_and_the_variable_to_export():
    """All of them at once: they share one cause, and fixing them one run at a time would mean one failed launch per
    directory. The message names ``OUTPUT_ROOT`` because the symptom does not point at it."""
    with pytest.raises(ValueError) as raised:
        setup_stage.setup(prepared='/family/prepared', tuning='/family/tuning', reports='/family/reports')

    message = str(raised.value)
    for key in ('prepared', 'tuning', 'reports'):
        assert key in message, key
    assert 'OUTPUT_ROOT' in message
    assert 'export OUTPUT_ROOT' in message, 'the message should say what to DO'


def test_a_VALID_call_still_creates_the_directories(tmp_path):
    """The guard must not have made the stage's actual job conditional."""
    setup_stage.setup(prepared=str(tmp_path / 'prepared'), reports=str(tmp_path / 'reports'))
    assert os.path.isdir(str(tmp_path / 'prepared'))
    assert os.path.isdir(str(tmp_path / 'reports'))


def test_every_SHIPPED_pipeline_passes_setup_a_usable_tree(repo_root, monkeypatch):
    """End to end against the real configs: with ``OUTPUT_ROOT`` exported, no pipeline's ``setup`` block trips the
    guard — and with it unset, every one of them does. The second half is what proves the guard is reachable from a
    real config rather than only from a hand-written path."""
    import glob

    from src.utils.io.parse_config import parse_config

    pipelines = [path for path in sorted(glob.glob(os.path.join(repo_root, 'config/*/*.yaml')))
                 if 'stages' in parse_config(path)]
    assert pipelines, 'no pipeline configs found — the check would be vacuous'

    for path in pipelines:
        monkeypatch.setenv('OUTPUT_ROOT', '/tmp')
        block = next((params for stage in parse_config(path)['stages']
                      for name, params in stage.items() if name == 'setup'), None)
        if block is None:
            continue
        offending = [key for key, value in block.items()
                     if setup_stage.looks_like_an_unset_root(str(value))]
        assert not offending, f'{os.path.basename(path)}: {offending}'

        monkeypatch.delenv('OUTPUT_ROOT', raising=False)
        unset_block = next(params for stage in parse_config(path)['stages']
                           for name, params in stage.items() if name == 'setup')
        assert any(setup_stage.looks_like_an_unset_root(str(value)) for value in unset_block.values()), \
            f'{os.path.basename(path)}: an unset OUTPUT_ROOT should trip the guard'
