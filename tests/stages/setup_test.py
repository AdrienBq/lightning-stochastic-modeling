"""Tests for src/stages/setup.py — the outputs-directory setup stage.

Untouched template code, and the one stage that ships with the repo. Thin, but not vacuous: ``make_dirs`` takes a
``hard_clean`` flag that calls ``shutil.rmtree``, and a destructive default would be a genuine hazard in a pipeline that
runs it before every experiment.
"""
import os

import pytest

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
    setup_module.setup(prepared='outputs/prepared', reports='outputs/reports')

    assert os.path.isdir(tmp_path / 'outputs' / 'prepared')
    assert os.path.isdir(tmp_path / 'outputs' / 'reports')


def test_setup_with_no_paths_is_a_no_op(tmp_path, monkeypatch):
    import setup as setup_module

    monkeypatch.setattr(setup_module, 'root_path', str(tmp_path))
    setup_module.setup()
    assert not list(tmp_path.iterdir())
