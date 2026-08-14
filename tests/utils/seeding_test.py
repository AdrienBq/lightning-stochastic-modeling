"""Tests for src/utils/seeding.py — the baseline deterministic seeding every stage gets for free.

Untouched template code. It matters because CLAUDE.md leans on it: *"Seeding is automatic — the orchestrator exports
``PIPELINE_SEED`` and ``src/stages/__init__.py`` applies it. Do not re-seed globally inside a stage."* That instruction is
only safe if this function really does seed everything the stages use, so these tests check each library actually moves.

The design choice worth pinning is the ``importlib.util.find_spec`` guard: it seeds only libraries that are IMPORTABLE,
so a stage not depending on torch pays no import cost. The failure mode it must avoid is the opposite one — silently
skipping a library that IS present, which would leave that RNG unseeded and the stage irreproducible.
"""
import random

import numpy as np
import pytest

from src.utils.seeding import seed_everything


def test_it_reports_which_libraries_were_seeded():
    seeded = seed_everything(1234)
    assert 'random' in seeded, 'the stdlib RNG is always seeded'
    assert isinstance(seeded, list)


def test_every_library_present_in_this_environment_is_seeded():
    """The venv has numpy, torch and lightning, so all four must be reported. A silently skipped library is the failure
    this guards: the stage would still run and still look deterministic in its logs."""
    seeded = seed_everything(1234)
    assert set(seeded) >= {'random', 'numpy', 'torch', 'lightning'}, seeded


def test_the_stdlib_rng_is_actually_reseeded():
    seed_everything(7)
    first = [random.random() for _ in range(5)]
    seed_everything(7)
    assert [random.random() for _ in range(5)] == first


def test_numpys_legacy_global_rng_is_actually_reseeded():
    """The legacy global ``np.random`` matters because ``pandas.DataFrame.sample`` and older library code use it, not a
    local ``default_rng``."""
    seed_everything(7)
    first = np.random.random(5)
    seed_everything(7)
    assert np.array_equal(np.random.random(5), first)


def test_torchs_global_rng_is_actually_reseeded():
    import torch

    seed_everything(7)
    first = torch.randn(5)
    seed_everything(7)
    assert torch.equal(torch.randn(5), first)


def test_two_different_seeds_give_different_draws():
    """The other half: a function that seeded nothing would also pass every reproducibility check above."""
    seed_everything(1)
    first = np.random.random(5)
    seed_everything(2)
    assert not np.array_equal(np.random.random(5), first)


def test_a_seed_beyond_the_32_bit_range_is_reduced_rather_than_rejected():
    """numpy's legacy seed and lightning both require ``[0, 2**32 - 1]``, while the orchestrator derives its seed from a
    sha256 truncation that can exceed it. Raising would make some cache keys unusable."""
    seeded = seed_everything(2 ** 40 + 5)
    assert 'numpy' in seeded


def test_the_reduction_is_consistent_so_the_seed_stays_deterministic():
    seed_everything(2 ** 40 + 5)
    first = np.random.random(3)
    seed_everything(2 ** 40 + 5)
    assert np.array_equal(np.random.random(3), first)


def test_the_python_hash_seed_is_exported():
    """Only affects CHILD processes, and the dataloader workers are children — so a set-based operation in a worker is
    reproducible across runs."""
    import os

    seed_everything(42)
    assert os.environ['PYTHONHASHSEED'] == str(42 % (2 ** 32))


def test_a_float_or_string_seed_is_coerced():
    """The seed arrives from an environment variable, so it is a string at the boundary."""
    assert seed_everything('99')
    assert seed_everything(99.0)


def test_lightning_is_seeded_with_workers_enabled():
    """``workers=True`` is what extends the seeding to the dataloader worker processes. Without it, a run with
    ``num_workers > 0`` has unseeded workers, and the day-grouped shuffling in hourly mode would differ per run."""
    import inspect

    from src.utils import seeding

    assert 'workers=True' in inspect.getsource(seeding.seed_everything)


def test_the_availability_probe_answers_for_present_and_absent_modules():
    from src.utils.seeding import _available

    assert _available('numpy') is True
    assert _available('a_module_that_is_not_installed_anywhere') is False


def test_the_availability_probe_does_not_IMPORT_what_it_finds():
    """The whole reason it uses ``find_spec``: a numpy-only stage should not pay for a torch import just to be told torch
    exists. A probe that imported would make ``seed_everything`` several seconds slower on every stage."""
    import sys

    from src.utils.seeding import _available

    if 'wave' in sys.modules:                                    # a stdlib module nothing here imports
        pytest.skip('wave was already imported by another test, so this cannot be observed')
    assert _available('wave') is True
    assert 'wave' not in sys.modules, 'find_spec must locate the module without executing it'


def test_a_malformed_module_name_is_ABSENT_rather_than_an_exception():
    """``find_spec('')`` raises ``ValueError``, and seeding runs on package import before the stage body — so anything
    raising here aborts a stage for a reason unrelated to its work."""
    from src.utils.seeding import _available

    assert _available('') is False


def test_a_lightning_failure_cannot_crash_a_stage(monkeypatch):
    """Defensive by design: seeding runs on package import, before the stage body, so an exception here would abort a
    stage for a reason unrelated to its work. It degrades to a debug log instead."""
    import lightning

    def explode(*args, **kwargs):
        raise RuntimeError('simulated lightning failure')

    monkeypatch.setattr(lightning, 'seed_everything', explode)
    seeded = seed_everything(5)
    assert 'lightning' not in seeded
    assert 'random' in seeded and 'numpy' in seeded, 'the other libraries must still be seeded'
