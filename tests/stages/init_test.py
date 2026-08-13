"""Tests for src/stages/__init__.py — the automatic per-stage seeding hook.

This is where CLAUDE.md's *"Seeding is automatic — the orchestrator exports ``PIPELINE_SEED`` and
``src/stages/__init__.py`` applies it. Do not re-seed globally inside a stage"* actually happens. It runs on PACKAGE
IMPORT, which is why every stage script must begin with ``from __init__ import root_path`` **before** any ``src.``
import: that line is what triggers the hook.

The hook cannot be re-triggered inside a test (the package is imported once), so the behaviour is tested by executing the
import in a SUBPROCESS with the environment set — which is also exactly how the orchestrator invokes a stage.
"""
import os
import subprocess
import sys
import textwrap

import pytest


def _run_stage_import(repo_root, env_extra, body='pass'):
    """Import the stages package in a fresh interpreter, the way a stage subprocess does."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        sys.path.insert(0, {os.path.join(repo_root, 'src/stages')!r})
        import random
        import __init__ as stages_package        # the line every stage script starts with
        {body}
    """)
    environment = {**os.environ, 'PYTHONPATH': repo_root, **env_extra}
    return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                          cwd=repo_root, env=environment)


def test_the_package_imports_without_a_pipeline_seed(repo_root):
    """The orchestrator's OWN import sees no ``PIPELINE_SEED`` — it is set only just before each stage subprocess — so the
    hook must be a no-op there rather than an error."""
    result = _run_stage_import(repo_root, {'PIPELINE_SEED': ''} if False else {})
    assert result.returncode == 0, result.stderr[-2000:]


def test_a_pipeline_seed_is_applied_on_import(repo_root):
    """The whole mechanism: setting the variable and importing the package must leave the RNGs seeded, with no stage code
    involved at all."""
    body = 'print("DRAW", random.random())'
    first = _run_stage_import(repo_root, {'PIPELINE_SEED': '4242'}, body)
    second = _run_stage_import(repo_root, {'PIPELINE_SEED': '4242'}, body)
    assert first.returncode == 0, first.stderr[-2000:]
    assert 'DRAW' in first.stdout
    assert first.stdout.strip().splitlines()[-1] == second.stdout.strip().splitlines()[-1]


def test_a_different_seed_changes_the_draw(repo_root):
    body = 'print("DRAW", random.random())'
    first = _run_stage_import(repo_root, {'PIPELINE_SEED': '1'}, body)
    second = _run_stage_import(repo_root, {'PIPELINE_SEED': '2'}, body)
    assert first.stdout.strip().splitlines()[-1] != second.stdout.strip().splitlines()[-1]


def test_an_unseeded_import_does_NOT_fix_the_draw(repo_root):
    """The converse, which is what makes the reproducibility test above mean something: without the variable the draw
    must vary between runs."""
    body = 'print("DRAW", random.random())'
    first = _run_stage_import(repo_root, {}, body)
    second = _run_stage_import(repo_root, {}, body)
    assert first.stdout.strip().splitlines()[-1] != second.stdout.strip().splitlines()[-1]


def test_a_malformed_seed_does_not_abort_the_stage(repo_root):
    """Seeding runs before the stage body, so an unparseable value must not be the reason a stage fails — it degrades to
    a warning and the stage proceeds UNSEEDED.

    The warning itself is not asserted: ``src/__init__.py`` calls ``logging.basicConfig(filename=...)``, so it lands in
    the repo's ``output.log`` rather than on stderr. What is asserted is the observable consequence — the body runs, and
    the RNG is genuinely not fixed, so the failure was not silently swallowed into a working seed.
    """
    body = 'print("BODY RAN"); print("DRAW", random.random())'
    first = _run_stage_import(repo_root, {'PIPELINE_SEED': 'not-an-int'}, body)
    second = _run_stage_import(repo_root, {'PIPELINE_SEED': 'not-an-int'}, body)

    assert first.returncode == 0, first.stderr[-2000:]
    assert 'BODY RAN' in first.stdout
    assert first.stdout.strip().splitlines()[-1] != second.stdout.strip().splitlines()[-1], \
        'a malformed seed must leave the RNG unseeded, not fall back to a fixed value'


def test_the_package_exposes_root_path(repo_root):
    """``from __init__ import root_path`` is the first line of every stage script, so the name has to be re-exported
    from the package even though it is defined in ``src/__init__.py``."""
    result = _run_stage_import(repo_root, {}, 'print("ROOT", stages_package.root_path)')
    assert result.returncode == 0, result.stderr[-2000:]
    assert repo_root in result.stdout
