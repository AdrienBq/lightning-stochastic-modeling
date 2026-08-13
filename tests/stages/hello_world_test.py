"""Tests for src/stages/hello_world.py — the plumber template's demo stage.

**Thin by design.** Eleven lines of template code that print a greeting; it exists so a fresh checkout can run
``python run_project.py config/hello_world.yaml`` end to end before any real stage exists. Nothing depends on it.

Its one genuine value is as the reference for the stage CONVENTION every real stage must follow — the
``from __init__ import root_path`` line before any ``src.`` import, and the ``Fire`` wrapper — so that is what is
asserted here rather than the greeting.
"""
import ast
import os

import pytest


def test_the_stage_runs(capsys):
    from hello_world import hello_world                       # bare name: see conftest.py

    hello_world()
    assert capsys.readouterr().out.strip(), 'the stage should print something'


def test_it_follows_the_stage_import_CONVENTION(repo_root):
    """CLAUDE.md: *"Each must begin with ``from __init__ import root_path`` before any ``src.`` import"* — because stages
    run from inside ``src/stages/``, and that line is also what triggers the automatic seeding hook.

    Checked here because this file is the template every real stage is copied from, so a convention broken here
    propagates.
    """
    path = os.path.join(repo_root, 'src/stages/hello_world.py')
    tree = ast.parse(open(path).read())

    import_lines = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    local_init = [i for i, node in enumerate(import_lines)
                  if isinstance(node, ast.ImportFrom) and node.module == '__init__']
    src_imports = [i for i, node in enumerate(import_lines)
                   if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('src')]

    assert local_init, 'a stage must import root_path from its local __init__'
    if src_imports:
        assert min(local_init) < min(src_imports), 'the __init__ import must come FIRST'


def test_it_is_wrapped_with_fire(repo_root):
    """Stages are standalone CLI scripts, so each needs the ``Fire`` entry point under ``__main__``."""
    source = open(os.path.join(repo_root, 'src/stages/hello_world.py')).read()
    assert 'Fire(' in source
    assert "__name__ == '__main__'" in source
