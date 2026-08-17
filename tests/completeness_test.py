"""Meta-tests ABOUT the suite: the mirror is complete, and every function in ``src/`` has a unit test.

This is the file that makes "tests/ mirrors src/" and "every function is tested" enforceable rather than aspirational.
It mirrors no module, which is why it sits at the ``tests/`` root.

Two independent checks:

1. **Structural** — every non-``__init__`` module under ``src/`` has a ``<module>_test.py`` in the mirrored location,
   and no test file outlives the module it tests. Both directions matter: the second is how a test survives a deleted
   function and quietly stops meaning anything.
2. **Function coverage** — every function and non-dunder method defined in ``src/`` is referenced somewhere under
   ``tests/``. Name-reference is the FLOOR, not a measure of quality: it catches "never exercised", not "exercised but
   its branches never run". Line coverage (Block 5c, ``pytest-cov``) is the measure.

Dunder methods are deliberately not enumerated: ``Dataset.__getitem__`` and ``__len__`` are covered by indexing a
dataset and ``__init__`` by every construction, so requiring a direct reference would force contrived calls.
"""
import ast
import os

import pytest

SRC = 'src'
TESTS = 'tests'

# Functions that cannot be unit-tested without a live MLflow tracking server or a real DATA_ROOT. Each entry carries
# its reason, and `test_exemption_list_has_not_grown` PINS THE LENGTH — adding an exemption must be a deliberate,
# visible edit rather than the path of least resistance when a test turns out to be awkward to write.
#
# ⚠️ EMPTY, and that is the Block 5c result rather than an oversight: all 291 functions have a test. The awkward ones
# were reachable after all — `run.execute_stage` against a stubbed `mlflow.run` and tracking client, `tuning`'s store
# and restart paths against a real journal file. What CANNOT be reached this way is line coverage of `run_sequential`
# / `run_prefect` / `run_sweep` / `_fit_trial`, which need a tracking server and a real fit; that gap is measured by
# `--cov-fail-under` and closed by Step 4's end-to-end gate, not by an exemption here.
EXEMPT = {}
EXEMPT_COUNT = 0


def _iter_source_files(root):
    for directory, _, files in os.walk(root):
        if '__pycache__' in directory:
            continue
        for name in sorted(files):
            if name.endswith('.py'):
                yield os.path.join(directory, name)


def _defined_functions(path):
    """Qualified names of every function and non-dunder method defined at module or class level in one file."""
    tree = ast.parse(open(path).read())
    module = os.path.splitext(os.path.basename(path))[0]
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append((f'{module}.{node.name}', node.name))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not child.name.startswith('__'):
                    names.append((f'{module}.{node.name}.{child.name}', child.name))
    return names


def source_modules(repo_root):
    """``[(src path, expected test path)]`` for every non-``__init__`` module, plus the two ``__init__.py`` that carry
    real code (``plotting``'s ``show_plot_and_save``, ``stages``' seed application) mapped to ``init_test.py``."""
    pairs = []
    for path in _iter_source_files(os.path.join(repo_root, SRC)):
        relative = os.path.relpath(path, repo_root)
        directory, name = os.path.split(relative)
        mirrored_directory = os.path.join(repo_root, directory.replace(SRC, TESTS, 1))
        if name == '__init__.py':
            if os.path.getsize(path) == 0 or directory in (SRC, os.path.join(SRC, 'utils')):
                continue                         # empty, or a package marker with only root_path plumbing
            pairs.append((path, os.path.join(mirrored_directory, 'init_test.py')))
        else:
            stem = os.path.splitext(name)[0]
            pairs.append((path, os.path.join(mirrored_directory, f'{stem}_test.py')))
    return pairs


def referenced_names(repo_root):
    """Every bare name and attribute referenced anywhere under ``tests/``, plus every string literal.

    String literals count because a good few functions are reached by NAME rather than by call — the registries and
    builders dispatch on strings (``build_regression_loss({'name': 'wmse_psd'})``, ``MODULE_REGISTRY['diffusion']``),
    and a test that drives ``wmse_psd`` through its builder is testing it.
    """
    names = set()
    for path in _iter_source_files(os.path.join(repo_root, TESTS)):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names.add(node.value)
    return names


# =====================================================================================================================
# 1. The mirror is complete, in both directions
# =====================================================================================================================
def test_every_source_module_has_a_test_file(repo_root):
    missing = [os.path.relpath(test_path, repo_root)
               for _, test_path in source_modules(repo_root) if not os.path.exists(test_path)]
    assert not missing, f'{len(missing)} source module(s) have no mirrored test file: {sorted(missing)}'


def test_no_test_file_outlives_its_module(repo_root):
    """A test file whose module was deleted still passes while asserting nothing about the codebase."""
    expected = {test_path for _, test_path in source_modules(repo_root)}
    expected |= {os.path.join(repo_root, TESTS, name)
                 for name in ('completeness_test.py',)}                      # this file mirrors nothing
    # ⚠️ EMPTY since Block 4d, and deliberately kept as an empty statement rather than deleted: every stage module the
    # shipped configs name now exists, so the mirror covers all of them normally. Block 4e adds `pipeline_e2e_test.py`
    # to the set above (it mirrors no single module because it exercises all of them), not to this one.

    orphans = [os.path.relpath(path, repo_root)
               for path in _iter_source_files(os.path.join(repo_root, TESTS))
               if path.endswith('_test.py') and path not in expected]
    assert not orphans, f'test file(s) with no corresponding src module: {sorted(orphans)}'


def test_mirror_directory_structure_matches(repo_root):
    source_directories = {
        os.path.relpath(directory, os.path.join(repo_root, SRC))
        for directory, _, _ in os.walk(os.path.join(repo_root, SRC)) if '__pycache__' not in directory
    }
    test_directories = {
        os.path.relpath(directory, os.path.join(repo_root, TESTS))
        for directory, _, _ in os.walk(os.path.join(repo_root, TESTS)) if '__pycache__' not in directory
    }
    assert source_directories == test_directories, (
        f'only in src: {sorted(source_directories - test_directories)}; '
        f'only in tests: {sorted(test_directories - source_directories)}'
    )


# =====================================================================================================================
# 2. Every function in src has a unit test
# =====================================================================================================================
def test_every_source_function_is_referenced_by_a_test(repo_root):
    """A HARD GATE since Block 5c. It was ``xfail(strict=False)`` through 5a and 5b while the gap closed — note that
    ``strict=False`` is why the marker had to come off by hand: once the last function got a test the gate would have
    reported ``xpassed`` forever, staying green even as new untested functions arrived."""
    referenced = referenced_names(repo_root)
    total = 0
    missing = []
    for qualified, bare in _all_functions(repo_root):
        total += 1
        if qualified not in EXEMPT and bare not in referenced:
            missing.append(qualified)

    assert not missing, (
        f'{len(missing)} of {total} functions have no test:\n  ' + '\n  '.join(sorted(missing))
    )


def _all_functions(repo_root):
    for path, _ in source_modules(repo_root):
        yield from _defined_functions(path)


def test_function_census_is_stable(repo_root):
    """Pins the testable surface, so a module arriving without tests shows up here as well as in the coverage gate
    above — and so the count MOVES in the same commit as the code, making the diff a statement of what was added.

    History: 291 at the end of Step 3. Step 4 block 4a added `prepare_modeling`'s 14 functions and
    `data.high_lightning_days` (306); block 4b added `tune`'s 2 and `retrain_best`'s 2 (310); block 4c added
    `evaluate`'s 3 (313); block 4c-r added `setup.looks_like_an_unset_root` (314); block 4d added
    `tabulate_metrics`' 4 and `combine_curves`' 13 (331) — the last two stages, so every stage the shipped configs
    name is now implemented.
    """
    total = sum(1 for _ in _all_functions(repo_root))
    assert total == 331, f'the testable surface moved from 331 to {total}; re-scope the coverage work-list'


def test_exemption_list_has_not_grown(repo_root):
    assert len(EXEMPT) == EXEMPT_COUNT, (
        f'EXEMPT holds {len(EXEMPT)} entries but EXEMPT_COUNT is {EXEMPT_COUNT}. Exempting a function from the '
        f'every-function requirement must be a deliberate edit of BOTH, with a reason on the entry.'
    )
