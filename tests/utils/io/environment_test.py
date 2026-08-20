"""Tests for src/utils/io/environment.py — the three things the process bootstrap does to ``os.environ``.

All three are portability fixes, and all three have the same danger: they run once, on import, before anything reports
what they did. So the tests care less about the happy path than about the three ways each could be silently wrong —
overriding a variable the user set, growing ``PATH`` on repeated imports, and pointing at a bundle that is not there.

⚠️ Every test that touches ``os.environ`` uses ``monkeypatch``, which restores it per test. Without that, a leaked
``DATA_ROOT`` would reach `pipeline_e2e_test.py` and make a real pipeline read the wrong dataset.
"""
import os
import subprocess
import sys

import pytest

from src.utils.io.environment import (
    load_env_file,
    parse_env_file,
    prepend_interpreter_to_path,
    use_bundled_cartopy_data,
)


@pytest.fixture
def env_file(tmp_path):
    def build(contents, name='.env'):
        path = tmp_path / name
        path.write_text(contents)
        return str(path)
    return build


# =====================================================================================================================
# Parsing
# =====================================================================================================================
def test_a_plain_assignment_is_parsed():
    assert parse_env_file('DATA_ROOT=/data/era5\n') == {'DATA_ROOT': '/data/era5'}


def test_blank_lines_and_comments_are_skipped():
    text = '\n# a note\n\nDATA_ROOT=/data\n   \n#OUTPUT_ROOT=/nope\n'
    assert parse_env_file(text) == {'DATA_ROOT': '/data'}


def test_an_export_prefix_is_accepted():
    """So the same file can be `source`d from a shell — which is how every existing launch script sets these."""
    assert parse_env_file('export OUTPUT_ROOT=/scratch/out\n') == {'OUTPUT_ROOT': '/scratch/out'}


def test_surrounding_whitespace_is_stripped_from_both_sides():
    assert parse_env_file('  DATA_ROOT  =  /data/era5  \n') == {'DATA_ROOT': '/data/era5'}


def test_only_the_FIRST_equals_splits():
    """A value may contain `=` — `MLFLOW_TRACKING_URI` and any query-string-ish path do."""
    parsed = parse_env_file('MLFLOW_TRACKING_URI=file:/out/mlruns?a=b\n')
    assert parsed == {'MLFLOW_TRACKING_URI': 'file:/out/mlruns?a=b'}


def test_a_trailing_comment_is_stripped_from_an_unquoted_value():
    """The trap this closes: `.env.example` documents each variable with an inline comment, so a user who copies it and
    edits the value in place would otherwise get the comment as part of their path."""
    assert parse_env_file('DATA_ROOT=/data/era5   # the read-only dataset\n') == {'DATA_ROOT': '/data/era5'}


def test_a_hash_without_preceding_whitespace_is_KEPT():
    """Requiring the whitespace is what lets a path that legitimately contains `#` survive the comment strip."""
    assert parse_env_file('DATA_ROOT=/data/era5#2\n') == {'DATA_ROOT': '/data/era5#2'}


@pytest.mark.parametrize('quote', ['"', "'"])
def test_a_quoted_value_is_taken_verbatim(quote):
    parsed = parse_env_file(f'NOTE={quote}keep  spaces # and the hash{quote}\n')
    assert parsed == {'NOTE': 'keep  spaces # and the hash'}


def test_an_empty_value_is_the_empty_string():
    """Which `parse_config` then treats exactly like unset — it maps a missing variable to `''` too, so `DATA_ROOT=`
    and no `DATA_ROOT` line fail identically, inside the stage."""
    assert parse_env_file('DATA_ROOT=\n') == {'DATA_ROOT': ''}


def test_a_later_line_wins_over_an_earlier_one():
    assert parse_env_file('DATA_ROOT=/first\nDATA_ROOT=/second\n') == {'DATA_ROOT': '/second'}


def test_a_line_that_is_not_an_assignment_RAISES_naming_the_line():
    """Raising rather than skipping is the whole point: a `.env` typo that silently sets nothing reproduces the exact
    class of bug this file was added to remove, and would surface as a missing variable deep inside a stage."""
    with pytest.raises(ValueError) as error:
        parse_env_file('DATA_ROOT=/data\njust some words\n', source='/tmp/.env')
    message = str(error.value)
    assert '/tmp/.env:2' in message and 'just some words' in message


def test_an_invalid_variable_name_RAISES():
    with pytest.raises(ValueError, match='not a valid environment variable name'):
        parse_env_file('not-a-name=1\n')


# =====================================================================================================================
# Applying — the part that must not clobber
# =====================================================================================================================
def test_a_missing_file_is_not_an_error(tmp_path):
    """`.env` is optional. The pipeline must stay runnable from a shell that exports the variables itself, which is how
    every launch script under job_scripts/ works."""
    assert load_env_file(str(tmp_path / 'absent.env')) == {}


def test_the_file_sets_variables_that_are_not_already_set(env_file, monkeypatch):
    monkeypatch.delenv('LSM_TEST_ROOT', raising=False)
    applied = load_env_file(env_file('LSM_TEST_ROOT=/from/file\n'))
    assert applied == {'LSM_TEST_ROOT': '/from/file'}
    assert os.environ['LSM_TEST_ROOT'] == '/from/file'


def test_an_ALREADY_SET_variable_is_NEVER_overridden(env_file, monkeypatch):
    """The load-bearing rule. slurm hands a job the environment of the shell it was submitted from, and every launch
    script exports its own paths — a `.env` that won over those would silently retarget a running job's data root."""
    monkeypatch.setenv('LSM_TEST_ROOT', '/from/shell')
    applied = load_env_file(env_file('LSM_TEST_ROOT=/from/file\n'))
    assert applied == {}
    assert os.environ['LSM_TEST_ROOT'] == '/from/shell'


def test_the_RETURN_VALUE_reports_only_what_was_actually_set(env_file, monkeypatch):
    """So a caller can tell the user which paths came from the file rather than from their shell — the two are
    indistinguishable afterwards, and that ambiguity is exactly what makes a wrong path hard to find."""
    monkeypatch.setenv('LSM_TEST_KEPT', '/from/shell')
    monkeypatch.delenv('LSM_TEST_NEW', raising=False)
    applied = load_env_file(env_file('LSM_TEST_KEPT=/from/file\nLSM_TEST_NEW=/from/file\n'))
    assert applied == {'LSM_TEST_NEW': '/from/file'}


def test_an_empty_string_in_the_environment_still_counts_as_SET(env_file, monkeypatch):
    """`UPSTREAM_MODEL=` is meaningful — it is how a stochastic family says "no warm start". A file value must not
    resurrect it, or `.env` would turn a deliberate standalone run into a warm-started one."""
    monkeypatch.setenv('LSM_TEST_UPSTREAM', '')
    assert load_env_file(env_file('LSM_TEST_UPSTREAM=/some/checkpoint.ckpt\n')) == {}
    assert os.environ['LSM_TEST_UPSTREAM'] == ''


# =====================================================================================================================
# The interpreter — mlflow's bare `python`
# =====================================================================================================================
def test_the_interpreter_directory_is_PUT_FIRST_on_path(monkeypatch):
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    assert prepend_interpreter_to_path() == os.path.dirname(sys.executable)
    assert os.environ['PATH'].split(os.pathsep)[0] == os.path.dirname(sys.executable)


def test_the_rest_of_path_is_PRESERVED(monkeypatch):
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    prepend_interpreter_to_path()
    assert os.environ['PATH'].endswith('/usr/bin:/bin')


def test_a_SECOND_call_is_a_no_op(monkeypatch):
    """`src` is imported once per process, but a test session imports it alongside subprocess launches, and a PATH that
    grows by one entry per import would eventually stop being a plausible PATH at all."""
    monkeypatch.setenv('PATH', '/usr/bin')
    prepend_interpreter_to_path()
    after_first = os.environ['PATH']
    assert prepend_interpreter_to_path() is None
    assert os.environ['PATH'] == after_first


def test_an_empty_path_is_handled(monkeypatch):
    """No trailing separator, which would make an empty entry that some shells read as the current directory."""
    monkeypatch.setenv('PATH', '')
    prepend_interpreter_to_path()
    assert os.environ['PATH'] == os.path.dirname(sys.executable)


# =====================================================================================================================
# The bundled cartopy data
# =====================================================================================================================
def test_the_bundle_is_used_when_it_EXISTS(tmp_path, monkeypatch):
    monkeypatch.delenv('CARTOPY_DATA_DIR', raising=False)
    (tmp_path / 'data' / 'cartopy').mkdir(parents=True)
    expected = str(tmp_path / 'data' / 'cartopy')
    assert use_bundled_cartopy_data(str(tmp_path)) == expected
    assert os.environ['CARTOPY_DATA_DIR'] == expected


def test_an_ABSENT_bundle_leaves_the_variable_unset(tmp_path, monkeypatch):
    """Which is what keeps this inert in a clone that has not fetched the bundle. Pointing cartopy at a directory that
    does not exist would be harmless (it falls back to the download cache), but setting a variable to a path with
    nothing in it is a misleading thing for `preflight` to find."""
    monkeypatch.delenv('CARTOPY_DATA_DIR', raising=False)
    assert use_bundled_cartopy_data(str(tmp_path)) is None
    assert 'CARTOPY_DATA_DIR' not in os.environ


def test_an_EXPLICIT_cartopy_dir_wins(tmp_path, monkeypatch):
    monkeypatch.setenv('CARTOPY_DATA_DIR', '/my/own/cartopy')
    (tmp_path / 'data' / 'cartopy').mkdir(parents=True)
    assert use_bundled_cartopy_data(str(tmp_path)) is None
    assert os.environ['CARTOPY_DATA_DIR'] == '/my/own/cartopy'


# =====================================================================================================================
# The bootstrap actually applies them
# =====================================================================================================================
# ⚠️ In a SUBPROCESS, because `src` is already imported in this session: the bootstrap runs once per process, so an
# in-process assertion would only re-observe whatever the test session's own import did. This is the same reason
# `tests/stages/init_test.py` shells out.
_PROBE = """
import json, os, sys
import src
print(json.dumps({
    'path_first': os.environ['PATH'].split(os.pathsep)[0],
    'interpreter_dir': os.path.dirname(sys.executable),
    'env_from_file': os.environ.get('LSM_BOOTSTRAP_PROBE'),
    'src_handlers': len(__import__('logging').getLogger('src').handlers),
    'root_handlers': len(__import__('logging').getLogger().handlers),
}))
"""


def _bootstrap_probe(repo_root, env_extra=None):
    environment = {**os.environ, 'PYTHONPATH': repo_root, **(env_extra or {})}
    environment.pop('LSM_BOOTSTRAP_PROBE', None)
    result = subprocess.run([sys.executable, '-c', _PROBE], cwd=repo_root, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    import json
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_src_puts_the_running_interpreter_first_on_path(repo_root):
    """The fix for mlflow's hardcoded `"python"`. Without it, `/path/to/venv/bin/python run_project.py` gives every
    stage subprocess whatever `python` happens to be on PATH, and the stage dies on `import mlflow` — reported as a
    broken stage rather than a broken environment."""
    probe = _bootstrap_probe(repo_root)
    assert probe['path_first'] == probe['interpreter_dir']


def test_importing_src_attaches_ONE_console_handler_to_the_src_logger(repo_root):
    """Exactly one, and none on root. Two would double every library record; the root handler is what used to send
    them all to `output.log` instead of the console."""
    probe = _bootstrap_probe(repo_root)
    assert probe['src_handlers'] == 1
    assert probe['root_handlers'] == 0


def test_importing_src_does_NOT_create_an_output_log(repo_root, tmp_path):
    """The `basicConfig(filename=...)` sink is gone. It is asserted from a COPY of the bootstrap's directory layout
    rather than the repo, because the real `output.log` is a 457 KB legacy file that predates the change."""
    import shutil
    sandbox = tmp_path / 'checkout'
    (sandbox / 'src' / 'utils' / 'io').mkdir(parents=True)
    for name in ('__init__.py',):
        shutil.copy(os.path.join(repo_root, 'src', name), sandbox / 'src' / name)
    shutil.copy(os.path.join(repo_root, 'src', 'utils', '__init__.py'), sandbox / 'src' / 'utils' / '__init__.py')
    open(sandbox / 'src' / 'utils' / 'io' / '__init__.py', 'w').close()
    shutil.copy(os.path.join(repo_root, 'src', 'utils', 'io', 'environment.py'),
                sandbox / 'src' / 'utils' / 'io' / 'environment.py')

    result = subprocess.run([sys.executable, '-c', 'import src'], cwd=str(sandbox),
                            env={**os.environ, 'PYTHONPATH': str(sandbox)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (sandbox / 'output.log').exists(), 'the bootstrap still creates output.log'


def test_a_dot_env_in_the_repo_root_is_read_by_the_bootstrap(repo_root, tmp_path, monkeypatch):
    """End to end through the real file location, so the path `src/__init__.py` builds is asserted rather than assumed.
    Written to the REAL repo root because that is the path the bootstrap computes from `__file__`; removed after."""
    target = os.path.join(repo_root, '.env')
    if os.path.exists(target):
        pytest.skip('a real .env exists in the checkout; not overwriting it')
    try:
        with open(target, 'w') as handle:
            handle.write('LSM_BOOTSTRAP_PROBE=/read/from/dot/env\n')
        assert _bootstrap_probe(repo_root)['env_from_file'] == '/read/from/dot/env'
    finally:
        os.remove(target)
