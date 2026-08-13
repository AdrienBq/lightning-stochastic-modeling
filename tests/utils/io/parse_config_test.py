"""Tests for src/utils/io/parse_config.py — the ``{{$VAR}}`` environment substitution.

Nineteen lines of untouched template code, and one of the two documented silent-failure modes in the whole repo, with
zero tests until now. CLAUDE.md states it as a footgun:

    Env vars in config use ``{{$VAR}}``, not ``${VAR}``. ``parse_config`` substitutes textually *before* the YAML parse,
    and an **unset variable becomes the empty string** rather than an error — so quote interpolated scalars
    (``data-path: '{{$DATA_ROOT}}'``) and expect a missing variable to fail inside the stage, not at parse time.

Both halves matter and both are tested here. Textual-before-parse is what makes an unquoted interpolation able to change
the YAML's STRUCTURE rather than just a value, and empty-string-on-unset is why a missing ``DATA_ROOT`` surfaces as a
confusing error deep inside a stage instead of at startup.
"""
import os

import pytest

from src.utils.io.parse_config import parse_config


@pytest.fixture
def config_file(tmp_path):
    def build(contents, name='config.yaml'):
        path = tmp_path / name
        path.write_text(contents)
        return str(path)
    return build


# =====================================================================================================================
# Substitution
# =====================================================================================================================
def test_a_set_variable_is_substituted(config_file, monkeypatch):
    monkeypatch.setenv('LSM_TEST_ROOT', '/data/lightning')
    config = parse_config(config_file("data-path: '{{$LSM_TEST_ROOT}}'\n"))
    assert config['data-path'] == '/data/lightning'


def test_the_dollar_brace_form_is_NOT_substituted(config_file, monkeypatch):
    """``${VAR}`` is the shell form and this parser does not implement it. A config written that way silently keeps the
    literal text, so the stage receives ``${DATA_ROOT}`` as a path — which is why CLAUDE.md names the syntax explicitly."""
    monkeypatch.setenv('LSM_TEST_ROOT', '/data/lightning')
    config = parse_config(config_file("data-path: '${LSM_TEST_ROOT}'\n"))
    assert config['data-path'] == '${LSM_TEST_ROOT}'


def test_the_same_variable_is_substituted_at_every_occurrence(config_file, monkeypatch):
    monkeypatch.setenv('LSM_TEST_ROOT', '/data')
    config = parse_config(config_file(
        "a: '{{$LSM_TEST_ROOT}}/one'\nb: '{{$LSM_TEST_ROOT}}/two'\n"
    ))
    assert config == {'a': '/data/one', 'b': '/data/two'}


def test_several_variables_in_one_scalar(config_file, monkeypatch):
    monkeypatch.setenv('LSM_TEST_ROOT', '/data')
    monkeypatch.setenv('LSM_TEST_MODE', 'daily')
    config = parse_config(config_file("path: '{{$LSM_TEST_ROOT}}/prepared/{{$LSM_TEST_MODE}}'\n"))
    assert config['path'] == '/data/prepared/daily'


# =====================================================================================================================
# ⚠️ An UNSET variable becomes the EMPTY STRING
# =====================================================================================================================
def test_an_unset_variable_becomes_the_empty_string_rather_than_raising(config_file, monkeypatch):
    """The documented footgun. The parse SUCCEEDS, so the failure moves from config-load time to somewhere inside the
    stage — typically as a ``FileNotFoundError`` on a path that looks like it lost its prefix."""
    monkeypatch.delenv('LSM_TEST_ABSENT', raising=False)
    config = parse_config(config_file("data-path: '{{$LSM_TEST_ABSENT}}'\n"))
    assert config['data-path'] == ''


def test_an_unset_variable_leaves_a_plausible_looking_relative_path(config_file, monkeypatch):
    """Concretely what the user sees: ``'{{$DATA_ROOT}}/samples'`` with DATA_ROOT unset becomes ``'/samples'``, an
    absolute path to a directory that does not exist — not obviously a missing-variable problem."""
    monkeypatch.delenv('LSM_TEST_ABSENT', raising=False)
    config = parse_config(config_file("data-path: '{{$LSM_TEST_ABSENT}}/samples'\n"))
    assert config['data-path'] == '/samples'


def test_an_empty_variable_is_indistinguishable_from_an_unset_one(config_file, monkeypatch):
    """Both collapse to the same empty string, so a stage cannot tell "not configured" from "configured to nothing"."""
    monkeypatch.setenv('LSM_TEST_EMPTY', '')
    monkeypatch.delenv('LSM_TEST_ABSENT', raising=False)
    empty = parse_config(config_file("a: '{{$LSM_TEST_EMPTY}}'\n"))
    absent = parse_config(config_file("a: '{{$LSM_TEST_ABSENT}}'\n"))
    assert empty == absent == {'a': ''}


# =====================================================================================================================
# ⚠️ Substitution happens BEFORE the YAML parse, so it can change the STRUCTURE
# =====================================================================================================================
def test_an_UNQUOTED_unset_interpolation_turns_the_value_into_None(config_file, monkeypatch):
    """This is why CLAUDE.md says to QUOTE interpolated scalars. Unquoted, the substitution leaves ``data-path:`` with
    nothing after the colon, and YAML parses that as ``None`` rather than as an empty string — a different type reaching
    the stage."""
    monkeypatch.delenv('LSM_TEST_ABSENT', raising=False)
    config = parse_config(config_file('data-path: {{$LSM_TEST_ABSENT}}\n'))
    assert config['data-path'] is None


def test_a_quoted_unset_interpolation_stays_a_string(config_file, monkeypatch):
    """The recommended form, contrasted directly with the test above — the whole reason the quoting advice exists."""
    monkeypatch.delenv('LSM_TEST_ABSENT', raising=False)
    assert parse_config(config_file("data-path: '{{$LSM_TEST_ABSENT}}'\n"))['data-path'] == ''


def test_a_substituted_value_is_YAML_INTERPRETED_not_kept_as_text(config_file, monkeypatch):
    """The consequence of substituting before parsing: an unquoted numeric value becomes an int, and a value like
    ``true`` becomes a bool. A stage expecting a string gets neither."""
    monkeypatch.setenv('LSM_TEST_NUMBER', '4')
    monkeypatch.setenv('LSM_TEST_FLAG', 'true')
    config = parse_config(config_file('count: {{$LSM_TEST_NUMBER}}\nflag: {{$LSM_TEST_FLAG}}\n'))
    assert config['count'] == 4 and isinstance(config['count'], int)
    assert config['flag'] is True


def test_a_value_containing_a_colon_can_break_the_parse(config_file, monkeypatch):
    """The sharpest form of "textual before parse": a variable whose value contains YAML syntax is not escaped, so it can
    make a valid config invalid — or worse, valid but different."""
    monkeypatch.setenv('LSM_TEST_WEIRD', 'a: b')
    with pytest.raises(Exception):
        parse_config(config_file('value: {{$LSM_TEST_WEIRD}}\n'))


# =====================================================================================================================
# The shipped configs
# =====================================================================================================================
def test_every_shipped_config_parses_with_the_environment_unset(repo_root, monkeypatch):
    """All 19 of them, with ``DATA_ROOT`` and ``UPSTREAM_MODEL`` deliberately absent — which is the state a fresh
    checkout is in. They must PARSE (the failure belongs inside the stage), so this catches a genuine YAML error rather
    than a missing variable."""
    import glob

    monkeypatch.delenv('DATA_ROOT', raising=False)
    monkeypatch.delenv('UPSTREAM_MODEL', raising=False)

    configs = sorted(glob.glob(os.path.join(repo_root, 'config/**/*.yaml'), recursive=True))
    assert len(configs) == 19, f'expected 19 configs, found {len(configs)}'
    for path in configs:
        assert parse_config(path) is not None, path


def test_the_interpolated_data_paths_are_quoted_in_every_shipped_config(repo_root):
    """The advice, enforced. An unquoted ``{{$DATA_ROOT}}`` would become ``None`` with the variable unset (see above),
    so the stage would receive a null path instead of an empty one."""
    import glob
    import re

    offenders = []
    for path in sorted(glob.glob(os.path.join(repo_root, 'config/**/*.yaml'), recursive=True)):
        for number, line in enumerate(open(path), start=1):
            stripped = line.split('#', 1)[0]
            if '{{$' in stripped and not re.search(r"['\"][^'\"]*\{\{\$", stripped):
                offenders.append(f'{os.path.relpath(path, repo_root)}:{number}')
    assert not offenders, f'unquoted env interpolation: {offenders}'
