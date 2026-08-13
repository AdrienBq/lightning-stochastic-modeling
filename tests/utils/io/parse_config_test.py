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


# =====================================================================================================================
# Pipeline integrity across the shipped configs
#
# These read the parsed configs rather than parse_config's mechanics, but they belong to the module whose job is
# reading those files — and the shared `pipelines` fixture parses all 19 exactly once for the whole group.
# =====================================================================================================================
@pytest.fixture(scope='module')
def pipelines(repo_root):
    """``{relative path: parsed config}`` for all 19 shipped configs, parsed once."""
    import glob

    parsed = {}
    for path in sorted(glob.glob(os.path.join(repo_root, 'config/**/*.yaml'), recursive=True)):
        parsed[os.path.relpath(path, repo_root)] = parse_config(path)
    return parsed


def _stage_params(config):
    """``[(stage name, params)]`` for a parsed pipeline; the non-pipeline configs (search spaces, metrics) have none."""
    for stage in config.get('stages', []) or []:
        for name, params in stage.items():
            if isinstance(params, dict):
                yield name, params


def test_no_stage_carries_a_target_variable_parameter(pipelines):
    """``target-variable`` was the key that selected between unbounded counts and lightning-hours. ``mode`` is the only
    selector now, and the preparation stage REJECTS the old key outright — so one left behind fails the pipeline at
    run time rather than at parse time."""
    survivors = [(path, name) for path, config in pipelines.items()
                 for name, params in _stage_params(config) if 'target-variable' in params]
    assert not survivors, survivors


def test_the_informal_target_name_appears_nowhere_in_config(repo_root):
    """``daily_lightning_hours`` is the informal English name of ``mode: daily``, never a config value. It survives only
    as a deprecated alias in ``normalize_mode`` so old prepared artifacts keep loading."""
    import glob

    offenders = [os.path.relpath(path, repo_root)
                 for path in glob.glob(os.path.join(repo_root, 'config/**/*.yaml'), recursive=True)
                 if 'daily_lightning_hours' in open(path).read()]
    assert not offenders, offenders


def test_every_consumer_reads_the_leaf_its_own_pipeline_PRODUCES(pipelines):
    """The real risk of the prepared-leaf rename: a half-renamed file where ``prepare_*`` writes one leaf and ``tune``
    reads another. Both parse, so the pipeline either fails at run time or — worse — silently reads a stale directory
    left over from a previous run."""
    for path, config in pipelines.items():
        produced, consumed = None, []
        for name, params in _stage_params(config):
            if name.startswith('prepare_') and 'output-path' in params:
                produced = params['output-path']
            elif 'input-path' in params:
                consumed.append((name, params['input-path']))
        if produced is None:
            continue
        mismatched = [entry for entry in consumed if entry[1] != produced]
        assert not mismatched, f'{path} produces {produced}, consumes {mismatched}'


def test_the_cross_family_eval_configs_read_a_real_family_output(pipelines):
    """``probabilistic_eval*.yaml`` consume directories the FAMILY pipelines produce, so their input paths cannot be
    checked against a producer in their own file — a dangling one would only surface as a missing directory."""
    family_outputs = {params['output-path'] for path, config in pipelines.items() if '/eval/' not in path
                      for name, params in _stage_params(config)
                      if name.startswith('prepare_') and 'output-path' in params}
    assert family_outputs, 'no family prepare output found — the check would be vacuous'

    for path, config in pipelines.items():
        if 'probabilistic_eval' not in path:
            continue
        dangling = [params['input-path'] for _, params in _stage_params(config)
                    if 'input-path' in params and params['input-path'] not in family_outputs]
        assert not dangling, f'{path}: {dangling}'


def test_every_config_valued_parameter_points_at_a_file_that_EXISTS(repo_root, pipelines):
    """The corollary of the ``OUTPUT_PARAM_KEYS`` rule: a stale path silently degrades to a plain scalar in
    ``classify_params``, so the lazy cache stops invalidating on that input and a changed ``metrics.yaml`` no longer
    busts the cache. Output paths under ``outputs/`` are absent by design and are not checked."""
    missing = [(path, name, key, value) for path, config in pipelines.items()
               for name, params in _stage_params(config)
               for key, value in params.items()
               if isinstance(value, str) and value.startswith('config/')
               and not os.path.exists(os.path.join(repo_root, value))]
    assert not missing, missing


def test_there_are_config_valued_parameters_to_check(pipelines):
    """The anti-vacuity guard for the test above: if the ``config/`` prefix convention ever changed, that test would
    pass by finding nothing to look at."""
    referencing = [(path, name) for path, config in pipelines.items()
                   for name, params in _stage_params(config)
                   if any(isinstance(value, str) and value.startswith('config/') for value in params.values())]
    assert len(referencing) >= 9, len(referencing)


def test_every_prepare_stage_declares_the_daily_mode(pipelines):
    """One mode across all nine pipelines today. An hourly pipeline is a deliberate future addition, and this is what
    makes adding one a visible edit rather than a drift."""
    modes = {params.get('mode') for config in pipelines.values()
             for name, params in _stage_params(config) if name.startswith('prepare_')}
    assert modes == {'daily'}, modes


def test_all_nine_prepare_blocks_keep_their_required_keys(pipelines):
    """Nine pipelines, nine prepare blocks. ``hourly-threshold`` is in the required set because it is what drops
    single-stroke hours before the daily aggregation — omitting it changes the target rather than erroring."""
    prepare_keys = [set(params) for config in pipelines.values()
                    for name, params in _stage_params(config) if name.startswith('prepare_')]
    assert len(prepare_keys) == 9, len(prepare_keys)
    required = {'data-path', 'split-config', 'output-path', 'mode', 'features', 'hourly-threshold'}
    for keys in prepare_keys:
        assert required <= keys, sorted(required - keys)
