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
    """All 23 of them, with ``DATA_ROOT`` and ``UPSTREAM_MODEL`` deliberately absent — which is the state a fresh
    checkout is in. They must PARSE (the failure belongs inside the stage), so this catches a genuine YAML error rather
    than a missing variable.

    19 through Step 4 block 4e; **23** after block 4f added the hourly pipeline — ``config/eval/metrics_hourly.yaml``,
    ``config/deterministic_unet/search_space_hourly.yaml`` and the two hourly tiers.
    """
    import glob

    monkeypatch.delenv('DATA_ROOT', raising=False)
    monkeypatch.delenv('UPSTREAM_MODEL', raising=False)

    configs = sorted(glob.glob(os.path.join(repo_root, 'config/**/*.yaml'), recursive=True))
    assert len(configs) == 23, f'expected 23 configs, found {len(configs)}'
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
# reading those files — and the shared `pipelines` fixture parses all 23 exactly once for the whole group.
# =====================================================================================================================
@pytest.fixture(scope='module')
def pipelines(repo_root):
    """``{relative path: parsed config}`` for all 23 shipped configs, parsed once."""
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
    ``classify_params``, so the lazy cache stops invalidating on that input and a changed ``metrics_daily.yaml`` no longer
    busts the cache. Paths under ``{{$OUTPUT_ROOT}}`` are absent by design and are not checked."""
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


def test_the_MODE_of_every_prepare_block_matches_its_FILE_NAME(pipelines):
    """``mode`` is the only key that selects between the two tasks, so a pipeline whose name says one thing and whose
    ``mode`` says the other is the single most expensive typo available here: it parses, it runs, and it trains a
    classifier on lightning-hours (or a bounded regressor on 0/1 labels) under a config whose metrics and search space
    were written for the other task.

    Through block 4e every pipeline was daily and this test asserted exactly that. Block 4f added the hourly tiers, so
    the assertion becomes the CORRESPONDENCE rather than a constant — and it is checked in both directions, which is
    what makes a mis-named copy of either pipeline fail here.
    """
    by_mode = {'daily': [], 'hourly': []}
    for path, config in pipelines.items():
        for name, params in _stage_params(config):
            if name.startswith('prepare_'):
                mode = params.get('mode')
                assert mode in by_mode, f'{path}: unknown mode {mode!r}'
                by_mode[mode].append(path)

    assert all('hourly' in path for path in by_mode['hourly']), by_mode['hourly']
    assert not any('hourly' in path for path in by_mode['daily']), by_mode['daily']
    assert len(by_mode['daily']) == 9 and len(by_mode['hourly']) == 2, \
        {mode: len(paths) for mode, paths in by_mode.items()}


def test_all_eleven_prepare_blocks_keep_their_required_keys(pipelines):
    """Eleven pipelines, eleven prepare blocks (nine daily + the two hourly tiers of block 4f). ``hourly-threshold`` is
    in the required set because it is what drops single-stroke hours — in daily mode before the aggregation, in hourly
    mode as the label itself — so omitting it changes the target rather than erroring.

    ``feature-aggregation`` is deliberately NOT required: it is daily-mode only (an hourly item is already one hour)
    and the hourly blocks omit it rather than carry a key the stage does not read.
    """
    prepare_keys = [set(params) for config in pipelines.values()
                    for name, params in _stage_params(config) if name.startswith('prepare_')]
    assert len(prepare_keys) == 11, len(prepare_keys)
    required = {'data-path', 'split-config', 'output-path', 'mode', 'features', 'hourly-threshold'}
    for keys in prepare_keys:
        assert required <= keys, sorted(required - keys)


# =====================================================================================================================
# Every written path is rooted at {{$OUTPUT_ROOT}}  (Step 4 block 4c-r)
#
# The outputs moved off the source tree because one daily prepared directory is ~20 GiB and a checkout cannot hold it.
# The change touched ~70 paths across 11 files, and the failure mode of such a sweep is a HALF-MOVED config: some paths
# under the new root, some still literal. That parses, runs, and reads a stale directory left by an earlier run.
# =====================================================================================================================
OUTPUT_KEYS = ('output-path', 'input-path', 'model-path', 'source-path', 'metrics-path', 'report-path')


def _output_root_paths(config):
    """Every stage parameter that names a place in the output tree, as ``(stage, key, value)``."""
    for name, params in _stage_params(config):
        if name == 'setup':                                      # setup's values are all output directories
            for key, value in params.items():
                if key != 'hard-clean':
                    yield name, key, value
            continue
        for key, value in params.items():
            if key in OUTPUT_KEYS or (name in ('tabulate_metrics', 'combine_curves') and key[0].isupper()):
                yield name, key, value


def test_NO_written_path_is_a_bare_relative_directory(repo_root, monkeypatch):
    """⭐ The regression guard for the whole relocation. Parsed with ``OUTPUT_ROOT`` set to a marker, every output-tree
    path must start with it — a path that does not is one the sweep missed, and it would be read or written inside the
    git checkout while its siblings went to the cluster's scratch space."""
    import glob

    from src.utils.io.parse_config import parse_config

    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    stragglers = []
    for path in sorted(glob.glob(os.path.join(repo_root, 'config/*/*.yaml'))):
        config = parse_config(path)
        if 'stages' not in config:
            continue
        for name, key, value in _output_root_paths(config):
            if isinstance(value, str) and not value.startswith('/MARKER'):
                stragglers.append(f'{os.path.basename(path)} :: {name}.{key} = {value}')
    assert not stragglers, 'paths not rooted at {{$OUTPUT_ROOT}}:\n  ' + '\n  '.join(stragglers)


def test_the_root_check_above_is_not_VACUOUS(repo_root, monkeypatch):
    """It would pass trivially if ``_output_root_paths`` yielded nothing — e.g. after a key rename. Twelve pipelines
    with six-odd written paths each means dozens; anything under ten means the collector stopped working."""
    import glob

    from src.utils.io.parse_config import parse_config

    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    collected = [entry for path in sorted(glob.glob(os.path.join(repo_root, 'config/*/*.yaml')))
                 for entry in _output_root_paths(parse_config(path))]
    assert len(collected) > 10, len(collected)


def test_an_UNSET_output_root_leaves_a_path_at_the_FILESYSTEM_ROOT(monkeypatch, tmp_path):
    """⚠️ The documented ``{{$VAR}}`` footgun, made executable. Substitution is textual and an unset variable becomes
    the EMPTY STRING rather than an error, so the path becomes absolute at ``/`` — and ``os.path.join(root_path, …)``
    then discards the repo root entirely. ``stages/setup.py`` is where that is caught; this pins WHY it has to be."""
    from src.utils.io.parse_config import parse_config

    monkeypatch.delenv('OUTPUT_ROOT', raising=False)
    probe = tmp_path / 'probe.yaml'
    probe.write_text("path: '{{$OUTPUT_ROOT}}/family/prepared/daily'\n")

    substituted = parse_config(str(probe))['path']
    assert substituted == '/family/prepared/daily'
    assert os.path.join('/some/repo/root', substituted) == '/family/prepared/daily', \
        'os.path.join discards everything before an absolute component — the reason this is not merely cosmetic'


# =====================================================================================================================
# The SHARED prepared directory  (Step 4 block 4c-r)
#
# The deterministic and MC-dropout families train on identical prepared data — the prepared directory is
# family-agnostic and their `prepare_modeling` blocks differed in `output-path` alone — so they now write ONE shared
# directory and the second pipeline to run skips preparation entirely. That saves ~20 GiB and a full pass per family.
#
# The invariant it creates: two pipelines writing one directory must ASK FOR THE SAME THING. They no longer can
# disagree silently, because `prepare_modeling` raises when `mode` or `hourly-threshold` disagree with the targets on
# disk — but the raise arrives at run time, after a preparation, and this catches it at test time instead.
# =====================================================================================================================
SHARED_PREPARED_FAMILIES = ('deterministic_unet', 'mc_dropout')


def _prepare_blocks(repo_root, tier=''):
    from src.utils.io.parse_config import parse_config

    blocks = {}
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        config = parse_config(os.path.join(repo_root, f'config/{family}/{family}_daily{tier}.yaml'))
        blocks[family] = next(params for stage in config['stages']
                              for name, params in stage.items() if name == 'prepare_modeling')
    return blocks


@pytest.mark.parametrize('tier', ['_smoke_cpu', '_smoke_gpu'])
def test_every_SMOKE_tier_sweeps_from_TRIAL_ZERO(tier, repo_root):
    """🐛 The hole the Step 4 block 4e gate fell into, closed.

    A smoke tier exists to prove every stage EXECUTES, and `lazy: false` is what stops the pipeline cache skipping one.
    But ``run_sweep`` keeps its own optuna store inside the tune stage's ``output-path`` and resumes from it, which
    `lazy` does not govern at all — so a second smoke run logged "keeping 2 recorded trial(s)", ran no trial, restored
    the previous best checkpoint and reported success. The gate passed while executing nothing, and the stale
    checkpoint it restored had been built by superseded code.

    `restart: true` is therefore not a preference in these tiers. A resume has no value at one or two trials, and its
    cost is a gate that cannot fail.
    """
    from src.utils.io.parse_config import parse_config

    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        config = parse_config(os.path.join(repo_root, f'config/{family}/{family}_daily{tier}.yaml'))
        tune_block = next(params for stage in config['stages']
                          for name, params in stage.items() if name == 'tune')
        assert tune_block.get('restart') is True, f'{family}{tier}: the sweep may resume and skip every trial'
        assert config['lazy'] is False, f'{family}{tier}: the pipeline cache may skip a stage'


def test_the_FULL_tiers_do_NOT_force_a_restart(repo_root):
    """The other side, so the rule above is not applied where it would hurt: a full sweep is 30-40 trials and hours
    long, and resuming an interrupted one is the feature. Only the smoke tiers, whose trials are worthless, discard it."""
    from src.utils.io.parse_config import parse_config

    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        config = parse_config(os.path.join(repo_root, f'config/{family}/{family}_daily.yaml'))
        tune_block = next(params for stage in config['stages']
                          for name, params in stage.items() if name == 'tune')
        assert 'restart' not in tune_block, f'{family}: a full sweep must be resumable after an interruption'


@pytest.mark.parametrize('tier', ['', '_smoke_cpu', '_smoke_gpu'])
def test_the_two_U_NET_families_prepare_into_ONE_shared_directory(tier, repo_root, monkeypatch):
    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    blocks = _prepare_blocks(repo_root, tier)

    paths = {family: blocks[family]['output-path'] for family in SHARED_PREPARED_FAMILIES}
    assert len(set(paths.values())) == 1, paths
    assert 'deterministic_and_mc_dropout' in next(iter(paths.values()))


@pytest.mark.parametrize('tier', ['', '_smoke_cpu', '_smoke_gpu'])
def test_families_sharing_a_directory_pass_IDENTICAL_prepare_parameters(tier, repo_root, monkeypatch):
    """⭐ The invariant the sharing creates. A parameter changed in one file and not the other means the second
    pipeline finds targets built under different settings. ``prepare_modeling`` raises on that rather than training on
    the wrong target — but only at run time, after somebody waited for a preparation. This is the same check, free."""
    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    blocks = _prepare_blocks(repo_root, tier)

    first, second = (blocks[family] for family in SHARED_PREPARED_FAMILIES)
    differing = {key: (first.get(key), second.get(key))
                 for key in set(first) | set(second) if first.get(key) != second.get(key)}
    assert not differing, f'{tier or "full"} tier: the shared prepare blocks disagree on {differing}'


@pytest.mark.parametrize('tier', ['', '_smoke_cpu', '_smoke_gpu'])
def test_DIFFUSION_keeps_its_OWN_prepared_directory(tier, repo_root, monkeypatch):
    """⚠️ Not an oversight. In residual mode diffusion's preparation writes ``upstream/<date>.npy`` maps and flips
    ``residual_target: true`` in ``prepared_config.json`` — and the DATASET keys the extra conditioning channel on
    that flag. Sharing would give the two U-net families a 6th input channel, and their checkpoints, built for
    ``5 x 24``, would mismatch on ``in_channels``.

    In FULL-target mode the directory would in fact be identical, which is exactly what makes sharing a trap rather
    than an optimisation: it would work until the first residual run."""
    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    blocks = _prepare_blocks(repo_root, tier)

    assert blocks['diffusion']['output-path'] != blocks['deterministic_unet']['output-path']
    assert 'upstream-model-path' in blocks['diffusion'], \
        'the key whose presence is the reason for the separation'
    for family in SHARED_PREPARED_FAMILIES:
        assert 'upstream-model-path' not in blocks[family], family


def test_the_MODEL_directories_stay_PER_FAMILY(repo_root, monkeypatch):
    """Only the prepared DATA is shared. ``tuning`` / ``best`` / ``evaluation`` / ``reports`` hold different models and
    different numbers, so a shared one would have the two families overwriting each other's ``best_model.ckpt``."""
    from src.utils.io.parse_config import parse_config

    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    per_family = {}
    for family in ('deterministic_unet', 'mc_dropout', 'diffusion'):
        config = parse_config(os.path.join(repo_root, f'config/{family}/{family}_daily.yaml'))
        block = next(params for stage in config['stages']
                     for name, params in stage.items() if name == 'setup')
        per_family[family] = {key: value for key, value in block.items() if key != 'prepared'}

    for key in ('tuning', 'best', 'evaluation', 'reports'):
        values = {family: paths[key] for family, paths in per_family.items()}
        assert len(set(values.values())) == 3, f'{key} is shared between families: {values}'


# =====================================================================================================================
# The HOURLY pipeline  (Step 4 block 4f)
#
# `mode: hourly` prepares a 0/1 occurrence target and the head emits a probability. Three files must move together —
# the metrics config, the search space and (implicitly) the prepared directory — and each of the three has a distinct,
# SILENT failure mode when it does not:
#
#   metrics-config   a daily suite puts the categorical cut at `> 0` on a probability field: POD ~ 1, FAR ~ the base
#                    rate, a full contingency table of nonsense. `run_metric_suite` warns; nothing raises.
#   model-config     a daily search space names `valid_regression_score`, which `selection_metric_for_mode` REJECTS —
#                    the one of the three that fails loudly, and only because it was made to.
#   prepared dir     `prepare_modeling` raises on a mode/threshold mismatch with the targets on disk, but only after a
#                    preparation has run.
#
# So the checks below are cheap versions of expensive discoveries. The loss-name check is the fourth of the same kind:
# nothing in the code rejects MAE on a binary target, it simply rewards the all-zero forecast.
# =====================================================================================================================
HOURLY_TIERS = ('config/deterministic_unet/deterministic_unet_hourly.yaml',
                'config/deterministic_unet/deterministic_unet_hourly_smoke_cpu.yaml')
HOURLY_METRICS = 'config/eval/metrics_hourly.yaml'
HOURLY_SEARCH_SPACE = 'config/deterministic_unet/search_space_hourly.yaml'


@pytest.fixture(scope='module')
def hourly_metrics(repo_root):
    import yaml

    with open(os.path.join(repo_root, HOURLY_METRICS)) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope='module')
def hourly_search_space(repo_root):
    import yaml

    with open(os.path.join(repo_root, HOURLY_SEARCH_SPACE)) as handle:
        return yaml.safe_load(handle)


def test_the_hourly_pipelines_swap_ALL_THREE_TASK_FILES(pipelines):
    """⭐ Both hourly tiers, every stage that takes one. A HALF-SWAPPED pipeline is the realistic failure — the metrics
    config is named on three stages and the search space on one, so forgetting a single line leaves `tune` selecting on
    the classification composite while `evaluate` scores the hour bands."""
    for path in HOURLY_TIERS:
        assert path in pipelines, path
        for name, params in _stage_params(pipelines[path]):
            if 'metrics-config' in params:
                assert params['metrics-config'] == HOURLY_METRICS, f'{path} :: {name}'
            if 'model-config' in params:
                assert params['model-config'] == HOURLY_SEARCH_SPACE, f'{path} :: {name}'


def test_the_hourly_pipelines_keep_the_SHARED_split(pipelines):
    """The split does NOT move with the task. Every pipeline partitions the same days by the same years — hourly items
    are those days expanded 24-fold, so a separate hourly split would break comparability with the daily families for
    no gain."""
    for path in HOURLY_TIERS:
        splits = {params['split-config'] for _, params in _stage_params(pipelines[path])
                  if 'split-config' in params}
        assert splits <= {'config/split/split.yaml', 'config/split/split_smoke_cpu.yaml'}, f'{path}: {splits}'


def test_NO_daily_pipeline_REACHES_FOR_an_hourly_task_file(pipelines):
    """The other direction, and the likelier mistake once these files exist: a copy-paste that leaves a daily pipeline
    pointing at ``metrics_hourly.yaml``. It would parse, run, and score a 0-24 regression against a 0.5 probability
    cut — every categorical score computed at ``pred >= 0.5 hours``."""
    offenders = [(path, name) for path, config in pipelines.items() if 'hourly' not in path
                 for name, params in _stage_params(config)
                 if HOURLY_METRICS in params.values() or HOURLY_SEARCH_SPACE in params.values()]
    assert not offenders, offenders


def test_the_hourly_metrics_config_cuts_the_PROBABILITY_not_the_occurrence(hourly_metrics):
    """⭐ THE load-bearing property of the hourly suite, and the one with no runtime guard that raises.

    ``kind: occurrence`` cuts BOTH sides at ``> 0``. On a probability field that marks every cell carrying any non-zero
    probability as a predicted event, so the contingency table degenerates to hits + false alarms with no misses and no
    correct negatives. ``kind: probability`` puts the decision threshold on the prediction and leaves the 0/1 labels
    alone. This is the same class of bug that made the DAILY occurrence table read "lightning everywhere" until
    ``pred_value: 1`` was added there — found by a human reading a confusion matrix, which is why it is pinned here.
    """
    thresholds = hourly_metrics['thresholds']
    categorical = hourly_metrics['metrics']['categorical']['thresholds']
    assert categorical, 'the categorical group has no threshold at all'
    for name in categorical:
        assert thresholds[name]['kind'] == 'probability', f'{name}: {thresholds[name]}'
        assert 0.0 < float(thresholds[name]['value']) < 1.0, f'{name}: not a probability cut'


def test_the_hourly_metrics_config_HAS_NO_HOUR_BANDS(hourly_metrics):
    """h3/h6/h12 are defined on a 0-24 daily count. On a 0/1 target ``>= 3 hours`` is an empty event, so every score at
    those bands would be NaN — reported, and indistinguishable from a model that failed to learn.

    ⚠️ This asserts the INTENT — no `absolute` hour band survives — rather than pinning the exact threshold set. It
    used to require exactly ``{occurrence, p50}``, which made it fail when the decision ladder grew to p10..p50: it was
    guarding the *number* of entries while claiming to guard their *kind*. A test that has to be edited whenever a
    level is added is not pinning the property its name states.
    """
    thresholds = hourly_metrics['thresholds']
    bands = {name: spec for name, spec in thresholds.items() if spec.get('kind') == 'absolute'}
    assert not bands, f'hour bands survive in the hourly suite: {bands}'
    assert set(thresholds) - {'occurrence'}, 'no decision cut at all'
    assert all(name.startswith('p') for name in set(thresholds) - {'occurrence'}), sorted(thresholds)


def test_the_hourly_OCCURRENCE_event_carries_NO_prediction_side_cut(hourly_metrics):
    """The mirror of the daily file's ``pred_value: 1``, and it is absent for a reason rather than by omission: nothing
    in the hourly suite cuts the PREDICTION at the occurrence level (that is ``p50``'s job), so ``occurrence`` is used
    only where a score conditions on the OBSERVED event. A ``pred_value`` here would be dead configuration at best, and
    at worst ``1`` — ``p >= 1``, which essentially never fires."""
    assert hourly_metrics['thresholds']['occurrence'] == {'kind': 'occurrence'}


def test_the_hourly_metrics_config_DROPS_the_groups_that_degenerate(hourly_metrics):
    """Three entries whose bins or conditions collapse on a binary observation, dropped rather than left to report
    constants: ``mae_stratified`` (its bins partition observed intensity, which has one non-empty band),
    ``estimation_tendency`` (``pred - obs = p - 1 <= 0`` always, so under ~ 1 and over = 0 for every model) and
    ``rank_correlation`` (Spearman against a dichotomous observation is a rescaled ``roc_auc``, and its obs > 0
    subgroups are constant)."""
    continuous = hourly_metrics['metrics']['continuous']
    assert 'mae_stratified' not in continuous
    assert 'estimation_tendency' not in continuous
    assert 'rank_correlation' not in hourly_metrics['metrics']['calibration']
    assert 'error_by_intensity_bin' not in hourly_metrics['reporting']['figures'], \
        'the figure reads curves["error_by_bin"], which this suite no longer populates'


def test_the_hourly_metrics_config_KEEPS_the_four_keys_the_task_EXISTS_FOR(hourly_metrics):
    """⭐ The positive half of the trim, and the reason this pipeline was built. All four need a calibrated probability
    and are structurally NaN or absent in every daily run, because the one-head design gives a daily model no
    probabilistic output."""
    skill = hourly_metrics['metrics']['skill']
    assert 'brier_skill_score' in skill and 'explained_deviance' in skill
    assert 'reliability_diagram' in hourly_metrics['metrics']['calibration']
    assert 'reliability' in hourly_metrics['reporting']['figures']
    assert 'dice' in hourly_metrics['metrics']['categorical']['scores'], \
        'soft dice on the probability is the eval-time complement of the dice training loss'


def test_the_hourly_metrics_config_KEEPS_the_ensemble_group(hourly_metrics):
    """Kept although the only hourly pipeline that ships is deterministic, for which the whole group is silently
    skipped. This is the hourly suite for ALL families, exactly as metrics_daily.yaml is the daily one — an hourly mc_dropout
    pipeline must not need a fourth metrics config to get its ensemble scores."""
    assert set(hourly_metrics['metrics']['ensemble']) == \
        {'crps', 'almost_fair_crps', 'spread_skill_ratio', 'rank_histogram'}


def test_the_hourly_search_space_names_NO_IMPROPER_loss(hourly_search_space):
    """⭐ MAE IS IMPROPER against a 0/1 observation: minimized by a sharp forecast, so at this base rate the all-zero
    prediction beats a calibrated one. A sweep offered ``weighted_mae`` / ``wmae_psd`` would be rewarded for the
    degenerate solution — while ``valid_classification_score``, which contains no mae term, disagreed about which trial
    won. Nothing in the code rejects the combination; this test is the rejection.

    ``asymmetric_huber`` goes for a different reason (it encodes conservativeness about a MAGNITUDE, which a
    probability does not have) and ``crps_binary`` for a third: it needs a real ensemble, and this family's single
    forward pass gives N = 1 with a spread term identically zero. The module raises on that one.
    """
    choices = hourly_search_space['loss']['name']['choices']
    for improper in ('weighted_mae', 'wmae_psd', 'asymmetric_huber', 'crps_binary'):
        assert improper not in choices, f'{improper} must not be reachable on a binary target: {choices}'
    assert choices, 'the loss space is empty'


def test_the_hourly_search_space_supplies_EVERY_key_its_losses_READ(hourly_search_space):
    """``build_binary_loss`` reads ``positive_class_weight`` and ``focal_gamma`` for ``focal_bce`` with ``[]``, not
    ``.get`` — a missing key is a KeyError inside a trial, i.e. after the sweep has started. Both are unreachable in
    the three daily spaces, so this file is the first to owe them."""
    loss = hourly_search_space['loss']
    if 'focal_bce' in loss['name']['choices']:
        assert 'positive_class_weight' in loss and 'focal_gamma' in loss
    if 'dice' in loss['name']['choices']:
        assert 'dice_smooth' in loss
    if any(name.endswith('_psd') for name in loss['name']['choices']):
        assert 'alpha' in loss


def test_the_hourly_search_space_OMITS_the_daily_only_blocks(hourly_search_space):
    """``output_activation`` and ``max_hours`` are unread in hourly mode (``_head_output`` is a pass-through and the
    sigmoid lives on the prediction path), and ``calibration.regression`` can only ever be a no-op there
    (``regression_calibration_enabled`` is ``(not hourly) and ...``). Carrying them would read as a configured choice
    with no effect, which is worse than their absence."""
    assert 'output_activation' not in hourly_search_space
    assert 'max_hours' not in hourly_search_space
    assert 'regression' not in hourly_search_space['calibration']
    assert 'occurrence' in hourly_search_space['calibration'], \
        'Platt scaling is the one calibrator this task CAN fit, and the first shipped pipeline that can'


def test_the_hourly_pipeline_keeps_its_OWN_prepared_directory(repo_root, monkeypatch):
    """It cannot share one, and the reason is stronger than diffusion's: the targets themselves differ in shape and
    dtype (``[T, H, W]`` uint8 0/1 against ``[H, W]`` float32 0-24) and the features are laid out time-major.
    ``prepare_modeling`` raises on the mode mismatch, so the failure is loud — but only after a preparation."""
    from src.utils.io.parse_config import parse_config

    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    daily = _prepare_blocks(repo_root)
    for path in HOURLY_TIERS:
        block = next(params for stage in parse_config(os.path.join(repo_root, path))['stages']
                     for name, params in stage.items() if name == 'prepare_modeling')
        assert block['output-path'].endswith('/hourly'), block['output-path']
        assert block['output-path'] not in {daily[family]['output-path'] for family in daily}


def test_the_hourly_SMOKE_tier_sweeps_from_TRIAL_ZERO(repo_root):
    """The same hole the block 4e gate fell into, closed for the fourth smoke tier: ``run_sweep`` keeps its own optuna
    store inside the tune stage's ``output-path`` and resumes from it, which the pipeline's ``lazy: false`` does not
    govern at all. Without ``restart: true`` a second run of this tier would execute no trial and restore a stale
    checkpoint, and the gate would pass having done nothing."""
    from src.utils.io.parse_config import parse_config

    config = parse_config(os.path.join(repo_root, HOURLY_TIERS[1]))
    tune_block = next(params for stage in config['stages'] for name, params in stage.items() if name == 'tune')
    assert tune_block.get('restart') is True
    assert config['lazy'] is False


def test_the_FULL_hourly_tier_does_NOT_force_a_restart(repo_root):
    """The other side of the rule: a full hourly sweep is 20 trials over 87 648 items per epoch, and resuming an
    interrupted one is the feature."""
    from src.utils.io.parse_config import parse_config

    config = parse_config(os.path.join(repo_root, HOURLY_TIERS[0]))
    tune_block = next(params for stage in config['stages'] for name, params in stage.items() if name == 'tune')
    assert 'restart' not in tune_block


def test_the_hourly_tiers_do_NOT_materialize_features(repo_root, monkeypatch):
    """⚠️ The one place the hourly pipeline deliberately diverges from every other shipped config, and it is load-bearing
    twice over: materializing would cost a SECOND ~20 GiB (the daily directory already holds every hour, laid out
    variable-major), and `materialize-features: false` is what turns on ``DayGroupedShuffleSampler``, which
    ``tuning.py`` builds when ``mode == hourly and not uses_materialized_features``. That sampler has never run under
    any shipped config.

    Both tiers, deliberately: a smoke tier that materialized would smoke a loader path the real run does not use.
    """
    from src.utils.io.parse_config import parse_config

    monkeypatch.setenv('OUTPUT_ROOT', '/MARKER')
    for path in HOURLY_TIERS:
        block = next(params for stage in parse_config(os.path.join(repo_root, path))['stages']
                     for name, params in stage.items() if name == 'prepare_modeling')
        assert block['materialize-features'] is False, path
        assert 'feature-dtype' not in block, \
            f'{path}: feature-dtype is the storage dtype of files this tier does not write'
        assert 'feature-aggregation' not in block, \
            f'{path}: feature-aggregation is daily-mode only -- an hourly item is already one hour'
