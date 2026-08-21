"""Tests for src/utils/io/environment.py — what the process bootstrap does to ``os.environ`` — plus the install surface
around it: the requirements/conda files, the committed cartopy bundle, and ``scripts/preflight.py``.

The bootstrap helpers share one danger: they run once, on import, before anything reports what they did. So the tests
care less about the happy path than about the ways each could be silently wrong — overriding a variable the user set,
growing ``PATH`` on repeated imports, pointing at a bundle that is not there.

The install-surface tests live here rather than in a file of their own because ``tests/`` mirrors ``src/`` one file per
module (``completeness_test.py`` enforces it in both directions), and a new root-level file would be flagged as an
orphan. They belong with this module by theme: every one of them asks "can this machine run the code".

⚠️ Every test that touches ``os.environ`` uses ``monkeypatch``, which restores it per test. Without that, a leaked
``DATA_ROOT`` would reach `pipeline_e2e_test.py` and make a real pipeline read the wrong dataset.
"""
import os
import subprocess
import sys

import pytest

from src.utils.io.environment import (
    is_git_lfs_pointer,
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


def test_an_ALREADY_IMPORTED_cartopy_has_its_config_REPAIRED(tmp_path, monkeypatch):
    """cartopy reads ``CARTOPY_DATA_DIR`` ONCE, at import, into ``config['pre_existing_data_dir']`` — so setting the
    variable afterwards is silently ineffective. Normal import order is safe (``src.utils.plotting.maps`` cannot load
    without this package's ``__init__`` running first), but a script that imports cartopy first would fall through to
    the download cache with no sign of why. Stubbed through ``sys.modules`` so the assertion is about the mechanism and
    not about cartopy being installed."""
    import sys
    import types

    stub = types.ModuleType('cartopy')
    stub.config = {'pre_existing_data_dir': '.'}
    monkeypatch.setitem(sys.modules, 'cartopy', stub)
    monkeypatch.delenv('CARTOPY_DATA_DIR', raising=False)
    (tmp_path / 'data' / 'cartopy').mkdir(parents=True)

    use_bundled_cartopy_data(str(tmp_path))
    assert stub.config['pre_existing_data_dir'] == str(tmp_path / 'data' / 'cartopy')


# =====================================================================================================================
# The git-lfs pointer check — and the bundle in THIS checkout
# =====================================================================================================================
_POINTER = (b'version https://git-lfs.github.com/spec/v1\n'
            b'oid sha256:797d675af9613f80b51ab6049fa32e589974d7a97c6497ca56772965f179ed26\nsize 1046728\n')


def test_a_pointer_file_is_RECOGNISED(tmp_path):
    path = tmp_path / 'ne_50m_coastline.shp'
    path.write_bytes(_POINTER)
    assert is_git_lfs_pointer(str(path))


def test_a_real_binary_is_NOT_a_pointer(tmp_path):
    """A shapefile starts with the big-endian magic 9994, nothing like the pointer text."""
    path = tmp_path / 'ne_50m_coastline.shp'
    path.write_bytes(b'\x00\x00\x27\x0a' + b'\x00' * 200)
    assert not is_git_lfs_pointer(str(path))


def test_a_file_SHORTER_than_the_magic_is_not_a_pointer(tmp_path):
    """The read is capped at 42 bytes, so a shorter file must not raise or over-read."""
    path = tmp_path / 'tiny'
    path.write_bytes(b'version')
    assert not is_git_lfs_pointer(str(path))


@pytest.mark.parametrize('kind', ['missing', 'directory'])
def test_an_unreadable_path_is_reported_as_NOT_a_pointer(tmp_path, kind):
    """Those are different problems with their own messages; conflating them here would mislabel a missing bundle as an
    un-pulled one and send the user to `git lfs pull` for nothing."""
    target = tmp_path / 'absent' if kind == 'missing' else tmp_path
    assert not is_git_lfs_pointer(str(target))


def test_THIS_checkout_has_the_cartopy_bundle_and_it_is_NOT_a_pointer(repo_root):
    """A checkout-integrity guard, not a unit test. Every map figure needs this one file, and a broken copy of it fails
    deep inside a shapefile reader — so asserting it here means `pytest` names the cause instead of a pipeline run
    naming a cartopy internal at the end of an hour's work. See data/cartopy/README.md.
    """
    bundle = os.path.join(repo_root, 'data', 'cartopy', 'shapefiles', 'natural_earth', 'physical')
    shapefile = os.path.join(bundle, 'ne_50m_coastline.shp')

    assert os.path.isfile(shapefile), f'the bundled coastline is missing: {shapefile}'
    assert not is_git_lfs_pointer(shapefile), (
        'data/cartopy/.../ne_50m_coastline.shp is a git-lfs POINTER, not the shapefile. The bundle is committed as '
        'ORDINARY blobs, so this means the `data/cartopy/shapefiles/** -filter` exemption was dropped from '
        '.gitattributes and '
        'something with git-lfs installed re-pointerised it. Restore the exemption (see the test below).'
    )
    # Reading a shapefile needs .shx and .dbf as well as .shp; .prj carries the CRS and .cpg the encoding.
    for extension in ('shx', 'dbf', 'prj', 'cpg'):
        assert os.path.isfile(os.path.join(bundle, f'ne_50m_coastline.{extension}')), extension


def test_the_INSTALL_files_carry_no_machine_specific_path(repo_root):
    """The whole point of Step 5. The requirements header used to open with
    `python -m venv /homedata/aburq/.venvs/lightning-stochastic-modeling` and `module load python/meso-3.11`, which a
    new user on a different cluster has to notice is not for them — install instructions are read as instructions.
    """
    offenders = []
    for name in ('minimal_requirements.txt', 'environment.yml', '.env.example'):
        text = open(os.path.join(repo_root, name)).read()
        for number, line in enumerate(text.splitlines(), start=1):
            if '/homedata' in line or '/home/aburq' in line or 'module load python' in line:
                offenders.append(f'{name}:{number}: {line.strip()}')
    assert not offenders, f'machine-specific paths in the install surface: {offenders}'


def test_the_conda_recipe_DEFERS_to_the_requirements_file(repo_root):
    """One dependency list, not two. Two enumerations agree the day they are written and diverge on the first version
    bump, after which a conda user and a pip user run different code — and the difference shows up as an
    unreproducible RESULT rather than an install error, which is the expensive kind.
    """
    from yaml import safe_load

    recipe = safe_load(open(os.path.join(repo_root, 'environment.yml')))
    dependencies = recipe['dependencies']
    pip_sections = [entry['pip'] for entry in dependencies if isinstance(entry, dict) and 'pip' in entry]

    assert pip_sections, 'environment.yml has no pip: section, so it cannot defer to minimal_requirements.txt'
    assert any('-r minimal_requirements.txt' in entry for section in pip_sections for entry in section), \
        'environment.yml must install `-r minimal_requirements.txt` rather than listing packages again'

    # Everything conda itself installs must be the interpreter and pip, nothing that the requirements file also names.
    conda_packages = {entry.split('=')[0].strip() for entry in dependencies if isinstance(entry, str)}
    assert conda_packages == {'python', 'pip'}, (
        f'environment.yml installs {sorted(conda_packages - {"python", "pip"})} through conda as well as pip through '
        f'minimal_requirements.txt — that is the second list this file exists to avoid'
    )


def test_the_requirements_file_does_NOT_pin_the_torch_BUILD(repo_root):
    """A regression guard on a measured, silent defect. `minimal_requirements.txt` used to carry
    `--extra-index-url https://download.pytorch.org/whl/cpu`, and a local version identifier sorts ABOVE the plain one
    (`Version('2.8.0+cpu') > Version('2.8.0')`), so pip preferred the CPU build on EVERY machine — including a GPU node,
    where `torch.cuda.is_available()` is then False and `accelerator: auto` silently trains on CPU. Nothing raises; the
    run is merely slow, which is why it survived. The CPU-only install is documented as an explicit opt-in instead.
    """
    lines = open(os.path.join(repo_root, 'minimal_requirements.txt')).read().splitlines()
    active = [line for line in lines if line.strip() and not line.lstrip().startswith('#')]
    offenders = [line for line in active if '--extra-index-url' in line or '--index-url' in line]
    assert not offenders, (
        f'an index override is active in minimal_requirements.txt: {offenders}. That silently changes which torch '
        f'build every machine gets; keep it in a comment as an opt-in.'
    )


def test_PREFLIGHT_passes_against_a_synthetic_dataset_root(repo_root, tmp_path):
    """`scripts/preflight.py` is the first command a new machine runs, so "it exits 0 when the machine is fine" is the
    property worth pinning — and it can only be asserted with a valid `DATA_ROOT`, which `build_dataset_root` supplies
    without the real 48 GB.
    """
    import subprocess
    import sys

    from tests.conftest import build_dataset_root

    data_root = build_dataset_root(str(tmp_path / 'data'), n_days=3)
    output_root = tmp_path / 'outputs'
    output_root.mkdir()

    environment = {**os.environ, 'PYTHONPATH': repo_root, 'DATA_ROOT': data_root,
                   'OUTPUT_ROOT': str(output_root)}
    result = subprocess.run([sys.executable, 'scripts/preflight.py'], cwd=repo_root, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, f'{result.stdout}\n{result.stderr[-1500:]}'
    assert '3 samples' in result.stdout, result.stdout


def test_PREFLIGHT_fails_and_NAMES_the_unset_variables(repo_root, tmp_path):
    """The failure path matters more than the success one: an unset `DATA_ROOT` substitutes to the empty string and
    fails deep inside `prepare_modeling`, so preflight's job is to say so first, by name."""
    import subprocess
    import sys

    # ⚠️ Set the two roots EMPTY rather than deleting them. Deleting only works on a machine with no `.env`, which is
    # every machine that has not followed the README's own `cp .env.example .env` step: `src/__init__.py` loads
    # `<root_path>/.env` from the location of `src/__init__.py` itself, so `cwd` cannot steer it away, and the file
    # then supplies both roots and the refusal never happens. An EMPTY value survives that, because `load_env_file`
    # skips any name already `in os.environ` and `''` counts as present -- and empty is the case that actually bites:
    # `{{$VAR}}` substitutes an unset variable to the empty string, which is what this test's docstring describes.
    environment = {**os.environ, 'DATA_ROOT': '', 'OUTPUT_ROOT': '', 'PYTHONPATH': repo_root}
    result = subprocess.run([sys.executable, 'scripts/preflight.py'], cwd=repo_root, env=environment,
                            capture_output=True, text=True)

    assert result.returncode == 1, result.stdout
    assert "['DATA_ROOT', 'OUTPUT_ROOT']" in result.stdout, result.stdout


# =====================================================================================================================
# The tracked launch scripts (Step 5 block 5e)
#
# `job_scripts/` is gitignored scratch; `job_scripts.example/` is the tracked, site-agnostic version. A tracked launch
# script is read as an instruction, so a machine-specific value in one is worse than no script at all.
# =====================================================================================================================
EXAMPLE_SCRIPTS = 'job_scripts.example'


def _example_scripts(repo_root):
    directory = os.path.join(repo_root, EXAMPLE_SCRIPTS)
    return [os.path.join(directory, name) for name in sorted(os.listdir(directory)) if name.endswith('.sh')]


def test_the_example_launch_scripts_are_SYNTACTICALLY_VALID(repo_root):
    """`bash -n` on each. They are never executed by the suite — a slurm submission is not something pytest can do — so
    without this a typo in one is found by a queued job failing in one second."""
    import subprocess

    broken = []
    for path in _example_scripts(repo_root):
        result = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
        if result.returncode != 0:
            broken.append(f'{os.path.basename(path)}: {result.stderr.strip()}')
    assert not broken, broken


def test_the_example_launch_scripts_carry_NO_machine_specific_value(repo_root):
    """Active lines only: the comments deliberately QUOTE the old `job_scripts/logs/...` path to explain why it changed,
    and a guard that cannot tell an explanation from a setting would forbid documenting the fix.
    """
    offenders = []
    for path in _example_scripts(repo_root):
        for number, line in enumerate(open(path), start=1):
            code = line.split('#', 1)[0]                       # drop comments, including the `#SBATCH` ones
            if any(needle in code for needle in ('/homedata', '/home/aburq', 'zen16')):
                offenders.append(f'{os.path.basename(path)}:{number}: {line.strip()}')
    assert not offenders, f'machine-specific values on active lines: {offenders}'


def test_the_example_scripts_name_NO_DIRECTORY_in_their_slurm_log_paths(repo_root):
    """The failure this encodes cost a real job. slurm resolves a RELATIVE `--output` against the directory you submit
    FROM, so `--output=job_scripts/logs/output/%x_%j.out` submitted from inside `job_scripts/` asks for
    `job_scripts/job_scripts/logs/`, which slurm cannot create — killing the job in one second with NO log to say why.
    Naming no directory means a log always appears wherever you stood.
    """
    import re

    offenders = []
    for path in _example_scripts(repo_root):
        for number, line in enumerate(open(path), start=1):
            match = re.match(r'#SBATCH\s+--(output|error)=(\S+)', line.strip())
            if match and os.path.dirname(match.group(2)):
                offenders.append(f'{os.path.basename(path)}:{number}: {line.strip()}')
    assert not offenders, f'slurm log paths with a directory component: {offenders}'


def test_every_example_stage_script_SOURCES_the_shared_common(repo_root):
    """The repo-root search, the `.env` load, the interpreter check and the family/mode path mapping all live in
    `_common.sh`. A script that re-derives any of them by hand is how the seven drift apart."""
    for path in _example_scripts(repo_root):
        if os.path.basename(path) == '_common.sh':
            continue
        text = open(path).read()
        assert '_common.sh' in text, f'{os.path.basename(path)} does not source _common.sh'
        assert 'FAMILY=' in text and 'STAGE_NAME=' in text, os.path.basename(path)


def test_the_example_common_REFUSES_to_run_without_the_two_roots(repo_root, tmp_path):
    """No fallback values, deliberately — the gitignored original hardcoded a machine's paths here. A
    wrong-but-plausible default is worse than a missing one, because the run SUCCEEDS against the wrong data.
    """
    import shutil
    import subprocess

    # ⚠️ Run against a repo root that has NO `.env`, not against the real one. Deleting the two variables from the
    # environment is not enough on a machine that has a `.env`: `_common.sh` sources `$REPO_ROOT/.env`, which supplies
    # both roots, and the refusal never happens -- so this test used to pass only on a clone that had not followed the
    # README. Worse, a shell `. file` ASSIGNS unconditionally, so unlike the Python loader it overrides an
    # already-exported value; setting the roots empty would not survive it either.
    #
    # `_common.sh` locates the root by searching for `src/stages/evaluate.py`, so a skeleton holding just that file is
    # a valid root, and `job_scripts.example/..` is one of the candidates it tries from `cwd`.
    root = tmp_path / 'repo'
    (root / 'src' / 'stages').mkdir(parents=True)
    (root / 'src' / 'stages' / 'evaluate.py').touch()
    shutil.copytree(os.path.join(repo_root, EXAMPLE_SCRIPTS), root / EXAMPLE_SCRIPTS)

    script = (
        'set -euo pipefail\n'
        'FAMILY=deterministic_unet; TIER=_smoke_cpu; MODE=daily; STAGE_NAME=probe\n'
        f'source {EXAMPLE_SCRIPTS}/_common.sh\n'
    )
    # SLURM_SUBMIT_DIR is the FIRST candidate `_common.sh` tries, so it has to go too: when the suite itself runs
    # inside a slurm job it points at the real repo, and the skeleton would never be reached.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('DATA_ROOT', 'OUTPUT_ROOT', 'SLURM_SUBMIT_DIR')}
    result = subprocess.run(['bash', '-c', script], cwd=str(root), env=environment, capture_output=True, text=True)

    assert result.returncode == 2, f'expected exit 2, got {result.returncode}\n{result.stdout}\n{result.stderr}'
    assert '.env.example' in result.stderr, result.stderr


def test_the_example_common_DERIVES_the_task_dependent_paths_TOGETHER(repo_root, tmp_path):
    """`MODE` must move the prepared directory, the search space AND the metrics config as one. Each has a silent
    failure mode alone: a daily metrics suite cuts a probability field at `> 0` (POD ~ 1, a contingency table of
    nonsense, nothing raised), and a daily search space names a selection metric the module rejects.
    """
    import subprocess

    from tests.conftest import build_dataset_root

    data_root = build_dataset_root(str(tmp_path / 'data'), n_days=2)
    script = (
        'set -euo pipefail\n'
        'FAMILY=deterministic_unet; TIER=_smoke_cpu; MODE=hourly; STAGE_NAME=probe\n'
        f'source {EXAMPLE_SCRIPTS}/_common.sh\n'
        'echo "PREPARED=$PREPARED_PATH"; echo "SEARCH=$SEARCH_SPACE"; echo "METRICS=$METRICS_CONFIG"\n'
    )
    environment = {**os.environ, 'DATA_ROOT': data_root, 'OUTPUT_ROOT': str(tmp_path / 'out')}
    result = subprocess.run(['bash', '-c', script], cwd=repo_root, env=environment, capture_output=True, text=True)

    assert result.returncode == 0, f'{result.stdout}\n{result.stderr}'
    assert 'prepared/hourly' in result.stdout, result.stdout
    assert 'SEARCH=config/deterministic_unet/search_space_hourly.yaml' in result.stdout, result.stdout
    assert 'METRICS=config/eval/metrics_hourly.yaml' in result.stdout, result.stdout


def test_the_example_common_REJECTS_hourly_for_a_family_without_an_hourly_pipeline(repo_root, tmp_path):
    """Only `deterministic_unet` has one. Left unchecked, `MODE=hourly FAMILY=diffusion` would name a prepared
    directory nothing ever writes and fail much later, looking like missing data."""
    import subprocess

    from tests.conftest import build_dataset_root

    data_root = build_dataset_root(str(tmp_path / 'data'), n_days=2)
    script = (
        'set -euo pipefail\n'
        'FAMILY=diffusion; TIER=_smoke_cpu; MODE=hourly; STAGE_NAME=probe\n'
        f'source {EXAMPLE_SCRIPTS}/_common.sh\n'
    )
    environment = {**os.environ, 'DATA_ROOT': data_root, 'OUTPUT_ROOT': str(tmp_path / 'out')}
    result = subprocess.run(['bash', '-c', script], cwd=repo_root, env=environment, capture_output=True, text=True)

    assert result.returncode == 2, result.stdout
    assert 'deterministic_unet only' in result.stderr, result.stderr


def test_the_cartopy_bundle_is_EXEMPT_from_the_git_lfs_filters(repo_root):
    """The load-bearing line, asserted through git itself rather than by grepping `.gitattributes`.

    `*.shp filter=lfs` still applies everywhere outside `data/cartopy/shapefiles/**`, so the exemption is what keeps this file an
    ordinary blob. Delete it and the next commit by anyone with git-lfs installed turns the shapefile into a pointer.

    It is also why the file is plain in the first place: an LFS object requires `git lfs install`, whose
    `filter.lfs.required = true` makes EVERY git command in the repo exit 128 without the binary — measured, including
    the `git diff HEAD` that `lazy.code_state_hash` runs, which then silently degrades the cache key to a hash of
    `src/` alone.
    """
    import subprocess

    path = 'data/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.shp'
    result = subprocess.run(['git', 'check-attr', 'filter', 'diff', 'merge', '--', path],
                            cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for line in result.stdout.strip().splitlines():
        attribute, value = line.rsplit(': ', 1)[0].rsplit(': ', 1)[-1], line.rsplit(': ', 1)[1]
        assert value == 'unset', (
            f'{attribute} is {value!r} for the bundled coastline — the `data/cartopy/shapefiles/** -filter -diff '
            f'-merge` '
            f'exemption in .gitattributes is gone, so this file will become a git-lfs pointer on the next commit made '
            f'with git-lfs installed'
        )


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
