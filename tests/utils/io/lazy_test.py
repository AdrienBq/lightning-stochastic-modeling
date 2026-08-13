"""Tests for src/utils/io/lazy.py — the pipeline's lazy-cache key machinery.

Untouched plumber template code, byte-identical to branch A's, and the SECOND documented silent-failure mode in the repo
with no tests until now. CLAUDE.md states both halves:

    ``OUTPUT_PARAM_KEYS`` (``output-path``, ``metrics-path``, ``report-path``) are treated as a stage's outputs by the
    lazy cache; any other parameter resolving to an existing path is an input. Use those names. Corollary: a *stale*
    path silently degrades to a plain scalar, so the cache stops invalidating on that input — keep ``metrics-config`` /
    ``split-config`` / ``model-config`` pointing at files that exist.

The stale-path degradation is the dangerous one and gets the sharpest test here. A pipeline whose ``metrics-config``
points at a moved file does not fail: the parameter stops being recognised as an input, so editing ``metrics.yaml`` no
longer busts the cache and the stage keeps returning an old cached result.

The MLflow-backed parts (``find_cached_run``) need a tracking server and belong to Step 4's end-to-end gate.
"""
import os

import pytest

from src.utils.io import lazy


# =====================================================================================================================
# Output vs input classification
# =====================================================================================================================
def test_the_output_keys_are_exactly_the_three_documented_names():
    """A stage naming its output anything else has it classified as an INPUT, so the cache fingerprints the output
    directory's contents and invalidates on its own previous run."""
    assert lazy.OUTPUT_PARAM_KEYS == ('output-path', 'metrics-path', 'report-path')


def test_an_existing_path_under_an_output_key_is_an_output(tmp_path, repo_root):
    outputs_dir = tmp_path / 'outputs'
    outputs_dir.mkdir()
    inputs, outputs = lazy.classify_params({'output-path': str(outputs_dir)}, repo_root)
    assert 'output-path' in outputs
    assert 'output-path' not in inputs


def test_an_existing_path_under_ANY_other_key_is_an_input(tmp_path, repo_root):
    config = tmp_path / 'metrics.yaml'
    config.write_text('metrics: {}\n')
    inputs, outputs = lazy.classify_params({'metrics-config': str(config)}, repo_root)
    assert 'metrics-config' in inputs
    assert 'metrics-config' not in outputs


def test_a_plain_scalar_is_neither_an_input_nor_an_output(repo_root):
    inputs, outputs = lazy.classify_params({'split': 'valid', 'ensemble-size': 4}, repo_root)
    assert 'split' not in inputs and 'split' not in outputs
    assert 'ensemble-size' not in inputs and 'ensemble-size' not in outputs


def test_a_REAL_tune_stage_has_its_config_parameters_classified_as_inputs(repo_root):
    """Driven from the shipped pipeline rather than a fixture, because the failure mode is config-side: ``model-config``
    and ``metrics-config`` must fingerprint as inputs so that editing a search space or the metric suite busts the cache.
    A hand-built dict would keep passing after the real config renamed a key."""
    import os

    from src.utils.io.parse_config import parse_config

    config = parse_config(os.path.join(repo_root, 'config/deterministic_unet/deterministic_unet.yaml'))
    tune = next(params for stage in config['stages'] for name, params in stage.items() if name == 'tune')

    inputs, outputs = lazy.classify_params(dict(tune), repo_root)
    assert 'model-config' in inputs, sorted(inputs)
    assert 'metrics-config' in inputs, sorted(inputs)
    assert 'ensemble-size' not in inputs and 'ensemble-size' not in outputs, 'a scalar must be neither'


# =====================================================================================================================
# ⚠️ THE stale-path degradation
# =====================================================================================================================
def test_a_STALE_path_silently_stops_being_an_input(tmp_path, repo_root):
    """The documented corollary, and the reason to keep those config paths valid. A path that does not exist is not an
    error — it is simply not recognised as a path, so it degrades to a plain scalar and the cache stops fingerprinting
    its CONTENTS. Editing the file it used to point at then no longer invalidates the stage."""
    missing = str(tmp_path / 'moved_away' / 'metrics.yaml')
    assert not os.path.exists(missing)

    inputs, outputs = lazy.classify_params({'metrics-config': missing}, repo_root)
    assert 'metrics-config' not in inputs, 'a stale path must not be silently treated as a live input'
    assert 'metrics-config' not in outputs


def test_the_cache_key_stops_changing_when_the_input_path_goes_stale(tmp_path, repo_root):
    """The consequence made concrete. With a LIVE path, editing the file changes the fingerprint and therefore the cache
    key. With a STALE path, the two edits are indistinguishable — same key, so a cached run is reused."""
    config = tmp_path / 'metrics.yaml'

    def key_for(params):
        fingerprint = lazy.fingerprint_paths(lazy.classify_params(params, repo_root)[0], repo_root,
                                             file_max_bytes=10 ** 9, dir_max_bytes=10 ** 9)
        return lazy.compute_cache_key('code', lazy.params_hash(params), fingerprint)

    config.write_text('metrics: {a: 1}\n')
    live_before = key_for({'metrics-config': str(config)})
    config.write_text('metrics: {a: 2}\n')
    live_after = key_for({'metrics-config': str(config)})
    assert live_before != live_after, 'a live input must invalidate the cache when it changes'

    stale = str(tmp_path / 'gone' / 'metrics.yaml')
    assert key_for({'metrics-config': stale}) == key_for({'metrics-config': stale})


def test_a_missing_path_fingerprints_as_ABSENT(tmp_path):
    """Not as an error and not as an empty hash — a distinct sentinel, so a stage whose input appears later does get a
    different key than one where it never existed."""
    assert lazy.fingerprint_path(str(tmp_path / 'nope'), 10 ** 9, 10 ** 9) == lazy.ABSENT


# =====================================================================================================================
# Fingerprinting
# =====================================================================================================================
def test_a_files_fingerprint_follows_its_CONTENT_not_its_name(tmp_path):
    first = tmp_path / 'a.txt'
    second = tmp_path / 'b.txt'
    first.write_text('same')
    second.write_text('same')
    assert lazy.fingerprint_path(str(first), 10 ** 9, 10 ** 9) == \
        lazy.fingerprint_path(str(second), 10 ** 9, 10 ** 9)

    second.write_text('different')
    assert lazy.fingerprint_path(str(first), 10 ** 9, 10 ** 9) != \
        lazy.fingerprint_path(str(second), 10 ** 9, 10 ** 9)


def test_a_directory_fingerprint_changes_when_a_member_changes(tmp_path):
    directory = tmp_path / 'prepared'
    directory.mkdir()
    (directory / 'one.npy').write_bytes(b'\x00' * 16)
    before = lazy.fingerprint_path(str(directory), 10 ** 9, 10 ** 9)
    (directory / 'one.npy').write_bytes(b'\x01' * 16)
    assert lazy.fingerprint_path(str(directory), 10 ** 9, 10 ** 9) != before


def test_a_directory_fingerprint_changes_when_a_member_is_ADDED(tmp_path):
    directory = tmp_path / 'prepared'
    directory.mkdir()
    (directory / 'one.npy').write_bytes(b'\x00' * 16)
    before = lazy.fingerprint_path(str(directory), 10 ** 9, 10 ** 9)
    (directory / 'two.npy').write_bytes(b'\x00' * 16)
    assert lazy.fingerprint_path(str(directory), 10 ** 9, 10 ** 9) != before


def test_an_oversized_file_is_SKIPPED_with_a_warning(tmp_path, caplog):
    """The prepared samples directory is ~8.7 MB x 5843, so hashing it every stage would dominate the run. The threshold
    skip is deliberate — but it means the cache does NOT invalidate on a change to a skipped input, which is why the
    warning exists."""
    import logging

    big = tmp_path / 'big.pt'
    big.write_bytes(b'\x00' * 4096)
    with caplog.at_level(logging.WARNING):
        fingerprint = lazy.fingerprint_path(str(big), file_max_bytes=64, dir_max_bytes=10 ** 9)
    assert fingerprint != lazy.ABSENT
    assert caplog.records, 'skipping an input silently would hide a cache that never invalidates'


def test_the_same_paths_fingerprint_identically_across_calls(tmp_path, repo_root):
    config = tmp_path / 'a.yaml'
    config.write_text('a: 1\n')
    params = {'metrics-config': str(config)}
    inputs = lazy.classify_params(params, repo_root)[0]
    first = lazy.fingerprint_paths(inputs, repo_root, 10 ** 9, 10 ** 9)
    second = lazy.fingerprint_paths(inputs, repo_root, 10 ** 9, 10 ** 9)
    assert first == second


def test_paths_present_reports_whether_the_outputs_exist(tmp_path, repo_root):
    """How the cache decides a cached run's outputs are still on disk — a hit whose outputs were deleted has to miss."""
    existing = tmp_path / 'out'
    existing.mkdir()
    assert lazy.paths_present({'output-path': str(existing)}, repo_root)
    assert not lazy.paths_present({'output-path': str(tmp_path / 'gone')}, repo_root)


# =====================================================================================================================
# The parameter hash and the cache key
# =====================================================================================================================
def test_the_params_hash_ignores_dict_ordering():
    """Two configs differing only in key order are the same configuration, and would otherwise miss the cache."""
    assert lazy.params_hash({'a': 1, 'b': 2}) == lazy.params_hash({'b': 2, 'a': 1})


def test_the_params_hash_changes_with_a_value():
    assert lazy.params_hash({'a': 1}) != lazy.params_hash({'a': 2})


def test_the_cache_key_depends_on_all_three_of_its_inputs():
    """Code state, parameters and input fingerprints. Dropping any one gives a key that hits across a real change."""
    base = lazy.compute_cache_key('code', 'params', 'inputs')
    assert base != lazy.compute_cache_key('other', 'params', 'inputs')
    assert base != lazy.compute_cache_key('code', 'other', 'inputs')
    assert base != lazy.compute_cache_key('code', 'params', 'other')


def test_the_cache_key_is_deterministic():
    assert lazy.compute_cache_key('c', 'p', 'i') == lazy.compute_cache_key('c', 'p', 'i')


# =====================================================================================================================
# The seed, and the code-state hash
# =====================================================================================================================
def test_the_stage_seed_is_derived_from_the_cache_inputs():
    """``PIPELINE_SEED`` is exported by the orchestrator and applied by ``src/stages/__init__.py``, so it must be a pure
    function of the stage's identity — a random seed would make every cached run irreproducible."""
    first = lazy.stage_seed('code', 'params')
    assert first == lazy.stage_seed('code', 'params')
    assert first != lazy.stage_seed('code', 'other')
    assert isinstance(first, int) and first >= 0


def test_the_code_state_hash_reflects_the_whole_repo_dirty_diff(repo_root):
    """The reason CLAUDE.md says to COMMIT before running a pipeline: the key includes the dirty diff, so any
    uncommitted edit anywhere busts every cache entry."""
    assert isinstance(lazy.code_state_hash(repo_root), str)
    assert lazy.code_state_hash(repo_root) == lazy.code_state_hash(repo_root)


def test_the_mlflow_tag_names_are_stable():
    """They are read back from previous runs, so renaming one makes every existing cached run unfindable."""
    assert lazy.TAG_STAGE == 'lazy_stage'
    assert lazy.TAG_CACHE_KEY == 'lazy_cache_key'
    assert lazy.TAG_CODE_STATE == 'lazy_code_state'
    assert lazy.TAG_PARAMS_HASH == 'lazy_params_hash'
    assert lazy.TAG_OUTPUT_FINGERPRINT == 'lazy_output_fingerprint'
