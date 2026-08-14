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

``find_cached_run`` is driven against a stub client rather than a live tracking server: what matters about it is the
filter string it builds and its refusal to propagate a lookup failure, neither of which needs MLflow to be running.
"""
import hashlib
import os
import subprocess

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


# =====================================================================================================================
# The hashing primitives, directly (Block 5c)
# =====================================================================================================================
def test_the_text_hash_is_plain_sha256_of_the_utf8_bytes():
    """Pinned against ``hashlib`` rather than against itself. It feeds ``params_hash`` and ``stage_seed``, so a change of
    algorithm or encoding invalidates every cache entry ever recorded — which is survivable, but must be deliberate."""
    assert lazy._sha256_hex('abc') == hashlib.sha256(b'abc').hexdigest()


def test_a_file_hashed_in_CHUNKS_matches_a_one_shot_hash(tmp_path):
    """``_hash_file_content`` reads in 1 MB chunks so a prepared sample does not land in memory whole. The property worth
    testing is that the loop is a true streaming hash: a file several chunks long must give the same digest as hashing
    its bytes in one go, or the fingerprint would depend on the buffer size."""
    payload = bytes(range(256)) * (lazy._CHUNK // 128)        # ~2 chunks, non-repeating within each
    big = tmp_path / 'multi_chunk.bin'
    big.write_bytes(payload)

    assert os.path.getsize(big) > lazy._CHUNK
    assert lazy._hash_file_content(str(big)) == hashlib.sha256(payload).hexdigest()


def test_an_empty_file_still_hashes(tmp_path):
    """The loop body never executes; the digest is sha256 of nothing rather than an error or an empty string."""
    empty = tmp_path / 'empty.npy'
    empty.write_bytes(b'')
    assert lazy._hash_file_content(str(empty)) == hashlib.sha256(b'').hexdigest()


@pytest.mark.parametrize('num_bytes,expected', [
    (0, '0.0 B'), (512, '512.0 B'), (1024, '1.0 KB'), (1024 ** 2, '1.0 MB'),
    (256 * 1024 ** 2, '256.0 MB'), (2048 * 1024 ** 2, '2.0 GB'),
])
def test_byte_sizes_render_in_the_unit_a_reader_expects(num_bytes, expected):
    """The two thresholds a user actually sets — ``lazy_content_max_file_mb: 256`` and
    ``lazy_content_max_dir_mb: 2048`` — appear in the skip warning through this, so they must read back as the numbers
    that were configured."""
    assert lazy._human(num_bytes) == expected


def test_the_size_ladder_stops_at_terabytes():
    """The loop's exit condition is ``unit == 'TB'``, so a petabyte-scale value reports as thousands of TB rather than
    falling off the end and returning ``None`` — which would crash the warning that is meant to be informational."""
    assert lazy._human(4096 * 1024 ** 4).endswith(' TB')


def test_a_relative_path_resolves_against_the_repo_root_and_an_absolute_one_is_untouched(repo_root):
    """Every path in a pipeline YAML is written relative to the repo root, and the cache has to fingerprint the same file
    the stage will open."""
    assert lazy._resolve('config/eval/metrics.yaml', repo_root) == \
        os.path.join(repo_root, 'config/eval/metrics.yaml')
    assert lazy._resolve('/absolute/elsewhere.yaml', repo_root) == '/absolute/elsewhere.yaml'


# =====================================================================================================================
# The skip warning — de-duplicated, because a silent skip is the failure it exists to make visible
# =====================================================================================================================
@pytest.fixture
def fresh_warning_state():
    """``_warned_paths`` is module-global and lives for the process. Tests that assert on WHETHER a warning fired must
    reset it, or they pass or fail depending on which test ran first."""
    saved = set(lazy._warned_paths)
    lazy._warned_paths.clear()
    yield
    lazy._warned_paths.clear()
    lazy._warned_paths.update(saved)


def test_the_skip_warning_names_the_path_the_size_and_the_budget(fresh_warning_state, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        lazy._warn_skip('/data/samples/sample_000001.pt', 9 * 1024 ** 2, 256 * 1024 ** 2, 'file budget', 'input')

    message = caplog.records[0].getMessage()
    assert '/data/samples/sample_000001.pt' in message
    assert '9.0 MB' in message and '256.0 MB' in message
    assert 'input' in message


def test_the_same_path_warns_only_ONCE_per_process(fresh_warning_state, caplog):
    """The prepared directory holds 5843 samples over the file budget. Without the de-duplication the console would carry
    one warning per file per stage, which is how a real warning stops being read."""
    import logging

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            lazy._warn_skip('/data/big.pt', 10 ** 9, 10 ** 6, 'file budget', 'input')
    assert len(caplog.records) == 1


def test_a_DIFFERENT_path_still_warns(fresh_warning_state, caplog):
    """The other half — de-duplication is per path, not a one-warning-per-process latch that would hide every skip after
    the first."""
    import logging

    with caplog.at_level(logging.WARNING):
        lazy._warn_skip('/data/one.pt', 10 ** 9, 10 ** 6, 'file budget', 'input')
        lazy._warn_skip('/data/two.pt', 10 ** 9, 10 ** 6, 'file budget', 'input')
    assert len(caplog.records) == 2


# =====================================================================================================================
# git, and the degraded fallback when it is not there
# =====================================================================================================================
def test_git_runs_against_the_given_repo_and_returns_its_stdout(repo_root):
    head = lazy._git(['rev-parse', 'HEAD'], repo_root).strip()
    assert len(head) == 40 and all(character in '0123456789abcdef' for character in head)


def test_a_failing_git_command_RAISES_rather_than_returning_empty(repo_root):
    """``check=True`` is what lets ``code_state_hash`` distinguish "git said nothing changed" from "git did not run". An
    empty return on failure would silently hash a constant and make every stage look cached."""
    with pytest.raises(subprocess.CalledProcessError):
        lazy._git(['rev-parse', 'definitely-not-a-ref'], repo_root)


def test_the_code_state_hash_DEGRADES_loudly_when_git_is_unavailable(repo_root, monkeypatch, caplog):
    """The fallback path, reached on a machine with no git. It must still return a usable hash — but a hash of ``src/``
    alone, which no longer notices a config or data-prep edit, so the degradation is warned about rather than assumed."""
    import logging

    def no_git(*args, **kwargs):
        raise FileNotFoundError('git')

    monkeypatch.setattr(lazy, '_git', no_git)
    with caplog.at_level(logging.WARNING):
        degraded = lazy.code_state_hash(repo_root)

    assert degraded.startswith('srctree:')
    assert any('degraded' in record.getMessage() for record in caplog.records)


# =====================================================================================================================
# The MLflow store lookup
# =====================================================================================================================
class _StubMlflowClient:
    """Records the ``search_runs`` keyword arguments instead of querying a tracking server."""

    def __init__(self, runs=(), error=None):
        self.runs = list(runs)
        self.error = error
        self.calls = []

    def search_runs(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.runs


def test_the_lookup_filters_on_the_stage_the_key_AND_a_finished_status():
    """All three clauses matter. Without the stage name a key collision across stages hits; without ``FINISHED`` a run
    that crashed halfway would be reused as a cache hit, with its partial outputs on disk."""
    client = _StubMlflowClient()
    lazy.find_cached_run(client, '42', 'tune', 'abc123')

    filter_string = client.calls[0]['filter_string']
    assert f"tags.{lazy.TAG_STAGE} = 'tune'" in filter_string
    assert f"tags.{lazy.TAG_CACHE_KEY} = 'abc123'" in filter_string
    assert "attributes.status = 'FINISHED'" in filter_string
    assert client.calls[0]['experiment_ids'] == ['42']


def test_the_lookup_asks_for_the_MOST_RECENT_matching_run_only():
    client = _StubMlflowClient()
    lazy.find_cached_run(client, '42', 'tune', 'abc123')
    assert client.calls[0]['order_by'] == ['attributes.start_time DESC']
    assert client.calls[0]['max_results'] == 1


def test_a_matching_run_is_returned_and_no_match_gives_NONE():
    marker = object()
    assert lazy.find_cached_run(_StubMlflowClient([marker]), '42', 'tune', 'k') is marker
    assert lazy.find_cached_run(_StubMlflowClient([]), '42', 'tune', 'k') is None


def test_a_lookup_FAILURE_is_a_cache_miss_not_a_pipeline_failure(caplog):
    """Deliberate: the cache is an optimisation, so an unreachable tracking store must degrade to running the stage. The
    warning is what keeps "everything re-ran today" from being a mystery."""
    import logging

    client = _StubMlflowClient(error=RuntimeError('tracking store unreachable'))
    with caplog.at_level(logging.WARNING):
        assert lazy.find_cached_run(client, '42', 'tune', 'k') is None
    assert any('cache lookup failed' in record.getMessage() for record in caplog.records)
