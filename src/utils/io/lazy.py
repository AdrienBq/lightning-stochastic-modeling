"""Lazy pipeline execution helpers: content/code fingerprinting and MLflow-store cache lookup.

The orchestrator (``src/stages/run.py``) uses these to skip a stage whose code, parameters and inputs are
identical to a previously recorded successful run (whose outputs are still on disk), and to detect
non-deterministic output changes (same code + inputs + params, different output).

State lives entirely in the MLflow tracking store: each executed stage's run is tagged with a content-derived
cache key and an output fingerprint (see the ``TAG_*`` constants), and a prior run is found by querying those
tags. No external database is required.

Checksum strategy is **hybrid**: files below ``file_max_bytes`` are content-hashed (byte-exact); larger files,
and every file inside a directory whose total size exceeds ``dir_max_bytes``, contribute only ``(relpath,
size)`` to the manifest. mtime is deliberately never used: it changes on every rewrite and would make a
deterministic re-run of a metadata-mode output always look "changed". The trade-off (a size-preserving content
change inside a metadata-mode file is not detected) is logged via a de-duplicated warning whenever a byte-exact
check is skipped.
"""
import os
import json
import hashlib
import logging
import subprocess

logger = logging.getLogger(__name__)

# tag keys written on each executed stage's MLflow run
TAG_STAGE = 'lazy_stage'
TAG_CACHE_KEY = 'lazy_cache_key'
TAG_CODE_STATE = 'lazy_code_state'
TAG_PARAMS_HASH = 'lazy_params_hash'
TAG_OUTPUT_FINGERPRINT = 'lazy_output_fingerprint'

# parameter keys whose values are stage OUTPUTS; every other param that resolves to an existing path is treated
# as an INPUT (matches the conventions already special-cased in src/stages/run.py)
OUTPUT_PARAM_KEYS = ('output-path', 'metrics-path', 'report-path')

# returned by fingerprint_path when the target does not exist
ABSENT = 'absent'

_CHUNK = 1024 * 1024
# de-duplicate "byte-exact check skipped" warnings within a single process
_warned_paths = set()


# --------------------------------------------------------------------------------------------------------------
# low-level hashing
# --------------------------------------------------------------------------------------------------------------
def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hash_file_content(abs_path: str) -> str:
    hasher = hashlib.sha256()
    with open(abs_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _human(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if value < 1024 or unit == 'TB':
            return f'{value:.1f} {unit}'
        value /= 1024


def _warn_skip(path: str, size: int, threshold: int, kind: str, label: str) -> None:
    if path in _warned_paths:
        return
    _warned_paths.add(path)
    logger.warning(
        'lazy: byte-exact check skipped for %s "%s" (%s > %s %s); the determinism guard uses a size-only '
        'manifest here', label, path, _human(size), _human(threshold), kind
    )


# --------------------------------------------------------------------------------------------------------------
# path fingerprinting (hybrid content / metadata)
# --------------------------------------------------------------------------------------------------------------
def fingerprint_path(abs_path: str, file_max_bytes: int, dir_max_bytes: int, label: str = 'input') -> str:
    """Hybrid content/metadata fingerprint of a file or directory.

    Args:
        abs_path: Absolute path to a file or directory.
        file_max_bytes: Files at or below this size are content-hashed; larger ones contribute size only.
        dir_max_bytes: A directory whose total size exceeds this is fingerprinted in size-only (metadata) mode.
        label: ``input`` or ``output`` (only used to make skip warnings readable).

    Returns:
        A short string fingerprint; ``ABSENT`` if the path does not exist.
    """
    if not os.path.exists(abs_path):
        return ABSENT

    if os.path.isfile(abs_path):
        size = os.path.getsize(abs_path)
        if size <= file_max_bytes:
            return 'f:c:' + _hash_file_content(abs_path)
        _warn_skip(abs_path, size, file_max_bytes, 'file budget', label)
        return f'f:m:{size}'

    # directory: collect (relpath, size) for every file, deterministically ordered
    entries = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(abs_path):
        dirnames.sort()
        for name in sorted(filenames):
            file_path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(file_path)
            except OSError:
                continue
            entries.append((os.path.relpath(file_path, abs_path), size, file_path))
            total += size
    entries.sort()

    metadata_only = total > dir_max_bytes
    if metadata_only:
        _warn_skip(abs_path, total, dir_max_bytes, 'dir budget', label)

    hasher = hashlib.sha256()
    for rel, size, file_path in entries:
        if metadata_only or size > file_max_bytes:
            if not metadata_only:
                _warn_skip(file_path, size, file_max_bytes, 'file budget', label)
            hasher.update(f'{rel}|m|{size}\n'.encode())
        else:
            hasher.update(f'{rel}|c|{_hash_file_content(file_path)}\n'.encode())
    return 'd:' + hasher.hexdigest()


def _resolve(value: str, root_path: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root_path, value)


def fingerprint_paths(paths: dict, root_path: str, file_max_bytes: int, dir_max_bytes: int,
                      label: str = 'input') -> str:
    """Combined fingerprint of a ``{param_key: path_value}`` mapping (order-independent)."""
    hasher = hashlib.sha256()
    for key in sorted(paths):
        abs_path = _resolve(paths[key], root_path)
        hasher.update(key.encode())
        hasher.update(b'=')
        hasher.update(fingerprint_path(abs_path, file_max_bytes, dir_max_bytes, label).encode())
        hasher.update(b'\n')
    return hasher.hexdigest()


def paths_present(paths: dict, root_path: str) -> bool:
    """True only if every value in ``paths`` resolves to an existing file/directory."""
    return all(os.path.exists(_resolve(value, root_path)) for value in paths.values())


# --------------------------------------------------------------------------------------------------------------
# parameter classification and hashing
# --------------------------------------------------------------------------------------------------------------
def classify_params(params: dict, root_path: str):
    """Split a stage's parameters into input-path and output-path mappings.

    Output paths are the keys in ``OUTPUT_PARAM_KEYS``; inputs are any other param whose value resolves to an
    existing path. Non-path scalars (mode, n-trials, seed, ...) belong in neither and are covered by
    ``params_hash`` instead.
    """
    inputs, outputs = {}, {}
    for key, value in params.items():
        if key in OUTPUT_PARAM_KEYS:
            outputs[key] = value
        elif isinstance(value, str) and os.path.exists(_resolve(value, root_path)):
            inputs[key] = value
    return inputs, outputs


def params_hash(params: dict) -> str:
    """Stable hash over the full resolved parameter dict (path strings included)."""
    return _sha256_hex(json.dumps(params, sort_keys=True, default=str))


# --------------------------------------------------------------------------------------------------------------
# code state, cache key and per-stage seed
# --------------------------------------------------------------------------------------------------------------
def _git(args, root_path: str) -> str:
    result = subprocess.run(
        ['git', '-C', root_path, *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def code_state_hash(root_path: str) -> str:
    """Hash of the whole-repo code state: git HEAD + working-tree diff + untracked (non-ignored) files.

    Any uncommitted edit busts the cache (conservative by design). Falls back to hashing the ``src/`` tree
    when git is unavailable, logging that code-state tracking is degraded.
    """
    try:
        hasher = hashlib.sha256()
        hasher.update(_git(['rev-parse', 'HEAD'], root_path).strip().encode())
        hasher.update(_git(['diff', 'HEAD'], root_path).encode())
        untracked = _git(['ls-files', '--others', '--exclude-standard'], root_path).split()
        for rel in sorted(untracked):
            file_path = os.path.join(root_path, rel)
            hasher.update(rel.encode())
            if os.path.isfile(file_path):
                try:
                    hasher.update(_hash_file_content(file_path).encode())
                except OSError:
                    pass
        return hasher.hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        logger.warning(
            'lazy: git unavailable (%s); code-state tracking degraded - hashing the src/ tree instead', error
        )
        return 'srctree:' + fingerprint_path(
            os.path.join(root_path, 'src'), file_max_bytes=10 * 1024 ** 2, dir_max_bytes=1024 ** 3, label='code'
        )


def compute_cache_key(code_hash: str, params_hash_value: str, inputs_fingerprint: str) -> str:
    """Combine the code, parameter and input fingerprints into a single stage cache key."""
    hasher = hashlib.sha256()
    for part in (code_hash, params_hash_value, inputs_fingerprint):
        hasher.update(part.encode())
        hasher.update(b'\x00')
    return hasher.hexdigest()


def stage_seed(code_hash: str, params_hash_value: str) -> int:
    """Deterministic per-stage seed derived from code + parameters (independent of input data bytes)."""
    digest = _sha256_hex(code_hash + '|' + params_hash_value)
    return int(digest[:16], 16) % (2 ** 31 - 1)


# --------------------------------------------------------------------------------------------------------------
# MLflow store lookup
# --------------------------------------------------------------------------------------------------------------
def find_cached_run(client, experiment_id: str, stage_name: str, cache_key: str):
    """Most recent FINISHED run of ``stage_name`` with a matching cache key, or None."""
    filter_string = (
        f"tags.{TAG_STAGE} = '{stage_name}' and tags.{TAG_CACHE_KEY} = '{cache_key}' "
        f"and attributes.status = 'FINISHED'"
    )
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            order_by=['attributes.start_time DESC'],
            max_results=1,
        )
    except Exception as error:  # pragma: no cover - never let a lookup failure abort the pipeline
        logger.warning('lazy: MLflow cache lookup failed (%s); running the stage', error)
        return None
    return runs[0] if runs else None
