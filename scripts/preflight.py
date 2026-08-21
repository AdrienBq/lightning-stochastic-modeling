"""Check that this machine can actually run the pipeline, and say precisely what is missing if it cannot.

Run it first on a fresh clone, before waiting on a real pipeline:

    python scripts/preflight.py

Every check here corresponds to something that otherwise fails LATE and in the wrong place — the reason this file
exists at all. In rough order of how misleading each failure is on its own:

* an unset ``DATA_ROOT`` surfaces inside ``prepare_modeling``, not at startup, because ``{{$VAR}}`` substitutes to the
  EMPTY STRING rather than erroring;
* an unset ``OUTPUT_ROOT`` makes every output path absolute at the filesystem root (``/family/prepared``), which
  ``setup`` catches — but only once the pipeline is already running;
* mlflow builds each stage's command from a hardcoded literal ``"python"``, so a stage subprocess can get a different
  interpreter than the one that launched the run and die on ``import mlflow``, reported as a broken STAGE;
* a git-lfs pointer where the cartopy coastline should be makes cartopy die inside a shapefile reader with
  ``KeyError: 828781878``, reported as corrupt data;
* a git that cannot run in this repo degrades the lazy cache key SILENTLY to a hash of ``src/`` alone, after which
  stages are re-run or wrongly skipped with nothing said.

Exit codes: ``0`` everything required passed · ``1`` at least one required check failed.
"""
import os
import subprocess
import sys

# `src` first, so the bootstrap has applied `.env`, the PATH fix and CARTOPY_DATA_DIR before anything is inspected.
# Checking before that would report the raw shell environment, which is not what a pipeline run sees.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src                                                                              # noqa: F401,E402

from src.utils.io.environment import is_git_lfs_pointer                                 # noqa: E402

ROOT = src.root_path

# Import name -> the distribution that provides it, for a message that names what to install.
RUNTIME_IMPORTS = [
    ('mlflow', 'mlflow'), ('fire', 'fire'), ('yaml', 'pyyaml'),
    ('torch', 'torch'), ('lightning', 'lightning'), ('optuna', 'optuna'),
    ('numpy', 'numpy'), ('pandas', 'pandas'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'),
    ('matplotlib', 'matplotlib'), ('cartopy', 'cartopy'),
]


class Report:
    """Check results, printed AS THEY COMPLETE. ``required=False`` entries are shown but never fail the run.

    Incremental rather than collected-then-rendered because the dependency check really does import torch, lightning
    and cartopy — ~26 s on a warm cache, more on a cold one — and a preflight that prints nothing for half a minute
    reads as hung, which is a poor first impression for the one command a new machine runs first.
    """

    WIDTH = 18                      # the longest check name, so rows line up without buffering them all first

    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail, required=True):
        self.rows.append((name, ok, detail, required))
        mark = 'ok  ' if ok else ('FAIL' if required else 'note')
        print(f'  [{mark}] {name.ljust(self.WIDTH)}  {detail}', flush=True)

    def failed(self):
        return [row for row in self.rows if row[3] and not row[1]]


def check_python(report):
    version = '.'.join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info[:2] >= (3, 11)
    report.add('python', ok, f'{version} at {sys.executable}'
                             + ('' if ok else '  <- 3.11+ required'))


def check_imports(report):
    """Really imports each one, rather than asking `importlib.util.find_spec` whether it exists.

    Slower (~26 s, mostly torch and lightning) and worth it: a present-but-broken compiled extension is one of the
    documented failure modes here — the `GLIBCXX_3.4.29` wall on an old cluster node raises at IMPORT, and `find_spec`
    would report it as fine.
    """
    print('  ...  importing the runtime stack (torch and lightning make this ~30 s)', flush=True)
    missing, broken = [], []
    for module, distribution in RUNTIME_IMPORTS:
        try:
            __import__(module)
        except ImportError as error:
            # ModuleNotFoundError means absent; any other ImportError means present and unusable, which needs a
            # different fix (usually LD_LIBRARY_PATH, not pip) and so must not be reported as "missing".
            if isinstance(error, ModuleNotFoundError) and error.name == module:
                missing.append(distribution)
            else:
                broken.append(f'{distribution}: {error}')

    if broken:
        report.add('dependencies', False, f'{len(broken)} installed but UNUSABLE — {broken[0]}'
                                         + ('' if len(broken) == 1 else f' (+{len(broken) - 1} more)'))
        return
    report.add('dependencies', not missing,
               f'{len(RUNTIME_IMPORTS)} runtime imports present' if not missing
               else f'missing {missing} — pip install -r minimal_requirements.txt')


def check_env_file(report):
    """Informational: `.env` is optional, since a launch script may export the variables itself."""
    path = os.path.join(ROOT, '.env')
    report.add('.env', os.path.isfile(path),
               'loaded' if os.path.isfile(path)
               else 'absent — fine if your shell or launch script exports the paths (cp .env.example .env otherwise)',
               required=False)


def check_data_root(report):
    root = os.environ.get('DATA_ROOT')
    if not root:
        report.add('DATA_ROOT', False,
                   'unset — a missing DATA_ROOT substitutes to the EMPTY STRING and fails inside prepare_modeling')
        return
    if not os.path.isdir(root):
        report.add('DATA_ROOT', False, f'{root} is not a directory')
        return

    missing = [name for name in ('metadata.json', 'metadata.csv', 'samples')
               if not os.path.exists(os.path.join(root, name))]
    if missing:
        report.add('DATA_ROOT', False, f'{root} is missing {missing} — not a dataset root')
        return

    samples = len([name for name in os.listdir(os.path.join(root, 'samples')) if name.endswith('.pt')])
    report.add('DATA_ROOT', samples > 0, f'{root}  ({samples} samples)'
                                        + ('' if samples else '  <- samples/ holds no .pt files'))


def check_output_root(report):
    root = os.environ.get('OUTPUT_ROOT')
    if not root:
        report.add('OUTPUT_ROOT', False,
                   'unset — every output path then resolves at the filesystem root; `setup` refuses the run')
        return

    probe = os.path.join(root, '.preflight_write_probe')
    try:
        os.makedirs(probe, exist_ok=True)
        os.rmdir(probe)
    except OSError as error:
        report.add('OUTPUT_ROOT', False, f'{root} is not writable ({error.strerror})')
        return
    report.add('OUTPUT_ROOT', True, f'{root}  (writable)')


def check_tracking_uri(report):
    """Informational: unset is valid, it just means the store lands in the checkout and grows without bound."""
    uri = os.environ.get('MLFLOW_TRACKING_URI')
    report.add('mlflow store', True,
               uri or f'unset — artifacts go to {os.path.join(ROOT, "mlruns")} inside the checkout',
               required=False)


def check_cartopy_bundle(report):
    bundle = os.path.join(ROOT, 'data', 'cartopy', 'shapefiles', 'natural_earth', 'physical')
    shapefile = os.path.join(bundle, 'ne_50m_coastline.shp')

    if not os.path.isfile(shapefile):
        report.add('cartopy data', False,
                   f'{shapefile} is missing — plotting will fetch it at PLOT time, which fails on an offline node')
        return
    if is_git_lfs_pointer(shapefile):
        report.add('cartopy data', False,
                   'the bundled coastline is a git-lfs POINTER, not the shapefile. The `data/cartopy/shapefiles/** '
                   '-filter` exemption in .gitattributes is gone; restore it and check the blob out again')
        return
    report.add('cartopy data', True,
               f'bundled ({os.path.getsize(shapefile):,} bytes), CARTOPY_DATA_DIR='
               f'{os.environ.get("CARTOPY_DATA_DIR", "<unset>")}')


def check_stage_interpreter(report):
    """The mlflow bare-`python` defect: a STAGE subprocess must get this interpreter, not whatever `PATH` had first.

    Asserted the way the failure happens — spawn a bare ``python`` and have it import mlflow — rather than by comparing
    strings, because that is the exact thing the stage subprocess does.
    """
    probe = 'import sys, mlflow; print(sys.executable)'
    try:
        result = subprocess.run(['python', '-c', probe], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        report.add('stage interpreter', False, f'could not run a bare `python` ({error})')
        return

    if result.returncode != 0:
        tail = (result.stderr or '').strip().splitlines()[-1:] or ['(no stderr)']
        report.add('stage interpreter', False,
                   f'a bare `python` cannot import mlflow: {tail[0]} — every stage subprocess would die this way')
        return

    resolved = result.stdout.strip()
    ok = resolved == sys.executable
    report.add('stage interpreter', ok,
               f'`python` -> {resolved}' + ('' if ok else f'  <- NOT this interpreter ({sys.executable})'))


def check_git(report):
    """A git that cannot run here degrades the lazy cache key to a hash of `src/` alone, and says so only to a log."""
    for arguments in (['rev-parse', 'HEAD'], ['diff', 'HEAD'], ['ls-files', '--others', '--exclude-standard']):
        result = subprocess.run(['git', '-C', ROOT, *arguments], capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or '').strip().splitlines()[-1:] or ['(no stderr)']
            report.add('git', False,
                       f'`git {" ".join(arguments)}` exits {result.returncode}: {tail[0]} — the lazy cache key would '
                       f'silently degrade to a hash of src/ alone')
            return
    report.add('git', True, 'rev-parse / diff / ls-files all work (the lazy cache can key on the real code state)')


def main() -> int:
    print(f'preflight for {ROOT}\n')
    report = Report()
    for check in (check_python, check_imports, check_env_file, check_data_root, check_output_root,
                  check_tracking_uri, check_cartopy_bundle, check_stage_interpreter, check_git):
        check(report)

    failures = report.failed()
    if failures:
        print(f'\n{len(failures)} required check(s) failed: {[name for name, *_ in failures]}')
        return 1
    print('\nReady. Next: `pytest tests/pipeline_e2e_test.py -q --no-cov` proves the pipeline is wired, then\n'
          '`python run_project.py config/deterministic_unet/deterministic_unet_daily_smoke_cpu.yaml MY_EXPERIMENT`.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
