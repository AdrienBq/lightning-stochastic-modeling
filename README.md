# Plumber

**A minimalistic ML pipeline template**

## Installation

Once you instantiate the template and clone the repository, install [`git lfs`](https://git-lfs.com/).
You can find the default options in the [`.gitattributes`](.gitattributes) file.

It is highly recommended to create a Python environment (3.11+) via `conda` or `Pipenv` for versioning.

## Usage

This template is intended for pipeline execution with [MLflow](https://mlflow.org/).
Pipelines are a tool for tracking the execution of code _while_ you develop.
To use the pipelines effectively, follow these few recommendations and your best judgement:
1. commit your code frequently, especially before running a pipeline;
2. break down your code into consecutive "stages", with obvious inputs and outputs;
3. version input data, your environment(e.g.: via
[`conda export`](https://docs.conda.io/projects/conda/en/stable/commands/export.html) and subsequent 
commit), and optionally the outputs;
4. seed **everything**.

The intended usage paradigm for the repository is build one or more pipelines in a readable `YAML` format
which contain all necessary specification to reproduce your results exactly
(see the [`hello_world.yaml`](config/hello_world.yaml) example).

### General Pipeline (Recommended) Structure
The `YAML` file should contain some boilerplate (used by the 
[`run_project.py`](run_project.py) and [`run.py`](src/stages/run.py) scripts to execute the pipeline), and under the 
`stages` attribute, a list of executable names under `scr/stages` without extensions (e.g.: `hello_world` to execute
[`src/stages/hello_world.py`](src/stages/hello_world.py)), each with a specification of the input arguments (use `{}`
if there are no inputs to a stage).

See the following pseudocode example.
```yaml
project_uri: 'src/stages'                                                           # the path of the directory where each stage scrip is saved
log_artifacts: false                                                                # whether to log the output file/directory as artifact
log_models: false                                                                   # whether to log the output model
orchestrator: mlflow                                                                # (optional) 'mlflow' (default, sequential) or 'prefect' (flow UI)
lazy: false                                                                         # (optional) skip stages whose code + params + inputs are unchanged
ensure_determinism: false                                                           # (optional) raise instead of warn on non-deterministic outputs

stages:
    - <YOUR SCRIPT NAME>: {<A DICT OF OPTIONS, SUCH AS INPUT AND OUTPUT SPECIFICATIONS>}
    - <SCRIPT NAMES CAN BE REPEATED>:
        OPTIONS: CAN EVEN
        BE: EXPANDED
    - <ANOTHER SCRIPT NAME>: 
        REPLACE-UNDERSCORES-WITH-HYPHENS: IF YOU USE THE FIRE PACKAGE
    
tags:                                                                               # tags to be applied to the run
    version: 1.0
```

### Lazy Execution and the Determinism Guard

Two optional top-level flags (placed right after the boilerplate, both default `false`) make re-running a
pipeline cheap and reproducibility-checked — backed entirely by the MLflow store, so no extra database is needed:

- **`lazy`** — before running a stage the orchestrator computes a *cache key* from (a) the whole-repo code state
  (git `HEAD` + working-tree diff + untracked files), (b) the stage's parameters and (c) checksums of its input
  files/folders. If a previously recorded **successful** run has the same cache key and its outputs are still
  present and byte-identical on disk, the stage is **skipped**. The cache key and an output fingerprint are stored
  as tags on each stage's MLflow run.
- **`ensure_determinism`** — when a stage *is* executed with an unchanged cache key but produces a **different**
  output than last recorded, that is non-deterministic behaviour. By default it is logged as a warning; set this to
  `true` to raise an exception instead.

Both are **overridable per stage** by adding a `lazy:` / `ensure-determinism:` key inside that stage's parameter
block (the orchestrator strips these before invoking the stage's CLI). Checksums are *hybrid*: small files are
content-hashed (byte-exact) while very large files/directories fall back to a size-only manifest — a skipped
byte-exact check is logged, and the thresholds are tunable via `lazy_content_max_file_mb` /
`lazy_content_max_dir_mb`. Because the code state is the whole-repo dirty diff, **any** uncommitted edit busts the
caches (conservative by design — commit before relying on lazy hits). Laziness applies only to the orchestrator;
running a stage directly via its own CLI always executes it.

The convention is that the keys in `OUTPUT_PARAM_KEYS` (`output-path`, `metrics-path`, `report-path`) are treated
as a stage's outputs and any other parameter resolving to an existing path is an input — see
[`src/utils/io/lazy.py`](src/utils/io/lazy.py). A `setup` stage (see [`src/stages/setup.py`](src/stages/setup.py),
the recommended stage that creates the output directories) is never cached.

### Deterministic Seeding

Recommendation #4 above ("seed **everything**") is automated: every stage is seeded deterministically before its
body runs, with a seed derived from its code state + parameters (so a fixed config reproduces the same run). The
orchestrator exports it as the `PIPELINE_SEED` environment variable and the auto-hook in
[`src/stages/__init__.py`](src/stages/__init__.py) applies it via
[`src/utils/seeding.py`](src/utils/seeding.py), which seeds Python's `random` plus `numpy` / `torch` /
`lightning` **when those libraries are present** (absent ones are skipped). A stage's own seeding (e.g. an
explicit `seed:` parameter) still applies on top of this baseline.

### Plotting

[`src/utils/plotting`](src/utils/plotting) bundles a small, opinionated matplotlib setup: it registers the bundled
Inter font ([`data/fonts/inter`](data/fonts/inter)) as the global font, exposes colour-blind-safe IBM/Tol
palettes and ready-made linear/diverging colormaps ([`palettes.py`](src/utils/plotting/palettes.py)), and provides
`show_plot_and_save(...)` for displaying + persisting a figure with a templated filename. Importing
`src.utils.plotting.palettes` also sets the default axis colour cycle. Requires `matplotlib`.

### Individual Stages
Each stage should be thought of as an independent command-line script that does one "block" of the pipeline and stores 
the results to disk; it is highly recommended to save all outputs in an `outputs` directory.

To wrap the function defined in an appropriate stage file under `src/stages` into a CLI, it is highly recommended to use
the `fire` package.

Notice that each stage is run from within the `scr/stages` folder, so it should handle imports accordingly;
it is recommended to always include the statement:
```python
from __init__ import root_path
```
before importing anything else from within the `src` directory.
See an example below.

```python
# stage "do something": src/stages/do_something.py
from fire import Fire

from __init__ import root_path
# === other src imports go below this line ===
from src.something import some_function

def do_something():
    ...


if __name__ == '__main__':
    Fire(do_something)
```
> Notice that PyCharm will often interpret these imports as an error, so do not panic if you see them highlighted 
> in red.

### Running a Pipeline

The [`run_project.py`](run_project.py) script parses a `YAML` file in the correct format, launches an MLflow 
orchestrator runner and each stage as a further runner (all locally by default) based on the provided options.
Anything that can run as a command-line python script with the provided options as command-line arguments can be
executed, but it is highly recommended to use `fire` to handle wrapping.

Once you are ready to launch a pipeline, you can run `run_project.py` from the root of the directory with two inputs:
1. the path of the `YAML` script of the pipeline;
2. an experiment name, so that MLflow can save all similar runs under one name.

For example, to run `my_pipeline.yaml` (it is recommended to always keep pipelines under the `config` 
dir.) from a terminal, go to the root of the repository and execute:
```shell
run_project.py config/my_pipeline.yaml AN_EXPERIMENT_NAME
```

> It is highly recommended to only run pipelines after committing the changes to the code, since MLflow also saves
> the commit hash for full reproducibility.

At start-up the orchestrator prints a welcome banner with the repository name (derived from the project directory,
so it auto-rebrands when you instantiate the template) and the version read from the pipeline's `tags.version`,
plus an optional one-line `description` from the pipeline YAML. The ASCII art is produced by a small, dependency-free
FIGlet renderer ([`src/utils/banner.py`](src/utils/banner.py) + the bundled
[`data/fonts/small.flf`](data/fonts/small.flf), whose smushing logic is adapted from the
[FIGfont spec](http://www.jave.de/figlet/figfont.html) and the [`pyfiglet`](https://github.com/pwaller/pyfiglet)
reference implementation); the name and subtitle are length-capped so the banner never overextends horizontally,
and it degrades gracefully to a plain box if the font cannot be loaded.

- **Disable it** with `banner: false` at the top of the pipeline YAML.
- **Customize the font:** `pip install pyfiglet` for greater customizability, then set `banner_font: <name>` to any of
  the hundreds of FIGlet fonts pyfiglet ships (without pyfiglet only the bundled `small` font is available).

```
╭────────────────────────────────────────────────╮
│        _            _                          │
│   _ __| |_  _ _ __ | |__  ___ _ _              │
│  | '_ \ | || | '  \| '_ \/ -_) '_|             │
│  | .__/_|\_,_|_|_|_|_.__/\___|_|               │
│  |_|                                           │
│                                                │
│  v1.0  ·  A minimalistic ML pipeline template  │
╰────────────────────────────────────────────────╯
```

### Orchestration Backends: MLflow (default) and Prefect

The pipeline can be driven by one of two interchangeable **orchestration backends**, selected with a single
optional top-level key in the pipeline YAML (defaults to `mlflow`, i.e. the historical behaviour):

```yaml
orchestrator: mlflow   # default — sequential loop, no extra dependencies
# orchestrator: prefect  # opt-in — same execution, wrapped in a Prefect flow for a UI / retries / remote submission
```

Prefect is an **optional dependency**: if `orchestrator: prefect` is selected but Prefect is not installed, the
orchestrator logs a warning and **falls back to the `mlflow` backend** rather than failing — so a pipeline never
breaks just because the optional dependency is missing. (Install Prefect via the
[`environment.yml`](environment.yml) below, or `pip install prefect`.)

Both backends run the **exact same per-stage logic** — the same subprocess dispatch, the same MLflow logging, the
same deterministic seeding, and crucially the **same lazy cache and determinism guard** described above. The shared
core is the `execute_stage(...)` function in [`src/stages/run.py`](src/stages/run.py); the MLflow backend calls it in
a `for` loop, the Prefect backend wraps it in a `@task`. Switching backends does **not** change results — a lazy hit
under `mlflow` is a lazy hit under `prefect`, byte-for-byte.

#### Why the cache stays in MLflow (and not in Prefect)

Prefect ships its own task caching (`@task(cache_key_fn=task_input_hash)`), but it is **not** a drop-in for this
template's strategy and is deliberately **not** used. Prefect's cache hashes a task's *Python call arguments*; it has
no notion of the whole-repo git code state, it does not content-fingerprint files on disk (a path string is just a
string to it), it does not verify that a stage's outputs are still present and byte-identical, and it persists the
task's *return value* rather than your on-disk artifacts. The template's cache — keyed on
*code state + parameters + input-file fingerprints*, with an output-presence/byte-identity check and a determinism
guard, all stored as MLflow run tags (see [`src/utils/io/lazy.py`](src/utils/io/lazy.py)) — is strictly stronger. The
Prefect backend therefore reuses that existing logic unchanged; Prefect is only a *scheduling shell*.

#### Two databases, cross-linked — not merged

A natural question is whether Prefect can reuse the same `mlflow.db` so there is only one store. The answer is **no,
and it should not** — but the overhead of having both is negligible:

| Store | Holds | Backend |
|---|---|---|
| `mlflow.db` (+ `mlruns/` artifacts) | experiments, runs, params, metrics, **and the lazy cache tags** | SQLite, managed by MLflow's migrations |
| `prefect.db` (default `~/.prefect/`) | flow runs, task runs, states, schedules, logs | SQLite, managed by Prefect's migrations |

The two have **incompatible schemas and independent migration chains**, so they cannot share one physical file —
pointing Prefect at `mlflow.db` would have each tool's migrations clobber the other's tables. But Prefect's metadata
DB is an *embedded SQLite file with no server process* (exactly like your MLflow SQLite backend), so adopting Prefect
adds **one lightweight `.db` file, not another daemon** (and `**/*.db` is already git-ignored).

What *is* shared is the **linkage, not the storage**: the Prefect flow opens one MLflow run, then each side is tagged
with the other's id (MLflow run carries the Prefect `flow_run_id`; the Prefect flow records the MLflow `run_id`), so
you can click straight from one UI to the other. MLflow remains the single source of truth for *experiments*; Prefect
owns *execution*.

#### Two UIs, two jobs

- **MLflow UI** (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) — the *experiment* view: parameters, metrics,
  artifacts, run comparison. This is where your "trials" live.
- **Prefect UI** (`pip install prefect`, then `prefect server start` → `localhost:4200`) — the *execution* view: which
  stages ran/failed/retried, the run timeline, durations, and logs. Flow runs are recorded to the local Prefect SQLite
  even without the server running; the server is only needed to *see* the UI.

The Prefect frontend does **not** display your metrics or model artifacts — for that you cross-link to MLflow. Its
value is execution observability, retries, scheduling, and (on a cluster) remote submission.

#### Concurrency and HPC (designed for, not yet wired)

The Prefect backend currently executes stages **linearly**, matching the MLflow backend exactly. Two extensions are
scaffolded with `TODO` markers in [`src/stages/run.py`](src/stages/run.py) so they have an obvious home:

- **Concurrent DAG execution.** Independent stages could run in parallel via a Prefect task runner. Because stages
  here communicate through *files on disk* rather than return values, the dependency edges must be made explicit —
  either declared (a `depends_on: [stage]` key in the YAML, stripped before the stage CLI like `lazy:`) or inferred
  (an edge wherever one stage's input path equals another's `output-path`). Parallelism is only safe across stages
  with no input/output overlap, and requires MLflow logging via the `MlflowClient` with explicit `run_id`s (the
  fluent `mlflow.start_run()` global is not thread-safe) plus a non-global per-stage seed.
- **Cluster execution.** Two levels, both via Prefect: (1) a `prefect worker` running inside an LSF/SLURM job picks
  up *whole-flow* deployments submitted from elsewhere; (2) a `DaskTaskRunner`/`RayTaskRunner` backed by
  `dask-jobqueue`'s `LSFCluster`/`SLURMCluster` turns *each stage* into its own resource-tailored cluster job. Either
  way the prerequisites are: stage outputs on a **shared filesystem** the orchestrator can `stat`, an MLflow store
  **reachable from every node** (the SQLite `mlflow.db` behind a `mlflow server`, or a shared-FS file store), and a
  Prefect API/DB reachable by the workers.

## Miscellaneous

1. [`git lfs`](https://git-lfs.com/) will not handle large files if you install it and track them _after_ committing
them, so beware!