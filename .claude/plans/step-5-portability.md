# Step 5 (block e) — Portability: user-agnostic & machine-agnostic

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 4](step-4-stages.md)

> **Status: 🔵 IN PROGRESS** (opened 2026-08-20). [Step 4](step-4-stages.md) is ✅ done at `ef19af0`.
>
> **The goal is one sentence:** clone the repo on another remote, install the minimal requirements, launch the
> pipeline. Everything below is either something that blocks that or something that makes it silently wrong.

This repo is meant to be used on other machines or remote servers and by other users, so the paths baked in
during Step 0 (`/homedata/aburq/.venvs/lightning-stochastic-modeling`, the `/homedata/aburq/batta_torch` data
root) will not be applicable to other contexts — this is the final step, once the pipelines themselves are
merged and stable. It adds a local, per-user config file where each user specifies their own paths (data root,
venv/output locations, etc.), so nothing user- or machine-specific is hardcoded in the pipeline YAMLs or code.
It also broadens installation beyond pip: `minimal_requirements.txt` currently assumes a plain `venv` + `pip`
workflow; this step adds equivalent install paths for `uv` and `conda` so the repo isn't tied to one tool.

**Build on the existing mechanism, don't invent one** (Step 1 finding #4): the configs already read a
**`DATA_ROOT`** environment variable, set by one launch script per machine — proven across three environments
(HPC + conda, Jean Zay + venv, local + venv). Formalise and document that, and let the per-user config file
supply `DATA_ROOT` plus the venv/output locations. Two concrete carry-overs:

- the launch scripts' commented-out `LD_LIBRARY_PATH="${CONDA_PREFIX}/lib..."` workaround (node `libstdc++`
  older than the compiled extensions need — `GLIBCXX_3.4.29`, gcc ≥ 11) is a known wart worth documenting;
- **cartopy downloads ~14 MB of Natural Earth shapefiles at *plot* time** into `~/.local/share/cartopy`. That is
  a network call from wherever plotting runs, so it fails on an offline compute node; the cache path is also
  user-specific. Needs a documented pre-warm step. *(Now unavoidable: cartopy is a hard requirement — the
  lazy-import fallback was dropped in Step 1.)*

## ✅ DONE in Step 4 block 4c-r — outputs live behind `{{$OUTPUT_ROOT}}`

Every written path in all 11 pipeline configs now resolves under `{{$OUTPUT_ROOT}}` — smoke tiers included, so there
is one rule rather than a rule with an exception. Done here rather than in Step 5 because the full dataset does not
fit in a checkout and the first real run cannot wait: measured against the real grid (101 x 149, 5843 days, 5
predictors x 24 h, float16 features), `features/` alone is **19.7 GiB per split**, so a daily prepared directory is
~20 GiB. `tuning/` adds a checkpoint per retained trial plus the optuna journal.

Three things landed with it:

1. **The `setup` guard.** `parse_config` maps an unset variable to the EMPTY STRING, so `'{{$OUTPUT_ROOT}}/family/prepared'`
   becomes `/family/prepared` — absolute, at the filesystem root — and `os.path.join(root_path, …)` then discards the
   repo root entirely. `setup.looks_like_an_unset_root` catches it before anything expensive, discriminating on the
   TOP-LEVEL segment (a real absolute path sits under an existing mount; an unset variable leaves a first segment that
   is a family name and does not exist). It lives in `setup` and **not** in `parse_config` because `UPSTREAM_MODEL`
   *relies* on the empty-string behaviour — unset means "no warm start" to both `tune` and `prepare_modeling`, so a
   blanket raise would break both stochastic families.
2. **The two U-net families now SHARE one prepared directory**, `$OUTPUT_ROOT/deterministic_and_mc_dropout/prepared`.
   Their `prepare_modeling` blocks differed in `output-path` alone, so the second pipeline to run now skips
   preparation entirely — ~20 GiB and one full pass saved. ⚠️ Diffusion does **not** share it and must not: in
   residual mode its preparation writes `upstream/` maps and flips `residual_target`, and the dataset keys the 6th
   conditioning channel on that flag, so the U-net checkpoints would mismatch on `in_channels`. In full-target mode
   the directory *would* be identical, which is what makes sharing a trap rather than an optimisation — it would work
   until the first residual run.
3. **Two regression guards in `parse_config_test.py`**: every output-tree path is rooted at the marker (the half-moved
   sweep is the failure mode of a 70-path edit), and the two families sharing a directory pass identical prepare
   parameters.

⚠️ **Corrections to what this file said before.** Two of the four hazards it listed were wrong:

* *"the half-moved-pipeline case"* — already guarded, and not by anything new:
  `parse_config_test.py::test_every_consumer_reads_the_leaf_its_own_pipeline_PRODUCES` has walked every pipeline since
  block 5a asserting each `input-path` equals its own file's `prepare_*` `output-path`. It catches the case whether
  `OUTPUT_ROOT` is set or not. The note also mis-attributed the harm to `prepare_modeling` "re-preparing into an empty
  directory", which is just the stage doing its job — the harm is entirely on the READING side, where `tune` trains on
  a stale directory from an earlier run.
* *"the 2048 MB dir budget"* — real, but the risk it implies is already covered. A 20 GiB prepared directory exceeds
  `lazy_content_max_dir_mb` (default 2048), so `lazy.fingerprint_path` switches to size-only mode: the cache key holds
  each file's path and byte SIZE, not its contents, and a same-size rewrite is invisible to the cache. That is the
  right trade — content-hashing 20 GiB per stage would dominate the run — and the specific danger (re-preparing under
  a different `hourly-threshold`) is caught from the other side by `prepare_modeling`'s own staleness check, which
  RAISES on a `mode` / `hourly-threshold` mismatch. The two mechanisms cover each other.

## The survey — what a fresh clone actually hits (measured 2026-08-20)

Ranked by whether it BLOCKS clone → install → launch, because that is this step's goal in one line.

| # | Defect | Where | Blocks? |
|---|---|---|---|
| 1 | `mlflow.projects` builds the stage command from a hardcoded literal `"python"`, so **every stage subprocess resolves its interpreter from `PATH`** | `run_project.py:41` + `src/stages/run.py:150` | ❌ **yes** |
| 2 | cartopy fetches its coastline shapefile at **plot** time | `plotting/maps.py:139` | ❌ **yes**, offline |
| 3 | No per-user config: the env vars must be exported by hand and nothing names them | — | ⚠️ `setup` catches `OUTPUT_ROOT` only |
| 4 | Install docs are pip-only and hardcode `/homedata/aburq/.venvs/...` + `module load python/meso-3.11` | `minimal_requirements.txt:4-6` | ⚠️ docs |
| 5 | `output.log`, `mlruns/`, `mlflow.db` are written **into the checkout** | `src/__init__.py:8`, mlflow default | ⚠️ grows |
| 6 | 590 `src/utils/*` records reach `output.log` and nothing else | 11 files | ergonomics |
| 7 | A fresh clone has **no launch material at all** — `job_scripts/` is gitignored | — | ⚠️ |
| 8 | `split_index.csv`'s `file` column is absolute into `$DATA_ROOT` | `prepare_modeling.py:422` | **DEFERRED** — see below |

Measured facts that shrank three of these, each recorded so nobody re-hunts them:

* **cartopy needs ONE file set, 1,128,501 bytes.** Instrumenting `shapereader.natural_earth` and rendering a real
  figure through `frame_map_axis` at the EuroPP extent requests exactly `('50m', 'physical', 'coastline')`. The ~14 MB
  in a warm `~/.local/share/cartopy` is other projects' leftovers — 10m coastline, admin boundary lines, land, ocean —
  none of which this repo ever asks for. `resolution='auto'` picked 50m from the extent; a future extent change could
  pick differently, and cartopy falls back to its writable cache when a file is absent, so bundling one resolution is
  safe rather than brittle.
* **`mlruns/` is only HALF the store.** Its run directories hold `artifacts/` and nothing else — no `meta.yaml`, no
  `params/`, no `tags/`. The metadata is in **`mlflow.db`** (sqlite, repo root, 1.7 MB): 5 experiments, 54 runs, 534
  params, 1798 metrics and **519 tags**, and the tags are where the lazy cache keys live. So a run's provenance and its
  figures sit in two different gitignored files in the checkout. `job_scripts/pipeline.sh:58` already exports
  `MLFLOW_TRACKING_URI=file:${OUTPUT_ROOT}/mlruns`, so sbatch is portable and `python run_project.py` is not — that
  asymmetry is what `.env` closes.
* **`git lfs` was NOT needed — and this step makes it needed.** `git lfs ls-files` is empty and neither tracked binary
  matches a filter in `.gitattributes`, so the README's "also install git lfs" was a `plumber` leftover. Bundling the
  coastline (decision 5) makes it true. ⚠️ **git-lfs is not installed on this host**; `module load git-lfs` gives
  2.11.0, which is enough to add and push. Consequence to guard: a clone WITHOUT git-lfs gets a ~130-byte pointer file
  where a shapefile should be, and plotting fails obscurely — so `preflight` must detect the pointer.

**No hardcoded data path in `src/` or `config/`.** `git grep` over the tracked tree finds `/homedata` only in doc
comments and the requirements header. The `{{$VAR}}` mechanism held; this step formalises it rather than replacing it.

## Decisions taken (2026-08-20, with the user)

1. **`.env` at the repo root, loaded in `src/__init__.py`.** Tracked `.env.example`, gitignored `.env`. Loaded at the
   earliest import so it reaches `run_project.py`, every standalone stage, `pytest` and `scripts/*` from ONE place, and
   **never overrides an already-set variable** — so a shell export or a slurm-inherited environment still wins. This
   makes `src/__init__.py` explicitly the *process bootstrap* (`sys.path`, `.env`, `PATH`, `CARTOPY_DATA_DIR`, logging)
   rather than an accidental one; it already mutated `sys.path`, the same class of side effect.
   Contents are small — the config interpolates exactly three variables (`DATA_ROOT` ×11, `OUTPUT_ROOT` ×261,
   `UPSTREAM_MODEL` ×6) plus mlflow's own `MLFLOW_TRACKING_URI`. `UPSTREAM_MODEL` ships commented out: it is a
   per-RUN checkpoint path, not per-machine config. `PIPELINE_SEED` must NEVER appear there — the orchestrator owns it.
   ⚠️ Consequence to watch: the four opt-in real-artifact tests in `evaluate_test.py` are gated on `OUTPUT_ROOT`, so a
   local `.env` turns them from skipped into live. That is the intent, but it moves the skip count.
2. **`output.log` is dropped**, console becomes the only sink. See the section below for what was in it.
3. **One dependency list.** `environment.yml` becomes a real minimal recipe (`python=3.11`, `pip`, then
   `-r minimal_requirements.txt`) so conda supplies only the interpreter; `uv` needs no file at all. Three documented
   install paths, one list, so they cannot drift — which is the actual failure mode of "add conda support".
4. **Track a generic `job_scripts.example/`.** ⚠️ Overrides the recommendation, which was to document
   `python run_project.py` only. Kept because `_common.sh`'s repo-root search encodes real debugging
   (`SLURM_SUBMIT_DIR` is where you ran `sbatch` FROM, and `sbatch` COPIES the script into `/var/spool/slurm/...`, so
   neither obvious guess works). It stays **slurm** — that is expected; what gets sanitised is the site-specific part:
   `--partition=zen16` (×7), the `--mem` 8G–128G / `--time` 15 min–3 days grid, and the `/homedata/aburq/...`
   fallbacks become documented placeholders. Cost accepted: a second copy that can drift from the gitignored one
   actually run.
5. **Bundle the coastline as PLAIN blobs**, with a `data/cartopy/shapefiles/** -filter -diff -merge` exemption in
   `.gitattributes`. ⚠️ **Decided three times**, and the reversals are worth recording because the third decision was
   forced by a measurement rather than a preference:
   1. recommended plain; the user chose **git-lfs**, as `.gitattributes` already dictates (`*.shp filter=lfs`);
   2. the `pip install git-lfs` question surfaced that the PyPI package is fetch-only and that pip runs after clone —
      raised, and the user **reaffirmed LFS**;
   3. building it broke the repo. `git lfs install` (which committing an LFS object requires) sets
      `filter.lfs.required = true`, after which **every git command exits 128 without the binary** —
      `git status --porcelain` measured at 128 in a plain shell — *including* the `git diff HEAD` that
      `lazy.code_state_hash` runs. That one is caught (`check=True` inside a `try`), so the pipeline does not crash: it
      **silently degrades** to hashing `src/` alone, dropping `config/` and the working-tree state from the cache key.
      A hard runtime dependency on every machine plus a quietly weakened cache is not worth 1.1 MB, and on that
      evidence the user chose plain.

   ⚠️ **The exemption is load-bearing**, not tidiness: `*.shp filter=lfs` still applies everywhere else, so deleting
   those two lines lets the next commit made with git-lfs installed convert the shapefile into a ~130-byte pointer —
   after which cartopy dies with `KeyError: 828781878` inside `shapefile.py` (the first bytes of "version" read as a
   shape type), which reads as corrupt data rather than a configuration mistake. Two tests guard it: one asserts
   `git check-attr` reports no filter for the path, the other that the file on disk is not a pointer.
6. **DEFERRED: the `split_index.csv` relative-path fix.** It does not block a fresh clone — if `prepare_modeling` runs
   on the remote, the absolute path it writes is correct and nothing breaks. Known limitation to record instead: **a
   prepared directory is not movable**, so copying ~20 GiB between machines (rather than re-preparing, which costs a
   full pass over 48 GB) fails, as does a `$DATA_ROOT` that moves under a scratch purge or remount. Only the two
   **hourly** tiers are exposed — the nine daily tiers materialise features and never open `file`. The fix, when it is
   wanted, is ~10 lines: write `sample_filename` relative, rejoin in `load_prepared_artifacts`, keep a legacy absolute
   `file` column loading. Because the in-memory column stays `file`, `dataset.py` and `compute_feature_stats` need no
   change at all.

## Blocks

| Block | Contents | Verified by |
|---|---|---|
| `5a` ✅ | **The bootstrap**: `.env` + the interpreter fix + the `CARTOPY_DATA_DIR` default + the logging sink | 29 loader unit tests; `.env.example` ↔ `config/` guards; three pipeline runs with the venv **removed** from `PATH`; the mutation check below |
| `5b` ✅ | Bundle `ne_50m_coastline` (1.1 MB, plain blobs) + `scripts/prewarm_cartopy.py` as the fallback for a changed extent | a cold-process render with the download cache EMPTY and every downloader refusing; both mutation-checked |
| `5c` | ~~Logging~~ — **merged into 5a**: it is three lines of the same 16-line file, and touching `src/__init__.py` twice would be worse than doing both at once | (see 5a) |
| `5d` | Install paths (pip/uv/conda) + `scripts/preflight.py`, including the LFS-pointer check | preflight unit tests; a requirements/conda consistency guard |
| `5e` | Docs + `job_scripts.example/`: the fresh-clone quickstart, `MLFLOW_TRACKING_URI`, the `GLIBCXX` wart | the by-hand gate — the user, on a different remote |

### ✅ 5a as built (2026-08-20)

`src/__init__.py` is now explicitly the **process bootstrap** — `sys.path`, `.env`, `PATH`, `CARTOPY_DATA_DIR`, logging
— with the four helpers in the new `src/utils/io/environment.py`, beside `parse_config.py`, the other module that reads
the environment. **1450 passed, 4 skipped, coverage 87.84 %** (from 1418 / 87.72).

**`sys.executable` is the source of truth; `PATH` is only the channel mlflow insists on using.** Worth stating plainly,
because it is what makes the fix work and what makes the new test honest:

* a process launched by ABSOLUTE interpreter path never consults `PATH` — `subprocess.run([sys.executable, ...])`
  bypasses it entirely — and `sys.executable` is set from how the process was invoked, not by searching `PATH`;
* so the venv location is always recoverable inside the process, whatever `PATH` says. `prepend_interpreter_to_path`
  copies `dirname(sys.executable)` to the front of `os.environ['PATH']`, and every child inherits that;
* measured with the venv stripped from `PATH`: before `import src`, bare `python` resolved to `/usr/bin/python`; after
  it, to the venv's — and a child could then `import mlflow`.

Two things worth keeping:

* **The e2e fixtures now REMOVE the venv from `PATH`** (`_path_without_the_venv`) instead of prepending it. Prepending
  made the runs work while hiding the defect; removing it turns all three pipeline runs into a regression test for the
  fix, and costs nothing because the parent is still launched by absolute path. Mutation-checked — commenting out that
  one call fails `test_the_pipeline_EXITS_ZERO` in 4.7 s with `ModuleNotFoundError: No module named 'mlflow'` surfacing
  as `Run (ID ...) failed`, exactly the misleading "broken stage" signature this defect produces in the wild.
* **`pip install git-lfs` does NOT substitute for the client.** A PyPI package of that name exists — but it is
  `git-lfs-fetch.py`, 5.6 KB of pure Python whose own metadata says it *"cannot fully replace the official git-lfs
  client"* and that *"uploading files is not implemented at all"*. So it can materialise a pointer on a consumer
  machine (as a separate `python -m git_lfs` command, not a git filter) but cannot author one. And the ordering is
  against it regardless: **`pip install` runs after `git clone`**, so the venv does not exist when the pointers would
  need smudging. ⚠️ This weakens decision 5 — the bundle's appeal was zero-setup-works-offline, and LFS adds a setup
  step. Re-confirm before building 5b.

Worth documenting as the FIRST post-install command, before the dataset is even copied:
`pytest tests/pipeline_e2e_test.py --no-cov` already builds a synthetic `$DATA_ROOT` and drives the real pipeline end
to end. It proves the install without the 48 GB.

### ✅ 5b as built (2026-08-20)

`data/cartopy/shapefiles/natural_earth/physical/ne_50m_coastline.{shp,shx,dbf,prj,cpg}` — cartopy's own layout, so
`CARTOPY_DATA_DIR` (set by the bootstrap) is consulted as `pre_existing_data_dir` before any download is attempted.

**The measurement that sized this**: instrumenting `shapereader.natural_earth` and rendering one figure through
`frame_map_axis` at the real EuroPP extent requests exactly `[('50m', 'physical', 'coastline')]` — one dataset,
1,128,501 bytes. A warm `~/.local/share/cartopy` may hold ~14 MB (10m coastline, admin boundary lines, land, ocean),
none of which this repo asks for. `coastlines()` uses `resolution='auto'`, so the requirement is a property of the
FIGURE, not the code — which is why both the script and the test discover it by rendering rather than by listing
filenames.

⚠️ **cartopy reads `CARTOPY_DATA_DIR` once, at ITS import**, into `config['pre_existing_data_dir']`. Setting it later is
silently ineffective. Normal order is safe (`src.utils.plotting.maps` cannot load without `src/__init__.py` running
first), and `use_bundled_cartopy_data` repairs the already-imported case through `sys.modules` rather than importing
cartopy itself.

⚠️ **cartopy MEMOISES resolved feature geometries**, so `natural_earth()` is called only on the first draw in a process.
That made the first version of both tests order-dependent — and would have let the offline assertion pass VACUOUSLY,
since with geometries already in memory no download is attempted whatever the bundle holds. Both now run in a
subprocess with `XDG_DATA_HOME` pointed at an empty directory, which is the only place either question has an answer.

Mutation-checked twice, both failing for the right reason:

* bundle moved away → 4 tests fail with `cartopy attempted a DOWNLOAD`;
* exemption removed from `.gitattributes` → `git check-attr` reports `filter: lfs` and the guard fires.

One flaw found and fixed in the script itself: `unpulled_pointers()` originally ran AFTER `requested_datasets()`, so
with a pointer in place the script crashed with cartopy's `KeyError: 828781878` instead of reporting the pointer —
diagnosing after the render means never diagnosing at all.

### Detail on the open items

* ⚠️ **`mlflow.projects` shells out to a BARE `python`, not `sys.executable`.** Found in Step 4 block 4e, the first
  thing the end-to-end test hit. `run_project.py` calls `mlflow.projects.run(...)`, and with no `MLproject` file
  MLflow's local backend builds the command `python src/stages/<stage>.py --...`. So **every stage subprocess resolves
  its interpreter from `PATH`**, not from the interpreter that launched the pipeline. Consequences:
  - it works today only because the launch scripts activate the venv first — which is precisely the machine-specific
    step this step exists to remove;
  - a job script that module-loads a system python, or any invocation by absolute interpreter path
    (`/path/to/venv/bin/python run_project.py ...`), silently gets the wrong interpreter and dies on `import mlflow`
    inside the stage. The traceback names the stage, so it reads as a broken stage rather than a broken environment.

  `tests/pipeline_e2e_test.py` works around it by prepending `os.path.dirname(sys.executable)` to the subprocess
  `PATH`. That is right for a test and is not the fix for a user: the options are an `MLproject` file whose entry-point
  commands are explicit about the interpreter, or documenting activation as a hard prerequisite of `run_project.py`.
  Decide it here.
* **`mlruns/` is not covered.** It is created next to `run_project.py`, holds the lazy cache's tags and every run's
  logged artifacts, and grows without bound. `run_project.py` already accepts a `tracking_uri`, so the choice is
  between pointing that at the new root and keeping the store local by design. ✅ **Confirmed in 4e that
  `MLFLOW_TRACKING_URI=file:/some/path` relocates the whole store** — the e2e test sets it so a test run never writes
  into the checkout, which means the per-user config file can supply it without any code change.
* **`OUTPUT_ROOT` and `DATA_ROOT` should come from the per-user config file** this step introduces, rather than from
  whatever exported them. The `setup` guard makes a missing `OUTPUT_ROOT` loud; it does nothing for a missing
  `DATA_ROOT`, which still fails inside `prepare_modeling`.
* **`.gitignore` still lists `/outputs/`** — harmless, and still correct for a local run with `OUTPUT_ROOT=outputs`.

## Carried in from Step 4

⚠️ **`split_index.csv`'s `file` column is an ABSOLUTE path into `$DATA_ROOT`, baked in at prepare time.** Found while
writing `prepare_modeling` (block 4a). Every other path in a prepared directory is a bare filename that
`load_prepared_artifacts` joins at read time — which is what makes a prepared directory movable — but `file` points
back at the source `samples/*.pt`, so it survives a move of the *outputs* and breaks on a move of the *dataset*.

⛔ **NO LONGER DORMANT — block 4f woke it up.** The paragraph below used to say "every shipped config sets
`materialize-features: true`, so `file` is never opened". That stopped being true when the two **hourly** tiers
shipped with `materialize-features: false` (deliberately: materialising would cost a second ~20 GiB, and it is what
turns on `DayGroupedShuffleSampler`). So the fallback reader is now on the shipped path, and with it the absolute
`file` column:

* it is read by `LightningMapsDataset._item_features_checkpoint`, the **fallback** feature reader — which the hourly
  pipeline uses for every item of every epoch;
* the nine daily tiers still materialise, so `file` is never opened there;
* **the hourly pipeline is therefore not machine-portable in the way the daily ones are.** It works today only
  because preparation and training happen on the same host with the same `$DATA_ROOT`. Prepare on one machine and
  train on another and it fails with a `FileNotFoundError` naming a path that exists on neither, which reads as a data
  problem rather than a portability one.

This raises the priority: it is a live defect on a shipped pipeline rather than a hazard waiting for a configuration
nobody uses.

Options for this step: store it relative to `data_path` and rejoin in `load_prepared_artifacts` (symmetric with the
other three columns, and the obvious fix), or drop the column entirely and require materialised features. ⛔ **The
second option is now closed**: the hourly tiers depend on the fallback path, so removing it would either double their
on-disk footprint or delete the pipeline. Take the relative-path fix.

---

## 🐛 Every `src/utils/` diagnostic goes to `output.log` and nowhere a user looks

⚠️ **CORRECTED 2026-08-19.** This section first said those records were "discarded". They are not — they are written
to a file, and the difference matters because it changes the fix. Measured, not inferred:

```python
src/__init__.py:8   logging.basicConfig(filename=os.path.join(root_path, 'output.log'), level=logging.INFO)
```

So the **root** logger owns a `FileHandler` at INFO. Then:

| logger | handlers | where its records go |
|---|---|---|
| `src.stages.<stage>` | `console_handler` (a `StreamHandler` → stderr), attached in each stage | **the console** *and*, by propagation to root, `output.log` |
| `src.utils.modeling.tuning` | `console_handler`, attached at module level — the only library module that does | the console *and* `output.log` |
| every other `src.utils.*` | none | **`output.log` ONLY** |

Verified by emitting a probe from `src.utils.metrics.evaluation` with the stage preamble loaded: nothing on stderr,
both records present in `output.log`. `reporting`'s "summed N hourly items into D days" — the line whose absence started
this — appears there **7 times**.

`output.log` sits at the repo root, is gitignored (`/output.log`), is currently **427 KB**, is appended to by every
process including concurrent stage subprocesses with no run separation or rotation, and **nothing in `CLAUDE.md`,
`README.md` or `job_scripts/README.md` mentions it exists**. A diagnostic you must already know to look for, in an
undocumented shared file, is not far from having none — but the record is there, which makes this an ergonomics defect
rather than a data-loss one.

What is affected, in rough order of how much it matters:

* ⚠️ `evaluation.run_metric_suite`'s **degenerate-configuration warning** — the one naming `kind: probability` when a
  shared occurrence cut meets a probability field. Three config files and a docstring describe it as the runtime guard
  behind a bug that produces an entire contingency table of nonsense without raising. It fires, into `output.log`. The
  DAILY instance of that same bug was caught by a human reading a confusion matrix, which is precisely what the guard
  was supposed to prevent — and it would not have been prevented, because nobody was reading that file.
* `search.apply_constraints`' two decision logs — "forcing `loss.intensity_weight_gamma` to 0" and "the sampled
  `unet` block is ignored for this trial". Both record a **change to the trial** that the trials table does not show.
* `reporting`'s "Requested plot date is not in the evaluated split; skipped" — a figure the user asked for, absent.
* everything in `data.py`, `dataset.py`, `registry.py`, `validation.py`, `scores.py`, `diagnostics.py`, `maps.py`.

## ✅ FIXED in block 5a — and the prescribed fix above was WRONG about scope

⚠️ **CORRECTED AGAIN 2026-08-20.** This section previously said the fix was *"attach `console_handler` once to
`logging.getLogger('src')`, then REMOVE the eight per-stage `addHandler` calls plus `tuning.py`'s and `run.py`'s
`lazy.logger` one … Ten files for a two-line idea."* **The eight stage calls must STAY.** A stage runs as
`python src/stages/<stage>.py`, so its `__name__` is **`__main__`**, not `src.stages.<stage>` — which is exactly why
`output.log` shows `INFO:__main__:` 551 times and `INFO:__init__:` 404 times. Those loggers sit OUTSIDE the `src.`
hierarchy, so the ancestor handler never reaches them and removing their own would have left every stage silent on the
console. The same applies to `run_project.py` (`__main__`) and `src/stages/__init__.py`'s seeding logger (`__init__`).

So the real fix was **three files, not ten**:

| | change |
|---|---|
| `src/__init__.py` | drop `basicConfig(filename=...)`; give `logging.getLogger('src')` the console handler **and** `setLevel(INFO)` — the library loggers are at `NOTSET`, so the level had to move with the handler or every `logger.info` would have gone silent |
| `src/utils/modeling/tuning.py` | remove its `addHandler` + `setLevel` (it is `src.utils.modeling.tuning`, so the ancestor now covers it) and the now-unused `console_handler` import — it was the ONLY library module with its own handler, which is why its diagnostics were the only library ones a user ever saw |
| `src/stages/run.py` | remove `lazy.logger.addHandler(...)` (`src.utils.io.lazy`, covered by the ancestor); **keep** its own `__main__` handler |

Verified after: a `src.utils.metrics.evaluation` record reaches stderr **exactly once**, `logging.getLogger('src')` has
one handler, the root logger has **zero**, and `output.log`'s mtime does not move. Third-party `lightning` /
`matplotlib` records (1,364 of them) are no longer collected anywhere — accepted, since lightning prints its own
progress and root now keeps its default WARNING level.

`output.log` itself is **dropped** (decision 2). The 457 KB file at the repo root is left in place and still
gitignored, so an existing checkout keeps it to read; nothing appends to it any more. Untracking it instead would make
it appear in `git ls-files --others`, which the lazy cache keys on — so the ignore line stays, marked legacy.

⚠️ **Caveat on the "it fires" claim**: `src.utils.*` loggers are at `NOTSET`, so their effective level is inherited
from root — which `basicConfig(level=INFO)` sets to INFO. Remove or change that call and every library `logger.info`
becomes invisible everywhere, since with no handler found Python's `lastResort` only emits WARNING and above. The two
halves of this defect are therefore coupled: fix the handler, and check the level.

⚠️ Until it lands, treat *"the code warns about X"* as false wherever X lives in `src/utils`: the record is emitted
and then dropped. Every such claim in a config or docstring is aspirational, and the configs that make it now say so.

## Open question carried from the architecture inventory

**Launch scripts** — port the three-script-per-machine pattern (`*.sh`), replace it with the per-user config file,
or both? (Question 6 in [`inventory-architecture.md`](inventory-architecture.md), still unanswered.)
