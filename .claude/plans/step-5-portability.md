# Step 5 (block e) — Portability: user-agnostic & machine-agnostic

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 4](step-4-stages.md)

> **Status: provisional — to be expanded** once [Step 4](step-4-stages.md) is done.

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

### Still open for this step

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

## 🐛 Only ONE module under `src/utils/` can be heard (found by the block 4f gate)

`src/__init__.py` builds `console_handler`, and **every stage attaches it to its own logger only** — plus
`lazy.logger` explicitly in `run.py` and, uniquely in the library, `src/utils/modeling/tuning.py` at module level.
Nothing attaches a handler to the `src` package logger and nothing configures the root logger, so **every
`logger.info` / `logger.warning` in every other `src/utils` module is discarded during a pipeline run.**

Found by chasing a log line that never appeared (`reporting`'s "summed N hourly items into D days"). What else is
silently lost, in rough order of how much it matters:

* ⚠️ `evaluation.run_metric_suite`'s **degenerate-configuration warning** — the one naming `kind: probability` when a
  shared occurrence cut meets a probability field. Three config files, a docstring and this plan all describe it as
  the runtime guard behind a bug that produces an entire contingency table of nonsense without raising. It has never
  been audible. The DAILY instance of that same bug was caught by a human reading a confusion matrix, which is
  precisely what the guard was supposed to prevent.
* `search.apply_constraints`' two decision logs — "forcing `loss.intensity_weight_gamma` to 0" and "the sampled
  `unet` block is ignored for this trial". Both record a **change to the trial** that the trials table does not show.
* `reporting`'s "Requested plot date is not in the evaluated split; skipped" — a figure the user asked for, absent.
* everything in `data.py`, `dataset.py`, `registry.py`, `validation.py`, `scores.py`, `diagnostics.py`, `maps.py`.

**The fix is small but not local:** attach `console_handler` once to `logging.getLogger('src')` in `src/__init__.py`,
then REMOVE the eight per-stage `addHandler` calls plus `tuning.py`'s and `run.py`'s `lazy.logger` one — otherwise
every stage record is emitted twice, once by its own handler and once by the ancestor's. Ten files for a two-line
idea, which is why it belongs to this step rather than to a block doing something else.

⚠️ Until it lands, treat *"the code warns about X"* as false wherever X lives in `src/utils`: the record is emitted
and then dropped. Every such claim in a config or docstring is aspirational, and the configs that make it now say so.

## Open question carried from the architecture inventory

**Launch scripts** — port the three-script-per-machine pattern (`*.sh`), replace it with the per-user config file,
or both? (Question 6 in [`inventory-architecture.md`](inventory-architecture.md), still unanswered.)
