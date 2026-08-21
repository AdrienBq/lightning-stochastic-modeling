# `job_scripts.example/` — slurm launchers, site-agnostic

Tracked, unlike the `job_scripts/` these were derived from, which is gitignored. Everything machine-specific has been
removed: no venv path, no partition, no data roots. Run them in place, or copy the directory to `job_scripts/` and edit
freely — that copy is ignored, so it is the right place for machine-specific defaults.

```shell
cp .env.example .env                       # then edit DATA_ROOT and OUTPUT_ROOT
python scripts/preflight.py                # check the environment before queueing anything
sbatch job_scripts.example/pipeline.sh     # from the REPO ROOT
```

`_common.sh` sources `.env` itself (`set -a`, so it exports), which is why the same file configures both the Python
side and these scripts. An already-exported value still wins, matching the Python loader.

## Where things come from

| | source | if absent |
|---|---|---|
| `DATA_ROOT`, `OUTPUT_ROOT` | `.env`, or the environment | **hard exit**, naming `.env.example`. No fallback values, deliberately — a wrong-but-plausible default is worse than a missing one, because the run succeeds against the wrong data |
| the interpreter | `$PYTHON`, else `python` on `PATH` | hard exit. Activate your environment before `sbatch`: slurm hands the job the submitting shell's environment, `PATH` included |
| `MLFLOW_TRACKING_URI` | `.env`, else `file:$OUTPUT_ROOT/mlruns` | defaulted, so the store never lands in the checkout |
| the partition | the cluster default | `#SBATCH --partition=` is commented out in every script; uncomment it if your cluster needs one (`sinfo` lists them) |

## Logs — two of them, and they go to different places

* **slurm's own** `--output`/`--error` name **no directory**, so they land in whatever directory you submitted from.
  ⚠️ This is a deliberate change from the `job_scripts/` originals, which used
  `--output=job_scripts/logs/output/%x_%j.out`. slurm resolves that against the SUBMIT directory, so submitting from
  inside `job_scripts/` asked for `job_scripts/job_scripts/logs/`, which slurm could not create — killing the job in
  one second **with no log at all to say why**. Naming no directory means you always get a log, wherever you stand.
* **the script's own transcript**, `$OUTPUT_ROOT/logs/<stage>_<family>_<mode><tier>_<timestamp>.log`, written by
  `tee`. Under `$OUTPUT_ROOT` like every other output, so nothing accumulates in the checkout.

## Use

Edit the `EDIT ME` block of one script, then submit **from the repo root**. `FAMILY`, `TIER` and `MODE` derive every
path, so switching family, tier or task is a one-line change.

| Script | What it runs | Rough cost (smoke / full) |
|---|---|---|
| `prepare_modeling.sh` | tensorise predictors + target | 10 s / hours, ~20 GiB per prepared dir |
| `tune.sh` | the sweep | minutes / days |
| `retrain_best.sh` | refit the winning trial | minutes / hours |
| `evaluate.sh` | metrics JSON + report | 15 s deterministic, 22 min diffusion (CPU) |
| `tabulate_metrics.sh` | the comparison CSV | seconds |
| `combine_curves.sh` | the overlaid figures | seconds |
| `pipeline.sh` | **all stages**, via `run_project.py` | the sum of the above |
| `_common.sh` | sourced by the others; never submit it | — |

## Per-stage scripts vs `pipeline.sh`

The per-stage scripts call `src/stages/<stage>.py` **directly**. That is what makes them useful for re-running one
thing — regenerating a report, re-sweeping — and it means: no mlflow run, no lazy cache, no orchestration. Metrics
land only in `--metrics-path`.

`pipeline.sh` runs `run_project.py` on a shipped YAML: stages in order, an mlflow run each with parameters and metrics
logged, configs attached as artifacts, and the lazy cache. Use it for a run you intend to trust.
⚠️ **Commit first** — the lazy cache keys on the whole-repo dirty diff.

## `MODE` — the one variable that switches the task

`MODE=daily | hourly` in a stage script's EDIT ME block. It is the same key `prepare_modeling` takes, and `_common.sh`
derives **all four** task-dependent paths from it together, so they cannot disagree:

| | `daily` | `hourly` |
|---|---|---|
| prepared dir | `$OUTPUT_ROOT/deterministic_and_mc_dropout<tier>/prepared/daily` (shared) | `$OUTPUT_ROOT/deterministic_unet_hourly<tier>/prepared/hourly` (its own) |
| run dir | `$OUTPUT_ROOT/<family><tier>` | `$OUTPUT_ROOT/deterministic_unet_hourly<tier>` |
| search space | `config/<family>/search_space_daily.yaml` | `config/deterministic_unet/search_space_hourly.yaml` |
| metrics | `config/eval/metrics_daily.yaml` | `config/eval/metrics_hourly.yaml` |

`MODE=hourly` is rejected for any family but `deterministic_unet` — no other family has an hourly pipeline. The split
does **not** move with the mode: hourly items are the same days expanded 24-fold.

⚠️ Setting one of these by hand instead has a silent failure mode in every direction. A daily metrics suite on a
probability field cuts the prediction at `> 0` (POD ≈ 1, a contingency table of nonsense, nothing raised); a daily
search space names `valid_regression_score`, which the module rejects; a mismatched prepared directory makes
`prepare_modeling` raise, but only after a preparation has run.

`pipeline.sh` deliberately does **not** set `MODE`: it runs a whole shipped YAML, which carries `mode:` itself.

## GPU

Two halves, and **either one alone fails**:

1. uncomment `#SBATCH --gres=gpu:1` (and a GPU partition), otherwise no device is allocated;
2. use a `_smoke_gpu`/full tier or set `ACCELERATOR=gpu`, otherwise the allocated GPU sits idle at CPU speed.

⚠️ A third half, off-cluster: install a CUDA-capable torch. `minimal_requirements.txt` no longer forces the `+cpu`
build, but an environment created before that change has one, and `torch.cuda.is_available()` is then False no matter
what slurm allocated. Check with `python scripts/preflight.py` and `python -c "import torch; print(torch.__version__)"`
— a `+cpu` suffix is the tell.

## Four traps these scripts encode

1. **`UPSTREAM_MODEL` means different things at different stages.** On `prepare_modeling` it is DIFFUSION's residual
   switch (the upstream's *predictions*, materialised as a conditioning channel). On `tune` it is MC-DROPOUT's warm
   start (the upstream's *weights*). Each script rejects the other family's usage with an explanation.
   `retrain_best` takes none: it reads the sweep's record, so the two cannot disagree.
2. **The two U-net families share one prepared directory; diffusion must not.** In residual mode diffusion's
   preparation appends a conditioning channel, and a U-net checkpoint pointed at that directory fails its
   `in_channels` check. `_common.sh` derives this from `FAMILY`.
3. **`mlflow.projects` shells out to a bare `python`.** A slurm job never has your venv activated, so `_common.sh`
   puts the interpreter first on `PATH`; without that every stage subprocess under `pipeline.sh` gets whatever `python`
   came first and dies on `import mlflow` — reported as a broken stage, which it is not. `src/__init__.py` does the
   same from inside Python, so the two cover each other.
4. **A missing root is silent in one direction and loud in the other.** `{{$VAR}}` substitutes to the EMPTY STRING, so
   an unset `DATA_ROOT` makes every dataset path relative and fails inside `prepare_modeling` with a bare
   `metadata.json`; an unset `OUTPUT_ROOT` makes every output path absolute at the filesystem root, which `setup`
   catches by name. `_common.sh` refuses both up front so neither happens.

Also: `ENSEMBLE_SIZE` must be ≥ 2 (`spread_skill_sums` uses `ddof=1`, so one member is a silent `NaN`), and any smoke
tier wants `RESTART=true` — `run_sweep` resumes from its own optuna store, which the pipeline's `lazy: false` does not
govern, so a re-run can report success having executed no trial.
