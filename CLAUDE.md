# lightning-stochastic-modeling

## What this project is

A **machine-learning parameterization of lightning within a reanalysis dataset**. Given a day's gridded ERA5
reanalysis predictors (`MU_LI`, `MU_MIXR`, `RH_500850`, `cp`, `lsm`), predict that day's gridded lightning field
(ERA5-gridded ATDnet observations).

**It is a diagnostic ERA5 → lightning mapping, not a temporal forecast.** There is no `t + dt` nowcasting
objective. Do not describe it as forecasting or nowcasting.

Three model families share one pipeline and report **the same metrics through one common evaluation**:

| Family | Nature |
|---|---|
| `deterministic_unet` | Deterministic U-net — the baseline, and the *upstream* model for the other two |
| `mc_dropout` | Stochastic via MC-dropout at inference. **Warm-starts** from the upstream's *weights* (`UPSTREAM_MODEL` set on `tune`) and runs the finetuning phase alone; unset ⇒ two-phase fit from scratch (train → finetune) |
| `diffusion` | Flow matching. Optionally **residual**: conditions on the upstream's *prediction* (`UPSTREAM_MODEL` set on `prepare_modeling`) and predicts a correction on top of it |

⚠️ Both stochastic families read `UPSTREAM_MODEL`, but at **different stages and for different things** —
MC-dropout wants the upstream's *weights* (a warm start, read by `tune`), diffusion wants its *predictions* (an
extra conditioning channel, materialised by `prepare_modeling`).

## Current scope: classification-first, no target transform

The **first** modelling target is **occurrence**, not unbounded stroke counts:

- **hourly** binary occurrence, or
- **daily regression of the number of hours with lightning, bounded `0–24`**.

**`mode` is the only key that selects between them.** `daily_lightning_hours` is the informal English name of
`mode: daily`, never a config value — there is no `target-variable` key, and the string appears nowhere in
`config/`. It survives only as a deprecated alias in `normalize_mode`, so artifacts prepared under the old name
keep loading.

The gamma F-transform (`GammaFTransform` / `LogStandardizeTransform`) is **removed**. It existed to condition an
unbounded heavy-tailed target and only adds complication here.

Two consequences that matter constantly:

1. **Training space == evaluation space.** There is no back-transform anywhere. If you find yourself asking "which
   space is this tensor in?", the answer is always *the target space*.
2. The `0–24` daily target is still a **regression**, just a *bounded* one. Ordinary distance losses (MSE / MAE /
   Huber) and error metrics apply. What does **not** apply is machinery for unbounded heavy-tailed counts —
   Tweedie, Poisson NLL, tail quantiles of the positive marginal. Poisson is actively wrong here: it is unbounded,
   so it puts probability mass above 24 hours/day.

## Data

Reached via the **`DATA_ROOT`** environment variable — **never hardcode a data path**, in code or in config.

⚠️ **Two roots, both required, both `{{$VAR}}`:** `DATA_ROOT` for the read-only dataset and **`OUTPUT_ROOT`** for
everything the pipelines write. Outputs are not in the checkout — one daily prepared directory is ~20 GiB (`features/`
alone is 19.7 GiB per split) and the three families come to ~60 GiB, so on a cluster the source tree and the bulk
storage are separate filesystems.

```shell
export DATA_ROOT=/path/to/era5_postprocess          # metadata.json, metadata.csv, samples/
export OUTPUT_ROOT=/scratch/$USER/lightning-outputs # prepared/, tuning/, best/, evaluation/, reports/
```

Or put them in a gitignored **`.env`** at the repo root (copy [`.env.example`](.env.example)). `src/__init__.py` loads
it on the first `src.` import, so it reaches `run_project.py`, every stage subprocess, `pytest` and `scripts/*` from one
place; `job_scripts.example/_common.sh` sources the same file, so one file configures both sides. ⚠️ **An already-set
variable always wins** — `.env` is the fallback, not the authority, so an `export` or slurm's inherited environment
still overrides it and it cannot silently retarget a running job. A line that is not a `KEY=VALUE` assignment raises
rather than being skipped. Run **`python scripts/preflight.py`** to check the whole environment at once (roots,
dependencies, the stage interpreter, the cartopy bundle, git) before queueing anything.

An **unset** variable substitutes to the empty string rather than erroring (see the `{{$VAR}}` note below), so
`'{{$OUTPUT_ROOT}}/family/prepared'` becomes `/family/prepared` — absolute, at the filesystem root. `setup`, the first
stage of every pipeline, refuses that and names the variable; a missing `DATA_ROOT` still fails inside the stage.

**The two U-net families share one prepared directory**, `$OUTPUT_ROOT/deterministic_and_mc_dropout/prepared`: their
`prepare_modeling` blocks are identical, so whichever pipeline runs first prepares it and the second skips. Their
`prepare_modeling` blocks must therefore stay in step — `parse_config_test.py` enforces the equality. **Diffusion keeps
its own**, because in residual mode its preparation writes `upstream/` maps and flips `residual_target`, which would
give the U-net families a 6th conditioning channel their checkpoints were not built for.

```
$DATA_ROOT/
  metadata.json     # 6 variables, 0.25 deg, 35-60N / -12-25E, 2008-01-02 -> 2023-12-31
  metadata.csv      # date,id,num_lightnings,pixels_with_lightning
  samples/          # 5843 x sample_XXXXXX.pt, ~8.7 MB each -- torch tensors, NOT netCDF
  scalers/          # final, old, split_0 .. split_12
```

| Invariant | Value |
|---|---|
| Grid | **101 × 149**, `origin='upper'` (array row 0 is the **north** edge) |
| Domain | 0.25°, 35–60 °N / −12–25 °E |
| Predictors | `MU_LI`, `MU_MIXR`, `RH_500850`, `cp`, `lsm` (+ `upstream` appended **last** in residual mode) |
| Target | `lightnings`, aggregated by `_daily_aggregation` (`hourly_threshold: 2` drops single-stroke hours) |
| Split | **by year** — test 2008/2015/2023 · valid 2009/2016/2022 · train 2010–2014, 2017–2021 |
| Sparsity | **hourly** target 99.57 % zero (positive class **0.43 %**) · **daily** target 95.30 % zero (positive **4.7 %**) · raw `lightnings == 0` 99.35 %, and a further 0.22 % carry exactly one stroke, which `hourly_threshold: 2` drops. Measured over all 5843 samples by [`scripts/sparsity.py`](scripts/sparsity.py); the table with the seasonal and yearly breakdown is in [README.md](README.md). Every design choice is downstream of this — ⚠️ **quote the right one of the six**: the repo previously asserted "~99.93 % of cells are zero" in ~40 places, which matches none of them and understated the hourly positive class by 6× and the daily one by 67× |

## Design invariants

- **One evaluation for all families.** `evaluate` is the single eval stage;
  `registry.load_model_module` dispatches by checkpoint marker (with `_sniff_family` as the legacy fallback).
  Never add a family-specific evaluation path.
- **The ensemble contract.** `predict_step(batch, idx)` returns a dict with `observation` `[B,H,W]`, `prediction`
  `[B,H,W]` (the ensemble *mean* for stochastic families), and `ensemble_members` `[B,M,H,W]` only when the family
  is stochastic and `eval_ensemble_size > 1`. MC-dropout is wrapped by `MCDropoutEnsembleModule` to satisfy this,
  which is what lets it feed the *shared* `scores.ensemble_partials`.
- **Streaming ensemble metrics return sums, not means.** `crps_sums` / `spread_skill_sums` return
  `(sum, ..., n_cells)` because the full `[N, M, H, W]` stack cannot be held in memory. Sums are additive across
  batches; means and ratios are not. Divide exactly once, at the end.
- **`ensemble-size` must be ≥ 2.** `spread_skill_sums` uses `ddof=1`, so a single-member ensemble silently yields
  `NaN` rather than erroring. This bites smoke configs in particular.
- **Every pointwise loss reduces through `_weighted_masked_mean`.** It normalises by the *sum of effective
  weights*, not the cell count. Inlining it risks a loss on a different scale from its siblings, which makes
  tuning results incomparable.
- **`intensity_weights(y, gamma) = (1 + y)^gamma` is computed from the raw target** and `gamma = 0` means
  unweighted — that is how `weighted_mae` covers plain `mae`, so the search space must allow `gamma: 0.0`.
- **Residual mode:** the upstream prediction is appended as the **last** conditioning channel and passed as a
  third batch item, `(x_cond, y, upstream)`; the model returns `clamp(upstream + residual)`. Flagged by the
  `residual_target` attribute; validate the channel count against `feature_mean.shape[0]`.

## Pipeline conventions

The repo is built on the `plumber` MLflow pipeline template. See [README.md](README.md) for the full mechanics.

- **Stages are standalone CLI scripts** under `src/stages/`, wrapped with `fire`. Each must begin with
  `from __init__ import root_path` **before** any `src.` import — stages run from within `src/stages/`.
- **Pipelines are YAML** under `config/`, grouped one directory per concern and run via
  `python run_project.py config/<family>/<pipeline>.yaml <EXPERIMENT>`:

  | Directory | Holds |
  |---|---|
  | `config/split/` | `split.yaml` (the year split) + `split_smoke_cpu.yaml` / `split_smoke_gpu.yaml` subsets — **task-agnostic**, shared by daily and hourly |
  | `config/eval/` | `metrics_daily.yaml` and `metrics_hourly.yaml` (**the** two shared suites) + the cross-family `probabilistic_eval*.yaml` |
  | `config/<family>/` | that family's pipelines, their smoke tiers, and one `search_space_<task>.yaml` per task |

  **Every task-specific file names its task**, so an unsuffixed name is never ambiguous: `metrics_daily.yaml` /
  `metrics_hourly.yaml`, `search_space_daily.yaml` / `search_space_hourly.yaml`, and
  `<family>_<task>[_smoke_cpu|_smoke_gpu].yaml`. The three daily families therefore have three tiers each
  (`<family>_daily.yaml`, `<family>_daily_smoke_cpu.yaml`, `<family>_daily_smoke_gpu.yaml`) and
  `deterministic_unet` additionally has `deterministic_unet_hourly.yaml` + `deterministic_unet_hourly_smoke_cpu.yaml`.
  ⚠️ **The `$OUTPUT_ROOT` directory names were deliberately NOT renamed** (`$OUTPUT_ROOT/deterministic_unet_smoke_cpu`,
  not `..._daily_smoke_cpu`): they are where ~60 GiB of prepared data and checkpoints already live, and the config
  rename was about disambiguating source files.
- **An hourly pipeline swaps exactly three files** — `metrics-config`, `model-config` and (implicitly) the prepared
  directory — and each has a *silent* failure mode if it does not move with the others. A daily metrics suite cuts a
  probability field at `> 0` (POD ≈ 1, a contingency table of nonsense, nothing raised); a daily search space names
  `valid_regression_score`, which `selection_metric_for_mode` **rejects**; a daily prepared directory makes
  `prepare_modeling` raise on the mode mismatch. `parse_config_test.py` pins all three.
- **Env vars in config use `{{$VAR}}`, not `${VAR}`.** `parse_config` substitutes textually *before* the YAML parse,
  and an **unset variable becomes the empty string** rather than an error — so quote interpolated scalars
  (`data-path: '{{$DATA_ROOT}}'`) and expect a missing variable to fail inside the stage, not at parse time.
- **`OUTPUT_PARAM_KEYS`** (`output-path`, `metrics-path`, `report-path`) are treated as a stage's outputs by the
  lazy cache; any other parameter resolving to an existing path is an input. Use those names. Corollary: a *stale*
  path silently degrades to a plain scalar, so the cache stops invalidating on that input — keep
  `metrics-config` / `split-config` / `model-config` pointing at files that exist.
- **Seeding is automatic** — the orchestrator exports `PIPELINE_SEED` and `src/stages/__init__.py` applies it. Do
  not re-seed globally inside a stage.
- **Commit before running a pipeline.** The lazy cache keys on the whole-repo dirty diff, so any uncommitted edit
  busts every cache entry.
- **Plotting** lives in `src/utils`, styled after the spec in
  [`.claude/plans/inventory-figures.md`](.claude/plans/inventory-figures.md) §1 — cartopy `EuroPP` axes with a
  `PlateCarree` data transform, `origin='upper'`, integer unit bins in lightning-hours, and **one warm palette per
  figure under a single colorbar**. Every map panel carries its own title (`observations`, `predictions` /
  `ensemble mean`, `ensemble std`, `member 1..3`) under a global `<date> event, <family> model`.
  ⛔ The **warm/cool over/under diff encoding was removed in Step 4 block 4e**: its second colorbar collided with the
  ensemble-std one and the pair read as clutter. `_BASE_COLORS_COOL` is kept, flagged legacy, one
  `make_lightning_cmap` call from being usable again; `draw_diff_map` and the two-colorbar helper are gone. Error
  direction is now read by comparing panels, and quantitatively from `bias` / `under_frac_*` / `over_frac_*`.
  Notebooks are **not** part of the repo.

## Rebuild in progress

This repo is being rebuilt block by block, merging three drifted branches of
`fers26p8-knowledge-guided-tail-ml`. **The plan is authoritative — read it before making structural changes:**

| Document | Contents |
|---|---|
| [`.claude/plans/rebuild-plan.md`](.claude/plans/rebuild-plan.md) | Index, cross-cutting merge tasks, verification gates |
| [`.claude/plans/00-context.md`](.claude/plans/00-context.md) | Framing, branch findings, target architecture |
| `.claude/plans/step-*.md` | One file per step (0 done, 1 done, 2 next) |
| `.claude/plans/inventory-*.md` | **The design decision record** — every loss, score, figure and module with a keep/change/remove decision |

Source branches are read **without checking out**, so the shared clone's working tree is never disturbed:

```shell
git -C /home/aburq/repos/fers/fers26p8-knowledge-guided-tail-ml show origin/<branch>:<path>
```

`aru-probabilistic-eval` is the base for shared infrastructure and **all** scores/evaluation;
`adrien-mc-dropout` is the source of the MC-dropout model, the losses superset, and the plotting style.

---

## Guidelines

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

Mark those goal / tasks as TODOs prior to beginning, and report them individually to the user.
Ask the user for verification of the goals, only _then_ begin the workflow.

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
