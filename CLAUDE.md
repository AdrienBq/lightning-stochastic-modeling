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
| `diffusion` | Flow matching. Optionally **residual**: conditions on the upstream's *prediction* (`UPSTREAM_MODEL` set on `prepare_regression`) and predicts a correction on top of it |

⚠️ Both stochastic families read `UPSTREAM_MODEL`, but at **different stages and for different things** —
MC-dropout wants the upstream's *weights* (a warm start, read by `tune`), diffusion wants its *predictions* (an
extra conditioning channel, materialised by `prepare_regression`).

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
| Sparsity | ~99.93 % of cells are zero. Every design choice is downstream of this |

## Design invariants

- **One evaluation for all families.** `evaluate_regression` is the single eval stage;
  `registry.load_regression_module` dispatches by checkpoint marker (with `_sniff_family` as the legacy fallback).
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
  | `config/split/` | `split.yaml` (the year split) + `split_smoke_cpu.yaml` / `split_smoke_gpu.yaml` subsets |
  | `config/eval/` | `metrics.yaml` (**the** shared suite) + the cross-family `probabilistic_eval*.yaml` |
  | `config/<family>/` | that family's pipeline, its two smoke tiers, and its `search_space.yaml` |

  Each family has three tiers: `<family>.yaml` (full), `<family>_smoke_cpu.yaml` and `<family>_smoke_gpu.yaml`.
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
  `PlateCarree` data transform, `origin='upper'`, integer unit bins in lightning-hours, and the warm/cool
  over/under diff encoding. Notebooks are **not** part of the repo.

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
