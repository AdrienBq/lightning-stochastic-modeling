# Rebuild plan: clean, merged `lightning-stochastic-modeling`

> **Deliverable location:** the user asked for this plan under
> `lightning-stochastic-modeling/.claude/plans/`. Plan mode only lets me edit this file; on approval the first
> action is to copy it to `lightning-stochastic-modeling/.claude/plans/rebuild-plan.md`.
>
> **Living plan:** this is a rolling plan. Early steps make design decisions (which metrics/losses/plots to
> keep) that reshape the later steps, so the later steps below are deliberately provisional. **At the end of
> every step I update this file** with the decisions made and refine the next step before starting it.

## Context

`fers/fers26p8-knowledge-guided-tail-ml` is a **machine-learning parameterization of lightning within a
reanalysis dataset**: given ERA5 reanalysis atmospheric predictors (`MU_LI, MU_MIXR, RH_500850, cp, lsm`) for a
day, predict that day's gridded lightning field (e.g. hours-per-cell with lightning, from ATDnet). It is a
diagnostic ERA5 → lightning mapping — **not** a temporal forecast; there is no t+dt nowcasting objective.

Several model families were built in parallel on separate branches and have drifted apart (re-implemented ideas,
stale vendored copies, diverged infra). The goal is one clean repo, rebuilt from the
`lightning-stochastic-modeling` template, where the **MC-dropout**, **flow-matching**, and **distr_regression
(U-net)** pipelines are merged, share as much code as possible, and report the **same metrics through one common
evaluation**. We rebuild block by block, verifying each block with a real CPU smoke run.

### Branch findings (exploration result)

| Branch | Role | Verdict |
|---|---|---|
| `aru-probabilistic-eval` | flow-matching + probabilistic eval; **already** has a family-agnostic shared architecture | **primary base** for shared code |
| `adrien-mc-dropout` | current, live MC-dropout pipeline (two-phase training) | **source of the MC-dropout model + losses** |
| `claude/quizzical-golick-7417bd` | strict ancestor of `aru-probabilistic-eval` (behind 10, 0 unique) | **ignore** |
| `aru-diffusion-model` | strict ancestor of `aru-probabilistic-eval` (behind 31, 0 unique) | **ignore** |

> ⚠️ **The local source clone is stale (found in Step 0).**
> `/home/aburq/repos/fers/fers26p8-knowledge-guided-tail-ml` holds only 4 commits and **no model code** — every
> branch above exists on `origin` (`git@github.com:aurelio-raffa/fers26p8-knowledge-guided-tail-ml.git`) but is
> unfetched. Network to GitHub works. **`git fetch --all` in that clone is a prerequisite for Step 1.**

`aru-probabilistic-eval` and `adrien-mc-dropout` genuinely diverged (merge-base `e3c13a0`; +126 / +38 commits).
The template's `src/stages/run.py`, `utils/io/lazy.py`, `utils/seeding.py`, `utils/plotting/*` are an evolved
superset of aru's infrastructure, so the template already provides the orchestration layer.

### Locked-in decisions
- Scope: **all three families** — MC-dropout, flow-matching (incl. residual/knowledge-guided mode), and
  distr_regression U-net (baseline **and** the upstream model for the residual mode).
- MC-dropout's two-phase (train → finetune) fit is folded into the **shared** tuning harness.
- Verification: **real CPU smoke runs — 1 epoch, 2 days of data** (full data is local, machine is CPU-only) at
  each block, plus unit tests + import/parse checks.

### Environment & data (established in Step 0)
- **Environment:** `minimal_requirements.txt` (Python 3.11, pip/venv) is the source of truth; loose version
  bounds. The venv lives at `/homedata/aburq/.venvs/lightning-stochastic-modeling` and must be activated
  explicitly (`~/.bash_aburq` auto-activates `$VENV_ROOT/default` in every new shell). `environment.yml` is
  retained but unmaintained.
- **Machine:** `nproc = 1`, 251 GB RAM, **no GPU** — hence the CPU torch wheel and genuinely tiny smoke runs.
- **Data root: `/homedata/aburq/batta_torch`** (~48 GB). *The path originally written in this plan
  (`/home/aburq/repos/fers/data/era5_post_process`) does not exist.* Layout: `metadata.json`
  (`batta_torch_2`; vars `MU_LI, MU_MIXR, RH_500850, cp, lsm, lightnings`; 0.25°; 35–60N / −12–25E;
  2008-01-02 → 2023-12-31), `metadata.csv` (`date,id,num_lightnings,pixels_with_lightning`),
  `samples/` with 5843 × ~8.7 MB `sample_XXXXXX.pt`, and `scalers/` (`final`, `old`, `split_0`…`split_12`).
  Raw ATDnet CSVs: `/homedata/aburq/lightning/ATDnet`. The upstream yearly ERA5 netCDFs at
  `/homedata/aburq/post_processed_era5` are **not** read by the pipeline.
- **Split (by year):** test 2008 / 2015 / 2023 · validation 2009 / 2016 / 2022 · train the rest
  (2010–2014, 2017–2021).

## Target architecture (high level — details firm up as we go)

One repo, one orchestrator, one evaluation. Code splits into **shared** (family-agnostic) vs **model-specific**.

- **Shared infra (reuse from template):** `run.py`, `setup.py`, `io/lazy.py`, `io/parse_config.py`,
  `seeding.py`, `banner.py`, `plotting/` (`show_plot_and_save`, palettes).
- **Shared pipeline surface:** `io/data.py`; a merged `dataset.py`; `unet.py` (backbone + `enable_mc_dropout`);
  `transforms.py`; a unified `losses.py`; `search.py`; a two-phase-capable `tuning.py`; `validation.py`
  (single selection score); `registry.py`; the `metrics/` package (`scores`, `evaluation`, `reporting`,
  `diagnostics`); `config/split.yaml`, `config/metrics.yaml`; the stages `prepare_regression.py`,
  **`evaluate_regression.py` (the common eval)**, `tabulate_metrics.py`, `combine_curves.py`.
- **Model-specific:** flow-matching (`diffusion.py`, `diffusion_module.py`, `tune_diffusion`);
  MC-dropout (`mc_dropout_module.py` from adrien + `mc_dropout_eval.py` adapter, `tune_mc_dropout`);
  distr_regression (`module.py`, `tune_distr_regression`); plus each family's config + search space.

The common eval already unifies families: `registry.load_regression_module` dispatches by checkpoint marker and
wraps MC-dropout in `MCDropoutEnsembleModule`, which re-expresses MC forward passes in the ensemble contract and
feeds the **shared** `scores.ensemble_partials`. distr_regression supplies the upstream for residual mode.

> **NB — the `.md`/design step (Step 1) will revise the metric, loss, and plotting inventories** (keep / modify /
> drop / add), and define the map-plotting spec (palette, CRS/projection, colorbar, quantization). Those
> decisions change what `config/metrics.yaml`, the search spaces (Step 2), `losses.py`, `metrics/*`, and the
> reporting/plotting code (Step 3) contain. So Steps 2–4 below are provisional and get rewritten after Step 1.

## Rebuild sequence

Block order a → b → c → d → e. Each step ends with the verification gate and a plan update.

### Step 0 — Bootstrap & hygiene — ✅ **DONE** (2026-07-27)
- **Rebranded:** `README.md` (title, framing paragraph — diagnostic mapping, not forecasting — plus new
  Installation and Data sections); `config/hello_world.yaml` `description` → the project tagline (feeds the
  banner; the banner *name* auto-derives from the directory, so no code change). `environment.yml` kept but
  marked **unmaintained** with a header pointing at `minimal_requirements.txt`.
- **`minimal_requirements.txt` written** — hand-built, loose bounds (`>=`,`<`): mlflow, fire, pyyaml, torch,
  lightning, optuna, numpy, pandas, scipy, scikit-learn, matplotlib, pytest. Carries
  `--extra-index-url https://download.pytorch.org/whl/cpu` because the machine has **no GPU** and the default
  PyPI linux torch wheel bundles ~2.5 GB of `nvidia-*` deps. Confirmed correct **not** to copy the old
  `requirements.txt`: it is a 154-line full env dump (horovod, poetry, fastapi, jupyter, rasterio…).
  Excluded until a step needs them: `cartopy`/`geopandas` (Step 1's map spec decides), `xarray`/`netCDF4`
  (data is `.pt` + CSV, not netCDF), `prefect` (`run.py` falls back to the mlflow orchestrator when absent).
  `torchmetrics` arrives anyway as a `lightning` transitive dep.
- **`.gitattributes` / `.gitignore`: no carry-over needed.** The template's are a strict superset of
  `fers26p8`'s — identical LFS lists plus the template's ML-checkpoint/tensor blocks (`*.pt`, `*.ckpt`,
  `*.safetensors`, `*.npz`…), and the template `.gitignore` adds `/outputs/` and `**/*.db`. Verified via
  `git check-attr`, not modified.
- **Verified:** venv is Python 3.11.7 and runs standalone (`module load python/meso-3.11` needed only to
  *create* it); `torch 2.13.0+cpu` with zero `nvidia-*` packages; all deps import; `parse_config` reads the
  YAML; `python run_project.py config/hello_world.yaml STEP0_SMOKE` exits 0 with the rebranded banner and
  seeding applied to `random, numpy, lightning, torch`; `git status` scoped to the 4 intended files.
- **Open follow-up (deliberately deferred):** `mlflow` resolved to **3.12.0**, one minor release below the
  `<3.13` ceiling where the local file-store backend is disabled — a fresh install in a few weeks will break.
  Tighten the bound when that bites.

### Step 1 (block a) — Docs = design decisions (the pivotal step)
**Prerequisite:** `git fetch --all` in `/home/aburq/repos/fers/fers26p8-knowledge-guided-tail-ml` — the clone is
stale and none of the source branches are present locally (see the branch-findings warning above). Without it
there is no `aru-probabilistic-eval` / `adrien-mc-dropout` code to inventory metrics and losses from.

This is a **design activity**, not just porting. Produce, and record decisions in:
- **Global** `CLAUDE.md` + `README.md`: correct framing (ML parameterization from reanalysis, not forecasting),
  data/split/design invariants, pipeline conventions.
- **`docs/`** (reworked, not copied): a metrics-&-losses design doc that **decides the inventory** — for each
  existing score/loss (from both branches) mark keep / modify / drop, and list additions; a plotting/reporting
  design doc that **defines the map spec** (color palette, CRS / cartopy projection, colorbar + quantization,
  which report figures survive); a pipeline/architecture doc (stage contracts, the shared-vs-specific split,
  the checkpoint `module_class` marker + `predict_step` ensemble contract).
- **Per-folder `README.md`** standards in `src/stages/`, `src/utils/metrics/`, `src/utils/modeling/`,
  `src/utils/plotting/` capturing the agreed contracts.
- **Gate:** the chosen metric/loss/plot inventory is the single source of truth for Steps 2–4. **Update this plan
  with the decisions and rewrite Steps 2–4 to match.**

### Step 2 (block b) — Config, driven by Step 1 decisions
- Shared `config/split.yaml` and `config/metrics.yaml` reflecting **only** the kept/added metrics.
- Per-family pipeline YAMLs + search spaces (losses/knobs limited to what Step 1 kept), the cross-model
  `probabilistic_eval.yaml`, and CPU **smoke variants** (`*_local.yaml`: 1 epoch, 2 days, tiny ensemble).

### Step 3 (block c) — Shared `src/utils` (provisional, refined after Steps 1–2)
Build in dependency order, implementing the Step-1 inventory: `io/data.py` → modeling (`dataset`, `unet`,
`transforms`, unified `losses`, `search`, `validation`, two-phase `tuning`, then model modules + `registry`) →
metrics (`scores`, `evaluation`, `reporting`, `diagnostics`) → confirm plotting covers the map spec.

### Step 4 (block d) — Stages + common evaluation (provisional)
`setup` → merged `prepare_regression` → shared-harness `tune_*` → `retrain_best_*` → **`evaluate_regression`
(common)** → `tabulate_metrics` → `combine_curves`; wire the per-family + cross-model pipelines; port the CPU
unit tests.

### Step 5 (block e) — Portability: user-agnostic & machine-agnostic

This repo is meant to be used on other machines or remote servers and by other users, so the paths baked in
during Step 0 (`/homedata/aburq/.venvs/lightning-stochastic-modeling`, the `/homedata/aburq/batta_torch` data
root) will not be applicable to other contexts — this is the final step, once the pipelines themselves are
merged and stable. It adds a local, per-user config file where each user specifies their own paths (data root,
venv/output locations, etc.), so nothing user- or machine-specific is hardcoded in the pipeline YAMLs or code.
It also broadens installation beyond pip: `minimal_requirements.txt` currently assumes a plain `venv` + `pip`
workflow; this step adds equivalent install paths for `uv` and `conda` so the repo isn't tied to one tool.

## Key merge tasks & risks (carried through Steps 3–4)
1. **MC-dropout de-duplication:** drop aru's stale vendored `modeling/mc_dropout/` package; use adrien's current
   `mc_dropout_module.py` on the shared `unet`/`losses`/`scores`.
2. **Unified `losses.py`** (superset of both branches, pruned to the Step-1 inventory).
3. **Single evaluation path:** aru's `run_metric_suite` + `finalize_ensemble_metrics` + streaming
   `ensemble_partials`; retire adrien's parallel suite; MC validation uses the shared `selection_score`.
4. **Two-phase `run_sweep`:** generalize adrien's train→finetune fit into the shared harness without regressing
   single-phase families (verify monitor / best-weight restore parity).
5. **Merge `dataset.py`** (`hourly_stack` aggregation + residual/upstream channel) and **`prepare_regression`**
   (full-target / residual / `daily_lightning_hours` target).
6. **Selection-score unification** (`valid_regression_score` ≡ `valid_tail_score` — one name).
7. **Registry markers + `_sniff_family`** so legacy checkpoints still load.
8. **Drop dead/superseded code:** deprecated occurrence classifier, adrien's inference ports, `hello_world`,
   `compute_high_lightning_days` (unless kept as a utility).

## Verification (per block; real CPU smoke, 1 epoch, 2 days)
1. Import/parse: `python -c "import ..."` for touched modules; `parse_config` on every YAML.
2. Unit tests: `pytest` on the relevant ported tests.
3. Smoke run: the affected `*_local.yaml` via `python run_project.py config/<family>_local.yaml <EXPERIMENT>`
   (CPU, `n-trials 1`, `max-epochs 1`, 2-day split); assert declared artifacts + expected metric keys.
4. Phase/final gate: run all three family `*_local` pipelines + `probabilistic_eval_local` end-to-end; assert
   `tabulate_metrics` emits one comparison CSV with identical metric-key columns across families and
   `combine_curves` emits the overlaid figures — proof the pipelines are merged and report the same metrics.
5. **After each step: update this plan** (decisions made, next step refined).
