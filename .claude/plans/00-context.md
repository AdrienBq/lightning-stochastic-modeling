# Rebuild: context, findings & target architecture

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md).
> Steps: [0](step-0-bootstrap.md) · [1](step-1-design.md) · [2](step-2-config.md) · [3](step-3-utils.md) ·
> [4](step-4-stages.md) · [5](step-5-portability.md)

> **Living plan:** this is a rolling plan. Early steps make design decisions (which metrics/losses/plots to
> keep) that reshape the later steps, so the later steps are deliberately provisional. **At the end of every step
> the relevant file is updated** with the decisions made, and the next step refined before starting it.

## Context

`fers/fers26p8-knowledge-guided-tail-ml` is a **machine-learning parameterization of lightning within a
reanalysis dataset**: given ERA5 reanalysis atmospheric predictors (`MU_LI, MU_MIXR, RH_500850, cp, lsm`) for a
day, predict that day's gridded lightning field (e.g. hours-per-cell with lightning, from ATDnet). It is a
diagnostic ERA5 → lightning mapping — **not** a temporal forecast; there is no t+dt nowcasting objective.

Several model families were built in parallel on separate branches and have drifted apart (re-implemented ideas,
stale vendored copies, diverged infra). The goal is one clean repo, rebuilt from the
`lightning-stochastic-modeling` template, where the **MC-dropout**, **flow-matching**, and **deterministic_unet**
pipelines are merged, share as much code as possible, and report the **same metrics through one common
evaluation**. We rebuild block by block, verifying each block with a real CPU smoke run.

> **Naming (Step 2):** the family formerly called `distr_regression` is now **`deterministic_unet`** everywhere in
> the target repo — the old name described a distributional regression this family does not do. Source-branch
> filenames, stage names and class names (`tune_distr_regression`, `distr_regression_aru.py`,
> `docs/distr_regression_pipeline.md`) keep the old spelling throughout the inventories, because those are records
> of what literally exists on the branches we read from.

### Branch findings (exploration result)

| Branch | Role | Verdict |
|---|---|---|
| `aru-probabilistic-eval` | flow-matching + probabilistic eval; **already** has a family-agnostic shared architecture | **primary base** for shared code |
| `adrien-mc-dropout` | current, live MC-dropout pipeline (two-phase training) | **source of the MC-dropout model + losses**, and of the **plotting reference** (notebook 02a) |
| `claude/quizzical-golick-7417bd` | strict ancestor of `aru-probabilistic-eval` (behind 10, 0 unique) | **ignore** |
| `aru-diffusion-model` | strict ancestor of `aru-probabilistic-eval` (behind 31, 0 unique) | **ignore** |

`aru-probabilistic-eval` and `adrien-mc-dropout` genuinely diverged (merge-base `e3c13a0`; +126 / +38 commits).
The template's `src/stages/run.py`, `utils/io/lazy.py`, `utils/seeding.py`, `utils/plotting/*` are an evolved
superset of aru's infrastructure, so the template already provides the orchestration layer.

> **Source clone:** `/home/aburq/repos/fers/fers26p8-knowledge-guided-tail-ml`. It was stale (4 commits, no model
> code); **`git fetch --all` was run in Step 1** and all branches are now present. Read source without checking
> out, so the clone's working tree is never disturbed: `git show origin/<branch>:<path>`.

## Locked-in decisions

> ### ⚠️ SCOPE CHANGE (2026-07-28) — classification-first, **no target transform**
> The **first** modelling target is **occurrence**, not unbounded stroke counts:
> - **hourly** binary occurrence, or
> - **daily regression of the number of hours with lightning, bounded `0–24`** (`daily_lightning_hours`).
>
> The **gamma F-transform is dropped** (`GammaFTransform` / `LogStandardizeTransform`) — it exists to condition an
> unbounded heavy-tailed target and only adds complication here.
>
> **Key distinction:** the daily `0–24` target is still a *regression*, just a **bounded** one, so ordinary
> distance losses (MSE / MAE / Huber) and error metrics **stay**. What becomes obsolete is machinery specific to
> **unbounded heavy-tailed counts** (Tweedie, Poisson, log1p spaces, tail quantiles) and to the transform.
>
> **Biggest win:** with no transform, **training space == evaluation space**, so the whole "which space am I in?"
> class of bug disappears, along with the back-transform plumbing.
>
> All four `inventory-*.md` files are flagged 🔢 `COUNT-REG` / 🔀 `TRANSFORM` / ✅ `CLASSIF` accordingly, and
> [`inventory-architecture.md`](inventory-architecture.md) §6 holds a **15-item transform-removal checklist**.

- Scope: **all three families** — MC-dropout, flow-matching (incl. residual/knowledge-guided mode), and the
  `deterministic_unet` (baseline **and** the upstream model for *both* stochastic families). We do not do the
  regression on the counts of flashes. The residual mode is done on the 0 to 24 daily occurence regression.
- MC-dropout's two-phase (train → finetune) fit is folded into the **shared** tuning harness.
- **Both stochastic families build on the upstream, via the same `UPSTREAM_MODEL` variable but different
  mechanisms** (Step 2): MC-dropout takes the upstream's **weights** — read by `tune`, which then warm-starts and
  runs the finetuning phase alone — while diffusion takes its **predictions**, materialised by
  `prepare_regression` as an extra conditioning channel it learns a residual on top of. Unset ⇒ each family trains
  standalone. This is why `deterministic_unet`'s search space fixes `unet.normalization: group`: a batch-norm
  upstream cannot be warm-started into an MC-dropout model, whose inference re-enables dropout.
- Verification: **real CPU smoke runs — 1 epoch, 2 days of data** (full data is local, machine is CPU-only) at
  each block, plus unit tests + import/parse checks.
- **Plotting:** no notebooks are ported. Plotting functions live in `src/utils`, **inspired by notebook 02a**
  (`adrien-mc-dropout:notebooks/02a_visualize_val_event_diffusion.ipynb`) — see
  [`inventory-figures.md`](inventory-figures.md) §1 for the full visual spec.

## Environment & data (established in Step 0)

- **Environment:** `minimal_requirements.txt` (Python 3.11, pip/venv) is the source of truth; loose version
  bounds. The venv lives at `/homedata/aburq/.venvs/lightning-stochastic-modeling` and must be activated
  explicitly (`~/.bash_aburq` auto-activates `$VENV_ROOT/default` in every new shell). `environment.yml` is
  retained but unmaintained.
- **Machine:** **no GPU** — smoke runs are CPU-only, hence the CPU torch wheel. Core/RAM counts differ between
  local and remote and are deliberately not recorded here (see [Step 5](step-5-portability.md)).
- **Data root: `/homedata/aburq/batta_torch`** (~48 GB). Layout: `metadata.json` (`batta_torch_2`; vars
  `MU_LI, MU_MIXR, RH_500850, cp, lsm, lightnings`; 0.25°; 35–60N / −12–25E; 2008-01-02 → 2023-12-31),
  `metadata.csv` (`date,id,num_lightnings,pixels_with_lightning`), `samples/` with 5843 × ~8.7 MB
  `sample_XXXXXX.pt`, and `scalers/` (`final`, `old`, `split_0`…`split_12`).
  Raw ATDnet CSVs: `/homedata/aburq/lightning/ATDnet`. The upstream yearly ERA5 netCDFs at
  `/homedata/aburq/post_processed_era5` are **not** read by the pipeline.
  Reached via the **`DATA_ROOT`** environment variable (existing mechanism — see [Step 5](step-5-portability.md)).
- **Grid: 101 × 149**, `origin='upper'` (row 0 = north).
- **Split (by year):** test 2008 / 2015 / 2023 · validation 2009 / 2016 / 2022 · train the rest
  (2010–2014, 2017–2021).

## Target architecture (high level — details firm up as we go)

One repo, one orchestrator, one evaluation. Code splits into **shared** (family-agnostic) vs **model-specific**.

- **Shared infra (reuse from template):** `run.py`, `setup.py`, `io/lazy.py`, `io/parse_config.py`,
  `seeding.py`, `banner.py`, `plotting/` (`show_plot_and_save`, palettes).
- **Shared pipeline surface:** `io/data.py`; a merged `dataset.py`; `unet.py` (backbone + `enable_mc_dropout`);
  a unified `losses.py`; `search.py`; a two-phase-capable `tuning.py`; `validation.py` (single selection score);
  `registry.py`; the `metrics/` package (`scores`, `evaluation`, `reporting`, `diagnostics`);
  `config/split/split.yaml`, `config/eval/metrics.yaml`; the stages `prepare_regression.py`,
  **`evaluate_regression.py` (the common eval)**, `tabulate_metrics.py`, `combine_curves.py`.
  *(`transforms.py` is **dropped** — see the scope change.)*
- **Model-specific:** flow-matching (`diffusion.py`, `diffusion_module.py`);
  MC-dropout (`mc_dropout_module.py` from adrien + `mc_dropout_eval.py` adapter);
  `deterministic_unet` (`module.py`, `DeterministicUnetModule`); plus each family's config + search space.
  *(Per-family `tune_*` stages are being **unified** — see [Step 4](step-4-stages.md).)*

The common eval already unifies families: `registry.load_regression_module` dispatches by checkpoint marker and
wraps MC-dropout in `MCDropoutEnsembleModule`, which re-expresses MC forward passes in the ensemble contract and
feeds the **shared** `scores.ensemble_partials`. The `deterministic_unet` supplies the upstream for both stochastic
families — its *weights* for MC-dropout's warm start, its *predictions* for diffusion's residual mode.

### The ensemble contract

```python
module = load_regression_module(path, map_location='cpu', model_family=None)   # None = auto-detect
module.eval_ensemble_size = M                                                  # M > 1 => ensemble
out = module.predict_step(batch, 0)
```

| Key | Shape | Present when |
|---|---|---|
| `observation` | `[B, H, W]` | always |
| `prediction` | `[B, H, W]` | always (ensemble **mean** for stochastic families) |
| `ensemble_members` | `[B, M, H, W]` | stochastic family **and** `eval_ensemble_size > 1` |

Family dispatch is by **checkpoint marker**, with `_sniff_family` as the fallback for legacy checkpoints.

**Residual mode:** `upstream = backbone.predict_step(...)['prediction']`, appended as the **last** conditioning
channel, and passed as a 3rd batch item: `batch = (x_cond, y, upstream)`; the diffusion returns
`clamp(upstream + residual)`. Flagged by the `residual_target` attribute; channel count validated against
`feature_mean.shape[0]`.
