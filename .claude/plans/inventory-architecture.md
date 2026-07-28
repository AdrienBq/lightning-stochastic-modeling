# Inventory: pipeline & architecture

**Purpose.** Factual inventory of the stage and modeling layers across both source branches, plus the
shared-vs-model-specific split. **The `Decision` column is for you to fill in.** Drives Steps 3–4.

## ⚠️ Scope decision (2026-07-28): classification-first, **no target transform**

First target is **occurrence** — hourly binary, or the **bounded daily count of hours with lightning, `0–24`**.
The gamma F-transform is **dropped**. The `0–24` target is still a regression, so distance losses and error
metrics stay; what goes is machinery for **unbounded heavy-tailed counts** and for the transform.

| Flag | Meaning |
|---|---|
| 🔢 `COUNT-REG` | Exclusive to the **unbounded count** task — obsolete under the new scope |
| 🔀 `TRANSFORM` | Serves the gamma F-transform only — **dies with it** |
| ✅ `CLASSIF` | Directly serves the occurrence task — **the new priority** |

## 1. Stages — aru unified them, adrien has them per-family

| Stage | A `aru-probabilistic-eval` | D `adrien-mc-dropout` | Note | Decision |
|---|---|---|---|---|
| `setup` | ✅ | ✅ | Identical; template also has it | keep |
| `run` | ✅ (richer: `_coerce_bool`, `_log_metrics_to_run`) | ✅ (bare) | Template's is an evolved superset of both | |
| **prepare** | `prepare_regression` — **one stage, all families** | `prepare_distr_regression` + `prepare_mc_dropout` — **two** | A unified it | keep A |
| **evaluate** | `evaluate_regression` — **one stage, all families** (+ legacy `evaluate_distr_regression`) | `evaluate_distr_regression` + `evaluate_mc_dropout` — **two** | A unified it | keep A|
| **tune** | `tune_distr_regression`, `tune_diffusion` — thin wrappers over shared `tuning.py` | `tune_distr_regression`, `tune_mc_dropout` — each with its **own** `_fit_trial` | A factored the harness out | modify to unify all tuning |
| `retrain_best_*` | `retrain_best_distr_regression`, `retrain_best_diffusion` | ✗ **absent** | A-only | keep |
| `tabulate_metrics` | ✅ | ✗ **absent** | A-only — the cross-model comparison CSV | keep |
| `combine_curves` | ✅ (`_combined_psd/_qq/_fss/_rank_histogram`) | ✗ **absent** | A-only — overlaid cross-model figures | modify : D has something similar for plotting in the notebook 02b_visualize_test_event_diffusion.ipynb, the plotting should follow this convention |
| `tune_mc_dropout` | ✗ **absent** | ✅ with **`_fit_phase`** (two-phase train→finetune) | D-only — the thing to fold into A's `run_sweep` | remove : has to be unified with other tuning (see above)|
| `compute_high_lightning_days` | ✗ | ✅ | Plan lists as drop candidate | keep : this is the "extremes" |
| `hello_world` | ✗ | ✅ | Template scaffold | check if it's useful and remove if not |

**The structural fact:** A has the *unified* stage surface (one prepare, one evaluate, one tuning harness, plus
the cross-model comparison stages). D has the *only* MC-dropout stages. So the merge is: **A's stage skeleton +
D's MC-dropout family folded into it.**

### Merge task #4 in concrete terms

| | Where the training loop lives | Two-phase? |
|---|---|---|
| A | `modeling/tuning.py` : `run_sweep` → `_fit_trial` (shared by all families) | ✗ single-phase |
| D | `stages/tune_mc_dropout.py` : `_fit_trial` → **`_fit_phase`** (called twice) | ✅ train → finetune |
| D | `stages/tune_distr_regression.py` : its own `_fit_trial` (duplicate of A's logic) | ✗ |

So `_fit_phase` is the unit to lift into A's `run_sweep`, and D's duplicated `_fit_trial`s are then deletable.

## 2. Modeling layer

### Files present on both — near-identical (safe merges)

| File | A | D | Status | Decision |
|---|---|---|---|---|
| `diffusion.py` | 10 defs/classes | 10, **same names at identical line numbers** | `IDENTICAL` — never diverged | keep |
| `transforms.py` | `GammaFTransform` (23), `LogStandardizeTransform` (112) | same, (23) / (109) | 🔀 `TRANSFORM` — **the whole file is the transform.** Both classes exist only to warp an unbounded heavy-tailed target. **Delete the file** under the new scope | remove |
| `search.py` | 6 functions | 6, only `flatten_trial` offset differs (97 vs 86) | near-identical, **but** 🔀 `apply_constraints` becomes near-empty — *both* its constraints are transform-related (see §6) | keep |
| `registry.py` | 5 functions | 4 — A adds `_mc_dropout_eval_overrides` | `A-SUPERSET` | modify : unify mc dropout with diffusion|
| `mc_dropout_eval.py` | `MCDropoutEnsembleModule` (37) | same class (18) | diverged in size — needs diffing | modify : unify |

### A-only — the shared infrastructure D never had

| File / symbol | Line | Role | Decision |
|---|---|---|---|
| **`tuning.py`** (whole file) | — | `run_sweep`, `retrain_best_config`, `_fit_trial`, `ThroughputDiagnostics`, `TrialProgressBar`, `OptunaPruningCallback`, `_journal_storage`, staleness checks | |
| **`validation.py`** (whole file) | — | `compute_selection_components`, `selection_score` — the single selection score (merge task #6) | |
| `dataset.DayGroupedShuffleSampler` | 213 | Day-grouped shuffling (needed for hourly mode) | |
| `unet.Fp32BilinearUpsample` | 80 | fp32 upsample — mixed-precision correctness | |
| `unet.PlattScaling` | 181 | ✅ `CLASSIF` Probability calibration head — calibrates **occurrence probabilities**, so it becomes *more* relevant under the new scope, not less | |
| `unet.MonotoneCalibration` | 206 | 🔢 `COUNT-REG` Monotone calibration head — works in **log1p space** and needs a non-negative prediction, i.e. a count regressor | |
| `module.log1p_huber` / `log1p_huber_quantile` | 33 / 46 | 🔢 `COUNT-REG` **Module-level losses NOT in `losses.py`** — `log1p` space is exactly the heavy-tailed-count trick; residual-mode losses in the wrong file | |
| `io/data.normalize_mode` | 47 | Mode-string normalisation (`hourly` / `daily`) — **stays**: the new scope still needs both modes | |
| `io/data.compute_target_transform_stats` | 434 | 🔀 `TRANSFORM` Fits the gamma (`gamma_shape`, `gamma_scale`) written to `target_stats.json`. **Delete** — sole consumer is `GammaFTransform.from_config` | |
| `io/data.compute_upstream_stats` | 506 | Upstream (residual mode) statistics | |
| `prepare_regression._materialize_upstream` | 500 | Writes the upstream prediction channel for residual mode | |
| `prepare_regression._daily_aggregation` (+`hourly_threshold`) | 182 | ✅ **`CLASSIF` — this is how the new target is built.** Aggregates hourly strokes into the daily `0–24` hours-with-lightning count; `hourly_threshold` is the single-stroke denoising knob (`2` = drop single-stroke hours). **Central to the new scope** | |

### D-only — the live MC-dropout model

| File / symbol | Line | Role | Decision |
|---|---|---|---|
| **`unet.enable_mc_dropout`** | 19 | Forces dropout layers active at inference — **the mechanism the whole family depends on** | |
| **`mc_dropout_module.MCDropoutModule`** | 75 | The current two-phase MC-dropout Lightning module | |
| `mc_dropout_module.build_output_activation` | 54 | Output activation factory (`max_hours` clamp) | |
| `mc_dropout_module_deprecated.py` | — | Superseded; drop | |
| `unet_aru.py`, `distr_regression_aru.py` | — | **Stale copies of aru's files** — the mirror image of aru's stale `mc_dropout/` package. Drop both | |

> Note the symmetry: each branch vendored a stale copy of the other's work. A has `modeling/mc_dropout/`
> (5 files); D has `unet_aru.py` + `distr_regression_aru.py`. **Both sets are deletable.**

## 3. The ensemble contract (from `registry.py` + notebook 02b)

The notebook documents the runtime contract better than any docstring:

```python
module = load_regression_module(path, map_location='cpu', model_family=None)  # None = auto-detect
module.eval_ensemble_size = M          # M > 1 => return the ensemble
out = module.predict_step(batch, 0)
```

`predict_step` returns a dict with:

| Key | Shape | Present when |
|---|---|---|
| `observation` | `[B, H, W]` | always |
| `prediction` | `[B, H, W]` | always (ensemble **mean** for stochastic families) |
| `ensemble_members` | `[B, M, H, W]` | only when the family is stochastic and `eval_ensemble_size > 1` |

Family dispatch is by **checkpoint marker** with `_sniff_family` as the fallback for legacy checkpoints
(merge task #7). `MCDropoutEnsembleModule` wraps MC forward passes into this same contract, which is what lets
MC-dropout and diffusion feed the *identical* `scores.ensemble_partials`.

**Residual mode**, also from the notebook:

```python
upstream = backbone.predict_step((x_feat, y), 0)['prediction']        # [B, H, W] raw space
x_cond   = torch.cat([x_feat, upstream.unsqueeze(1)], dim=1)          # upstream appended LAST
batch    = (x_cond, y, upstream)                                      # 3-tuple, not 2
# diffusion then returns clamp(upstream + residual)
```

Flagged by the `residual_target` attribute; channel count validated against `feature_mean.shape[0]`.

## 4. Portability — the mechanism already exists ⚠️ important for Step 5

D's branch carries **three launch scripts, one per machine**, and they reveal the existing pattern:

| Script | `DATA_ROOT` | Env activation |
|---|---|---|
| `mc_dropout.sh` | `/work/ext/st17/group8/data/` | `conda activate fers26p8` |
| `mc_dropout_jz.sh` | `/lustre/fswork/projects/rech/udt/uzn71za/batta_torch` | `source ~/.venvs/fers-minimal/bin/activate` |
| `mc_dropout_local.sh` | `/home/aburq/repos/fers/data/era5_post_process` | `source ~/.venvs/fers-minimal/bin/activate` |

**`DATA_ROOT` is an environment variable the configs already read.** Step 5 should formalise and document this
(and add a per-user config file) rather than invent a new mechanism — the pattern is proven across three machines.

**This also explains the rebuild plan's wrong data path.** `/home/aburq/repos/fers/data/era5_post_process` was
copied from `mc_dropout_local.sh` — it is correct *for the local machine*, wrong for this remote (where the data
is `/homedata/aburq/batta_torch`, note the same `batta_torch` naming as the Jean Zay script). Exactly the class of
error Step 5 exists to eliminate.

Both non-conda scripts also carry a commented-out `LD_LIBRARY_PATH` workaround: the node's system `libstdc++`
predates the one the compiled extensions need (`GLIBCXX_3.4.29`, gcc ≥ 11). Worth documenting as a known wart.

### Pre-existing `minimal_requirements.txt` on D

D already had one — I hand-built ours in Step 0 without knowing. They agree on the package **set**:

| | D (existing) | Ours (Step 0) |
|---|---|---|
| torch | `==2.0.1` | `>=2.4,<3` → resolved 2.13.0+cpu |
| lightning | `==2.0.9` | `>=2.4,<3` → 2.6.5 |
| numpy / scipy / sklearn | `==1.26.4` / `==1.11.3` / `==1.3.1` | loose → 2.4.6 / 1.17.1 / 1.9.0 |
| pandas | `==3.0.3` ⚠️ (likely a typo for 2.0.3 — pandas 3.x against torch 2.0.1 is implausible) | `>=2.2,<3` → 2.3.3 |
| mlflow / optuna / fire / matplotlib / PyYAML | present, mostly unpinned | present, bounded |
| **pytest** | ✗ | ✅ added |
| **cartopy** | ✗ | ✅ added (consistent with D's `reporting.py` having no geography) |

So the hand-build was sound — ours is that same set, newer and bounded, plus test and mapping deps. No change
needed, but D's file is worth citing as corroboration.

## 5. Pre-existing design docs on D — Step 1 source material

| Doc | Lines | Contents |
|---|---|---|
| `docs/metrics_and_losses.md` | 285 | Reference with LaTeX math: intensity weighting, weighted MSE, asymmetric Huber, Tweedie, Poisson NLL, focal BCE, hierarchy + target transform; then continuous / categorical / skill / calibration / spatial metrics; the tuning selection score; a best-practice checklist |
| `docs/distr_regression_pipeline.md` | 140 | Running it, stages and artifacts, code map, design decisions, extending |

Together with A's heavily-commented `config/metrics.yaml`, **most of Step 1's written material already exists.**
The work is consolidation and reconciliation against your annotations, not authoring from scratch.

## 6. Transform removal checklist (scope decision, 2026-07-28)

Every touchpoint of the gamma F-transform, so the removal can be done completely rather than leaving orphans.

| # | Location | What to do |
|---|---|---|
| 1 | `modeling/transforms.py` | **Delete the file** — `GammaFTransform` *and* `LogStandardizeTransform` both exist solely to warp an unbounded heavy-tailed target |
| 2 | `losses.TRANSFORM_COMPATIBLE_LOSSES` | Delete the constant |
| 3 | `losses.build_regression_loss(loss_config, transform_enabled)` | **Drop the 2nd parameter** and the guard that raises on incompatible combinations |
| 4 | `losses.REGRESSION_LOSSES` | Remove `'tweedie'`, `'poisson_nll'` |
| 5 | `losses.tweedie_deviance`, `losses.poisson_nll` | 🔢 Remove (unbounded-count likelihoods) |
| 6 | `search.apply_constraints` | **Both** constraints are transform-conditioned → the function becomes a no-op. Delete it, or keep as an empty hook for future constraints |
| 7 | `io/data.compute_target_transform_stats` | Delete; drop `gamma_shape` / `gamma_scale` from `target_stats.json` |
| 8 | `module.DistrRegressionModule` | Remove `transform_enabled` / `self.transform`, the `transform.forward(y)` call in the training step, and the `transform.inverse(...)` on the validation/predict path |
| 9 | `unet.MonotoneCalibration` | 🔢 Remove (log1p space ⇒ count regressor) — and with it the second `apply_constraints` rule |
| 10 | `module.log1p_huber` / `log1p_huber_quantile` | 🔢 Remove, or re-derive without log1p if residual mode needs them |
| 11 | `scores.tweedie_deviance_score`, `quantile_ratios`, `quantile_quantile` | 🔢 Remove or re-scope (see `inventory-scores.md`) |
| 12 | `scores.uniform_histogram_ks` (PIT) | 🔀 **Decide.** The function is generic (a uniformity test); the dependency is in `evaluation.py:468-470`, which reads `target_stats['gamma_shape']`/`['gamma_scale']` and calls `gammainc` to build the PIT values. Item 7 removes those parameters ⇒ those two call sites break. Either drop PIT, or re-derive without a fitted CDF (**ensemble-rank PIT**, needing only member ranks vs the observation) |
| 13 | `config/metrics.yaml` | Remove the `power: 1.9` Tweedie entry, `pit_histogram.space: transformed`, and the tail-quantile thresholds (see §"Threshold redefinition" in the scores inventory) |
| 14 | search-space YAMLs | Remove the whole `target_transform:` block (`enabled`, `zero_handling`, `clip_eps`, `gaussianize`) and the `calibration.regression` monotone option |
| 15 | Metrics-space invariant | `metrics.yaml` states *"all metrics are computed in the ORIGINAL target space… back-transformed through the inverse CDF first"*. **With no transform, training space == target space** — the invariant becomes trivially true and the back-transform plumbing can go |

**Net simplification:** item 15 is the real prize — training space and evaluation space become the same space, so
an entire class of "which space am I in?" bug disappears. That is the strongest argument for this scope decision.

**One thing the transform was genuinely buying**, worth naming so its loss is deliberate: on an unbounded
heavy-tailed target it conditioned the regression so rare extremes didn't dominate the gradient. On a **bounded
`0–24`** target that problem is far milder (the target spans one order of magnitude, not several), and for
**binary occurrence** it doesn't arise at all — `focal_bce_with_logits` handles the imbalance directly. So the
justification for dropping it is sound *for these targets*; it would need revisiting if an unbounded
stroke-count target ever returns.

## Open questions for you

1. **`log1p_huber` / `log1p_huber_quantile`** live in `module.py`, not `losses.py`, and never appeared in the
   losses inventory because of that. Move them into the unified `losses.py`, or leave them module-local?
2. **`PlattScaling` / `MonotoneCalibration`** (A-only calibration heads) — in scope, or drop with the classifier?
3. **`mc_dropout_eval.MCDropoutEnsembleModule`** differs in size between branches (line 37 vs 18). Want me to diff
   them properly before Step 3?
4. **Legacy stages** — keep A's `evaluate_distr_regression` alongside the unified `evaluate_regression` for
   backwards compatibility, or delete it?
5. **`compute_high_lightning_days`** — the plan says drop "unless kept as a utility". Which?
6. **Launch scripts** — port the three-script pattern (`*.sh` per machine), or replace with the Step 5 per-user
   config file, or both?

1. move them in the losses.py
2. In scope, keep, and extend to mc-dropout
3. Aru's branch is the more up-ot-date eval branch, so scores should come from there
4. delete
5. Keep
6. Keep only one, replace particularities with step 5 per-user config file.
