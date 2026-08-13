# Step 3 (block c) — Shared `src/utils`

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 2](step-2-config.md) · Next: [Step 4](step-4-stages.md)

> **Status: 🔵 next.** [Step 2](step-2-config.md) fixed the config contract this step implements. Build order and
> per-module content are settled below; the stages that call these modules are [Step 4](step-4-stages.md).

**Goal.** Rebuild `src/utils` as *one* shared library that the three families sit on top of, implementing exactly
what the annotated inventories kept. Every module here is a merge of two drifted branches, so each section below
names the **merge direction** and the **traps**.

`transforms.py` is **not** built — the target transform is dropped ([`00-context.md`](00-context.md)).

Source branches are read without checking out:
`git -C /home/aburq/repos/fers/fers26p8-knowledge-guided-tail-ml show origin/<branch>:<path>`.
**A** = `aru-probabilistic-eval` · **D** = `adrien-mc-dropout`.

---

## Build order

The user-specified order, which also happens to satisfy the import graph (each block only imports from blocks
above it):

| # | Block | Files | Merge direction |
|---|---|---|---|
| **1** | **Data I/O** | `io/data.py` | **A** (542 vs 353), minus the transform stats |
| **2** | **Metrics** | `metrics/scores.py` → `evaluation.py` → `reporting.py` → `diagnostics.py` | **A** for all four; port 1 function from D |
| **3a** | **Modeling: data** | `modeling/dataset.py` | **A**, + a **new** hourly 0/1 target path (on neither branch) |
| **3b** | **Modeling: shared model layer** | `unet.py`, `losses.py`, `search.py`, `validation.py`, `tuning.py`, `registry.py` | ⚠️ **mixed** — see each |
| **3c** | **Deterministic U-net** | `modeling/module.py` | **A** (452 vs 315) |
| **3d** | **MC-dropout** | `mc_dropout_module.py`, `mc_dropout_eval.py` | **D** for the module, **A** for the adapter |
| **3e** | **Diffusion** | `diffusion.py`, `diffusion_module.py` | **A** (`diffusion.py` is identical on both) |
| **4** | **Plotting** | `plotting/` figure functions per the 02a spec | new, from [`inventory-figures.md`](inventory-figures.md) §1 |

**Already in the template — do NOT rebuild:** `io/lazy.py`, `io/parse_config.py`, `seeding.py`, `banner.py`,
`plotting/__init__.py` (`show_plot_and_save`), `plotting/palettes.py`. The template's versions are
**byte-identical** to A's (md5-checked), not an evolved superset.

~~Update the plotting/palettes.py to include the 02a specs~~ — **not done, by decision.** The 02a colours live in
`plotting/maps.py` beside their only consumer, `make_lightning_cmap`; see Block 4 for the reasoning.

**Verification after every block:** `python -c "import ..."` on the touched module, plus `pytest` on its ported
tests. The end-to-end smoke run is Step 4's gate — nothing here is runnable on its own.

---

## ⚠️ Three traps to know before starting

1. **`modeling/unet.py` is the same path holding two different networks.** A's `unet.py` (349) has the calibration
   heads (`PlattScaling`, `MonotoneCalibration`), `Fp32BilinearUpsample`, and `DistrRegressionNet`, but **no**
   `enable_mc_dropout`. D's `unet.py` (236) has `enable_mc_dropout` but none of the calibration machinery. Each
   branch *also* vendors the other's copy (D's `unet_aru.py` = A's `unet.py`; A's `mc_dropout/unet.py` = D's
   `unet.py`). **The merged file must be the union**, and a naive "take A" silently loses the mechanism the whole
   MC-dropout family depends on.
2. **Three `DistrRegressionModule` classes exist** across the two branches (A's `module.py` 452, D's `module.py`
   315, D's `distr_regression_aru.py` 144). Take A's, rename to `DeterministicUnetModule`, delete the other two.
3. **`crps_ensemble` / `almost_fair_crps_ensemble` are a silent name collision** — A returns `float` and accepts
   `condition=`; D returns `np.ndarray` per element. Merging by name would fail *quietly*. Resolved in favour of
   A's contract; **write a test pinning the return type** (carried from
   [`rebuild-plan.md`](rebuild-plan.md) risk #9).

---

## 1. `io/data.py` — the data layer

**Take A** (542 lines; D's 353 is a subset). A relocated the mode constants here from `dataset.py`, which is the
layering we want: the mode vocabulary belongs with the data, not with the torch `Dataset`.

Keep, unchanged in spirit: `METADATA_JSON_FILENAME`, `METADATA_CSV_FILENAME`, `SAMPLES_DIRNAME`,
`TARGET_VARIABLE_NAME`, `MODE_DAILY`/`MODE_HOURLY`/`MODES`/`LEGACY_MODE_ALIASES`, `normalize_mode`,
`load_dataset_metadata`, `metadata_variable_names`, `index_samples`, `load_sample_tensor`, `SPLIT_NAMES`,
`load_split_config`, `assign_splits_by_year`, `assign_splits_by_sample_id`, `assign_splits_from_config`,
`compute_feature_stats`, `load_prepared_artifacts`, `compute_upstream_stats`.

**Changes:**

- ❌ **Delete `compute_target_transform_stats`** (checklist item 7) and drop `gamma_shape` / `gamma_scale` from
  `target_stats.json`. Its only consumer was `GammaFTransform.from_config`, which is gone. This also removes A's
  deferred in-function `import transforms`, so the module has **no** internal dependencies at all.
- `assign_splits_by_sample_id` + `cross_check: false` must work on a **subset** — the smoke splits
  (`config/split/split_smoke_{cpu,gpu}.yaml`) name 8 and 18 specific sample ids. Confirm samples not named by any
  range are dropped rather than erroring, and that the year cross-check is genuinely skippable.
- `compute_feature_stats` must accumulate in **float64** regardless of the stored `feature-dtype: float16`, so
  storage precision never affects normalization.

### The preparation stage splits in two, over one shared implementation

The hourly task is a classification, so `prepare_regression` is the wrong name for half of its job. Split it by
task:

| Stage | Mode | Target |
|---|---|---|
| `prepare_regression` | `daily` | 0–24 lightning-hours per cell |
| `prepare_classification` | `hourly` | 0/1 occurrence per cell per hour |

**Two stage scripts, one implementation.** Everything except the target derivation is shared: sample indexing,
split assignment, feature materialisation, the streamed feature statistics, and the upstream-prediction pass. So the
shared `_prepare_base` keeps all of it and only `_derive_target` branches, with each stage script being a thin
wrapper that hard-codes its mode. There is already precedent for exactly this on branch A, where
`prepare_distr_regression.py` is a 21-line shim delegating to `prepare_regression`.

Splitting this way has a benefit beyond the name: **`mode` stops being a stage parameter**, because the stage
identity carries it. That removes an entire class of misconfiguration — a `mode` that disagrees with the
`target-variable`, the selection score, or the output activation is no longer expressible.

The two stages' parameter sets genuinely differ, which is further reason not to force them into one signature:

- `feature-aggregation: hourly_stack` is **daily-only**. It stacks 24 hourly maps into `C*24` channels; in hourly
  mode the features are already one map per hour, `[C, H, W]`, with nothing to stack.
- `hourly-threshold` applies to **both**, and identically: it is the ≥2-stroke cutoff that decides whether an hour
  counts at all. Daily mode then counts the qualifying hours; hourly mode emits 0 or 1 per hour directly. Sharing
  the cutoff is what keeps the two tasks' denoising consistent (see Invariant 1 below).

`evaluate_regression` has the same naming problem but must **not** be split — it is the single shared evaluation, and
its metric suite already serves both tasks. It is renamed to **`evaluate`**, which drops the misleading word without
adding a second evaluation path.

> **⚠️ STAGE FLAG.** These are `src/stages/` changes, so they land in [Step 4](step-4-stages.md); they are recorded
> here because this step defines the shared implementation they wrap. The config consequences, for Step 3 or 4:
> - The nine pipeline files under `config/{deterministic_unet,mc_dropout,diffusion}/` name the stage
>   `prepare_regression`; daily pipelines keep that name, and any hourly pipeline uses `prepare_classification`.
> - Remove the `mode:` key from every `prepare_*` block — the stage name now carries it.
> - Rename the `evaluate_regression` stage to `evaluate` in all nine pipeline files and in the three
>   `config/eval/probabilistic_eval*.yaml` blocks.
> - `feature-aggregation` stays only in `prepare_regression` blocks.

### Two invariants this module must preserve

Both are relied on by the configs Step 2 wrote, and both are easy to break while refactoring, so they are written
out in full rather than left as a "check this" note.

**Invariant 1: the single-stroke denoising is permanent, and that is why evaluation has no threshold of its own.**
The raw data for one day is an hourly stack of stroke counts, `lightning[24, H, W]`. ATDnet occasionally logs a
single stroke in an hour that is a detection or location error rather than real lightning, and discarding those
hours is the denoising. `_daily_aggregation` (`prepare_regression.py:182-199`) applies it before any aggregation
happens: `kept = lightning >= hourly_threshold` marks the hours that qualify, and `kept.sum(axis=0)` counts them to
produce the 0–24 daily target. Sub-threshold hours are therefore excluded at the moment `targets/<date>.npy` is
written, so the file on disk excludes them permanently.

There are two consequences. First, evaluation only has to ask whether the stored target is positive, because any
positive value has already survived the two-stroke filter. This is why `config/eval/metrics.yaml` deliberately has
no `occurrence_threshold`, and why `resolve_occurrence_event` asserts that none has reappeared: an evaluation-side
threshold would be a second, competing definition of "this cell had lightning", and the two definitions could
disagree with each other. Second, changing `hourly_threshold` changes the target itself, so it requires re-running
preparation and re-training. It cannot be adjusted at evaluation time.

**Invariant 2: `max_samples` cannot express a small multi-split subset, which is why the smoke tiers need their own
split files.** After assigning splits, `prepare_regression` loops over `train`, `valid` and `test`, and raises
`ValueError('The "<name>" split is empty; ...')` if any of the three received no samples (`:384-386`).
`max_samples` is implemented as `index.head(N)`, which keeps the first N rows of an index ordered by sample id, and
that ordering is chronological from 2008-01-02. The year-based split assigns all of 2008 to *test*. So
`max_samples: 2` keeps sample ids 0 and 1, which are two January 2008 days and both belong to test. Train and valid
receive nothing, and preparation raises that error before doing any work. This is why the smoke tiers name explicit
per-split sample-id ranges in `config/split/split_smoke_{cpu,gpu}.yaml` instead of capping the number of days.

## 2. Metrics

Build in this order — `scores` has no internal deps, `evaluation` needs `scores` + `io/data`, `reporting` needs
`plotting/palettes`, `diagnostics` is standalone.

### 2.1 `metrics/scores.py` — **take A** (778 vs 550), port one function from D

A is a superset apart from two functions. Per [`inventory-scores.md`](inventory-scores.md):

- ➕ **Port `dice_coefficient` from D** — the only genuinely unique score on that branch, and it pairs with
  `dice_loss` in the occurrence head.
- ❌ **Drop D's `psd_full_fidelity`** as a function. **Unify `psd_fidelity` behind one `band:` argument**
  (`high`/`mid`/`low`/`full`); the *metric keys* `psd_full_fidelity` and `psd_high_fidelity` stay, served by it via
  the runner's `*_fidelity`-suffix rule. `config/eval/metrics.yaml` needs no change for this.
- ❌ **Remove** `tweedie_deviance_score` (🔢) and `uniform_histogram_ks` (the PIT test — its call site read the
  deleted gamma stats).
- 🔧 **`stratified_mae`** — its bins were positive-count quantiles; they are now the absolute hour bands.
- 🔧 **`fss_useful_scale`** — reuse the per-scale FSS already computed by `fss` instead of recomputing all scales.
- ➕ **New: `roc_auc`** — per event threshold, emitted through the categorical `<score>_<threshold>` grammar.
- ➕ **New: `explained_deviance`** — Bernoulli explained deviance on the occurrence head vs climatology,
  `1 - logloss(model)/logloss(clim)`. Binary head **only**: with Tweedie gone the bounded `0–24` target has no
  likelihood to take a deviance of.
- ⚠️ Resolve the `crps_ensemble` collision in favour of A + **add the return-type test**.

### 2.2 `metrics/evaluation.py` — **take A** (646 vs 555)

A's `run_metric_suite` + `merge_ensemble_partials` + `finalize_ensemble_metrics` are *the* single evaluation path.

- ❌ **Retire D's `regression_metric_suite` and `EnsembleProbabilisticAccumulator`** — the whole point of the merge
  is that there is one suite and one streaming accumulator.
- ❌ **Remove the persistence baseline** from `build_baselines` and `_climatology_tables`. Not applicable to a
  diagnostic parameterization (no past observation at inference).
- ❌ **Remove the PIT block** (`evaluation.py:468-470` on A) that read `target_stats['gamma_shape']`/`['gamma_scale']`
  and called `gammainc`. Those keys no longer exist, so these are hard breaks, not optional cleanups.
- ✅ **`resolve_threshold` already supports `kind: absolute`** (it is the default kind) — the hour bands work with
  no change. `resolve_occurrence_event` keeps its hard assertion that no `occurrence_threshold` reappears.
- ➕ Wire the two new scores into the suite: `roc_auc` into the categorical group's score loop,
  `explained_deviance` into the skill group (emitting the flat key `explained_deviance`).
- 🔒 **Streaming contract, do not break it:** `crps_sums` / `spread_skill_sums` return `(sum, …, n_cells)` because
  the full `[N, M, H, W]` stack does not fit in memory. Sums are additive across batches; means and ratios are
  not — divide exactly **once**, at the end, in `finalize_ensemble_metrics`.

### 2.3 `metrics/reporting.py` — **take A** (911 vs 198; D's is superseded wholesale)

D's 198-line version renders raw pixel indices with no projection; A's has the full map + residual suite. Only
`write_report` is shared API.

- 🔧 **Restyle the map figures to the 02a spec** ([`inventory-figures.md`](inventory-figures.md) §1): cartopy
  `EuroPP` axes with a `PlateCarree` data transform, `origin='upper'` (row 0 = north), integer unit bins in
  lightning-hours, warm/cool over/under diff encoding. Replace `_resolve_map_norm`/`_quantize_field` with a
  `make_lightning_cmap`, and `_count_panel`/`_std_panel`/`_overunder_panel` with `draw_map`/`draw_diff_map`.
- 🔧 **Rename** `_maps_per_day` → `maps_most_extreme_days` and **split** `_reliability_and_pit` → `reliability`
  (PIT half deleted).
- ❌ **Remove** `LogNorm` / `colorbar_scale: log` (🔢) and the never-implemented `qq_plot`.
- ➕ **New: `roc_pr_curves`** (ROC + PR per threshold — plotting both is the point: at a 0.07 % base rate ROC
  flatters while PR exposes the real trade-off) and **`confusion_matrix`** (2×2 counts per threshold).
  Both must be registered in whatever `write_report` uses to dispatch `reporting.figures` names.
- **How one config serves every family without branching.** `write_report` holds a dictionary mapping each figure
  name to a lambda, and each line-or-table figure function begins by fetching its entry from the `curves` dictionary
  and returning immediately if it is missing. A deterministic run never populates `curves['rank_histogram']`, and a
  non-residual run never populates `curves['residual']`, so those figures skip themselves and
  `config/eval/metrics.yaml` can list all fourteen unconditionally. Two caveats worth knowing before adding figures:
  - `maps_most_extreme_days` does **not** consult `curves` at all. It receives `prediction`, `observation`, `items`
    and `ensemble_members` as direct arguments and picks its layout by checking whether `ensemble_members` is
    `None`. So the self-skipping mechanism covers the line and table figures only, not the map figure.
  - Every figure call is wrapped in `try`/`except Exception` that only logs a warning, on the reasoning that a broken
    figure must not lose the whole run. A genuine bug in a figure is therefore swallowed in exactly the same way as a
    deliberately absent curve. This matters here because this step adds two new figures: if `roc_pr_curves` raises,
    the run still reports success and the only trace is a warning in the log.

### 2.4 `metrics/diagnostics.py` — **take A** (298, A-only), as-is

`residual_diagnostics` populates `curves['residual']`, which is what makes the six `residual_*` figures appear only
for a residual diffusion run. No changes needed.

## 3. Modeling

### 3a. `modeling/dataset.py` — **take A** (247 vs 139)

A has `DayGroupedShuffleSampler` (needed for hourly mode) and drops the mode constants in favour of `io/data`'s.

- ➕ **Residual/upstream channel:** append the upstream prediction as the **last** conditioning channel and yield a
  third batch item, `(x_cond, y, upstream)`. Validate the channel count against `feature_mean.shape[0]`.

#### The mode vocabulary — one axis, not two

The two branches disagree about what `mode` means, and neither matches what this project now needs. The
reconciliation:

| Branch | Scheme |
|---|---|
| **D** | Three modes that conflate aggregation level with quantity: `daily_lightning_hours`, `hourly_counts`, `hourly_occurrence`. |
| **A** | Two axes. `mode` is `daily` or `hourly` and names only the aggregation level; `target_variable` names the quantity. D's three names are accepted as `LEGACY_MODE_ALIASES` and produce a deprecation warning. |
| **Ours** | **One axis.** `mode` fully determines the target and the task, and `target_variable` is deleted. |

| `mode` | Target | Task |
|---|---|---|
| `daily` | hours per cell per day with lightning, a **0–24** integer | **regression** |
| `hourly` | **0 or 1** per cell per hour | **classification** |

Collapsing to one axis is coherent rather than merely convenient. The other two target quantities that used to
exist, `lightning_counts` and `lightning_peak`, are unbounded counts and are out of scope under the
classification-first framing. Once they are gone, each mode has exactly one legal target, so a second axis carries
no information at all.

The concrete work in `io/data.py` and `prepare_regression`:

- Delete the `target_variable` parameter, the `TARGET_VARIABLES` tuple, and the diagnostic that compared the
  zero-proportion of all three daily aggregations. Remove `target_variable` from `target_stats.json`.
- **Drop the `hourly_counts` legacy alias.** Keeping it would map a request for unbounded hourly counts onto a
  binary target, which changes the meaning of the request while looking like a harmless rename. The
  `daily_lightning_hours` alias is safe and can stay, because it still resolves to the same target it always did.
- ➕ **The hourly 0/1 target has to be written from scratch; it is not a port.** On branch A, `mode: hourly` forces
  `target_variable = 'lightning_counts'` (`prepare_regression.py:330`), which is the unbounded hourly count and is
  out of scope. The only implementation that produces a binary target is D's `hourly_occurrence`, and D's
  preparation stage is the one being retired. So `_derive_target` needs a new hourly branch that thresholds to 0 or
  1 using the same `hourly_threshold` cutoff as the daily path, so that both tasks denoise identically.
- `normalize_mode` survives, now covering two modes and one alias.

> **⚠️ CONFIG FLAG — MODES.** This decision requires editing configs that Step 2 already wrote. Recorded here as a
> Step 3 work item rather than Step 2 rework:
> - Remove the `target-variable:` line from the `prepare_regression` block of all nine pipeline files under
>   `config/{deterministic_unet,mc_dropout,diffusion}/`. Its current value, `daily_lightning_hours`, is rejected
>   outright by `prepare_regression.py:310-311` in any case, because it is a legacy *mode* alias rather than a
>   target-variable value. Every pipeline would fail at its first stage as currently written.
> - Remove the `mode:` key as well. Once the preparation stage is split by task (see §1, "The preparation stage
>   splits in two"), the stage name carries the mode and the key is redundant. Note this supersedes an earlier
>   version of this flag that said to *extend* the `mode:` comment.
> - Rename the prepared-data leaf directory from `daily_lightning_hours` to `daily`, so the path states the task and
>   nothing more. This touches `output-path` on `prepare_regression` and `input-path` on `tune`, `retrain_best` and
>   the evaluation stage in all nine files, plus the three evaluation blocks in `config/eval/`.
> - `CLAUDE.md` refers to the target as `daily_lightning_hours`. Under a single-axis scheme that string is the
>   informal name for the daily task rather than a config value, which is worth one clarifying sentence there.

### 3b. Shared model layer

**`unet.py` — UNION, not a pick.** A's backbone + `ConvBlock` + `BottleneckAttention` + `Fp32BilinearUpsample` +
`UpBlock` + `UNetBackbone` + `PlattScaling` + `MonotoneCalibration` + `REGRESSION_CALIBRATION_STRUCTURES`, **plus
D's `enable_mc_dropout`**. Delete `unet_aru.py`, `distr_regression_aru.py`, and A's whole vendored
`modeling/mc_dropout/` package (6 files, ~1930 lines).
`MonotoneCalibration` **stays** — checklist item 9's removal is struck (Step 1 Q2), and the search spaces expose it
as `calibration.regression`. Rename `DistrRegressionNet` → `DeterministicUnetNet`.

**`losses.py` — merge direction is D → shared** (408 vs 130; D is a strict superset). This is the *opposite* of
the scores file, which is the easy mistake to make.
Keep: `intensity_weights`, `_weighted_masked_mean`, `weighted_mse`, `weighted_mae`, `weighted_rmse`,
`asymmetric_huber`, `psd_penalty`, `wmae_psd`, `afcrps_psd`, `focal_bce_with_logits`, `dice_loss`, `brier_loss`,
`crps_binary`, `crps`, `almost_fair_crps`.
- ❌ Remove `mae`, `rmse` (absorbed at `γ=0`), `tweedie_deviance`, `poisson_nll`, `TRANSFORM_COMPATIBLE_LOSSES`,
  `build_finetune_loss`; drop `'tweedie'`/`'poisson_nll'` from `REGRESSION_LOSSES`.
- ➕ **New `wmse_psd`** — `alpha * weighted_mse + (1 - alpha) * psd_penalty`, the `weighted_mse` sibling of
  `wmae_psd`. A few lines reusing both.
- 🔧 `build_regression_loss` loses its `transform_enabled` argument and **absorbs the `finetuning` option**; same
  for `build_binary_loss`. Document the config alias → function map: `focal_bce`→`focal_bce_with_logits`,
  `dice`→`dice_loss`, `brier`→`brier_loss`.
- 🔒 **Every pointwise loss reduces through `_weighted_masked_mean`**, which normalises by the *sum of effective
  weights*, not the cell count. Inlining it puts one loss on a different scale from its siblings and makes tuning
  results incomparable.
- 🔒 `intensity_weights(y, γ) = (1 + y)^γ` from the **raw** target, and **`γ` must reach 0.0** — that is what makes
  `weighted_mae ≡ mae`. The search spaces sample `γ ∈ [0, 5]`.
- 🔒 The CRPS in `losses.py` and in `scores.py` must agree; `scores.crps_ensemble` is the reference contract.

**`search.py`** — identical on both branches; take either. But `apply_constraints` is **not** the no-op checklist
item 6 predicted: both its old rules were transform-conditioned and die, and **two new rules replace them**:
1. `occurrence_head.loss == focal_bce` ⇒ force `intensity_weight_gamma = 0`. On a binary target `(1+y)^γ` collapses
   to `{1, 2^γ}`, duplicating `positive_class_weight`; their product is uninterpretable.
2. `tune` given an `upstream-model-path` ⇒ force `finetuning.enabled = true` and **ignore the `unet:` block**
   (architecture comes from the checkpoint).

**`validation.py`** — A-only (128). It currently holds one selection score; it needs **two**, one per task, both
retiring the old `valid_tail_score` name:

| `valid_regression_score` (`mode: daily`) | weight | `valid_classification_score` (`mode: hourly`) | weight |
|---|---|---|---|
| `mae_cond_ss_climatology` | 0.60 | `average_precision_occurrence` | 0.50 |
| `psd_full_fidelity` | 0.40 | `brier_skill_score` | 0.20 |
| | | `psd_full_fidelity` | 0.30 |

The classification composite uses `brier_skill_score` where the regression one uses `mae_cond_ss_climatology`,
because on a target that is only ever 0 or 1 a proper probabilistic score is far more informative than a mean
absolute error, which on binary values reduces to little more than a rescaled error rate. `average_precision` leads
that composite because it is the base-rate-robust discrimination measure and discrimination *is* the classification
task. `psd_full_fidelity` appears in both composites and serves the same purpose in each: it stops an over-smoothed
model from winning. The cost is that the two tasks no longer share a common point term, which is acceptable because
comparing a 0–24 hour error against a binary error was never meaningful.

Which composite applies is **derived from the prepared data's `mode`**, not read from config, so there is a single
source of truth. `validation.py` should raise if the search space's `selection.metric` disagrees with the name
derived from the mode, so that flipping the mode and forgetting to update the score fails loudly instead of quietly
optimising the wrong objective.

`compute_selection_components` therefore has to emit four component keys rather than three:
`mae_cond_ss_climatology`, `psd_full_fidelity`, `average_precision_occurrence` and `brier_skill_score`. Delete D's
in-module `valid_regression_score` so that MC-dropout validates through this same shared path.

> **⚠️ CONFIG FLAG — SELECTION SCORES.** All three files
> `config/{deterministic_unet,mc_dropout,diffusion}/search_space.yaml` currently carry the single composite decided
> earlier in Step 2, which was `0.40 average_precision_occurrence + 0.30 mae_cond_ss_climatology +
> 0.30 psd_full_fidelity`. Every pipeline runs `mode: daily`, so each `selection:` block should become
> `metric: valid_regression_score` with `mae_cond_ss_climatology: 0.60` and `psd_full_fidelity: 0.40`. The
> `average_precision_occurrence` term is removed from those files, because it belongs to the classification
> composite that a `mode: hourly` pipeline would select. The explanatory comment in each block currently derives the
> PR-AUC weighting and needs rewriting to match.

**`tuning.py`** — A-only (942), the largest single piece of work. Generalize `run_sweep` / `_fit_trial` /
`retrain_best_config` to serve all three families:
- ➕ Fold D's `_fit_phase` **two-phase** train→finetune fit into the shared harness *without* regressing
  single-phase families — verify monitor and best-weight-restore parity for both shapes.
- ➕ **Warm-start path (new, exists on no branch):** when handed an `upstream-model-path`, load the upstream U-net's
  weights into the MC-dropout net, **skip phase 1**, run the finetuning phase alone.
- ➕ **`selection` is read from `model-config` and recorded into `best_trial.json`**; `retrain_best_config` reads it
  back from `source-path/best_trial.json` and takes **no** `model-config` and no `selection-metric`/`-mode`. This
  is what makes a retrain/sweep mismatch unrepresentable (Step 2 decision 3).
- Keep `_check_retrain_staleness`: a structural change (mode/target/features/residual/channels) is a hard error;
  a changed target distribution or code state only warns.

**`registry.py`** — take A (150 vs 111).

This file is the mechanism that makes one evaluation stage work for all three families, so it is worth stating what
it does before changing it. Each module writes a `module_class` marker into its checkpoint from
`on_save_checkpoint`. The loader then resolves the family in three steps, in priority order: an explicit
`model_family` argument wins if given; otherwise the checkpoint's marker is used; otherwise `_sniff_family` guesses
from the saved trial configuration, which works because the families have disjoint sections (a `flow` or
`transformer` section means diffusion, an `mc_inference` section means MC-dropout). Having resolved a family, the
loader instantiates the matching class and, if that family registers one in `EVAL_WRAPPERS`, wraps it in an
evaluation adapter — which is how MC-dropout arrives at the shared evaluation already satisfying the ensemble
contract. There are only two callers: `evaluate_regression.py:191` and `prepare_regression.py:559`, the latter being
the upstream-prediction pass.

Three changes:

- 🔧 **Rename `load_regression_module` to `load_model_module`.** Now that hourly classification is in scope, the
  families this function returns are not all regressors, so the current name is misleading. Update both call sites.
- 🔧 Change the `MODULE_REGISTRY` keys to the new family tokens, `deterministic_unet`, `mc_dropout` and `diffusion`,
  and update `DEFAULT_MODULE_CLASS` to match. Keep `_sniff_family` so older checkpoints still load.
- ❌ **Delete `_mc_dropout_eval_overrides`.** It exists only because A's stale vendored copy of `losses.py` cannot
  build `afcrps_psd`, so it disabled the finetuning section when loading a checkpoint for evaluation. Once there is
  a single shared `losses.py` implementing every loss we keep, the workaround has nothing left to work around.

### 3c. Deterministic U-net — `modeling/module.py`

**Take A** (452 vs 315), rename `DistrRegressionModule` → **`DeterministicUnetModule`**.
- ❌ Remove `transform_enabled` / `self.transform`, the `transform.forward(y)` in the training step, and the
  `transform.inverse(...)` on the validation/predict paths (checklist item 8). **This is the prize**: training space
  == evaluation space, so the whole "which space is this tensor in?" class of bug disappears.
- ❌ Remove `log1p_huber` / `log1p_huber_quantile` (🔢) — or re-derive without log1p if residual mode needs them.
- Keep A's calibration `PHASES` (`joint`, `classifier`, `regressor`, `classifier_calibration`,
  `regression_calibration`) — both calibration heads are in scope.
- 🔧 **Output head is mode-dependent** (see "Settled: the output head, per mode" below): `softplus` plus a
  `predict_step` clamp to `max_hours` for daily, `sigmoid` on a raw logit for hourly. Never clamp during training.
  In hourly mode this module is the occurrence classifier, so `PlattScaling` composes *inside* the head as the
  affine term before the sigmoid — it is not a second nonlinearity stacked on top.
- Must write a **checkpoint family marker** and use group norm, so it is a valid upstream for both stochastic
  families.

### 3d. MC-dropout

**`mc_dropout_module.py` — take D** (398; A's copy is the stale vendored one). Confirmed stale by identical
function order with newer additions missing.
- 🔧 **Add the warm-start entry point** — the module has `set_phase('train'|'finetune')` and a `finetuning_enabled`
  gate but **nothing that loads foreign weights**. Needs an init-from-checkpoint path plus a load-time
  compatibility check over ~~`in_channels`/`base_channels`/`depth`/`activation`/`normalization`~~ that **raises naming
  the offending field**. A silent partial `load_state_dict` is the failure mode to prevent.
  > ⚠️ **Corrected during block 5a — the field list above is wrong.** Only `in_channels`, `mode` and a missing
  > `hyper_parameters.trial` raise; the sampled ARCHITECTURE (`base_channels` / `depth` / `activation` / …) is
  > **overridden from the checkpoint and logged**, never rejected. Rejecting it would fail 26 warm-start trials in 27,
  > since the sweep samples architecture independently of the frozen upstream — and the override is what discharges
  > `apply_constraints`' "the sampled `unet` block is ignored" obligation. `WARM_START_ARCHITECTURE_KEYS` names the
  > fields whose discard is REPORTED, not fields that must agree; its own comment said otherwise until 5a fixed it.
- 🔧 `build_output_activation` — dispatch on the mode: plain `softplus` for daily, `sigmoid` for hourly (see
  "Settled: the output head, per mode" below). In daily mode the `max_hours` ceiling is applied by a **clamp in
  `predict_step`**, not by the activation, and the ensemble path must clamp each member, not just their mean. In
  hourly mode no clamp is needed and the head must emit a **logit** for the loss while the sigmoid is applied on the
  prediction path only.
- ❌ Delete `mc_dropout_module_deprecated.py`.
- Normalization is **group only**: MC inference re-enables dropout while the rest of the model is in `eval()`, and
  batch-norm running statistics would shift under that.

**`mc_dropout_eval.py` — take A** (116 vs 76). `MCDropoutEnsembleModule` re-expresses MC forward passes in the
shared ensemble contract, which is what lets this family feed the *shared* `scores.ensemble_partials` rather than a
bespoke accumulator.

### 3e. Diffusion

**`diffusion.py`** — byte-identical size on both branches; take either as-is (`FlowVelocityNet`, `DiTBlock`,
`ConvStem`/`ConvDecoder`, `flow_matching_targets`, `sample`).

**`diffusion_module.py` — take A** (470 vs 175).
- ❌ Remove the transform plumbing and the `log_warp` generation-space switch — it selected between a
  log1p-standardized and a raw space, i.e. it *was* the target transform under another name. One fixed generation
  space means `valid_regression_score` is comparable across every trial.
- **Residual mode:** conditions on the upstream prediction (appended **last**), returns
  `clamp(upstream + residual, 0, max_hours)`, flagged by the `residual_target` attribute. This is the same
  inference-time clamp the other two families apply; the full-target path needs it too, and each ODE draw must be
  clamped, not only the ensemble mean.
- Rank trials on the target-space composite, **not** `valid_flow_loss` — the flow loss lives in the generation
  space and says nothing about occurrence skill or structure fidelity.

## 4. Plotting — ✅ DONE (98/98 gate checks)

Figure functions in `src/utils/plotting/`, built to the 02a spec
([`inventory-figures.md`](inventory-figures.md) §1) on top of the template's existing `show_plot_and_save` and
`palettes`. Keep `_geographic_context` and `_select_plot_indices`; add `make_lightning_cmap`, `draw_map`,
`draw_diff_map`. **Notebooks are not ported.**

**Most of this shipped early, inside Block 2.** Restyling `reporting.py` to the 02a grammar is not separable from
writing the grammar, so `src/utils/plotting/maps.py` (the colour system, the projection, `draw_map` /
`draw_diff_map` / `add_shared_diff_colorbars`) landed with the metric suite in commit `1133cb6`, and all fourteen
figures came with it. What was left for this block was therefore an audit against §1 plus the gate — and the audit
found one real deviation:

- 🐛 **cartopy was still optional.** `geographic_context` caught `ImportError` and returned `(None, None)`, and six
  call sites branched on it, against §5 answer 2 ("keep cartopy as a hard requirement"). cartopy is a hard
  dependency in `minimal_requirements.txt:39`, so the fallback was unreachable in any working install and reachable
  only in a broken one — where it would emit figures in raw pixel indices that look plausible and are not maps,
  which is exactly what makes branch D's reporting unusable (§3). Fixed: module-scope `import cartopy.crs as ccrs`,
  the fallback deleted, and the dead `is not None` branches collapsed in `maps.py` and `reporting.py`.
  `projection` also leaves `draw_map` / `draw_diff_map` / `frame_map_axis`, whose only use for it was the `is None`
  test; axis construction still takes it.

### Two ambiguities resolved

1. **§5 answer 1, "Discard coastlines", meant discard BORDERS.** The question offered "add borders" or "coastlines
   only" and the answer named neither, so it read as removing the basemap entirely. Confirmed with the user
   (2026-08-12): **coastlines stay, country borders are not added.** The gate pins both halves, so neither drifts.
2. **The 02a colours stay in `maps.py`, not `palettes.py`** — a deliberate departure from this file's line 40. The
   warm/cool/grey ramps define the *lightning-hours value axis* and have exactly one consumer,
   `make_lightning_cmap`, in the same file; `palettes.py` holds the general-purpose IBM/Tol design libraries that
   reach the line figures through the `rcParams` prop cycle. Moving them would separate `make_lightning_cmap` from
   its own data for no second caller. `palettes.py` is therefore untouched by Step 3.

**Known gap, deliberately left open** (inventory-figures.md §4): `make_lightning_cmap` has no probability scale, so
an hourly run's maps collapse to two colours (`nanmax(obs) == 1` ⇒ levels `[0, 0.5, 1]`). By decision, not oversight.

**Gate** (`gate_block4.py`, 98 checks) — the emphasis is on what no metric can catch, because every score in this
repo is computed on the arrays rather than on the rendered picture:

- **The north-edge claim, proved end to end by rasterising.** Two probes light one array row each, north and south,
  and their painted centroids are compared in image coordinates. Mutation-tested: flipping `origin` to `'lower'`
  trips five checks. Not readable off the artifact — cartopy regrids the field into the target CRS and re-emits it
  as `origin='lower'` in projected metres — so the source kwargs are checked by AST alongside, and the
  reprojection itself is pinned so nobody "fixes" the source to match what they see.
- Rows are chosen inside the displayed window: §5 answer 5 crops the view at 55 °N, which hides the **top ~20 array
  rows** of every map. The gate asserts that crop, so it stays a decision rather than a surprise.
- The colour axis: 26 boundaries / 25 bands at `max_val = 24`, white `[0, 0.5)`, grey `[0.5, 1)`, ceil-rounding, the
  degenerate `max_val <= 1` floor, and warm/cool sharing an identical boundary array (the comparability invariant
  the two diff colorbars rest on).
- The over/under encoding, tested **behaviourally**: a field that over-predicts west of 7.5 °E and under-predicts
  east of it must come out red on the left and blue on the right, and `pred == obs` must render warm — otherwise a
  perfect forecast reads as under-prediction. Swapping the two masks trips both checks.
- Both layouts (panel and colorbar counts, figsize, the `M = 2` smoke case blanking its third slot), the std panel's
  vmax being **derived from the day** rather than 02b's hardcoded `vmax=8` (§5 issue 2), the PSD figure's kilometre
  axis and inverted x, and `metrics.yaml` ⇄ `write_report` parity in **both** directions (no unconfigured builder,
  no undispatched figure, `qq_plot` absent from both).

## 5a. Tests — ✅ DONE (598 passing, 613 collected, 14 skipped)

`tests/` **mirrors `src/`**: every directory under `src/` has the same directory here, and every non-`__init__` module a
`<module>_test.py` beside it. 32 test files for the 27 modules plus the two `__init__.py` that carry real code
(`plotting`'s `show_plot_and_save`, `stages`' seeding hook, as `init_test.py`) and three Step-4 placeholders.

`<module>_test.py` **singular** is deliberate — it is one of pytest's two default patterns, so no `python_files`
override is needed and there is no config whose absence silently collects zero tests and exits 0.

`tests/completeness_test.py` is a meta-test that makes the layout enforceable rather than aspirational: the mirror is
complete in **both** directions (no module without a test file, no test file outliving its module), and every function
in `src/` is referenced by some test. The second is `xfail` until 5c — it currently reports **137 of 291 untested**,
which is 5c's work-list.

### Where A's 1479 lines went

All four of A's non-stage files were ported. `test_ensemble_scores.py` split across `metrics/scores_test.py` +
`evaluation_test.py`; `test_residual_diagnostics.py` across `metrics/diagnostics_test.py` +
`modeling/diffusion_module_test.py` + `metrics/reporting_test.py`; `test_probabilistic_eval_compat.py` across
`modeling/registry_test.py` (the cross-family `predict_step` parity — making the families interchangeable is
registry.py's job) + `mc_dropout_module_test.py` + `mc_dropout_eval_test.py`. A's two stage-level files became the
three skipped placeholders, with their paths **pre-fixed** for Step 2's contract so Step 4 deletes one `pytestmark`
line per file rather than re-deriving the adaptation.

**Five of A's tests were deliberately dropped**, each replaced by a positive test of what superseded it:
`test_resolve_map_norm_quantize_and_cap` + the two `quantize` layout tests (the whole map-colour configuration surface
is gone under the 02a grammar) and `test_mc_dropout_loads_for_eval_despite_unimplemented_finetune_loss` (it asserted
`registry._mc_dropout_eval_overrides` exists; the replacement asserts `build_ensemble_loss('afcrps_psd')` now builds,
so the workaround's premise is gone). A's four-pair `combine_curves` test lost its `qq` pair with `quantile_quantile`.

### Findings — the code was right, my premises were wrong, ~12 times

Worth recording because several are things the plan documents *incorrectly*:

1. **`mc_forward` DOES clamp.** The ceiling is in `_to_prediction`, applied per member — so A's `<= 24` assertion
   survives for a different reason than A's `scaled_sigmoid`. Only `_to_prediction_differentiable` skips it.
2. **`from_upstream` OVERRIDES architecture rather than rejecting it**, and raises on `in_channels` / `mode` /
   missing hyperparameters. This file's block 3d text says it "raises naming the offending field" for
   `base_channels` / `depth` / `activation` — the code is right and the text is stale, since rejecting those would
   fail 26 warm-start trials in 27. `WARM_START_ARCHITECTURE_KEYS`' comment ("must agree") also overstates: it only
   drives the override *log*.
3. **adaLN-Zero makes a fresh `FlowVelocityNet` emit exactly zero velocity**, so any "changing the input changes the
   output" test is vacuously false at initialisation. Now pinned as a property, with a `perturbed_net` fixture for the
   sensitivity tests.
4. **`UNetBackbone` cannot take the real 101 × 149 grid** — 101 → 50 → 100 breaks the skip concat. `DeterministicUnetNet`
   pads to a multiple of `2 ** depth` and crops back. **Every Step 3 gate used a 24 × 32 fixture, divisible by 8, so
   this was never exercised**; the test now runs the real grid at depths 3–5.
5. **`ConvBlock` emits `Dropout2d` only when `dropout > 0`** — the layer is omitted, not inert. Which is exactly why
   `MCDropoutModule` must reject `dropout_p <= 0`.
6. **No shipped search space offers a binary loss** — all three are daily spaces, consistent with the config's own note
   that an hourly space "just sets `loss.name` to one of focal_bce / dice / brier / crps_binary".
7. **`apply_constraints` mutates in place.** Harmless at its call site (`trials.csv` records the repaired trial), but
   the optuna path diverges from its own record: `study.ask()` registers the sampled gamma, the repair forces it to 0,
   and `study.tell` attributes the score to the phantom value. Unreachable today (focal_bce only). Documented.
8. **`build_ensemble_loss` ignores `enabled`** — the phase gate lives in the module.
9. **`DiffusionModule` keeps its marker as a module-level constant**, not a class attribute, being standalone. So the
   marker contract must be checked through `on_save_checkpoint`, not `cls.CHECKPOINT_MARKER`.
10. **Stage modules are not importable as `src.stages.X`** — `from __init__ import root_path` resolves only with
    `src/stages/` on `sys.path`. That is the convention (it triggers the seeding hook), and `tests/stages/conftest.py`
    replicates it at *module* scope, because the test files import their stage during collection.
11. **`run._coerce_bool` falls back to the default for `None` only** — any unrecognised string becomes `False`, so a
    typo'd `lazy: ture` silently disables the cache. Documented; `False` is the safe direction for every flag it governs.
12. **`show_plot_and_save` with a placeholder-less pattern silently writes every figure to one file.**

### Notes

- `reporting_test.py` was 216 s before the cartopy renders were shared via module-scoped fixtures; 96 s after. The
  whole suite is ~150 s.
- Three files are **thin by design** and say so (`banner`, `stages/hello_world`, `stages/setup` — untouched template
  code); 5c gives their functions real tests under the every-function requirement.
- The nine `gate_block*.py` scripts still pass **788 / 0** unchanged, so nothing here required a source edit.

---

## Settled: the output head, per mode

### The activation is determined by the mode, not searched

The two tasks need different output ranges, and in each case there is exactly one correct activation, so there is
nothing to tune:

| `mode` | Activation | Range | Ceiling |
|---|---|---|---|
| `daily` | `softplus` | 0 to ∞ | **not** structural — see the clamp decision below |
| `hourly` | `sigmoid` | 0 to 1 | structural; a probability cannot leave the interval |

`softplus` gives the 0-hour floor structurally, which is why `min_hours` was dropped. `sigmoid` is what makes the
hourly head emit a genuine probability, which is what `brier_skill_score`, `reliability_diagram`,
`explained_deviance` and the ROC/PR curves all require: none of them is meaningful on an unbounded score.

Because the mode fixes the answer, the activation is **derived from the mode in code** rather than read from config.
This is the same reasoning that removed `min_hours` and that picks the selection score: a config key with one legal
value per mode adds a way to be wrong and no way to be right.

> **⚠️ CONFIG FLAG — OUTPUT ACTIVATION.** Remove the `output_activation` key from all three
> `config/{deterministic_unet,mc_dropout,diffusion}/search_space.yaml` files. Make `max_hours` daily-only, and say so
> in its comment: in hourly mode the ceiling is 1 and is guaranteed by the sigmoid, so `max_hours` does not apply.
> This also retires the dangling cross-reference those three files currently carry to the deleted
> "open question: where max_hours is enforced" section.

### The Platt-scaling interaction: one sigmoid, not two

`PlattScaling` is `p = sigmoid(a·z + b)` and it operates on the **logit** `z`, not on a probability. So the two are
the same operation with and without a fitted affine term, and they compose only in one order:

```
network -> logit z -> [optional Platt: a·z + b] -> sigmoid -> probability
```

Applying the sigmoid first and then Platt would squash an already-squashed value, which flattens the probability
range and quietly destroys calibration. So the network must emit a **logit** in hourly mode, the head applies the
sigmoid last, and enabling `calibration.occurrence: platt` inserts the affine term *before* it rather than adding a
second nonlinearity. When Platt is disabled the head is a bare sigmoid, which is the identity-initialised case of
the same expression.

### ⚠️ The loss path needs logits, the prediction path needs probabilities

This is the same trap as the clamp below, so treat it the same way. `focal_bce_with_logits` takes **logits** — its
numerically-stable formulation requires them, and feeding it probabilities is silently wrong rather than an error.
`dice_loss`, `brier_loss` and `crps_binary` take **probabilities**. The module must therefore keep the raw logit
available and hand each loss the space it expects, applying the sigmoid only on the prediction path
(`predict_step`, validation scoring, metrics). `build_binary_loss` should record which space each loss wants so this
cannot be got wrong by accident.

### The `max_hours` clamp: at inference, not during training

In daily mode `softplus` is unbounded above, so the activation alone does not enforce the 24-hour ceiling.

**Decision: train without clamping, and clamp inside `predict_step`.** Training stays unclamped so gradients remain
live everywhere — a hard clamp during training has zero gradient above the ceiling, which lets a badly-scaled model
get stuck there. Clamping at prediction time means no physically impossible value (more than 24 hours in a day) ever
reaches the metric suite, the report figures or the residual reconstruction. `build_output_activation` therefore
returns a plain `softplus` for daily mode, and the clamp is one line in each of the three modules' `predict_step`
(see §3c, §3d, §3e). Hourly mode needs no clamp at all, since the sigmoid already bounds the output.

## Carried into this step from earlier steps

- **The transform-removal checklist** — [`inventory-architecture.md`](inventory-architecture.md) §6. Item 9 is
  struck (`MonotoneCalibration` stays) and item 13 is done (Step 2). Everything else applies here.
- **Per-folder `README.md`s** in `src/stages/`, `src/utils/metrics/`, `src/utils/modeling/`, `src/utils/plotting/`,
  capturing the agreed contracts. Deferred from Step 1 because those folders did not exist yet, and documenting
  contracts before the code invites drift — so write each one *as* its folder is finished.
- **Dead code to delete** (rebuild-plan #1, #8): A's vendored `modeling/mc_dropout/` package, D's `unet_aru.py` /
  `distr_regression_aru.py` / `mc_dropout_module_deprecated.py`, and `transforms.py`.
  **`compute_high_lightning_days` is kept** — it is "the extremes".
