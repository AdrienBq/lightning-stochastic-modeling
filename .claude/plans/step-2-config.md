# Step 2 (block b) — Config ✅ **DONE (2026-07-29)**

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 1](step-1-design.md) · Next: [Step 3](step-3-utils.md)

**Goal.** Write every YAML the pipelines need, containing **only** what the annotated inventories kept. Config
comes before code (Step 3) deliberately: the configs are the contract the code implements, and writing them first
surfaces missing decisions cheaply.

**Verification for this step is parse-only** (no code exists yet): `parse_config` must read every file, and every
metric/loss name referenced must exist in an annotated `keep`/`modify` row.

---

## ▶ OUTCOME — what was actually built, and how it differs from this spec

**18 files**, laid out one directory per concern (this supersedes the flat tree in §1):

```
config/
├── split/               split.yaml · split_smoke_cpu.yaml · split_smoke_gpu.yaml
├── eval/                metrics_daily.yaml · probabilistic_eval.yaml · probabilistic_eval_smoke_cpu.yaml
├── deterministic_unet/  deterministic_unet_daily.yaml · …_smoke_cpu.yaml · …_smoke_gpu.yaml · search_space_daily.yaml
├── mc_dropout/          mc_dropout_daily.yaml          · …_smoke_cpu.yaml · …_smoke_gpu.yaml · search_space_daily.yaml
└── diffusion/           diffusion_daily.yaml           · …_smoke_cpu.yaml · …_smoke_gpu.yaml · search_space_daily.yaml
```

All gates pass: 19/19 files parse; `{{$DATA_ROOT}}` and `{{$UPSTREAM_MODEL}}` interpolate to `''` when unset;
every loss/metric/figure name resolves to an inventory row; every `thresholds:`/`bins:` reference and every
`*_fidelity` band resolves; all three `selection` blocks sum to 1.00 and name emittable keys; no forbidden name
survives in active YAML *or* in any parsed config; no absolute path; every smoke-split id maps to a real,
lightning-active, year-disjoint mid-July day.

### Corrections to this spec, found while executing it

| # | This document said | Reality | Resolution |
|---|---|---|---|
| 1 | §3.2 + §7: `kind: absolute` needs a **new** branch in `evaluation.resolve_threshold` | `absolute` **already exists and is the default kind** (`aru:src/utils/metrics/evaluation.py:58-76`) | Hour bands work with no new code. **Step 3 deferral deleted.** |
| 2 | §5.1: `data-path: ${DATA_ROOT}` | `parse_config` matches `{{\$([^}]+)}}` **only**; `${VAR}` passes through literally and an unset var becomes `''` | Wrote `'{{$DATA_ROOT}}'`; documented the token in `CLAUDE.md` + `README.md`, which had never stated it |
| 3 | §5.3/§5.4: smoke split via `max-days: 2` | No such parameter exists. `prepare_regression` has a *global* `max_samples` = `index.head(N)` **and raises if any split is empty** — on the year split `head(2)` is 2 days of 2008, i.e. test-only → hard error | Dedicated `split_smoke_{cpu,gpu}.yaml` (`by_sample_id`, `cross_check: false`), **mid-July** so the occurrence base rate is non-zero. No new stage parameter |
| 4 | §3.3: `tweedie_deviance` removal "pending conflict #3" | Already resolved in `step-1-design.md` — **remove** is final | Removed; the "pending" marker was stale |
| 5 | §4.1 `calibration:` shows only `occurrence` | §6 item 9 (remove `MonotoneCalibration`) is superseded by Step 1 Q2 ("keep and extend") | Both `calibration.occurrence` **and** `calibration.regression` are exposed; item 9 struck in the inventory |
| 6 | §5.1: `prepared-path` / `checkpoint-path` / `search-space` | The aru stages use `input-path` / `model-path` / `model-config`, and `run.py` **special-cases `model-config` and `metrics-config` to log them as MLflow artifacts** | Kept aru's names — renaming would lose the artifact logging for nothing |
| 7 | §5.3/§5.4 both titled `*_local.yaml` with contradictory values | Two tiers, one name | Renamed to **`*_smoke_cpu.yaml`** and **`*_smoke_gpu.yaml`** ("local" described a machine, not a tier) |

### Decisions taken beyond this spec

1. **`distr_regression` → `deterministic_unet`** (§0 below asked for `deterministic_module`; `_module` is a
   code-layer suffix, wrong for a pipeline/directory/`model-family` token).
2. **Selection score is PR-AUC driven**, one name `valid_regression_score` for all families:
   `0.40 average_precision_occurrence + 0.30 mae_cond_ss_climatology + 0.30 psd_full_fidelity`.
3. **`selection-metric`/`selection-mode` stage parameters dropped.** They duplicated the search space's
   `selection:` block, which was the *only* reason a "MUST match the sweep's" warning existed. `tune` now reads the
   block from `model-config` and records it into `best_trial.json`; `retrain_best` reads it back. The mismatch is
   now unrepresentable rather than merely warned about.
4. **MC-dropout warm-starts from the deterministic U-net** (`UPSTREAM_MODEL` on **`tune`**), running the finetuning
   phase alone; unset ⇒ two-phase fit from scratch. ⚠️ Deliberately *not* the diffusion mechanism: MC-dropout takes
   the upstream's **weights**, diffusion takes its **predictions** via `prepare_regression`.
5. **`deterministic_unet.unet.normalization` fixed to `group`** — a batch-norm upstream cannot be warm-started into
   an MC-dropout model (MC inference re-enables dropout, which would unfreeze batch-norm running statistics), so
   leaving the choice open would let a 40-trial sweep return a best model no downstream family can consume.
6. **`min_hours: 0` alongside `max_hours: 24`** in every search space — the target is bounded on both sides.
   ⚠️ `softplus` only guarantees `>= 0`, so it would silently violate a non-zero `min_hours`.
7. **No `persistence` baseline.** A diagnostic parameterization never sees a past observation.
8. **Five new names added, with inventory rows written for each** (§6 verification rule 2 required them):
   `wmse_psd` (loss), `roc_auc` + `explained_deviance` (scores), `roc_pr_curves` + `confusion_matrix` (figures).

---

## 0. change distr_regression name and 

This naming is confusing. The job of this network is to provide a deterministic U-Net which is then used to compute the residuals that are the targets of the diffusion (flow-matching) model. 
Rename to deterministic_module.
Change everywhere that applies (also the README and other steps plans for example).


## 1. File inventory

> ⚠️ **SUPERSEDED** — the flat tree below was the plan; the built layout is the per-concern one in the OUTCOME
> section above (`config/split/`, `config/eval/`, `config/<family>/`), and the smoke tiers are
> `*_smoke_cpu.yaml` / `*_smoke_gpu.yaml`, not `*_local.yaml`. Kept for the rename record only.

```
config/
├── split.yaml                        # shared: year-based train/valid/test   -> config/split/split.yaml
├── metrics_daily.yaml                      # shared: the one metric suite          -> config/eval/metrics_daily.yaml
├── distr_regression.yaml             # -> config/deterministic_unet/deterministic_unet_daily.yaml
├── distr_regression/
│   └── search_space_daily.yaml             # -> config/deterministic_unet/search_space_daily.yaml
├── mc_dropout_daily.yaml                   # -> config/mc_dropout/mc_dropout_daily.yaml
├── mc_dropout/
│   └── search_space_daily.yaml
├── diffusion_daily.yaml                    # -> config/diffusion/diffusion_daily.yaml
├── diffusion/
│   └── search_space_daily.yaml
├── probabilistic_eval.yaml           # -> config/eval/probabilistic_eval.yaml
└── *_local.yaml                      # -> *_smoke_cpu.yaml, plus a *_smoke_gpu.yaml tier
```

**Renames from the source branches:** aru's `config/distr_regression/split.yaml` → `config/split/split.yaml`
(it is shared, not family-specific); aru's `config/diffusion_model*.yaml` → `diffusion*` for consistency; adrien's
`config/{distr_regression,mc_dropout}/metrics_daily.yaml` and `metrics_old*.yaml` collapse into the single shared
`config/eval/metrics_daily.yaml`. Aru's `*_fast_retrain.yaml` variants are **dropped** — `retrain_best` is a stage in the
main pipeline, not a separate config.

---

## 2. `config/split.yaml`

The year-based split, matching [`00-context.md`](00-context.md). Reuses the existing
`io/data.assign_splits_by_year` reader (`load_split_config` → `assign_splits_from_config`).

```yaml
# Train/validation/test assignment by calendar year. `by_sample_id` is the alternative
# (contiguous sample-index ranges) supported by io/data.assign_splits_by_sample_id.
by_year:
    test:  [2008, 2015, 2023]
    valid: [2009, 2016, 2022]
    train: [2010, 2011, 2012, 2013, 2014, 2017, 2018, 2019, 2020, 2021]
```

---

## 3. `config/metrics_daily.yaml` — the shared suite

Structure carried from aru (`baselines`, `thresholds`, `metrics.{continuous,categorical,skill,calibration,spatial,ensemble}`,
`reporting`), pruned to the annotated inventory. Per the scores open-question #4 answer, this file is **restated as
a clean reference doc** — every entry keeps a one-line rationale.

### 3.1 `baselines` — unchanged

```yaml
baselines:
    zero: {}                          # all-zero predictor: the imbalance-exploiting cheat
    climatology: {window_days: 90}    # per-cell day-of-year mean, ±45 days
```
Drop the persistence baseline : only acceptable for forecasting tasks. Here we are doing reanalysis parameterization. We do not have observations in the past so persistence is not applicable and is therefore not a good baseline.

### 3.2 `thresholds` — ⚠️ **must be redefined**

The old spec used quantiles of the **positive count marginal**, which collapse on a bounded target: on `0–24` the
0.999 quantile of positives lands on a single hour value, so `p99` and `p99_9` stop being distinct events; on
binary, only `occurrence` survives.

**Proposal — absolute hour bands** (explicit, interpretable, and stable across splits):

```yaml
thresholds:
    occurrence: {kind: occurrence}          # target > 0 — THE primary event under the new scope
    h3:  {kind: absolute, value: 3}         # >= 3 hours with lightning that day
    h6:  {kind: absolute, value: 6}
    h12: {kind: absolute, value: 12}
```

This needs a **new `kind: absolute`** branch in `evaluation.resolve_threshold` (currently supports
`occurrence` and `train_positive_quantile`) — a Step 3 task. For the **hourly binary** target only `occurrence`
applies, so that pipeline's `thresholds` list is a single entry.

### 3.3 `metrics.continuous`

Keep: `rmse`, `mae`, `bias`, `rmse_conditional_positive`, `mae_conditional_positive`, `mae_stratified`, `r2`
(aru's `condition=` version), `estimation_tendency`.
Remove: `tweedie_deviance` ⚠️ *pending conflict #3*.
`mae_stratified` and `r2`/`estimation_tendency` `thresholds` lists switch to the new hour bands.

### 3.4 `metrics.categorical` — now the headline group

```yaml
categorical:
    thresholds: [occurrence, h3, h6, h12]
    scores: [pod, far, csi, ets, hss, sedi, frequency_bias]
```

All seven kept. ETS is chance-corrected; SEDI stays non-degenerate as the base rate → 0; `frequency_bias > 1`
flags the preferred conservative over-forecasting.
Add ROC-AUC and PR-AUC

### 3.5 `metrics.skill` — unchanged

`mse_skill_score` (vs zero/climatology/persistence), `mae_conditional_skill_score` (vs zero/climatology,
conditioned on `obs_positive`), `brier_skill_score` (vs climatology, target `occurrence`).
Add the explained deviance

### 3.6 `metrics.calibration` — reduced

| Entry | Fate |
|---|---|
| `reliability_diagram` | **Keep** — promoted to a headline calibration diagnostic for the occurrence head |
| `rank_correlation` | **Keep** (spearman, over the new threshold list) |
| `pit_histogram` | **Remove** — PIT dropped |
| `quantile_ratio` | ⚠️ 🔢 `COUNT-REG` — quantiles collapse on `0–24`. Recommend **remove** |
| `qq_plot` | **Remove** — 🔢 `COUNT-REG`, *and* it was declared here without ever being implemented |

### 3.7 `metrics.spatial` — unchanged in substance

`psd_band_ratio` (bands `full [2,inf]`, `low [32,inf]`, `mid [8,32]`, `high [2,8]` in pixel wavelengths),
`psd_high_fidelity {band: high}`, `psd_full_fidelity {band: full}` — **note the key stays** but is now computed via
`psd_band_ratios(...)['full']` + `psd_fidelity` since `scores.psd_full_fidelity` is removed;
`log_spectral_distance`, `fss` (thresholds × scales `[1,3,5,9,17]`, `report_useful_scale: true`),
`sharpness_ratio`, `variance_ratio`.

Unify the psd_fidelity : one function with band as argument that can take "high", "mid", "low" or "full" as values. psd_band_ratios can take similar arguments as input

### 3.8 `metrics.ensemble` — unchanged

`crps`, `almost_fair_crps`, `spread_skill_ratio`, `rank_histogram` (conditioned on occurrence, M+1 bins).
**`ensemble-size` is a stage parameter, not set here.**

### 3.9 `reporting`

```yaml
reporting:
    figures:
        - maps_most_extreme_days      # reports the most extreme day in test / val styled to the 02a spec (inventory-figures.md §1)
        - psd_curves
        - fss_vs_scale
        - reliability                 # was `reliability_and_pit` -- PIT half removed
        - error_by_intensity_bin      # bins -> hour bands
        - rank_histogram              # ensemble runs only
        # residual-mode diffusion diagnostics; self-skip unless curves['residual'] is populated.
        # Lower priority than the occurrence task, but kept.
        - residual_bias_map
        - residual_surprise
        - residual_histograms
        - residual_qq
        - residual_scatters
        - residual_heteroscedasticity
    formats: [png, csv]
```

Add the two occurrence-task figures that exist nowhere yet — **ROC / PR curves** (backed by
`average_precision`) and a **confusion matrix per threshold**. 

---

## 4. Search spaces

One per family, same node grammar as `search.py` (`{type: categorical|int|float, low, high, log}`; plain values
pass through). Three structural consequences of Step 1:

1. **No `target_transform:` block at all** — the whole section goes, along with `calibration.regression`'s
   monotone option (`MonotoneCalibration` is 🔢 `COUNT-REG`).
2. **`intensity_weight_gamma` low bound → `0.0`.** Required by the `mae`/`rmse` removal: γ=0 makes
   `weighted_mae` ≡ `mae` and `weighted_rmse` ≡ `rmse`. Without this the unweighted variants become unreachable.
3. **Finetuning is a nested block**, not a separate builder — `build_finetune_loss` is removed and the option folds
   into `build_regression_loss` / `build_binary_loss`.

### 4.1 Shared skeleton (all families)

```yaml
loss:
    name: {type: categorical, choices: [weighted_mae, weighted_rmse, weighted_mse, asymmetric_huber, wmae_psd]}
    intensity_weight_gamma: {type: float, low: 0.0, high: 1.0}   # 0.0 => unweighted (replaces mae/rmse)
    asymmetry_tau: {type: float, low: 0.5, high: 0.9}            # asymmetric_huber only
    huber_delta:   {type: float, low: 0.5, high: 3.0}
    alpha:         {type: float, low: 0.5, high: 1.0}            # wmae_psd: pointwise weight; (1-alpha) -> PSD

occurrence_head:                                                 # kept -- "occurrence head is still in scope"
    enabled: {type: categorical, choices: [true, false]}
    loss: {type: categorical, choices: [focal_bce, dice, brier, crps_binary]}
    focal_gamma: {type: float, low: 1.0, high: 3.0}
    positive_class_weight: {type: float, low: 1.0, high: 50.0}

optimizer:
    lr: {type: float, low: 1.0e-4, high: 1.0e-2, log: true}
    weight_decay: {type: float, low: 1.0e-6, high: 1.0e-3, log: true}

unet:
    base_channels: {type: categorical, choices: [16, 32, 64]}
    depth: {type: int, low: 3, high: 5}
    normalization: {type: categorical, choices: [batch, group, none]}
    activation: {type: categorical, choices: [relu, gelu, silu]}
    bottleneck_attention: {type: categorical, choices: [true, false]}

calibration:
    occurrence: {type: categorical, choices: [none, platt]}       # PlattScaling kept (probability calibration)

batch_size: {type: int, low: 4, high: 32}
max_epochs: 50
```

> **Class-weighting overlap to settle:** `intensity_weights` with γ>0 and `focal_bce`'s `positive_class_weight`
> both up-weight positives. On a *binary* target `(1+y)^γ` collapses to `{1, 2^γ}` — i.e. a plain positive-class
> weight — so the two mechanisms duplicate. if focal_bce is selected then intensity_weights must be disabled.

Add the wmae-psd and wmse-psd variants in the possible losses. These are necessary for the mc-dropout pipeline (see the afcrps_psd with loss_weight below).

### 4.2 `mc_dropout/search_space_daily.yaml` — adds

```yaml
dropout_p: {type: float, low: 0.05, high: 0.3}
finetuning:                                    # phase 2 of the two-phase fit
    enabled: {type: categorical, choices: [true, false]}
    loss: {type: categorical, choices: [crps, almost_fair_crps, afcrps_psd]}
    loss_weight: {type: float, low: 0.1, high: 1.0}
    samples: {type: int, low: 8, high: 32}      # MC samples for the CRPS spread term
    max_epochs: 10
output_activation: {type: categorical, choices: [softplus, clamped_sigmoid]}
max_hours: 24                                  # bounded target ceiling
```

### 4.3 `diffusion/search_space_daily.yaml` — adds

```yaml
residual_target: {type: categorical, choices: [true, false]}   # residual mode needs an upstream checkpoint
flow:
    n_steps: {type: int, low: 8, high: 32}                     # ODE integration steps
    hidden_dim: {type: categorical, choices: [128, 256]}
    n_blocks: {type: int, low: 2, high: 6}                     # DiTBlock count
    patch_size: {type: categorical, choices: [1, 2]}
```

### 4.4 `distr_regression/search_space_daily.yaml`

The shared skeleton only — it is the deterministic baseline and the residual upstream, so no ensemble or
diffusion knobs.

---

## 5. Pipeline YAMLs

Template boilerplate (`project_uri`, `log_artifacts`, `log_models`, `lazy`, `ensure_determinism`, `banner`,
`description`, `tags.version`) plus a `stages:` list. Data paths come from **`DATA_ROOT`** (see
[Step 5](step-5-portability.md)), never hardcoded.

### 5.1 Per-family shape (e.g. `config/mc_dropout_daily.yaml`)

```yaml
project_uri: 'src/stages'
log_artifacts: false
log_models: false
lazy: true                     # skip unchanged stages; safe because prepare is expensive
ensure_determinism: false
banner: true
description: 'MC-dropout occurrence model'

stages:
    - setup:
        output-path: outputs/mc_dropout
    - prepare_regression:
        data-path: ${DATA_ROOT}
        split-config: config/split.yaml
        output-path: outputs/mc_dropout/prepared/daily_lightning_hours
        mode: daily                        # daily | hourly
        target-variable: daily_lightning_hours
        hourly-threshold: 2                # >=2 strokes for an hour to count (single-stroke denoising)
    - tune:                                # UNIFIED tuning stage (was tune_mc_dropout)
        model-family: mc_dropout
        prepared-path: outputs/mc_dropout/prepared/daily_lightning_hours
        search-space: config/mc_dropout/search_space_daily.yaml
        output-path: outputs/mc_dropout/tuning
        n-trials: 30
        sampler: tpe
        max-epochs: 50
    - retrain_best:
        model-family: mc_dropout
        source-path: outputs/mc_dropout/tuning
        output-path: outputs/mc_dropout/best
    - evaluate_regression:
        checkpoint-path: outputs/mc_dropout/best
        prepared-path: outputs/mc_dropout/prepared/daily_lightning_hours
        metrics-config: config/metrics_daily.yaml
        metrics-path: outputs/mc_dropout/metrics
        report-path: outputs/mc_dropout/reports
        ensemble-size: 32

tags:
    version: 1.0
```

Note `output-path` / `metrics-path` / `report-path` are the template's `OUTPUT_PARAM_KEYS`, so the lazy cache
treats them as outputs and everything else resolving to a path as an input — no extra wiring needed.

### 5.2 `config/probabilistic_eval.yaml` — cross-model

Runs the **common** evaluation over the three trained families, then compares:

```yaml
stages:
    - evaluate_regression: {checkpoint-path: outputs/distr_regression/best, ...}
    - evaluate_regression: {checkpoint-path: outputs/mc_dropout/best, ...}
    - evaluate_regression: {checkpoint-path: outputs/diffusion/best, ...}
    - tabulate_metrics:
        distr_regression: outputs/distr_regression/metrics
        mc_dropout: outputs/mc_dropout/metrics
        diffusion: outputs/diffusion/metrics
        output-path: outputs/comparison
    - combine_curves:
        <same three report dirs>
        output-path: outputs/comparison/curves
```

This is the **proof the merge worked**: `tabulate_metrics` must emit one CSV whose metric-key columns are
*identical* across families.

### 5.3 / 5.4 The two smoke tiers — `*_smoke_cpu.yaml` and `*_smoke_gpu.yaml`

*(Both sections were titled `*_local.yaml` with contradictory values; renamed so each tier has its own name.)*
Each tier is its parent by copy, overriding only what makes the run small:

| Parameter | `*_smoke_cpu` | `*_smoke_gpu` | Why |
|---|---|---|---|
| `split-config` | `config/split/split_smoke_cpu.yaml` (8 days: 4/2/2) | `config/split/split_smoke_gpu.yaml` (18 days: 10/4/4) | `by_sample_id`, mid-July — **not** a `max-days` parameter, which does not exist (see correction 3) |
| `output-path` prefixes | `outputs/<family>_smoke_cpu/…` | `outputs/<family>_smoke_gpu/…` | a smoke run must never clobber a real one |
| `n-trials` | `1` | `2` | GPU tier runs two so the sampler / trial-comparison path is exercised |
| `sampler` | `random` | `tpe` | TPE is pointless at one trial |
| `max-epochs` | `1` | `1` | |
| `pruning` | `false` | `false` | a 1-epoch trial has nothing to prune |
| `accelerator` / `num-workers` | `cpu` / `0` | `gpu` / `8` | in-process loading avoids worker start-up dominating an 8-day run |
| `precision` | (omitted) | `bf16-mixed` | the CPU tier cannot exercise mixed precision |
| `feature-stats-days` | `4` | `10` | cannot exceed the train days available |
| `ensemble-size` | **`2`** | **`5`** | ⚠️ **never `1`** — `spread_skill_sums` uses `ddof=1`, so `M=1` silently yields `NaN` |
| `sampling-steps` (diffusion) | `8` | `16` | ODE integration dominates a CPU diffusion eval |
| `lazy` | `false` | `false` | never cache a smoke run |

Both tiers point at the **full** search spaces: `n-trials: 1` + `max-epochs: 1` is what makes them tiny, and three
more search-space files would only drift out of sync.

⚠️ With 8 days the rarest threshold (`h12`) may see zero events, so its categorical scores can come back `NaN`.
That is expected at this size and is not a failed run.

## 6. Step 2 verification — ✅ all gates pass

1. ✅ `parse_config` on every YAML under `config/` — 19/19 (18 + the template's `hello_world`).
2. ✅ Every loss name in every `search_space_daily.yaml`, every metric key in `metrics_daily.yaml`, and every
   `reporting.figures` entry resolves to a `keep`/`modify` row in
   [`inventory-losses.md`](inventory-losses.md) / [`inventory-scores.md`](inventory-scores.md) /
   [`inventory-figures.md`](inventory-figures.md) — the five new names got rows written for them.
3. ✅ No forbidden name (`target_transform`, `gamma_shape`, `gamma_scale`, `pit_histogram`, `tweedie`, `poisson`,
   `qq_plot`, `quantile_ratio`, `train_positive_quantile`, `persistence`, `log_warp`, `occurrence_threshold`,
   `distr_regression`) survives in **active YAML** or in any **parsed** config. ⚠️ A naive whole-file grep *does*
   hit several of these — every hit is comment prose *explaining the absence*, so the gate must strip comments or
   inspect the parsed structure. Removed "subject to conflict resolution" from `tweedie`/`poisson`: Step 1 settled
   both as remove.
4. ✅ `DATA_ROOT` is referenced as `{{$DATA_ROOT}}` and interpolates to `''` when unset; no absolute path anywhere.
5. ✅ Every `thresholds:` / `bins:` reference resolves to a declared threshold, and every `*_fidelity` key names a
   declared PSD band.
6. ✅ All three `selection` blocks name the same metric, sum to 1.00, and reference only emittable keys.
7. ✅ Every smoke-split sample id maps to a real, lightning-active, year-disjoint mid-July day in `metadata.csv`.
8. ✅ `upstream-model-path` is on `tune` for mc_dropout and on `prepare_regression` for diffusion, nowhere else;
   `selection-metric`/`selection-mode` are gone; `min_hours`+`max_hours` in all three search spaces.

## 7. Deliberately deferred to Step 3

- ~~`evaluation.resolve_threshold` needs its new `kind: absolute` branch~~ — **not needed**, it already supports it
  (correction 1).
- `psd_full_fidelity` as a *metric key* backed by `psd_band_ratios` + `psd_fidelity` rather than its own function.
- `fss_useful_scale` reusing precomputed per-scale FSS instead of recomputing (annotated `modify`).
- Implementations for the five new names: `wmse_psd`, `roc_auc`, `explained_deviance`, `roc_pr_curves`,
  `confusion_matrix`.
- **`MCDropoutRegressionModule` init-from-checkpoint** plus a load-time architecture-compatibility check that
  *raises*, naming the offending field, rather than silently partial-loading. The module has
  `set_phase('train'|'finetune')` and a `finetuning_enabled` gate but nothing that loads foreign weights, so this is
  new code, not wiring.
- `search.apply_constraints`: force `finetuning.enabled = true` and ignore the `unet:` block when `tune` gets an
  `upstream-model-path`; force `intensity_weight_gamma = 0` when `occurrence_head.loss == focal_bce`.
- `build_output_activation` reading the `min_hours`/`max_hours` pair; decide whether `softplus` with a non-zero
  `min_hours` is constrained out or clamped after the activation.
- `tune` records the resolved `selection` block into `best_trial.json`; `retrain_best` reads it back and drops
  `model-config` entirely.
