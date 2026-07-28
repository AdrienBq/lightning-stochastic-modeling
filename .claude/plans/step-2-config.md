# Step 2 (block b) — Config

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 1](step-1-design.md) · Next: [Step 3](step-3-utils.md)

**Goal.** Write every YAML the pipelines need, containing **only** what the annotated inventories kept. Config
comes before code (Step 3) deliberately: the configs are the contract the code implements, and writing them first
surfaces missing decisions cheaply.

**Verification for this step is parse-only** (no code exists yet): `parse_config` must read every file, and every
metric/loss name referenced must exist in an annotated `keep`/`modify` row.

---

## 0. change distr_regression name and 

This naming is confusing. The job of this network is to provide a deterministic U-Net which is then used to compute the residuals that are the targets of the diffusion (flow-matching) model. 
Rename to deterministic_module.
Change everywhere that applies (also the README and other steps plans for example).


## 1. File inventory

```
config/
├── split.yaml                        # shared: year-based train/valid/test
├── metrics.yaml                      # shared: the one metric suite, all families
├── distr_regression.yaml             # pipeline: U-net baseline (also the residual upstream)
├── distr_regression/
│   └── search_space.yaml
├── mc_dropout.yaml                   # pipeline: MC-dropout
├── mc_dropout/
│   └── search_space.yaml
├── diffusion.yaml                    # pipeline: flow matching (incl. residual mode)
├── diffusion/
│   └── search_space.yaml
├── probabilistic_eval.yaml           # cross-model: evaluate + tabulate + combine
└── *_local.yaml                      # CPU smoke variant of each of the four above
```

**Renames from the source branches:** aru's `config/distr_regression/split.yaml` → top-level `config/split.yaml`
(it is shared, not family-specific); aru's `config/diffusion_model*.yaml` → `diffusion*` for consistency; adrien's
`config/{distr_regression,mc_dropout}/metrics.yaml` and `metrics_old*.yaml` collapse into the single shared
`config/metrics.yaml`. Aru's `*_fast_retrain.yaml` variants are **dropped** — `retrain_best_*` is a stage in the
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

## 3. `config/metrics.yaml` — the shared suite

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

### 4.2 `mc_dropout/search_space.yaml` — adds

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

### 4.3 `diffusion/search_space.yaml` — adds

```yaml
residual_target: {type: categorical, choices: [true, false]}   # residual mode needs an upstream checkpoint
flow:
    n_steps: {type: int, low: 8, high: 32}                     # ODE integration steps
    hidden_dim: {type: categorical, choices: [128, 256]}
    n_blocks: {type: int, low: 2, high: 6}                     # DiTBlock count
    patch_size: {type: categorical, choices: [1, 2]}
```

### 4.4 `distr_regression/search_space.yaml`

The shared skeleton only — it is the deterministic baseline and the residual upstream, so no ensemble or
diffusion knobs.

---

## 5. Pipeline YAMLs

Template boilerplate (`project_uri`, `log_artifacts`, `log_models`, `lazy`, `ensure_determinism`, `banner`,
`description`, `tags.version`) plus a `stages:` list. Data paths come from **`DATA_ROOT`** (see
[Step 5](step-5-portability.md)), never hardcoded.

### 5.1 Per-family shape (e.g. `config/mc_dropout.yaml`)

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
        search-space: config/mc_dropout/search_space.yaml
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
        metrics-config: config/metrics.yaml
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

### 5.3 `*_local.yaml` — CPU smoke variants

Override only what makes the run tiny; inherit everything else by copy:

| Parameter | Smoke value | Why |
|---|---|---|
| `n-trials` | `1` | one trial |
| `max-epochs` | `1` | one epoch |
| split | 2-day slice | via a `max-days: 2` parameter on `prepare_regression`, or a dedicated `split_local.yaml` |
| `ensemble-size` | **`2`** | ⚠️ **not `1`** — `spread_skill_sums` uses `ddof=1`, so `M=1` silently yields `NaN` |
| `lazy` | `false` | never cache a smoke run |
| `n-trials` sampler | `random` | TPE is pointless at one trial |

---

### 5.4 `*_local.yaml` — GPU smoke variants

Override only what makes the run tiny; inherit everything else by copy:

| Parameter | Smoke value | Why |
|---|---|---|
| `n-trials` | `2` | one trial |
| `max-epochs` | `1` | one epoch |
| split | 10-day slice | via a `max-days: 10` parameter on `prepare_regression`, or a dedicated `split_local.yaml` |
| `ensemble-size` | **`5`** | ⚠️ **not `1`** — `spread_skill_sums` uses `ddof=1`, so `M=1` silently yields `NaN` |
| `lazy` | `false` | never cache a smoke run |
| `n-trials` sampler | `TPE` | TPE is pointless at one trial |

## 6. Step 2 verification

1. `parse_config` on **every** YAML under `config/` — exits 0.
2. Cross-check: every metric name in `metrics.yaml` and every loss name in every `search_space.yaml` appears as a
   `keep`/`modify` row in [`inventory-scores.md`](inventory-scores.md) / [`inventory-losses.md`](inventory-losses.md).
3. Grep for regressions: **no** occurrence of `target_transform`, `gamma_shape`, `gamma_scale`, `pit_histogram`,
   `tweedie`, `poisson` (subject to conflict resolution), `qq_plot`, or `train_positive_quantile` anywhere in
   `config/`.
4. `DATA_ROOT` is referenced, and no absolute machine-specific path is hardcoded, in any config.
5. Assert every `thresholds:` reference resolves to a name defined in `metrics.yaml`'s `thresholds` block.

## 7. Deliberately deferred to Step 3

- `evaluation.resolve_threshold` needs its new `kind: absolute` branch before the hour-band thresholds work.
- `psd_full_fidelity` as a *metric key* backed by `psd_band_ratios` + `psd_fidelity` rather than its own function.
- `fss_useful_scale` reusing precomputed per-scale FSS instead of recomputing (annotated `modify`).
