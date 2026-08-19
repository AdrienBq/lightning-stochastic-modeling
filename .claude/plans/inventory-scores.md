# Inventory: scores & metrics

**Purpose.** Complete factual inventory of every score defined on the two source branches, grouped by category
and by cross-branch similarity. **The `Decision` column is for you to fill in** (`keep` / `change` / `remove`,
plus notes). Once annotated, this file is the single source of truth for the unified `src/utils/metrics/scores.py`
and for `config/metrics_daily.yaml` in Step 2.

Nothing here is a recommendation — it is what exists, with the differences made explicit.

## Sources

| Key | Branch : path | Count | Character |
|---|---|---|---|
| **A** | `aru-probabilistic-eval` : `src/utils/metrics/scores.py` | 40 | Evolved; adds **streaming ensemble** machinery + `condition` support |
| **V** | `aru-probabilistic-eval` : `src/utils/modeling/mc_dropout/scores.py` | 30 | **Stale vendored copy** of D |
| **D** | `adrien-mc-dropout` : `src/utils/metrics/scores.py` | 34 | Older base + two unique additions |

V is again a stale snapshot: identical function order and line offsets to D, missing D's `psd_full_fidelity`.
It contributes nothing unique and is listed only so its removal is traceable.

## Status legend

| Status | Meaning |
|---|---|
| `IDENTICAL` | Same name, same signature in A and D |
| `A-SUPERSET` | Same name in both, but **A adds parameters** — A is backwards-compatible |
| ⚠️ `CONTRACT-DIFFERS` | Same name in both but **different return type or semantics** — silent-bug risk on merge |
| `A-ONLY` / `D-ONLY` | Exists on one branch only |

## ⚠️ Scope decision (2026-07-28): classification-first, **no target transform**

First target is **occurrence** — hourly binary, or the **bounded daily count of hours with lightning, `0–24`**.
The gamma F-transform is **dropped**. The daily `0–24` target is still a regression, so ordinary error metrics
stay; what goes is machinery for **unbounded heavy-tailed counts** and for the transform.

| Flag | Meaning |
|---|---|
| 🔢 `COUNT-REG` | Exclusive to the **unbounded count** task — obsolete under the new scope |
| 🔀 `TRANSFORM` | Serves the gamma F-transform only — dies with it |
| ✅ `CLASSIF` | Directly serves the occurrence task — **the new priority** |
| 🎯 `TAIL-THRESH` | Still valid, but its `p90/p99/p99_9` **positive-count quantile** thresholds need redefining for a `0–24` (or binary) target |

---

## 1. Event / threshold helpers — ✅ **now the core of the suite**

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `exceedance` | 29 | 29 | `IDENTICAL` | ✅ `CLASSIF` Boolean mask `values > t` (or `>=`, via `strict`) | keep |
| `contingency_counts` | 34 | 34 | `IDENTICAL` | ✅ `CLASSIF` `(hits, misses, false_alarms, correct_negatives)` at a threshold | keep |
| `categorical_scores` | 50 | 50 | `IDENTICAL` | ✅ `CLASSIF` From a contingency table → `pod, far, csi, ets, hss, sedi, frequency_bias`. ETS is chance-corrected; SEDI stays non-degenerate as base rate → 0 | keep |

> With occurrence as the primary task, this group **is** the headline metric set — the `occurrence` threshold
> (`target > 0`) becomes the main evaluation event rather than one of four.

## 2. Continuous errors

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `rmse` | 92 | 92 | `IDENTICAL` | Root mean squared error | keep |
| `mae` | 96 | 96 | `IDENTICAL` | Mean absolute error | keep |
| `bias` | 100 | 100 | `IDENTICAL` | `mean(pred - obs)`; `>= 0` preferred (conservativeness) | keep |
| `conditional_error` | 190 | 105 | `A-SUPERSET` | Error restricted to a cell subset. **A adds `condition=` array**; D hardcodes its own subset. A's version is what `validation.py` calls for the selection score | keep |
| `stratified_mae` | 208 | 113 | `IDENTICAL` | 🎯 `TAIL-THRESH` MAE within observed-intensity bins (tail profile). Bins are positive-count quantiles — **redefine for `0–24`** (e.g. explicit hour bands); degenerate for binary | keep |
| `tweedie_deviance_score` | 231 | 136 | `IDENTICAL` | 🔢 `COUNT-REG` Tweedie deviance as a metric — same unbounded-continuous assumption as the loss | remove |
| `r2_score` | 105 | 155 | `A-SUPERSET` | 🎯 `TAIL-THRESH` Coefficient of determination. **A adds `condition=`** for per-subgroup R². Valid for the `0–24` regression; **meaningless for binary** (use Brier / AP instead) | keep A |
| `estimation_tendency` | 127 | 171 | `IDENTICAL` | 🎯 `TAIL-THRESH` Under-/over-estimated cell fractions with a `tolerance` dead-band. Valid for `0–24`; for binary it collapses into the contingency table | keep |
| `skill_score` | 243 | 148 | `IDENTICAL` | `1 - model_error / baseline_error` | keep |
| `explained_deviance` | — | — | **NEW (Step 2)** | ✅ `CLASSIF` **Bernoulli** explained deviance on the occurrence head vs the climatology baseline: `1 - logloss(model) / logloss(climatology)`. Deliberately scoped to the *binary* head — with `tweedie_deviance_score` removed, the bounded `0–24` target has no likelihood left to take a deviance of, so a "regression explained deviance" would be undefined here. On the occurrence probability it is well-defined, proper, and the standard likelihood-based complement to `brier_skill_score` (which is the *quadratic* rather than *logarithmic* score) | keep — **new code in Step 3**; lives in `metrics.skill`, emits the flat key `explained_deviance` |

## 3. Rank / ordering agreement

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `rank_correlation` | 156 | 187 | `IDENTICAL` | Spearman/Kendall within observed subgroups (all-cell coefficient is degenerate under the zero mass) | keep |

## 4. Spatial structure / spectral fidelity — challenge (B)

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `fss` | 253 | 223 | `A-SUPERSET` | Fractions skill score at threshold × neighbourhood scale. A takes extra params | keep |
| `fss_useful_scale` | 291 | 256 | `IDENTICAL` | Smallest scale where `FSS > 0.5 + base_rate/2` | modify : do not recompute the fss at all scales, but used the precomputed scores from the passes of fss |
| `mean_power_spectrum` | 313 | 277 | `A-SUPERSET` | Mean 2-D power spectrum. **A adds a `progress` callback** (for long streaming passes) | keep |
| `_wavelength_grid` | 323 | 285 | `IDENTICAL` | Private: pixel-wavelength grid for radial binning | keep — the shared basis of every PSD score, so the band edges in `metrics_daily.yaml` mean the same thing everywhere |
| `radial_psd` | 332 | 294 | `A-SUPERSET` | Radially-averaged PSD; A's signature is richer | keep |
| `radial_psd_per_map` | 364 | — | `A-ONLY` | Per-map PSD instead of pre-averaged — required for **pooled ensemble** structure scoring (avoids averaging maps before spectra) | keep |
| `psd_band_ratios` | 410 | 315 | `IDENTICAL` | Pred/obs PSD ratio per named wavelength band (`full`/`low`/`mid`/`high`, pixels) | keep |
| `psd_fidelity` | 445 | 342 | `IDENTICAL` | `clip(1 - abs(1 - ratio), 0, 1)` — scalar summary of one band ratio | modify to include the psd_full_fidelity |
| `psd_full_fidelity` | — | 349 | `D-ONLY` | Convenience wrapper: full-band fidelity direct from `(pred, obs)`. A achieves the same via `psd_band_ratios(...)['full']` + `psd_fidelity` — so **functionally covered**, but the named entry point differs | remove |
| `log_spectral_distance` | 452 | 367 | `IDENTICAL` | Log-domain spectral distance | keep |
| `sharpness_ratio` | 470 | 378 | `IDENTICAL` | Std of spatial gradient magnitude, pred/obs — cheap blur detector | keep |
| `variance_ratio` | 480 | 388 | `IDENTICAL` | Spatial variance ratio, pred/obs | keep |

## 5. Distributional calibration

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `quantile_ratios` | 489 | 397 | `IDENTICAL` | 🔢 `COUNT-REG` `pred_q / obs_q` at tail quantiles (0.99, 0.999) to expose under-dispersion. On a `0–24` target those quantiles sit at a handful of integers — **near-useless**; on binary, meaningless | modify to fit 0-24 regression|
| `uniform_histogram_ks` | 507 | 412 | `IDENTICAL` | 🔀 `TRANSFORM` KS distance to Uniform(0,1) = the PIT flatness test (`pit_ks`). The function itself is **generic** — it just tests uniformity. The transform dependency is in its **caller**: `evaluation.py:468-470` reads `target_stats['gamma_shape']`/`['gamma_scale']` and calls `gammainc` to build the PIT values. Deleting `compute_target_transform_stats` (removal item 7) removes those parameters, so the call sites break for lack of a fitted `F` | remove |
| `reliability_curve` | 524 | 491 | `IDENTICAL` | ✅ `CLASSIF` Binned reliability (calibration) curve for a probability head — **now a primary calibration diagnostic** | keep |
| `quantile_quantile` | 548 | — | `A-ONLY` | 🔢 `COUNT-REG` Marginal pred-vs-obs QQ at occurrence cells; backs `qq_plot`. Same coarseness issue on a `0–24` target | modify |

## 6. Binary / probabilistic (single-valued) — ✅ **now headline metrics**

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `brier_score` | 572 | 515 | `IDENTICAL` | ✅ `CLASSIF` Brier score on occurrence probabilities — a **proper** score for the new primary task | keep |
| `average_precision` | 576 | 519 | `IDENTICAL` | ✅ `CLASSIF` Area under precision-recall, `max_samples` subsampling cap. The right summary under extreme imbalance (PR beats ROC when positives are rare) | keep — **and it is now the discrimination term of the tuning selection score**, as `average_precision_occurrence` (Step 2) |
| `dice_coefficient` | — | 533 | `D-ONLY` | ✅ `CLASSIF` Dice/F1 overlap on binarised fields. **Not in A at all** — and its value rises sharply under the new scope | keep |
| `roc_auc` | — | — | **NEW (Step 2)** | ✅ `CLASSIF` Area under the ROC curve, per event threshold. Threshold-free like `average_precision`, but reported *alongside* it rather than instead: ROC-AUC is the familiar, cross-study-comparable number, while being **optimistic when negatives dominate** (at the hourly 0.43 % base rate a useless model still scores well on it). Having both makes that gap visible — a high `roc_auc` with a low `average_precision` is the signature of imbalance-exploitation. Emitted through the categorical group's `<score>_<threshold>` grammar, and backs the new `roc_pr_curves` figure | keep — **new code in Step 3** |

## 7. Ensemble scores — ⚠️ the merge hazard

| Score | A | D | Status | What it computes | Decision |
|---|---|---|---|---|---|
| `_crps_terms` | 602 | — | `A-ONLY` | Private: order-statistic CRPS terms, factored for streaming reuse | keep |
| `crps_sums` | 626 | — | `A-ONLY` | **Partial sums** of CRPS for incremental accumulation | keep |
| `crps_ensemble` | 647 | 429 | ⚠️ `CONTRACT-DIFFERS` | **A returns `float`** (aggregated, accepts `condition=`). **D returns `np.ndarray`** (per-element). Same name, incompatible contracts | keep A |
| `almost_fair_crps_ensemble` | 653 | 451 | ⚠️ `CONTRACT-DIFFERS` | Same divergence: A → `float` + `condition=`, D → `np.ndarray` | keep A |
| `spread_skill_sums` | 659 | — | `A-ONLY` | Partial sums for spread-skill streaming | keep |
| `spread_skill_ratio` | 680 | — | `A-ONLY` | `sqrt(mean ensemble variance) / RMSE(ensemble mean)`; ~1 calibrated, `<1` over-confident | keep — ⚠️ its partials (`spread_skill_sums`) use `ddof=1`, so **`ensemble-size` must be ≥ 2**: `M = 1` divides by zero and yields a silent `NaN` rather than raising |
| `rank_histogram_counts` | 690 | 465 | `A-SUPERSET` | Talagrand rank counts (randomised tie-breaking); A's signature is richer | keep |
| `rank_histogram_reliability` | 725 | — | `A-ONLY` | Scalar flatness of the rank histogram (0 = calibrated) | keep |
| `ensemble_partials` | 739 | — | `A-ONLY` | **The streaming ensemble contract.** Emits per-batch partials that `evaluation.merge_ensemble_partials` accumulates and `finalize_ensemble_metrics` reduces. This is the mechanism that lets MC-dropout and diffusion report *identical* ensemble metrics | keep |

> ⚠️ **`crps_ensemble` / `almost_fair_crps_ensemble` are the one genuine trap in this inventory.** Identical names,
> different return types (`float` vs `np.ndarray`). Merging by name without checking would produce silently wrong
> numbers rather than an error. A's contract is the one the streaming evaluation depends on.

## 7b. Scope-change notes for the spatial and ensemble groups

Neither group is flagged — both are **task-agnostic** — but two points matter:

- **Spatial / spectral (§4)** operates on any `[N, H, W]` field, so it works on binary occurrence maps and on
  `0–24` hour maps alike. On a *binary* field the PSD measures the spectrum of a 0/1 indicator, which is still a
  meaningful over-smoothing detector (a blurred probability field has too little high-frequency power). Challenge
  (B) therefore survives the scope change intact.
- **Ensemble (§7)** is likewise target-agnostic. Note `spread_skill_sums` uses `ddof=1`, so **`M ≥ 2` is
  required** — an `ensemble-size: 1` smoke config yields silent `NaN`.

## Threshold redefinition needed (🎯 `TAIL-THRESH` rows)

`config/metrics_daily.yaml` currently defines thresholds as quantiles of the **positive train-count marginal**:

```yaml
thresholds:
    occurrence: {kind: occurrence}                      # target > 0   <- ✅ becomes the primary event
    p90:   {kind: train_positive_quantile, value: 0.90}
    p99:   {kind: train_positive_quantile, value: 0.99}
    p99_9: {kind: train_positive_quantile, value: 0.999}
```

Under the new targets this needs rethinking: on a `0–24` integer target the 0.999 quantile of positives collapses
onto a single hour value, so `p99`/`p99_9` stop being distinct events; on a binary target only `occurrence`
survives at all. Options — explicit hour bands (e.g. `≥1, ≥3, ≥6, ≥12` hours/day), fewer thresholds, or
`occurrence` alone. This is a Step 2 decision.

## 8. Retired / superseded

| Item | Where | Note | Decision |
|---|---|---|---|
| `EnsembleProbabilisticAccumulator` | D : `metrics/evaluation.py:502` | D's **parallel** ensemble accumulation — the alternative to A's `ensemble_partials` + `merge_ensemble_partials` + `finalize_ensemble_metrics`. Plan merge task #3 retires this | remove |
| `regression_metric_suite` | D : `metrics/evaluation.py:354` | D's second suite alongside `run_metric_suite`. A has **one** suite only | remove |
| whole `mc_dropout/` package | A : `src/utils/modeling/mc_dropout/` | 5 files (`scores`, `losses`, `evaluation`, `unet`, `mc_dropout_module`), all stale snapshots | remove |

---

## Summary

| | Count |
|---|---|
| `IDENTICAL` | 22 |
| `A-SUPERSET` (A backwards-compatible) | 6 |
| ⚠️ `CONTRACT-DIFFERS` | 2 |
| `A-ONLY` | 8 |
| `D-ONLY` | 2 (`psd_full_fidelity` — functionally covered; `dice_coefficient` — genuinely unique) |

**The structural fact — opposite to the losses file:** on *scores*, `aru-probabilistic-eval` is essentially a
**superset** of `adrien-mc-dropout`, adding `condition=` support and the entire streaming-ensemble layer. Only
`dice_coefficient` would be genuinely lost by taking A wholesale.

So the merge directions differ per file: **scores → take A**, **losses → take D**. That asymmetry is worth
recording explicitly, because doing it uniformly in either direction loses real work.

## Open questions for you

1. **`dice_coefficient`** — the only genuinely unique score on D. Keep it? It pairs with `dice_loss`, so its fate
   probably follows the occurrence-classifier decision.
2. **`psd_full_fidelity`** — keep D's named convenience wrapper, or standardise on A's
   `psd_band_ratios(...)['full']` + `psd_fidelity` composition? (`config/metrics_daily.yaml` already declares
   `psd_full_fidelity` as a metric key either way.)
3. **The two `CONTRACT-DIFFERS` functions** — confirm A's `float`-returning contract wins, so I can assert it in
   the design doc and add a test pinning the return type.
4. **`config/metrics_daily.yaml` is already an extensive design doc** on A (baselines, thresholds, and 6 metric groups
   with rationale). Should the metrics design doc *reference* it as the source of truth rather than restating it?

1. Keep
2. standardise on A
3. Keep A
4. Restate it as a clean reference doc