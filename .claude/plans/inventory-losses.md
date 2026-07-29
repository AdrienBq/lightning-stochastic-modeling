# Inventory: loss functions

**Purpose.** Complete factual inventory of every loss defined on the two source branches, grouped by category
and by cross-branch similarity. **The `Decision` column is for you to fill in** (`keep` / `change` / `remove`,
plus notes). Once annotated, this file is the single source of truth for the unified `src/utils/modeling/losses.py`
built in Step 3.

Nothing here is a recommendation — it is what exists, with the differences made explicit.

## Sources

| Key | Branch : path | Count | Character |
|---|---|---|---|
| **A** | `aru-probabilistic-eval` : `src/utils/modeling/losses.py` | 8 | Lean, **regression-only** set |
| **V** | `aru-probabilistic-eval` : `src/utils/modeling/mc_dropout/losses.py` | 19 | **Stale vendored copy** of D (see below) |
| **D** | `adrien-mc-dropout` : `src/utils/modeling/losses.py` | 23 | **Superset**: regression + binary + ensemble + spectral |

### V is a stale snapshot of D — confirmed, not assumed

V and D share identical function order and near-identical line offsets up to D's line 105, where D introduces
`psd_penalty`. V contains **none** of D's three spectral losses (`psd_penalty`, `wmae_psd`, `afcrps_psd`). So V is
an earlier copy of D that was vendored into aru and then left behind while D moved on. This is the evidence for
the rebuild plan's merge task #1 ("drop aru's stale vendored `modeling/mc_dropout/` package").

**Consequence:** there are only **two live sources** — A and D. V contributes nothing unique and is listed below
only so its removal is traceable.

## Status legend

| Status | Meaning |
|---|---|
| `IDENTICAL` | Same function in A and D, same signature |
| `D-ONLY` | Exists only in `adrien-mc-dropout` — would be **lost** if D is not merged |
| `A-ONLY` | Exists only in `aru-probabilistic-eval` |
| `STALE-DUP` | Present in V only as an old copy; no unique content |

## ⚠️ Scope decision (2026-07-28): classification-first, **no target transform**

The first target is the **occurrence** task — either hourly binary occurrence, or the **daily count of hours with
lightning, bounded `0–24`** (`daily_lightning_hours`). The gamma F-transform is **dropped**: it exists to condition
an unbounded heavy-tailed count target and only complicates matters here.

**Important distinction used for the flags below.** The daily `0–24` target is still a *regression*, just a
**bounded** one — so ordinary distance losses (MSE / MAE / Huber) remain fully in scope. What becomes obsolete is
machinery specific to **unbounded, heavy-tailed stroke counts** and to the transform.

| Flag | Meaning |
|---|---|
| 🔢 `COUNT-REG` | Exclusive to the **unbounded count** regression task — obsolete under the new scope |
| 🔀 `TRANSFORM` | Exists only to serve the gamma F-transform — **dies with it** |
| ✅ `CLASSIF` | Directly serves the occurrence task — **the new priority** |
| *(untagged)* | Task-agnostic, or valid for the bounded `0–24` regression |

---

## 1. Weighting helpers

| Loss | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `intensity_weights` | 26 | 27 | 27 | `IDENTICAL` | Per-cell weight `∝ y_raw^gamma`; up-weights intense cells against the zero mass | keep — **and `gamma: 0.0` must stay reachable in every search space**: `gamma = 0` is what makes `weighted_mae ≡ mae` and `weighted_rmse ≡ rmse`, i.e. what licenses removing the unweighted variants below (Step 2) |
| `_weighted_masked_mean` | 31 | 32 | 32 | `IDENTICAL` | Private reduction: weighted mean over valid (masked) cells; shared denominator for all pointwise losses | keep — every pointwise loss must reduce through it; it normalises by the **sum of effective weights**, not the cell count, so inlining it would put one loss on a different scale from its siblings and make tuning results incomparable |

## 2. Pointwise regression losses

| Loss | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `weighted_mse` | 36 | 37 | 37 | `IDENTICAL` | Intensity-weighted squared error | keep |
| `mae` | — | 41 | 41 | `D-ONLY` | Masked mean absolute error | remove : we can take w=1 with the weighted_mae. Need to extend the search space for gamma to reach 0 |
| `weighted_mae` | — | 45 | 45 | `D-ONLY` | Intensity-weighted MAE | keep |
| `rmse` | — | 49 | 49 | `D-ONLY` | Root of masked MSE | remove : same reason as mae |
| `weighted_rmse` | — | 54 | 54 | `D-ONLY` | Root of intensity-weighted MSE | keep — it *absorbs* the removed `rmse` at `gamma = 0`, so removing `rmse` above is only valid if this one stays (Step 2) |
| `asymmetric_huber` | 40 | 58 | 58 | `IDENTICAL` | Huber with different slopes for over- vs under-prediction — encodes the *conservativeness* preference (penalise misses harder than false alarms) | keep |

> **Note:** A carries only `weighted_mse` + `asymmetric_huber`. The four plain/weighted MAE/RMSE variants exist
> only on D. If the unified file is built from A alone, those four disappear.

## 3. Distributional / count losses — 🔢 **both `COUNT-REG`: the whole group is out of scope**

| Loss | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `tweedie_deviance` | 61 | 79 | 79 | `IDENTICAL` | 🔢 `COUNT-REG` Tweedie deviance, `power` configurable — proper score for zero-inflated non-negative **continuous/unbounded** targets (eval config uses `power: 1.9`). **Wrong family for a bounded `0–24` target** | remove |
| `poisson_nll` | 81 | 99 | 99 | `IDENTICAL` | 🔢 `COUNT-REG` Poisson NLL — Poisson is **unbounded**, so it puts mass on >24 hours/day. For `0–24`-of-24 the principled likelihood is **binomial**, not Poisson | remove |

## 4. Spectral / structure losses — **all `D-ONLY`, all newest work**

| Loss | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `psd_penalty` | — | — | 105 | `D-ONLY` | Penalises mismatch between predicted and observed radially-averaged power spectra — a **differentiable anti-over-smoothing term**. Directly attacks challenge (B) in `metrics.yaml` | keep |
| `wmae_psd` | — | — | 163 | `D-ONLY` | Composite: `alpha * weighted_mae + (1 - alpha) * psd_penalty` | keep |
| `wmse_psd` | — | — | — | **NEW (Step 2)** | Composite: `alpha * weighted_mse + (1 - alpha) * psd_penalty` — the exact `weighted_mse` sibling of `wmae_psd`. Added so the *squared-error* branch of the pointwise family can also carry the anti-over-smoothing term, instead of forcing a choice between MSE-shaped gradients and spectral fidelity | keep — **new code in Step 3**, a few lines reusing `psd_penalty` and `weighted_mse` |
| `afcrps_psd` | — | — | 270 | `D-ONLY` | Composite: almost-fair CRPS + `psd_penalty`, `beta=0.7` | keep |

> These three are the **highest-risk items in this inventory**. They are absent from A *and* from A's vendored
> copy V, meaning they postdate the vendoring. They are also the only losses that optimise the spectral fidelity
> the shared metric suite measures — a training/eval mismatch if dropped.

## 5. Binary / classification losses — ✅ **the new priority group**

| Loss | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `focal_bce_with_logits` | 87 | 105 | 179 | `IDENTICAL` | ✅ `CLASSIF` Focal BCE — down-weights easy negatives, for the extreme class imbalance | keep |
| `dice_loss` | — | 119 | 193 | `D-ONLY` | ✅ `CLASSIF` Soft Dice on the occurrence mask, `smooth=1.0` | keep |
| `brier_loss` | — | 132 | 206 | `D-ONLY` | ✅ `CLASSIF` Brier score as a differentiable loss on probabilities | keep |
| `crps_binary` | — | 196 | 288 | `D-ONLY` | ✅ `CLASSIF` CRPS specialised to a binary target | keep |

> Under the new scope this group moves from "the hierarchy's auxiliary head" to **the primary objective**. Note the
> consequence for the `mask` argument (see `inventory-architecture.md`): with occurrence as the main task rather
> than a gate for a count regressor, `hierarchy_enabled` and its `mask = (y > 0)` semantics need re-deciding.

## 6. Ensemble / probabilistic losses

| Loss | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `crps` | — | 137 | 211 | `D-ONLY` | Sample-based CRPS over ensemble members | keep |
| `almost_fair_crps` | — | 164 | 238 | `D-ONLY` | Almost-fair CRPS — removes small-ensemble spread bias (the eval suite reports the matching *metric*) | keep |
Make sure that the way crps and almost-fair crps are implemented is consistent in both losses and scores

> Both are `D-ONLY` at the **loss** level, yet aru's `scores.py` computes the corresponding **metrics**
> (`crps_ensemble`, `almost_fair_crps_ensemble`). So A can score CRPS but cannot train on it.

## 7. Loss builders (config → callable dispatch)

| Builder | A | V | D | Status | What it does | Decision |
|---|---|---|---|---|---|---|
| `build_regression_loss(loss_config, transform_enabled)` | 101 | 270 | 368 | present in all | 🔀 `TRANSFORM` **in part**: the `transform_enabled` parameter and its `TRANSFORM_COMPATIBLE_LOSSES` guard become dead — **drop the second argument entirely**. The dispatch body itself stays | modify : incorporate option for finetuning |
| `build_binary_loss(loss_config)` | — | 223 | 315 | `D-ONLY` | ✅ `CLASSIF` Builder for the occurrence head — now the **primary** builder | mofidy : incorporate option for finetuning |
| `build_finetune_loss(finetuning_config)` | — | 249 | 341 | `D-ONLY` | Builder for **phase 2** of the two-phase MC-dropout fit — the hook the shared two-phase harness needs (merge task #4) | remove |

---

## Summary of what is at stake

| | Count | |
|---|---|---|
| `IDENTICAL` (safe either way) | 7 | Merge trivially |
| `D-ONLY` | 15 | **Lost unless `adrien-mc-dropout` is merged in** |
| `A-ONLY` | 0 | A is a strict subset of D, except that A's `build_regression_loss` is the version wired to the shared tuning harness |
| `STALE-DUP` (V) | 19 files' worth | Delete wholesale; zero unique content |

**The single structural fact:** on losses, `adrien-mc-dropout` is a **strict superset** of
`aru-probabilistic-eval`. The merge direction for this file is therefore D → shared, not A → shared, which is the
opposite of the shared-infrastructure direction (where aru is the base). Worth being deliberate about.

## Scope-change consequences (2026-07-28)

**Module-level constants that die with the transform** (both in `losses.py`):

| Constant | Fate |
|---|---|
| `TRANSFORM_COMPATIBLE_LOSSES = ('weighted_mse', 'asymmetric_huber')` | 🔀 **delete** — its only consumers are `build_regression_loss`'s guard and `search.apply_constraints` |
| `REGRESSION_LOSSES = (…, 'tweedie', 'poisson_nll', …)` | drop the two 🔢 entries; keep the rest |

This also **retires the stale-allowlist problem** described in the architecture notes: `weighted_mae` /
`weighted_rmse` were silently unreachable whenever the transform was on, purely because the allowlist was never
updated. With the transform gone, every distance loss is reachable and the question disappears.

**`intensity_weights` under the new targets** — deliberately *not* flagged, but its behaviour changes:

| Target | `w = (1 + y)^γ` becomes |
|---|---|
| hourly binary `{0, 1}` | two values, `{1, 2^γ}` → degenerates into a plain **positive-class weight** |
| daily `0–24` | `{1 … 25^γ}` → still a genuine intensity weighting, and well-behaved (no heavy tail) |

So it stays useful for the daily target and becomes redundant-but-harmless for the binary one, where
`focal_bce_with_logits`'s own `positive_class_weight` already does that job. Worth deciding which of the two
mechanisms owns class weighting rather than having both.

## Open questions for you

1. **The three spectral losses** (`psd_penalty`, `wmae_psd`, `afcrps_psd`) — keep all three, or only the
   composites, or only the primitive `psd_penalty` and build composites via config?
2. **Redundant pointwise family** — `mae`/`weighted_mae`/`rmse`/`weighted_rmse` are four thin wrappers over
   `_weighted_masked_mean`. Keep all four as named losses, or collapse to config knobs on one function?
3. **`crps_binary` vs `dice_loss` vs `brier_loss`** — all three target the occurrence head. Is the classifier head
   still in scope at all? (The plan lists "deprecated occurrence classifier" as a drop candidate, which may make
   this whole group removable.)

1. keep all three
2. Collapse mae and rmse into their weighted versions
Occurence head is still in scope. Keep it.
