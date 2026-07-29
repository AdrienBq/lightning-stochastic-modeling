# Step 3 (block c) — Shared `src/utils`

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 2](step-2-config.md) · Next: [Step 4](step-4-stages.md)

> **Status: 🔵 next.** [Step 2](step-2-config.md) is done, so the config contract this step implements is now fixed
> — see §"Obliged by Step 2" below for the concrete list. Still to be expanded into per-module detail, in the same
> way Step 2 was expanded from its one-line placeholder.

Build in dependency order, implementing the annotated inventories: `io/data.py` → modeling (`dataset`, `unet`,
unified `losses`, `search`, `validation`, two-phase `tuning`, then model modules + `registry`) → metrics
(`scores`, `evaluation`, `reporting`, `diagnostics`) → plotting functions covering the 02a map spec.

*(`transforms.py` is **not** built — the target transform is dropped. See [`00-context.md`](00-context.md).)*

## Carried into this step from earlier steps

- **The 15-item transform-removal checklist** — [`inventory-architecture.md`](inventory-architecture.md) §6.
- **Per-folder `README.md` standards** in `src/stages/`, `src/utils/metrics/`, `src/utils/modeling/`,
  `src/utils/plotting/`, capturing the agreed contracts. *(Deferred here from Step 1 — those folders don't exist
  until now, and documenting contracts before the code invites drift.)*
- **Plotting functions in `src/utils` inspired by notebook 02a** (not a notebook port). Full visual spec:
  [`inventory-figures.md`](inventory-figures.md) §1.
## Obliged by Step 2 — the config contract this step must satisfy

Every item below is *already referenced by a written config file*, so Step 3 is not free to skip it. Source:
[`step-2-config.md`](step-2-config.md) §7.

**New code (nothing on any branch implements these):**

| What | Where | Notes |
|---|---|---|
| `wmse_psd` | `modeling/losses.py` | `alpha * weighted_mse + (1 - alpha) * psd_penalty` — the `weighted_mse` sibling of the existing `wmae_psd`. A few lines reusing both |
| `roc_auc` | `metrics/scores.py` | Per event threshold, emitted through the categorical group's `<score>_<threshold>` grammar |
| `explained_deviance` | `metrics/scores.py` | **Bernoulli** explained deviance on the occurrence head vs climatology: `1 - logloss(model)/logloss(clim)`. Binary head only — the bounded `0–24` target has no likelihood left once Tweedie is gone |
| `roc_pr_curves` | `metrics/reporting.py` | ROC **and** PR curves per threshold. Plotting both is the point: at a 0.07 % base rate ROC flatters while PR exposes the real trade-off |
| `confusion_matrix` | `metrics/reporting.py` | 2×2 counts per threshold — the raw hits/misses behind `pod`/`far`/`csi`/`ets` |
| **MC-dropout warm start** | `modeling/mc_dropout_module.py` + `tuning.py` | When `tune` gets an `upstream-model-path`: load the upstream U-net's weights, **skip phase 1**, run the finetuning phase alone. The module has `set_phase('train'\|'finetune')` and a `finetuning_enabled` gate but **nothing that loads foreign weights**. Needs a load-time architecture check (`in_channels`, `base_channels`, `depth`, `activation`, `normalization`) that **raises naming the offending field** — silent partial `load_state_dict` is the failure mode to prevent |

**Modifications to ported code:**

- `losses.build_regression_loss` / `build_binary_loss` absorb the `finetuning` option; **`build_finetune_loss` is
  deleted** (it is a nested config block now, not a separate builder).
- `build_output_activation` reads the **`min_hours`/`max_hours` pair**: `clamped_sigmoid` becomes
  `min_hours + (max_hours - min_hours) * sigmoid(z)`. ⚠️ `softplus` only guarantees `>= 0`, so it silently violates
  a non-zero `min_hours` — either constrain that combination out or clamp after the activation.
- `search.apply_constraints` — **not** the no-op §6 item 6 predicted. Two live rules: force
  `intensity_weight_gamma = 0` when `occurrence_head.loss == focal_bce` (on a binary target `(1+y)^γ` collapses to
  `{1, 2^γ}`, duplicating `positive_class_weight`); and force `finetuning.enabled = true` + ignore the `unet:` block
  when `tune` is given an `upstream-model-path`.
- `tune` records the resolved `selection` block into `best_trial.json`; **`retrain_best` reads it back** from
  `source-path/best_trial.json` and takes no `model-config` and no `selection-metric`/`selection-mode`.
- `psd_fidelity` unified behind a single `band:` argument (`high`/`mid`/`low`/`full`), with `psd_full_fidelity` and
  `psd_high_fidelity` kept as *metric keys* served by it — `scores.psd_full_fidelity` as its own function is gone.
- `fss_useful_scale` reuses the precomputed per-scale FSS instead of recomputing it.
- `unet.MonotoneCalibration` **stays** (exposed as `calibration.regression`) — §6 item 9's removal is struck.
- Every family's module must honour `predict_step`'s ensemble contract, and the deterministic one must be
  warm-start-compatible, i.e. **group norm** (its search space fixes `normalization: group`).

~~`evaluation.resolve_threshold` gains a `kind: absolute` branch~~ — **already supported**; `absolute` is in fact the
default kind (`aru:src/utils/metrics/evaluation.py:58-76`). No work needed.
