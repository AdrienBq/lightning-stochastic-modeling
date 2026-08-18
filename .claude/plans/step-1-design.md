
# Step 1 (block a) — Design decisions via the inventories — ✅ **DONE** (2026-07-28)

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 0](step-0-bootstrap.md) · Next: [Step 2](step-2-config.md)

> ## ▶ RESUME HERE — Step 1 complete, **start [Step 2](step-2-config.md)**
>
> **Done:** prerequisite fetch; all four inventories written **and fully annotated**; the classification-first /
> no-transform scope change; plotting reference fixed as notebook 02a; plan split into per-step files; all three
> conflicts and all six architecture open questions resolved; **global `CLAUDE.md` + `README.md` written**.
>
> **Next action:** execute [Step 2](step-2-config.md) — write the 14 config files. Nothing blocks it.
>
> Do **not** re-run the inventory — it is complete and annotated on disk.

**Prerequisite — ✅ done.** `git fetch --all` run in `/home/aburq/repos/fers/fers26p8-knowledge-guided-tail-ml`;
`aru-probabilistic-eval`, `aru-diffusion-model`, `claude/quizzical-golick-7417bd`, `split-config`,
`zth_foundational_model` fetched, `adrien-mc-dropout` updated `1a39896..3f489f1`. All source code readable via
`git show origin/<branch>:<path>` (read-only, no checkout).

## Deliverables — the four inventories (all annotated)

| File | Contents | Annotated |
|---|---|---|
| [`inventory-losses.md`](inventory-losses.md) | 24 losses, 7 categories, 3 sources (A / V / D) | ✅ |
| [`inventory-scores.md`](inventory-scores.md) | 40 scores, 8 categories | ✅ |
| [`inventory-figures.md`](inventory-figures.md) | 02a visual spec, 13 pipeline figures, issues | ✅ |
| [`inventory-architecture.md`](inventory-architecture.md) | stages, modeling layer, ensemble contract, portability, 15-item transform-removal checklist | ✅ (Decision columns; open questions still unanswered) |

These four files **are** the Step 1 design decision record — they replace the originally-planned prose design
docs. Sources already consolidated into them: adrien's `docs/metrics_and_losses.md` (285 lines) and
`docs/distr_regression_pipeline.md` (140 lines), and aru's commented `config/metrics_daily.yaml`.

## Decisions taken

| Question | Decision |
|---|---|
| Modelling target | **Occurrence first** — hourly binary, or daily `0–24` hours-with-lightning. See the scope change in [`00-context.md`](00-context.md) |
| Target transform | **Dropped.** 15-item removal checklist in [`inventory-architecture.md`](inventory-architecture.md) §6 |
| Map style | **cartopy, hard requirement** (no lazy-import fallback) |
| Coastlines / borders | **No country borders.** |
| Plotting reference | **Notebook 02a**, style only. **Do not port the notebooks** — write plotting functions in `src/utils` inspired by 02a |
| Colour scale | **Per-date** (`ceil(nanmax(obs))` for the plotted day); drop 02b's hardcoded `vmax=8` |
| Date selector | **Drop `ipywidgets`** — date selection is a pipeline concern, not part of the visual grammar |
| Map extent | **Keep `set_extent([-5, 20, 30, 55])`** — do *not* make 55–60 °N visible, despite data existing there |
| PIT | **Drop** |
| `psd_full_fidelity` | **Remove**; standardise on A's `psd_band_ratios(...)['full']` + `psd_fidelity` |
| `crps_ensemble` / `almost_fair_crps_ensemble` | **Keep aru's `float` contract** |
| `dice_coefficient` / occurrence head | **Keep** — "occurrence head is still in scope" |
| Spectral losses | **Keep all three** (`psd_penalty`, `wmae_psd`, `afcrps_psd`) |
| `mae` / `rmse` | **Collapse into their weighted versions** → requires `intensity_weight_gamma` low bound extended to **0.0** |
| `build_finetune_loss` | **Remove** — finetuning becomes an option inside `build_regression_loss` / `build_binary_loss` |
| Per-folder `README.md`s | **Deferred to [Step 3](step-3-utils.md)**, when those folders exist |
| `metrics_daily.yaml` as design doc | **Restate as a clean reference doc** rather than pointing at the YAML |
| Tuning | **Unify all tuning** — one harness; `tune_mc_dropout` removed as a separate stage |
| `compute_high_lightning_days` | **Keep** — "this is the extremes" |
| `hello_world` | Check usefulness; remove if not |


## Key findings (evidence-backed, not assumed)

1. **The merge direction is opposite per file.** On **scores**, aru is a near-superset (adds `condition=` support
   and the whole streaming-ensemble layer; only `dice_coefficient` is lost by taking it wholesale). On **losses**,
   adrien is a *strict* superset — 15 of 24 are D-only, including the three spectral losses (`psd_penalty`,
   `wmae_psd`, `afcrps_psd`) that exist nowhere else and are the **only** losses optimising the spectral fidelity
   the shared suite measures. Merging uniformly in either direction loses real work.
2. ⚠️ **One genuine trap.** `crps_ensemble` / `almost_fair_crps_ensemble` exist on both branches with identical
   names but different return types — `float` (aru, aggregated, accepts `condition=`) vs `np.ndarray` (adrien,
   per-element). Merging by name yields silently wrong numbers, not an error. **Resolved: keep aru's.**
3. **⭐ Plotting is based on notebook 02a.** adrien's `src/utils/metrics/reporting.py` is **superseded and not a
   design input**. 02a beats 02b: one plotting cell instead of two near-identical ones, and `vmin=0` instead of a
   hardcoded `vmax=8`; everything else is identical. Full spec in [`inventory-figures.md`](inventory-figures.md) §1
   (EuroPP axes + PlateCarree transform, `origin='upper'`, labelled dashed gridlines, `make_lightning_cmap`,
   `draw_diff_map`'s complementary-mask over/under encoding). aru converged independently on
   EuroPP / `origin='upper'` / integer bins, corroborating the spec. aru still supplies the plumbing and the
   non-map figures.
4. **`DATA_ROOT` already exists — Step 5's mechanism is proven.** adrien's branch carries three launch scripts,
   one per machine (`mc_dropout.sh` → `/work/ext/st17/group8/data/` + conda; `mc_dropout_jz.sh` → Jean Zay
   `/lustre/.../batta_torch` + venv; `mc_dropout_local.sh` → the local machine + venv), each exporting
   `DATA_ROOT`, which the configs read. **This also explains this plan's original wrong data path** — copied from
   `mc_dropout_local.sh` (correct locally, wrong on the remote).
5. **Most of Step 1's prose already existed.** adrien's `docs/metrics_and_losses.md` + `docs/distr_regression_pipeline.md`
   plus aru's commented `config/metrics_daily.yaml` meant Step 1 was **consolidation, not authoring**.
6. **Both branches vendored stale copies of each other.** aru has `modeling/mc_dropout/` (5 files); adrien has
   `unet_aru.py` + `distr_regression_aru.py`. Confirmed stale by identical function order / line offsets with the
   newer additions missing. **Both sets deletable** — merge task #1, now evidence-backed.
7. **Two losses hid from the losses inventory:** `log1p_huber` / `log1p_huber_quantile` live in aru's `module.py`,
   not `losses.py`. Both are 🔢 `COUNT-REG` (log1p space). Placement still an open question.
8. **`TRANSFORM_COMPATIBLE_LOSSES` was a stale allowlist.** `('weighted_mse', 'asymmetric_huber')` — exactly aru's
   non-likelihood losses — so `weighted_mae`/`weighted_rmse`/`wmae_psd` were **silently resampled away** by
   `search.apply_constraints` whenever the transform was on. Dropping the transform dissolves this entirely.
9. **Grid confirmed 101 × 149**, `origin='upper'`, from the committed report figure. That figure also shows
   adrien's pipeline reporting has **no geography at all** — and is a vivid illustration of challenge (B): the
   prediction is visibly over-smoothed vs the speckled observation.
10. **`spread_skill_sums` uses `ddof=1`**, so **`M ≥ 2` is required** — an `ensemble-size: 1` smoke config yields
    silent `NaN`. Carried into [Step 2](step-2-config.md)'s smoke variants.
11. **PIT's real dependency** is on the *fitted gamma parameters* (`evaluation.py:468-470` reads
    `target_stats['gamma_shape']`/`['gamma_scale']` and calls `gammainc`), **not** on the transform being active.
    Removing `compute_target_transform_stats` is what breaks it. **Resolved: drop PIT.**

## Resolved conflicts (were blocking Step 2)

| Item | Resolution |
|---|---|
| `transforms.py` | **remove** — the whole file *is* the transform |
| `tweedie_deviance`, `poisson_nll` (losses) | **remove** |
| `tweedie_deviance_score` (score) | **remove** |
| `quantile_ratios`, `quantile_quantile` | **modify** to fit the `0–24` regression |
| `uniform_histogram_ks` (PIT) | **remove** |

## Resolved architecture questions

| # | Question | Answer |
|---|---|---|
| 1 | `log1p_huber` / `log1p_huber_quantile` placement | **Move into `losses.py`** |
| 2 | `PlattScaling` / `MonotoneCalibration` | **In scope, keep — and extend to MC-dropout.** *(`MonotoneCalibration`'s only incompatibility was with the gaussianized transform's signed values; with the transform gone and a non-negative `0–24` target, log1p space is valid again — so the earlier 🔢 flag no longer applies.)* |
| 3 | `MCDropoutEnsembleModule` divergence | **Take aru's** — it is the more up-to-date evaluation branch, so all scores/eval come from there |
| 4 | Legacy `evaluate_distr_regression` | **Delete** |
| 5 | `compute_high_lightning_days` | **Keep** — it is "the extremes" |
| 6 | Launch scripts | **Keep only one**; per-machine particularities move to the [Step 5](step-5-portability.md) per-user config |

## Final deliverables — ✅ done

- [`CLAUDE.md`](../../CLAUDE.md) — project framing (diagnostic mapping, *not* forecasting), the classification-first
  scope and its two consequences, data invariants (grid `101 × 149` / `origin='upper'`, `DATA_ROOT`, year split),
  **design invariants** (one evaluation, the ensemble contract, sums-not-means streaming, `ensemble-size ≥ 2`,
  `_weighted_masked_mean` as the shared reduction, `gamma = 0` ⇒ unweighted, residual-mode channel order),
  pipeline conventions, and a pointer to this plan. The four working Guidelines sections are preserved verbatim.
- [`README.md`](../../README.md) — new **Modelling scope** section (hourly binary vs daily `0–24`, no transform,
  the three families, why evaluation leads with base-rate-robust scores); **Data** section rewritten around
  `DATA_ROOT` with the grid/domain/zero-fraction invariants and the `ensemble-size ≥ 2` warning; template-only
  sections marked *(template)*. Also fixed a stray `++` typo in the install command.

**Gate satisfied:** the annotated inventories alone are the source of truth for Steps 2–4.
