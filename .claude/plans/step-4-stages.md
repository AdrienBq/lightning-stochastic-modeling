# Step 4 (block d) — Stages + common evaluation

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 3](step-3-utils.md) · Next: [Step 5](step-5-portability.md)

> **Status: 🔵 next.** Expanded 2026-08-14, after Step 3 closed.

## Context

Step 3 finished `src/utils` as one merged library with a durable suite (1102 passing, 291/291 functions tested, 86 %
line coverage). **`src/stages/` still holds only the plumber template** — `run.py`, `setup.py`, `hello_world.py`,
`__init__.py`. The twelve shipped configs name **seven** stages and only `setup` exists, so *nothing in this repo runs
end to end today*: every model family, every loss, every score and every figure built in Step 3 is reachable only from
a test.

Step 4 writes the five missing stages so all three families run one pipeline and report the same metrics.

The framing from [`inventory-architecture.md`](inventory-architecture.md) §1 still holds — **A has the unified stage
surface, D has the only MC-dropout stages** — but the merge is far smaller than that table suggests, because **A had
already factored the sweep harness out into `src/utils/modeling/tuning.py`, and Step 3 folded D's two-phase fit into
it.** A's `tune_distr_regression.py` is 121 lines of pure argument forwarding. So the stages are thin wrappers over
code that already exists and is tested.

### What is actually on the two branches

Measured with `git -C …fers26p8-knowledge-guided-tail-ml show origin/<branch>:src/stages/<file>`:

| | A `aru-probabilistic-eval` | D `adrien-mc-dropout` |
|---|---|---|
| prepare | `prepare_regression` **760** (one stage, all families) | `prepare_distr_regression` 254 + `prepare_mc_dropout` 307 |
| tune | `tune_distr_regression` **121** + `tune_diffusion` 165 — thin wrappers | `tune_distr_regression` 297 + `tune_mc_dropout` 349 — each with its own `_fit_trial` |
| retrain | `retrain_best_distr_regression` **104** + `retrain_best_diffusion` 109 | ✗ absent |
| evaluate | `evaluate_regression` **407** (one stage, all families) | `evaluate_distr_regression` 182 + `evaluate_mc_dropout` 360 |
| compare | `tabulate_metrics` **116** + `combine_curves` **252** | ✗ absent |
| other | — | `compute_high_lightning_days` 54, `hello_world` 11 |

**Base = A throughout.** D's stages are superseded: `tune_mc_dropout._fit_phase` is already inside
`tuning.py::_fit_trial` (block 3b-2), and `evaluate_mc_dropout` is replaced by the shared evaluation — the design
invariant CLAUDE.md states as *"One evaluation for all families. Never add a family-specific evaluation path."*

## Decisions taken (2026-08-14)

1. **One shared prepare stage keeping `mode:`, but both misleading names go.** `prepare_regression` →
   **`prepare_modeling`**, `evaluate_regression` → **`evaluate`**.
   ⚠️ This **supersedes [`step-3-utils.md`](step-3-utils.md) §3's stage-split note**, which proposed
   `prepare_regression` (daily) + `prepare_classification` (hourly) and dropping `mode:`. That note contradicted the
   twelve configs Step 2 shipped *and* CLAUDE.md's *"`mode` is the only key that selects between them"*. Struck there,
   with the reason, as part of block 4a — a design record that disagrees with the code is worse than no record.
2. **Full hourly, with a pipeline.** The 0/1 occurrence branch of `_derive_target` is new code and gets a runnable
   pipeline + smoke tier. Otherwise `build_binary_loss`, `PlattScaling`, `kind: probability`, `climatology_brier`,
   `explained_deviance`, `dice_occurrence`, the threshold-free FSS and `valid_classification_score` all stay dead
   code — built and tested in Step 3, reachable by nothing.
3. **`compute_high_lightning_days` becomes a utility, not a stage.** Into `src/utils/io/data.py` beside
   `index_samples` / `load_dataset_metadata`, which already parse the same `metadata.csv`, so the CSV-reading knowledge
   stays in one file. No `src/stages/` entry, no mirrored stage test.
4. ~~**`hello_world` is deleted**~~ — ⛔ **REVERSED in block 4a, after the check the decision asked for.** The
   criterion was "template scaffold with no consumer", and it has two: `README.md` links `config/hello_world.yaml`
   twice as *the* worked example of the pipeline-config format, and `run_project.py`'s own docstring points at it for
   the `lazy` / `ensure_determinism` keys. It is also the only pipeline in the repo that runs with **no `$DATA_ROOT`
   at all** — the cheapest possible smoke of the orchestrator itself, which is exactly what Step 0 used as its gate.
   Its config documents four top-level keys the family pipelines never show (`banner_font`,
   `lazy_content_max_file_mb`, `lazy_content_max_dir_mb`, and the bare-minimum `stages` form). **Kept.**
5. **A synthetic-root end-to-end test; the smoke configs stay where they are.** See §"End-to-end verification". The
   real-`$DATA_ROOT` smoke run remains a human-invoked gate.

## The seven stages after the merge

| Stage | Base | Est. lines | Note |
|---|---|---|---|
| `setup` | already in repo | 44 | unchanged |
| `prepare_modeling` | A `prepare_regression` | ~700 | the only large stage; `target_variable` out, hourly binary target in |
| `tune` | A's two wrappers, unified | ~170 | dispatches `module_factory` by `model-family`; owns the MC-dropout warm-start `partial` |
| `retrain_best` | A's two wrappers, unified | ~130 | same dispatch; selection metric read back from `best_trial.json` |
| `evaluate` | A `evaluate_regression` | ~370 | the shared evaluation; loses five dead map-colour arguments |
| `tabulate_metrics` | A | ~116 → **135** | near as-is; the display-name map dropped as a bug |
| `combine_curves` | A | ~260 → **355** | `_combined_qq` deleted, reliability + ROC/PR overlays added |

---

## Per-stage work

### `prepare_modeling` — base A `prepare_regression.py` (760)

Keep the whole structure: `_prepare_base` (A:287), `_write_feature_file`, `_backfill_features`,
`_feature_layout_for_mode`, the accumulator / `_summarize` / `_zero_proportion_report` block, and
`_materialize_upstream` (A:500). Deltas:

- ❌ **`target_variable` is gone.** Drop the parameter and collapse
  `_daily_aggregation(lightning, target_variable, hourly_threshold)` to `_daily_aggregation(lightning,
  hourly_threshold)` returning `kept.sum(axis=0)` — the lightning-hours count. The `lightning_counts` /
  `lightning_peak` branches go with it.
- ➕ **The hourly target is NEW code.** A returns raw `uint16` counts `[T, H, W]` in hourly mode; this project needs
  **0/1 occurrence**, thresholded with the *same* `hourly_threshold`:

  ```python
  if mode == MODE_HOURLY:
      return (lightning >= hourly_threshold).astype(np.float32)   # [T, H, W], 0/1
  return _daily_aggregation(lightning, hourly_threshold)          # [H, W], 0-24
  ```

  Sharing the cutoff is what keeps the two tasks' denoising consistent — daily counts the qualifying hours, hourly
  emits whether the hour qualified. Also **remove A's loud "hourly mode ignores hourly_threshold" warning**, which is
  no longer true.
- 🔧 `load_regression_module` → **`load_model_module`** inside `_materialize_upstream`.
- 🔧 Defaults to match the shipped configs: `split_config='config/split/split.yaml'`, `feature_dtype='float16'`,
  `hourly_threshold=2`.
- ✅ `positive_quantiles` **stays** — `_prepare_base` writes it (A:441-461); only `gamma_shape` / `gamma_scale` went,
  and they went with `compute_target_transform_stats` in block 1.
- ✅ `_materialize_upstream`'s idempotency signature (resolved path + size + mtime) is right as-is: an in-place retune
  of the same path invalidates rather than silently reusing stale predictions.

**Parameters, from the configs verbatim:** `data-path`, `split-config`, `output-path`, `mode`, `feature-aggregation`,
`features`, `hourly-threshold`, `overwrite`, `materialize-features`, `feature-dtype`, plus diffusion's
`upstream-model-path`, `upstream-accelerator`, `upstream-devices`, `upstream-num-workers`, `upstream-batch-size`.

### `tune` — NEW, unifying A's two wrappers

A's wrappers differ only in `module_factory` + `study_name`, so one stage with a family dispatch replaces both. The
two pieces of real logic:

```python
MODULE_FACTORIES = {                       # or reuse registry.MODULE_REGISTRY / FAMILY_NAMES
    'deterministic_unet': DeterministicUnetModule,
    'mc_dropout':         MCDropoutModule,
    'diffusion':          DiffusionModule,
}

# ⚠️ THE fork: one `upstream-model-path` string, two independent uses.
# MC-dropout wants the upstream's WEIGHTS (a warm start, loaded here);
# run_sweep separately hands the same string to apply_constraints, which forces finetuning.enabled.
factory = MODULE_FACTORIES[model_family]
if model_family == 'mc_dropout' and upstream_model_path:
    factory = partial(MCDropoutModule.from_upstream, upstream_model_path)
```

- `study_name=f'tune_{model_family}'` so the three families' optuna journals cannot collide on resume.
- ❌ **No `selection-metric` / `selection-mode` parameters** — the search space's `selection:` block is the single
  source of truth (rebuild-plan #6); `run_sweep` reads it from `model-config` and records it into `best_trial.json`, so
  a retrain can no longer disagree with the sweep that chose the configuration.
- ⚠️ **`upstream-model-path` must reach only `mc_dropout`.** Diffusion's sits on `prepare_modeling` (its `tune` block
  has no such key), so it must not be quietly forwarded for that family — a diffusion trial receiving one would be
  constrained by a rule written for MC-dropout.
- Every other argument forwards to `run_sweep`, whose signature already accepts them (verified against the current
  `tuning.py`, `upstream_model_path` included).

### `retrain_best` — NEW, unifying A's two wrappers

Same `MODULE_FACTORIES` dispatch over `retrain_best_config`. Parameters from the configs: `model-family`,
`model-type`, `source-path`, `input-path`, `metrics-config`, `output-path`, `metrics-path`, `max-epochs`,
`early-stopping-patience`, `accelerator`, `devices`, `num-workers`, `progress-bar`. No `selection-metric` /
`selection-mode` — read back from `source-path/best_trial.json`.

➕ **A warm-started sweep is retrained WARM-STARTED** (user decision, block 4b — this plan first said the opposite).
The claim that "a retrain fits from fresh weights by definition" was an assertion, not something the design implies,
and it breaks the stage's own contract: the sweep's hyperparameters were chosen under a one-phase fit from the
upstream's weights, so a from-scratch retrain runs two phases from random weights and answers a different question.
`tuning.retrain_best_config` already said as much at its warm-start branch — *"the stage must supply them to
module_factory"* — so the stage was contradicting the harness it calls.

Resolution order, through the same `tune._module_factory` so the two stages cannot drift: an explicit
`--upstream-model-path` (for when the upstream itself was retrained), else **whatever the sweep recorded in
`best_trial.json`**. The shipped pipeline therefore needs no new config key, and a recorded checkpoint that no longer
exists **raises** rather than falling back to a from-scratch fit — that fallback is the silent regime change the whole
arrangement exists to prevent.

### `evaluate` — base A `evaluate_regression.py` (407)

The orchestration is correct as written, and every function it calls exists with a matching signature (checked:
`run_metric_suite`'s 7-positional + 3-keyword call, `build_baselines`, `merge_ensemble_partials`,
`finalize_ensemble_metrics`, `dataset.items_frame()`). Deltas:

- 🔧 `load_regression_module` → **`load_model_module`**.
- ❌ **Five dead arguments go:** `colorbar_scale`, `colorbar_integer_bins`, `quantize`, `max_val` and
  `occurrence_event` are all gone from `write_report` under the 02a grammar — the scale is always unit bins in
  lightning-hours driven by `ceil(nanmax(obs))` per date. Current signature:
  `write_report(report_path, reporting_config, metrics_flat, curves, prediction, observation, items,
  ensemble_members=None, plot_dates=None)`.
- ⚠️ **Name collision to handle:** the stage's own `residual_diagnostics: Optional[bool]` parameter shadows the
  function. A imported it aliased; keep that —
  `from src.utils.metrics.diagnostics import residual_diagnostics as compute_residual_diagnostics`.
- ❌ **Delete the `occurrence_threshold` warning block.** `resolve_occurrence_event` now hard-asserts that no
  `occurrence_threshold` reappears, so the branch is unreachable.
- 🔧 **The headline log names are stale** and would silently print nothing: `ets_p99` → `ets_h6`, `fss_p90_s3` →
  `fss_h6_s3`, plus `rank_corr_p99`, `over_frac_p99`, `mae_cond_pos`. Rebuild the list from keys the suite actually
  emits — `evaluation_test.py`'s `EXPECTED_DAILY_KEYS` (34 keys) is the authority.
- ➕ **Hourly branch:** in hourly mode the model's own output IS the probability, so `probability` must be populated
  from `prediction` rather than left `None` — otherwise the reliability diagram, `explained_deviance` and
  `dice_occurrence` silently vanish on the one task they were built for.
- ✅ Keep the ensemble pooling (`[N*M, H, W]` structure stack with obs replicated per member), the `> 8 GB` warning,
  and the CUDA-reported-but-unusable fallback.

### `tabulate_metrics` — A (116) → 135. ✅ DONE in 4d

`**kwargs` family→JSON mapping, union columns, sorted, `selected-metrics` subsetting. Two changes to A:

- 🐛 **`_DISPLAY_NAMES` is GONE, and it was a bug, not a simplification.** A's map covered `U_net` / `Diffusion_Model`
  — *neither of them a label any shipped config uses* — so ported as-is, `Deterministic-UNet` would have fallen through
  to A's underscore→space fallback and appeared as `Deterministic UNet`, disagreeing with both the config and
  `combine_curves`' legends. `_display` now undoes Fire's substitution instead (`key.replace('_', '-')`), which
  restores all three shipped labels with no map to go stale when a family is added. A's own tests asserted its invented
  labels, so the mislabelling was invisible; the replacement test derives the labels from
  `probabilistic_eval*.yaml` and asserts each round-trips.
- ➕ **The per-family missing-metric list is logged.** The phase gate wants *identical metric-key columns across
  families*, but the columns are identical **by construction** — one DataFrame has one column set — so the CSV alone
  cannot show a disagreement. What shows it is the NaN pattern, and the log is where it is stated in words: a
  deterministic family missing `crps` is expected, a family missing `mae` is a merge failure.

### `combine_curves` — A (252) → 355, 02a restyle. ✅ DONE in 4d

- ❌ **`_combined_qq` is gone**, as planned: `qq_table.csv` is a file this repo never writes, so it would have logged
  *"no model had a usable qq_table.csv; skipped"* on every run forever.
- ➕ **`_combined_reliability`** from `reliability_table.csv` — two panels, the curve and the bin populations, because a
  reliability curve carried by one populated bin is indistinguishable from a calibrated one without them.
- ⚠️ **`_combined_roc_pr` could NOT be built from `roc_pr_summary.csv`** as this plan said. That file holds four
  *scalars* per threshold (`roc_auc`, `average_precision`, `base_rate`, `from_probability`); the curve points live only
  in the in-memory `curves['roc_pr']` block. From the summary alone the figure would have been a bar chart of numbers
  already in `tabulate_metrics`' table — strictly worse than the table. So `reporting._roc_pr_curves` now also writes
  **`roc_pr_curves.csv`** (long-format points), which is what every other curve figure already did; the summary keeps
  its role, supplying the legend annotations and the PR no-skill floor. ~10 lines in the existing `if 'csv' in formats`
  block, two tests, no signature change.
- 🔧 **One threshold, not four.** `metrics_daily.yaml` declares `[occurrence, h3, h6, h12]`, so overlaying every one across
  three families is twelve lines per panel and a PR panel on a log axis is unreadable at that density.
  `_headline_threshold` picks `occurrence` (the event `average_precision_occurrence`, the selection score's
  discrimination term, is defined on) and falls back to the **first-listed** threshold — which is what makes it work
  for the hourly task, whose thresholds are probability cuts with no `occurrence` among them.
- 🔧 The PSD x-axis reads **`wavelength_km`**, inverted. Reading `wavelength_px` instead is a 27.75× error that still
  draws a plausible loglog figure, which is why a test asserts the plotted x-data equals the kilometre column.
- ✅ Kept: the ±1σ ensemble band, `_combined_fss`'s two legends, `_combined_rank_histogram`, `_model_colors` over
  sorted names, and `_read_csv`'s never-raise contract.

**On the tests.** Both placeholder files were rewritten rather than un-skipped. A's `report_dirs` fixture wrote
`x,y\n1,0.5\n...` for every curve — a schema no figure matches — so every figure would have self-skipped and every test
passed while drawing nothing. The fixture now builds each report directory by calling the real
`reporting._psd_curves` / `_fss_vs_scale` / `_reliability` / `_roc_pr_curves` / `_rank_histogram`, so a column renamed
on either side fails here; and the figures are inspected as **figures** (a `_save` spy over `axis.lines`) rather than as
files, since a png existing says nothing about what is on it. Five mutations were run against the result — px for km,
no inversion, no occurrence preference, a no-skill line per family, no finite mask — and each failed exactly the test
that claims to catch it.

### Deletions and the utility move

- ~~`src/stages/hello_world.py`~~ — kept, see decision 4.
- `compute_high_lightning_days` → `src/utils/io/data.py` as `high_lightning_days(data_path, quantile=0.95)`, tested in
  the existing `data_test.py`.
  ⚠️ It carries a documented leak caveat: the quantile is taken over **every** day in `metadata.csv`, test years
  included. That is right for describing the dataset and wrong for anything feeding a model. Documented rather than
  changed (filtering would make it the wrong tool for its actual job), and a test asserts the caveat survives edits.
- Not ported: D's `prepare_*` / `tune_*` / `evaluate_*`; A's `evaluate_distr_regression` and
  `prepare_distr_regression` shims, and `tune_diffusion` / `retrain_best_diffusion`.

---

## The hourly pipeline — ✅ DONE in 4f (4 configs, 23 tests, no `src/` change)

The stage code above makes hourly *derivable*; these make it *runnable*.

**The headline finding: block 4f wrote no Python at all.** Every layer was already mode-aware and the wiring was
already right — `prepare_modeling._derive_target` emits the 0/1 field, `unet_module_base` reads `mode` from
`target_stats.json` and flips head/prediction/calibration, the loss dispatch is by NAME so both loss families are
admissible, `deterministic_module.predict_step` returns `probability=prediction` when hourly, `run_metric_suite`
detects a probability field from the arrays, and `reporting` already suffixes figure names with the hour. So this block
is **four config files, twenty-three tests, and a gate** — which is what blocks 4a–4c were paying for.

- ➕ **`config/eval/metrics_hourly.yaml`** — a sibling of `metrics_daily.yaml`, whose own prose already specified what
  must change. The load-bearing edit is `metrics.categorical.thresholds`: entries must be `kind: probability`
  (`p50: {kind: probability, value: 0.5}`), **not `occurrence`**. An `occurrence` entry resolves to `> 0` on a
  probability field, so it fires on every cell with any non-zero probability — POD ≈ 1, FAR ≈ base rate, a full
  contingency table of nonsense with no error raised. `run_metric_suite` warns at that configuration; the config must
  not reach it. Dropped as planned: `mae_stratified`, `estimation_tendency`, `rank_correlation` and the
  `error_by_intensity_bin` figure that reads the first one's curve. **Kept, deliberately:** the whole `ensemble` group
  (this is the hourly suite for *all* families, as `metrics_daily.yaml` is the daily one) and `mae`, which is
  IMPROPER against a 0/1 observation but stays reported for comparability with the daily suite — no selection score
  and no loss may use it.
- ➕ **`config/deterministic_unet/search_space_hourly.yaml`** — ⚠️ **not in the original plan, and required.**
  `selection_metric_for_mode` RAISES when a search space's `selection.metric` disagrees with the prepared mode, so an
  hourly pipeline cannot reuse a daily space. Its four deltas: binary/proper losses only, `calibration.occurrence`
  (Platt) with the daily-only `regression` warp gone, `output_activation`/`max_hours` omitted (unread in hourly), and
  `valid_classification_score` with the classification weights.
  ⛔ **Four loss names are excluded and each exclusion is load-bearing:** `weighted_mae` / `wmae_psd` because MAE is
  IMPROPER on a 0/1 target (minimized by a sharp forecast, so the all-zero prediction wins and nothing in the code
  objects); `asymmetric_huber` because conservativeness is a statement about a magnitude a probability does not have;
  `crps_binary` because it needs a real ensemble and this family's single forward pass gives N = 1.
- ➕ **`config/deterministic_unet/deterministic_unet_hourly.yaml`** + its `*_smoke_cpu` tier. Deterministic family
  only — see the note below on why the other two are not a copy-paste away.
- ⚠️ **24× the items**, and the plan's sampler claim needed one correction. `DayGroupedShuffleSampler` becomes live
  when `mode == hourly` **and** `not uses_materialized_features` — hourly alone is not enough. So the hourly tiers set
  **`materialize-features: false`**, which is load-bearing twice: materializing would cost a SECOND ~20 GiB (the daily
  directory already holds every hour, laid out variable-major) and it is the only way the sampler ever runs. Both
  tiers, so the smoke tier smokes the loader path the real run uses.
- ⚠️ **`ensemble-size` ≥ 2** as always: `spread_skill_sums` uses `ddof=1`, and a single member yields a silent `NaN`.
- ⭐ **The hourly maps plot the DAILY TOTALS** (`reporting._sum_hours_into_days`, added in 4f-r after the gate showed
  the problem). The hourly stack is summed over each date's hours before anything is drawn, so there is one figure per
  DATE and one plotting grammar for both tasks. It is a change of **units**, not a plotting convenience, and it is
  exact in both panels: a 0/1 observation summed over a date *is* the `0-24` lightning-hours field `mode: daily`
  prepares (same `hourly-threshold`), and the predicted probabilities summed over a date are
  `sum_h P(lightning at hour h)` — the **expected** lightning-hours that day, which is what a daily model predicts
  directly. So an hourly and a daily figure of the same date are comparable panel for panel.
  * ⚠️ **What it hides, accepted:** errors that cancel across a day's hours cancel here too, so a model that puts the
    day's lightning at the wrong *hour* but the right cell looks perfect on these maps. Every metric stays hourly —
    the reliability diagram, the ROC/PR curves and the `p50` confusion matrix read the un-aggregated arrays.
  * The per-hour file name and title (`maps_<date>_hHH_*`, `<date> hHH event`) became **unreachable** and were
    removed: after the sum, every plotted item is a day in both modes.
  * **`HOURLY_PLOT_CATEGORIES = ('most_active',)`** — an hourly run plots the most-active days only, named
    `maps_<date>.png` with no category tag. Both omissions are reasoned: `worst_error` ranks on the error in the DAILY
    TOTAL once the hours are summed, so it would select on the very quantity the sum hides, and `median_activity`
    exists for a typical-vs-extreme contrast that is the daily task's product rather than this one's. A DAILY run
    keeps all three — the narrowing is hourly-only, checked in both directions.
  * What it replaced: the colour axis is observation-driven (`ceil(nanmax(obs))`), so a 0/1 observation pinned
    `max_val` at 1 and the warm palette collapsed to white-below-0.5 / grey-above under an `h / day` label naming a
    unit that was not on the axis.
  * ⚠️ **`evaluate` never cleans its `report-path`.** Three successive runs of the hourly tier left nine superseded
    map figures beside the two current ones, indistinguishable in a listing except by mtime — which read as "the
    change did not work". Not fixed here (deleting a declared output on write is its own decision), but it is the
    second time a stale report file has cost a debugging round: worth either clearing the directory in `write_report`
    or logging the files a run did NOT write.
  * Fixed while looking at the result: gridspec `wspace` 0.05 → **0.14** in both map layouts. Cartopy draws gridline
    labels OUTSIDE the axes and gridspec does not count them, so adjacent panels ran their longitude labels together
    into `20°E5°W`. This was one of the two cosmetics flagged-and-unassigned at the end of 4e.

### ⚠️ Why there is no hourly `mc_dropout` or `diffusion`, and what each would cost

Not an oversight, and the two are not the same distance away:

| | reachable? | what it needs |
|---|---|---|
| `mc_dropout` | **yes, config only** | its own hourly search space (`dropout_p` + the `finetuning` block) and a tier pair. The module is fully mode-aware: `_to_prediction_differentiable` sigmoids when hourly, `predict_step` returns `probability`, and phase 2's ensemble loss consumes the TARGET space — probabilities — where continuous CRPS against a 0/1 outcome is a proper score. It could also warm-start from the hourly deterministic upstream and share its prepared directory, exactly as the daily pair does. |
| `diffusion` | **no — a code-level gap** | its `predict_step` returns `'probability': None` by design ("no occurrence-probability head in this family") and its prediction is `clamp(upstream + residual, 0, max_hours)`, a continuous field rather than a value in [0, 1]. So `prediction_is_probability` is false and all four keys the hourly task exists for are absent or NaN, while `selection_metric_for_mode` still forces `valid_classification_score` with its 0.20 Brier term permanently NaN. Making flow matching emit a calibrated probability (generate logits and sigmoid? generate in probability space?) is a **design decision**, not a copy. |

### The daily/hourly naming rename (4f, unplanned)

With `metrics_hourly.yaml` beside `metrics.yaml`, an unsuffixed name became ambiguous — it read as "the metrics
config" while being one of two. So **every task-specific config now names its task**: 13 files renamed with `git mv`
(`metrics_daily.yaml`, 3 × `search_space_daily.yaml`, and 9 × `<family>_daily[_smoke_cpu|_smoke_gpu].yaml`) plus a
60-file textual sweep over configs, `src/`, `tests/`, the plans, `README.md`, `CLAUDE.md` and `job_scripts/`.

⚠️ **The `$OUTPUT_ROOT` directory names were deliberately NOT renamed.** `$OUTPUT_ROOT/deterministic_unet_smoke_cpu`
stays as it is: ~60 GiB of prepared data and checkpoints from the 4e gates live there, and renaming would orphan every
one of them to disambiguate a *source* file. `job_scripts/_common.sh` gained a `MODE=daily|hourly` variable that
derives the prepared dir, the run dir, the search space and the metrics config **together**, so a stage run by hand
cannot mix a daily search space with an hourly prepared directory.

## The rename surface

`prepare_regression` → `prepare_modeling`, `evaluate_regression` → `evaluate`. Mechanical but wide — 36 files. The
functional ones:

| Where | What |
|---|---|
| 9 × `config/{family}/{family}_daily{,_smoke_cpu,_smoke_gpu}.yaml` | the two stage keys |
| 2 × `config/eval/probabilistic_eval{,_smoke_cpu}.yaml` | `evaluate_regression:` × 3 each |
| `tests/stages/evaluate_regression_test.py` | → `evaluate_test.py` |
| `tests/completeness_test.py` | the hardcoded Step-4 set (L112-114) and the census (L170) |

Prose-only (docstrings and comments naming the stage): `CLAUDE.md`, `README.md`, `config/eval/metrics_daily.yaml`,
`config/split/split*.yaml`, the three `search_space_daily.yaml`, `src/utils/io/data.py`, `registry.py`, `search.py`,
`mc_dropout_module.py`, `mc_dropout_eval.py`, and 4 test files. Update them — a comment naming a stage that does not
exist is how the next reader loses an hour.

---

## End-to-end verification

A smoke run (`python run_project.py config/<family>/<family>_daily_smoke_cpu.yaml <EXPERIMENT>`) has exactly the right
**coverage**: it drives `run_project.py` → the orchestrator → every stage subprocess → the whole library, and it is the
only thing reaching the 413 statements of `tuning.py` + `stages/run.py` that no unit test can. It has none of the
properties that make a test a test:

| | smoke run today | what is needed |
|---|---|---|
| assertions | a human reading a log | asserted in code |
| data | real `$DATA_ROOT` | runnable by anyone |
| location | a shell command in a plan file | inside `pytest` |

The first row is the important one. *"Assert declared artifacts + expected metric keys"* is a human-read check — the
exact failure mode block 5b eliminated for the nine gate scripts. Repeating it for the pipeline layer would be the same
mistake in a new place.

### The smoke configs stay in `config/<family>/`

Their value is being **the real pipeline at a smaller scale** — same schema, same stage list, adjacent to the full
config so a change to one is a visible change to the other. The entire delta is `lazy: false`, a smaller
`split-config`, smaller output paths, and `n-trials` / `max-epochs` / `ensemble-size` turned down. Moving them under
`tests/` would let them drift from the pipeline they exist to smoke-test — the one thing they must not do — and would
break CLAUDE.md's documented three-tier convention plus the 19-config parse sweep in `parse_config_test.py`.

### `tests/pipeline_e2e_test.py` — synthetic root, always runs. ✅ DONE in 4e (18 tests, ~115 s)

**Four things the design above got wrong or did not know.** The rest of this section stands as written.

1. 🐛 **`mlflow.projects` shells out to a BARE `python`, not `sys.executable`** — the first thing the test hit, and a
   real portability wart rather than a test problem. With no `MLproject` file, MLflow's local backend builds
   `python src/stages/<stage>.py ...`, so every stage subprocess resolves its interpreter from `PATH`. It works today
   only because the launch scripts activate the venv first. The test prepends `os.path.dirname(sys.executable)` to the
   subprocess `PATH`; the user-facing fix is Step 5's
   ([recorded there](step-5-portability.md) with the options).
2. ⚠️ **`EXPECTED_DAILY_KEYS` could not be asserted as a subset.** `evaluation_test.py`'s `daily_suite` fixture calls
   `run_metric_suite` with a **non-None `probability`**, which no daily pipeline produces — `deterministic_module`
   returns `probability=None` in daily mode because there is no occurrence head. So `explained_deviance` is
   *structurally* absent from a real daily run, and two more keys (`fss_useful_scale_occurrence`,
   `mae_bin_occurrence_h3`) are absent on a 2-day untrained split. The test imports the list and subtracts two named
   exclusion tables, asserting **set equality on the difference** — so it fails both when a key disappears and when an
   excluded key becomes emittable, which would mean the exclusion note is stale.
3. ➕ **Only ONE parameter had to be rewritten** — `split-config`. Sizing the synthetic dataset at 12 days (8 train)
   lets the shipped `feature-stats-days: 4` stand, so the derived config is otherwise the shipped pipeline verbatim,
   and a test pins that exactly one non-comment line differs. Both roots come from the environment
   (`DATA_ROOT` / `OUTPUT_ROOT`), which is the shipped mechanism, and `MLFLOW_TRACKING_URI` redirects the store into
   `tmp_path` so a test run never writes into the checkout.
4. ⚠️ **The comparison layer is tested with ONE family's artifacts under all three shipped labels.** Three trained
   models would triple the runtime, and "the three families emit identical metric keys" genuinely needs three models —
   it stays with the by-hand cross-family gate. What the e2e test proves instead is the plumbing no unit test can
   reach: that `--Deterministic-UNet <path>` survives `run_project` → MLflow's parameter list → Fire's hyphen
   substitution → the stage's `**kwargs`, and comes back out as a row label spelled the way the config spelled it.
   Block 4d's unit tests call `tabulate` directly, so they cover `_display` and neither layer in front of it.

**Anti-vacuity, checked:** three simultaneous mutations in different subsystems — branch A's space fallback in
`_display`, one figure builder unwired in `combine_curves`, and a deliberately stale exclusion entry — each failed
exactly its own test while the other 15 stayed green.

Lives at the `tests/` root, like `completeness_test.py`, because it mirrors no single module — it exercises all of
them. ⚠️ **Add it to `completeness_test.py`'s expected set (L110-111)** or
`test_no_test_file_outlives_its_module` flags it as an orphan.

Two halves, both deliberate:

1. **A `$DATA_ROOT`-shaped fixture** — `metadata.json`, `metadata.csv`, `samples/sample_XXXXXX.pt` on a small grid.
   `data_test.py`'s `sample_directory` fixture already builds exactly this shape and is the thing to lift into
   `conftest.py`.
2. **A config DERIVED from the shipped smoke YAML**, not a fixture copy: parse
   `config/<family>/<family>_daily_smoke_cpu.yaml`, rewrite only `data-path`, `split-config` and the output paths into
   `tmp_path`, dump it, invoke `run_project.py` on that. Derived means it **cannot drift** from the real pipeline —
   the same principle as `test_a_REAL_tune_stage_has_its_config_parameters_classified_as_inputs`, which reads the
   shipped `deterministic_unet_daily.yaml` rather than a hand-built dict.

What it asserts, in code:

- every declared `output-path` / `metrics-path` / `report-path` exists and is non-empty;
- the metrics JSON's keys ⊇ `EXPECTED_DAILY_KEYS` — the 34 keys `evaluation_test.py` already pins, so the unit suite
  and the pipeline suite cannot disagree about what the evaluation emits;
- the report directory holds the configured figure set;
- **the comparison CSV's columns are identical across the three families**, with the deterministic family's ensemble
  scalars `NaN` — the same property the by-hand cross-family gate reads, checked here on synthetic data;
- the run exits 0 and the orchestrator logged one child run per stage.

**What it deliberately does NOT prove** — the accepted cost of the synthetic-only choice:

- the real 101 × 149 grid, the real sample `.pt` layout, the real year split;
- that 8.7 MB × 5843 is tractable;
- anything about GPU execution.

Those stay with the **three by-hand gates** of the verification list above, as in Steps 2 and 3 — the per-family smoke
run on real data, the cross-family comparison over all three, and the hourly smoke. The e2e test proves the pipeline is
*wired*; the smoke run proves it *works on the data*. Both are needed and they are not substitutes.

⚠️ **It will not necessarily move the coverage number.** `run_project.py` dispatches each stage through `mlflow.run` as
a subprocess, and `pytest-cov` does not see into subprocesses without the `COV_CORE_*` `.pth` hook. Either enable
subprocess coverage explicitly, or state plainly that `tuning.py` / `run.py` gain confidence without gaining measured
lines. Do not raise `--cov-fail-under` on the strength of this test until the number is actually re-measured.

⚠️ **The lazy cache must be off or irrelevant.** The smoke tiers already set `lazy: false`; keep it that way in the
derived config, or a second run in one session skips the stages the test means to execute.

---

## ⚠️ Tests are part of the implementation, not a block at the end

Step 3 learned this the expensive way: blocks 1–4 were verified by nine throwaway scratchpad scripts, and blocks
5a–5c then spent three commits turning that into a durable suite — an audit of 641 check sites, plus 287 new tests
written against code that had shipped weeks earlier. Writing the test beside the function costs a fraction of that.

**So in Step 4: every new function gets its test in the SAME commit as the function.** Concretely:

- `tests/completeness_test.py` is a **hard gate** as of block 5c — 291 of 291 functions referenced, `EXEMPT` empty. A
  new stage function with no test **fails the suite**. That is what makes "tests are part of the implementation"
  enforceable rather than aspirational.
- `test_function_census_is_stable` pins the count at **291**. Every stage function added here moves it, so the number
  is updated *in the same commit* — which makes the census diff a visible statement of what the commit added.
- `tests/` mirrors `src/`: a new `src/stages/<name>.py` needs `tests/stages/<name>_test.py`, or
  `test_every_source_module_has_a_test_file` fails. **Three mirrored files already exist and are skipped**, paths
  pre-fixed, waiting for this step: `evaluate_regression_test.py` (→ `evaluate_test.py`), `tabulate_metrics_test.py`,
  `combine_curves_test.py`. Flipping one `pytest.mark.skip` per file is the intended first move, not a rewrite.
  **New files needed:** `prepare_modeling_test.py`, `tune_test.py`, `retrain_best_test.py`, plus the non-mirrored
  `pipeline_e2e_test.py`.
- `pytest.ini` carries `--cov-fail-under=85`. **Re-measure before touching it** (see the subprocess caveat above);
  raise it against a measurement, never against an expectation.
- Write the test that would **catch a plausible break**, not one that merely mentions the function. The gate matches on
  the bare name and counts string literals, so `assert fn(x) is not None` satisfies it — and a test whose only job is
  to satisfy the gate is worse than an acknowledged gap, because it makes the suite look complete while asserting
  nothing. Where a function has no behaviour worth pinning beyond "it runs", say so in the docstring: that is a
  reviewable claim.
- The stages are thin, so most of their tests are **contract** tests: the family dispatch resolves all three names, the
  MC-dropout warm-start `partial` is built only for that family and only with an upstream path, the five dead
  `write_report` arguments are absent, the hourly branch populates `probability`.

## Verification

Per [`rebuild-plan.md`](rebuild-plan.md) §"Verification", in order:

1. **Import/parse** — `python -c "import ..."` on each new stage. They run from inside `src/stages/`, so each must
   begin `from __init__ import root_path` **before** any `src.` import. `parse_config` on every YAML.
2. **`pytest tests/ -q`** — green, coverage floor met, and the skip count drops as the three placeholders flip.
3. **`pipeline_e2e_test.py`** — the synthetic-root run, in the suite. This is what makes "the pipeline is wired"
   re-runnable by anyone, with no dataset.
4. **THE PER-FAMILY SMOKE RUN** — by hand, on real data, once per family (block `4e`):

   ```shell
   export DATA_ROOT=/path/to/era5_postprocess          # metadata.json, metadata.csv, samples/
   export OUTPUT_ROOT=/scratch/$USER/lightning-outputs # everything the pipeline writes
   python run_project.py config/<family>/<family>_daily_smoke_cpu.yaml <EXPERIMENT>
   ```

   It proves the three things a synthetic fixture structurally cannot: the real **101 × 149** grid, the real
   `samples/sample_XXXXXX.pt` layout, and the real year-based split. Green means all five stages ran in sequence and
   each wrote the artifacts the next one reads.
   ⚠️ **Commit before running** — the lazy cache keys on the whole-repo dirty diff, so any uncommitted edit busts every
   entry and the run tells you nothing about the committed state.

5. **THE CROSS-FAMILY COMPARISON** — by hand, and the gate that says whether the merge actually worked (block `4e`).
   Run all three `*_smoke_cpu` pipelines, then `config/eval/probabilistic_eval_smoke_cpu.yaml` on top of them (it
   needs three trained checkpoints, so it is downstream of all three and is the expensive gate, not a quick check).

   The requirement: `tabulate_metrics` emits one CSV whose **metric-key COLUMNS are identical across the three
   families**, and `combine_curves` emits the overlaid figures. Identical *columns*, not identical values — the
   deterministic family's ensemble scalars are `NaN` and it contributes no rank histogram, which is correct.

   ⚠️ **Read the LOG, not just the CSV.** The columns are identical *by construction* — one DataFrame has one column
   set — so the CSV alone cannot fail this gate. What it is really reading is the **NaN pattern**, which is why
   `tabulate_metrics` logs, per family, exactly which metrics it lacks. A deterministic family missing `crps` is
   expected; a family missing `mae` is the merge failure the gate exists to catch.

6. **THE HOURLY SMOKE** — by hand (block `4f`):
   `python run_project.py config/deterministic_unet/deterministic_unet_hourly_smoke_cpu.yaml <EXPERIMENT>`, end to end.
   Its metrics JSON must carry the keys that are **absent** in daily mode — `brier_skill_score`,
   `explained_deviance`, `dice_p50` — plus a `reliability_table.csv` in the report directory. Their presence is the
   proof that `probability` was populated and reached the metric suite; in daily mode their absence is equally correct.
   Four more things this gate is the only place to read:
   * `pod_p50 < 1` and a finite `csi_p50` — i.e. the `kind: probability` cut is on the PREDICTION side. A degenerate
     table here is the `occurrence`-on-a-probability bug, and it does not raise.
   * a non-zero occurrence base rate. The positive (cell, hour) pairs are the same strokes the daily smoke sees, spread
     over 24× the cells, so mid-July gives percent-level — but a base rate of zero makes every categorical score,
     skill score and reliability bin `NaN` while the run still reports success.
   * `fss_s<scale>` keys, NOT `fss_occurrence_s<scale>` — the threshold-free FSS form, switched on by detecting the
     probability field rather than by a config key.
   * ⚠️ `ets_h6` is `NaN` in the trials table (its band is `>= 6 hours` on a 0/1 target, so the contingency table is
     empty) and the hourly maps collapse to two colours. Both expected; see the block-4f section above.
7. **The closing review of `tests/`** (below).
8. **Update this file and `rebuild-plan.md`**; flip Step 4 to ✅.

## Closing review of `tests/` — the last thing Step 4 does

After the gates pass, review the whole suite **as if new to the repo** — deliberately not from inside the history that
built it, because the author of a test is the worst judge of whether it asserts anything. Three questions, answered
with evidence rather than impression:

1. **Are the tests relevant?** Does each one pin a decision this project actually made, or a property of the library it
   happens to call? (Block 5b found two gate checks that tested *matplotlib* and five that were literal
   `check(label, True)`. Both patterns are easy to write and invisible in a green run.)
2. **Are they catching possible bugs?** Spot-check by **mutation**: break a source line, confirm the test that claims
   to cover it fails, revert. ⚠️ Verify the edit landed on an *executed* line — 5b's first gate-4 mutation hit a
   docstring, the code never changed, and the result read as "not caught".
3. **Is the code well covered?** Both measurements, and the disagreement between them — they are not interchangeable
   ([Step 3](step-3-utils.md) §5c). Name what is still uncovered and why, rather than reporting a single percentage.

The output is a written assessment, not a pass/fail: which tests earn their place, which are decorative, and where the
real gaps are.

## Block order

| Block | Contents |
|---|---|
| `4a` ✅ | `prepare_modeling` + the rename surface + the `compute_high_lightning_days` utility move (`hello_world` kept) |
| `4b` ✅ | `tune` + `retrain_best` (one commit — same dispatch, same wrappers) |
| `4c` ✅ | `evaluate` |
| `4c-r` ✅ | *(unplanned)* every output behind `{{$OUTPUT_ROOT}}`, the `setup` unset-root guard, one shared U-net prepared dir |
| `4d` ✅ | `tabulate_metrics` + `combine_curves` + `src/stages/README.md` — **every stage the configs name now exists** |
| `4e` | `tests/pipeline_e2e_test.py` ✅ (18 tests, ~115 s), then **by hand on real data**: (a) each family's `*_smoke_cpu` pipeline end to end, proving the real 101×149 grid / `samples/*.pt` layout / year split; (b) `probabilistic_eval_smoke_cpu.yaml` on top of all three, proving `tabulate_metrics` emits **identical metric-key columns across the families** and `combine_curves` the overlaid figures |
| `4f` | the hourly pipeline — **no `src/` change**: `metrics_hourly.yaml` + `search_space_hourly.yaml` (unplanned, required by `selection_metric_for_mode`) + the two hourly tiers + 23 tests, plus the unplanned daily/hourly naming rename (13 `git mv` + a 60-file sweep). Then **by hand**: the hourly smoke runs and its metrics JSON carries the classification keys absent in daily mode (`brier_skill_score`, `explained_deviance`, `dice_p50`, the reliability table) |
| `4g` | the closing review of `tests/` |

One commit per block, tests included, stopping after each to report — the pattern Steps 2 and 3 used.

⚠️ **`4e` before `4f` deliberately.** The e2e test is what tells us the daily pipeline is wired before the hourly task
adds a second target derivation, a second metrics config and 24× the items on top of it. Debugging both at once is the
avoidable version of this step going wrong.

## Carried into this step from earlier steps

- **Tuning unified into one stage** (annotated decision) — done in substance by Step 3: D's `_fit_phase` is already in
  `run_sweep`, so 4b is the CLI wrapper only.
- **`registry.py` and `mc_dropout_eval.py`: "modify — unify"** so MC-dropout and diffusion share one ensemble path —
  done in Step 3 blocks 3d/3e; `evaluate` consumes it.
- **`combine_curves`: "modify"** so its plotting follows the 02a convention — block 4d.
- **`compute_high_lightning_days`: keep** — as a utility, per decision 3.
- **`hello_world`:** checked; **KEPT** — decision 4 was reversed once the usefulness check it mandated actually ran (it
  has two consumers: `README.md` and `config/hello_world.yaml`). This line said "deleted" until 4d; it was the one place
  the plan still contradicted the tree.
- The pipeline YAML shapes are already drafted in [Step 2](step-2-config.md) §5 and **shipped** — the twelve configs
  are the contract these stages must satisfy, not a draft to revise.
- **Per-folder `README.md`** for `src/stages/`, capturing the agreed contracts (deferred from Step 1; write it as the
  folder is finished, in 4d).
