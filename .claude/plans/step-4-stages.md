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
| `tabulate_metrics` | A | ~116 | near as-is |
| `combine_curves` | A | ~260 | `_combined_qq` deleted, reliability + ROC/PR overlays added |

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

### `tabulate_metrics` — A (116), near as-is

`**kwargs` model→JSON mapping, `_display` restoring `Diffusion-Model` from Fire's `Diffusion_Model`. The substantive
requirement is the phase gate's: **identical metric-key columns across families**, with the deterministic family's
ensemble scalars `NaN`. Verify `_DISPLAY_NAMES` covers the three shipped labels (`Deterministic-UNet`, `MC-Dropout`,
`Diffusion`).

### `combine_curves` — A (252), 02a restyle

- ❌ **`_combined_qq` goes.** It reads `qq_table.csv`, which no longer exists — the target-space `qq_plot` was removed
  with the 02a grammar and `reporting.py` never writes that file. Left in, it would log *"no model had a usable
  qq_table.csv; skipped"* on every run forever.
- ➕ **`_combined_reliability`** from `reliability_table.csv` and **`_combined_roc_pr`** from `roc_pr_summary.csv` —
  the classification-first headline diagnostics, and the two figures a cross-family comparison most wants.
- ✅ Keep `_combined_psd` (with its ±1σ ensemble band), `_combined_fss`, `_combined_rank_histogram`, `_model_colors`
  (deterministic over sorted names), and `_read_csv`'s never-raise contract.
- 🔧 The PSD x-axis in **kilometres**, inverted, matching `reporting._psd_curves` — A plots `wavelength_px`; the CSV
  carries both columns.

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

## The hourly pipeline

The stage code above makes hourly *derivable*; these make it *runnable*.

- ➕ **`config/eval/metrics_hourly.yaml`** — a sibling of `metrics.yaml`, whose own prose (L37-64) already specifies
  what must change. The load-bearing edit is `metrics.categorical.thresholds`: entries must be `kind: probability`
  (`p50: {kind: probability, value: 0.5}`), **not `occurrence`**. An `occurrence` entry resolves to `> 0` on a
  probability field, so it fires on every cell with any non-zero probability — POD ≈ 1, FAR ≈ base rate, a full
  contingency table of nonsense with no error raised. `run_metric_suite` warns at that configuration; the config must
  not reach it.
  Also drop `mae_stratified`, `estimation_tendency` and `rank_correlation` (block 2r2 §1: their bins or conditions
  degenerate on a binary observation).
- ➕ **`config/deterministic_unet/deterministic_unet_hourly.yaml`** + its `*_smoke_cpu` tier. Deterministic family
  only — it is the upstream for the other two, and the point is to prove the classification path runs, not to sweep it.
- ⚠️ **24× the items.** The 8-day CPU smoke split becomes 192 hourly items; `feature-aggregation` is ignored in hourly
  mode (an item is already one hour) and **`DayGroupedShuffleSampler` becomes live for the first time** — it is dormant
  today because every config sets `materialize-features: true`.
- ⚠️ **`ensemble-size` ≥ 2** as always: `spread_skill_sums` uses `ddof=1`, and a single member yields a silent `NaN`.

## The rename surface

`prepare_regression` → `prepare_modeling`, `evaluate_regression` → `evaluate`. Mechanical but wide — 36 files. The
functional ones:

| Where | What |
|---|---|
| 9 × `config/{family}/{family}{,_smoke_cpu,_smoke_gpu}.yaml` | the two stage keys |
| 2 × `config/eval/probabilistic_eval{,_smoke_cpu}.yaml` | `evaluate_regression:` × 3 each |
| `tests/stages/evaluate_regression_test.py` | → `evaluate_test.py` |
| `tests/completeness_test.py` | the hardcoded Step-4 set (L112-114) and the census (L170) |

Prose-only (docstrings and comments naming the stage): `CLAUDE.md`, `README.md`, `config/eval/metrics.yaml`,
`config/split/split*.yaml`, the three `search_space.yaml`, `src/utils/io/data.py`, `registry.py`, `search.py`,
`mc_dropout_module.py`, `mc_dropout_eval.py`, and 4 test files. Update them — a comment naming a stage that does not
exist is how the next reader loses an hour.

---

## End-to-end verification

A smoke run (`python run_project.py config/<family>/<family>_smoke_cpu.yaml <EXPERIMENT>`) has exactly the right
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

### `tests/pipeline_e2e_test.py` — synthetic root, always runs

Lives at the `tests/` root, like `completeness_test.py`, because it mirrors no single module — it exercises all of
them. ⚠️ **Add it to `completeness_test.py`'s expected set (L110-111)** or
`test_no_test_file_outlives_its_module` flags it as an orphan.

Two halves, both deliberate:

1. **A `$DATA_ROOT`-shaped fixture** — `metadata.json`, `metadata.csv`, `samples/sample_XXXXXX.pt` on a small grid.
   `data_test.py`'s `sample_directory` fixture already builds exactly this shape and is the thing to lift into
   `conftest.py`.
2. **A config DERIVED from the shipped smoke YAML**, not a fixture copy: parse
   `config/<family>/<family>_smoke_cpu.yaml`, rewrite only `data-path`, `split-config` and the output paths into
   `tmp_path`, dump it, invoke `run_project.py` on that. Derived means it **cannot drift** from the real pipeline —
   the same principle as `test_a_REAL_tune_stage_has_its_config_parameters_classified_as_inputs`, which reads the
   shipped `deterministic_unet.yaml` rather than a hand-built dict.

What it asserts, in code:

- every declared `output-path` / `metrics-path` / `report-path` exists and is non-empty;
- the metrics JSON's keys ⊇ `EXPECTED_DAILY_KEYS` — the 34 keys `evaluation_test.py` already pins, so the unit suite
  and the pipeline suite cannot disagree about what the evaluation emits;
- the report directory holds the configured figure set;
- **the comparison CSV's columns are identical across the three families** (gate 5's real content), with the
  deterministic family's ensemble scalars `NaN`;
- the run exits 0 and the orchestrator logged one child run per stage.

**What it deliberately does NOT prove** — the accepted cost of the synthetic-only choice:

- the real 101 × 149 grid, the real sample `.pt` layout, the real year split;
- that 8.7 MB × 5843 is tractable;
- anything about GPU execution.

Those stay **gates 4–6, run by hand**, as in Steps 2 and 3. The e2e test proves the pipeline is *wired*; the smoke run
proves it *works on the data*. Both are needed and they are not substitutes.

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
4. **Smoke run, per family** (by hand, real data): `python run_project.py config/<family>/<family>_smoke_cpu.yaml
   <EXPERIMENT>` with `DATA_ROOT` exported. Proves what the synthetic test cannot — the real grid, the real sample
   layout, the real split.
   ⚠️ **Commit before running** — the lazy cache keys on the whole-repo dirty diff.
5. **Phase gate** (by hand): all three `*_smoke_cpu` pipelines plus `config/eval/probabilistic_eval_smoke_cpu.yaml`
   end to end. `tabulate_metrics` must emit one CSV with **identical metric-key columns across families**, and
   `combine_curves` the overlaid figures. Identical *columns* is the requirement, not identical values — the
   deterministic family's ensemble scalars are `NaN` and it contributes no rank histogram.
6. **The hourly smoke** runs, and its metrics JSON carries the classification keys (`brier_skill_score`,
   `explained_deviance`, `dice_occurrence`, `average_precision_occurrence`) that are absent in daily mode.
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
| `4a` | `prepare_modeling` + the rename surface + the `compute_high_lightning_days` utility move + delete `hello_world` |
| `4b` | `tune` + `retrain_best` (one commit — same dispatch, same wrappers) |
| `4c` | `evaluate` |
| `4d` | `tabulate_metrics` + `combine_curves` |
| `4e` | `tests/pipeline_e2e_test.py`, then gates 4 + 5 by hand |
| `4f` | the hourly pipeline (`metrics_hourly.yaml` + the hourly YAML + smoke tier), then gate 6 |
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
- **`hello_world`:** checked; deleted, per decision 4.
- The pipeline YAML shapes are already drafted in [Step 2](step-2-config.md) §5 and **shipped** — the twelve configs
  are the contract these stages must satisfy, not a draft to revise.
- **Per-folder `README.md`** for `src/stages/`, capturing the agreed contracts (deferred from Step 1; write it as the
  folder is finished, in 4d).
