# Rebuild plan: clean, merged `lightning-stochastic-modeling` — index

Split into per-step files (2026-07-28). This file is the **index** plus the two cross-cutting sections that belong
to no single step.

## Files

| File | Contents | Status |
|---|---|---|
| [`00-context.md`](00-context.md) | Project framing, branch findings, locked-in decisions incl. the **scope change**, environment & data, target architecture, the ensemble contract | reference |
| [`step-0-bootstrap.md`](step-0-bootstrap.md) | Bootstrap & hygiene | ✅ **done** (2026-07-27) |
| [`step-1-design.md`](step-1-design.md) | Design decisions via the inventories; global `CLAUDE.md` + `README.md` | ✅ **done** (2026-07-28) |
| [`step-2-config.md`](step-2-config.md) | Every config file and its contents | ✅ **done** (2026-07-29) |
| [`step-3-utils.md`](step-3-utils.md) | Shared `src/utils` | ✅ **done** (2026-08-14) |
| [`step-4-stages.md`](step-4-stages.md) | Stages + common evaluation — **expanded 2026-08-14**, blocks `4a`–`4g` | 🔵 **in progress** — `4a`–`4f` done; every stage exists, all three families ran on real data and reported one 129-metric comparison table, and the **hourly** task is now runnable. Next: `4g`, the fresh-eyes review of `tests/` |
| [`step-5-portability.md`](step-5-portability.md) | User- and machine-agnostic | ⚪ provisional |

### Design decision record (the Step 1 deliverable — all annotated)

| File | Contents |
|---|---|
| [`inventory-losses.md`](inventory-losses.md) | 24 losses, 7 categories, 3 sources |
| [`inventory-scores.md`](inventory-scores.md) | 40 scores, 8 categories |
| [`inventory-figures.md`](inventory-figures.md) | The 02a visual spec, 13 pipeline figures, issues |
| [`inventory-architecture.md`](inventory-architecture.md) | Stages, modeling layer, ensemble contract, portability, 15-item transform-removal checklist |
| [`inventory-gate-migration.md`](inventory-gate-migration.md) | All 641 checks of the nine deleted `gate_block*.py`, and where each one went (Step 3 block 5b) |

**Block order** a → b → c → d → e. Each step ends with the verification gate below and a plan update.

> **▶ To resume work:** Step 4 is at block **`4g`**, the last one. Done: all seven stages (`4a`–`4d`), every written
> path behind `{{$OUTPUT_ROOT}}` (`4c-r`), the synthetic-root e2e test and the three by-hand real-data gates (`4e`), and
> the hourly pipeline (`4f`). `tests/` is the verification of record — 1378 passing, 329/329 functions tested,
> ~88 % line coverage.
>
> What `4e` and `4f` settled, so it is not re-litigated:
>
> * **The merge works.** All three families ran their `*_smoke_cpu` pipeline on the real dataset and
>   `probabilistic_eval_smoke_cpu.yaml` produced **one CSV, 3 families × 129 metrics**. Every point / categorical /
>   skill / spatial metric is present for every family; the only gaps are capability-explained (6 ensemble scalars
>   absent for the deterministic family, 33 `resid_*` present only for the residual diffusion run) plus one
>   data-dependent `fss_useful_scale_h3`.
> * **Six bugs came out of `4e`, five of them visible only in real output** — three found by a human reading a figure.
>   Cartopy's non-finite gridliner extents were behind two of them (cropped saves, invisible titles).
> * **`4f` needed no `src/` change**: every layer was already mode-aware, so the hourly task was four configs and 23
>   tests. It did need an unplanned `search_space_hourly.yaml` (`selection_metric_for_mode` raises otherwise) and
>   brought the daily/hourly **config naming rename** with it — every task-specific file now names its task.
>
> ⚠️ Standing rules for `4g` and beyond: **a new function's test lands in the same commit as the function**, and
> **commit before running a pipeline** — the lazy cache keys on the whole-repo dirty diff. Read
> [`step-4-stages.md`](step-4-stages.md) §Verification for all eight checks and §"Closing review of `tests/`" for what
> `4g` actually asks for (an assessment with evidence, not a pass/fail).

---

## Key merge tasks & risks (carried through Steps 3–4)

> ⚠️ **Two stage renames land in Step 4 block 4a**, so every reference below and in the other plan files is due an
> update at that point: `prepare_regression` → **`prepare_modeling`** and `evaluate_regression` → **`evaluate`**. The
> single shared prepare stage keeps its `mode:` key; the split into `prepare_regression`/`prepare_classification`
> proposed in [`step-3-utils.md`](step-3-utils.md) §3 is **superseded** (struck there, with the reason).

1. **MC-dropout de-duplication:** drop aru's stale vendored `modeling/mc_dropout/` package **and** adrien's
   `unet_aru.py` / `distr_regression_aru.py`; use adrien's current `mc_dropout_module.py` on the shared
   `unet`/`losses`/`scores`. *(Both stale sets confirmed by identical function order with newer additions missing.)*
2. **Unified `losses.py`** — superset of both branches, pruned to the annotated inventory. Merge direction is
   **D → shared** for losses (adrien is a strict superset), the *opposite* of the scores file.
3. **Single evaluation path:** aru's `run_metric_suite` + `finalize_ensemble_metrics` + streaming
   `ensemble_partials`; retire adrien's `EnsembleProbabilisticAccumulator` and `regression_metric_suite`;
   MC validation uses the shared `selection_score`.
4. **Two-phase `run_sweep`:** generalize adrien's `_fit_phase` train→finetune fit into the shared harness without
   regressing single-phase families (verify monitor / best-weight restore parity). **Plus, per Step 2:** a
   *warm-start* path where `tune` is handed an `upstream-model-path` and runs the finetuning phase **alone**,
   loading the upstream U-net's weights. That path does not exist on any branch — `MCDropoutRegressionModule` has
   `set_phase()` and a `finetuning_enabled` gate but nothing that loads foreign weights — and it needs a load-time
   architecture check that *raises* rather than silently partial-loading.
5. **Merge `dataset.py`** (`hourly_stack` aggregation + residual/upstream channel) and **`prepare_regression`**
   (full-target / residual / `daily_lightning_hours` target).
6. **Selection-score unification** — one name, `valid_regression_score` (retires `valid_tail_score`), and per Step 2
   **one source of truth**: the search space's `selection:` block. `tune` reads it from `model-config` and records it
   into `best_trial.json`; `retrain_best` reads it back. The `selection-metric`/`selection-mode` stage parameters are
   gone, so a retrain can no longer disagree with the sweep that chose the configuration.
7. **Registry markers + `_sniff_family`** so legacy checkpoints still load.
8. **Drop dead/superseded code:** `mc_dropout_module_deprecated.py`, adrien's inference ports, `hello_world`
   (pending a usefulness check). **`compute_high_lightning_days` is kept** — it is "the extremes".
9. ⚠️ **The `crps_ensemble` name collision** — `float` (aru) vs `np.ndarray` (adrien). Resolved in favour of aru's
   contract; **add a test pinning the return type**, since merging by name would fail silently rather than loudly.

## Verification (per block; real CPU smoke, 1 epoch, 2 days)

1. Import/parse: `python -c "import ..."` for touched modules; `parse_config` on every YAML.
2. **Unit tests: `pytest` — `tests/` is the verification of record.** It mirrors `src/` (one `<module>_test.py` per
   module) and `tests/completeness_test.py` enforces both the mirror and the every-function requirement. The nine
   `gate_block*.py` scripts that verified Steps 3's blocks 1–4 were **deleted in block 5b**, after each of their 641
   check sites was migrated or dropped with a stated reason — the record is
   [`inventory-gate-migration.md`](inventory-gate-migration.md).

   **Two gates, measuring different things — neither is a superset of the other:**

   * `test_every_source_function_is_referenced_by_a_test` — **hard since block 5c**, 331 of 331, `EXEMPT` empty. It
     catches "never exercised". A new function with no test fails the suite; `test_function_census_is_stable` pins
     the count, so adding one is a visible two-line edit.
   * `--cov-fail-under=85` in `pytest.ini` — line coverage, currently **87.6 %**. It catches "exercised, but its
     branches never run", which the name gate structurally cannot. Raise the floor whenever a step lifts the number;
     the residue today is `tuning.py` (25 %) and `stages/run.py` (57 %), and both need a **real pipeline run** —
     `run_project.py` dispatches every stage as an `mlflow.run` subprocess, which `pytest-cov` does not see into
     without the `COV_CORE_*` hook, so those lines are reachable only by the by-hand smoke gates of Step 4.

   > ⚠️ **At the END of the rebuild, delete the `source_invariant` tests as a group.** They assert on SOURCE TEXT
   > rather than behaviour — that an identifier removed by the three-branch merge stayed removed, that a function
   > delegates rather than re-implements. They are merge guards and have no value once the merge is history.
   > `pytest -m 'not source_invariant'` shows what the suite proves without them; `grep -rn source_invariant tests/`
   > finds every one. Retire the whole set, not a file at a time.
   > ➕ Retiring them will DROP line coverage, since a tokenize sweep still executes the module it parses. Re-measure
   > and re-set the floor at that point rather than treating the fall as a regression.
3. Smoke run: the affected `*_smoke_cpu.yaml` via
   `python run_project.py config/<family>/<family>_daily_smoke_cpu.yaml <EXPERIMENT>` (CPU, `n-trials 1`,
   `max-epochs 1`, the 8-day mid-July split, **`ensemble-size 2`** — never 1, see below); assert declared
   artifacts + expected metric keys. A `*_smoke_gpu.yaml` tier exists for GPU hosts (18 days, 2 trials,
   `bf16-mixed`, `ensemble-size 5`) but cannot be run here.
4. Phase/final gate: run all three family `*_smoke_cpu` pipelines + `config/eval/probabilistic_eval_smoke_cpu.yaml`
   end-to-end; assert `tabulate_metrics` emits one comparison CSV with **identical metric-key columns across
   families** and `combine_curves` emits the overlaid figures — proof the pipelines are merged and report the same
   metrics. (The deterministic family's ensemble scalars are `NaN` and it contributes no rank histogram: identical
   *columns* is the requirement, not identical values.)
5. **After each step: update the relevant step file** (decisions made, next step refined).
6. **A new function's test lands in the SAME commit as the function.** Step 3 verified blocks 1–4 with throwaway
   scratchpad scripts and then spent three commits (5a–5c) turning that into a durable suite — an audit of 641 check
   sites plus 287 tests written against code that had shipped weeks earlier. The hard gate in §2 is what makes this
   enforceable from Step 4 onward rather than a good intention. See
   [`step-4-stages.md`](step-4-stages.md) for the concrete rules and for the **closing review of `tests/`** that each
   remaining step ends with: read the suite as if new to the repo and answer, with evidence, whether the tests are
   relevant, whether they would catch a real bug (spot-check by mutation), and where the coverage genuinely is.

> ⚠️ **`ensemble-size` must be ≥ 2 in smoke configs.** `scores.spread_skill_sums` computes variance with `ddof=1`,
> so a single-member ensemble divides by zero and yields a silent `NaN` rather than an error.
