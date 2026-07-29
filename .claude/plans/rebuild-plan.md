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
| [`step-3-utils.md`](step-3-utils.md) | Shared `src/utils` | 🔵 **next** — see its "obliged by Step 2" list |
| [`step-4-stages.md`](step-4-stages.md) | Stages + common evaluation | ⚪ provisional |
| [`step-5-portability.md`](step-5-portability.md) | User- and machine-agnostic | ⚪ provisional |

### Design decision record (the Step 1 deliverable — all annotated)

| File | Contents |
|---|---|
| [`inventory-losses.md`](inventory-losses.md) | 24 losses, 7 categories, 3 sources |
| [`inventory-scores.md`](inventory-scores.md) | 40 scores, 8 categories |
| [`inventory-figures.md`](inventory-figures.md) | The 02a visual spec, 13 pipeline figures, issues |
| [`inventory-architecture.md`](inventory-architecture.md) | Stages, modeling layer, ensemble contract, portability, 15-item transform-removal checklist |

**Block order** a → b → c → d → e. Each step ends with the verification gate below and a plan update.

> **▶ To resume work:** read the OUTCOME block at the top of [`step-2-config.md`](step-2-config.md) — it records
> what Step 2 built, the seven corrections it made to its own spec, and the list Step 3 is now obliged to implement.

---

## Key merge tasks & risks (carried through Steps 3–4)

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
2. Unit tests: `pytest` on the relevant ported tests.
3. Smoke run: the affected `*_smoke_cpu.yaml` via
   `python run_project.py config/<family>/<family>_smoke_cpu.yaml <EXPERIMENT>` (CPU, `n-trials 1`,
   `max-epochs 1`, the 8-day mid-July split, **`ensemble-size 2`** — never 1, see below); assert declared
   artifacts + expected metric keys. A `*_smoke_gpu.yaml` tier exists for GPU hosts (18 days, 2 trials,
   `bf16-mixed`, `ensemble-size 5`) but cannot be run here.
4. Phase/final gate: run all three family `*_smoke_cpu` pipelines + `config/eval/probabilistic_eval_smoke_cpu.yaml`
   end-to-end; assert `tabulate_metrics` emits one comparison CSV with **identical metric-key columns across
   families** and `combine_curves` emits the overlaid figures — proof the pipelines are merged and report the same
   metrics. (The deterministic family's ensemble scalars are `NaN` and it contributes no rank histogram: identical
   *columns* is the requirement, not identical values.)
5. **After each step: update the relevant step file** (decisions made, next step refined).

> ⚠️ **`ensemble-size` must be ≥ 2 in smoke configs.** `scores.spread_skill_sums` computes variance with `ddof=1`,
> so a single-member ensemble divides by zero and yields a silent `NaN` rather than an error.
