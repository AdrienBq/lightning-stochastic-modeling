# Step 4 (block d) — Stages + common evaluation

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 3](step-3-utils.md) · Next: [Step 5](step-5-portability.md)

> **Status: provisional — to be expanded** once [Step 3](step-3-utils.md) is done.

`setup` → merged `prepare_regression` → shared-harness `tune` → `retrain_best` → **`evaluate_regression`
(common)** → `tabulate_metrics` → `combine_curves`; wire the per-family + cross-model pipelines; port the CPU
unit tests.

## Carried into this step

- **Tuning is unified into one stage** (annotated decision): the per-family `tune_distr_regression` /
  `tune_diffusion` / `tune_mc_dropout` stages collapse into a single `tune` stage taking `model-family`.
  adrien's `_fit_phase` (the two-phase train→finetune fit) lifts into aru's `run_sweep`; both duplicated
  `_fit_trial`s are then deletable.
- **`registry.py` and `mc_dropout_eval.py`: "modify — unify"** so MC-dropout and diffusion share one ensemble
  path.
- **`combine_curves`: "modify"** so its plotting follows the 02a convention.
- **`compute_high_lightning_days`: keep** — "this is the extremes".
- **`hello_world`:** check whether it is still useful; remove if not.
- The pipeline YAML shapes are already drafted in [Step 2](step-2-config.md) §5.
