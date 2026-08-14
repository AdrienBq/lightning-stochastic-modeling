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

## ⚠️ Tests are part of the implementation, not a block at the end

Step 3 learned this the expensive way: blocks 1–4 were verified by nine throwaway scratchpad scripts, and blocks
5a–5c then spent three commits turning that into a durable suite — an audit of 641 check sites, plus 287 new tests
written against code that had shipped weeks earlier. Writing the test beside the function costs a fraction of that.

**So in Step 4: every new function gets its test in the SAME commit as the function.** Concretely:

- `tests/completeness_test.py` is a **hard gate** as of Step 3 block 5c — 291 of 291 functions referenced, `EXEMPT`
  empty. A new stage function with no test **fails the suite**. That is deliberate: the gate is what makes "tests are
  part of the implementation" enforceable rather than aspirational.
- `test_function_census_is_stable` pins the count at **291**. Every stage function added here moves it, so the number
  is updated *in the same commit* — which makes the census diff a visible statement of what the commit added.
- `tests/` mirrors `src/`: a new `src/stages/<name>.py` needs `tests/stages/<name>_test.py`, or
  `test_every_source_module_has_a_test_file` fails. **Three mirrored files already exist and are skipped**, with
  their paths pre-fixed, waiting for this step: `evaluate_regression_test.py`, `tabulate_metrics_test.py`,
  `combine_curves_test.py`. Flipping one `pytest.mark.skip` per file is the intended first move, not a rewrite.
- `pytest.ini` carries `--cov-fail-under=85`. Step 4 should **raise** it: a real `*_smoke_cpu.yaml` run is what
  finally exercises `tuning.py` (25 %) and `stages/run.py` (57 %), the two files holding 413 of the 549 uncovered
  statements. If the number goes up, the floor goes up with it.
- Write the test that would **catch a plausible break**, not one that merely mentions the function. The gate matches
  on the bare name and counts string literals, so `assert fn(x) is not None` satisfies it — and a test whose only job
  is to satisfy the gate is worse than an acknowledged gap, because it makes the suite look complete while asserting
  nothing. Where a function has no behaviour worth pinning beyond "it runs", say so in the docstring: that is a
  reviewable claim.

## Closing review of `tests/` — the last thing Step 4 does

After the end-to-end gate passes, review the whole suite **as if new to the repo** — deliberately not from inside the
history that built it, because the author of a test is the worst judge of whether it asserts anything. Three
questions, answered with evidence rather than impression:

1. **Are the tests relevant?** Does each one pin a decision this project actually made, or a property of the library
   it happens to call? (Step 5b found two gate checks that tested *matplotlib* and five that were literal
   `check(label, True)`. Both patterns are easy to write and invisible in a green run.)
2. **Are they catching possible bugs?** Spot-check by **mutation**: break a source line, confirm the test that claims
   to cover it fails, revert. ⚠️ Verify the edit landed on an *executed* line — 5b's first gate-4 mutation hit a
   docstring, the code never changed, and the result read as "not caught".
3. **Is the code well covered?** Both measurements, and the disagreement between them — they are not
   interchangeable (Step 3 §5c). Name what is still uncovered and why, rather than reporting a single percentage.

The output is a written assessment, not a pass/fail: which tests earn their place, which are decorative, and where
the real gaps are.
