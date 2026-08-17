# `src/stages/` — the pipeline stages

Every stage is a **standalone CLI script** wrapped with `fire`, invoked by the orchestrator as a subprocess:

```shell
python src/stages/<name>.py --param-name value ...
```

Nothing here is imported as a library. `src/utils/` holds the code; a stage is the thin argument-forwarding layer
between a YAML block and that code. If a stage grows logic worth testing on its own, that logic belongs in
`src/utils/`.

This file documents the **cross-stage contracts** — the things no single module's docstring can state. For what a
given stage does, read its own docstring; for the project's modelling scope, read [`CLAUDE.md`](../../CLAUDE.md).

---

## The rule that breaks everything if forgotten

Every stage must begin with

```python
from __init__ import root_path
```

**before any `src.` import.** A stage runs with `src/stages/` as its script directory, so the repo root is not on
`sys.path` until `src/stages/__init__.py` puts it there — and that import is what does it. Put a `src.` import first
and the stage dies with `ModuleNotFoundError: No module named 'src'`, at import time, in a subprocess, with a
traceback that points at the wrong line.

That same `__init__.py` also **applies `PIPELINE_SEED`**, exported by the orchestrator. Seeding is therefore
automatic: do not re-seed globally inside a stage.

## The stages, and what flows between them

| Stage | Reads | Writes |
|---|---|---|
| `setup` | the output paths themselves | the output tree (and refuses an unset `OUTPUT_ROOT`) |
| `prepare_modeling` | `$DATA_ROOT/{metadata.json,metadata.csv,samples/}` | `targets/`, `features/`, `split_index.csv`, `target_stats.json`, `prepared_config.json` (+ `upstream/` in residual mode) |
| `tune` | a prepared directory | `trials.csv`, `best_trial.json`, `best_model.ckpt`, `best_trial_metrics.json` |
| `retrain_best` | a prepared directory + the sweep's `best_trial.json` | the same four, for the winning configuration |
| `evaluate` | a prepared directory + `best_model.ckpt` | `test_metrics.json`, `predictions.npz`, a report directory of figures + CSVs |
| `tabulate_metrics` | N families' `test_metrics.json` | one families × metrics comparison CSV |
| `combine_curves` | N families' report directories | the overlaid comparison figures (png + pdf) |
| `hello_world` | — | nothing; the template's smoke target for `run_project.py` itself |

`evaluate` is **the** evaluation, for every family and both tasks. `registry.load_model_module` dispatches on the
checkpoint's family marker, so there is no family-specific evaluation path and adding one would defeat the merge —
two families' numbers are comparable only because one code path produced them.

The last two stages are the **comparison layer**, and they run in their own pipeline
(`config/eval/probabilistic_eval.yaml`) after the per-family ones. They recompute nothing: `tabulate_metrics` reads
the metrics JSONs and `combine_curves` the report CSVs, so a comparison costs seconds and can be redrawn without
re-running an evaluation.

## Parameters

**Names are a contract with the lazy cache.** `OUTPUT_PARAM_KEYS` — `output-path`, `metrics-path`, `report-path` — are
treated as a stage's *outputs*; **any other parameter resolving to an existing path is an input**, and its contents
are fingerprinted into the cache key. Two consequences:

* use those three names for anything the stage writes, or the cache will fingerprint your output as an input and
  invalidate the stage against itself;
* a **stale** path silently degrades to a plain scalar, so the cache stops invalidating on that input. Keep
  `metrics-config` / `split-config` / `model-config` pointing at files that exist.

**Two roots, both `{{$VAR}}`, both required.** `DATA_ROOT` is the read-only dataset; `OUTPUT_ROOT` is everything the
pipelines write (~20 GiB per prepared directory, ~60 GiB for three families — which is why it is not in the
checkout). Never hardcode either, in code or in config.

⚠️ **`parse_config` substitutes textually before the YAML parse, and an unset variable becomes the EMPTY STRING**, not
an error. So `'{{$OUTPUT_ROOT}}/mc_dropout/prepared'` becomes `/mc_dropout/prepared` — absolute, at the filesystem
root, and `os.path.join(root_path, …)` then discards the repo root entirely. `setup` catches that before anything
expensive and names the variable. The empty-string behaviour is **load-bearing** for `UPSTREAM_MODEL`, where unset
legitimately means "no warm start", which is why the guard lives in `setup` and not in `parse_config`.

## Two things that are not symmetric, and look like they should be

**`UPSTREAM_MODEL` is read by two different stages, for two different things.**

| Family | Stage | What it wants |
|---|---|---|
| `mc_dropout` | `tune` | the upstream's **weights** — a warm start, so only the finetuning phase runs |
| `diffusion` | `prepare_modeling` | the upstream's **predictions** — materialised as `upstream/<date>.npy` and appended as the last conditioning channel |

Forwarding it to the wrong stage is not an error you will see: a diffusion `tune` given an upstream would be
constrained by a rule written for MC-dropout, and an MC-dropout `prepare_modeling` given one would build a 6th
channel its checkpoints have no input for.

**The two U-net families share one prepared directory** (`$OUTPUT_ROOT/deterministic_and_mc_dropout/prepared`) —
their `prepare_modeling` blocks are identical, so whichever pipeline runs first prepares it and the second skips.
Keep those blocks in step; `parse_config_test.py` enforces the equality. **Diffusion keeps its own**, and in
full-target mode its directory *would* be identical — which is what makes sharing a trap rather than an
optimisation: it would work until the first residual run flipped `residual_target`.

## Adding a stage

1. The script, `from __init__ import root_path` first, `Fire(<function>)` last.
2. A mirrored `tests/stages/<name>_test.py` — `tests/completeness_test.py` fails on a module without one, and on a
   test file without a module.
3. Move `test_function_census_is_stable`'s count in the **same commit**, so the diff states what was added.
4. `python run_project.py config/<family>/<pipeline>.yaml <EXPERIMENT>` to run it. **Commit first** — the lazy cache
   keys on the whole-repo dirty diff, so any uncommitted edit busts every entry.
