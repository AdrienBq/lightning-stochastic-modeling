# Step 5 (block e) — Portability: user-agnostic & machine-agnostic

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 4](step-4-stages.md)

> **Status: provisional — to be expanded** once [Step 4](step-4-stages.md) is done.

This repo is meant to be used on other machines or remote servers and by other users, so the paths baked in
during Step 0 (`/homedata/aburq/.venvs/lightning-stochastic-modeling`, the `/homedata/aburq/batta_torch` data
root) will not be applicable to other contexts — this is the final step, once the pipelines themselves are
merged and stable. It adds a local, per-user config file where each user specifies their own paths (data root,
venv/output locations, etc.), so nothing user- or machine-specific is hardcoded in the pipeline YAMLs or code.
It also broadens installation beyond pip: `minimal_requirements.txt` currently assumes a plain `venv` + `pip`
workflow; this step adds equivalent install paths for `uv` and `conda` so the repo isn't tied to one tool.

**Build on the existing mechanism, don't invent one** (Step 1 finding #4): the configs already read a
**`DATA_ROOT`** environment variable, set by one launch script per machine — proven across three environments
(HPC + conda, Jean Zay + venv, local + venv). Formalise and document that, and let the per-user config file
supply `DATA_ROOT` plus the venv/output locations. Two concrete carry-overs:

- the launch scripts' commented-out `LD_LIBRARY_PATH="${CONDA_PREFIX}/lib..."` workaround (node `libstdc++`
  older than the compiled extensions need — `GLIBCXX_3.4.29`, gcc ≥ 11) is a known wart worth documenting;
- **cartopy downloads ~14 MB of Natural Earth shapefiles at *plot* time** into `~/.local/share/cartopy`. That is
  a network call from wherever plotting runs, so it fails on an offline compute node; the cache path is also
  user-specific. Needs a documented pre-warm step. *(Now unavoidable: cartopy is a hard requirement — the
  lazy-import fallback was dropped in Step 1.)*

## TODO — move ALL outputs off the source tree, behind `{{$OUTPUT_ROOT}}`

⭐ **Decided 2026-08-14 (user).** Every config currently writes to `outputs/…`, i.e. inside the git checkout. On a
remote cluster the source tree and the bulk storage are deliberately separate filesystems with very different quotas,
so this does not survive a real run. **One variable, `OUTPUT_ROOT`, replaces the literal `outputs/` prefix
everywhere — smoke tiers included**, so there is a single rule rather than a rule with an exception.

The scale is the argument. Measured against the real grid (101 × 149, 5843 days, 5 predictors × 24 h, float16
features):

| | per day | per split |
|---|---|---|
| `features/` | 3.44 MiB | **19.7 GiB** |
| `targets/` daily (float32 0–24) | 58.8 KiB | 0.33 GiB |
| `targets/` hourly (uint8 0/1) | 352.7 KiB | 1.97 GiB |

**One daily prepared directory is ~20 GiB and the three families are ~60 GiB** — each family keeps its own copy
deliberately, so the pipelines can run and be cached independently. An hourly tier adds ~21.6 GiB per family. And
`prepared/` is only the largest part: `tuning/` holds a checkpoint per retained trial plus the optuna journal,
`best/` another checkpoint, `evaluation/` the `predictions.npz` stacks, and `reports/` the png+pdf pair per rendered
day. All of it grows with every experiment, and none of it belongs in a checkout.

**Use the mechanism that already exists** (Step 1 finding #4, and the rule CLAUDE.md states for `DATA_ROOT`: *never
hardcode a data path*) — the same `{{$VAR}}` substitution, read from the per-user config file this step introduces:

```yaml
- setup:
    outputs:    '{{$OUTPUT_ROOT}}'
    prepared:   '{{$OUTPUT_ROOT}}/deterministic_unet/prepared'
    tuning:     '{{$OUTPUT_ROOT}}/deterministic_unet/tuning'
    best:       '{{$OUTPUT_ROOT}}/deterministic_unet/best'
    evaluation: '{{$OUTPUT_ROOT}}/deterministic_unet/evaluation'
    reports:    '{{$OUTPUT_ROOT}}/deterministic_unet/reports'
- prepare_modeling:
    output-path: '{{$OUTPUT_ROOT}}/deterministic_unet/prepared/daily'
# …and every input-path / model-path / metrics-path / report-path that points into the tree
```

Five things to get right when doing it:

1. **⚠️ `{{$VAR}}` substitutes to the EMPTY STRING when unset**, so an unset `OUTPUT_ROOT` silently yields
   `/deterministic_unet/prepared/daily` — an absolute path at the filesystem root, which fails late and confusingly
   (or, worse, succeeds as root). `DATA_ROOT` has the same footgun today, so solve both together: either default the
   variable in the launch script, or add one guard that rejects a resolved path whose root segment is empty.
2. **It is a sweep across all 12 configs plus `config/hello_world.yaml`**, and every `input-path` / `model-path` /
   `metrics-path` / `report-path` / `source-path` must move with the `output-path` that produced it — a half-moved
   pipeline writes to the new root and reads from the old one, and `prepare_modeling`'s overwrite=false fast path
   would then quietly re-prepare into an empty directory.
3. **Quote every interpolated scalar** (`'{{$OUTPUT_ROOT}}/…'`), per the CLAUDE.md convention — substitution is
   textual and happens *before* the YAML parse.
4. **The lazy cache already copes, but check the number**: `lazy_content_max_dir_mb` defaults to 2048, so a 20 GiB
   prepared directory is fingerprinted in size-only metadata mode with a de-duplicated warning. That is the right
   behaviour — content-hashing 20 GiB per stage would dominate the run — but it means a size-preserving change to a
   prepared file is not detected. Worth stating in the config comment rather than leaving to be discovered.
5. **Decide whether the MLflow store moves too.** `mlruns/` is created next to `run_project.py`, holds the lazy
   cache's tags and every run's logged artifacts, and grows without bound. It is not covered by `OUTPUT_ROOT` as
   written above; `run_project.py` already accepts a `tracking_uri`, so the choice is between pointing that at the
   new root and leaving the store local by design.

## Carried in from Step 4

⚠️ **`split_index.csv`'s `file` column is an ABSOLUTE path into `$DATA_ROOT`, baked in at prepare time.** Found while
writing `prepare_modeling` (block 4a). Every other path in a prepared directory is a bare filename that
`load_prepared_artifacts` joins at read time — which is what makes a prepared directory movable — but `file` points
back at the source `samples/*.pt`, so it survives a move of the *outputs* and breaks on a move of the *dataset*.

The consequence is narrow and currently dormant, which is why it needs writing down rather than fixing now:

* it is read only by `LightningMapsDataset._item_features_checkpoint`, the **fallback** feature reader;
* every shipped config sets `materialize-features: true`, so the materialised reader is used and `file` is never
  opened. A stale `file` column therefore costs nothing today;
* the moment someone prepares with `materialize-features: false` on one machine and trains on another, it fails —
  with a `FileNotFoundError` naming a path that exists on neither, which reads as a data problem rather than a
  portability one.

Options for this step: store it relative to `data_path` and rejoin in `load_prepared_artifacts` (symmetric with the
other three columns, and the obvious fix), or drop the column entirely and require materialised features. The second
is tempting — the fallback path is dormant and `DayGroupedShuffleSampler` exists for it — but it removes the only way
to prepare without doubling the on-disk footprint, so decide it here rather than by accident.

## Open question carried from the architecture inventory

**Launch scripts** — port the three-script-per-machine pattern (`*.sh`), replace it with the per-user config file,
or both? (Question 6 in [`inventory-architecture.md`](inventory-architecture.md), still unanswered.)
