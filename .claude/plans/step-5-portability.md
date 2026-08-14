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
