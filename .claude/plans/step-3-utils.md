# Step 3 (block c) — Shared `src/utils`

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Prev: [Step 2](step-2-config.md) · Next: [Step 4](step-4-stages.md)

> **Status: provisional — to be expanded** once [Step 2](step-2-config.md) is done, in the same way Step 2 was
> expanded from its one-line placeholder.

Build in dependency order, implementing the annotated inventories: `io/data.py` → modeling (`dataset`, `unet`,
unified `losses`, `search`, `validation`, two-phase `tuning`, then model modules + `registry`) → metrics
(`scores`, `evaluation`, `reporting`, `diagnostics`) → plotting functions covering the 02a map spec.

*(`transforms.py` is **not** built — the target transform is dropped. See [`00-context.md`](00-context.md).)*

## Carried into this step from earlier steps

- **The 15-item transform-removal checklist** — [`inventory-architecture.md`](inventory-architecture.md) §6.
- **Per-folder `README.md` standards** in `src/stages/`, `src/utils/metrics/`, `src/utils/modeling/`,
  `src/utils/plotting/`, capturing the agreed contracts. *(Deferred here from Step 1 — those folders don't exist
  until now, and documenting contracts before the code invites drift.)*
- **Plotting functions in `src/utils` inspired by notebook 02a** (not a notebook port). Full visual spec:
  [`inventory-figures.md`](inventory-figures.md) §1.
- **From [Step 2](step-2-config.md) §7:** `evaluation.resolve_threshold` gains a `kind: absolute` branch;
  `psd_full_fidelity` becomes a metric key backed by `psd_band_ratios` + `psd_fidelity`; `fss_useful_scale` reuses
  precomputed per-scale FSS instead of recomputing it.
