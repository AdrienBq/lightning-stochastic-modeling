# Inventory: figures, reporting & the map spec

**Purpose.** Factual inventory of the plotting code across the source branches, and the **visual grammar** the
rebuilt plotting should follow. **The `Decision` column is for you to fill in** (`keep` / `change` / `remove`).

**Scope:** this document defines *what a plot looks like* — panels, projection, colour encoding, layout. It does
**not** define the plotting pipeline (which dates are chosen, figure naming, persistence); those are Step 3/4
concerns. See "Out of scope" in §1.

## ⭐ The design reference is `notebooks/02a_visualize_val_event_diffusion.ipynb`

**Decided (2026-07-28).** The repo's plotting is based on **adrien's notebooks**, and specifically on **02a**.
This is the best plotting codebase across all branches. It is the *style and
structure* reference; the pipeline's job is to reproduce it non-interactively.

`adrien-mc-dropout`'s `src/utils/metrics/reporting.py` is **superseded** — it is not a design input. (Its
committed output, `main_outs/.../maps_worst_best_days.png`, is retained below only as a record of what the *old*
pipeline emitted and why it is being replaced — not as a reference.)

### 02a vs 02b — 02a is canonical

| | 02a (val event) | 02b (test event) |
|---|---|---|
| Split | `split == 'valid'` | `split == 'test'` |
| Plotting cells | **one** (cell 7) | **two** — cell 7 plus a near-identical "7b" that redefines `run_inference` *and* `plot_maps_and_psd` in full |
| Ensemble-std panel | `vmin=0` — scale adapts to the data | `vmin=0, **vmax=8**` — hardcoded ceiling |
| PSD placement | **separate figure** below the maps | 7b moves it into the std slot + a `fig.canvas.draw()` position-matching hack |

Everything else is byte-identical: the palettes, `make_lightning_cmap`, `draw_diff_map`, the projection setup and
the grid layout. So **02a is 02b minus the duplication and minus the magic number** — take 02a.

## Sources

| Key | Location | Size | Role |
|---|---|---|---|
| ⭐ **N** | `adrien-mc-dropout` : `notebooks/02a_visualize_val_event_diffusion.ipynb` | 9 cells | **THE design reference** — palettes, diff maps, layout, PSD |
| **A** | `aru-probabilistic-eval` : `src/utils/metrics/reporting.py` | 40 functions | The **pipeline plumbing** and the complete non-map figure set (curves, residual diagnostics). Structure to port N's styling *into* |
| ~~D~~ | `adrien-mc-dropout` : `src/utils/metrics/reporting.py` | 7 functions | **Superseded — not a design input** |

Sibling notebooks, same lineage: `01a_visualize_val_event`, `01b_visualize_val_event_mc_dropout`,
`01c_visualize_test_event_mc_dropout`, `02b_…test_event_diffusion`, `03_inspect_hyperparameters`,
`04_validation_power_spectrum`.

---

## 1. The map spec — as defined by 02a

This is the specification to implement. Every value below is what 02a actually does.

### Projection and framing

| Element | 02a | Decision |
|---|---|---|
| Axes projection | `ccrs.EuroPP()` | |
| Data transform | `ccrs.PlateCarree()` | |
| Array origin | `origin='upper'` (row 0 = north) | |
| Data extent | `[-12, 25, 35, 60]` — from `metadata.json` | |
| Display extent | `set_extent([-5, 20, 35, 55])` | |
| Coastlines | `ax.coastlines(linewidth=0.8)` | |
| Gridlines | `draw_labels=True, linestyle='--', alpha=0.7, linewidth=0.5`; `top_labels=False`, `right_labels=False`; left labels **only on the leftmost panel** | |
| Aspect | `ax.set_aspect('equal')` | |
| Borders | ✗ none (A has them — union or not?) | |

### The colour system — `make_lightning_cmap`

Two structurally identical 10-stop palettes plus one grey:

```python
_BASE_COLORS_WARM = ["#FFFFFF","#FFF5A6","#FFE37B","#FACA57","#F5AD37",
                     "#F08C1E","#E36C16","#D24D17","#B33117","#992015"]   # white->yellow->red
_BASE_COLORS_COOL = ["#FFFFFF","#D6E6F5","#A8CCE8","#7FB0DB","#5594CE",
                     "#3576C0","#1F5BA8","#13428A","#0A2E6B","#06204D"]   # white->blue->navy
_GREY = "#9E9E9E"                                                        # values in [0.5, 1) h/day
```

`make_lightning_cmap(max_val, base_colors)` returns `(ListedColormap, BoundaryNorm)` with

```python
levels = [0.0, 0.5] + list(range(1, max_val + 1))     # max_val = ceil(nanmax(obs))
colors = [gradient[0], grey] + gradient[1:]
```

so the value axis is: **`[0, 0.5)` white · `[0.5, 1)` grey · then one bin per whole hour** up to `ceil(max)`. Both
palettes are built against the *same* `max_val`, so the two `BoundaryNorm`s span an identical range and the two
colorbars are directly comparable.

| Property | Value | Decision |
|---|---|---|
| Quantization | unit-width integer bins in **lightning-hours** | |
| Sub-1 treatment | `[0, 0.5)` white, `[0.5, 1)` grey — near-zero visually separated from genuine low counts | |
| Scale source | `global_max = ceil(nanmax(obs))` — **observation-driven and per-date**; pred, obs and members all share it within a figure | keep per-date |

### `draw_diff_map` — the signature idea

One panel showing **magnitude and error direction simultaneously**, by drawing the prediction twice under
complementary masks:

```python
over  = np.ma.masked_where(pred <  obs_, pred)   # pred >= obs -> WARM palette
under = np.ma.masked_where(pred >= obs_, pred)   # pred <  obs -> COOL palette
ax.imshow(over,  cmap=cmap_warm, norm=norm_warm, origin='upper', transform=PC, extent=EXTENT)
ax.imshow(under, cmap=cmap_cool, norm=norm_cool, origin='upper', transform=PC, extent=EXTENT)
```

with two colorbars labelled `pred > obs (h / day)` and `pred < obs (h / day)`, sharing the value scale. **No
equivalent exists in A or D.** A's `_overunder_panel` is the nearest thing and is less informative.

### Layout

**Ensemble model** — 2 × 3 grid, `figsize=(NCOLS*5, NROWS*5)`, `hspace=0.17`, `wspace=0.05`,
`left=0.05, right=0.78, top=0.90, bottom=0.08`:

| | col 0 | col 1 | col 2 |
|---|---|---|---|
| row 0 | Observed *(warm)* | Predicted, ensemble mean *(diff)* | Predicted, ensemble std *(`viridis`, `vmin=0`)* |
| row 1 | Member 1 *(diff)* | Member 2 *(diff)* | Member 3 *(diff)* |

**Deterministic model** — 1 × 2: `Observed` | `Predicted (diff)`, `figsize=(11, 5.5)`.

Colorbars: the std panel gets a detached `viridis` bar spanning **row 0 only**
(`fig.add_axes([0.80, row0_bottom, 0.016, cell_h])`); the two diff bars span **both rows** at x `0.85` / `0.91`.
Title is `fig.suptitle(str(date), fontsize=14, fontweight='bold')`.

### The PSD figure

A **separate** figure (`figsize=(8, 5)`) drawn after the maps:

- individual members in `darkorange`, `alpha=0.5, linewidth=1`, capped at `N_MEMBERS_PLOT = 6`, single legend entry
- `Observed` in `steelblue`, `Predicted (ensemble mean)` in `darkorange`, both `linewidth=2`
- `loglog`, **`invert_xaxis()`** so large wavelengths are on the left
- axes: `Wavelength [pixels]` / `Radially-averaged power`; `grid(True, which='both', alpha=0.3)`. Modify the pixel scale to have km instead.
- built on `scores.radial_psd(field[np.newaxis])` — the **1-D** radial profile

### Colour scale: per-date — **decided**

`make_lightning_cmap` is called with `global_max = ceil(nanmax(obs))` **for the date being plotted**, so the scale
adapts per day. **Keep this.** Both palettes are built against that same `max_val`, so within a figure every panel
shares one scale and the two diff colorbars stay aligned — which is what matters. Panels from *different* days
being on different scales is accepted.

### Out of scope for this document

02a is interactive — an `ipywidgets.Dropdown` drives re-running the plot cell. **Drop the widget**, and do not
treat this document as a specification of the plotting *pipeline*. It defines the **visual grammar**: what a plot
should look like, what the panels are, how colour encodes value and error direction. It deliberately does **not**
decide which dates get plotted, how figures are named, or how they're selected and persisted — those are
pipeline concerns for Step 3/4, and A's existing machinery (`_select_plot_indices`, `_save_figure`, the
`reporting.figures` config list) already covers them.

---

## 2. Pipeline figures — A's set, to be restyled per §1

| Figure | A | in `metrics_daily.yaml` | What it shows | Decision |
|---|---|---|---|---|
| `maps_worst_best_days` | `_maps_per_day` (413) | ✅ | Observed vs predicted day maps → **restyle to §1** | keep — **renamed `maps_most_extreme_days`** in Step 2's `metrics_daily.yaml`: "worst/best" reads as a judgement of the *model*, when what the figure actually selects is the most extreme *observed* day (plus the median) |
| `psd_curves` | 481 | ✅ | Radially-averaged PSD vs obs and baselines → **restyle to §1's PSD figure** | keep |
| `fss_vs_scale` | 514 | ✅ | FSS curve per threshold | keep |
| `reliability_and_pit` | 540 | ✅ | Reliability diagram + PIT histogram — ⚠️ **split, see §4** | **renamed `reliability`** in Step 2's `metrics_daily.yaml` — the reliability half is kept and promoted to a headline diagnostic; the PIT half is dropped with the transform that fitted its CDF |
| `error_by_intensity_bin` | 598 | ✅ | Stratified error across intensity bins | keep — bins become the explicit hour bands (`occurrence`/`h3`/`h6`/`h12`) |
| `rank_histogram` | 615 | ✅ | Talagrand diagram (ensemble runs only) | keep |
| `roc_pr_curves` | — | **NEW (Step 2)** | ROC **and** precision-recall curves, one pair per event threshold. PR is backed by the existing `average_precision`; ROC by the new `roc_auc`. Plotting them together is the point: at the hourly 0.43 % base rate the ROC curve looks flattering while the PR curve exposes the real operating trade-off | keep — **new code in Step 3** |
| `confusion_matrix` | — | **NEW (Step 2)** | 2×2 contingency counts per threshold — the raw hits/misses/false-alarms/correct-negatives *behind* `pod`/`far`/`csi`/`ets`, which are otherwise only visible as ratios | keep — **new code in Step 3** |
| `residual_bias_map` | 694 | ✅ | Mean predicted discrepancy (diverging) | |
| `residual_surprise` | 710 | ✅ | Magnitude + direction surprise vs true discrepancy | |
| `residual_histograms` | 737 | ✅ | `D_pred` vs `D_true` distributions | |
| `residual_qq` | 767 | ✅ | Residual QQ | |
| `residual_scatters` | 784 | ✅ | `D_pred` vs target / discrepancy / upstream | |
| `residual_heteroscedasticity` | 807 | ✅ | Correction-error spread vs upstream / target | |
| `qq_plot` | — | ✅ **declared, never implemented** | `scores.quantile_quantile` exists but no figure function does | |

The six `residual_*` figures are backed by `metrics/diagnostics.py` (`residual_diagnostics`, A-only) and self-skip
unless a residual run populated `curves['residual']`.

### A's map plumbing — keep the structure, replace the styling

| Helper | Line | Fate under §1 |
|---|---|---|
| `_geographic_context` | 56 | **Keep** — returns `(EuroPP, PlateCarree)`, matching 02a; also holds the lazy-import fallback |
| `_add_map_axis` / `_map_imshow` / `_decorate_map_axis` | 67–80 | Keep as structure; `_decorate_map_axis` gains 02a's labelled gridlines |
| `_resolve_map_norm` / `_quantize_field` | 179 / 122 | **Replace** with `make_lightning_cmap` |
| `_colorbar_ticks` / `_integer_bin_formatter` | 153 / 169 | Simplify — 02a's `levels` already encode the tick positions |
| `_thin_horizontal_colorbar` / `_double_colorbar` | 235 / 247 | Replace with `add_shared_diff_colorbars` (02a uses **vertical**, right-hand bars) |
| `_count_panel` / `_std_panel` / `_overunder_panel` | 270–289 | Replace with `draw_map` / `draw_diff_map` |
| `_deterministic_day_figure` / `_stochastic_day_figure` | 303 / 327 | Replace with 02a's two layouts |
| `_select_plot_indices` | 380 | Keep (A-only) — but **out of scope here**: which dates get plotted is a pipeline decision, not part of the visual grammar |
| `_occurrence` / `_occurrence_mask_below` / `_active_maxima` | 108–143 | Reconcile with 02a's observation-driven `global_max` |
| `_diverging_norm` / `_solid_cmap` / `_residual_map_panel` / `_diverging_colorbar` | 652–686 | Keep for the residual figures |

---

## 3. Record: why D's `reporting.py` is superseded

Not a design input — kept only to justify the replacement. Its committed output
(`main_outs/trial_7_regress_psd_full/reports/maps_worst_best_days.png`, extracted to
`outputs/ref_adrien_maps_worst_best_days.png`) shows: **no projection or coastlines** (raw pixel indices,
confirming the grid is **101 × 149**, `origin='upper'`), a `magma`-style map rendering **zero as black** so a
95.3 %-zero field reads as a black rectangle, **one colorbar per panel** so observed and predicted sit on
different scales, and `most_active` / `worst_error` resolving to the same day so a third of the figure repeats.

It is, however, a good illustration of challenge (B): the prediction is visibly over-smoothed against the
speckled observation.

---

## 4. Scope decision (2026-07-28): classification-first, no transform

| Item | Impact |
|---|---|
| `make_lightning_cmap`'s integer bins | ✅ **Better fit than before.** A `0–24` target has exactly 25 natural unit bins, so `ceil(max_val)` is bounded by 24 and no capped-top-bin handling is needed |
| `LogNorm` / `colorbar_scale: log` (A only) | 🔢 `COUNT-REG` — exists for the heavy-tailed count field; pointless on `0–24`. **Remove** (02a has no log option) |
| `draw_diff_map`, warm/cool palettes | Unaffected, and *more* interpretable: over/under prediction in **whole hours** |
| White/grey sub-1 band | Unaffected — still the right treatment for `y = 0` |
| `psd_curves`, `fss_vs_scale` | Unaffected — spectral and neighbourhood scores work on binary and `0–24` fields alike |
| `reliability_and_pit` | ⚠️ **Split it.** *Reliability* is ✅ `CLASSIF` and becomes a headline figure. **PIT must be re-derived or dropped** — see below |
| `error_by_intensity_bin` | 🎯 Bins are positive-count quantiles → redefine as explicit hour bands; degenerate for binary |
| `qq_plot` | 🔢 `COUNT-REG` — a staircase over ~24 integers. Dropping it also closes issue #3 (declared but never implemented) |
| Six `residual_*` figures | Not wrong, but residual-mode diffusion is **lower priority** than the occurrence task now |

### PIT — corrected

An earlier note in this file said PIT depends on `pit_histogram: {space: transformed}`. **That config key does not
exist.** What the code actually does (`evaluation.py:464-474`):

```python
shape, scale = float(target_stats['gamma_shape']), float(target_stats['gamma_scale'])
u_prediction  = gammainc(shape, np.clip(prediction[occurrence], 0, None) / scale)
u_observation = gammainc(shape, observation[occurrence] / scale)
```

PIT reaches **directly into `target_stats['gamma_shape']` / `['gamma_scale']`** and calls `gammainc` (the gamma
CDF) itself — independent of whether the model trained under the transform. Both prediction and observation are
pushed through that CDF at occurrence cells; a marginally-calibrated model gives ~Uniform(0,1), and
`uniform_histogram_ks` reports the KS distance to uniform (`pit_ks`) plus the histogram densities.

**So the dependency is on the *fitted gamma parameters*, not on the transform being active.** Removal-checklist
item 7 deletes `compute_target_transform_stats`, which is what writes those parameters — so PIT's two call sites
break for lack of an `F`, not for lack of a transform. Options: drop PIT, or re-derive it without a fitted CDF
(**ensemble-rank PIT**, which needs only member ranks against the observation, or a binned empirical CDF on the
`0–24` target).

### Candidate new figures for the occurrence task

Nothing in any of the three implementations covers these, and they'd be the natural headline diagnostics:
**ROC / precision-recall curves** (backed by the existing `average_precision`) and a
**confusion matrix per threshold**.

---

## 5. Issues found

| # | Issue | Where | Severity |
|---|---|---|---|
| 1 | **Display extent ≠ data extent.** Data drawn at `[-12, 25, 35, 60]`, view set to `[-5, 20, 30, 55]` — shows 5° of empty space south of the data and **clips 55–60 °N where real data exists** (southern Scandinavia, northern UK) | **02a and 02b**, `_frame` | ⚠️ likely bug |
| 2 | Hardcoded `vmax=8` on the std panel | **02b only** — 02a uses `vmin=0` | avoided by taking 02a. drop the hardcoded vmax and derive it with max for the given day |
| 3 | Cell 7 / 7b near-total duplication | **02b only** — 02a has one plotting cell | avoided by taking 02a |
| 4 | `qq_plot` declared in `metrics_daily.yaml` but no figure function exists | A/D vs config | inconsistency |
| 5 | Stray `µ` character inside `run_inference`'s docstring | 02a, cell 7 | cosmetic |

## Open questions for you

1. **Borders?** A draws country borders, 02a does not. Add them to the 02a style, or keep coastlines only?
2. **Keep cartopy optional?** A imports it lazily and degrades to plain axes; we now have it as a hard
   requirement. Keep the fallback or let missing cartopy be a hard error?
3. **PIT** — drop, or re-derive as ensemble-rank PIT?
4. **Notebooks in scope?** Port 02a (deduplicated, widget removed) into the new repo, and what about the other six?
5. **Fix issue #1** (the extent mismatch)? If 55–60 °N should be visible, every map changes.

*Resolved: colour scale stays per-date; the `ipywidgets` date selector is dropped.*

1. Discard coastlines
2. Keep cartopy as a hard requirement for now
3. drop
4. Do not port the notebooks for now. We will make plotting functions in src.utils that are inspired by 02a
5. don't make 55-60 °N visible

### ⚠️ Clarification of answer 1 (2026-08-12) — it meant discard BORDERS

Question 1 offered two options, "add borders to the 02a style" or "keep coastlines only", and the answer named
neither, so on re-reading in Step 3 Block 4 it parsed as *remove the basemap entirely*. Confirmed with the user:

> **Coastlines stay** (`ax.coastlines(linewidth=0.8)`, the geographic anchor). **Country borders are not added** —
> A draws them, 02a does not, and their line density would compete with a field that is 95.3 % zero.

`gate_block4.py` pins both halves — `coastlines` present, `BORDERS` / `add_feature` absent — in both `maps.py` and
`reporting.py`, so neither side of the decision can drift back silently.

### Answers 2 and 5, as implemented

- **Answer 2 was not honoured until Block 4.** Block 2 shipped `geographic_context` with a `try/except ImportError`
  that returned `(None, None)`, plus six call sites branching on it. Since cartopy is a hard dependency
  (`minimal_requirements.txt:39`), that path was dead in any working install and live only in a broken one — where
  it would emit exactly the projection-less pixel-index figures §3 condemns. The fallback is now gone and a missing
  cartopy raises at import.
- **Answer 5 costs the top ~20 array rows.** `DISPLAY_EXTENT` stops at 55 °N while the data runs to 60 °N, so rows
  0–20 of every `[101, 149]` field are never drawn (issue 1's northern clip, accepted). The gate asserts the crop so
  it stays visible as a decision, and its own north/south orientation probes light rows *inside* the shown window.