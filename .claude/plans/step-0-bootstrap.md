# Step 0 (block —) — Bootstrap & hygiene — ✅ **DONE** (2026-07-27)

> Part of the split rebuild plan. Index: [`rebuild-plan.md`](rebuild-plan.md) ·
> Context: [`00-context.md`](00-context.md) · Next: [Step 1](step-1-design.md)

- **Rebranded:** `README.md` (title, framing paragraph — diagnostic mapping, not forecasting — plus new
  Installation and Data sections); `config/hello_world.yaml` `description` → the project tagline (feeds the
  banner; the banner *name* auto-derives from the directory, so no code change). `environment.yml` kept but
  marked **unmaintained** with a header pointing at `minimal_requirements.txt`.
- **`minimal_requirements.txt` written** — hand-built, loose bounds (`>=`,`<`): mlflow, fire, pyyaml, torch,
  lightning, optuna, numpy, pandas, scipy, scikit-learn, matplotlib, pytest. Carries
  `--extra-index-url https://download.pytorch.org/whl/cpu` because the machine has **no GPU** and the default
  PyPI linux torch wheel bundles ~2.5 GB of `nvidia-*` deps. Confirmed correct **not** to copy the old
  `requirements.txt`: it is a 154-line full env dump (horovod, poetry, fastapi, jupyter, rasterio…).
  Excluded until a step needs them: `geopandas`, `xarray`/`netCDF4` (data is `.pt` + CSV, not netCDF),
  `prefect` (`run.py` falls back to the mlflow orchestrator when absent).
  `torchmetrics` arrives anyway as a `lightning` transitive dep.
- **`.gitattributes` / `.gitignore`: no carry-over needed.** The template's are a strict superset of
  `fers26p8`'s — identical LFS lists plus the template's ML-checkpoint/tensor blocks (`*.pt`, `*.ckpt`,
  `*.safetensors`, `*.npz`…), and the template `.gitignore` adds `/outputs/` and `**/*.db`. Verified via
  `git check-attr`, not modified.
- **Verified:** venv is Python 3.11.7 and runs standalone (`module load python/meso-3.11` needed only to
  *create* it); `torch 2.13.0+cpu` with zero `nvidia-*` packages; all deps import; `parse_config` reads the
  YAML; `python run_project.py config/hello_world.yaml STEP0_SMOKE` exits 0 with the rebranded banner and
  seeding applied to `random, numpy, lightning, torch`; `git status` scoped to the intended files.
- **Open follow-up (deliberately deferred):** `mlflow` resolved to **3.12.0**, one minor release below the
  `<3.13` ceiling where the local file-store backend is disabled — a fresh install in a few weeks will break.
  Tighten the bound when that bites.
- **Amended during Step 1:** `cartopy>=0.22,<1` added to `minimal_requirements.txt` once the map spec required it
  (lifting Step 0's deferral). Installed from prebuilt wheels — cartopy 0.25.0 + pyproj 3.7.2 + pyshp 3.1.6 +
  shapely 2.1.2, **no system GEOS/PROJ build needed** — and verified by rendering a real map at the domain extent.
  **Decision (2026-07-28): cartopy stays a hard requirement** (aru's lazy-import-with-fallback is not preserved).
- **Corroboration found in Step 1:** adrien's branch already had a `minimal_requirements.txt` with the same
  package set; ours is newer, bounded, plus `pytest` + `cartopy`. No change needed. (Its `pandas==3.0.3` alongside
  `torch==2.0.1` looks like a typo for `2.0.3`.)
