# Bundled Natural Earth data

`shapefiles/natural_earth/physical/ne_50m_coastline.*` — the coastline `src/utils/plotting/maps.py` draws, committed so
plotting needs no network.

## Why it is here

cartopy downloads Natural Earth data lazily, at **plot** time, into a per-user cache
(`~/.local/share/cartopy`). That is a network call from wherever plotting happens — usually a compute node, often
offline — and it arrives at the *end* of a pipeline run rather than the start. Bundling removes it.

The layout is cartopy's own, not ours. `src/__init__.py` sets `CARTOPY_DATA_DIR` to this directory, which cartopy reads
into `config['pre_existing_data_dir']` and consults **before** attempting any download:

```
{CARTOPY_DATA_DIR}/shapefiles/natural_earth/{category}/ne_{resolution}_{name}.shp
```

⚠️ cartopy reads that variable **once, at import**. A script that does `import cartopy` before `import src` falls
through to the download cache with no sign of why — `use_bundled_cartopy_data` repairs the already-imported case, but
importing `src` first is the rule.

## Why exactly this one file

Measured, not guessed. `frame_map_axis` calls `ax.coastlines()` at cartopy's default `resolution='auto'`, which picks
the shapefile from the **axes extent** — so the requirement is a property of the figures, not of the code. Instrumenting
`shapereader.natural_earth` and rendering one figure at the real `EuroPP` extent requests exactly:

```
[('50m', 'physical', 'coastline')]
```

One dataset, 1,128,501 bytes across the five sidecar files. A warm `~/.local/share/cartopy` may hold ~14 MB (10m
coastline, admin boundary lines, land, ocean) — none of which this repo ever asks for.

Change the projection or the domain and `auto` may ask for `10m` or `110m` instead, which is not bundled; cartopy then
falls back to the download cache. **`scripts/prewarm_cartopy.py` reports that case** (it discovers the requirement by
rendering, so it cannot drift from `maps.py`) and fetches what is missing with `--fetch`.

## Why these are NOT git-lfs objects

`*.shp` is LFS-tracked in [`.gitattributes`](../../.gitattributes) — but `data/cartopy/shapefiles/**` is **exempted** there, so
all five files are ordinary blobs and a plain `git clone` receives the real data.

That exemption was added after measuring what LFS actually costs here. Committing an LFS object requires
`git lfs install`, which sets `filter.lfs.required = true`, and then:

* **every** git command in the repo exits **128** when the binary is absent — `git status`, `git diff`, `git commit` —
  not a warning, a hard failure;
* including the `git diff HEAD` that `lazy.code_state_hash` runs on every pipeline run. That one is caught, so the
  pipeline does not crash — it **silently degrades** to hashing `src/` alone, dropping `config/` and the working-tree
  state from the cache key, after which stages are re-run or wrongly skipped.

A hard runtime dependency on every machine, plus a silently weakened cache, is not a good trade for 1.1 MB. (And
`pip install git-lfs` is not an escape: that PyPI package is a fetch-only reimplementation that cannot upload, and pip
runs *after* `git clone` anyway.)

⚠️ **The exemption is load-bearing.** Remove those two lines from `.gitattributes` and the next commit by anyone with
git-lfs installed converts the shapefile into a ~130-byte pointer — after which cartopy dies with
`KeyError: 828781878` inside `shapefile.py`, the first bytes of the word "version" read as a shape type. That reads as
corrupt data rather than a configuration mistake, which is why two tests guard it: one asserts `git check-attr` reports
no filter for this path, the other that the file on disk is not a pointer.

## Provenance and licence

[Natural Earth](https://www.naturalearthdata.com/) 50m physical coastline, obtained through cartopy's own downloader
(`https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip`). Natural Earth is in the **public domain**:

> All versions of Natural Earth raster + vector map data found on this website are in the public domain. You may use
> the maps in any manner, including modifying the content and design, electronic dissemination, and offset printing.

No attribution is required; it is recorded here because provenance of a committed binary should never be a mystery.
