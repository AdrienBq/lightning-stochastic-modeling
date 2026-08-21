#!/bin/bash
# Tensorise the predictors + the daily target for one split. See job_scripts.example/README.md.
#SBATCH --job-name=prep_lightning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# #SBATCH --partition=<your-cpu-partition>    omit to use the cluster default; `sinfo` lists them
#
# ⚠️ Submit from the REPO ROOT: `sbatch job_scripts.example/<this>.sh`. The script finds the repo either way, but every
# path it passes to a stage is relative to the root.
# The --output/--error above deliberately name NO directory, so they land in whatever directory you submitted from and
# you ALWAYS get a log. The earlier `job_scripts/logs/output/%x_%j.out` form resolved against the SUBMIT directory, so
# submitting from inside job_scripts/ asked for job_scripts/job_scripts/logs/, which slurm could not create -- killing
# the job in one second with no log at all to say why. That is a bad way to learn where to stand.
set -euo pipefail

# ===== EDIT ME =======================================================================================================
FAMILY=deterministic_unet          # deterministic_unet | mc_dropout | diffusion  (the first two SHARE a prepared dir)
TIER=_smoke_cpu                    # _smoke_cpu | _smoke_gpu | '' (full)
MODE=daily                         # daily | hourly.  ⚠️ ONE variable, read by _common.sh too: it picks the prepared
                                   # directory, the search space and the metrics config, so the three cannot disagree.
                                   # hourly gives 24x the items, deterministic_unet only, and does NOT materialize
                                   # features (that is what turns on DayGroupedShuffleSampler).
FEATURE_DTYPE=float32              # float32 for a smoke tier; float16 for the full one (halves 19.7 GiB per split).
                                   # DAILY ONLY -- hourly writes no feature files.
OVERWRITE=false                    # false reuses an existing prepared dir -- how the 2nd U-net family skips this stage
HOURLY_THRESHOLD=2                 # >= 2 strokes for an hour to count. BAKED INTO THE TARGET: changing it means
                                   # re-preparing AND re-training everything downstream.
# diffusion only: set to build RESIDUAL artifacts (upstream/ maps + residual_target). Leave empty for full-target.
UPSTREAM_MODEL="${UPSTREAM_MODEL:-}"
UPSTREAM_ACCELERATOR=cpu
UPSTREAM_BATCH_SIZE=2
UPSTREAM_NUM_WORKERS=0
# =====================================================================================================================

STAGE_NAME=prepare
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh
require_path "$SPLIT_CONFIG"
banner
echo "prepared -> $PREPARED_PATH   (mode=$MODE dtype=$FEATURE_DTYPE overwrite=$OVERWRITE)"

EXTRA=()
# The task-dependent half of the argument list, mirroring the two shipped pipelines exactly. `feature-aggregation` is
# daily-mode only (an hourly item is already one hour) and `feature-dtype` is the storage dtype of files hourly does
# not write, so both are OMITTED there rather than passed and ignored.
if [ "$MODE" = "hourly" ]; then
    EXTRA+=(--materialize-features False)
else
    EXTRA+=(--feature-aggregation hourly_stack
            --materialize-features True
            --feature-dtype "$FEATURE_DTYPE")
fi
if [ -n "$UPSTREAM_MODEL" ]; then
    [ "$FAMILY" = diffusion ] || { echo "UPSTREAM_MODEL on prepare_modeling is a DIFFUSION-only switch (residual mode). MC-dropout takes its upstream at the TUNE stage instead." >&2; exit 2; }
    require_path "$UPSTREAM_MODEL"
    EXTRA+=(--upstream-model-path "$UPSTREAM_MODEL"
            --upstream-accelerator "$UPSTREAM_ACCELERATOR"
            --upstream-devices 1
            --upstream-num-workers "$UPSTREAM_NUM_WORKERS"
            --upstream-batch-size "$UPSTREAM_BATCH_SIZE")
fi

"$PYTHON" src/stages/prepare_modeling.py \
    --data-path "$DATA_ROOT" \
    --split-config "$SPLIT_CONFIG" \
    --output-path "$PREPARED_PATH" \
    --mode "$MODE" \
    --features MU_LI,MU_MIXR,RH_500850,cp,lsm \
    --hourly-threshold "$HOURLY_THRESHOLD" \
    --overwrite "$OVERWRITE" \
    "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "done: prepared -> $PREPARED_PATH"
