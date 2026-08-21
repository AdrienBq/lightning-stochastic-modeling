#!/bin/bash
# Refit the winning trial. See job_scripts.example/README.md.
#SBATCH --job-name=retrain_lightning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=1-00:00:00
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
FAMILY=deterministic_unet
TIER=_smoke_cpu
MODE=daily                         # daily | hourly (deterministic_unet only); see _common.sh
MAX_EPOCHS=2                       # full tier: 80
PATIENCE=1                         # full tier: 10
ACCELERATOR=cpu
NUM_WORKERS=0
PROGRESS_BAR=False                 # False for a batch job: a tqdm bar in a redirected log is noise

# ⚠️ Deliberately NO upstream switch. The stage reads the SWEEP'S RECORD from source-path/best_trial.json, so a
# warm-started sweep is retrained warm-started automatically and the two cannot disagree. Pass --upstream-model-path
# by hand only to OVERRIDE (a re-trained upstream); it warns when it differs from the record.
# =====================================================================================================================

STAGE_NAME=retrain
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh
require_path "$PREPARED_PATH" "$TUNING_PATH/best_trial.json"
banner
echo "sweep=$TUNING_PATH -> best=$BEST_PATH"

"$PYTHON" src/stages/retrain_best.py \
    --model-family "$FAMILY" \
    --model-type "$FAMILY" \
    --source-path "$TUNING_PATH" \
    --input-path "$PREPARED_PATH" \
    --metrics-config "$METRICS_CONFIG" \
    --output-path "$BEST_PATH" \
    --metrics-path "$BEST_PATH/best_trial_metrics.json" \
    --max-epochs "$MAX_EPOCHS" \
    --early-stopping-patience "$PATIENCE" \
    --accelerator "$ACCELERATOR" \
    --devices 1 \
    --num-workers "$NUM_WORKERS" \
    --progress-bar "$PROGRESS_BAR" 2>&1 | tee "$LOG"
echo "done: checkpoint -> $BEST_PATH/best_model.ckpt"
