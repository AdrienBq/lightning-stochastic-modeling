#!/bin/bash
# Hyperparameter sweep for one family. See job_scripts.example/README.md.
#SBATCH --job-name=tune_lightning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# #SBATCH --partition=<your-cpu-partition>    omit to use the cluster default; `sinfo` lists them
# #SBATCH --gres=gpu:1             uncomment TOGETHER WITH ACCELERATOR=gpu (and a GPU partition above). Either
#                                  half alone fails: no allocation -> lightning raises on accelerator: gpu;
#                                  no ACCELERATOR change -> an idle GPU and a CPU-speed run.
#
# ⚠️ Submit from the REPO ROOT: `sbatch job_scripts.example/<this>.sh`. The script finds the repo either way, but every
# path it passes to a stage is relative to the root.
# The --output/--error above deliberately name NO directory, so they land in whatever directory you submitted from and
# you ALWAYS get a log. The earlier `job_scripts/logs/output/%x_%j.out` form resolved against the SUBMIT directory, so
# submitting from inside job_scripts/ asked for job_scripts/job_scripts/logs/, which slurm could not create -- killing
# the job in one second with no log at all to say why. That is a bad way to learn where to stand.
set -euo pipefail

# ===== EDIT ME =======================================================================================================
FAMILY=deterministic_unet          # deterministic_unet | mc_dropout | diffusion
TIER=_smoke_cpu                    # _smoke_cpu | _smoke_gpu | '' (full)
MODE=daily                         # daily | hourly (deterministic_unet only). Picks the prepared dir, the search
                                   # space and the metrics config together -- see _common.sh.
N_TRIALS=2                         # full tier ships 40 (deterministic) / 30 (mc_dropout, diffusion)
SAMPLER=random                      # random | tpe  (tpe needs enough trials to learn from; full tier uses tpe)
MAX_EPOCHS=2                       # full tier: 50. ⚠️ mc_dropout WARM START ignores this (phase 1 is skipped)
PATIENCE=1                         # full tier: 10
PRUNING=false                      # true needs sampler=tpe
PROGRESS_BAR=False                 # False for a batch job: a tqdm bar in a redirected log is noise
RESTART=true                       # ⚠️ true for any smoke tier. run_sweep resumes from its own optuna store inside
                                   # output-path, which the pipeline's `lazy: false` does NOT govern -- a re-run
                                   # otherwise reports success having executed no trial at all.
FEATURE_STATS_DAYS=4               # cannot exceed the TRAIN days of the split (smoke split has 4); full tier: 256
ACCELERATOR=cpu                    # cpu | gpu | auto
NUM_WORKERS=0                      # full tier: 8
# mc_dropout only: a deterministic_unet checkpoint -> WARM START (load its weights, run the finetune phase alone).
UPSTREAM_MODEL="${UPSTREAM_MODEL:-}"
# =====================================================================================================================

STAGE_NAME=tune
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh
require_path "$PREPARED_PATH" "$SEARCH_SPACE" "$METRICS_CONFIG"
banner
echo "prepared=$PREPARED_PATH -> tuning=$TUNING_PATH  (trials=$N_TRIALS sampler=$SAMPLER restart=$RESTART)"

EXTRA=()
if [ -n "$UPSTREAM_MODEL" ]; then
    [ "$FAMILY" = mc_dropout ] || { echo "UPSTREAM_MODEL on tune is an MC-DROPOUT-only switch (it wants the upstream WEIGHTS). Diffusion takes its upstream at PREPARE_MODELING instead, where it becomes a conditioning channel." >&2; exit 2; }
    require_path "$UPSTREAM_MODEL"
    EXTRA+=(--upstream-model-path "$UPSTREAM_MODEL")
    echo "WARM START from $UPSTREAM_MODEL"
fi

"$PYTHON" src/stages/tune.py \
    --model-family "$FAMILY" \
    --model-type "$FAMILY" \
    --input-path "$PREPARED_PATH" \
    --model-config "$SEARCH_SPACE" \
    --metrics-config "$METRICS_CONFIG" \
    --output-path "$TUNING_PATH" \
    --metrics-path "$TUNING_PATH/best_trial_metrics.json" \
    --n-trials "$N_TRIALS" \
    --sampler "$SAMPLER" \
    --max-epochs "$MAX_EPOCHS" \
    --early-stopping-patience "$PATIENCE" \
    --pruning "$PRUNING" \
    --restart "$RESTART" \
    --accelerator "$ACCELERATOR" \
    --devices 1 \
    --num-workers "$NUM_WORKERS" \
    --progress-bar "$PROGRESS_BAR" \
    --feature-stats-days "$FEATURE_STATS_DAYS" \
    "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "done: sweep -> $TUNING_PATH"
