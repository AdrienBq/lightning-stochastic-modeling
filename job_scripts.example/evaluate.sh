#!/bin/bash
# The shared evaluation for one family: metrics JSON + the report directory. See job_scripts.example/README.md.
#SBATCH --job-name=eval_lightning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# #SBATCH --partition=<your-cpu-partition>    omit to use the cluster default; `sinfo` lists them
# #SBATCH --gres=gpu:1             uncomment TOGETHER WITH ACCELERATOR=gpu (and a GPU partition above)
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
MODE=hourly                         # daily | hourly (deterministic_unet only); see _common.sh
SPLIT=test                         # test | valid

ENSEMBLE_SIZE=2                    # members per item. ⚠️ NEVER 1: spread_skill_sums uses ddof=1, so one member gives a
                                   # silent NaN. Shipped values:  smoke: 2 / 2 / 2      full: 2 / 32 / 32
                                   # (deterministic emits no members at any size -- its ensemble scalars are NaN by
                                   # design and dropped from the JSON.)
SAMPLING_STEPS=8                   # DIFFUSION ONLY, ignored otherwise. Dominates a CPU diffusion evaluation: 2 items
                                   # x 2 members x 8 steps took 22 min on the real grid, vs 15 s for deterministic.
ACCELERATOR=cpu                    # cpu | gpu | auto
NUM_WORKERS=0                      # full tier: 8
BATCH_SIZE=2                       # full tier: 16
PROGRESS_BAR=False                 # False for a batch job: a tqdm bar in a redirected log is noise
# =====================================================================================================================

STAGE_NAME=evaluate
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh

[ "$ENSEMBLE_SIZE" -ge 2 ] || { echo "ENSEMBLE_SIZE must be >= 2: spread_skill_sums uses ddof=1, so 1 member gives NaN with no error." >&2; exit 2; }
require_path "$PREPARED_PATH" "$BEST_PATH/best_model.ckpt" "$METRICS_CONFIG"
banner
echo "model=$BEST_PATH/best_model.ckpt  split=$SPLIT ensemble=$ENSEMBLE_SIZE"
echo "report -> $REPORT_PATH"

EXTRA=()
[ "$FAMILY" = diffusion ] && EXTRA+=(--sampling-steps "$SAMPLING_STEPS")

"$PYTHON" src/stages/evaluate.py \
    --input-path "$PREPARED_PATH" \
    --model-path "$BEST_PATH/best_model.ckpt" \
    --output-path "$EVALUATION_PATH" \
    --metrics-config "$METRICS_CONFIG" \
    --metrics-path "$EVALUATION_PATH/${SPLIT}_metrics.json" \
    --report-path "$REPORT_PATH" \
    --split "$SPLIT" \
    --baselines zero,climatology \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --accelerator "$ACCELERATOR" \
    --devices 1 \
    --num-workers "$NUM_WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --progress-bar "$PROGRESS_BAR" \
    "${EXTRA[@]}" 2>&1 | tee "$LOG"
echo "done: metrics -> $EVALUATION_PATH/${SPLIT}_metrics.json ; report -> $REPORT_PATH"
