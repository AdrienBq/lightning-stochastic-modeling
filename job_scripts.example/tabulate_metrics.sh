#!/bin/bash
# The comparison TABLE: one CSV, families as rows. Seconds, not minutes.
#SBATCH --job-name=tabulate_lightning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
TIER=_smoke_cpu                    # which tier's evaluations to compare
SPLIT=test
# =====================================================================================================================

FAMILY=deterministic_unet          # only to satisfy _common.sh; this stage spans all three
STAGE_NAME=tabulate
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh

COMPARISON_DIR="${OUTPUT_ROOT}/comparison${TIER}"
mkdir -p "$COMPARISON_DIR"
banner
echo "comparison -> $COMPARISON_DIR/metrics_comparison_${SPLIT}.csv"

# ⚠️ The flag NAME is the row label: Fire maps --MC-Dropout to the kwargs key MC_Dropout and the stage restores the
# hyphen, so these must stay hyphenated to match config/eval/probabilistic_eval.yaml and combine_curves' legends.
ARGS=()
for family in deterministic_unet mc_dropout diffusion; do
    metrics="${OUTPUT_ROOT}/${family}${TIER}/evaluation/${SPLIT}_metrics.json"
    case "$family" in
        deterministic_unet) label=Deterministic-UNet ;;
        mc_dropout)         label=MC-Dropout ;;
        diffusion)          label=Diffusion ;;
    esac
    if [ -f "$metrics" ]; then
        ARGS+=(--"$label" "$metrics")
    else
        echo "skipping $label: no $metrics"      # a partial comparison is legal; the stage raises only if ALL are absent
    fi
done

"$PYTHON" src/stages/tabulate_metrics.py \
    --output-path "$COMPARISON_DIR/metrics_comparison_${SPLIT}.csv" \
    "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "done: table -> $COMPARISON_DIR/metrics_comparison_${SPLIT}.csv"
