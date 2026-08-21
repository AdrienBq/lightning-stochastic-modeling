#!/bin/bash
# The comparison FIGURES: the families' curves overlaid. Seconds, not minutes.
#SBATCH --job-name=curves_lightning
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
TIER=_smoke_cpu
# =====================================================================================================================

FAMILY=deterministic_unet          # only to satisfy _common.sh; this stage spans all three
STAGE_NAME=curves
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh

CURVES_DIR="${OUTPUT_ROOT}/comparison${TIER}/curves"
mkdir -p "$CURVES_DIR"
banner
echo "curves -> $CURVES_DIR"

ARGS=()
for family in deterministic_unet mc_dropout diffusion; do
    reports="${OUTPUT_ROOT}/${family}${TIER}/reports"
    case "$family" in
        deterministic_unet) label=Deterministic-UNet ;;
        mc_dropout)         label=MC-Dropout ;;
        diffusion)          label=Diffusion ;;
    esac
    if [ -d "$reports" ]; then
        ARGS+=(--"$label" "$reports")
    else
        echo "skipping $label: no $reports"
    fi
done

"$PYTHON" src/stages/combine_curves.py --output-path "$CURVES_DIR" "${ARGS[@]}" 2>&1 | tee "$LOG"
echo "done: figures -> $CURVES_DIR"
