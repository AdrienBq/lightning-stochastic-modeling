#!/bin/bash
# =====================================================================================================================
# END TO END: every stage of one family's pipeline, through the ORCHESTRATOR.
#
# Unlike the per-stage scripts this does NOT call src/stages/*.py directly — it runs run_project.py on the shipped
# YAML, which is the supported path and the only one that gives you:
#   * an mlflow run per stage, with the parameters and metrics logged and the configs attached as artifacts;
#   * the lazy cache (skip a stage whose code + params + input checksums are unchanged);
#   * the stages in the right order, with each one's outputs wired to the next.
# The per-stage scripts are for re-running ONE stage (a report, a fresh sweep); this is for a real run.
#
# ⚠️ COMMIT FIRST. The lazy cache keys on the whole-repo dirty diff, so an uncommitted edit busts every entry and the
# run tells you nothing about the committed state.
# =====================================================================================================================
#SBATCH --job-name=pipeline_lightning
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# #SBATCH --partition=<your-cpu-partition>    omit to use the cluster default; `sinfo` lists them
# #SBATCH --gres=gpu:1             for a *_smoke_gpu tier. ⚠️ The tier's YAML already sets `accelerator: gpu`,
#                                  so WITHOUT this line the job gets no GPU and lightning raises.
#
# ⚠️ Submit from the REPO ROOT: `sbatch job_scripts.example/<this>.sh`. The script finds the repo either way, but every
# path it passes to a stage is relative to the root.
# The --output/--error above deliberately name NO directory, so they land in whatever directory you submitted from and
# you ALWAYS get a log. The earlier `job_scripts/logs/output/%x_%j.out` form resolved against the SUBMIT directory, so
# submitting from inside job_scripts/ asked for job_scripts/job_scripts/logs/, which slurm could not create -- killing
# the job in one second with no log at all to say why. That is a bad way to learn where to stand.
set -euo pipefail

# ===== EDIT ME =======================================================================================================
CONFIG=config/deterministic_unet/deterministic_unet_daily_smoke_cpu.yaml
# The shipped pipelines. Every task-specific config names its task, so there is no unsuffixed ambiguity:
#   daily  (0-24 lightning-hours regression)   config/<family>/<family>_daily[_smoke_cpu|_smoke_gpu].yaml
#   hourly (0/1 occurrence classification)     config/deterministic_unet/deterministic_unet_hourly[_smoke_cpu].yaml
#   cross-family comparison                    config/eval/probabilistic_eval[_smoke_cpu].yaml
# ⚠️ MODE is not set here on purpose: the CONFIG already carries `mode:` on its prepare_modeling block, and this script
# runs the whole pipeline through run_project.py rather than assembling stage arguments itself.
EXPERIMENT=lightning_smoke
# For a family whose pipeline reads it (mc_dropout at TUNE, diffusion at PREPARE_MODELING). Leave empty otherwise:
# unset means "no warm start" / "full-target", and both stages treat an empty string that way.
UPSTREAM_MODEL="${UPSTREAM_MODEL:-}"
# =====================================================================================================================

FAMILY=deterministic_unet          # only to satisfy _common.sh; the CONFIG is what selects the family here
STAGE_NAME=pipeline
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh" 2>/dev/null \
    || source job_scripts/_common.sh 2>/dev/null \
    || source job_scripts.example/_common.sh
require_path "$CONFIG"
export UPSTREAM_MODEL              # the config substitutes {{$UPSTREAM_MODEL}} from the environment

# mlflow.projects runs each stage as `python src/stages/<stage>.py`, a BARE `python` resolved from PATH rather than
# sys.executable, so the interpreter must be first on PATH — done in _common.sh, and again inside Python by
# `src/__init__.py`. Belt and braces on purpose: this is the failure that reads as a broken STAGE when it is a broken
# environment. MLFLOW_TRACKING_URI is set in _common.sh too, so it applies to every script rather than only this one.
[ "$(command -v python)" = "$(dirname "$PYTHON")/python" ] \
    || echo "WARNING: \`python\` on PATH is $(command -v python), not $PYTHON — mlflow would use it for every stage"

banner
echo "config=$CONFIG experiment=$EXPERIMENT"
echo "tracking=$MLFLOW_TRACKING_URI"
[ -n "$UPSTREAM_MODEL" ] && echo "UPSTREAM_MODEL=$UPSTREAM_MODEL" || echo "UPSTREAM_MODEL unset (no warm start / full-target)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "WARNING: the working tree is DIRTY. The lazy cache keys on the whole-repo diff, so every entry is busted and"
    echo "         this run does not describe the committed state. Commit first for a run you intend to trust."
fi

"$PYTHON" run_project.py "$CONFIG" "$EXPERIMENT" 2>&1 | tee "$LOG"
echo "done: see $MLFLOW_TRACKING_URI and $OUTPUT_ROOT"
