# =====================================================================================================================
# Shared by every job_scripts.example/<stage>.sh — sourced, never submitted.
#
# Holds the things every stage script needs and none of them should re-derive: the repo root, the interpreter, the two
# data roots, the family/mode -> path mapping, and the guards. Seven copies of this would have drifted apart by the
# second edit.
#
# A stage script must, IN THIS ORDER:
#   1. set FAMILY / TIER / MODE (and whatever it needs of its own)
#   2. source this file
#   3. run "$PYTHON" src/stages/<stage>.py ... using the paths defined here
#
# ⚠️ NOTHING IN HERE IS SITE-SPECIFIC, deliberately — that is the difference between this directory and the gitignored
# `job_scripts/` it was derived from. Paths come from `.env` (or the environment); the interpreter comes from PATH.
# Copy this directory to `job_scripts/` and edit freely if you want machine-specific defaults; that copy is gitignored.
# =====================================================================================================================

# --- the repo root ---------------------------------------------------------------------------------------------------
# Found by SEARCHING, because neither obvious guess is reliable:
#   * SLURM_SUBMIT_DIR is the directory you ran sbatch FROM, so `cd job_scripts.example && sbatch tune.sh` makes it
#     job_scripts.example, not the repo;
#   * sbatch COPIES the script into /var/spool/slurm/..., so at run time ${BASH_SOURCE[0]} is not where you submitted
#     from — trusting it alone resolved to / and gave "mkdir: cannot create directory 'job_scripts'".
REPO_ROOT=""
for _candidate in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
                  "$(dirname "${BASH_SOURCE[1]:-$0}")" "$(dirname "${BASH_SOURCE[1]:-$0}")/.." "$PWD" "$PWD/.."; do
    if [ -n "$_candidate" ] && [ -f "$_candidate/src/stages/evaluate.py" ]; then
        REPO_ROOT="$(cd "$_candidate" && pwd)"
        break
    fi
done
if [ -z "$REPO_ROOT" ]; then
    echo "Cannot locate the repo root: no src/stages/evaluate.py in the submit dir, its parent, this script's dir, its parent, or \$PWD. Submit from inside the repo." >&2
    exit 2
fi
cd "$REPO_ROOT"                        # every stage resolves `config/...` and its relative outputs against the root

# --- the per-user config ---------------------------------------------------------------------------------------------
# `.env` is KEY=VALUE, which is readable by BOTH `src/__init__.py` and a shell — one config file, not two. `set -a`
# exports what it sets, so the stage subprocesses inherit it.
#
# ⚠️ AN ALREADY-EXPORTED VALUE STILL WINS, matching `load_env_file` in src/utils/io/environment.py: `.env` is the
# fallback, not the authority. This used to be a claim rather than a behaviour — a plain `. file` ASSIGNS
# unconditionally, so `.env` silently overrode a job that exported its own DATA_ROOT, which is precisely the
# retargeting that `test_an_ALREADY_SET_variable_is_NEVER_overridden` forbids on the Python side.
#
# Snapshot the exported environment, source the file, then re-assert the snapshot: a name that was already set keeps
# its value, a name the file introduces survives. Doing it this way rather than parsing KEY=VALUE here keeps the
# `.env` grammar defined in ONE place (parse_env_file) instead of two that agree until the first edit.
# `grep -v` drops readonly entries (bash exports SHELLOPTS/BASHOPTS as `declare -rx` when they are exported at all);
# re-declaring one of those fails, and under `set -e` that would kill the job instead of loading the config.
# An EMPTY exported value counts as set and wins too, exactly as it does in Python — `UPSTREAM_MODEL=` means
# "no warm start", and `.env` must not resurrect it.
if [ -f "$REPO_ROOT/.env" ]; then
    _env_snapshot="$(export -p | grep -vE '^declare -[a-zA-Z]*r')"
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
    eval "$_env_snapshot"
    unset _env_snapshot
fi

# --- the interpreter -------------------------------------------------------------------------------------------------
# ⚠️ NOT hardcoded. Export PYTHON, or activate your environment before `sbatch` — slurm hands the job the environment of
# the shell you submitted from, PATH included, so an activated venv carries over.
PYTHON="${PYTHON:-$(command -v python || true)}"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "No usable interpreter: PYTHON is unset and no \`python\` is on PATH. Activate your environment before" >&2
    echo "submitting, or export PYTHON=/path/to/venv/bin/python. See minimal_requirements.txt for the install." >&2
    exit 2
fi
# mlflow.projects shells out to a BARE `python`, so the interpreter must be FIRST on PATH or every stage subprocess
# under pipeline.sh gets a different one and dies on `import mlflow`. `src/__init__.py` also does this from inside
# Python (prepend_interpreter_to_path); doing it here too means the per-stage scripts are covered before any import.
export PATH="$(dirname "$PYTHON"):$PATH"

# --- the two roots ---------------------------------------------------------------------------------------------------
# ⚠️ NO FALLBACK VALUES. The gitignored `job_scripts/` this was derived from hardcoded a machine's paths here, which is
# fine for scratch and wrong for anything tracked: a wrong-but-plausible default is worse than a missing one, because
# the run succeeds against the wrong data.
if [ -z "${DATA_ROOT:-}" ] || [ -z "${OUTPUT_ROOT:-}" ]; then
    echo "DATA_ROOT and OUTPUT_ROOT must both be set (got DATA_ROOT='${DATA_ROOT:-}' OUTPUT_ROOT='${OUTPUT_ROOT:-}')." >&2
    echo "  cp .env.example .env    # then edit the two required lines" >&2
    echo "Run \`python scripts/preflight.py\` to check the whole environment at once." >&2
    exit 2
fi
export DATA_ROOT OUTPUT_ROOT

[ -f "$DATA_ROOT/metadata.json" ] || { echo "DATA_ROOT='$DATA_ROOT' has no metadata.json: not a dataset root." >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

# Keep the mlflow store with the outputs rather than in the checkout, where it grows without bound. Confirmed to
# relocate the whole store; unset, artifacts land in ./mlruns next to run_project.py.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${OUTPUT_ROOT}/mlruns}"

# --- family + mode -> paths ------------------------------------------------------------------------------------------
FAMILY="${FAMILY:?a stage script must set FAMILY before sourcing _common.sh}"
TIER="${TIER-}"                        # unset is fine and means the FULL tier
# The MODE, and the three files that must move together with it (see CLAUDE.md). `daily` is the 0-24 lightning-hours
# regression; `hourly` is the 0/1 occurrence classification, which only deterministic_unet has a pipeline for.
# It is the same key `prepare_modeling` takes, and the only thing that selects between the two tasks.
MODE="${MODE:-daily}"
case "$MODE" in
    daily|hourly) ;;
    *) echo "Unknown MODE '$MODE' (expected daily or hourly)." >&2; exit 2 ;;
esac

# ⚠️ The two U-net families SHARE one prepared directory; diffusion has its OWN, and that is not tidiness. In residual
# mode diffusion's preparation appends an upstream conditioning channel, so a U-net checkpoint pointed at that
# directory fails its in_channels check. See CLAUDE.md.
case "$FAMILY" in
    deterministic_unet|mc_dropout) PREPARED_OWNER="deterministic_and_mc_dropout" ;;
    diffusion)                     PREPARED_OWNER="diffusion" ;;
    *) echo "Unknown FAMILY '$FAMILY' (expected deterministic_unet, mc_dropout or diffusion)." >&2; exit 2 ;;
esac

# ⚠️ The hourly pipeline shares NOTHING: its targets are [T, H, W] 0/1 rather than [H, W] 0-24, so it owns both its run
# directory and its prepared directory. Its config names them `deterministic_unet_hourly*`, and these must agree with
# that config or a stage run through these scripts writes somewhere the pipeline never reads.
if [ "$MODE" = "hourly" ]; then
    [ "$FAMILY" = "deterministic_unet" ] || {
        echo "MODE=hourly exists for deterministic_unet only (no hourly pipeline for '$FAMILY')." >&2; exit 2; }
    RUN_DIR="${OUTPUT_ROOT}/${FAMILY}_hourly${TIER}"
    PREPARED_PATH="${RUN_DIR}/prepared/hourly"
else
    RUN_DIR="${OUTPUT_ROOT}/${FAMILY}${TIER}"
    PREPARED_PATH="${OUTPUT_ROOT}/${PREPARED_OWNER}${TIER}/prepared/daily"
fi
TUNING_PATH="${RUN_DIR}/tuning"
BEST_PATH="${RUN_DIR}/best"
EVALUATION_PATH="${RUN_DIR}/evaluation"
REPORT_PATH="${RUN_DIR}/reports"
SEARCH_SPACE="config/${FAMILY}/search_space_${MODE}.yaml"
METRICS_CONFIG="${METRICS_CONFIG:-config/eval/metrics_${MODE}.yaml}"
SPLIT_CONFIG="${SPLIT_CONFIG:-config/split/split${TIER}.yaml}"

# --- logging ---------------------------------------------------------------------------------------------------------
# The script's own full transcript, under $OUTPUT_ROOT rather than in the checkout — same rule as every other output.
# slurm's own --output/--error land in the directory you submitted from (they name no directory on purpose, so a
# submit from anywhere still produces a log).
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "$LOG_DIR"
STAGE_NAME="${STAGE_NAME:-stage}"
# ⚠️ MODE is in the name. Without it an hourly evaluate logged to `evaluate_deterministic_unet_smoke_cpu_*.log`,
# indistinguishable from the daily one it sits beside -- and the two report on different prepared data.
LOG="${LOG_DIR}/${STAGE_NAME}_${FAMILY}_${MODE}${TIER}_$(date +%Y%m%d_%H%M%S).log"

require_path() {                       # fail naming the missing input, not 40 lines into the stage
    for _required in "$@"; do
        [ -e "$_required" ] || { echo "missing input: $_required" >&2; exit 3; }
    done
}

banner() {
    echo "stage=$STAGE_NAME family=$FAMILY mode=$MODE tier=${TIER:-full} python=$PYTHON"
    echo "repo=$REPO_ROOT"
    echo "data=$DATA_ROOT"
    echo "out=$OUTPUT_ROOT"
    echo "log=$LOG"
}
