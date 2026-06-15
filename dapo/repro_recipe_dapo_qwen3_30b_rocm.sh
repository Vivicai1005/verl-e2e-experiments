#!/usr/bin/env bash
# =============================================================================
# Reproduce: DAPO | Qwen3-30B-A3B (MoE) | Megatron | ROCm (MI350/MI355)
# Path under test: recipe/dapo + `ray job submit`  (NOT the mainline main_ppo path)
# =============================================================================
#
# This records the *exact* sequence needed to get the recipe-based DAPO launcher
#   /workspace/verl-recipe/dapo/run_dapo_qwen3_moe_30b_megatron_rocm.sh
# running on a single 8-GPU node, starting from a fresh `verl` container.
#
# It is the companion to the mainline launcher
#   mi355_qwen3_30b_a3b_megatron_tp1ep8_infertp4.sh   (python3 -m verl.trainer.main_ppo)
# which does NOT use the recipe submodule or ray-job submission and therefore
# does NOT need steps 2-3 below. Use this script only when you specifically want
# the `recipe.dapo.main_dapo` entry point.
#
# WHERE TO RUN: inside the `verl` container, e.g.
#   docker exec -it verl bash
#   bash /workspace/verl-e2e-experiments/dapo/repro_recipe_dapo_qwen3_30b_rocm.sh
#
# The script is idempotent — safe to re-run. Each step checks before mutating.
#
# Prerequisites (see ./README.md): model weights + parquet datasets present at
# the paths the run script expects (defaults: $HOME/verl/models, $HOME/verl/data).
# -----------------------------------------------------------------------------

set -euo pipefail

# ----------------------------- configuration ---------------------------------
VERL_DIR=${VERL_DIR:-/workspace/verl}                       # verl core checkout
RECIPE_SRC=${RECIPE_SRC:-/workspace/verl-recipe}            # your verl-recipe fork
RECIPE_DST=${RECIPE_DST:-${VERL_DIR}/recipe}               # submodule mount point
RUN_SCRIPT=${RUN_SCRIPT:-${RECIPE_SRC}/dapo/run_dapo_qwen3_moe_30b_megatron_rocm.sh}

NOFILE=${NOFILE:-524288}                  # raised open-file limit (fixes raylet EMFILE)
RESTART_RAY=${RESTART_RAY:-1}             # 1 = stop+start the head node (kills running jobs)
TRAIN_ETP=${TRAIN_ETP:-2}                 # expert_tensor_parallel_size so ETP*EP*PP=8 on 8 GPUs

RAY_PORT=${RAY_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
OBJECT_STORE_MEMORY=${OBJECT_STORE_MEMORY:-10000000000}
SUBMIT=${SUBMIT:-1}                       # 1 = actually submit the job at the end

log() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

[ -d "$VERL_DIR" ] || { echo "ERROR: $VERL_DIR not found — run this inside the verl container." >&2; exit 1; }

# =============================================================================
# Step 1 — Raise the open-file limit and (re)start Ray
# -----------------------------------------------------------------------------
# Symptom we hit: `ray job submit` failed with HTTP 500 / aiohttp
#   "ServerDisconnectedError: Server disconnected". The real cause (in
#   /tmp/ray/session_latest/logs/raylet.err) was the raylet aborting with
#   SIGABRT: "Unhandled exception ... open: Too many open files [system:24]".
# The container's soft nofile limit was 1024 — far too low for Ray.
#
# CRITICAL: ulimit must be raised BEFORE `ray start`, in the same shell, because
# the raylet inherits the limit at launch. Raising it only before `ray job
# submit` does nothing to an already-running raylet.
# =============================================================================
log "Step 1: ulimit -n $NOFILE + Ray head"
ulimit -n "$NOFILE"
echo "soft nofile = $(ulimit -Sn)  (hard = $(ulimit -Hn))"

if [ "$RESTART_RAY" = 1 ]; then
    ray stop || true
    ray start --head --port="$RAY_PORT" \
        --dashboard-host=0.0.0.0 --dashboard-port="$DASHBOARD_PORT" \
        --object-store-memory="$OBJECT_STORE_MEMORY"
    # Verify the raylet actually inherited the raised limit.
    rl_pid=$(pgrep -f "raylet/raylet" | head -1 || true)
    if [ -n "${rl_pid:-}" ]; then
        echo "raylet $rl_pid -> $(grep 'Max open files' /proc/"$rl_pid"/limits)"
    fi
else
    echo "RESTART_RAY=0 -> assuming a Ray head started with ulimit -n $NOFILE already runs."
fi

# =============================================================================
# Step 2 — Populate the `recipe` git submodule
# -----------------------------------------------------------------------------
# Symptom: ModuleNotFoundError: No module named 'recipe.dapo'.
# Cause: `recipe` is a git submodule of verl (-> verl-recipe.git) that was never
# checked out, so ${VERL_DIR}/recipe was empty. The job uploads ${VERL_DIR} as
# its working dir, so `recipe.dapo` resolved to the empty folder.
# Fix: fill it from your verl-recipe fork (excluding .git).
# =============================================================================
log "Step 2: populate $RECIPE_DST from $RECIPE_SRC"
if [ -f "${RECIPE_DST}/dapo/main_dapo.py" ]; then
    echo "already populated -> skipping"
else
    [ -d "$RECIPE_SRC" ] || { echo "ERROR: $RECIPE_SRC not found." >&2; exit 1; }
    rm -f "${RECIPE_DST}/.git"          # submodule gitlink file; conflicts with copy
    mkdir -p "$RECIPE_DST"
    if command -v rsync >/dev/null; then
        rsync -a --exclude=".git" "${RECIPE_SRC}/" "${RECIPE_DST}/"
    else
        (cd "$RECIPE_SRC" && tar --exclude=.git -cf - .) | (cd "$RECIPE_DST" && tar -xf -)
    fi
    echo "populated -> $(ls "${RECIPE_DST}/dapo/main_dapo.py")"
fi

# =============================================================================
# Step 3 — Patch recipe imports for the current verl API
# -----------------------------------------------------------------------------
# Symptom: ImportError: cannot import name 'TaskRunner' from
#   'verl.trainer.main_ppo' (Did you mean 'TaskRunnerV1'?), then
#   ActorClassInheritanceException when subclassing the @ray.remote TaskRunner.
# Cause: this verl HEAD refactored main_ppo:
#   - run_ppo            -> still in verl.trainer.main_ppo (new signature)
#   - TaskRunner (+add_* builder methods) -> moved to verl.trainer.main_ppo_v0
#                                            and decorated with @ray.remote
#   - create_rl_dataset / create_rl_sampler -> moved to verl.trainer.ppo.utils
# Fix: redirect the imports, and recover the *undecorated* TaskRunner via
#   __ray_metadata__.modified_class so the recipe can subclass it and apply
#   ray.remote itself (as recipe/dapo/main_dapo.py already does).
# Applied to both the fork source and the uploaded copy.
# =============================================================================
log "Step 3: patch main_dapo.py imports"
for f in "${RECIPE_SRC}/dapo/main_dapo.py" "${RECIPE_DST}/dapo/main_dapo.py"; do
    [ -f "$f" ] || continue
    F="$f" python3 - <<'PY'
import os
f = os.environ["F"]
old = "from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo"
new = ("from verl.trainer.main_ppo import run_ppo\n"
       "from verl.trainer.main_ppo_v0 import TaskRunner as _RemoteTaskRunner\n"
       "# main_ppo_v0.TaskRunner is @ray.remote-decorated in this verl; recover the\n"
       "# plain class so the recipe can subclass it and apply ray.remote itself.\n"
       "TaskRunner = _RemoteTaskRunner.__ray_metadata__.modified_class\n"
       "from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler")
s = open(f).read()
if "_RemoteTaskRunner.__ray_metadata__.modified_class" in s:
    print(f"  already patched: {f}")
elif old in s:
    open(f, "w").write(s.replace(old, new))
    print(f"  patched: {f}")
else:
    print(f"  WARN: import line not found (manual check needed): {f}")
PY
done
# sanity: the patched imports must resolve and expose the builder method
( cd "$VERL_DIR" && python3 - <<'PY'
from verl.trainer.main_ppo import run_ppo  # noqa
from verl.trainer.main_ppo_v0 import TaskRunner as R
T = R.__ray_metadata__.modified_class
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler  # noqa
assert hasattr(T, "add_actor_rollout_worker"), "plain TaskRunner missing add_actor_rollout_worker"
print("  import sanity OK")
PY
)

# =============================================================================
# Step 4 — Fit the MoE parallelism to 8 GPUs
# -----------------------------------------------------------------------------
# Symptom: RuntimeError: world_size (8) is not divisible by
#   expert_tensor_model_pipeline_parallel size (16).
# Cause: the run script sets tp=4, ep=2, pp=2. expert_tensor_parallel_size
#   defaults to tp=4, so the expert group = ETP(4)*EP(2)*PP(2) = 16, which does
#   not divide world_size 8.
# Fix: pin expert_tensor_parallel_size=${TRAIN_ETP} (=2) -> 2*2*2 = 8 ✓, while
#   the dense layout tp4*pp2*cp1 = 8 already fills the node (dp=1).
#   (Alternative: set train_ep=1 in the run script -> 4*1*2 = 8.)
# The run script hardcodes its hydra args (no "$@" forwarding), so we inject the
# two ETP overrides directly, idempotently.
# =============================================================================
log "Step 4: set expert_tensor_parallel_size=$TRAIN_ETP in run script"
[ -f "$RUN_SCRIPT" ] || { echo "ERROR: $RUN_SCRIPT not found." >&2; exit 1; }
if grep -q "expert_tensor_parallel_size" "$RUN_SCRIPT"; then
    echo "already present -> skipping"
else
    # Append an ETP override line right after each expert_model_parallel_size arg,
    # preserving whether it targets .actor. or .ref.
    TRAIN_ETP="$TRAIN_ETP" perl -0pi -e '
        my $etp = $ENV{TRAIN_ETP};
        s{(actor_rollout_ref\.(actor|ref)\.megatron\.expert_model_parallel_size=\$\{train_ep\} \\\n)}
         {$1    actor_rollout_ref.$2.megatron.expert_tensor_parallel_size=$etp \\\n}g;
    ' "$RUN_SCRIPT"
    echo "inserted:"; grep -nE "expert_(model|tensor)_parallel_size" "$RUN_SCRIPT"
fi

# =============================================================================
# Step 5 — Submit
# -----------------------------------------------------------------------------
log "Step 5: submit DAPO job"
if [ "$SUBMIT" = 1 ]; then
    cd "$VERL_DIR"
    ulimit -n "$NOFILE"          # belt-and-suspenders for the submitting shell
    bash "$RUN_SCRIPT"
    echo
    echo "Submitted. Track with:"
    echo "  ray job list"
    echo "  ray job logs <job_id> -f"
else
    echo "SUBMIT=0 -> skipping. To launch:  cd $VERL_DIR && ulimit -n $NOFILE && bash $RUN_SCRIPT"
fi
