# Reproduce: DAPO · Qwen3-30B-A3B (MoE) · Megatron · ROCm (MI350/MI355)

Step-by-step guide to get the **recipe-based** DAPO launcher running on a single
8-GPU node, starting from a fresh `verl` container.

- **Path under test:** `recipe/dapo` + `ray job submit` → `recipe.dapo.main_dapo`
- **Launcher script:** `/workspace/verl-recipe/dapo/run_dapo_qwen3_moe_30b_megatron_rocm.sh`
- **One-shot equivalent:** [`repro_recipe_dapo_qwen3_30b_rocm.sh`](./repro_recipe_dapo_qwen3_30b_rocm.sh)
  runs every step below automatically (idempotent).

> **Not the same as** [`mi355_qwen3_30b_a3b_megatron_tp1ep8_infertp4.sh`](./mi355_qwen3_30b_a3b_megatron_tp1ep8_infertp4.sh).
> That one uses the mainline `python3 -m verl.trainer.main_ppo` entry point and
> does **not** touch the `recipe` submodule or `ray job submit`, so it does not
> need Steps 2–3 here. Use this guide only when you specifically want the
> `recipe.dapo.main_dapo` entry point.

---

## Environment

| | |
| --- | --- |
| Hardware | 8× GPU single node (MI350 / MI355 class) |
| Container | `verl` |
| verl core | `/workspace/verl` (volcengine/verl HEAD) |
| recipe fork | `/workspace/verl-recipe` (Vivicai1005/verl-recipe) |
| Backend | Megatron + vLLM rollout |

Get a shell in the container:

```bash
docker exec -it verl bash
cd /workspace/verl
```

## Prerequisites

Model weights and datasets must already be present at the paths the run script
expects (defaults below; see [`README.md`](./README.md) for download commands):

- model: `$HOME/verl/models/Qwen3-30B-A3B`
- converted mcore ckpt: `$HOME/verl/models/Qwen3-30B-A3B-dist_ckpt`
- train: `$HOME/verl/data/dapo-math-17k.parquet`
- eval: `$HOME/verl/data/aime-2024.parquet`

---

## Step 1 — Raise the open-file limit, then start Ray

**Symptom.** `ray job submit` fails with HTTP 500 and the client prints:

```
RuntimeError: Request failed with status code 500
aiohttp.client_exceptions.ServerDisconnectedError: Server disconnected
```

**Real cause.** The raylet crashed. Look at the raylet log:

```bash
grep -n "Too many open files" /tmp/ray/session_latest/logs/raylet.err
# (raylet) Unhandled exception ... what(): open: Too many open files [system:24]  -> SIGABRT
```

The container's soft `nofile` limit was **1024** — far too low for Ray. When the
job opened more sockets than that, the raylet aborted, taking the dashboard/job
agents down with it (hence "Server disconnected").

**Fix.** Raise the limit **before** starting Ray, in the **same shell** — the
raylet inherits the limit at launch. Raising it only before `ray job submit`
does nothing to an already-running raylet.

```bash
ulimit -n 524288
ulimit -Sn                 # verify -> 524288

ray stop
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --object-store-memory=10000000000
```

**Verify** the raylet actually got the new limit:

```bash
pid=$(pgrep -f "raylet/raylet" | head -1); grep "Max open files" /proc/$pid/limits
# -> Max open files   524288   524288   files
```

> **Make it permanent (optional).** Start the container with
> `--ulimit nofile=524288:524288`, or add `* soft/hard nofile 524288` to
> `/etc/security/limits.conf`, or put `ulimit -n 524288` at the top of your
> launch script.

---

## Step 2 — Populate the `recipe` submodule

**Symptom.**

```
/usr/bin/python3: Error while finding module specification for
'recipe.dapo.main_dapo' (ModuleNotFoundError: No module named 'recipe.dapo')
```

**Cause.** `recipe` is a git submodule of verl (→ `verl-recipe.git`) that was
never checked out, so `/workspace/verl/recipe` is empty. The job uploads
`/workspace/verl` as its working dir, so `recipe.dapo` resolves to that empty
folder. Your actual recipe code lives in the fork at `/workspace/verl-recipe`.

```bash
cat /workspace/verl/.gitmodules            # shows: submodule "recipe" -> verl-recipe.git
ls /workspace/verl/recipe                  # empty
```

**Fix.** Fill the submodule directory from your fork (excluding `.git`):

```bash
rm -f /workspace/verl/recipe/.git          # submodule gitlink file conflicts with the copy
rsync -a --exclude=".git" /workspace/verl-recipe/ /workspace/verl/recipe/

ls /workspace/verl/recipe/dapo/main_dapo.py   # verify present
```

---

## Step 3 — Patch `main_dapo.py` imports for the current verl API

**Symptoms (in order, as you fix each one):**

```
ImportError: cannot import name 'TaskRunner' from 'verl.trainer.main_ppo'
             (Did you mean: 'TaskRunnerV1'?)
...
ray.actor.ActorClassInheritanceException: Attempted to define subclass
'DAPOTaskRunner' of actor class 'TaskRunner'. Inheriting from actor classes is
not currently supported.
```

**Cause.** This verl HEAD refactored `main_ppo`:

| What the recipe imports | New location in this verl |
| --- | --- |
| `run_ppo` | still `verl.trainer.main_ppo` (new signature, recipe already uses it) |
| `TaskRunner` (+ `add_*` builder methods) | `verl.trainer.main_ppo_v0`, now `@ray.remote`-decorated |
| `create_rl_dataset`, `create_rl_sampler` | `verl.trainer.ppo.utils` |

Because `main_ppo_v0.TaskRunner` is now an actor class, the recipe can't subclass
it directly — it needs the **plain** class, which Ray keeps at
`TaskRunner.__ray_metadata__.modified_class`.

**Fix.** Edit **both** copies of `main_dapo.py`
(`/workspace/verl-recipe/dapo/` and `/workspace/verl/recipe/dapo/`). Replace the
single import line:

```python
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
```

with:

```python
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import TaskRunner as _RemoteTaskRunner
# main_ppo_v0.TaskRunner is @ray.remote-decorated in this verl; recover the
# plain class so the recipe can subclass it and apply ray.remote itself.
TaskRunner = _RemoteTaskRunner.__ray_metadata__.modified_class
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
```

**Verify** the imports resolve and the plain class still has the builder method:

```bash
cd /workspace/verl && python3 -c "
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_v0 import TaskRunner as R
T = R.__ray_metadata__.modified_class
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
print('OK, add_actor_rollout_worker =', hasattr(T, 'add_actor_rollout_worker'))
"
# -> OK, add_actor_rollout_worker = True
```

---

## Step 4 — Fit the MoE parallelism to 8 GPUs

**Symptom.**

```
RuntimeError: world_size (8) is not divisible by
expert_tensor_model_pipeline_parallel size (16)
```

**Cause.** The run script sets `tp=4, ep=2, pp=2`. The expert group size is
`expert_tensor_parallel × expert_model_parallel × pipeline`. With
`expert_tensor_parallel` defaulting to `tp=4`, that's `4 × 2 × 2 = 16`, which
doesn't divide `world_size = 8`. (The dense layout `tp4 × pp2 × cp1 = 8` already
fills the node, so data-parallel = 1; only the expert group is over-sized.)

**Fix — pin `expert_tensor_parallel_size=2`** so `2 × 2 × 2 = 8` ✓. Add these two
args to the `ray job submit ... -- python3 -m recipe.dapo.main_dapo` command in
`run_dapo_qwen3_moe_30b_megatron_rocm.sh` (next to the existing
`expert_model_parallel_size` lines):

```bash
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=2 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=2 \
```

> **Alternative:** set `train_ep=1` near the top of the run script → `4 × 1 × 2 = 8`
> (disables expert parallelism instead of shrinking expert-TP).

**General rule on N GPUs:** `ETP × EP × PP` must divide the world size.

---

## Step 5 — Submit and monitor

```bash
cd /workspace/verl
ulimit -n 524288
bash /workspace/verl-recipe/dapo/run_dapo_qwen3_moe_30b_megatron_rocm.sh

# track it
ray job list
ray job logs <job_id> -f
```

A healthy start looks like: `DAPOTaskRunner` boots → 8 `WorkerDict` actors spin
up → NCCL/Gloo process groups connect across all 8 ranks → Megatron model init →
vLLM rollout → reward computation, with no assertion errors.

---

## Issues & fixes (quick reference)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ServerDisconnectedError` / HTTP 500 on `ray job submit`; raylet SIGABRT `Too many open files` | soft `nofile` = 1024 | `ulimit -n 524288` **before** `ray start` (Step 1) |
| `ModuleNotFoundError: No module named 'recipe.dapo'` | empty `recipe` submodule | copy fork → `verl/recipe` (Step 2) |
| `ImportError: cannot import name 'TaskRunner'` | verl API refactor | redirect imports (Step 3) |
| `ActorClassInheritanceException` subclassing `TaskRunner` | `main_ppo_v0.TaskRunner` is `@ray.remote` | use `__ray_metadata__.modified_class` (Step 3) |
| `world_size (8) not divisible by ... size (16)` | `ETP(4)×EP(2)×PP(2)=16` | `expert_tensor_parallel_size=2` (Step 4) |

---

## One-shot script

To run all of the above automatically (idempotent — safe to re-run):

```bash
# copy into the container (verl has no bind mounts), then run
docker cp ~/verl-e2e-experiments/dapo/repro_recipe_dapo_qwen3_30b_rocm.sh \
  verl:/workspace/verl-e2e-experiments/dapo/
docker exec -it verl bash -lc \
  'bash /workspace/verl-e2e-experiments/dapo/repro_recipe_dapo_qwen3_30b_rocm.sh'
```

Tunable via env vars: `TRAIN_ETP`, `NOFILE`, `RESTART_RAY`, `SUBMIT`, `VERL_DIR`,
`RECIPE_SRC`, `RUN_SCRIPT`. Example — patch/setup only, don't submit:

```bash
SUBMIT=0 bash repro_recipe_dapo_qwen3_30b_rocm.sh
```
