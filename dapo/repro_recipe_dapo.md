# Reproduce: DAPO · Qwen3-30B-A3B (MoE) · Megatron · ROCm (MI350/MI355)


- **Path under test:** `recipe/dapo` + `ray job submit` → `recipe.dapo.main_dapo`
- **Launcher script:** `/workspace/verl-recipe/dapo/run_dapo_qwen3_moe_30b_megatron_rocm.sh`

---

## Environment
```bash
docker exec -it verl bash
cd /workspace/verl
```

## Prerequisites

- model: `$HOME/verl/models/Qwen3-30B-A3B`
- train: `$HOME/verl/data/dapo-math-17k.parquet`
- eval: `$HOME/verl/data/aime-2024.parquet`

---

## Step 1 — Raise the open-file limit, then start Ray

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

## Step 2 — Populate the `recipe` submodule

```bash
rm -f /workspace/verl/recipe/.git          # submodule gitlink file conflicts with the copy
rsync -a --exclude=".git" /workspace/verl-recipe/ /workspace/verl/recipe/

ls /workspace/verl/recipe/dapo/main_dapo.py   # verify present
```

---

## Step 3 — Submit and monitor

```bash
cd /workspace/verl
ulimit -n 524288
bash /workspace/verl-recipe/dapo/run_dapo_qwen3_moe_30b_megatron_rocm.sh

# track it
ray job list
ray job logs <job_id> -f
```

---

## Issues & fixes (quick reference)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ServerDisconnectedError` / HTTP 500 on `ray job submit`; raylet SIGABRT `Too many open files` | soft `nofile` = 1024 | `ulimit -n 524288` **before** `ray start` (Step 1) |
| `ModuleNotFoundError: No module named 'recipe.dapo'` | empty `recipe` submodule | copy fork → `verl/recipe` (Step 2) |
| `ImportError: cannot import name 'TaskRunner'` | verl API refactor | redirect imports |
| `ActorClassInheritanceException` subclassing `TaskRunner` | `main_ppo_v0.TaskRunner` is `@ray.remote` | use `__ray_metadata__.modified_class` |

---
