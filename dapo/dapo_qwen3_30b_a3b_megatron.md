# DAPO — Qwen3-30B-A3B (Megatron + vLLM)

End-to-end DAPO training run for `Qwen/Qwen3-30B-A3B` (MoE) on the verl
Megatron + vLLM backend. This doc records the **exact data prep, config, and
launch commands** so the run is reproducible after a container/cluster rebuild.

DAPO = Decoupled clip + dynamic sampling policy optimization. Key pieces vs.
vanilla GRPO:
- **clip-higher** — asymmetric PPO clip (`clip_ratio_low` < `clip_ratio_high`).
- **dynamic sampling** — filter prompt groups whose rewards are all-equal
  (all-correct / all-wrong) so every batch carries gradient signal.
- **token-level policy loss** — loss aggregated over tokens, not sequences.
- **overlong reward shaping** — soft length penalty instead of a hard cut.

---

## Environment

| | |
| --- | --- |
| Hardware | `<e.g. 8× MI300X / node count>` |
| Container | `<image tag>` |
| verl commit | `<git rev-parse HEAD in /workspace/verl>` |
| vLLM | `<version>` (+ patches in `../patches/`) |
| Backend | Megatron (`strategy=megatron`) + vLLM rollout |

> If you applied the vllm patches, note it here and link
> [`../patches/README.md`](../patches/README.md).

---

## Parallelism / model config

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B` |
| TP | `<COMMON_TP>` |
| PP | `<COMMON_PP>` |
| EP | `<COMMON_EP>` |
| ETP | `<COMMON_ETP>` |
| CP | `<COMMON_CP>` |
| Rollout TP (`INFER_TP`) | `<…>` |
| mbridge | `<USE_MBRIDGE / VANILLA_MBRIDGE>` |

---

## Prerequisites: dataset and model downloads

Run once before training. Paths below are what the launch command expects —
adjust if you change them there.

- train set: `~/data/DAPO-Math-17k/data/dapo-math-17k.parquet`
- eval set: `~/data/AIME-2024/data/aime-2024.parquet`
- base model: `~/models/Qwen/Qwen3-30B-A3B`

### Dataset (DAPO-Math-17k train + AIME-2024 eval)

Pull the pre-built parquet files straight from the Hugging Face dataset repos:

```bash
mkdir -p $HOME/data/DAPO-Math-17k
mkdir -p $HOME/data/AIME-2024

hf download BytedTsinghua-SIA/DAPO-Math-17k \
  data/dapo-math-17k.parquet \
  --repo-type dataset \
  --local-dir $HOME/data/DAPO-Math-17k

hf download BytedTsinghua-SIA/AIME-2024 \
  data/aime-2024.parquet \
  --repo-type dataset \
  --local-dir $HOME/data/AIME-2024
```

### Base model — full weights

DAPO trains real weights (not a dummy model), so download the full checkpoint:

```bash
mkdir -p $HOME/models/Qwen/Qwen3-30B-A3B

hf download Qwen/Qwen3-30B-A3B \
  --local-dir $HOME/models/Qwen/Qwen3-30B-A3B
```

---

## DAPO hyperparameters

| Knob | Value | Notes |
| --- | --- | --- |
| `adv_estimator` | `grpo` | DAPO builds on GRPO advantages |
| `clip_ratio_low` | `0.2` | clip-higher |
| `clip_ratio_high` | `0.28` | clip-higher |
| `loss_agg_mode` | `token-mean` | token-level loss |
| dynamic sampling | `filter_groups.enable=True` | metric `acc`, drop all-equal groups |
| `max_num_gen_batches` | `<…>` | resample cap for dynamic sampling |
| overlong shaping | `overlong_buffer.enable=True` | `len=<…>`, `penalty_factor=<…>` |
| `max_prompt_length` | `<…>` | |
| `max_response_length` | `<…>` | |
| rollout `n` | `<…>` | samples per prompt |
| `train_batch_size` | `<…>` | |
| `ppo_mini_batch_size` | `<…>` | |
| `actor.optim.lr` | `<…>` | |
| `total_epochs` / `total_training_steps` | `<…>` | |

---

## Launch

```bash
ray stop --force

# <paste the exact command actually used, e.g. based on recipe/dapo/run_dapo_*.sh>
bash recipe/dapo/run_dapo_qwen3_30b_a3b_megatron.sh
```

> Tip: capture the exact env-var overrides the same way the mi300x doc does, so
> the command is copy-pasteable. After a successful run, paste the resolved
> command from the Ray/console log here.

---

## Results

| Run date | Steps | Eval (AIME / acc) | Notes |
| --- | --- | --- | --- |
| `<YYYY-MM-DD>` | `<…>` | `<…>` | `<pass/fail, throughput, issues>` |

Logs / checkpoints:
- console log: `<path>`
- checkpoints: `<path>`
- wandb / tensorboard: `<link or run id>`

---

## Issues & fixes

Record anything you hit and how you resolved it (OOM, mbridge/LoRA limits,
vllm patches needed, etc.), mirroring the format in
[`../mi300x/e2e_ppo_trainer_megatron-moe-expert-parallel.md`](../mi300x/e2e_ppo_trainer_megatron-moe-expert-parallel.md).
