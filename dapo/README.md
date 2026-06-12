# DAPO experiments (Megatron + vLLM)

End-to-end DAPO training runs on the verl Megatron + vLLM backend. This doc
records the **exact data prep, config, and launch commands** so each run is
reproducible after a container/cluster rebuild.

DAPO = Decoupled clip + dynamic sampling policy optimization. Key pieces vs.
vanilla GRPO:
- **clip-higher** — asymmetric PPO clip (`clip_ratio_low` < `clip_ratio_high`).
- **dynamic sampling** — filter prompt groups whose rewards are all-equal
  (all-correct / all-wrong) so every batch carries gradient signal.
- **token-level policy loss** — loss aggregated over tokens, not sequences.
- **overlong reward shaping** — soft length penalty instead of a hard cut.

## Models

| Model | Script | Status |
| --- | --- | --- |
| Qwen3-30B-A3B (MoE) | `mi355_qwen3_30b_a3b_megatron_tp1ep8_infertp4.sh` | ✅ validated end-to-end |
| Qwen3.5-35B-A3B (MoE) | _TBD_ | _planned_ |

---

## Shared prerequisites: datasets

Run once before training. Paths below are what the launch commands expect —
override with `TRAIN_FILES` / `VAL_FILES` if you change them.

- train set: `~/data/DAPO-Math-17k/data/dapo-math-17k.parquet`
- eval set: `~/data/AIME-2024/data/aime-2024.parquet`

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

---

# Qwen3-30B-A3B

DAPO training run for `Qwen/Qwen3-30B-A3B` (MoE, 128 experts).
Script: `mi355_qwen3_30b_a3b_megatron_tp1ep8_infertp4.sh`.

## Environment

| | |
| --- | --- |
| Hardware | 8× GPU single node (MI350/MI355 class) |
| Backend | Megatron (`model_engine=megatron`) + vLLM rollout |
| verl | `/workspace/verl` in the `verl` container |
| vLLM patches | see [`../patches/README.md`](../patches/README.md) if applied |

## Model download — full weights

DAPO trains real weights (not a dummy model), so download the full checkpoint:

```bash
mkdir -p $HOME/models/Qwen/Qwen3-30B-A3B

hf download Qwen/Qwen3-30B-A3B \
  --local-dir $HOME/models/Qwen/Qwen3-30B-A3B
```

## Parallelism (validated)

| Component | Setting | Value |
| --- | --- | --- |
| Actor/ref TP | `ACTOR_TP` | 1 |
| Actor/ref PP | `ACTOR_PP` | 1 |
| Actor/ref EP | `ACTOR_EP` | 8 |
| Actor/ref ETP | `ACTOR_ETP` | 1 |
| Actor/ref CP | `ACTOR_CP` | 1 |
| Rollout TP | `INFER_TP` | 4 |
| Rollout EP | `GEN_MOE_EP` | 4 (= `INFER_TP`) |
| Rollout MoE TP | `GEN_MOE_TP` | 1 (trtllm-only; ignored by vLLM) |

Constraints to keep in mind on N GPUs:
- Megatron: `ETP × EP × PP` must divide world size (`1 × 8 × 1 = 8` ✓).
- Rollout: verl asserts `expert_parallel_size == rollout_TP × rollout_DP`, and
  rollout `data_parallel_size` defaults to **1** (not auto-derived), so
  `GEN_MOE_EP` must equal `INFER_TP`.

## Key hyperparameters

| Knob | Value |
| --- | --- |
| `adv_estimator` | `grpo` |
| `clip_ratio_low` / `clip_ratio_high` / `clip_ratio_c` | `0.2` / `0.28` / `10.0` |
| `loss_agg_mode` | `token-mean` |
| overlong shaping | `len=4096`, `penalty_factor=1.0` |
| `max_prompt_length` / `max_response_length` | `2048` / `8192` |
| `ppo_max_token_len_per_gpu` | `max_prompt + max_response` (= 10240) |
| rollout `n` | `8` |
| `train_batch_size` / `ppo_mini_batch_size` | `512` / `16` |
| `actor.optim.lr` | `1e-5` |
| `save_freq` | `-1` (disabled) |

## Launch

```bash
cd /workspace/verl-e2e-experiments/dapo

WANDB_MODE=offline \
EXPERIMENT_NAME=mi350_qwen3_30b_a3b_megatron \
MODEL_PATH=$HOME/models/Qwen/Qwen3-30B-A3B \
TRAIN_FILES=$HOME/data/DAPO-Math-17k/data/dapo-math-17k.parquet \
VAL_FILES=$HOME/data/AIME-2024/data/aime-2024.parquet \
ALL_OFFLOAD=False \
bash mi355_qwen3_30b_a3b_megatron_tp1ep8_infertp4.sh
```

`EXPERIMENT_NAME` is auto-suffixed with the parallelism, e.g.
`mi350_qwen3_30b_a3b_megatron_tp1ep8_infertp4`.

**wandb**: the script logs to `["console","wandb"]`. Either `wandb login` in the
container, or set `WANDB_MODE=offline` (as above), or override
`trainer.logger='["console"]'` — otherwise it crashes with
`No API key configured`.

## Validation status

Validated end-to-end in the `verl` container (8 GPUs): Megatron actor/ref init →
vLLM rollout generation → reward computation, no assertion errors, no OOM with
`ALL_OFFLOAD=False`.

## Issues & fixes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `world_size (8) not divisible by expert_tensor_model_pipeline size (16)` | ETP defaulted to TP=4 → `ETP×EP×PP=16` | set `expert_tensor_parallel_size` explicitly; default `ACTOR_ETP=1` |
| `expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size` | rollout `GEN_MOE_EP` ≠ `INFER_TP` (rollout DP=1) | default `GEN_MOE_EP=${INFER_TP}` |
| `wandb: No API key configured` | no wandb auth in container | `WANDB_MODE=offline` or `wandb login` |

---

# Qwen3.5-35B-A3B

_Planned — to be added._
