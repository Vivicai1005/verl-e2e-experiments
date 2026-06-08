# Megatron MoE Expert-Parallel E2E (GSM8K) on AMD MI300X

End-to-end smoke tests for the verl Megatron + vLLM GRPO pipeline with MoE
expert parallelism, adapted from
`.github/workflows/e2e_ppo_trainer_megatron_vllm_2.yml` to run on 8× MI300X
(ROCm) instead of 8× L20 (CUDA).

## ROCm adaptations vs. the upstream CI commands

The CI jobs target NVIDIA L20 + NVIDIA Megatron-Bridge. On MI300X we change:

- `VANILLA_MBRIDGE=True VALUE_VANILLA_MBRIDGE=True` — use the installed
  `mbridge` package; the NVIDIA `megatron.bridge` package is **not** installed
  in the ROCm container (`ModuleNotFoundError: No module named 'megatron.bridge'`).
- `*.profiler.tool=null` overrides — no nsys/nvtx on ROCm.
- `ALL_OFFLOAD=False` — the dummy 2-layer model has tiny GPU footprint
  (~8 GB allocated), so CPU offload only floods host RAM and triggers the
  Linux OOM-killer on a DataLoader worker at shutdown. Disabling it keeps the
  run clean.

All three are single-step smoke tests (`TOTAL_TRAIN_STEPS` defaults to `1` in
`tests/special_e2e/run_ppo_trainer_megatron.sh`). They verify the pipeline
wires together end-to-end; they do not test convergence.

Common config: PP=2, TP=4, EP=4, ETP=1, CP=1; dummy model
`tests/special_e2e/ppo_trainer/expert_parallel/qwen2moe_minimal.json` derived
from `Qwen/Qwen3-30B-A3B-Instruct-2507`.

---

## Prerequisites: dataset and model downloads

Run these once before the tests. Defaults expected by
`tests/special_e2e/run_ppo_trainer_megatron.sh`:
- dataset: `~/data/gsm8k/{train,test}.parquet`
- base model: `~/models/Qwen/Qwen3-30B-A3B-Instruct-2507`
- reward model: `~/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B`

### Dataset (GSM8K)

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k
```

### Base model — config + tokenizer only

`USE_DUMMY_MODEL=True` builds a random-weight 2-layer model via
`scripts/init_random_model.py`, which only reads `AutoConfig` and
`AutoTokenizer` from the base model path — it does **not** load the 30B
weights. So download just the config/tokenizer files (a few MB), not the full
checkpoint:

```bash
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir $HOME/models/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --include "config.json" "generation_config.json" "tokenizer*" "*.json"
```

### Reward model — full weights

The reward model is served by vLLM (`reward.reward_model.enable=True`), so it
needs the full checkpoint:

```bash
hf download Skywork/Skywork-Reward-V2-Llama-3.2-1B \
  --local-dir $HOME/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B
```

---

## (1) 3D parallelism with mbridge — PASS

```bash
ray stop --force
ADV_ESTIMATOR=grpo USE_DUMMY_MODEL=True DUMMY_MODEL_CONFIG_PATH=tests/special_e2e/ppo_trainer/expert_parallel/qwen2moe_minimal.json \
  PPO_MAX_TOKEN_LEN=1024 FWD_MAX_TOKEN_LEN=1024 \
  MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=512 \
  MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507 USE_MBRIDGE=True VANILLA_MBRIDGE=True VALUE_VANILLA_MBRIDGE=True \
  COMMON_PP=2 COMMON_VPP=null COMMON_CP=1 COMMON_TP=4 COMMON_EP=4 COMMON_ETP=1 INFER_TP=8 \
  USE_DIST_CKPT=False ALL_OFFLOAD=False SKIP_SAVE_HF_MODEL=1 \
  bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  global_profiler.tool=null actor_rollout_ref.actor.profiler.tool=null actor_rollout_ref.ref.profiler.tool=null actor_rollout_ref.rollout.profiler.tool=null critic.profiler.tool=null
```

Result: step 1 completes, full metrics printed, exit code `0`.

---

## (2) 3D parallelism with FP8 rollout + mbridge — PASS

Same as (1) but rollout served in FP8 (`ROLLOUT_QUANTIZATION=fp8`) and rollout
tensor parallel reduced to `INFER_TP=2`.

```bash
ray stop --force
ADV_ESTIMATOR=grpo USE_DUMMY_MODEL=True DUMMY_MODEL_CONFIG_PATH=tests/special_e2e/ppo_trainer/expert_parallel/qwen2moe_minimal.json \
  PPO_MAX_TOKEN_LEN=1024 FWD_MAX_TOKEN_LEN=1024 \
  MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=512 \
  MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507 USE_MBRIDGE=True VANILLA_MBRIDGE=True VALUE_VANILLA_MBRIDGE=True \
  COMMON_PP=2 COMMON_VPP=null COMMON_CP=1 COMMON_TP=4 COMMON_EP=4 COMMON_ETP=1 INFER_TP=2 \
  USE_DIST_CKPT=False ALL_OFFLOAD=False SKIP_SAVE_HF_MODEL=1 ROLLOUT_QUANTIZATION=fp8 \
  bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  global_profiler.tool=null actor_rollout_ref.actor.profiler.tool=null actor_rollout_ref.ref.profiler.tool=null actor_rollout_ref.rollout.profiler.tool=null critic.profiler.tool=null
```

Result: step 1 completes, full metrics printed, exit code `0`.

---

## (3) 3D parallelism with mbridge LoRA — BLOCKED

```bash
ray stop --force
ADV_ESTIMATOR=grpo USE_DUMMY_MODEL=True DUMMY_MODEL_CONFIG_PATH=tests/special_e2e/ppo_trainer/expert_parallel/qwen2moe_minimal.json \
  PPO_MAX_TOKEN_LEN=1024 FWD_MAX_TOKEN_LEN=1024 \
  MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=512 LORA_RANK=8 CRITIC_LORA_RANK=8 \
  MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507 USE_MBRIDGE=True VANILLA_MBRIDGE=True VALUE_VANILLA_MBRIDGE=True \
  COMMON_PP=2 COMMON_VPP=null COMMON_CP=1 COMMON_TP=4 COMMON_EP=2 COMMON_ETP=1 INFER_TP=8 \
  USE_DIST_CKPT=False LORA_MERGE=True ALL_OFFLOAD=False SKIP_SAVE_HF_MODEL=1 \
  bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  global_profiler.tool=null actor_rollout_ref.actor.profiler.tool=null actor_rollout_ref.ref.profiler.tool=null actor_rollout_ref.rollout.profiler.tool=null critic.profiler.tool=null
```

Result:

```
Error: LoRA/PEFT only supported via Megatron-Bridge
```

### Why it fails

LoRA on the Megatron backend is implemented **only** through NVIDIA
Megatron-Bridge, not the vanilla `mbridge` package. From
`verl/workers/config/megatron_peft.py:26`:

```python
assert bridge is not None and provider is not None, "LoRA/PEFT only supported via Megatron-Bridge"
from megatron.bridge.peft.utils import create_peft
```

With `VANILLA_MBRIDGE=True`, `bridge`/`provider` are `None`, so the assertion
fails. Even if it passed, `megatron.bridge.peft.utils` is not installed in the
ROCm container.

So this test cannot run with vanilla mbridge.

(LoRA-specific args kept from the CI command: `LORA_RANK=8`,
`CRITIC_LORA_RANK=8`, `LORA_MERGE=True`, and `COMMON_EP=2`.)
