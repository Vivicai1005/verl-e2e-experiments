# VERL Training on AMD MI300

End-to-end steps to run VERL GRPO training (Qwen3-8B) on AMD MI300 GPUs.

## 1. Pull the Docker image

```bash
docker pull amdagi/training_ubuntu_rocm7.0.2_56_py312:verl_te2.10_vllm0.20_gfx942_950
```

## 2. Launch the container

```bash
docker run -it --name verl --device /dev/kfd --device /dev/dri \
  --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace \
  amdagi/training_ubuntu_rocm7.0.2_56_py312:verl_te2.10_vllm0.20_gfx942_950 \
  /bin/bash
```

## 3. Download the model

```bash
python3 -c "import transformers; transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B')"
```

## 4. Prepare the datasets

```bash
python3 examples/data_preprocess/gsm8k.py --local_dir $HOME/data/gsm8k
python3 examples/data_preprocess/math_dataset.py --local_dir $HOME/data/math
```

## 5. Run training

```bash
export WANDB_API_KEY="YOUR_WANDB_API"
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

To cap the run at 500 steps and enable FSDP param/optimizer offload (works around the OOM described below) without editing the script:

```bash
EXPERIMENT_NAME=mi300x-qwen3_8b_grpo_vllm_fsdp bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  trainer.total_training_steps=500 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

## Known issue: OOM at step 4

With the stock `run_qwen3_8b_fsdp.sh` config, training completes steps 1–3 and then crashes during `update_actor` on step 4 with a HIP OOM:

```
:0:rocdevice.cpp :3672: Callback: Queue ... Aborting with error :
HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources.
... Available Free mem : 62 MB
*** SIGABRT received ***
```

Ray then tears the job down with `ActorDiedError` on the worker that hit the OOM.

Context from the run:
- 8× MI300 (192 GB), colocated vLLM rollout + FSDP actor/ref on the same GPUs.
- `actor/perf/max_memory_reserved_gb` holds ~179 GB between steps; `gpu_memory_utilization=0.6` reserves another large slab for vLLM KV cache.
- ROCm logs `expandable_segments not supported on this platform`, so allocator fragmentation accumulates across steps until step 4 can no longer fit the actor update.
- `save_freq=20` means no checkpoint exists yet when the crash happens — the run is lost.
