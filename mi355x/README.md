# VERL Training on AMD MI355

End-to-end steps to run VERL GRPO training (Qwen3-8B) on AMD MI355 GPUs.

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
  -w /workspace/verl \
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
EXPERIMENT_NAME=mi355-qwen3_8b_grpo_vllm_fsdp bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh trainer.total_training_steps=500
```

## Known issue: OOM at step 106 during vLLM `wake_up`

With the stock `run_qwen3_8b_fsdp.sh` config, training ran cleanly for 105 steps (~10.5 h) and then crashed on step 106 inside `update_weights` → `rollout.resume(tags=["weights"])` → vLLM `engine.wake_up()`:

```
ray.exceptions.RayTaskError: ray::vLLMHttpServer.wake_up() ...
Exception: Call to wake_up method failed: Worker failed with error
'CUDA Error: out of memory at /workspace/vllm/csrc/cumem_allocator.cpp:151'
...
RuntimeError: CUDA Error: out of memory at /workspace/vllm/csrc/cumem_allocator.cpp:151
  File ".../vllm/device_allocator/cumem.py", line 234, in wake_up
    create_and_map(handle)
  File ".../vllm/device_allocator/cumem.py", line 59, in create_and_map
    python_create_and_map(*allocation_handle)
```

The actor's FSDP update itself succeeded — only the vLLM wake-up that follows (which remaps the rollout weights into HBM via the cumem allocator) failed.

Step-106 metrics right before the crash:
- `actor/perf/max_memory_allocated_gb: 56.87`
- `actor/perf/max_memory_reserved_gb: **264.36**` — within ~24 GB of the 288 GB HBM ceiling on MI355X.
- `actor_rollout_ref.actor.fsdp_config.param_offload=False`, `optimizer_offload=False`.
- `rollout.gpu_memory_utilization=0.6`, `rollout.tensor_model_parallel_size=2`.
- `ppo_max_token_len_per_gpu=24576`, `response_length/clip_ratio: 0.236`.

Why it survived 105 steps and died on 106: vLLM's cumem allocator bypasses the PyTorch caching allocator and asks the HIP driver for fresh physical pages. PyTorch's reserved high-water mark drifts upward over steps as variable-length batches force new larger blocks; at step 106 the residual free physical HBM dropped below what cumem needs to remap vLLM's weight tensors.

Mitigations to try (cheapest first):
- Lower `actor_rollout_ref.rollout.gpu_memory_utilization` to ~0.5.
- Drop `ppo_max_token_len_per_gpu` and `log_prob_max_token_len_per_gpu` from 24576 to 16384 to cap FSDP's reserved peak.
- Enable `actor_rollout_ref.actor.fsdp_config.param_offload=True` and `optimizer_offload=True` to free HBM during rollout.
- Raise rollout `tensor_model_parallel_size` from 2 to 4 to halve vLLM's per-GPU footprint.
