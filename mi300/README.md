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
