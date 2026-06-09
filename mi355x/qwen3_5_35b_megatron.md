# VERL Megatron Training (Qwen3.5-35B-A3B) on AMD MI355

End-to-end steps to run VERL GRPO training with the Megatron backend for
Qwen3.5-35B-A3B on AMD MI355 GPUs.

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
export HF_TOKEN=YOUR_HF_TOKEN
HF_MODEL_PATH=$HOME/models/Qwen/Qwen3.5-35B-A3B
hf download Qwen/Qwen3.5-35B-A3B --local-dir $HOME/models/Qwen/Qwen3.5-35B-A3B
```

## 4. Prepare the dataset

```bash
python3 examples/data_preprocess/geo3k.py --local_save_dir $HOME/data/geo3k
```

## 5. Run training

```bash
export WANDB_API_KEY="YOUR_WANDB_API"
HF_MODEL_PATH=$HOME/models/Qwen/Qwen3.5-35B-A3B \
bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

W&B run: https://wandb.ai/wei-cai/verl_grpo_qwen3_5_35b_geo3k?nw=nwuserweicai
