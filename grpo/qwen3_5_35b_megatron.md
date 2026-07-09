# VERL Megatron Training (Qwen3.5-35B-A3B) on AMD MI355

End-to-end steps to run VERL GRPO training with the Megatron backend for
Qwen3.5-35B-A3B on AMD MI355 GPUs.

## 1. Pull the Docker image

```bash
docker pull amdagi/verl-dev:rocm7.14_torch2.12_vllm0.22.1_0708
```

## 2. Launch the container

```bash
docker run -it --name verl --device /dev/kfd --device /dev/dri \
  --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace/verl \
  amdagi/verl-dev:rocm7.14_torch2.12_vllm0.22.1_0708 \
  /bin/bash
```

## 3. Download the model

```bash
export HF_TOKEN=YOUR_HF_TOKEN
hf download Qwen/Qwen3.5-35B-A3B --local-dir $HOME/models/Qwen3.5-35B-A3B
```

## 4. Prepare the dataset

```bash
python3 examples/data_preprocess/geo3k.py --local_save_dir $HOME/data/geo3k
```

## 5. Apply vllm patch

```bash
cd /workspace/vllm
bash ~/verl-e2e-experiments/patches/apply.sh
```

## 6. Run training

```bash
export WANDB_API_KEY="YOUR_WANDB_API"
HF_MODEL_PATH=$HOME/models/Qwen3.5-35B-A3B \
bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh trainer.save_freq=-1
```

To run without wandb logging (console only):

```bash
HF_MODEL_PATH=$HOME/models/Qwen3.5-35B-A3B \
bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh trainer.save_freq=-1 trainer.logger=['console']
```

If enable aiter use:

```bash
VLLM_ROCM_USE_AITER=1 bash examples/grpo_trainer/run_qwen3_5_35b_megatron.sh
```

W&B run: https://wandb.ai/wei-cai/verl_grpo_qwen3_5_35b_geo3k?nw=nwuserweicai
