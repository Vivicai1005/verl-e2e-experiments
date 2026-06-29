"""
Inspect vLLM rollout outputs for geo3k at the SAME parallelism/sampling as
examples/grpo_trainer/run_qwen3_5_35b_megatron.sh, and score each generation
with the geo3k reward function.
"""
import io
import re
import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams

from verl.utils.reward_score import geo3k
from transformers import AutoProcessor

MODEL = "/root/models/Qwen3.5-35B-A3B"
PARQUET = "/root/data/geo3k/train.parquet"
N_PROMPTS = 4


def build_inputs(processor, row):
    """Replicate verl rl_dataset multimodal prompt building."""
    msg = row["prompt"][0]  # {'role':'user','content':'<image>Find x...'}
    content = msg["content"]
    pil = Image.open(io.BytesIO(row["images"][0]["bytes"])).convert("RGB")

    parts = []
    for seg in re.split(r"(<image>)", content):
        if seg == "<image>":
            parts.append({"type": "image", "image": pil})
        elif seg:
            parts.append({"type": "text", "text": seg})
    messages = [{"role": "user", "content": parts}]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return text, pil


def main():
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    df = pd.read_parquet(PARQUET).head(N_PROMPTS)
    prompts, gts = [], []
    for _, row in df.iterrows():
        text, pil = build_inputs(processor, row)
        prompts.append({"prompt": text, "multi_modal_data": {"image": pil}})
        gts.append(row["reward_model"]["ground_truth"])

    print("=" * 80)
    print("TEMPLATED PROMPT TAIL (last 200 chars):")
    print(repr(prompts[0]["prompt"][-200:]))
    print("=" * 80, flush=True)

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.6,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=3072,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )

    def report(tag, sp):
        print("\n" + "#" * 80 + f"\n# {tag}\n" + "#" * 80, flush=True)
        outs = llm.generate(prompts, sp)
        for i, out in enumerate(outs):
            gt = gts[i]
            print(f"\n----- prompt {i}  (ground_truth={gt!r}) -----")
            for j, comp in enumerate(out.outputs):
                resp = comp.text
                box = geo3k.extract_boxed_content(resp)
                fmt = geo3k.format_reward(resp)
                acc = geo3k.acc_reward(resp, gt)
                score = geo3k.compute_score(resp, gt)
                starts_think = resp.lstrip().startswith("<think>")
                has_box = "\\boxed{" in resp
                print(f"  [n{j}] score={score:.2f} fmt={fmt} acc={acc} "
                      f"starts_<think>={starts_think} has_\\boxed={has_box} len={len(resp)}")
                print(f"        boxed_extracted={box!r}")
                print(f"        resp_head={resp[:160]!r}")
                print(f"        resp_tail={resp[-160:]!r}", flush=True)

    report("TRAINING SAMPLING  temp=1.0 top_p=1.0 top_k=-1 n=5",
           SamplingParams(n=5, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=2048))
    report("GREEDY  temp=0 n=1",
           SamplingParams(n=1, temperature=0.0, max_tokens=2048))
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
