"""Generate with dlmrt from Python: one GPU, or tensor parallel under torchrun.

    python examples/offline_inference.py --model google/diffusiongemma-26B-A4B-it
    torchrun --nproc-per-node 8 examples/offline_inference.py --model CKPT --role orl

Under torchrun every rank runs this same program; rank 0 prints.
"""

import argparse

import torch.distributed as dist

from dlmrt import LLM

PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip?",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="checkpoint directory or Hugging Face repo id")
    p.add_argument("--role", default="base", choices=["base", "orl"])
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    llm = LLM(args.model, role=args.role)
    lead = not dist.is_initialized() or dist.get_rank() == 0

    def on_block(tokens, steps, elapsed_s, frames):
        if lead:
            print(f"  block: {len(tokens)} tokens in {steps} denoise rounds ({elapsed_s:.3f} s)", flush=True)

    for prompt in PROMPTS:
        out = llm.generate(prompt, max_new_tokens=args.max_new_tokens, on_block=on_block)
        if lead:
            print(f"\n>>> {prompt}\n{out.text}\n[{len(out.token_ids)} tokens, {out.request_tps:.0f} tok/s, "
                  f"{len(out.token_ids) / sum(out.denoise_steps):.1f} tokens per denoise round]\n")


if __name__ == "__main__":
    main()
