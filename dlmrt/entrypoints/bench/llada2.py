"""LLaDA2 benchmarks, dispatched by `dlmrt bench` when the checkpoint's model_type is llada2_moe.

step: fixed-context round latency and one prefill chunk. smoke: one chat generation. suite: the paper's generation
protocol (its SGLang client's prompt rendering) over a problem file: rows with chat `messages`
(GSM8K, MATH-500) go through the chat template as they are; HumanEval rows (`prompt`) become system + instruction +
fenced prompt; 4096 new tokens, greedy. Records carry the paper's field names (`raw`, `finish`,
`completion_tokens`, `fwd_denoise`, `fwd_total`, `wall_s`) for its isolated scorer; `--shard I/N` takes every N-th
problem (the paper's modulo partition) and a rerun with the same settings resumes after the rows already written.
Tensor / expert parallel: launch under `torchrun --nproc-per-node N`; every rank runs the same program, rank 0 reports
and writes.
"""

import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import torch

from ...distributed import parallel
from ...models.llada2 import ROLES, LladaConfig, LladaModel
from ...models.llada2.engine import LladaEngine
from ...quantization.precision import PRECISIONS
from .common import chat_messages, check_contexts, cuda_ms, quick_order, report, run_suite, tp_size


def decoder(args):
    """The paper's decoder for the checkpoint's role; --threshold and --edit-threshold override the role's, and
    --temperature-min / --temperature-decrement set Speed mode's step_linear confidence temperature."""
    base = ROLES[args.role]
    if args.threshold is not None:
        base = dataclasses.replace(base, prob_threshold=args.threshold)
    if args.edit_threshold is not None:
        base = dataclasses.replace(base, edit_threshold=args.edit_threshold)
    if (args.temperature_min is None) != (args.temperature_decrement is None):
        raise SystemExit("--temperature-min and --temperature-decrement set the step_linear schedule together")
    if args.temperature_min is not None:
        base = dataclasses.replace(base, confidence_temperature_min=args.temperature_min,
                                   confidence_temperature_step_decrement=args.temperature_decrement)
    return base


def load(args):
    """The (sharded) model and whether this process reports; under torchrun every rank runs the same program."""
    group = parallel.init()
    model = LladaModel.load(args.model, max_seq_len=args.max_seq_len, group=group,
                            precision=getattr(args, "precision", None))
    return model, group is None or group.rank == 0


def build(args):
    """The model, whether this process reports, and the engine for --numerics: dlmrt's kernels, or SGLang's (bitwise
    the paper's runs; BF16 on one GPU)."""
    if args.numerics == "sglang":
        from ...models.llada2.sglang_numerics import SglangLladaEngine, SglangLladaModel

        if args.precision == "nvfp4":
            raise SystemExit("--numerics sglang runs BF16 checkpoints")
        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            raise SystemExit("--numerics sglang runs on one GPU: launch it without torchrun")
        model = SglangLladaModel(args.model, max_seq_len=args.max_seq_len)
        return model, True, SglangLladaEngine(model, decoder(args), fused=not args.in_place)
    model, lead = load(args)
    return model, lead, LladaEngine(model, decoder(args), fused=not args.in_place)


def tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def render(tok, problem):
    """The paper's client prompt: the chat template with the generation prompt (SGLang then encodes the text)."""
    if problem.get("messages") and problem.get("assistant_prefill"):
        raise ValueError("assistant_prefill rows are not part of the LLaDA protocol")
    return tok.apply_chat_template(chat_messages(problem), tokenize=False, add_generation_prompt=True)


def step(args):
    # The prefill timed below covers [0, max_rows), which building the engine already writes.
    check_contexts(args.contexts, LladaConfig.load(args.model).block_size, args.max_seq_len)
    model, lead = load(args)
    engine = LladaEngine(model, decoder(args))
    c, b = model.config, engine.block
    generator = torch.Generator(device=model.device).manual_seed(0)
    result = {"model": "llada2_moe", "precision": model.precision, "round_ms": {}, "prefill_ms": {}}
    for ctx in args.contexts:
        end = ctx + b
        engine.tokens.copy_(torch.randint(0, c.vocab_size, (b,), device=model.device, generator=generator))
        engine.start.fill_(ctx)
        engine.end.fill_(end)
        graph = engine.graphs[engine.splits(end), "round"]

        def round_():
            engine.ctr[0] = 1
            engine.ctr[2] = 0
            graph.replay()

        result["round_ms"][ctx] = cuda_ms(round_)
        report(lead, ctx, "round", round(result["round_ms"][ctx], 3), "splits", engine.splits(end))
    ids = torch.randint(0, c.vocab_size, (model.max_rows,), device=model.device, generator=generator)
    result["prefill_ms"][model.max_rows] = cuda_ms(lambda: engine._prefill(ids, model.max_rows), reps=5)
    report(lead, "prefill", model.max_rows, "rows", round(result["prefill_ms"][model.max_rows], 3))
    if args.out and lead:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


def smoke(args):
    tok = tokenizer(args.model)
    model, lead = load(args)
    engine = LladaEngine(model, decoder(args))
    ids = tok.encode(render(tok, {"messages": [{"role": "user", "content": args.prompt}]}))
    for _ in range(2):
        result = engine.generate(ids, args.max_new_tokens)
    report(lead, tok.decode(result.token_ids, skip_special_tokens=True))
    report(lead, json.dumps({"tokens": len(result.token_ids), "tps": round(result.request_tps, 1),
                             "tpf": round(len(result.token_ids) / sum(result.denoise_steps), 3),
                             "denoise": result.denoise_steps, "refine": result.refine_steps,
                             "prefill_s": round(result.prefill_s, 4)}))
    if args.out and lead:
        Path(args.out).write_text(json.dumps({"token_ids": result.token_ids, "denoise": result.denoise_steps,
                                              "refine": result.refine_steps}) + "\n")


def suite(args):
    tok = tokenizer(args.model)
    model, lead, engine = build(args)
    eos = model.config.eos_token_id

    def solve(problem, max_new_tokens):
        ids = tok.encode(render(tok, problem))
        result = engine.generate(ids, max_new_tokens)
        denoise = sum(result.denoise_steps)
        return {"entry_point": problem.get("entry_point"), "raw": tok.decode(result.token_ids, skip_special_tokens=True),
                "finish": "stop" if result.token_ids[-1] == eos else "length",
                "completion_tokens": len(result.token_ids), "wall_s": result.elapsed_s,
                "n_blocks": len(result.denoise_steps), "fwd_denoise": denoise, "fwd_refine": sum(result.refine_steps),
                "fwd_total": sum(result.forwards), "fwd_nocommit": denoise, "prompt_tokens": len(ids),
                "prefill_s": result.prefill_s, "request_tps": result.request_tps, "denoise_steps": result.denoise_steps,
                "refine_steps": result.refine_steps, "output_ids": result.token_ids}

    run_suite(args, lead, {
        "backend": "dlmrt-llada2_moe", "model": args.model, "precision": model.precision, "numerics": args.numerics,
        "role": args.role, "decoder": decoder(args).__dict__, "fused": engine.fused,
        "prompt_format": "the paper's client prompt: chat template (messages, or system + instruction + fenced prompt), "
                         "text encoded as SGLang does",
        "tp": tp_size(model)}, solve)


def quick_ids(benchmark, n):
    """The paper's frozen TPS quick-v1 selection: one warmup and eight measured tasks."""
    order = quick_order(benchmark, n)
    return order[0], order[1:9]


def tps(args):
    """The paper's TPS quick check: serial, warmup excluded, each request timed from prompt rendering to the decoded
    text (as its LLaDA worker times engine.generate(render(prompt))); pooled TPS over non-length responses (None when
    there are none, and then no macro TPS)."""
    tok = tokenizer(args.model)
    model, lead, engine = build(args)
    eos = model.config.eos_token_id
    result = {"model": args.model, "precision": model.precision, "numerics": args.numerics, "role": args.role,
              "fused": engine.fused, "tp": tp_size(model), "benchmarks": {}}
    for name, path in (("gsm8k", args.gsm8k), ("humaneval", args.humaneval)):
        problems = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        warmup, measured = quick_ids(name, len(problems))
        rows = []
        for index in [warmup] + measured:
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = engine.generate(tok.encode(render(tok, problems[index])), args.max_new_tokens)
            text = tok.decode(out.token_ids, skip_special_tokens=True)
            seconds = time.perf_counter() - started
            rows.append({"task_id": problems[index]["task_id"], "tokens": len(out.token_ids), "seconds": seconds,
                         "finish": "stop" if out.token_ids[-1] == eos else "length", "chars": len(text),
                         "denoise": sum(out.denoise_steps), "forwards": sum(out.forwards)})
        clean = [r for r in rows[1:] if r["finish"] == "stop"]
        tokens, seconds = sum(r["tokens"] for r in clean), sum(r["seconds"] for r in clean)
        result["benchmarks"][name] = {"n": len(clean), "tokens": tokens, "seconds": seconds,
                                      "pooled_tps": tokens / seconds if seconds else None, "rows": rows}
        report(lead, name, json.dumps({k: v for k, v in result["benchmarks"][name].items() if k != "rows"}))
    rates = [b["pooled_tps"] for b in result["benchmarks"].values()]
    result["macro_tps"] = None if None in rates else statistics.mean(rates)
    report(lead, "macro", None if result["macro_tps"] is None else round(result["macro_tps"], 1))
    if args.out and lead:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


def add_decoder_arguments(p, prefix="llada2_moe only: "):
    p.add_argument("--temperature-min", type=float,
                   help=prefix + "step_linear confidence temperature, from one down to this minimum (Speed mode); "
                                 "with --temperature-decrement")
    p.add_argument("--temperature-decrement", type=float,
                   help=prefix + "the step_linear schedule's decrement per round; with --temperature-min")


NUMERICS_HELP = "sglang: SGLang's kernels and reduction orders, bitwise the paper's runs (BF16, one GPU, needs flashinfer)"


def add_engine_arguments(p, prefix=""):
    p.add_argument("--in-place", action="store_true", help=prefix + "refine every block in place (no fused rounds)")
    p.add_argument("--numerics", choices=("dlmrt", "sglang"), default="dlmrt", help=prefix + NUMERICS_HELP)


def add_arguments(sub):
    """After diffusion_gemma.add_arguments, whose --role (base / orl) step and smoke share."""
    for name in ("step", "smoke"):
        p = sub.choices[name]
        p.add_argument("--threshold", type=float, help="llada2_moe only: override the role's acceptance threshold")
        p.add_argument("--edit-threshold", type=float, help="llada2_moe only: override the role's edit threshold")
        add_decoder_arguments(p)
    p = sub.add_parser("tps", help="llada2_moe only: the paper's TPS quick check (GSM8K and HumanEval, 8 tasks each)")
    p.add_argument("--model", required=True)
    p.add_argument("--gsm8k", required=True)
    p.add_argument("--humaneval", required=True)
    p.add_argument("--out")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--role", choices=sorted(ROLES), default="base")
    p.add_argument("--threshold", type=float)
    p.add_argument("--edit-threshold", type=float)
    add_decoder_arguments(p, prefix="")
    p.add_argument("--precision", choices=PRECISIONS, help="bf16 or nvfp4 (an NVFP4 export); default the checkpoint's")
    add_engine_arguments(p)


COMMANDS = {"step": step, "smoke": smoke, "suite": suite, "tps": tps}
