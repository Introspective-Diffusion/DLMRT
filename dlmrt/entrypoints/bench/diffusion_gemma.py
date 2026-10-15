"""DiffusionGemma benchmarks, dispatched by `dlmrt bench` when the checkpoint's model_type is diffusion_gemma.

step: fixed-context round/commit latency. smoke: one chat generation. humaneval: the paper's DiffusionGemma
protocol (its vLLM client's prompt: system message, instruction + fenced prompt through the chat template, 2048 new
tokens, per-request seed); generation only, scored separately with EvalPlus base tests. suite: the same protocol over the
paper's problem files (chat `messages` rows as they are, HumanEval `prompt` rows as above), records in the schema of
its vLLM clients for the paper's scorers.
Tensor/expert parallel: launch under `torchrun --nproc-per-node N`; every rank runs the same program, rank 0 reports.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from ...distributed import parallel
from ...models.diffusion_gemma import PRECISIONS, ROLES, GemmaConfig, GemmaModel
from ...models.diffusion_gemma.engine import GemmaEngine
from .common import aggregate, chat_messages, check_contexts, cuda_ms, evaluation_ids, report, run_suite, tp_size


def load(args):
    """The (sharded) model and whether this process reports."""
    group = parallel.init()
    model = GemmaModel.load(args.model, max_seq_len=args.max_seq_len, group=group, precision=args.precision)
    return model, group is None or group.rank == 0


def decoder(args):
    """The paper's decoder for the checkpoint's role; --entropy-bound overrides the role's bound."""
    base = ROLES[args.role]
    bound = base.entropy_bound if args.entropy_bound is None else args.entropy_bound
    return dataclasses.replace(base, entropy_bound=bound, seed=args.seed)


def chat_ids(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def step(args):
    check_contexts(args.contexts, GemmaConfig.load(args.model).block_size, args.max_seq_len)
    model, lead = load(args)
    engine = GemmaEngine(model, decoder(args))
    w = model.config.sliding_window
    result = {"model": "diffusion_gemma", "tp": tp_size(model), "round_ms": {}, "commit_ms": {}}
    # Distinct tokens, as in real rounds: identical rows would all route to the same 8 experts.
    generator = torch.Generator(device=model.device).manual_seed(0)
    for buffer in (engine.tokens, engine.canvas):
        buffer.copy_(torch.randint(0, model.config.vocab_size, buffer.shape, device=model.device, generator=generator))
    # Every replay starts from the same state: a round rewrites its block and the self-conditioning and temperature
    # state, and replays from the drifted state route to fewer experts than real rounds.
    state = [engine.tokens, engine.logits, engine.probs, engine.temp, engine.h_ref, engine.t_prev, engine.accepted,
             engine.remaining]
    initial = [t.clone() for t in state]
    for ctx in args.contexts:
        engine.start.fill_(ctx)
        engine.end.fill_(ctx + engine.block)
        engine.leftpad.fill_(max(0, ctx - (w - 1)))

        def round_():
            for t, t0 in zip(state, initial):
                t.copy_(t0)
            engine.step.fill_(1)
            engine.graphs["round"].replay()

        result["round_ms"][ctx] = cuda_ms(round_)
        result["commit_ms"][ctx] = cuda_ms(engine.graphs["commit"].replay)
        report(lead, ctx, "round", round(result["round_ms"][ctx], 3), "commit", round(result["commit_ms"][ctx], 3))
    if args.out and lead:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


def smoke(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model, lead = load(args)
    engine = GemmaEngine(model, decoder(args))
    ids = chat_ids(tokenizer, [{"role": "user", "content": args.prompt}])
    for _ in range(2):
        result = engine.generate(ids, args.max_new_tokens)
    report(lead, tokenizer.decode(result.token_ids, skip_special_tokens=True))
    report(lead, json.dumps({"tokens": len(result.token_ids), "tps": result.request_tps, "steps": result.denoise_steps,
                             "tokens_per_forward": len(result.token_ids) / sum(result.denoise_steps)}))
    if args.out and lead:
        Path(args.out).write_text(json.dumps({"token_ids": result.token_ids, "steps": result.denoise_steps}) + "\n")


def load_problems(path):
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if [r["task_id"] for r in rows] != [f"HumanEval/{i}" for i in range(164)]:
        raise ValueError("Expected all 164 HumanEval tasks in their original order")
    return rows


def humaneval(args):
    from transformers import AutoTokenizer

    problems = load_problems(args.inputs)
    warmup_id, measured_ids = evaluation_ids(args.full, args.count)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The engine's length check (the prompt plus whole canvases), for every request before the model loads and --out
    # exists.
    b = GemmaConfig.load(args.model).block_size
    for i in [warmup_id, *measured_ids]:
        n = len(chat_ids(tokenizer, chat_messages(problems[i])))
        if n + (args.max_new_tokens + b - 1) // b * b > args.max_seq_len:
            raise SystemExit(f"{problems[i]['task_id']}: {n} prompt tokens + --max-new-tokens {args.max_new_tokens}, in "
                             f"whole {b}-token canvases, exceed --max-seq-len {args.max_seq_len}")
    model, lead = load(args)
    tp = tp_size(model)
    if lead:
        args.out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "backend": "dlmrt-diffusion_gemma" + (f"-tp{tp}" if tp > 1 else ""), "model": args.model,
        "precision": model.precision,
        "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "warmup_id": warmup_id, "measured_ids": measured_ids, "max_new_tokens": args.max_new_tokens,
        "decoder": decoder(args).__dict__, "prompt_format": "DiffusionGemma paper: system + instruction + fenced prompt",
        "repeats": args.repeats, "scope": "full_humaneval" if args.full else "quick_v1", "scoring": "not_run",
        "timing": "tokenization/detokenization excluded",
    }
    if lead:
        (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    engine = GemmaEngine(model, decoder(args))
    eos = set(model.config.eos_token_ids)

    def run(index, kind, repeat):
        p = problems[index]
        messages = chat_messages(p)
        ids = chat_ids(tokenizer, messages)
        result = engine.generate(ids, args.max_new_tokens, seed=args.seed)
        count = len(result.token_ids)
        text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
        row = {
            "task_id": p["task_id"], "entry_point": p["entry_point"], "index": index, "kind": kind, "repeat": repeat,
            "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
            "elapsed_s": result.elapsed_s, "request_tps": result.request_tps, "completion_tokens": count,
            "prompt_tokens": len(ids), "finish_type": "stop" if result.token_ids[-1] in eos else "length",
            "denoise_forwards": sum(result.denoise_steps), "commit_forwards": result.commit_forwards,
            "filled_tokens": len(result.denoise_steps) * engine.block, "denoise_steps": result.denoise_steps,
            "prefill_s": result.prefill_s, "text": text, "output_ids": result.token_ids, "seed": args.seed,
        }
        # Field names of the paper's DiffusionGemma records, which its EvalPlus scorer binds.
        row.update(finish=row["finish_type"], raw=text, wall_s=result.elapsed_s, fwd_denoise=row["denoise_forwards"],
                   fwd_total=row["denoise_forwards"] + row["commit_forwards"])
        if lead:
            with (args.out / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        report(lead, json.dumps({k: row[k] for k in ("task_id", "kind", "completion_tokens", "elapsed_s", "request_tps",
                                                     "denoise_forwards", "finish_type")}))
        return row

    run(warmup_id, "warmup", -1)
    measured = [run(i, "measured", r) for r in range(args.repeats) for i in measured_ids]
    summary = {"all": aggregate(measured),
               "per_repeat": {str(r): aggregate([x for x in measured if x["repeat"] == r]) for r in range(args.repeats)}}
    if lead:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report(lead, json.dumps(summary["all"]))


def suite(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model, lead = load(args)
    engine = GemmaEngine(model, decoder(args))
    eos = set(model.config.eos_token_ids)

    def solve(problem, max_new_tokens):
        ids = chat_ids(tokenizer, chat_messages(problem))
        result = engine.generate(ids, max_new_tokens, seed=args.seed)
        denoise = sum(result.denoise_steps)
        return {"entry_point": problem.get("entry_point"),
                "raw": tokenizer.decode(result.token_ids, skip_special_tokens=True),
                "finish": "stop" if result.token_ids[-1] in eos else "length",
                "completion_tokens": len(result.token_ids), "wall_s": result.elapsed_s,
                "n_blocks": len(result.denoise_steps), "fwd_denoise": denoise, "fwd_commit": result.commit_forwards,
                "fwd_total": denoise + result.commit_forwards, "prompt_tokens": len(ids), "prefill_s": result.prefill_s,
                "request_tps": result.request_tps, "denoise_steps": result.denoise_steps,
                "output_ids": result.token_ids, "seed": args.seed}

    run_suite(args, lead, {
        "backend": "dlmrt-diffusion_gemma", "model": args.model, "precision": model.precision, "role": args.role,
        "decoder": decoder(args).__dict__, "seed": args.seed, "seed_protocol": "reset_each_request",
        "prompt_format": "the paper's vLLM client prompt through the chat template",
        "tp": tp_size(model)}, solve)


def add_arguments(sub):
    for name in ("step", "smoke", "humaneval"):
        p = sub.choices[name]
        p.add_argument("--role", choices=sorted(ROLES), default="base",
                       help="diffusion_gemma, and llada2_moe for step / smoke: the paper's decoder for a base or ORL "
                            "checkpoint")
        p.add_argument("--entropy-bound", type=float, help="diffusion_gemma only: override the role's bound")
        p.add_argument("--seed", type=int, default=1234, help="diffusion_gemma only: per-request canvas seed")
        p.add_argument("--precision", choices=PRECISIONS,
                       help="diffusion_gemma and llada2_moe: bf16 or nvfp4 (an NVFP4 export); default the checkpoint's")


COMMANDS = {"step": step, "smoke": smoke, "humaneval": humaneval, "suite": suite}
