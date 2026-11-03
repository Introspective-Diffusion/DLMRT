"""DiffusionGemma benchmarks, dispatched by `dlmrt bench` when the checkpoint's model_type is diffusion_gemma.

step: fixed-context round/commit latency. smoke: one chat generation. humaneval: the paper's DiffusionGemma
protocol (its vLLM client's prompt: system message, instruction + fenced prompt through the chat template, 2048 new
tokens, per-request seed); generation only, scored separately with EvalPlus base tests. suite: the same protocol over the
paper's problem files (chat `messages` rows as they are, HumanEval `prompt` rows as above), records in the schema of
its vLLM clients for the paper's scorers.
Tensor/expert parallel: launch under `torchrun --nproc-per-node N`; every rank runs the same program, rank 0 reports.

throughput: the batched engine (one GPU) in closed loop over a problem file: at each concurrency level N, N requests
in flight until every request of the level has finished. identity: the batched engine's tokens and denoise steps for
one request at a time against GemmaEngine's, and their batch-1 speed.
"""

import argparse
import dataclasses
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from ...distributed import parallel
from ...models.diffusion_gemma import PRECISIONS, ROLES, GemmaConfig, GemmaModel
from ...models.diffusion_gemma.batched import LIMIT, BatchedGemmaEngine, LoadPolicy, sizes
from ...models.diffusion_gemma.config import STOP_RULES
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


def problem_ids(tokenizer, path, count, skip=0):
    """The first `count` rows of a problem file from row `skip`, and their prompts through the chat template."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()][skip:skip + count]
    if len(rows) < count:
        raise SystemExit(f"{path}: {count} problems from row {skip} needed, {len(rows)} there")
    return rows, [chat_ids(tokenizer, chat_messages(r)) for r in rows]


def one_gpu(command):
    if parallel.init() is not None:
        raise SystemExit(f"dlmrt bench {command} runs the batched engine on one GPU (not under torchrun)")


def throughput(args):
    from transformers import AutoTokenizer

    one_gpu("throughput")
    levels = [n for n, _ in args.phases] if args.phases else args.levels
    max_batch = args.max_batch or max(levels)
    blocks = tuple(sorted({b for _, b in args.tiers}, reverse=True))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = BatchedGemmaEngine.load(args.model, decoder(args), max_batch, max_seq_len=args.max_seq_len,
                                     precision=args.precision, blocks=blocks, stops=args.stops,
                                     policy=LoadPolicy(args.tiers), fused=args.fused, lane_share=args.lane_share,
                                     prefill=args.prefill)
    eos = set(engine.model.config.eos_token_ids)
    settings = {"model": args.model, "precision": engine.model.precision, "decoder": engine.decoder.__dict__,
                "blocks": engine.blocks, "stops": engine.stops, "tiers": args.tiers, "fused": engine.fused,
                "lane_share": args.lane_share, "prefill": args.prefill, "max_batch": max_batch,
                "max_rows": engine.max_rows, "max_new_tokens": args.max_new_tokens}
    report(True, json.dumps(settings))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    def measure(name, concurrency, count):
        rows, ids = problem_ids(tokenizer, args.problems, count, args.skip)
        engine.run(ids[:1], 64, concurrency=1)  # every launch graph is built at start; this warms the host path
        torch.cuda.synchronize()
        done_s = {}
        started = time.perf_counter()
        results = engine.run(ids, args.max_new_tokens, concurrency=concurrency,
                             on_finish=lambda r: done_s.__setitem__(r.rid, time.perf_counter() - started))
        wall = time.perf_counter() - started
        tokens = sum(len(r.token_ids) for r in results)
        trace = engine.trace
        forwards = max(1, sum(t["forwards"] for t in trace))
        graph_s = sum(t["graph_s"] for t in trace)
        canvases = Counter(b for r in results for b in r.blocks)
        firsts = [engine.first_canvas_s(r) for r in results]
        summary = {
            "requests": len(results), "tokens": tokens, "wall_s": wall, "tps": tokens / wall,
            "request_tps_median": statistics.median(r.request_tps for r in results),
            "tokens_per_forward": tokens / max(1, sum(sum(r.denoise_steps) for r in results)),
            "first_canvas_s_median": statistics.median(firsts), "launches": len(trace), "forwards": forwards,
            "ms_per_forward": 1e3 * graph_s / forwards, "host_s": wall - graph_s,
            "canvases": {str(b): k / max(1, canvases.total()) for b, k in sorted(canvases.items(), reverse=True)},
            "slots": engine.mode_shares(), "lane_commits": engine.lane_commits(),
            "nan_positions": engine.nan_positions(),
        }
        if args.out:
            (args.out / name).mkdir()
            with (args.out / name / "requests.jsonl").open("w") as stream:
                for r, row, first in zip(results, rows, firsts):
                    record = {
                        "task_id": row.get("task_id", r.rid), "kind": "measured", "repeat": 0, "index": args.skip + r.rid,
                        "raw": tokenizer.decode(r.token_ids, skip_special_tokens=True),
                        "finish": "stop" if r.token_ids and r.token_ids[-1] in eos else "length",
                        "completion_tokens": len(r.token_ids), "elapsed_s": r.elapsed_s, "done_s": done_s[r.rid],
                        "first_canvas_s": first, "blocks": r.blocks, "denoise_steps": r.denoise_steps,
                        "output_ids": r.token_ids}
                    record.update({key: row[key] for key in ("gold", "meta") if key in row})
                    stream.write(json.dumps(record) + "\n")
        return results, summary

    out = []
    if args.phases:
        results, summary = measure("phased", args.phases, sum(k for _, k in args.phases))
        phase = [i for i, (_, k) in enumerate(args.phases) for _ in range(k)]
        summary["phases"] = [{"concurrency": n, "requests": k, "request_tps_median": statistics.median(
            r.request_tps for r, j in zip(results, phase) if j == i)} for i, (n, k) in enumerate(args.phases)]
        out.append(summary)
        report(True, json.dumps(summary))
    for level in [] if args.phases else levels:
        _, summary = measure(f"c{level}", level, max(args.count, 4 * level))
        out.append({"concurrency": level, **summary})
        report(True, json.dumps(out[-1]))
    if args.out:
        (args.out / "summary.json").write_text(json.dumps(out, indent=2) + "\n")


def identity(args):
    from transformers import AutoTokenizer

    one_gpu("identity")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.problems:
        _, prompts = problem_ids(tokenizer, args.problems, args.count)
    else:
        prompts = [chat_ids(tokenizer, [{"role": "user", "content": args.prompt}])]
    block = args.canvas_length or GemmaConfig.load(args.model).block_size
    _, _, buckets = sizes((block,), 1)
    model = GemmaModel.load(args.model, args.max_seq_len, precision=args.precision, canvas_length=block, slots=2,
                            max_rows=buckets[-1], row_buckets=buckets)
    d = dataclasses.replace(decoder(args), stop_rule=args.stop_rule)
    solo, batched = GemmaEngine(model, d), BatchedGemmaEngine(model, d, 1)
    same, ratios = 0, []
    for i, ids in enumerate(prompts):
        a = [solo.generate(ids, args.max_new_tokens, stop_on_eos=not args.ignore_eos) for _ in range(args.repeats)]
        b = [batched.run([ids], args.max_new_tokens, stop_on_eos=not args.ignore_eos)[0] for _ in range(args.repeats)]
        ok = a[0].token_ids == b[0].token_ids and a[0].denoise_steps == b[0].denoise_steps
        same += ok
        ratios.append(min(x.elapsed_s for x in a) / min(x.elapsed_s for x in b))
        report(True, json.dumps({"prompt": i, "identical": ok, "tokens": [len(a[0].token_ids), len(b[0].token_ids)],
                                 "steps": [sum(a[0].denoise_steps), sum(b[0].denoise_steps)], "speed": ratios[-1]}))
    report(True, json.dumps({"canvas_length": block, "stop_rule": d.stop_rule, "prompts": len(prompts),
                             "identical": same, "speed": statistics.mean(ratios)}))
    if same < len(prompts):
        raise SystemExit(1)


def tiers(value):
    """--tiers 2:256,11:128,0:32 as LoadPolicy tiers (0: no bound)."""
    try:
        out = tuple((int(n) or LIMIT, int(b)) for n, b in (x.split(":") for x in value.split(",")))
        if out and all(b > 0 and not b & (b - 1) for _, b in out) and out[-1][0] == LIMIT and len({b for _, b in out}) <= 4:
            return out
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected MOST:LENGTH,...,0:LENGTH (up to four power-of-two lengths), got {value!r}")


def stop_rules(value):
    """--stops 32:freeze_argmax,... as {canvas length: stop rule}."""
    try:
        out = {int(b): s for b, s in (x.split(":") for x in value.split(","))} if value else {}
        if set(out.values()) <= set(STOP_RULES):
            return out
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected LENGTH:RULE,... with rules among {STOP_RULES}, got {value!r}")


def phases(value):
    """--phases 1:4,4:16,... as [(concurrency, requests), ...]."""
    try:
        out = [tuple(int(v) for v in x.split(":")) for x in value.split(",")]
        if out and all(len(x) == 2 and min(x) > 0 for x in out):
            return out
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected CONCURRENCY:REQUESTS,..., got {value!r}")


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
    t = sub.add_parser("throughput", help="diffusion_gemma only: the batched engine in closed loop at each concurrency")
    i = sub.add_parser("identity", help="diffusion_gemma only: the batched engine against GemmaEngine at batch 1")
    for p in (t, i):
        p.add_argument("--model", required=True)
        p.add_argument("--role", choices=sorted(ROLES), default="base", help="the paper's decoder for a base or ORL checkpoint")
        p.add_argument("--entropy-bound", type=float, help="override the role's bound")
        p.add_argument("--seed", type=int, default=1234, help="per-request canvas seed")
        p.add_argument("--precision", choices=PRECISIONS, help="bf16 or nvfp4 (an NVFP4 export); default the checkpoint's")
        p.add_argument("--max-new-tokens", type=int, default=4096)
        p.add_argument("--max-seq-len", type=int, default=4608,
                       help="KV positions per request: the KV cache holds max_batch + 1 of them")
    t.add_argument("--problems", type=Path, required=True,
                   help="problem file (JSON lines: chat `messages` or a HumanEval `prompt`), e.g. the paper's gsm8k.jsonl")
    t.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    t.add_argument("--count", type=int, default=128, help="requests per level: max(count, 4 x the level)")
    t.add_argument("--skip", type=int, default=0, help="start at this row of the problem file")
    t.add_argument("--phases", type=phases,
                   help="instead of --levels: one run whose concurrency changes, CONCURRENCY:REQUESTS,... in prompt order")
    t.add_argument("--max-batch", type=int, help="request slots (default the largest level)")
    t.add_argument("--tiers", type=tiers, default=tiers("2:256,11:128,0:32"),
                   help="canvas length by load: up to MOST requests, LENGTH (0: no bound); one tier is a fixed length")
    t.add_argument("--stops", type=stop_rules, default=stop_rules("32:freeze_argmax"),
                   help="stop rule per canvas length (others: the role's, renoise)")
    t.add_argument("--no-fused", dest="fused", action="store_false",
                   help="commit every canvas in its own rows (no commit lanes)")
    t.add_argument("--lane-share", type=float, help="fixed lane claims per request and forward (default measured)")
    t.add_argument("--prefill", type=int, help="rows of a prompt chunk (default the canvas length)")
    t.add_argument("--out", type=Path, help="settings.json, summary.json and <level>/requests.jsonl")
    i.add_argument("--problems", type=Path, help="problem file (default one chat prompt, --prompt)")
    i.add_argument("--prompt", default="Write a Python function that checks whether a number is prime.")
    i.add_argument("--count", type=int, default=3, help="problems from the file")
    i.add_argument("--canvas-length", type=int, help="default the checkpoint's canvas_length")
    i.add_argument("--stop-rule", choices=STOP_RULES, default="renoise")
    i.add_argument("--repeats", type=int, default=1, help="runs per prompt and engine; the speed takes the fastest")
    i.add_argument("--ignore-eos", action="store_true", help="generate --max-new-tokens tokens (for the speed)")


COMMANDS = {"step": step, "smoke": smoke, "humaneval": humaneval, "suite": suite, "throughput": throughput,
            "identity": identity}
