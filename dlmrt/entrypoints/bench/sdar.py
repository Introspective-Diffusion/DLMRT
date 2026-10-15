"""SDAR benchmarks (see dlmrt.entrypoints.bench): fixed-context round latency, a smoke generation, the paper's
HumanEval protocol (the paper's fixed <think> wrapper, entropy bound 0.6, generation only), and the paper's QwenDLM
suite over problem files (`prompt` rows in the same wrapper, the role's decoder).
"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path


from ...distributed import parallel
from ...models.sdar import ROLES, DecodeConfig, Engine, Model, ModelConfig, render_prompt
from .common import aggregate, check_contexts, cuda_ms, evaluation_ids, report, run_suite, tp_size


def load(args, max_seq_len):
    """The (sharded) model and whether this process reports."""
    group = parallel.init()
    model = Model.load(args.model, max_seq_len=max_seq_len, fp8=getattr(args, "fp8", False), group=group)
    return model, group is None or group.rank == 0


def load_inputs(path):
    import pyarrow.parquet as pq

    rows = []
    for i, item in enumerate(pq.read_table(path).to_pylist()):
        prompt = item["prompt"] if isinstance(item["prompt"], str) else item["prompt"][0]["content"]
        rows.append({"task_id": f"HumanEval/{i}", "prompt": prompt})
    if len(rows) != 164:
        raise ValueError("Expected all 164 HumanEval tasks in their original order")
    return rows


def fp8_projections(value):
    return True if value == "all" else frozenset(value.split(","))


def fp8_label(value):
    return "" if not value else "-fp8" if value is True else "-fp8-" + "-".join(sorted(value))


def step(args):
    check_contexts(args.contexts, ModelConfig.load(args.model).block_size, args.max_seq_len)
    model, lead = load(args, args.max_seq_len)
    engine = Engine(model)
    b = engine.block
    tp = tp_size(model)
    result = {"fp8": fp8_label(args.fp8), "tp": tp, "round_ms": {}, "commit_ms": {}}
    for ctx in args.contexts:
        end = ctx + b
        engine.start.fill_(ctx)
        engine.end.fill_(end)
        splits = engine.splits(end)

        def round_():
            engine.step.fill_(1)
            engine.graphs[splits, "round"].replay()

        result["round_ms"][ctx] = cuda_ms(round_)
        result["commit_ms"][ctx] = cuda_ms(engine.graphs[splits, "commit"].replay)
        report(lead, ctx, "round", round(result["round_ms"][ctx], 3), "commit", round(result["commit_ms"][ctx], 3))
    if args.out and lead:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


def smoke(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model, lead = load(args, args.max_seq_len)
    engine = Engine(model, DecodeConfig(entropy_bound=0.6))
    ids = tokenizer.encode(render_prompt(args.prompt), add_special_tokens=False)
    for _ in range(2):
        result = engine.generate(ids, args.max_new_tokens)
    report(lead, tokenizer.decode(result.token_ids, skip_special_tokens=True))
    report(lead, json.dumps({"tokens": len(result.token_ids), "tps": result.request_tps,
                             "steps": result.denoise_steps}))
    if args.out and lead:
        Path(args.out).write_text(json.dumps({"token_ids": result.token_ids, "steps": result.denoise_steps}) + "\n")


def humaneval(args):
    from transformers import AutoTokenizer

    source = load_inputs(args.inputs)
    warmup_id, measured_ids = evaluation_ids(args.full, args.count)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # The engine's length check (a prompt of at least one block is prefilled, a shorter one shares the first block;
    # whole blocks), for every request before the model loads and --out exists.
    b = ModelConfig.load(args.model).block_size
    for i in [warmup_id, *measured_ids]:
        n = len(tokenizer.encode(render_prompt(source[i]["prompt"]), add_special_tokens=False))
        prefix = n if n >= b else 0
        if prefix + (n - prefix + args.max_new_tokens + b - 1) // b * b > args.max_seq_len:
            raise SystemExit(f"{source[i]['task_id']}: {n} prompt tokens + --max-new-tokens {args.max_new_tokens}, in "
                             f"whole {b}-token blocks, exceed --max-seq-len {args.max_seq_len}")
    model, lead = load(args, args.max_seq_len)
    tp = tp_size(model)
    if lead:
        args.out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "backend": "dlmrt" + fp8_label(args.fp8) + (f"-tp{tp}" if tp > 1 else ""),
        "model": args.model,
        "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "warmup_id": warmup_id, "measured_ids": measured_ids, "max_new_tokens": args.max_new_tokens,
        "entropy_bound": 0.6, "randtok_seed": args.randtok_seed,
        "prompt_format": "paper fixed user/assistant/<think> wrapper", "repeats": args.repeats,
        "scope": "full_humaneval" if args.full else "quick_v1", "scoring": "not_run",
        "timing": "tokenization/detokenization excluded",
    }
    if lead:
        (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    engine = Engine(model, DecodeConfig(entropy_bound=0.6, randtok_seed=args.randtok_seed))
    eos = model.config.eos_token_id

    def run(index, kind, repeat):
        prompt = render_prompt(source[index]["prompt"])
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        result = engine.generate(ids, args.max_new_tokens)
        count = len(result.token_ids)
        row = {
            "task_id": source[index]["task_id"], "index": index, "kind": kind, "repeat": repeat,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "elapsed_s": result.elapsed_s, "request_tps": result.request_tps, "completion_tokens": count,
            "prompt_tokens": len(ids), "finish_type": "stop" if result.token_ids[-1] == eos else "length",
            "denoise_forwards": sum(result.denoise_steps), "commit_forwards": result.commit_forwards,
            "filled_tokens": len(result.denoise_steps) * engine.block - (len(ids) if len(ids) < engine.block else 0),
            "denoise_steps": result.denoise_steps, "prefill_s": result.prefill_s,
            "text": tokenizer.decode(result.token_ids, skip_special_tokens=True), "output_ids": result.token_ids,
        }
        if lead:
            with (args.out / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        report(lead, json.dumps({k: row[k] for k in ("task_id", "kind", "completion_tokens", "elapsed_s",
                                                     "request_tps", "denoise_forwards", "finish_type")}))
        return row

    run(warmup_id, "warmup", -1)
    measured = [run(i, "measured", r) for r in range(args.repeats) for i in measured_ids]
    summary = {
        "all": aggregate(measured),
        "per_repeat": {str(r): aggregate([x for x in measured if x["repeat"] == r]) for r in range(args.repeats)},
    }
    if lead:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report(lead, json.dumps(summary["all"]))


def decoder(args):
    """The paper's decoder for the checkpoint's role; --threshold (Base only; also its edit threshold),
    --entropy-bound and --edit-threshold override the role's."""
    base = ROLES[args.role]
    changes = {"randtok_seed": args.randtok_seed}
    if args.threshold is not None:
        if base.accept_rule != "prob_threshold":
            raise SystemExit("--threshold sets Base's probability threshold; --role orl accepts by --entropy-bound")
        changes["prob_threshold"] = changes["edit_threshold"] = args.threshold
    if args.entropy_bound is not None:
        changes["entropy_bound"] = args.entropy_bound
    if args.edit_threshold is not None:
        changes["edit_threshold"] = args.edit_threshold
    return dataclasses.replace(base, **changes)


def suite(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.numerics == "sglang":
        from ...models.sdar.sglang_numerics import SglangSdarEngine, SglangSdarModel

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            raise SystemExit("--numerics sglang runs on one GPU: launch it without torchrun")
        model, lead = SglangSdarModel(args.model, max_seq_len=args.max_seq_len), True
        engine = SglangSdarEngine(model, decoder(args))
    else:
        model, lead = load(args, args.max_seq_len)
        engine = Engine(model, decoder(args))
    eos = model.config.eos_token_id

    def solve(problem, max_new_tokens):
        ids = tokenizer.encode(render_prompt(problem["prompt"]), add_special_tokens=False)
        result = engine.generate(ids, max_new_tokens)
        denoise = sum(result.denoise_steps)
        return {"raw": tokenizer.decode(result.token_ids, skip_special_tokens=True),
                "finish": "stop" if result.token_ids[-1] == eos else "length",
                "completion_tokens": len(result.token_ids), "wall_s": result.elapsed_s,
                "n_blocks": result.commit_forwards, "fwd_denoise": denoise, "fwd_commit": result.commit_forwards,
                "fwd_total": denoise + result.commit_forwards, "prompt_tokens": len(ids), "prefill_s": result.prefill_s,
                "request_tps": result.request_tps, "denoise_steps": result.denoise_steps,
                "output_ids": result.token_ids}

    run_suite(args, lead, {
        "backend": "dlmrt-sdar", "model": args.model, "numerics": args.numerics, "role": args.role,
        "decoder": decoder(args).__dict__,
        "prompt_format": "paper fixed user/assistant/<think> wrapper",
        "tp": tp_size(model)}, solve)


COMMANDS = {"step": step, "smoke": smoke, "humaneval": humaneval, "suite": suite}
