"""Shared benchmark helpers: CUDA timing, rank-0 reporting, the papers' client prompt, the frozen quick-v1 orders,
aggregation, the suite loop."""

import argparse
import hashlib
import json
import os
import statistics

import torch


def report(lead, *values, **kwargs):
    if lead:
        print(*values, **kwargs, flush=True)


SYSTEM = "You are a helpful assistant good at coding."
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"


def chat_messages(problem):
    """The papers' client prompt as chat messages: a row's `messages`, or system + instruction + fenced HumanEval
    prompt."""
    if problem.get("messages"):
        return [dict(m) for m in problem["messages"]]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION + f"\n```python\n{problem['prompt'].strip()}\n```"}]


def quick_order(benchmark, n):
    """The frozen quick-v1 order of a benchmark's n tasks (a fixed hash); the first is the warmup."""
    return sorted(range(n), key=lambda i: hashlib.sha256(f"tps-quick-v1|{benchmark}|{i}".encode()).hexdigest())


def evaluation_ids(full=False, count=8):
    """Frozen quick-v1 subset of HumanEval: the warmup and the measured tasks."""
    order = quick_order("humaneval", 164)
    return order[0], list(range(164)) if full else order[1:count + 1]


def aggregate(rows):
    seconds = sum(r["elapsed_s"] for r in rows)
    tokens = sum(r["completion_tokens"] for r in rows)
    forwards = sum(r["denoise_forwards"] for r in rows)
    return {
        "n": len(rows), "tokens": tokens, "seconds": seconds, "pooled_tps": tokens / seconds if seconds else None,
        "median_request_tps": statistics.median(r["request_tps"] for r in rows) if rows else None,
        "denoise_forwards": forwards, "emitted_tokens_per_denoise": tokens / forwards if forwards else None,
        "filled_tokens": sum(r["filled_tokens"] for r in rows),
        "commit_forwards": sum(r["commit_forwards"] for r in rows),
        "length_finishes": sum(r["finish_type"] == "length" for r in rows),
    }


def tp_size(model):
    """The ranks the model is sharded over (1 without tensor parallelism)."""
    return model.group.world if model.group else 1


def check_contexts(contexts, block, max_seq_len):
    """`bench step` runs one block at each context: [ctx, ctx + block) must lie in the KV cache and RoPE table."""
    for ctx in contexts:
        if not 0 <= ctx <= max_seq_len - block:
            raise SystemExit(f"--contexts {ctx}: its {block}-token block does not fit in --max-seq-len {max_seq_len}")


def cuda_ms(fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def shard(value):
    """`--shard I/N` as (I, N)."""
    try:
        index, count = (int(x) for x in value.split("/"))
        if 0 <= index < count:
            return index, count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected I/N with 0 <= I < N, got {value!r}")


def run_suite(args, lead, protocol, solve):
    """The paper's generation suite: each problem file in turn (its first --limit rows; every N-th from the I-th for
    --shard I/N, the paper's modulo partition; a shard with no rows does nothing). Records go to
    <out>/<file stem>/shard_w<I>.jsonl, which a rerun resumes, with the rows' `gold` and `meta` for the paper's
    scorers; protocol.json records each file's settings, and a rerun whose settings other than --limit differ stops.
    `solve(problem, max_new_tokens)` generates one problem and returns its record fields."""
    index, count = args.shard
    for problems in args.problems:
        rows = [json.loads(x) for x in problems.read_text().splitlines() if x.strip()]
        mine = rows[:args.limit][index::count]
        if not mine:
            continue
        out = args.out / problems.stem
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"shard_w{index}.jsonl"
        done = {json.loads(x)["task_id"] for x in path.read_text().splitlines() if x.strip()} if path.exists() else set()
        recorded = out / "protocol.json"
        text = json.dumps({
            **protocol, "problems": str(problems), "input_sha256": hashlib.sha256(problems.read_bytes()).hexdigest(),
            "limit": args.limit, "shards": count, "max_new_tokens": args.max_new_tokens,
            "timing": "engine.generate only (tokenization and detokenization excluded)"}, indent=2) + "\n"
        if recorded.exists() and recorded.read_text() != text:
            old, new = json.loads(recorded.read_text()), json.loads(text)
            # --limit only lengthens or shortens each shard's prefix of the file: the rows already written stay valid.
            changed = ", ".join(sorted(k for k in old.keys() | new.keys()
                                        if k != "limit" and json.dumps(old.get(k)) != json.dumps(new.get(k))))
            if changed:
                raise SystemExit(f"{recorded} records other settings ({changed}): resume with those, or use a new --out")
        if lead and index == 0 and (not recorded.exists() or recorded.read_text() != text):
            # Every rank and shard checks it: a reader sees the whole file or none.
            (out / "protocol.json.tmp").write_text(text)
            os.replace(out / "protocol.json.tmp", recorded)
        solve(mine[0], 64)
        with (path.open("a") if lead else open(os.devnull, "w")) as stream:
            for problem in mine:
                if problem["task_id"] in done:
                    continue
                row = {"task_id": problem["task_id"], **solve(problem, args.max_new_tokens)}
                row.update({key: problem[key] for key in ("gold", "meta") if key in problem})
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                report(lead, json.dumps({k: row[k] for k in ("task_id", "finish", "completion_tokens", "fwd_denoise",
                                                             "wall_s")}))
