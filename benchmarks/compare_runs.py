"""Token / step identity of two dlmrt runs (same seed and protocol), per (task, repeat), plus pooled TPS of each.

usage: python3 benchmarks/compare_runs.py RUN_A RUN_B   (two `dlmrt bench humaneval --out` directories, or two
                                                       `dlmrt bench suite` record directories <out>/<file stem>)
"""
import json
import sys
from pathlib import Path


def load(arm):
    arm = Path(arm)
    if (arm / "requests.jsonl").exists():
        rows = [json.loads(x) for x in (arm / "requests.jsonl").read_text().splitlines()]
        return {(r["task_id"], r["repeat"]): r for r in rows if r["kind"] == "measured"}
    rows = [json.loads(x) for f in sorted(arm.glob("shard_w*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
    return {(r["task_id"], 0): r for r in rows}


a, b = load(sys.argv[1]), load(sys.argv[2])
tasks = sorted(a.keys() & b.keys())
same_tokens = sum(a[k]["output_ids"] == b[k]["output_ids"] for k in tasks)
same_steps = sum(a[k]["denoise_steps"] == b[k]["denoise_steps"] for k in tasks)


def tps(arm):
    rows = [arm[k] for k in tasks]
    return sum(r["completion_tokens"] for r in rows) / sum(r.get("elapsed_s", r.get("wall_s")) for r in rows)


print(json.dumps({"tasks": len(tasks), "token_identical": same_tokens, "step_identical": same_steps,
                  "tps_a": round(tps(a), 1), "tps_b": round(tps(b), 1), "speedup": round(tps(b) / tps(a), 4)}))
