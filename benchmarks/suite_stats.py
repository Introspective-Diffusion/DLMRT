"""TPF and request TPS of `dlmrt bench suite` records, as docs/performance.md defines them: over the responses that
did not hit the token budget, TPF is the mean of completion tokens / denoise rounds and TPS the completion tokens
over the generation time, pooled. Several record directories (seeds of one setting) are averaged per metric.
Accuracy comes from the paper's scorers.

usage: python benchmarks/suite_stats.py DIR [DIR ...]   (each DIR an <out>/<problem file stem> of `dlmrt bench suite`)
"""
import json
import statistics
import sys
from pathlib import Path


def stats(directory):
    rows = [json.loads(line) for f in sorted(Path(directory).glob("shard_w*.jsonl"))
            for line in f.read_text().splitlines() if line.strip()]
    clean = [r for r in rows if r["finish"] != "length"]
    return {"tasks": len(rows), "length_capped": len(rows) - len(clean),
            "tpf": statistics.mean(r["completion_tokens"] / r["fwd_denoise"] for r in clean),
            "tps": sum(r["completion_tokens"] for r in clean) / sum(r["wall_s"] for r in clean)}


runs = [stats(d) for d in sys.argv[1:]]
for d, s in zip(sys.argv[1:], runs):
    print(f"{d}: {s['tasks']} tasks, {s['length_capped']} length-capped, TPF {s['tpf']:.2f}, TPS {s['tps']:.0f}")
if len(runs) > 1:
    print(f"mean of {len(runs)}: TPF {statistics.mean(s['tpf'] for s in runs):.2f}, "
          f"TPS {statistics.mean(s['tps'] for s in runs):.0f}")
