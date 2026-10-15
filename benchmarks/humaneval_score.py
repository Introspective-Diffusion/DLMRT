"""Quick HumanEval check of `dlmrt bench humaneval` output: the original tests (openai_humaneval), 20 s per program.
Not the paper's isolated EvalPlus scorer; for regressions between engine versions only. Writes quick_scores.json
({repeat: {task_id: passed}}) into each run directory.

usage: python3 benchmarks/humaneval_score.py openai_humaneval.parquet RUN_DIR [RUN_DIR ...]
"""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq


def program(test, text):
    answer = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.S)
    code = blocks[-1] if blocks else answer
    imports = "\n".join(l for l in test["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return f"{imports}\n{code}\n\n{test['test']}\n\ncheck({test['entry_point']})\n"


# The program runs as module "candidate", not __main__: an answer's own `if __name__ == "__main__":` block (input(),
# sys.exit(0), unittest.main()) would otherwise run before check(). Its directory stays first on sys.path (cwd).
RUNNER = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='candidate')"


def run(path):
    # No stdin (input() fails instead of waiting) and a session of its own, so a timeout kills its children too.
    with subprocess.Popen([sys.executable, "-c", RUNNER, str(path)], cwd=path.parent, stdin=subprocess.DEVNULL,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True) as proc:
        try:
            return proc.wait(timeout=20) == 0
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return False


def main():
    tests = {t["task_id"]: t for t in pq.read_table(sys.argv[1]).to_pylist()}
    for arm in map(Path, sys.argv[2:]):
        rows = [json.loads(x) for x in (arm / "requests.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r["kind"] == "measured"]
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, r in enumerate(rows):
                p = Path(tmp) / f"{i:04d}.py"
                p.write_text(program(tests[r["task_id"]], r["text"]))
                paths.append(p)
            with ThreadPoolExecutor(32) as pool:
                ok = list(pool.map(run, paths))
        tokens = sum(r["completion_tokens"] for r in rows)
        elapsed = sum(r["elapsed_s"] for r in rows)
        forwards = sum(r["denoise_forwards"] for r in rows)
        commits = sum(r["commit_forwards"] for r in rows)
        print(json.dumps({"arm": arm.name, "pass": sum(ok), "n": len(rows), "pooled_tps": round(tokens / elapsed, 1),
                          "tokens": tokens, "tokens_per_denoise": round(tokens / forwards, 2),
                          "ms_per_forward": round(1000 * elapsed / (forwards + commits), 3),
                          "length_finishes": sum(r["finish_type"] == "length" for r in rows)}), flush=True)
        scores = {}
        for r, o in zip(rows, ok):
            scores.setdefault(r["repeat"], {})[r["task_id"]] = o
        (arm / "quick_scores.json").write_text(json.dumps(scores) + "\n")


if __name__ == "__main__":
    main()
