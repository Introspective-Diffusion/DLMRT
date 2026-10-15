"""The `dlmrt` command: `dlmrt bench ...`, `dlmrt serve ...` (also `python -m dlmrt ...`, e.g. under torchrun)."""

import sys

USAGE = """usage: dlmrt {bench,serve} ...

    bench   latency, smoke, HumanEval, suite and TPS benchmarks (dlmrt bench --help)
    serve   streaming HTTP server, one request at a time (dlmrt serve --help)
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return
    command, rest = argv[0], argv[1:]
    if command == "bench":
        from .bench import main as bench
        return bench(rest)
    if command == "serve":
        from .server import main as serve
        return serve(rest)
    raise SystemExit(f"dlmrt: unknown command {command!r}\n{USAGE}")
