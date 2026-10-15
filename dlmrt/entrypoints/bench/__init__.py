"""Benchmarks of every model family, dispatched by the checkpoint's model_type.

    dlmrt bench step --model CKPT                round latency, and commit (prefill for LLaDA2) latency, at fixed
                                                 contexts
    dlmrt bench smoke --model CKPT               one generation, printed
    dlmrt bench humaneval --model CKPT --inputs openai_humaneval.parquet --out DIR [--full]
                                                 the paper's HumanEval generation protocol (QwenDLM and
                                                 DiffusionGemma); programs are scored
                                                 separately (benchmarks/humaneval_score.py)
    dlmrt bench suite --model CKPT --role base|orl --problems F.jsonl --out DIR
                                                 the paper's generation protocol over problem files (records for the
                                                 paper's isolated scorers)
    dlmrt bench tps --model LLADA_CKPT           LLaDA2 only: the paper's TPS quick check

Tensor parallel: launch any command under `torchrun --nproc-per-node N -m dlmrt bench ...`; every rank runs the same
program and rank 0 reports (`--numerics sglang` runs on one GPU only). Family-specific options are listed under each
command's --help.
"""

import argparse
from pathlib import Path

from ... import registry
from ...quantization.precision import PRECISIONS
from . import diffusion_gemma, llada2, sdar
from .common import shard


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dlmrt bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("step", "smoke", "humaneval"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--fp8", nargs="?", const="all", type=sdar.fp8_projections, default=False,
                       help="QwenDLM only: FP8 projections: bare flag = qkv,gate_up,down; or a comma list of "
                            "qkv,o,gate_up,down,sc")
        p.add_argument("--max-seq-len", type=int, help="default 16384 (QwenDLM humaneval: 65536)")
    sub.choices["step"].add_argument("--contexts", type=int, nargs="+", default=[0, 2048, 8192])
    sub.choices["step"].add_argument("--out")
    sub.choices["smoke"].add_argument("--prompt", default="Write a Python function that checks whether a number is prime.")
    sub.choices["smoke"].add_argument("--max-new-tokens", type=int, default=2048)
    sub.choices["smoke"].add_argument("--out")
    he = sub.choices["humaneval"]
    he.add_argument("--inputs", type=Path, required=True)
    he.add_argument("--out", type=Path, required=True)
    he.add_argument("--full", action="store_true")
    he.add_argument("--count", type=int, default=8)
    he.add_argument("--repeats", type=int, default=2)
    he.add_argument("--max-new-tokens", type=int, default=32768)
    he.add_argument("--randtok-seed", type=int, default=12345)
    p = sub.add_parser("suite", help="the paper's generation protocol over problem files")
    p.add_argument("--model", required=True)
    p.add_argument("--problems", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", default="0/1", type=shard,
                   help="I/N, 0 <= I < N: every N-th problem from the I-th; a rerun with the same settings resumes")
    p.add_argument("--limit", type=int, help="the first N problems of each file")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--role", choices=("base", "orl"), default="base",
                   help="the paper's decoder for a base or ORL checkpoint")
    p.add_argument("--threshold", type=float,
                   help="LLaDA2, QwenDLM Base: override the role's acceptance threshold (QwenDLM: its edit threshold too)")
    p.add_argument("--edit-threshold", type=float, help="LLaDA2, QwenDLM: override the role's edit threshold")
    p.add_argument("--entropy-bound", type=float, help="DiffusionGemma, QwenDLM: override the role's entropy bound")
    p.add_argument("--seed", type=int, default=1234, help="DiffusionGemma: per-request canvas seed")
    p.add_argument("--randtok-seed", type=int, default=12345, help="QwenDLM: the random-token canvas seed")
    p.add_argument("--precision", choices=PRECISIONS,
                   help="LLaDA2, DiffusionGemma: bf16 or nvfp4 (an NVFP4 export); default the checkpoint's")
    llada2.add_decoder_arguments(p, prefix="LLaDA2: ")
    p.add_argument("--in-place", action="store_true", help="LLaDA2: refine every block in place (no fused rounds)")
    p.add_argument("--numerics", choices=("dlmrt", "sglang"), default="dlmrt",
                   help="LLaDA2, QwenDLM: " + llada2.NUMERICS_HELP)
    diffusion_gemma.add_arguments(sub)
    llada2.add_arguments(sub)
    args = parser.parse_args(argv)
    family = registry.family(args.model)
    commands = {"sdar": sdar.COMMANDS, "llada2": llada2.COMMANDS, "diffusion_gemma": diffusion_gemma.COMMANDS}[family]
    if args.command not in commands:
        raise SystemExit(f"dlmrt bench {args.command} is not implemented for {registry.model_type(args.model)}")
    if args.max_seq_len is None:  # step, smoke, humaneval
        args.max_seq_len = 65536 if (family, args.command) == ("sdar", "humaneval") else 16384
    commands[args.command](args)
