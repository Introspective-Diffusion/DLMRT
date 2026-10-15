# Benchmarks

`dlmrt bench` dispatches on the checkpoint's `model_type`; `--model` is a local checkpoint directory
(`hf download REPO --local-dir DIR`). Run any command under `torchrun --nproc-per-node N -m dlmrt bench ...` for
tensor parallelism (every rank runs the same program; rank 0 reports and writes); `--numerics sglang` runs on one GPU
and stops under torchrun. Family-specific options are listed in each command's `--help`.

## Round latency

```bash
python -m dlmrt bench step --model CKPT [--contexts 0 2048 8192] [--out step.json]
```

Times one captured denoise round at each context (CUDA-event median of 20 replays), and one commit at each context
(QwenDLM, DiffusionGemma) or one 256-row prefill (LLaDA2, median of 5).
QwenDLM (`sdar`) also takes `--fp8`. Every context plus one block must fit in `--max-seq-len` (default 16384); the
command checks this before it loads the model.

## HumanEval (QwenDLM, DiffusionGemma)

```bash
torchrun --nproc-per-node 8 -m dlmrt bench humaneval --model CKPT --inputs openai_humaneval.parquet \
    --out results/he-s1234 --full --repeats 1 --max-new-tokens 2048 --seed 1234 [--role orl] [--precision nvfp4]
python benchmarks/humaneval_score.py openai_humaneval.parquet results/he-s1234
```

`--inputs` is the `openai_humaneval` test split as parquet. The prompt protocols follow the papers the checkpoints
come from: QwenDLM uses its fixed `<think>` wrapper (entropy bound 0.6, `--randtok-seed` fixes the noise); DiffusionGemma
uses the chat template with a system message and an instruction + fenced prompt (`--seed` fixes the noise). Without
`--full` a frozen 8-task subset runs (the first task of a fixed hash order is the warm-up). `--max-new-tokens`
defaults to 32768 and `--max-seq-len` to 65536 for QwenDLM and 16384 for DiffusionGemma, so DiffusionGemma needs a
smaller budget (its paper's is 2048) or a longer context: a request whose prompt and budget, in whole blocks, exceed
`--max-seq-len` stops the command before the model loads and before `--out` is created.

The command writes `protocol.json`, one `requests.jsonl` row per generation (text, token ids, timing, denoise steps
per block) and `summary.json` (pooled request TPS, tokens per denoise). Generation and scoring are separate:
`benchmarks/humaneval_score.py` runs each answer's last code block against HumanEval's original tests (20 s each,
without stdin, not as `__main__`, so an answer's own main block does not run) and writes `quick_scores.json` per
repeat. It executes model-written code, so run it in a sandbox; for the papers' numbers the answers were scored with
EvalPlus in an isolated container.

## LLaDA2

```bash
python -m dlmrt bench smoke --model LLADA_CKPT --role orl
python -m dlmrt bench suite --model LLADA_CKPT --role base --problems gsm8k.jsonl math500.jsonl humaneval.jsonl --out DIR
python -m dlmrt bench tps --model LLADA_CKPT --role orl --gsm8k gsm8k.jsonl --humaneval humaneval.jsonl
# Speed mode (ORL): edit threshold 0, step_linear confidence temperature
python -m dlmrt bench suite --model LLADA_CKPT --role orl --threshold 0.6 --edit-threshold 0 --temperature-min 0.6 \
    --temperature-decrement 0.08 --problems humaneval.jsonl --out DIR
# the paper runtime's numerics (BF16, one GPU)
python -m dlmrt bench suite --model LLADA_CKPT --role orl --numerics sglang --problems humaneval.jsonl --out DIR
```

`suite` runs the paper's generation protocol (chat `messages` rows go through the chat template as they are; HumanEval
`prompt` rows become system + instruction + fenced prompt; 4096 new tokens, greedy) and writes records with the
paper's field names for its scorers; `--shard I/N` (0 <= I < N) takes every N-th problem from the I-th; `--limit N`
keeps the first N problems of each file. A rerun into the same `--out` resumes after the rows already written if its
settings other than `--limit` match the `protocol.json` there, and stops otherwise. `tps` is the paper's TPS quick
check over 8-task GSM8K and HumanEval subsets (pooled over the responses that stop before the token budget; `null` if
none does).
The paper's decoders: Quality mode is the role's threshold (Base 0.7, ORL 0.85) with edit threshold 0.5; Speed mode
lowers the threshold (Base 0.5, ORL 0.6) and the edit threshold to 0 (`--edit-threshold 0`) and, for ORL, anneals the
confidence temperature (`--temperature-min 0.6 --temperature-decrement 0.08`, always given together). `--in-place`
refines every block in place instead of in fused rounds (the same tokens, one forward more per block).

## DiffusionGemma suite

```bash
python -m dlmrt bench suite --model DG_CKPT --role orl --seed 1234 --problems gsm8k.jsonl math500.jsonl \
    humaneval.jsonl mbpp.jsonl --out DIR [--precision nvfp4]
```

The same records over the paper's problem files with DiffusionGemma's prompt format (chat `messages` rows as they
are, HumanEval `prompt` rows as system + instruction + fenced prompt, through the chat template); `--seed` resets
the canvas noise for every request. The paper's settings: 4096 new tokens (2048 for HumanEval), seeds 1234-1236
averaged, each checkpoint's decoder (`--role base` / `--role orl`) on all four benchmarks.

## QwenDLM suite

```bash
python -m dlmrt bench suite --model QWENDLM_CKPT --role base --threshold 0.85 --max-new-tokens 49152 --max-seq-len 65536 \
    --problems gsm8k.jsonl math500.jsonl mbpp.jsonl --out DIR
python -m dlmrt bench suite --model QWENDLM_CKPT --role orl --entropy-bound 0.6 --max-new-tokens 32768 --max-seq-len 65536 \
    --problems humaneval.jsonl --out DIR
```

Rows carry a `prompt`, sent in the paper's fixed `<think>` wrapper. `--role base` is Base's probability threshold at
confidence temperature one (its edit threshold follows `--threshold` unless `--edit-threshold` is given); `--role orl`
is ORL's entropy budget (`--entropy-bound`; it takes no `--threshold`) with the confidence temperature annealed from
1.0 to 0.6. The paper's settings: Base threshold 0.85 (0.9 for HumanEval) with 49152 new tokens (32768 for
HumanEval); ORL entropy bound 0.4 for GSM8K and MATH-500 and 0.6 for HumanEval and MBPP, with 32768 new tokens (49152
for MBPP). `--numerics sglang` runs the paper runtime's numerics.

## Reproducing the paper's table

The table in [performance](performance.md#all-models-bf16-on-one-h100) runs each job on one H100 in the
`lmsysorg/sglang:v0.5.13` image (on H100 the SGLang-numerics rows give the paper runtime's answers; the counts are
under the table). The
problem files and the isolated scorers are those of the paper's
[code repository](https://github.com/Introspective-Diffusion/ORL); dlmrt's records carry the field names its scorers
read. `benchmarks/suite_stats.py` gives the TPF and TPS columns from the records.

The checkpoints the paper evaluated, at these revisions (`hf download REPO --revision REVISION --local-dir DIR`):

| Placeholder | Checkpoint | Hugging Face repository (revision) |
| --- | --- | --- |
| `QWENDLM_BASE` | QwenDLM 4B | `yifanyu/QwenDLM-4B` (`94c4a956af73bb7be5ae5474e85a490a96ccb2b2`) |
| `QWENDLM_ORL` | QwenDLM 4B + ORL | `yifanyu/QwenDLM-4B-ORL` (`e12b90bf8267aa5d994bcf73996f00e00d484239`) |
| `LLADA_BASE` | LLaDA2.1-mini | `inclusionAI/LLaDA2.1-mini` (`20e64e2ad21644d0e5248586ed9c942cdd45de0f`) |
| `LLADA_ORL` | LLaDA2.1-mini + ORL | `yifanyu/LLaDA2.1-mini-ORL` (`eb9e57b7ced51a750ec4d94aa7d4cee46243f597`) |
| `DG_BASE` | DiffusionGemma 26B-A4B | `google/diffusiongemma-26B-A4B-it` (`f7f5b7f5fa82ffc52addd066915886d497f5517b`) |
| `DG_ORL` | DiffusionGemma 26B-A4B + ORL | `yifanyu/DiffusionGemma-26B-A4B-it-ORL` (`a5f9c6d23c585eefacf3986a12c3e809061f01ce`) |
| | LLaDA2.1-mini + ORL, NVFP4 experts | `yifanyu/LLaDA2.1-mini-ORL-NVFP4` (`291928fa5f4bdfe10c79e1f10b570b1750af6336`) |
| | DiffusionGemma 26B-A4B + ORL, NVFP4 experts | `yifanyu/DiffusionGemma-26B-A4B-it-ORL-NVFP4` (`73e088db981412fcbff76e1ba8ccf99985fed8e3`) |

```bash
# QwenDLM: SGLang numerics, the QwenDLM problem files
Q="--numerics sglang --max-seq-len 65536"
python -m dlmrt bench suite --model QWENDLM_BASE --role base --threshold 0.85 --max-new-tokens 49152 $Q \
    --problems gsm8k.jsonl math500.jsonl mbpp.jsonl --out out/qwendlm-base
python -m dlmrt bench suite --model QWENDLM_BASE --role base --threshold 0.9 --max-new-tokens 32768 $Q \
    --problems humaneval.jsonl --out out/qwendlm-base
python -m dlmrt bench suite --model QWENDLM_ORL --role orl --entropy-bound 0.4 --max-new-tokens 32768 $Q \
    --problems gsm8k.jsonl math500.jsonl --out out/qwendlm-orl
python -m dlmrt bench suite --model QWENDLM_ORL --role orl --entropy-bound 0.6 --max-new-tokens 32768 $Q \
    --problems humaneval.jsonl --out out/qwendlm-orl
python -m dlmrt bench suite --model QWENDLM_ORL --role orl --entropy-bound 0.6 --max-new-tokens 49152 $Q \
    --problems mbpp.jsonl --out out/qwendlm-orl

# LLaDA2.1-mini: SGLang numerics, Quality (q) and Speed (s) modes
L="--numerics sglang --max-new-tokens 4096 --problems gsm8k.jsonl math500.jsonl humaneval.jsonl mbpp.jsonl"
python -m dlmrt bench suite --model LLADA_BASE --role base --threshold 0.7 --edit-threshold 0.5 $L \
    --out out/llada-base-q
python -m dlmrt bench suite --model LLADA_ORL --role orl --threshold 0.85 --edit-threshold 0.5 $L --out out/llada-orl-q
python -m dlmrt bench suite --model LLADA_BASE --role base --threshold 0.5 --edit-threshold 0 $L --out out/llada-base-s
python -m dlmrt bench suite --model LLADA_ORL --role orl --threshold 0.6 --edit-threshold 0 --temperature-min 0.6 \
    --temperature-decrement 0.08 $L --out out/llada-orl-s

# DiffusionGemma: one run per seed (the seeds listed under the table), each checkpoint with its role
for role in base orl; do
  model=$([ $role = base ] && echo DG_BASE || echo DG_ORL)
  for seed in 1234 1235 1236 1237 1238 1239 1240 1241; do
    python -m dlmrt bench suite --model $model --role $role --seed $seed --max-new-tokens 4096 \
        --problems gsm8k.jsonl math500.jsonl mbpp.jsonl --out out/dg-$role-s$seed
    python -m dlmrt bench suite --model $model --role $role --seed $seed --max-new-tokens 2048 \
        --problems humaneval.jsonl --out out/dg-$role-s$seed
  done
done

python benchmarks/suite_stats.py out/llada-orl-q/humaneval
python benchmarks/suite_stats.py out/dg-orl-s1236/gsm8k out/dg-orl-s1238/gsm8k out/dg-orl-s1241/gsm8k
```

The NVFP4 results run the NVFP4 exports with `--precision nvfp4` on one B200 or one RTX 5090: LLaDA2.1-mini ORL in
Quality mode, and DiffusionGemma ORL at the seeds given with its results.

## Checking a change: token identity

Every engine change that is not meant to change numerics is checked token for token: run the same protocol and seed
on the old and new code and compare (`compare_runs.py` pairs the generations by task and repeat).

```bash
python benchmarks/compare_runs.py results/he-old results/he-new
# {"tasks": 164, "token_identical": 164, "step_identical": 164, "tps_a": 7288.4, "tps_b": 7303.4, ...}
PYTHONPATH=old python benchmarks/kernel_dump.py old.pt; PYTHONPATH=new python benchmarks/kernel_dump.py new.pt
python benchmarks/kernel_dump.py --compare old.pt new.pt          # DiffusionGemma's row kernels, bit for bit
```

Changes that do alter numerics (different tiles, reduction orders, precisions) are judged as the
[numerics policy](../CONTRIBUTING.md#the-numerics-policy) says.

## Kernel sweeps

`benchmarks/attention_sweep.py` times DiffusionGemma's Triton attention over tile configurations at a rank's shapes
(e.g. `2 1 2 1` for TP=8); `benchmarks/row_kernels.py` times the per-layer row kernels as PDL chains in a CUDA graph.
A kernel that faults takes its GPU down (on managed clusters often the whole node): run configurations that never ran
on a GPU type under `compute-sanitizer --tool memcheck` first (`CHECK=1` makes the sweep run each configuration once).
