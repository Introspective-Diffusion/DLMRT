"""Streaming HTTP server (QwenDLM, DiffusionGemma or LLaDA2, by the checkpoint's model_type): one GPU, or tensor
parallel under torchrun.

    torchrun --nproc-per-node 8 -m dlmrt serve --model CKPT --port 18080     # TP=8
    python -m dlmrt serve --model CKPT --port 18081                          # one GPU

Rank 0 serves HTTP on --host (default 127.0.0.1) and broadcasts each request to the other ranks; every rank runs the
same generate() call. One request runs at a time (a single-request
engine); later requests wait in a queue.

GET  /info      -> {"tp", "gpu", "model", "model_type", ...}
POST /generate  {"prompt" or "messages", "max_new_tokens", "diffusion", "thinking"} -> NDJSON, one line per block
                (256 tokens; 32 for LLaDA2), then a summary. "messages" is a chat ([{"role", "content"}], ending
                with a user turn; earlier assistant turns without their thinking); thinking, default true, is
                DiffusionGemma's enable_thinking (QwenDLM always thinks; LLaDA2 ignores it):
                {"type": "block", "text", "tokens", "elapsed_s", "steps"} ... {"type": "done", ...}
With "diffusion": true every block line also carries "frames", one per denoise round: the block that round
started from, random tokens included, as {"tokens": [text of each position], "settled": "0110..."} with 1 where
the position already holds its emitted token; and "final", the emitted token's text per position. The emitted
text is the same with or without it.
Speeds are engine-side: generated tokens / time from the first prefill kernel to the last block
(tokenization and the network excluded), as in `dlmrt bench`.
"""

import argparse
import json
import queue
import threading
import time
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.distributed as dist

from .. import registry
from ..distributed import parallel
from ..llm import load_runtime, resolve


class Job:
    def __init__(self, ids, max_new_tokens, diffusion, thinking):
        self.ids, self.max_new_tokens, self.diffusion, self.thinking = ids, max_new_tokens, diffusion, thinking
        self.events = queue.Queue()


def parse_request(data, thinking_switch):
    """A /generate body as (messages, max_new_tokens, diffusion, thinking); ValueError with the 400's message."""
    try:
        body = json.loads(data or b"{}")
    except (ValueError, RecursionError) as e:  # also bytes that are not UTF-8, and nesting too deep to parse
        raise ValueError(f"body: {e}") from None
    if not isinstance(body, dict):
        raise ValueError("body: a JSON object")
    messages = body.get("messages")
    if not messages:
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str):
            raise ValueError("prompt: a string")
        messages = [{"role": "user", "content": prompt.strip()}]
    if not (isinstance(messages, list) and messages
            and all(isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")
                    and isinstance(m.get("content"), str) for m in messages)
            and messages[-1]["role"] == "user" and messages[-1]["content"].strip()):
        raise ValueError("messages: role/content turns ending with a nonempty user turn")
    max_new, diffusion, thinking = body.get("max_new_tokens", 2048), body.get("diffusion", False), body.get("thinking", True)
    # Whatever int() read before and means a whole number: 1024, 1024.0 (JSON from typed clients), "1024"
    if isinstance(max_new, float) and max_new.is_integer() or isinstance(max_new, str) and max_new.strip().isdigit():
        max_new = int(max_new)
    if not isinstance(max_new, int) or isinstance(max_new, bool):
        raise ValueError("max_new_tokens: an integer")
    flags = []
    for key, value in (("diffusion", diffusion), ("thinking", thinking)):
        if value is None or type(value) is int and value in (0, 1):  # read as before; only "false" was misread
            value = bool(value)
        if not isinstance(value, bool):
            raise ValueError(f"{key}: true or false")
        flags.append(value)
    diffusion, thinking = flags
    return ([{"role": m["role"], "content": m["content"]} for m in messages], max(1, min(max_new, 16384)), diffusion,
            thinking or not thinking_switch)


def main(argv=None):
    p = argparse.ArgumentParser(prog="dlmrt serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint directory or Hugging Face repo id")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--max-seq-len", type=int, default=32768)
    p.add_argument("--entropy-bound", type=float, help="default 0.6 for QwenDLM, the role's bound for DiffusionGemma")
    p.add_argument("--role", choices=["base", "orl"], default="base",
                   help="DiffusionGemma and LLaDA2: the decoder of a base or ORL checkpoint")
    p.add_argument("--threshold", type=float, help="LLaDA2 only: override the role's acceptance threshold")
    p.add_argument("--precision", choices=["bf16", "nvfp4"],
                   help="DiffusionGemma and LLaDA2: bf16 or nvfp4 (an NVFP4 export); default the checkpoint's")
    args = p.parse_args(argv)
    name = Path(args.model).name if Path(args.model).is_dir() else args.model  # a repo id, not its snapshot's hash
    args.model = resolve(args.model)
    registry.family(args.model)  # before the tokenizer, so an unsupported checkpoint's code never runs

    from transformers import AutoTokenizer

    group = parallel.init()
    lead = group is None or group.rank == 0
    control = dist.new_group(backend="gloo", timeout=timedelta(days=7)) if group else None
    # Every rank tokenizes the warm-up prompt, so all of them run the same rounds. The SDAR checkpoint ships its own
    # config code; without trust_remote_code the tokenizer asks on stdin, which a server cannot answer.
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    runtime = load_runtime(args.model, tokenizer, group, role=args.role, entropy_bound=args.entropy_bound,
                           threshold=args.threshold, precision=args.precision, max_seq_len=args.max_seq_len)
    engine, encode, eos_ids, hidden, markers = (runtime.engine, runtime.encode, runtime.eos_ids, runtime.hidden,
                                                runtime.markers)
    described = dict(runtime.info)
    tp = group.world if group else 1
    warm = encode([{"role": "user", "content": "Write a Python function that checks whether a number is prime."}], True)
    engine.generate(warm, 512, stop_on_eos=False)  # warm every graph once before serving
    engine.generate(warm, 256, stop_on_eos=False, trace=True)

    numerics, fused = described.pop("numerics", "BF16"), described.pop("fused_tp")
    info = {"tp": tp, "gpus": tp, "gpu": torch.cuda.get_device_name(), "model": name, **described,
            "engine": f"dlmrt ({numerics}, device-side denoise loop" + (", fused TP reduction)" if fused else ")"),
            "block_size": engine.block, "max_seq_len": args.max_seq_len, "diffusion_mode": True}
    jobs = queue.Queue()

    def run(job):
        box = [job.ids, job.max_new_tokens, job.diffusion] if lead else [None, None, None]
        if control is not None:
            dist.broadcast_object_list(box, src=0, group=control)
        ids, max_new, diffusion = box
        on_block = None
        if lead:
            def on_block(tokens, steps, elapsed, frames):
                job.events.put(("block", tokens, steps, elapsed, frames))
        result = engine.generate(ids, max_new, on_block=on_block, trace=diffusion)
        if lead:
            job.events.put(("done", result))

    if lead:
        start_server(args.host, args.port, info, jobs, tokenizer, engine, encode, eos_ids, hidden, markers,
                     args.max_seq_len)
        print(f"serving {described['model_type']} TP={tp} on {args.host}:{args.port}", flush=True)
        while True:
            run(jobs.get())
    else:
        while True:
            run(None)


def start_server(host, port, info, jobs, tokenizer, engine, encode, eos_ids, hidden, markers, max_seq_len):
    hidden = [tokenizer.decode([t]) for t in eos_ids] + hidden
    quiet = hidden + markers  # blank in diffusion frames
    pieces = {}  # token id -> its text, for diffusion frames

    def decode(ids):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        for h in hidden:
            text = text.replace(h, "")
        return text

    def diffusion_fields(frames, emitted):
        """Per-round frames and the emitted token's text per position (empty inside hidden strings and thinking
        markers), so the page can keep every position at its final width while it shows the noise."""
        new = sorted(({t for f in frames for t in f} | set(emitted)) - pieces.keys())
        pieces.update(zip(new, tokenizer.batch_decode([[t] for t in new], skip_special_tokens=False)))
        final = [pieces[t] for t in emitted]
        starts, joined = [], ""
        for p in final:
            starts.append(len(joined))
            joined += p
        shown = list(final)
        for h in filter(None, quiet):
            k = joined.find(h)
            while k >= 0:
                for j, s in enumerate(starts):
                    if k <= s and s + len(final[j]) <= k + len(h):
                        shown[j] = ""
                k = joined.find(h, k + len(h))
        return {"final": shown,
                "frames": [{"tokens": [pieces[t] for t in f],
                            "settled": "".join("1" if t == u else "0" for t, u in zip(f, emitted))} for f in frames]}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # close-delimited bodies stream without chunking

        def log_message(self, *_):
            pass

        def send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/info":
                self.send(200, {**info, "queued": jobs.qsize()})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self.send(404, {"error": "not found"})
            length = self.headers.get("Content-Length", "0").strip()
            if not length.isdecimal():
                return self.send(400, {"error": "Content-Length: a nonnegative integer"})
            try:
                messages, max_new, diffusion, thinking = parse_request(self.rfile.read(int(length)), info["thinking_switch"])
            except ValueError as e:
                return self.send(400, {"error": str(e)})
            try:
                ids = encode(messages, thinking)
            except Exception as e:  # the chat template rejects the conversation (e.g. turns that do not alternate)
                return self.send(400, {"error": f"chat template: {e}"})
            if len(ids) + max_new + 2 * engine.block > max_seq_len:
                return self.send(400, {"error": f"conversation too long for max_seq_len={max_seq_len}"})
            job = Job(ids, max_new, diffusion, thinking)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.line({"type": "queued", "position": jobs.qsize(), "prompt_tokens": len(ids)})
            submitted = time.perf_counter()
            jobs.put(job)
            generated, text = [], ""
            while True:
                kind, *rest = job.events.get()
                if kind == "block":
                    tokens, steps, elapsed, frames = rest
                    generated.extend(tokens)
                    full = decode(generated)
                    # A block boundary can split a multi-byte character; then the client replaces its text.
                    update = {"text": full[len(text):]} if full.startswith(text) else {"full_text": full}
                    text = full
                    if frames is not None:
                        update.update(diffusion_fields(frames, tokens))
                    self.line({"type": "block", **update, "tokens": len(tokens), "total_tokens": len(generated),
                               "steps": steps, "elapsed_s": elapsed})
                    continue
                r = rest[0]
                forwards = sum(r.forwards) if hasattr(r, "forwards") else sum(r.denoise_steps) + r.commit_forwards
                self.line({"type": "done", "tokens": len(r.token_ids), "elapsed_s": r.elapsed_s,
                           "prefill_s": r.prefill_s, "tps": r.request_tps, "blocks": len(r.denoise_steps),
                           "denoise_steps": sum(r.denoise_steps),
                           "tokens_per_step": len(r.token_ids) / max(1, sum(r.denoise_steps)),
                           "ms_per_forward": 1e3 * (r.elapsed_s - r.prefill_s) / max(1, forwards),
                           "finish": "stop" if r.token_ids and r.token_ids[-1] in eos_ids else "length",
                           "diffusion": job.diffusion, "thinking": job.thinking,
                           "queue_s": time.perf_counter() - submitted - r.elapsed_s,
                           **info})
                return

        def line(self, obj):
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client left; the engine still finishes the request

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()


if __name__ == "__main__":
    main()
