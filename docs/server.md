# Server

`dlmrt serve` streams generations over HTTP, on one GPU or tensor parallel under torchrun:

```bash
torchrun --nproc-per-node 8 -m dlmrt serve --model CKPT --port 18080 [--role orl] [--precision nvfp4]
python -m dlmrt serve --model CKPT --port 18081                       # one GPU
```

| Option | |
| --- | --- |
| `--model` | checkpoint directory or Hugging Face repo id |
| `--host`, `--port` | where rank 0 listens (default `127.0.0.1:18080`) |
| `--max-seq-len` | prompt + generation limit (default 32768) |
| `--role` | `base` or `orl`: the decoder settings of LLaDA2 and DiffusionGemma ([models](models.md)); QwenDLM runs its entropy-budget decoder at `--entropy-bound` |
| `--entropy-bound` | QwenDLM (default 0.6) and DiffusionGemma (default: the role's) acceptance bound |
| `--threshold` | LLaDA2: override the role's acceptance threshold |
| `--precision` | DiffusionGemma and LLaDA2: `bf16` or `nvfp4` (default: the checkpoint's) |

Rank 0 serves HTTP and broadcasts each request to the other ranks; every rank runs the same generation. dlmrt is a
single-request engine: one request runs at a time and later ones wait in a queue. Before serving, the server runs a
warm-up generation, so every CUDA graph exists when the first request arrives.

The tokenizer is loaded with `trust_remote_code=True`, as in the Python API, so a checkpoint's own Python code runs
once its `model_type` is one dlmrt serves: serve only checkpoints you trust. The server has no authentication; it
listens on `127.0.0.1` unless `--host` says otherwise.

## `GET /info`

The server's configuration: `tp` (also as `gpus`), `gpu`, `model` (the `--model` repo id, or the checkpoint
directory's name), `model_type`, the decoder settings (`role`, `entropy_bound` or `prob_threshold`), `precision`
(DiffusionGemma, LLaDA2), `thinking_switch` (whether `thinking` applies), `block_size`, `max_seq_len`, `engine`
(numerics, and whether TP sums run in dlmrt's fused reduction), `diffusion_mode`, and `queued` (requests waiting).

## `POST /generate`

```json
{"messages": [{"role": "user", "content": "Write quicksort in Python."}], "max_new_tokens": 1024}
```

| Field | |
| --- | --- |
| `messages` | a chat: `{"role": "system" \| "user" \| "assistant", "content"}` turns ending with a nonempty user turn (earlier assistant turns without their thinking) |
| `prompt` | instead of `messages`: one user turn |
| `max_new_tokens` | default 2048, at most 16384 |
| `thinking` | DiffusionGemma's `enable_thinking` (default true); QwenDLM always thinks and LLaDA2 ignores it |
| `diffusion` | add every denoise round of each block to the stream (default false); the emitted text is the same |

The response is NDJSON (`application/x-ndjson`), one object per line, written as the generation proceeds:

1. `{"type": "queued", "position", "prompt_tokens"}` as soon as the request is accepted.
2. One `{"type": "block", "text", "tokens", "total_tokens", "steps", "elapsed_s"}` per block (256 tokens; 32 for
   LLaDA2): `text` is the new text since the previous line. When a block boundary splits a multi-byte character, the
   line carries `full_text` (all text so far) instead, which replaces what the client has.
3. `{"type": "done", ...}`: `tokens`, `elapsed_s`, `prefill_s`, `tps`, `blocks`, `denoise_steps`, `tokens_per_step`,
   `ms_per_forward`, `finish` (`stop` at EOS, else `length`), `diffusion`, `thinking`, `queue_s`, and the `/info`
   fields.

With `"diffusion": true` each block line also has `frames`, one per denoise round: the block that round started from,
random tokens included, as `{"tokens": [text of each position], "settled": "0110..."}` with `1` where the position
already holds its emitted token; and `final`, the emitted token's text per position (blank inside hidden strings and
thinking markers).

Speeds are engine-side, as in `dlmrt bench`: generated tokens over the time from the first prefill kernel to the last
block (tokenization and the network excluded). A request gets a 400 with `{"error"}` when its `Content-Length` is not
a nonnegative integer, its body is not a JSON object, its `messages` are malformed, its `prompt` is empty or not a
string, `max_new_tokens` is not a whole number, `thinking` or `diffusion` is not a boolean (or 0, 1, null), the chat
template rejects the conversation, or it is too long for `--max-seq-len`.

```bash
curl -N localhost:18080/generate -d '{"prompt": "Hi!", "max_new_tokens": 256}'
curl localhost:18080/info
```
