"""The Python API: load a checkpoint of any supported family and generate from a prompt or a chat.

    from dlmrt import LLM

    llm = LLM("path/or/hf-repo")                    # one GPU; under torchrun, tensor parallel over the node
    out = llm.generate("Write a haiku about GPUs.", max_new_tokens=512)
    print(out.text, f"{out.request_tps:.0f} tok/s")

Under torchrun every rank constructs the same LLM and makes the same generate() calls, one request at a time.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import registry
from .distributed import parallel


@dataclass
class Runtime:
    """A loaded model as the API and the server drive it."""

    engine: object
    encode: Callable  # (messages, thinking) -> prompt token ids
    eos_ids: set
    info: dict  # model_type, decoder settings, thinking_switch, fused_tp; precision and numerics where they apply
    hidden: list = field(default_factory=list)  # strings removed from the text: the EOS token, an empty thought
    markers: list = field(default_factory=list)  # thinking markers: kept in the text, blank in diffusion frames


def resolve(model):
    """A local checkpoint directory: `model` itself, or the Hugging Face Hub snapshot of the repo it names."""
    if Path(model).is_dir():
        return str(model)
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def load_runtime(model, tokenizer, group=None, *, role="base", entropy_bound=None, threshold=None, precision=None,
                 max_seq_len=32768, trace=True):
    """The engine and chat encoding for the checkpoint's family. `role` (base / orl) picks the paper's decoder for
    DiffusionGemma and LLaDA2; `entropy_bound` (QwenDLM, DiffusionGemma) and `threshold` (LLaDA2) override it;
    `precision` (DiffusionGemma, LLaDA2) is bf16 or nvfp4, default the checkpoint's own. `trace` also captures the graphs
    that record every denoise round, for `generate(trace=True)`."""
    return {"sdar": _sdar, "llada2": _llada2, "diffusion_gemma": _diffusion_gemma}[registry.family(model)](
        model, tokenizer, group, role=role, entropy_bound=entropy_bound, threshold=threshold, precision=precision,
        max_seq_len=max_seq_len, trace=trace)


def _sdar(model, tokenizer, group, *, entropy_bound, max_seq_len, trace, **_):
    from .models.sdar import DecodeConfig, Engine, Model, render_prompt

    bound = 0.6 if entropy_bound is None else entropy_bound
    m = Model.load(model, max_seq_len=max_seq_len, group=group)
    engine = Engine(m, DecodeConfig(entropy_bound=bound), trace=trace)

    def encode(messages, thinking):
        # Trained with the paper's think wrapper only: an empty think block (the template's enable_thinking=False)
        # or /no_think does not stop it from thinking, so SDAR always thinks. One user turn is render_prompt.
        turns = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages[:-1])
        return tokenizer.encode(turns + render_prompt(messages[-1]["content"]), add_special_tokens=False)

    return Runtime(engine, encode, {m.config.eos_token_id},
                   {"model_type": "sdar", "entropy_bound": bound, "thinking_switch": False, "fused_tp": m.fused}, [],
                   ["</think>"])


def _llada2(model, tokenizer, group, *, role, threshold, precision, max_seq_len, trace, **_):
    from .models.llada2 import ROLES, LladaModel
    from .models.llada2.engine import LladaEngine
    from .models.llada2.model import REDUCE

    decoder = ROLES[role] if threshold is None else dataclasses.replace(ROLES[role], prob_threshold=threshold)
    m = LladaModel.load(model, max_seq_len=max_seq_len, group=group, precision=precision)
    engine = LladaEngine(m, decoder, trace=trace)
    numerics = "BF16" if m.precision == "bf16" else "NVFP4 experts, FP8 KV"

    def encode(messages, thinking):
        # The chat template fixes "detailed thinking off"; the text is encoded as the reference runtime does.
        return tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return Runtime(engine, encode, {m.config.eos_token_id},
                   {"model_type": "llada2_moe", "role": role, "prob_threshold": decoder.prob_threshold,
                    "thinking_switch": False, "precision": m.precision, "numerics": numerics,
                    "fused_tp": group is not None and REDUCE == "fused"}, ["<|role_end|>"], [])


def _diffusion_gemma(model, tokenizer, group, *, role, entropy_bound, precision, max_seq_len, trace, **_):
    from .models.diffusion_gemma import ROLES, GemmaModel
    from .models.diffusion_gemma.engine import GemmaEngine

    m = GemmaModel.load(model, max_seq_len=max_seq_len, group=group, precision=precision)
    bound = ROLES[role].entropy_bound if entropy_bound is None else entropy_bound
    engine = GemmaEngine(m, dataclasses.replace(ROLES[role], entropy_bound=bound), trace=trace)
    numerics = "BF16" if m.precision == "bf16" else "NVFP4 experts, FP8 KV"

    def encode(messages, thinking):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
        return tokenizer.encode(text, add_special_tokens=False)

    # Without thinking the answer opens with an empty thought channel, which is hidden.
    return Runtime(engine, encode, set(m.config.eos_token_ids),
                   {"model_type": "diffusion_gemma", "role": role, "entropy_bound": bound, "thinking_switch": True,
                    "precision": m.precision, "numerics": numerics, "fused_tp": group is not None},
                   ["<|channel>thought\n<channel|>"], ["<|channel>thought\n", "<channel|>"])


@dataclass
class RequestOutput:
    text: str
    generation: object  # runtime.generation.Generation: token ids, timing, denoise steps per block

    @property
    def token_ids(self):
        return self.generation.token_ids

    @property
    def request_tps(self):
        return self.generation.request_tps

    @property
    def denoise_steps(self):
        return self.generation.denoise_steps


class LLM:
    def __init__(self, model, *, role="base", entropy_bound=None, threshold=None, precision=None, max_seq_len=32768,
                 trace=False):
        """`model`: a checkpoint directory or a Hugging Face repo id. Decoder options as in `load_runtime`."""
        from transformers import AutoTokenizer

        self.model = resolve(model)
        registry.family(self.model)  # before the tokenizer, so an unsupported checkpoint's code never runs
        self.group = parallel.init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.runtime = load_runtime(self.model, self.tokenizer, self.group, role=role, entropy_bound=entropy_bound,
                                    threshold=threshold, precision=precision, max_seq_len=max_seq_len, trace=trace)

    @property
    def engine(self):
        return self.runtime.engine

    def encode(self, prompt=None, *, messages=None, thinking=True):
        """Prompt token ids for one user turn (`prompt`) or a chat (`messages`, ending with a user turn)."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        return self.runtime.encode(messages, thinking)

    def decode(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        for h in [self.tokenizer.decode([t]) for t in self.runtime.eos_ids] + self.runtime.hidden:
            text = text.replace(h, "")
        return text

    def generate(self, prompt=None, *, messages=None, prompt_ids=None, max_new_tokens=2048, thinking=True,
                 stop_on_eos=True, on_block=None, trace=False):
        """One request. `on_block(token_ids, denoise_steps, elapsed_s, frames)` is called after every block; `trace`
        (with LLM(..., trace=True)) passes every round's input block as `frames`."""
        ids = prompt_ids if prompt_ids is not None else self.encode(prompt, messages=messages, thinking=thinking)
        generation = self.engine.generate(ids, max_new_tokens, stop_on_eos=stop_on_eos, on_block=on_block, trace=trace)
        return RequestOutput(self.decode(generation.token_ids), generation)
