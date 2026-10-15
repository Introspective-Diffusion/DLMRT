"""Dense SDAR checkpoint contract and the paper's greedy block decoder."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True
    eos_token_id: int = 151645
    self_conditioning_prob: float = 0.0
    self_conditioning_intermediate_size: int | None = None
    self_conditioning_post_norm: bool = False
    self_conditioning_post_norm_scale: float = 1.0
    self_conditioning_prenorm_scale: float = 1.0
    self_conditioning_reparam_out_scale: float = 1.0
    self_conditioning_residual_scale: float = 1.0
    self_conditioning_chunk_size: int = 256

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim != 128:
            raise ValueError("GQA with head_dim 128 is required")
        if not self.tie_word_embeddings:
            raise ValueError("The SC path assumes tied embeddings")
        if type(self.eos_token_id) is not int:
            raise ValueError(f"eos_token_id must be one token id, not {self.eos_token_id!r}")

    @property
    def sc(self):
        return self.self_conditioning_prob > 0

    @classmethod
    def load(cls, path):
        raw = json.loads((Path(path) / "config.json").read_text())
        if raw.get("model_type") != "sdar":
            raise ValueError("Only dense SDAR checkpoints are supported")
        for key in ("attention_bias", "use_sliding_window", "quantization_config", "num_experts"):
            if raw.get(key):
                raise ValueError(f"Unsupported model setting: {key}")
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Scaled RoPE is not implemented")
        raw.setdefault("rope_theta", rope.get("rope_theta", 10000.0))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DecodeConfig:
    """Greedy SDAR decoder. Entropy acceptance uses the exclusive cumulative budget."""

    accept_rule: str = "entropy_budget"
    entropy_bound: float = 0.4
    prob_threshold: float = 0.85
    edit_threshold: float = 0.9
    allow_accepted_change: bool = True
    confidence_temperature: float = 1.0
    confidence_temperature_min: float = 0.6
    confidence_temperature_step_decrement: float = 0.01
    random_token_high: int = 151669
    randtok_seed: int = 12345
    logits_shift: bool = True

    def __post_init__(self):
        if self.accept_rule not in ("entropy_budget", "prob_threshold"):
            raise ValueError("accept_rule must be entropy_budget or prob_threshold")
        for name in ("entropy_bound", "prob_threshold", "edit_threshold", "confidence_temperature",
                     "confidence_temperature_min", "confidence_temperature_step_decrement"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.confidence_temperature_min <= self.confidence_temperature:
            raise ValueError("Invalid confidence temperature range")


# The paper's QwenDLM decoders: Base accepts by probability threshold at confidence temperature one, with the edit
# threshold equal to it; ORL by entropy budget, confidence temperature annealed from 1.0 down to 0.6.
ROLES = {
    "base": DecodeConfig(accept_rule="prob_threshold", prob_threshold=0.85, edit_threshold=0.85,
                         confidence_temperature_min=1.0, confidence_temperature_step_decrement=0.0),
    "orl": DecodeConfig(entropy_bound=0.4),
}


def render_prompt(content):
    """The paper's fixed user/assistant/<think> wrapper; no tokenizer chat template."""
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n"
