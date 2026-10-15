"""LLaDA2.1-mini (llada2_moe): checkpoint contract and the paper's threshold decoder."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from ...quantization import nvfp4


# The reference caps a block's round index for its confidence-temperature schedule at max_steps_per_block.
MAX_STEPS_PER_BLOCK = 32


@dataclass(frozen=True)
class LladaConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    first_k_dense_replace: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    vocab_size: int
    rms_norm_eps: float = 1e-6
    block_size: int = 32
    eos_token_id: int = 156892
    mask_token_id: int = 156895
    use_self_conditioning: bool = False
    self_conditioning_intermediate_size: int | None = None
    self_conditioning_prenorm_scale: float = 1.0
    self_conditioning_reparam_out_scale: float = 1.0
    self_conditioning_residual_scale: float = 1.0
    self_conditioning_post_norm: bool = False
    self_conditioning_post_norm_scale: float = 1.0

    @classmethod
    def load(cls, path):
        raw = json.loads((Path(path) / "config.json").read_text())
        if raw.get("model_type") != "llada2_moe":
            raise ValueError("Only llada2_moe checkpoints are supported")
        checks = {"hidden_act": "silu", "score_function": "sigmoid", "moe_router_enable_expert_bias": True,
                  "norm_topk_prob": True, "num_shared_experts": 1, "tie_word_embeddings": False}
        for key, want in checks.items():
            if raw.get(key) != want:
                raise ValueError(f"Unsupported model setting: {key}={raw.get(key)!r} (expected {want!r})")
        for key in ("use_bias", "use_qkv_bias", "norm_head"):
            if raw.get(key):
                raise ValueError(f"Unsupported model setting: {key}")
        if raw.get("quantization_config") and nvfp4.quant_config(path)[0] is None:
            raise ValueError("Unsupported quantization_config (only ModelOpt NVFP4 exports)")
        if not raw.get("use_qk_norm", True) or raw.get("use_sliding_window"):
            raise ValueError("Expected QK norm and full attention")
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Scaled RoPE is not implemented")
        partial = rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 1.0))
        sc = {k: v for k, v in raw.items() if k.startswith("self_conditioning_") and k in cls.__dataclass_fields__}
        if (sc.get("self_conditioning_prenorm_scale", 1.0) == 1.0) != \
                (sc.get("self_conditioning_reparam_out_scale", 1.0) == 1.0):
            raise ValueError("self_conditioning prenorm / reparam-out scales must be set together")
        return cls(hidden_size=raw["hidden_size"], intermediate_size=raw["intermediate_size"],
                   moe_intermediate_size=raw["moe_intermediate_size"], num_experts=raw["num_experts"],
                   num_experts_per_tok=raw["num_experts_per_tok"], n_group=raw["n_group"],
                   topk_group=raw["topk_group"], routed_scaling_factor=raw["routed_scaling_factor"],
                   first_k_dense_replace=raw["first_k_dense_replace"], num_hidden_layers=raw["num_hidden_layers"],
                   num_attention_heads=raw["num_attention_heads"], num_key_value_heads=raw["num_key_value_heads"],
                   head_dim=raw["head_dim"], rotary_dim=int(raw["head_dim"] * partial),
                   rope_theta=rope.get("rope_theta", raw.get("rope_theta")), vocab_size=raw["vocab_size"],
                   rms_norm_eps=raw["rms_norm_eps"], use_self_conditioning=bool(raw.get("use_self_conditioning")),
                   **sc)


@dataclass(frozen=True)
class LladaDecode:
    """The paper's LLaDA decoder (SGLang's SDARPrefixKV).

    Every round proposes the argmax token at each position with its softmax probability as confidence. While the
    block has `<|mask|>` positions, those above `prob_threshold` are accepted, or else the single most confident
    one; accepted positions whose proposal changed with confidence above `edit_threshold` are rewritten. Once all
    are accepted, up to `refine_steps` edit-only rounds run, stopping at the first that changes nothing.

    The confidence is the softmax probability at temperature one, or, with a `confidence_temperature_min` below one,
    at the reference's step_linear temperature max(min, 1 - decrement * r), r the block's round index (zero for its
    first round, refine rounds included, capped at MAX_STEPS_PER_BLOCK - 1)."""

    prob_threshold: float = 0.7
    edit_threshold: float = 0.5
    refine_steps: int = 16
    allow_accepted_change: bool = True
    confidence_temperature_min: float = 1.0
    confidence_temperature_step_decrement: float = 0.0

    def __post_init__(self):
        if not 0 <= self.prob_threshold < 1 or not 0 <= self.edit_threshold < 1 or self.refine_steps < 0:
            raise ValueError("Invalid decoder settings")
        if not 0 < self.confidence_temperature_min <= 1 or self.confidence_temperature_step_decrement < 0:
            raise ValueError("Invalid confidence temperature schedule")

    @property
    def anneals(self):
        return self.confidence_temperature_min < 1 and self.confidence_temperature_step_decrement > 0

    @property
    def confidence(self):
        """The schedule as row_stats takes it (temperature max(min, max - decrement * round)), the cap folded into
        the minimum."""
        if not self.anneals:
            return SimpleNamespace(confidence_temperature=1.0, confidence_temperature_min=1.0,
                                   confidence_temperature_step_decrement=0.0)
        dec = self.confidence_temperature_step_decrement
        low = max(self.confidence_temperature_min, 1.0 - dec * (MAX_STEPS_PER_BLOCK - 1))
        return SimpleNamespace(confidence_temperature=1.0, confidence_temperature_min=low,
                               confidence_temperature_step_decrement=dec)


ROLES = {"base": LladaDecode(prob_threshold=0.7), "orl": LladaDecode(prob_threshold=0.85)}
