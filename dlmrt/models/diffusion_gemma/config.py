"""DiffusionGemma's checkpoint contract (GemmaConfig) and its uniform-state decoder settings (GemmaDecode, ROLES)."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GemmaConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    top_k_experts: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_global_key_value_heads: int
    global_head_dim: int
    layer_types: tuple
    sliding_window: int
    vocab_size: int
    rope_theta_sliding: float
    rope_theta_full: float
    rope_partial_full: float
    eos_token_ids: tuple
    block_size: int = 256
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())
        if raw.get("model_type") != "diffusion_gemma":
            raise ValueError("Only diffusion_gemma checkpoints are supported")
        t = raw["text_config"]
        if t.get("attention_bias") or raw.get("attention_bias"):
            raise ValueError("Unsupported model setting: attention_bias")
        for q in (t.get("quantization_config"), raw.get("quantization_config")):
            if q and (q.get("quant_method") != "modelopt" or q.get("quant_algo") != "NVFP4"):
                raise ValueError(f"Unsupported quantization: {q.get('quant_method')} {q.get('quant_algo')}")
        if t.get("hidden_activation") != "gelu_pytorch_tanh" or not t.get("tie_word_embeddings", True):
            raise ValueError("Expected GeGLU (gelu_pytorch_tanh) and tied embeddings")
        if t.get("use_bidirectional_attention") == "all":
            raise ValueError("A bidirectional encoder is not implemented")
        rope = t["rope_parameters"]
        if rope["sliding_attention"].get("rope_type", "default") != "default" or \
                rope["full_attention"].get("rope_type") != "proportional":
            raise ValueError("Expected default RoPE on sliding and proportional RoPE on full layers")
        types = tuple(t["layer_types"])
        if set(types) != {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unknown layer types {set(types)}")
        canvas = raw.get("canvas_length", 256)
        if type(canvas) is not int or canvas < 1 or canvas & (canvas - 1):
            raise ValueError(f"canvas_length {canvas!r} must be a power of two (the decoder kernels' tl.arange spans it)")
        eos = raw.get("eos_token_id", t.get("eos_token_id"))
        generation = path / "generation_config.json"
        if generation.exists():
            eos = json.loads(generation.read_text()).get("eos_token_id", eos)
        return cls(hidden_size=t["hidden_size"], intermediate_size=t["intermediate_size"],
                   moe_intermediate_size=t["moe_intermediate_size"], num_experts=t["num_experts"],
                   top_k_experts=t["top_k_experts"], num_hidden_layers=t["num_hidden_layers"],
                   num_attention_heads=t["num_attention_heads"], num_key_value_heads=t["num_key_value_heads"],
                   head_dim=t["head_dim"], num_global_key_value_heads=t["num_global_key_value_heads"],
                   global_head_dim=t["global_head_dim"], layer_types=types, sliding_window=t["sliding_window"],
                   vocab_size=t["vocab_size"], rope_theta_sliding=rope["sliding_attention"]["rope_theta"],
                   rope_theta_full=rope["full_attention"]["rope_theta"],
                   rope_partial_full=rope["full_attention"].get("partial_rotary_factor", 1.0),
                   eos_token_ids=tuple(eos) if isinstance(eos, list) else (eos,),
                   block_size=canvas, rms_norm_eps=t["rms_norm_eps"],
                   # transformers' DiffusionGemmaTextConfig default when absent
                   final_logit_softcapping=t.get("final_logit_softcapping", 30.0))


@dataclass(frozen=True)
class GemmaDecode:
    """Uniform-state decoder of the reference: sample every canvas position at an annealed temperature, keep the
    lowest-entropy samples within `entropy_bound`, re-noise the rest; stop when the argmax canvas is stable for one
    round and its mean entropy is below `confidence_threshold`.

    `temperature_mode`: "linear" (t_max -> t_min over max_denoising_steps, the HF reference) or "adaptive_scalar"
    (the ORL decoder, as the paper's vLLM decoder applies it): round 0 runs at t_max and records the normalized
    raw-logit entropy averaged over the canvas; later rounds run at min(previous T, t_min + (t_max - t_min) *
    (entropy / reference) ** adaptive_power), the entropy averaged over the positions the previous round did not
    accept."""

    entropy_bound: float = 0.1
    t_min: float = 0.4
    t_max: float = 0.8
    max_denoising_steps: int = 48
    confidence_threshold: float = 0.005
    stability_threshold: int = 1
    seed: int = 1234
    temperature_mode: str = "linear"
    adaptive_power: float = 0.5

    def __post_init__(self):
        if self.stability_threshold != 1:
            raise ValueError("Only stability_threshold = 1 is implemented")
        if not 0 <= self.t_min < self.t_max or self.entropy_bound <= 0 or self.max_denoising_steps < 1:
            raise ValueError("Invalid decoder settings")
        if self.temperature_mode not in ("linear", "adaptive_scalar") or self.adaptive_power <= 0:
            raise ValueError("temperature_mode must be linear or adaptive_scalar (with a positive power)")

    @property
    def adaptive(self):
        return self.temperature_mode == "adaptive_scalar"


# The paper's DiffusionGemma decoder settings per checkpoint role.
ROLES = {
    "base": GemmaDecode(),
    "orl": GemmaDecode(entropy_bound=0.12, t_min=0.5, t_max=0.8, temperature_mode="adaptive_scalar"),
}
