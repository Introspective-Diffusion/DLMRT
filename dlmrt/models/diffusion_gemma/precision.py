"""DiffusionGemma's checkpoint reading at a precision (see dlmrt.quantization.precision)."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from ...quantization import nvfp4
from ...quantization import precision as precision_switch


def read_weights(path, config, precision, device, group=None):
    """The checkpoint's tensors on `device`, BF16, with this rank's experts only. An NVFP4 export's experts come from
    its per-expert NVFP4 tensors (not its BF16 copies), as `layers.{i}.experts.fp4` for this GPU's NVFP4 MoE backend
    (nvfp4) or as BF16 `experts.{gate_up,down}_proj` (bf16)."""
    path = Path(path)
    checkpoint = precision_switch.checkpoint_precision(path)
    world, rank = (group.world, group.rank) if group else (1, 0)
    local = config.num_experts // world
    lo = rank * local
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    weights, parts = {}, {}
    for shard in sorted(set(index.values())):
        with safe_open(path / shard, framework="pt", device="cpu") as source:
            for key in source.keys():  # noqa: SIM118 - safe_open is not a mapping.
                if key.startswith(("model.encoder.vision_tower.", "model.encoder.embed_vision.")):
                    continue
                expert = nvfp4.EXPERT_KEY.match(key)
                if expert:
                    i, e = int(expert[1]), int(expert[2])
                    if lo <= e < lo + local:
                        parts[(i, e, expert[3], expert[4])] = source.get_tensor(key)
                    continue
                if key.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                    if checkpoint == "nvfp4":
                        continue
                    if world > 1:
                        weights[key] = source.get_slice(key)[lo:lo + local].to(device=device, dtype=torch.bfloat16)
                        continue
                weights[key] = source.get_tensor(key).to(device=device, dtype=torch.bfloat16)
    # backend() imports FlashInfer's MoE, which a BF16 run does not need.
    experts = nvfp4.cutlass_experts if precision == "nvfp4" and nvfp4.backend(device) == "cutlass" else nvfp4.trtllm_experts
    for i in range(config.num_hidden_layers if checkpoint == "nvfp4" else 0):
        p = f"model.decoder.layers.{i}.experts."
        if precision == "nvfp4":
            weights[p + "fp4"] = experts(parts, i, lo, local, device)
            continue
        weights[p + "gate_up_proj"], weights[p + "down_proj"] = nvfp4.expert_weights(parts, i, lo, local, device)
    return weights
