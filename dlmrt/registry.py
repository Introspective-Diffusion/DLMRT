"""Model families by checkpoint `model_type`."""

import json
from pathlib import Path

FAMILIES = {"sdar": "sdar", "llada2_moe": "llada2", "diffusion_gemma": "diffusion_gemma"}


def model_type(path):
    return json.loads((Path(path) / "config.json").read_text()).get("model_type")


def family(path):
    """The dlmrt.models package that serves the checkpoint at `path`."""
    kind = model_type(path)
    if kind not in FAMILIES:
        raise ValueError(f"Unsupported model_type {kind!r}; dlmrt serves {sorted(FAMILIES)}")
    return FAMILIES[kind]
