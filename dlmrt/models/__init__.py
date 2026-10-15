"""Model families. Each package holds a checkpoint contract and decoder (config.py), the forward (model.py), the
Triton kernels particular to it, if any (kernels.py; QwenDLM uses the shared ones in dlmrt.ops.triton), and its
single-request engine (engine.py)."""
